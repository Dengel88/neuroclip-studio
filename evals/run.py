"""Offline regression suite for the pipeline.

Runs every brief in `cases.yaml` end to end against the mock provider - no
network, no API keys, deterministic - and checks the properties that must hold
whatever the model says: the durations add up, every clip length is renderable,
each scene has exactly one prompt with all template dimensions, and the router
picked a generation mode. Briefs marked `expect: reject` must be refused, and
refused before a model call.

    python evals/run.py [--provider mock|gemini] [--quiet]

Exit code is non-zero when the pass rate is below 100%, so this is usable as a
CI gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import domain  # noqa: E402
from config import config  # noqa: E402
from errors import (  # noqa: E402
    DomainValidationError,
    InfeasibleBriefError,
    NeuroclipError,
)
from llm import get_llm_provider  # noqa: E402
from pipeline import PipelineOrchestrator, PipelineState  # noqa: E402

CASES_PATH = Path(__file__).with_name("cases.yaml")
RESULTS_DIR = Path(__file__).with_name("results")

logger = logging.getLogger("NeuroclipEvals")

BRIEF_FIELDS = (
    "video_format",
    "total_duration",
    "business_goal",
    "target_audience",
    "topic_idea",
    "needs_voiceover",
)


class CheckFailed(Exception):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


async def run_case(case: Dict[str, Any], provider) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": case["id"],
        "expect": case.get("expect", "pass"),
        "passed": False,
        "checks": {},
        "error": None,
    }
    orchestrator = PipelineOrchestrator(provider=provider)
    state = PipelineState(**{field: case[field] for field in BRIEF_FIELDS})
    calls_before = getattr(provider, "calls", None)

    try:
        state = await orchestrator.generate_concepts(state)

        if result["expect"] == "reject":
            raise CheckFailed(
                f"brief should have been refused ({case.get('reason', 'no reason given')}) "
                "but the pipeline accepted it"
            )

        expect(
            len(state.concepts) == config.domain.concepts_count,
            f"expected {config.domain.concepts_count} concepts, got {len(state.concepts)}",
        )
        result["checks"]["concepts"] = "ok"

        state.selected_concept_id = state.concepts[0].id
        state = await orchestrator.generate_storyboard(state)
        domain.validate_storyboard(state.scenes, state.total_duration)
        result["checks"]["duration_sum"] = (
            f"{sum(s.duration for s in state.scenes)}s == {state.total_duration}s"
        )
        result["checks"]["scene_count"] = len(state.scenes)

        state = await orchestrator.generate_prompts(state)
        domain.validate_prompts(state.prompts, state.scenes)
        result["checks"]["dimensions"] = "all present"

        modes = {p.generation_type for p in state.prompts}
        expect(
            modes <= {"text-to-video", "image-to-video"} and modes,
            f"router produced unexpected modes: {modes}",
        )
        result["checks"]["router_modes"] = sorted(modes)

        # A targeted retake must not disturb the arithmetic.
        retake_target = state.scenes[0].scene_number
        state = await orchestrator.regenerate_scene(state, retake_target)
        domain.validate_storyboard(state.scenes, state.total_duration)
        result["checks"]["retake_keeps_total"] = "ok"

        result["passed"] = True

    except (InfeasibleBriefError, DomainValidationError) as exc:
        if result["expect"] == "reject":
            result["checks"]["rejected"] = str(exc)
            if calls_before is not None:
                spent = provider.calls - calls_before
                expect(spent == 0, f"refusal cost {spent} model call(s), expected 0")
                result["checks"]["cost"] = "0 model calls"
            result["passed"] = True
        else:
            result["error"] = f"{type(exc).__name__}: {exc}"
    except (CheckFailed, NeuroclipError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - an eval run should report, not crash
        result["error"] = f"unexpected {type(exc).__name__}: {exc}"

    return result


async def run_evals(provider_name: str, quiet: bool) -> int:
    provider = get_llm_provider(provider_name)
    cases: List[dict] = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    logger.info("provider=%s cases=%d", provider_name, len(cases))
    for line in domain.describe_rules():
        logger.info("rule: %s", line)

    results = [await run_case(case, provider) for case in cases]
    passed = sum(1 for r in results if r["passed"])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "models": config.models.chains.reasoning,
        "rules": list(domain.describe_rules()),
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results) * 100, 1) if results else 0.0,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS_DIR / f"report_{provider_name}_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not quiet:
        print_table(results, report)
    print(f"\nReport: {report_path.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1


def print_table(results: List[dict], report: dict) -> None:
    width = max(len(r["id"]) for r in results) + 2
    print()
    print(f"{'CASE':<{width}} {'EXPECT':<8} {'RESULT':<8} DETAIL")
    print("-" * (width + 60))
    for r in results:
        verdict = "PASS" if r["passed"] else "FAIL"
        detail = r["error"] or ", ".join(f"{k}={v}" for k, v in r["checks"].items())
        print(f"{r['id']:<{width}} {r['expect']:<8} {verdict:<8} {detail[:80]}")
    print("-" * (width + 60))
    print(f"Pass rate: {report['passed']}/{report['total']} ({report['pass_rate']}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Neuroclip eval suite.")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "gemini"],
        help="mock (default, offline and free) or gemini (spends real API quota)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the report path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
    )
    # Keep the per-attempt provider chatter out of the table.
    logging.getLogger("NeuroclipStudio.LLM").setLevel(logging.WARNING)
    logging.getLogger("NeuroclipStudio.Pipeline").setLevel(logging.WARNING)

    os.environ["LLM_PROVIDER"] = args.provider
    config.provider = args.provider
    return asyncio.run(run_evals(args.provider, args.quiet))


if __name__ == "__main__":
    raise SystemExit(main())
