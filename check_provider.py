"""Verify the API keys and the configured model chains against the real API.

Run this before deploying, or whenever the app reports that the upstream model
API is unavailable. It answers the three questions that error cannot:

    Is a key present and accepted?
    Do the models named in config.yaml actually exist for this key?
    Does a real structured call succeed?

    python check_provider.py

Keys are never printed - only how many were found and which position failed.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from config import config  # noqa: E402
from errors import ProviderError  # noqa: E402

OK = "  [ok]  "
BAD = "  [!!]  "
INFO = "        "


class Ping(BaseModel):
    """Smallest possible structured response, to prove the round trip works."""

    answer: str


def keys_present() -> list[int]:
    return [i for i in range(1, 10) if (os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()]


def list_available_models(api_key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    names = []
    for model in client.models.list():
        raw = getattr(model, "name", "") or ""
        names.append(raw.split("/")[-1] if "/" in raw else raw)
    return sorted(n for n in names if n)


async def try_one_call(model_name: str, api_key: str) -> tuple[bool, str]:
    from google import genai
    from google.genai import types

    def _call() -> str:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with the single word: pong",
            config=types.GenerateContentConfig(
                system_instruction="You answer in JSON.",
                response_mime_type="application/json",
                response_schema=Ping,
                temperature=0.0,
            ),
        )
        return response.text or ""

    try:
        raw = await asyncio.to_thread(_call)
        Ping.model_validate_json(raw)
        return True, "structured call succeeded"
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"


async def main() -> int:
    print("\n=== Keys ===")
    positions = keys_present()
    if not positions:
        print(BAD + "No GEMINI_API_KEY_1..9 found.")
        print(INFO + "Put the key in .env as GEMINI_API_KEY_1=... and run again.")
        return 1
    print(OK + f"{len(positions)} key(s) found, at position(s): {positions}")

    primary_key = os.getenv(f"GEMINI_API_KEY_{positions[0]}", "").strip()

    print("\n=== Models this key can actually see ===")
    try:
        available = list_available_models(primary_key)
    except Exception as exc:  # noqa: BLE001
        print(BAD + f"Could not list models: {type(exc).__name__}: {str(exc)[:200]}")
        print(INFO + "A 401/403 here means the key itself is rejected.")
        return 1

    if not available:
        print(BAD + "The API returned an empty model list.")
        return 1
    print(OK + f"{len(available)} models available. Generative ones:")
    for name in available:
        if "embedding" not in name and "aqa" not in name:
            print(INFO + name)

    print("\n=== Configured chains vs reality ===")
    problems: list[str] = []
    for role in ("reasoning", "image"):
        chain = getattr(config.models.chains, role)
        print(f"  {role}:")
        for entry in chain:
            mark = OK if entry in available else BAD
            note = "" if entry in available else "  <- NOT available to this key"
            print(f"{mark}{entry}{note}")
            if entry not in available:
                problems.append(f"{role}: {entry}")
        if not any(entry in available for entry in chain):
            problems.append(f"{role}: the whole chain is unavailable")

    print("\n=== Real call against the reasoning chain ===")
    reachable = [m for m in config.models.chains.reasoning if m in available]
    if not reachable:
        print(BAD + "Nothing in the reasoning chain exists for this key - skipping.")
    else:
        model_name = reachable[0]
        ok, detail = await try_one_call(model_name, primary_key)
        print((OK if ok else BAD) + f"{model_name}: {detail}")
        if not ok:
            problems.append(f"live call to {model_name} failed")

    print("\n=== Verdict ===")
    if not problems:
        print(OK + "Everything checks out. The app should be able to generate.")
        return 0

    print(BAD + "Problems found:")
    for problem in problems:
        print(INFO + problem)
    print()
    print(INFO + "Fix: edit `models.chains` in config.yaml so every entry is a name")
    print(INFO + "from the list above, then redeploy. The first entry is the primary")
    print(INFO + "model; the rest are fallbacks tried in order.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
