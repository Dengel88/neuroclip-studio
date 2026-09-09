"""Business rules enforced in code, not only in prompts.

The prompts ask the model to respect these rules; this module is what actually
decides whether an answer is acceptable. Every message raised here is written
for two audiences at once - the human reading the logs and the model reading it
back in the repair prompt - so it always states what was wrong *and* what is
expected.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

from config import AppConfig, config
from errors import DomainValidationError, InfeasibleBriefError


@lru_cache(maxsize=None)
def _reachable_totals(durations: tuple[int, ...], limit: int) -> frozenset[int]:
    """Every total obtainable as a sum of the clip lengths, up to limit."""
    reachable = {0}
    for value in range(1, limit + 1):
        if any(value - d in reachable for d in durations):
            reachable.add(value)
    return frozenset(reachable)


def is_total_duration_feasible(total: int, app_config: AppConfig | None = None) -> bool:
    """Can total seconds be cut into clips of the allowed lengths at all?"""
    active = app_config or config
    if total < active.domain.min_total_duration or total > active.domain.max_total_duration:
        return False
    durations = tuple(sorted(active.domain.allowed_scene_durations))
    return total in _reachable_totals(durations, active.domain.max_total_duration)


def assert_brief_feasible(total: int, app_config: AppConfig | None = None) -> None:
    """Reject an impossible brief before spending a single model call."""
    active = app_config or config
    allowed = active.domain.allowed_scene_durations
    if total < active.domain.min_total_duration or total > active.domain.max_total_duration:
        raise InfeasibleBriefError(
            f"Total duration {total}s is outside the supported range "
            f"{active.domain.min_total_duration}-{active.domain.max_total_duration}s."
        )
    if not is_total_duration_feasible(total, active):
        raise InfeasibleBriefError(
            f"Total duration {total}s cannot be composed from clips of "
            f"{allowed} seconds. Pick a duration that is a sum of those values."
        )


def plan_scene_durations(total: int, app_config: AppConfig | None = None) -> List[int]:
    """A concrete split of total into allowed clip lengths.

    Used by the mock provider and by the eval suite as a reference answer.
    Prefers the longest clips but backtracks through a DP table, so 30s becomes
    [8, 8, 8, 6] instead of failing on a greedy dead end.
    """
    active = app_config or config
    durations = sorted(active.domain.allowed_scene_durations, reverse=True)

    best: List[List[int] | None] = [None] * (total + 1)
    best[0] = []
    for value in range(1, total + 1):
        for clip in durations:
            previous = best[value - clip] if value - clip >= 0 else None
            if previous is not None:
                best[value] = previous + [clip]
                break
    if best[total] is None:
        raise InfeasibleBriefError(
            f"Total duration {total}s cannot be composed from clips of "
            f"{active.domain.allowed_scene_durations} seconds."
        )
    return best[total]


def validate_aspect_ratio(video_format: str, app_config: AppConfig | None = None) -> None:
    active = app_config or config
    if video_format not in active.domain.supported_aspect_ratios:
        raise DomainValidationError(
            f"Aspect ratio '{video_format}' is not supported. "
            f"Use one of: {', '.join(active.domain.supported_aspect_ratios)}."
        )


def validate_concepts(concepts: Sequence, app_config: AppConfig | None = None) -> None:
    active = app_config or config
    expected = active.domain.concepts_count
    if len(concepts) != expected:
        raise DomainValidationError(
            f"Returned {len(concepts)} concepts, expected exactly {expected}."
        )
    ids = [c.id for c in concepts]
    if len(set(ids)) != len(ids):
        raise DomainValidationError(f"Concept ids must be unique, got {ids}.")


def validate_storyboard(
    scenes: Sequence, total_duration: int, app_config: AppConfig | None = None
) -> None:
    """Sum of durations, allowed clip lengths, sequential numbering, scene cap."""
    active = app_config or config
    allowed = active.domain.allowed_scene_durations

    if not scenes:
        raise DomainValidationError("The storyboard contains no scenes.")

    if len(scenes) > active.limits.max_scenes:
        raise DomainValidationError(
            f"The storyboard has {len(scenes)} scenes, the maximum is "
            f"{active.limits.max_scenes}."
        )

    offenders = [
        f"scene {s.scene_number} ({s.duration}s)" for s in scenes if s.duration not in allowed
    ]
    if offenders:
        raise DomainValidationError(
            f"These scenes use a duration the renderer cannot produce: "
            f"{', '.join(offenders)}. Every duration must be one of {allowed}."
        )

    total = sum(s.duration for s in scenes)
    if total != total_duration:
        raise DomainValidationError(
            f"Scene durations add up to {total}s but the video must be exactly "
            f"{total_duration}s. Adjust the shot breakdown - add, drop or re-time "
            f"shots - so the sum matches; every duration must stay in {allowed}."
        )

    numbers = [s.scene_number for s in scenes]
    if numbers != list(range(1, len(scenes) + 1)):
        raise DomainValidationError(
            f"Scenes must be numbered 1..{len(scenes)} with no gaps, got {numbers}."
        )


def missing_dimensions(technical_prompt: str, app_config: AppConfig | None = None) -> List[str]:
    """Which of the template dimensions are absent from a generated prompt."""
    active = app_config or config
    lowered = technical_prompt.lower()
    return [d for d in active.domain.veo_prompt_dimensions if f"{d.lower()}:" not in lowered]


def validate_prompts(
    prompts: Sequence, scenes: Sequence, app_config: AppConfig | None = None
) -> None:
    """One prompt per scene, matching durations, all template dimensions present."""
    active = app_config or config
    dimensions = active.domain.veo_prompt_dimensions

    expected_numbers = [s.scene_number for s in scenes]
    got_numbers = [p.scene_number for p in prompts]
    if got_numbers != expected_numbers:
        raise DomainValidationError(
            f"Expected exactly one prompt per scene, numbered {expected_numbers}, "
            f"got {got_numbers}."
        )

    by_scene = {s.scene_number: s for s in scenes}
    problems: List[str] = []

    for prompt in prompts:
        scene = by_scene[prompt.scene_number]
        if prompt.omni_duration != scene.duration:
            problems.append(
                f"scene {prompt.scene_number}: omni_duration is {prompt.omni_duration}s "
                f"but the storyboard allots {scene.duration}s"
            )
        if prompt.generation_type not in ("text-to-video", "image-to-video"):
            problems.append(
                f"scene {prompt.scene_number}: generation_type "
                f"'{prompt.generation_type}' must be 'text-to-video' or 'image-to-video'"
            )
        if prompt.generation_type == "image-to-video" and not prompt.image_prompt.strip():
            problems.append(
                f"scene {prompt.scene_number}: generation_type is 'image-to-video' "
                "but image_prompt is empty"
            )
        absent = missing_dimensions(prompt.technical_prompt, active)
        if absent:
            problems.append(
                f"scene {prompt.scene_number}: technical_prompt is missing the "
                f"dimension(s) {', '.join(absent)}"
            )

    if problems:
        raise DomainValidationError(
            "The generated prompts break the template contract: "
            + "; ".join(problems)
            + f". Every technical_prompt must contain all {len(dimensions)} labelled "
            f"dimensions ({', '.join(dimensions)}) and omni_duration must equal the "
            "scene duration from the storyboard."
        )


def describe_rules(app_config: AppConfig | None = None) -> Iterable[str]:
    """Human-readable summary, used in logs and the eval report header."""
    active = app_config or config
    yield f"allowed clip durations: {active.domain.allowed_scene_durations}"
    yield f"concepts per brief: {active.domain.concepts_count}"
    yield f"prompt dimensions: {', '.join(active.domain.veo_prompt_dimensions)}"
