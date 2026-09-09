"""The rules that must hold regardless of what the model returns."""

from __future__ import annotations

import pytest

import domain
from errors import DomainValidationError, InfeasibleBriefError
from models import OmniPrompt, Scene


def scene(number: int, duration: int, description: str = "A shot") -> Scene:
    return Scene(
        scene_number=number,
        visual_description=description,
        camera_movement="Static",
        duration=duration,
    )


def prompt_for(sc: Scene, app_config, *, technical: str | None = None, duration=None) -> OmniPrompt:
    body = technical or " ".join(
        f"{d}: value." for d in app_config.domain.veo_prompt_dimensions
    )
    return OmniPrompt(
        scene_number=sc.scene_number,
        generation_type="text-to-video",
        image_prompt="",
        technical_prompt=body,
        omni_duration=duration if duration is not None else sc.duration,
    )


# -- duration arithmetic -------------------------------------------------


@pytest.mark.parametrize("total", [4, 6, 8, 10, 12, 14, 30, 60])
def test_feasible_totals_can_be_planned(total, app_config):
    plan = domain.plan_scene_durations(total, app_config)
    assert sum(plan) == total
    assert all(d in app_config.domain.allowed_scene_durations for d in plan)


@pytest.mark.parametrize("total", [1, 2, 3, 5, 7, 9, 11, 15])
def test_totals_that_no_combination_reaches(total, app_config):
    """With 4/6/8-second clips, odd totals and small numbers are impossible."""
    assert not domain.is_total_duration_feasible(total, app_config)
    with pytest.raises(InfeasibleBriefError):
        domain.assert_brief_feasible(total, app_config)


def test_out_of_range_total_is_rejected(app_config):
    too_long = app_config.domain.max_total_duration + 2
    with pytest.raises(InfeasibleBriefError, match="outside the supported range"):
        domain.assert_brief_feasible(too_long, app_config)


# -- storyboard ----------------------------------------------------------


def test_valid_storyboard_passes(app_config):
    scenes = [scene(1, 8), scene(2, 8), scene(3, 8), scene(4, 6)]
    domain.validate_storyboard(scenes, 30, app_config)


def test_sum_mismatch_is_reported_with_both_numbers(app_config):
    scenes = [scene(1, 8), scene(2, 8)]
    with pytest.raises(DomainValidationError) as exc:
        domain.validate_storyboard(scenes, 30, app_config)
    assert "16s" in str(exc.value) and "30s" in str(exc.value)


def test_disallowed_duration_is_reported_per_scene(app_config):
    scenes = [scene(1, 8), scene(2, 7), scene(3, 8), scene(4, 7)]
    with pytest.raises(DomainValidationError, match="scene 2"):
        domain.validate_storyboard(scenes, 30, app_config)


def test_gaps_in_numbering_are_rejected(app_config):
    scenes = [scene(1, 8), scene(3, 8), scene(4, 8), scene(5, 6)]
    with pytest.raises(DomainValidationError, match="numbered"):
        domain.validate_storyboard(scenes, 30, app_config)


def test_empty_storyboard_is_rejected(app_config):
    with pytest.raises(DomainValidationError, match="no scenes"):
        domain.validate_storyboard([], 30, app_config)


def test_scene_count_cap(app_config):
    count = app_config.limits.max_scenes + 1
    scenes = [scene(i, 4) for i in range(1, count + 1)]
    with pytest.raises(DomainValidationError, match="maximum"):
        domain.validate_storyboard(scenes, 4 * count, app_config)


# -- prompt template -----------------------------------------------------


def test_all_dimensions_present(app_config):
    scenes = [scene(1, 8)]
    domain.validate_prompts([prompt_for(scenes[0], app_config)], scenes, app_config)


def test_missing_dimension_is_named(app_config):
    dimensions = app_config.domain.veo_prompt_dimensions
    partial = " ".join(f"{d}: value." for d in dimensions[:-1])
    scenes = [scene(1, 8)]
    with pytest.raises(DomainValidationError, match=dimensions[-1]):
        domain.validate_prompts(
            [prompt_for(scenes[0], app_config, technical=partial)], scenes, app_config
        )


def test_duration_drift_between_prompt_and_scene(app_config):
    scenes = [scene(1, 8)]
    with pytest.raises(DomainValidationError, match="omni_duration"):
        domain.validate_prompts(
            [prompt_for(scenes[0], app_config, duration=4)], scenes, app_config
        )


def test_image_to_video_without_image_prompt(app_config):
    scenes = [scene(1, 8)]
    bad = prompt_for(scenes[0], app_config).model_copy(
        update={"generation_type": "image-to-video", "image_prompt": "  "}
    )
    with pytest.raises(DomainValidationError, match="image_prompt is empty"):
        domain.validate_prompts([bad], scenes, app_config)


def test_one_prompt_per_scene(app_config):
    scenes = [scene(1, 8), scene(2, 8)]
    with pytest.raises(DomainValidationError, match="one prompt per scene"):
        domain.validate_prompts([prompt_for(scenes[0], app_config)], scenes, app_config)


def test_aspect_ratio_must_be_supported(app_config):
    domain.validate_aspect_ratio(app_config.domain.supported_aspect_ratios[0], app_config)
    with pytest.raises(DomainValidationError, match="not supported"):
        domain.validate_aspect_ratio("21:9", app_config)
