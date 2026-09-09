"""Pipeline stages, repair loop and targeted scene retakes."""

from __future__ import annotations

import pytest

import domain
from config import config
from errors import (
    DomainValidationError,
    InfeasibleBriefError,
    ProviderError,
    RepairExhaustedError,
)
from llm.mock import MODE_BROKEN_DURATION, MODE_BROKEN_SCHEMA, MockProvider
from models import StoryboarderOutput
from pipeline import PipelineOrchestrator, PipelineState


async def run_to_prompts(orchestrator, state) -> PipelineState:
    state = await orchestrator.generate_concepts(state)
    state.selected_concept_id = state.concepts[0].id
    state = await orchestrator.generate_storyboard(state)
    return await orchestrator.generate_prompts(state)


# -- happy path ----------------------------------------------------------


async def test_full_pipeline_satisfies_every_domain_rule(orchestrator, state, app_config):
    state = await run_to_prompts(orchestrator, state)

    domain.validate_storyboard(state.scenes, state.total_duration, app_config)
    domain.validate_prompts(state.prompts, state.scenes, app_config)
    assert sum(s.duration for s in state.scenes) == state.total_duration
    assert len(state.concepts) == app_config.domain.concepts_count


async def test_history_records_each_stage(orchestrator, state):
    state = await run_to_prompts(orchestrator, state)
    assert len(state.history) == 3
    assert any("storyboard" in entry for entry in state.history)


# -- state ---------------------------------------------------------------


def test_concept_id_zero_is_a_real_selection(state):
    """`if not state.selected_concept_id` would treat a valid id 0 as unset."""
    from models import Concept

    state.concepts = [
        Concept(
            id=0,
            working_title="Zero",
            logline="l",
            visual_style="s",
            pacing="p",
            voiceover_tone="",
        )
    ]
    state.selected_concept_id = 0
    assert state.selected_concept is not None
    assert state.selected_concept.id == 0


async def test_storyboard_without_a_selected_concept(orchestrator, state):
    state = await orchestrator.generate_concepts(state)
    with pytest.raises(DomainValidationError, match="No concept selected"):
        await orchestrator.generate_storyboard(state)


async def test_prompts_without_a_storyboard(orchestrator, state):
    with pytest.raises(DomainValidationError, match="No storyboard"):
        await orchestrator.generate_prompts(state)


# -- infeasible briefs ---------------------------------------------------


async def test_impossible_duration_fails_before_any_model_call(mock_provider, brief):
    brief = {**brief, "total_duration": 15}
    orchestrator = PipelineOrchestrator(provider=mock_provider)

    with pytest.raises(InfeasibleBriefError, match="cannot be composed"):
        await orchestrator.generate_concepts(PipelineState(**brief))

    assert mock_provider.calls == 0, "an impossible brief must not cost a model call"


async def test_unsupported_aspect_ratio_is_rejected(mock_provider, brief):
    brief = {**brief, "video_format": "21:9"}
    orchestrator = PipelineOrchestrator(provider=mock_provider)

    with pytest.raises(DomainValidationError, match="not supported"):
        await orchestrator.generate_concepts(PipelineState(**brief))
    assert mock_provider.calls == 0


# -- repair loop ---------------------------------------------------------


async def test_repair_loop_recovers_from_a_broken_storyboard(state):
    """The mock breaks the duration sum once, then answers correctly."""
    provider = MockProvider(mode=MODE_BROKEN_DURATION, fail_times=1)
    orchestrator = PipelineOrchestrator(provider=provider)

    state = await orchestrator.generate_concepts(state)
    state.selected_concept_id = state.concepts[0].id
    state = await orchestrator.generate_storyboard(state)

    assert provider.injected_failures == 1
    assert sum(s.duration for s in state.scenes) == state.total_duration


async def test_repair_loop_recovers_from_a_schema_violation(state):
    provider = MockProvider(mode=MODE_BROKEN_SCHEMA, fail_times=1)
    orchestrator = PipelineOrchestrator(provider=provider)

    state = await orchestrator.generate_concepts(state)
    assert provider.injected_failures == 1
    assert len(state.concepts) == config.domain.concepts_count


async def test_repair_loop_gives_up_and_says_why(state, monkeypatch):
    monkeypatch.setattr(config.repair, "max_attempts", 2)
    # Break more times than the loop is allowed to retry.
    provider = MockProvider(mode=MODE_BROKEN_DURATION, fail_times=99)
    orchestrator = PipelineOrchestrator(provider=provider)

    state = await orchestrator.generate_concepts(state)
    state.selected_concept_id = state.concepts[0].id

    with pytest.raises(RepairExhaustedError) as exc:
        await orchestrator.generate_storyboard(state)
    assert "add up to" in exc.value.last_error
    assert exc.value.attempts == 2


async def test_repair_prompt_carries_the_reason(state, monkeypatch):
    """The retry must tell the model what was wrong - that is the whole point."""
    provider = MockProvider(mode=MODE_BROKEN_DURATION, fail_times=1)
    orchestrator = PipelineOrchestrator(provider=provider)
    seen: list[str] = []

    original = provider.generate_structured

    async def spy(system, user, schema, model_alias, temperature=0.7, context=None):
        seen.append(user)
        return await original(system, user, schema, model_alias, temperature, context)

    monkeypatch.setattr(provider, "generate_structured", spy)

    state = await orchestrator.generate_concepts(state)
    state.selected_concept_id = state.concepts[0].id
    await orchestrator.generate_storyboard(state)

    retry_prompt = seen[-1]
    assert "REASON:" in retry_prompt
    assert "add up to" in retry_prompt


async def test_transport_errors_are_never_shown_to_the_model(state, monkeypatch):
    """A 503 must propagate, not get fed back as 'fix your previous answer'."""
    provider = MockProvider()
    orchestrator = PipelineOrchestrator(provider=provider)
    attempts = {"n": 0}

    async def failing(*args, **kwargs):
        attempts["n"] += 1
        raise ProviderError("503 upstream unavailable")

    monkeypatch.setattr(provider, "generate_structured", failing)

    with pytest.raises(ProviderError):
        await orchestrator.generate_concepts(state)
    assert attempts["n"] == 1, "a transport failure must not be retried by the repair loop"


# -- targeted retake -----------------------------------------------------


async def test_regenerate_scene_touches_only_that_scene(orchestrator, state):
    state = await run_to_prompts(orchestrator, state)
    before = [s.model_copy(deep=True) for s in state.scenes]
    target = before[1].scene_number

    state = await orchestrator.regenerate_scene(state, target, note="make it wider")

    assert len(state.scenes) == len(before)
    for old, new in zip(before, state.scenes):
        if new.scene_number == target:
            assert new.duration == old.duration, "the retake must preserve the duration"
        else:
            assert new == old, f"scene {new.scene_number} was modified by the retake"

    assert sum(s.duration for s in state.scenes) == state.total_duration
    # The stale prompt for the replaced shot must be dropped.
    assert all(p.scene_number != target for p in state.prompts)


async def test_regenerate_unknown_scene(orchestrator, state):
    state = await run_to_prompts(orchestrator, state)
    with pytest.raises(DomainValidationError, match="does not exist"):
        await orchestrator.regenerate_scene(state, 999)


async def test_retake_rejects_a_duration_change(orchestrator, state, monkeypatch):
    state = await run_to_prompts(orchestrator, state)
    target = state.scenes[0].scene_number
    wrong_duration = next(
        d for d in config.domain.allowed_scene_durations if d != state.scenes[0].duration
    )

    async def wrong_length(system, user, schema, model_alias, temperature=0.7, context=None):
        return StoryboarderOutput.model_validate(
            {
                "scenes": [
                    {
                        "scene_number": target,
                        "visual_description": "A different shot",
                        "camera_movement": "Static",
                        "duration": wrong_duration,
                    }
                ]
            }
        )

    monkeypatch.setattr(orchestrator.provider, "generate_structured", wrong_length)

    with pytest.raises(RepairExhaustedError) as exc:
        await orchestrator.regenerate_scene(state, target)
    assert "must last exactly" in exc.value.last_error


# -- storyboard invalidation --------------------------------------------


async def test_new_storyboard_drops_stale_prompts(orchestrator, state):
    state = await run_to_prompts(orchestrator, state)
    assert state.prompts
    state = await orchestrator.generate_storyboard(state)
    assert state.prompts == []
