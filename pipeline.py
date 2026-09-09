"""Pipeline state and the stages that operate on it.

The state is server-side and explicit: without it "orchestration" would just be
three independent endpoints, and a targeted retake of one scene would be
impossible because the browser would have to resend everything.

Failure handling has exactly three shapes here:

* ``ValidationError`` / ``DomainValidationError`` -> repairable, fed back to the
  model with the reason attached;
* ``ProviderError``                              -> transport, surfaced as-is,
  never shown to the model;
* ``InfeasibleBriefError``                       -> the brief itself is
  impossible, caught before any model call.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

import domain
from config import config
from errors import DomainValidationError, RepairExhaustedError
from llm import LLMProvider, get_llm_provider
from models import (
    Concept,
    OmniPrompt,
    PromptEngineerOutput,
    Scene,
    ScriptwriterOutput,
    StoryboarderOutput,
)
from prompt_loader import load_prompt

logger = logging.getLogger("NeuroclipStudio.Pipeline")

T = TypeVar("T", bound=BaseModel)


class PipelineState(BaseModel):
    """Everything the funnel knows about one video, in one object."""

    # The brief.
    video_format: str
    total_duration: int
    business_goal: str
    target_audience: str
    topic_idea: str
    needs_voiceover: bool = False

    # Stage outputs.
    concepts: List[Concept] = []
    selected_concept_id: Optional[int] = None
    scenes: List[Scene] = []
    prompts: List[OmniPrompt] = []

    # Audit trail: what was generated, repaired or retaken, in order.
    history: List[str] = []

    @property
    def selected_concept(self) -> Optional[Concept]:
        # `is None` on purpose: concept id 0 is falsy but perfectly valid.
        if self.selected_concept_id is None:
            return None
        return next((c for c in self.concepts if c.id == self.selected_concept_id), None)

    def note(self, message: str) -> None:
        logger.info("pipeline.%s", message)
        self.history.append(message)


class PipelineOrchestrator:
    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        # Resolved lazily so a provider swapped in after construction (tests,
        # evals) is picked up, and so importing this module never builds a client.
        return self._provider or get_llm_provider(config.provider)

    # -- repair loop -----------------------------------------------------

    async def _generate_validated(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        model_alias: str,
        temperature: float,
        context: Optional[Mapping[str, Any]] = None,
        validate: Optional[Callable[[T], None]] = None,
        stage: str = "stage",
    ) -> T:
        """Ask the model, validate, and on a repairable failure ask again.

        `validate` raises `DomainValidationError` for rule violations. Transport
        errors are not caught here at all - they belong to the provider and are
        allowed to propagate untouched.
        """
        attempts = config.repair.max_attempts
        last_error = ""

        for attempt in range(1, attempts + 1):
            prompt = user
            if last_error:
                prompt = (
                    f"{user}\n\n"
                    "Your previous answer was rejected by the pipeline validator.\n"
                    f"REASON: {last_error}\n"
                    "Return a corrected answer that fixes exactly this problem."
                )
            try:
                result = await self.provider.generate_structured(
                    system, prompt, schema, model_alias, temperature, context
                )
                if validate is not None:
                    validate(result)
                if attempt > 1:
                    logger.info("repair.recovered stage=%s attempt=%d", stage, attempt)
                return result
            except (ValidationError, DomainValidationError) as exc:
                last_error = _readable(exc)
                logger.warning(
                    "repair.attempt stage=%s attempt=%d/%d reason=%s",
                    stage, attempt, attempts, last_error,
                )

        logger.error("repair.exhausted stage=%s attempts=%d", stage, attempts)
        raise RepairExhaustedError(attempts, last_error)

    # -- stages ----------------------------------------------------------

    async def generate_concepts(self, state: PipelineState) -> PipelineState:
        domain.validate_aspect_ratio(state.video_format)
        # Fail before spending a model call on a brief the renderer cannot serve.
        domain.assert_brief_feasible(state.total_duration)

        agent = config.agents.scriptwriter
        user = (
            f"Format: {state.video_format}\n"
            f"Total duration: {state.total_duration}s\n"
            f"Business goal: {state.business_goal}\n"
            f"Target audience: {state.target_audience}\n"
            f"Topic: {state.topic_idea}\n"
            f"Needs voiceover: {state.needs_voiceover}"
        )

        result = await self._generate_validated(
            system=load_prompt(agent.prompt_file),
            user=user,
            schema=ScriptwriterOutput,
            model_alias=agent.model_alias,
            temperature=agent.temperature,
            context=state.model_dump(include={
                "topic_idea", "total_duration", "needs_voiceover", "video_format",
            }),
            validate=lambda out: domain.validate_concepts(out.concepts),
            stage="concepts",
        )

        state.concepts = result.concepts
        state.note(f"generated {len(result.concepts)} concepts")
        return state

    async def generate_storyboard(self, state: PipelineState) -> PipelineState:
        concept = state.selected_concept
        if concept is None:
            raise DomainValidationError(
                "No concept selected. Generate concepts and pick one first."
            )

        agent = config.agents.storyboarder
        user = (
            f"Title: {concept.working_title}\n"
            f"Concept: {concept.logline}\n"
            f"Style: {concept.visual_style}\n"
            f"Pacing: {concept.pacing}\n"
            f"Total duration: {state.total_duration}s"
        )

        result = await self._generate_validated(
            system=load_prompt(agent.prompt_file),
            user=user,
            schema=StoryboarderOutput,
            model_alias=agent.model_alias,
            temperature=agent.temperature,
            context={
                "total_duration": state.total_duration,
                "topic_idea": state.topic_idea,
                "video_format": state.video_format,
            },
            validate=lambda out: domain.validate_storyboard(out.scenes, state.total_duration),
            stage="storyboard",
        )

        state.scenes = result.scenes
        # A new storyboard invalidates prompts generated for the old one.
        state.prompts = []
        state.note(f"generated storyboard with {len(result.scenes)} scenes")
        return state

    async def generate_prompts(self, state: PipelineState) -> PipelineState:
        if not state.scenes:
            raise DomainValidationError("No storyboard yet. Generate the storyboard first.")

        concept = state.selected_concept
        style = concept.visual_style if concept else "unspecified"
        agent = config.agents.prompt_engineer

        scenes_text = "\n".join(
            f"Scene {s.scene_number}: {s.visual_description} "
            f"| Camera: {s.camera_movement} | Duration: {s.duration}s"
            for s in state.scenes
        )
        user = (
            f"Target video format: {state.video_format}\n"
            f"Overall style: {style}\n\n"
            f"Scenes:\n{scenes_text}"
        )

        result = await self._generate_validated(
            system=load_prompt(agent.prompt_file),
            user=user,
            schema=PromptEngineerOutput,
            model_alias=agent.model_alias,
            temperature=agent.temperature,
            context={
                "video_format": state.video_format,
                "scenes": [s.model_dump() for s in state.scenes],
            },
            validate=lambda out: domain.validate_prompts(out.prompts, state.scenes),
            stage="prompts",
        )

        state.prompts = result.prompts
        state.note(f"generated {len(result.prompts)} prompts")
        return state

    async def regenerate_scene(
        self, state: PipelineState, scene_number: int, note: Optional[str] = None
    ) -> PipelineState:
        """Re-shoot a single scene, leaving every other scene untouched.

        The replacement must keep the same duration, otherwise the total stops
        matching and the whole storyboard would have to be redone - which is the
        thing this method exists to avoid.
        """
        target = next((s for s in state.scenes if s.scene_number == scene_number), None)
        if target is None:
            raise DomainValidationError(
                f"Scene {scene_number} does not exist. "
                f"Available scenes: {[s.scene_number for s in state.scenes]}."
            )

        concept = state.selected_concept
        agent = config.agents.storyboarder
        neighbours = "\n".join(
            f"Scene {s.scene_number} ({s.duration}s): {s.visual_description}"
            for s in state.scenes
            if s.scene_number != scene_number
        )
        user = (
            f"Style: {concept.visual_style if concept else 'unspecified'}\n"
            f"You are re-shooting ONE scene of an approved storyboard.\n"
            f"Return a storyboard containing exactly one scene, numbered {scene_number}, "
            f"with a duration of exactly {target.duration}s.\n\n"
            f"Scene to replace: {target.visual_description} "
            f"(camera: {target.camera_movement})\n"
            f"{f'Requested change: {note}' if note else ''}\n\n"
            f"Surrounding scenes, which must stay coherent with the new shot:\n{neighbours}"
        )

        def _validate(out: StoryboarderOutput) -> None:
            if len(out.scenes) != 1:
                raise DomainValidationError(
                    f"Expected exactly one replacement scene, got {len(out.scenes)}."
                )
            replacement = out.scenes[0]
            if replacement.duration != target.duration:
                raise DomainValidationError(
                    f"The replacement scene must last exactly {target.duration}s "
                    f"to keep the total at {state.total_duration}s, got "
                    f"{replacement.duration}s."
                )

        result = await self._generate_validated(
            system=load_prompt(agent.prompt_file),
            user=user,
            schema=StoryboarderOutput,
            model_alias=agent.model_alias,
            temperature=agent.temperature,
            context={"total_duration": target.duration, "topic_idea": state.topic_idea},
            validate=_validate,
            stage=f"retake-scene-{scene_number}",
        )

        replacement = result.scenes[0].model_copy(update={"scene_number": scene_number})
        state.scenes = [
            replacement if s.scene_number == scene_number else s for s in state.scenes
        ]
        # The old prompt for this scene no longer describes the shot.
        state.prompts = [p for p in state.prompts if p.scene_number != scene_number]
        # Cheap insurance: the invariant the whole method is built to preserve.
        domain.validate_storyboard(state.scenes, state.total_duration)
        state.note(f"retook scene {scene_number}" + (f" ({note})" if note else ""))
        return state


def _readable(exc: Exception) -> str:
    """A compact, model-friendly rendering of a validation failure."""
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
            for err in exc.errors()[:8]
        ]
        return "The JSON did not match the required schema - " + "; ".join(parts)
    return str(exc)
