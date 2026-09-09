"""Offline provider.

It exists so the eval suite and the tests can exercise the whole pipeline
without a network call or a burnt API key. It is not a stub: it produces answers
that actually satisfy the domain rules, because an offline run that cannot pass
its own validators tells you nothing.

Three modes:

* ``valid``            - well-formed, rule-abiding answers;
* ``broken_duration``  - schema-valid JSON whose scene durations do not add up,
  which is what the repair loop exists for. It heals after
  ``fail_times`` attempts so a repair loop can be observed succeeding;
* ``broken_schema``    - text that is not valid JSON for the schema at all.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel

from config import AppConfig, config
from domain import plan_scene_durations

from .base import LLMProvider

T = TypeVar("T", bound=BaseModel)

MODE_VALID = "valid"
MODE_BROKEN_DURATION = "broken_duration"
MODE_BROKEN_SCHEMA = "broken_schema"


class MockProvider(LLMProvider):
    def __init__(
        self,
        mode: str = MODE_VALID,
        fail_times: int = 1,
        app_config: Optional[AppConfig] = None,
    ):
        self.mode = mode
        self.fail_times = fail_times
        self.config = app_config or config
        self.calls = 0
        self.image_calls = 0
        self.injected_failures = 0

    # -- helpers ---------------------------------------------------------

    def _consume_failure_budget(self) -> bool:
        """True while this provider still owes the caller a deliberate failure.

        Counted per injected failure rather than per call, so a broken mode
        breaks the stage it targets and then heals - which is exactly the
        sequence a repair loop has to survive.
        """
        if self.mode == MODE_VALID or self.injected_failures >= self.fail_times:
            return False
        self.injected_failures += 1
        return True

    def _dimension_block(self, subject: str) -> str:
        return " ".join(
            f"{name}: mock {name.lower()} for {subject}."
            for name in self.config.domain.veo_prompt_dimensions
        )

    # -- LLMProvider -----------------------------------------------------

    async def generate_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        model_alias: str,
        temperature: float = 0.7,
        context: Optional[Mapping[str, Any]] = None,
    ) -> T:
        self.calls += 1
        context = context or {}
        name = schema.__name__

        if self.mode == MODE_BROKEN_SCHEMA and self._consume_failure_budget():
            # Exercises the pydantic-ValidationError branch of the repair loop.
            return schema.model_validate_json(json.dumps({"unexpected": "payload"}))

        builder = {
            "ScriptwriterOutput": self._concepts,
            "StoryboarderOutput": self._storyboard,
            "PromptEngineerOutput": self._prompts,
            "VideoEditOutput": self._edit,
        }.get(name)

        if builder is None:
            raise NotImplementedError(
                f"MockProvider has no fixture for schema '{name}'. Add one in llm/mock.py."
            )

        return schema.model_validate(builder(context))

    async def generate_image(
        self, prompt: str, aspect_ratio: str, model_alias: str = "image"
    ) -> Optional[bytes]:
        self.image_calls += 1
        # A one-pixel JPEG header is enough for the storage layer to be exercised.
        return b"\xff\xd8\xff\xdb" + b"mock-reference-frame"

    # -- fixtures --------------------------------------------------------

    def _concepts(self, context: Mapping[str, Any]) -> dict:
        topic = str(context.get("topic_idea", "the brief"))
        angles = ["documentary", "hyper-stylised", "character-driven", "abstract", "retro"]
        return {
            "concepts": [
                {
                    "id": index,
                    "working_title": f"Mock concept {index}",
                    "logline": f"A {angles[(index - 1) % len(angles)]} take on {topic}.",
                    "visual_style": f"{angles[(index - 1) % len(angles)]}, high contrast",
                    "pacing": "fast" if index % 2 else "measured",
                    "voiceover_tone": "warm" if context.get("needs_voiceover") else "",
                }
                for index in range(1, self.config.domain.concepts_count + 1)
            ]
        }

    def _storyboard(self, context: Mapping[str, Any]) -> dict:
        total = int(context.get("total_duration") or self.config.domain.min_total_duration)
        durations = plan_scene_durations(total, self.config)

        if self.mode == MODE_BROKEN_DURATION and self._consume_failure_budget():
            # Schema-valid, rule-breaking: one scene too many. The pipeline must
            # catch this in code and hand the message back to the model.
            durations = durations + [self.config.domain.allowed_scene_durations[0]]

        return {
            "scenes": [
                {
                    "scene_number": number,
                    "visual_description": f"Mock shot {number}: {context.get('topic_idea', 'subject')}.",
                    "camera_movement": "Static" if number % 2 else "Slow push in",
                    "duration": duration,
                }
                for number, duration in enumerate(durations, start=1)
            ]
        }

    def _prompts(self, context: Mapping[str, Any]) -> dict:
        scenes = context.get("scenes") or []
        prompts = []
        for scene in scenes:
            number = scene["scene_number"]
            # Alternate the router so evals cover both branches.
            is_i2v = number % 2 == 1
            prompts.append(
                {
                    "scene_number": number,
                    "generation_type": "image-to-video" if is_i2v else "text-to-video",
                    "image_prompt": f"Mock reference frame for scene {number}." if is_i2v else "",
                    "technical_prompt": self._dimension_block(f"scene {number}"),
                    "omni_duration": scene["duration"],
                }
            )
        return {"prompts": prompts}

    def _edit(self, context: Mapping[str, Any]) -> dict:
        return {
            "optimized_edit_prompt": (
                "Anchor: keep the subject untouched. Before -> After: mock change applied "
                "at 0:01. Audio follows the visual beat."
            )
        }
