"""Provider-agnostic LLM interface.

Nothing above this layer imports `google.genai`. Swapping the backend means
adding one module next to `gemini.py` and one line in `factory.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        model_alias: str,
        temperature: float = 0.7,
        context: Optional[Mapping[str, Any]] = None,
    ) -> T:
        """Return an instance of `schema` parsed from the model's answer.

        `context` carries structured facts about the call that a real backend
        ignores (they are already spelled out in `user`) but that an offline
        provider needs in order to synthesise a *valid* answer - notably the
        requested total duration and the storyboard being worked on.

        Raises:
            pydantic.ValidationError: the answer did not fit the schema. The
                caller may repair it by asking again with the error attached.
            errors.ProviderError: transport failure. Never repairable by the
                model, never fed back into a prompt.
        """

    @abstractmethod
    async def generate_image(
        self, prompt: str, aspect_ratio: str, model_alias: str = "image"
    ) -> Optional[bytes]:
        """Render a still frame, or return None when the image models are unavailable."""
