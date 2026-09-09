"""Gemini backend: key rotation, model fallback, exponential backoff.

The important design point is the *split* between two kinds of failure:

* transport failures (429, 503, timeouts) are the provider's problem - it
  retries, backs off and rotates keys, and never lets the model see them;
* a well-delivered answer that does not fit the schema is the *caller's*
  problem - the `ValidationError` is raised straight through so the pipeline's
  repair loop can ask the model again with the error attached.

Mixing the two is what makes a single malformed answer burn every API key.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from enum import Enum
from typing import Any, Callable, List, Mapping, Optional, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import config
from errors import ProviderError, ProviderExhaustedError

from .base import LLMProvider

logger = logging.getLogger("NeuroclipStudio.LLM")

T = TypeVar("T", bound=BaseModel)

_STATUS_IN_TEXT = re.compile(r"\b([45]\d{2})\b")
# Key-specific: this key is bad or throttled out, another key may work.
_KEY_FATAL_STATUSES = frozenset({401, 403})
# Model-specific: the model name is wrong or gone, no key will fix it.
_MODEL_FATAL_STATUSES = frozenset({404})


class Failure(Enum):
    TRANSIENT = "transient"      # retry this key after a backoff
    KEY_FATAL = "key_fatal"      # abandon this key, try the next one
    MODEL_FATAL = "model_fatal"  # abandon this model, try the next in the chain
    FATAL = "fatal"              # our own bug (bad request, bad schema) - give up now


def _status_of(error: BaseException) -> Optional[int]:
    """Best-effort HTTP status extraction from a google-genai error."""
    for attribute in ("code", "status_code", "http_status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    match = _STATUS_IN_TEXT.search(str(error))
    return int(match.group(1)) if match else None


def classify(error: BaseException, retry_on_status: List[int]) -> Failure:
    status = _status_of(error)
    if status is not None:
        if status in retry_on_status:
            return Failure.TRANSIENT
        if status in _KEY_FATAL_STATUSES:
            return Failure.KEY_FATAL
        if status in _MODEL_FATAL_STATUSES:
            return Failure.MODEL_FATAL
        return Failure.FATAL
    if isinstance(error, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return Failure.TRANSIENT
    # Unrecognised: treat as key-specific rather than aborting the whole request,
    # but never as transient - we do not want to sit in a backoff loop on a bug.
    return Failure.KEY_FATAL


class GeminiProvider(LLMProvider):
    """Talks to the Gemini API over a pool of rotating keys."""

    def __init__(self, api_keys: Optional[List[str]] = None):
        self.keys = api_keys if api_keys is not None else self._keys_from_env()
        if not self.keys:
            logger.warning(
                "No Gemini API keys found. Set GEMINI_API_KEY_1..9 in .env, "
                "or run with LLM_PROVIDER=mock."
            )

    @staticmethod
    def _keys_from_env() -> List[str]:
        raw = (os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 10))
        return [key.strip() for key in raw if key and key.strip()]

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with symmetric jitter, bounded by the config."""
        base = config.retry.backoff_base_seconds * (2 ** attempt)
        capped = min(base, config.retry.backoff_max_seconds)
        jitter = capped * config.retry.jitter_ratio
        return max(0.0, capped + random.uniform(-jitter, jitter))

    async def _execute_with_rotation(
        self, model_chain: List[str], attempt_func: Callable[..., Any]
    ) -> Any:
        """Run `attempt_func(client, model_name)` until something works.

        Only transport-shaped failures are retried here. Anything the caller
        could repair (a `ValidationError`) is raised by the caller, not inside
        this loop - `attempt_func` deliberately returns raw text.
        """
        if not self.keys:
            raise ProviderError(
                "No API keys available. Set GEMINI_API_KEY_1 or use the mock provider."
            )

        retry_on = config.retry.retry_on_status
        attempts_per_key = config.retry.attempts_per_key
        last_error: Optional[BaseException] = None

        for model_name in model_chain:
            model_is_dead = False
            for key_index, key in enumerate(self.keys, start=1):
                client = genai.Client(api_key=key)
                for attempt in range(attempts_per_key):
                    try:
                        logger.info(
                            "llm.attempt model=%s key=#%d try=%d/%d",
                            model_name, key_index, attempt + 1, attempts_per_key,
                        )
                        return await asyncio.to_thread(attempt_func, client, model_name)
                    except Exception as exc:  # noqa: BLE001 - re-raised after classification
                        last_error = exc
                        verdict = classify(exc, retry_on)
                        logger.warning(
                            "llm.failure model=%s key=#%d try=%d verdict=%s error=%s",
                            model_name, key_index, attempt + 1,
                            verdict.value, type(exc).__name__,
                        )
                        if verdict is Failure.FATAL:
                            raise ProviderError(
                                f"Non-retryable error from {model_name}: {exc}"
                            ) from exc
                        if verdict is Failure.MODEL_FATAL:
                            model_is_dead = True  # no key will resurrect it
                            break
                        if verdict is Failure.KEY_FATAL:
                            break  # this key is done, try the next one
                        delay = self._backoff_delay(attempt)
                        logger.info("llm.backoff seconds=%.2f", delay)
                        await asyncio.sleep(delay)
                if model_is_dead:
                    break

        logger.error("llm.exhausted models=%s keys=%d", model_chain, len(self.keys))
        raise ProviderExhaustedError(
            f"All {len(self.keys)} key(s) and {len(model_chain)} model(s) were tried "
            f"without success. Last error: {last_error}"
        )

    async def generate_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        model_alias: str,
        temperature: float = 0.7,
        context: Optional[Mapping[str, Any]] = None,
    ) -> T:
        model_chain = getattr(config.models.chains, model_alias)

        def _call(client: "genai.Client", model_name: str) -> str:
            cfg = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            )
            response = client.models.generate_content(
                model=model_name, contents=user, config=cfg
            )
            if not response.text:
                raise ProviderError(f"{model_name} returned an empty response body.")
            return response.text

        raw = await self._execute_with_rotation(model_chain, _call)
        # Outside the rotation on purpose: a ValidationError here is repairable
        # by the pipeline and must not consume another key.
        return schema.model_validate_json(raw)

    async def generate_image(
        self, prompt: str, aspect_ratio: str, model_alias: str = "image"
    ) -> Optional[bytes]:
        model_chain = getattr(config.models.chains, model_alias)

        def _call(client: "genai.Client", model_name: str) -> bytes:
            result = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio=aspect_ratio,
                ),
            )
            images = getattr(result, "generated_images", None)
            if not images:
                raise ProviderError(f"{model_name} returned no image data.")
            return images[0].image.image_bytes

        try:
            return await self._execute_with_rotation(model_chain, _call)
        except ProviderError as exc:
            # A missing reference frame degrades the result but must not fail
            # the whole request - the text prompt is still usable.
            logger.error("image.unavailable error=%s", exc)
            return None
