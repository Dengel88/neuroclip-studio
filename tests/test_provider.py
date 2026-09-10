"""Provider behaviour: rotation, fallback, and the transport/validation split.

The bug these tests exist to prevent: catching everything inside the rotation
loop, so a schema violation is retried against every key and every fallback
model, burns the whole key pool, and reaches the caller as a transport error the
repair loop cannot recognise.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from config import config
from errors import ProviderError, ProviderExhaustedError
from llm.factory import get_llm_provider, reset_llm_providers
from llm.gemini import Failure, GeminiProvider, classify


class Tiny(BaseModel):
    value: int


class FakeAPIError(Exception):
    """Stands in for google.genai's error type: carries an HTTP status."""

    def __init__(self, code: int, message: str = "api failure"):
        super().__init__(f"{code} {message}")
        self.code = code


@pytest.fixture
def fast_retries(monkeypatch):
    """Same control flow, no wall-clock cost."""
    monkeypatch.setattr(config.retry, "backoff_base_seconds", 0.0)
    monkeypatch.setattr(config.retry, "backoff_max_seconds", 0.0)
    monkeypatch.setattr(config.retry, "attempts_per_key", 2)


@pytest.fixture
def no_real_client(monkeypatch):
    monkeypatch.setattr("llm.gemini.genai.Client", lambda api_key: object())


# -- classification ------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_documented_statuses_are_transient(status):
    assert classify(FakeAPIError(status), config.retry.retry_on_status) is Failure.TRANSIENT


@pytest.mark.parametrize("status,expected", [(401, Failure.KEY_FATAL), (403, Failure.KEY_FATAL),
                                             (404, Failure.MODEL_FATAL), (400, Failure.FATAL)])
def test_non_transient_statuses(status, expected):
    assert classify(FakeAPIError(status), config.retry.retry_on_status) is expected


def test_timeouts_are_transient():
    assert classify(TimeoutError("timed out"), config.retry.retry_on_status) is Failure.TRANSIENT


# -- rotation ------------------------------------------------------------


async def test_rotation_moves_to_the_next_key_after_429(fast_retries, no_real_client):
    provider = GeminiProvider(api_keys=["k1", "k2", "k3"])
    seen: list[str] = []

    def attempt(client, model_name):
        seen.append(model_name)
        if len(seen) < 3:
            raise FakeAPIError(429, "rate limited")
        return "ok"

    result = await provider._execute_with_rotation(["model-a"], attempt)
    assert result == "ok"
    assert len(seen) == 3


async def test_all_keys_and_models_exhausted(fast_retries, no_real_client):
    provider = GeminiProvider(api_keys=["k1", "k2"])
    calls = {"n": 0}

    def attempt(client, model_name):
        calls["n"] += 1
        raise FakeAPIError(503, "unavailable")

    with pytest.raises(ProviderExhaustedError, match="Last error"):
        await provider._execute_with_rotation(["model-a", "model-b"], attempt)

    # 2 models x 2 keys x 2 attempts per key.
    assert calls["n"] == 8


async def test_model_fallback_after_404(fast_retries, no_real_client):
    """A missing model must cost one attempt, not one attempt per key."""
    provider = GeminiProvider(api_keys=["k1", "k2", "k3"])
    seen: list[str] = []

    def attempt(client, model_name):
        seen.append(model_name)
        if model_name == "gone":
            raise FakeAPIError(404, "model not found")
        return "ok"

    assert await provider._execute_with_rotation(["gone", "present"], attempt) == "ok"
    assert seen == ["gone", "present"]


async def test_bad_request_aborts_immediately(fast_retries, no_real_client):
    provider = GeminiProvider(api_keys=["k1", "k2", "k3"])
    calls = {"n": 0}

    def attempt(client, model_name):
        calls["n"] += 1
        raise FakeAPIError(400, "malformed request")

    with pytest.raises(ProviderError, match="Non-retryable"):
        await provider._execute_with_rotation(["model-a", "model-b"], attempt)
    assert calls["n"] == 1


async def test_auth_failure_tries_the_next_key_once_each(fast_retries, no_real_client):
    provider = GeminiProvider(api_keys=["k1", "k2", "k3"])
    calls = {"n": 0}

    def attempt(client, model_name):
        calls["n"] += 1
        raise FakeAPIError(403, "permission denied")

    with pytest.raises(ProviderExhaustedError):
        await provider._execute_with_rotation(["model-a"], attempt)
    # One attempt per key, no backoff retries: a rejected key will not heal.
    assert calls["n"] == 3


async def test_no_keys_is_a_clear_error():
    provider = GeminiProvider(api_keys=[])
    with pytest.raises(ProviderError, match="No API keys"):
        await provider._execute_with_rotation(["model-a"], lambda c, m: "never")


# -- the split that matters ---------------------------------------------


def _client_returning(text: str, counter: dict):
    """A stand-in genai.Client that always answers with `text`."""

    class Response:
        def __init__(self):
            self.text = text

    class Models:
        def generate_content(self, **kwargs):
            counter["n"] += 1
            return Response()

    class Client:
        models = Models()

    return lambda api_key: Client()


async def test_validation_error_does_not_consume_keys(fast_retries, monkeypatch):
    """A schema violation must surface on the first try, with keys untouched.

    Three keys and two models in the chain: if the rotation loop swallowed the
    `ValidationError`, this would be six API calls and a `ProviderExhaustedError`
    instead of one call and a repairable error.
    """
    calls = {"n": 0}
    monkeypatch.setattr(
        "llm.gemini.genai.Client", _client_returning('{"value": "not an int"}', calls)
    )
    monkeypatch.setattr(config.models.chains, "reasoning", ["model-a", "model-b"])
    provider = GeminiProvider(api_keys=["k1", "k2", "k3"])

    with pytest.raises(ValidationError):
        await provider.generate_structured("sys", "user", Tiny, "reasoning")

    assert calls["n"] == 1


async def test_valid_json_is_parsed(fast_retries, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr("llm.gemini.genai.Client", _client_returning('{"value": 7}', calls))
    provider = GeminiProvider(api_keys=["k1"])

    result = await provider.generate_structured("sys", "user", Tiny, "reasoning")
    assert result.value == 7
    assert calls["n"] == 1


async def test_empty_response_is_a_provider_error(fast_retries, monkeypatch):
    """An empty body is a transport problem, so it does rotate - and then fails."""
    calls = {"n": 0}
    monkeypatch.setattr("llm.gemini.genai.Client", _client_returning("", calls))
    monkeypatch.setattr(config.models.chains, "reasoning", ["model-a"])
    provider = GeminiProvider(api_keys=["k1"])

    with pytest.raises(ProviderError):
        await provider.generate_structured("sys", "user", Tiny, "reasoning")
    assert calls["n"] >= 1


def test_backoff_grows_and_stays_within_bounds(monkeypatch):
    monkeypatch.setattr(config.retry, "backoff_base_seconds", 1.0)
    monkeypatch.setattr(config.retry, "backoff_max_seconds", 8.0)
    monkeypatch.setattr(config.retry, "jitter_ratio", 0.25)
    provider = GeminiProvider(api_keys=["k1"])

    for attempt in range(6):
        delay = provider._backoff_delay(attempt)
        expected = min(1.0 * (2 ** attempt), 8.0)
        assert 0.0 <= delay <= expected * 1.25 + 1e-9


# -- factory -------------------------------------------------------------


def test_provider_instance_is_reused():
    """Rebuilding the provider per request would reset key rotation state."""
    reset_llm_providers()
    assert get_llm_provider("mock") is get_llm_provider("mock")
    reset_llm_providers()


def test_unknown_provider_is_rejected():
    reset_llm_providers()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider("hal9000")
    reset_llm_providers()


# -- diagnosis -----------------------------------------------------------


async def test_exhaustion_reports_which_model_and_status(fast_retries, no_real_client):
    """'Service unavailable' is not a diagnosis. Say what refused, and how."""
    provider = GeminiProvider(api_keys=["k1"])

    def attempt(client, model_name):
        raise FakeAPIError(404 if model_name == "missing-model" else 403)

    with pytest.raises(ProviderExhaustedError) as exc:
        await provider._execute_with_rotation(["missing-model", "other-model"], attempt)

    assert "missing-model:404" in exc.value.trail
    assert "other-model:403" in exc.value.trail


async def test_trail_does_not_repeat_identical_failures(fast_retries, no_real_client):
    provider = GeminiProvider(api_keys=["k1", "k2", "k3"])

    def attempt(client, model_name):
        raise FakeAPIError(403)

    with pytest.raises(ProviderExhaustedError) as exc:
        await provider._execute_with_rotation(["one-model"], attempt)

    assert exc.value.trail == ["one-model:403"]


# -- image generation ----------------------------------------------------


async def test_image_is_read_from_an_inline_data_part(fast_retries, monkeypatch):
    """Gemini image models answer through generate_content, not generate_images."""

    class Inline:
        data = b"\xff\xd8jpeg-bytes"

    class Part:
        inline_data = Inline()

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()

    class Response:
        candidates = [Candidate()]

    class Models:
        def generate_content(self, **kwargs):
            return Response()

        def generate_images(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("generate_images is the deprecated Imagen-only call")

    class Client:
        models = Models()

    monkeypatch.setattr("llm.gemini.genai.Client", lambda api_key: Client())
    monkeypatch.setattr(config.models.chains, "image", ["an-image-model"])
    provider = GeminiProvider(api_keys=["k1"])

    assert await provider.generate_image("a red apple", "16:9") == b"\xff\xd8jpeg-bytes"


async def test_missing_image_degrades_to_none(fast_retries, monkeypatch):
    """A quota wall on images must not fail the whole prompt request.

    Image models are the first thing to hit a free-tier limit, and a storyboard
    without reference frames is still a usable deliverable.
    """
    monkeypatch.setattr("llm.gemini.genai.Client", lambda api_key: object())
    monkeypatch.setattr(config.models.chains, "image", ["img-a", "img-b"])
    provider = GeminiProvider(api_keys=["k1"])

    async def refuse(chain, fn):
        raise ProviderExhaustedError("all image models refused")

    monkeypatch.setattr(provider, "_execute_with_rotation", refuse)

    assert await provider.generate_image("a red apple", "16:9") is None


async def test_fatal_error_also_reports_the_status(fast_retries, no_real_client):
    """An invalid key (400) must not read as 'the upstream API is unavailable'."""
    provider = GeminiProvider(api_keys=["k1"])

    def attempt(client, model_name):
        raise FakeAPIError(400, "API key not valid")

    with pytest.raises(ProviderError) as exc:
        await provider._execute_with_rotation(["model-a"], attempt)

    assert exc.value.trail == ["model-a:400"]


def test_safe_reason_redacts_anything_key_shaped():
    """An upstream error can quote the request back; a key must not survive it."""
    from llm.gemini import safe_reason

    leaked = Exception(
        "400 INVALID_ARGUMENT: API key not valid: AIzaSyD-EXAMPLEKEY_1234567890abc"
    )
    reason = safe_reason(leaked)
    assert "AIzaSy" not in reason
    assert "[REDACTED_KEY]" in reason
    assert "API key not valid" in reason


def test_safe_reason_is_capped():
    from llm.gemini import safe_reason

    assert len(safe_reason(Exception("x" * 5000))) <= 223


async def test_fatal_error_carries_the_upstream_reason(fast_retries, no_real_client):
    provider = GeminiProvider(api_keys=["k1"])

    def attempt(client, model_name):
        raise FakeAPIError(400, "API key not valid. Please pass a valid API key.")

    with pytest.raises(ProviderError) as exc:
        await provider._execute_with_rotation(["model-a"], attempt)

    assert "API key not valid" in exc.value.upstream_reason
