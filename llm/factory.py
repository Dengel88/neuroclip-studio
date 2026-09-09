"""Provider selection.

The instance is cached per provider name. That is not a micro-optimisation:
`GeminiProvider` owns the key pool, and rebuilding it on every request would
throw away the rotation state, so a throttled key would be picked first again
and again.
"""

from __future__ import annotations

from typing import Dict, Optional

from config import config

from .base import LLMProvider

_INSTANCES: Dict[str, LLMProvider] = {}


def _build(name: str) -> LLMProvider:
    if name == "gemini":
        from .gemini import GeminiProvider  # imported lazily: pulls in google.genai

        return GeminiProvider()
    if name == "mock":
        from .mock import MockProvider

        return MockProvider()
    raise ValueError(f"Unknown LLM provider '{name}'. Supported: 'gemini', 'mock'.")


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Return the shared provider instance for `provider_name`."""
    name = (provider_name or config.provider).strip().lower()
    if name not in _INSTANCES:
        _INSTANCES[name] = _build(name)
    return _INSTANCES[name]


def set_llm_provider(provider_name: str, provider: LLMProvider) -> None:
    """Install a provider explicitly. Used by tests and by the eval suite."""
    _INSTANCES[provider_name.strip().lower()] = provider


def reset_llm_providers() -> None:
    """Drop every cached instance. Used by tests."""
    _INSTANCES.clear()
