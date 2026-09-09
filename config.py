"""Typed application configuration.

Everything tunable lives in `config.yaml`; this module only loads and validates
it. Two environment variables may override the file:

* ``NEUROCLIP_CONFIG``  - path to an alternative YAML file
* ``LLM_PROVIDER``      - overrides ``provider`` (used by tests and evals)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"
PROMPTS_DIR = BASE_DIR / "prompts"


class ChainsConfig(BaseModel):
    reasoning: List[str] = Field(..., min_length=1)
    image: List[str] = Field(..., min_length=1)


class ModelsConfig(BaseModel):
    chains: ChainsConfig


class AgentConfig(BaseModel):
    prompt_file: str
    model_alias: str
    temperature: float = Field(..., ge=0.0, le=2.0)


class AgentsConfig(BaseModel):
    scriptwriter: AgentConfig
    storyboarder: AgentConfig
    prompt_engineer: AgentConfig
    vfx_supervisor: AgentConfig


class RetryConfig(BaseModel):
    attempts_per_key: int = Field(..., ge=1)
    backoff_base_seconds: float = Field(..., ge=0.0)
    backoff_max_seconds: float = Field(..., ge=0.0)
    jitter_ratio: float = Field(..., ge=0.0, le=1.0)
    retry_on_status: List[int]


class RepairConfig(BaseModel):
    max_attempts: int = Field(..., ge=1)


class DomainConfig(BaseModel):
    allowed_scene_durations: List[int] = Field(..., min_length=1)
    concepts_count: int = Field(..., ge=1)
    min_total_duration: int = Field(..., ge=1)
    max_total_duration: int = Field(..., ge=1)
    supported_aspect_ratios: List[str] = Field(..., min_length=1)
    veo_prompt_dimensions: List[str] = Field(..., min_length=1)
    edit_rules: List[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> "DomainConfig":
        if self.min_total_duration > self.max_total_duration:
            raise ValueError("domain.min_total_duration must not exceed max_total_duration")
        if any(d <= 0 for d in self.allowed_scene_durations):
            raise ValueError("domain.allowed_scene_durations must be positive")
        return self


class LimitsConfig(BaseModel):
    max_short_text_chars: int = Field(..., ge=1)
    max_long_text_chars: int = Field(..., ge=1)
    max_scenes: int = Field(..., ge=1)


class SessionsConfig(BaseModel):
    backend: str = Field(default="memory")
    ttl_seconds: int = Field(..., ge=1)
    max_sessions: int = Field(..., ge=1)
    sweep_interval_seconds: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _check_backend(self) -> "SessionsConfig":
        if self.backend not in ("memory", "stateless"):
            raise ValueError("sessions.backend must be either 'memory' or 'stateless'")
        return self


class RateLimitConfig(BaseModel):
    enabled: bool
    requests_per_minute: int = Field(..., ge=1)
    generation_requests_per_minute: int = Field(..., ge=1)


class ImagesConfig(BaseModel):
    storage: str
    directory: str
    retention_seconds: int = Field(..., ge=1)
    cleanup_interval_seconds: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _check_storage(self) -> "ImagesConfig":
        if self.storage not in ("static", "base64"):
            raise ValueError("images.storage must be either 'static' or 'base64'")
        return self


class AppConfig(BaseModel):
    provider: str
    models: ModelsConfig
    agents: AgentsConfig
    retry: RetryConfig
    repair: RepairConfig
    domain: DomainConfig
    limits: LimitsConfig
    sessions: SessionsConfig
    rate_limit: RateLimitConfig
    images: ImagesConfig

    @property
    def images_dir(self) -> Path:
        """Absolute path to the image directory, created on demand."""
        path = Path(self.images.directory)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    def prompt_variables(self) -> dict:
        """Values substituted into the `$placeholders` of `prompts/*.md`.

        Every domain constant that also appears in a prompt is rendered from
        here, so `config.yaml` stays the single source of truth.
        """
        durations = self.domain.allowed_scene_durations
        dimensions = self.domain.veo_prompt_dimensions
        return {
            "allowed_durations": ", ".join(str(d) for d in durations),
            "allowed_durations_list": _humanised_list([str(d) for d in durations]),
            "concepts_count": str(self.domain.concepts_count),
            "dimension_count": str(len(dimensions)),
            "veo_dimensions": ", ".join(dimensions),
            "veo_template": "\n".join(
                f"{i}. {name}: <...>" for i, name in enumerate(dimensions, start=1)
            ),
            "edit_rules": "\n".join(
                f"{i}. {rule}" for i, rule in enumerate(self.domain.edit_rules, start=1)
            ),
        }


def _humanised_list(items: List[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} or {items[-1]}"


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    """Load and validate the YAML config.

    Resolution order: explicit `path` > `$NEUROCLIP_CONFIG` > `config.yaml`
    next to this file. `$LLM_PROVIDER` then overrides `provider`.
    """
    resolved = Path(path or os.getenv("NEUROCLIP_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    _apply_env_overrides(data)
    return AppConfig(**data)


def _apply_env_overrides(data: dict) -> None:
    """Let the deployment environment override a few file settings.

    Serverless hosts get sane defaults automatically: their filesystem is
    read-only and their processes are short-lived, so writing JPEGs to disk and
    keeping sessions in a dict both quietly break. Vercel sets `VERCEL` itself.
    """
    on_serverless = bool(os.getenv("VERCEL"))

    provider = os.getenv("LLM_PROVIDER")
    if provider:
        data["provider"] = provider

    storage = os.getenv("IMAGES_STORAGE") or ("base64" if on_serverless else None)
    if storage:
        data.setdefault("images", {})["storage"] = storage

    backend = os.getenv("SESSIONS_BACKEND") or ("stateless" if on_serverless else None)
    if backend:
        data.setdefault("sessions", {})["backend"] = backend


config = load_config()
