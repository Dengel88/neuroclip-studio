"""Shared fixtures.

No test in this suite touches the network: the pipeline always runs on
`MockProvider`, and the provider tests drive `GeminiProvider` with a fake
`attempt_func` instead of a real client.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Set before `main` is imported: the app refuses to start without credentials.
os.environ.setdefault("BASIC_AUTH_USER", "test-user")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-password")

from config import load_config  # noqa: E402
from llm.mock import MockProvider  # noqa: E402
from pipeline import PipelineOrchestrator, PipelineState  # noqa: E402


@pytest.fixture
def app_config():
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def brief() -> dict:
    return {
        "video_format": "16:9",
        "total_duration": 30,
        "business_goal": "Brand awareness",
        "target_audience": "Developers",
        "topic_idea": "An agent pipeline that writes its own storyboard",
        "needs_voiceover": False,
    }


@pytest.fixture
def state(brief) -> PipelineState:
    return PipelineState(**brief)


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def orchestrator(mock_provider) -> PipelineOrchestrator:
    return PipelineOrchestrator(provider=mock_provider)
