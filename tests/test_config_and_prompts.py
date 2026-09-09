"""Config loading and prompt rendering.

The point of these tests is the promise in the README: a domain constant lives
in `config.yaml` and nowhere else. If a placeholder ever ships unsubstituted,
the model receives the literal string `$allowed_durations` inside its system
instruction and silently ignores the rule - a failure that is invisible in
production and obvious here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import PROMPTS_DIR, AppConfig, load_config
from prompt_loader import preload_all_prompts, render_prompt

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILES = sorted(p.name for p in Path(PROMPTS_DIR).glob("*.md"))


def test_config_loads_and_validates(app_config):
    assert isinstance(app_config, AppConfig)
    assert app_config.domain.allowed_scene_durations
    assert app_config.models.chains.reasoning


def test_env_overrides_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert load_config(ROOT / "config.yaml").provider == "mock"


def test_invalid_config_is_rejected(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["images"]["storage"] = "dropbox"
    broken = tmp_path / "config.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="images.storage"):
        load_config(broken)


def test_min_greater_than_max_duration_is_rejected(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["domain"]["min_total_duration"] = 500
    broken = tmp_path / "config.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="min_total_duration"):
        load_config(broken)


@pytest.mark.parametrize("filename", PROMPT_FILES)
def test_no_placeholder_survives_rendering(filename, app_config):
    rendered = render_prompt(filename, app_config)
    assert "$" not in rendered, f"{filename} still contains an unsubstituted placeholder"


def test_rendered_prompt_carries_the_config_values(app_config):
    storyboarder = render_prompt("storyboarder.md", app_config)
    for duration in app_config.domain.allowed_scene_durations:
        assert str(duration) in storyboarder

    engineer = render_prompt("prompt_engineer.md", app_config)
    for dimension in app_config.domain.veo_prompt_dimensions:
        assert dimension in engineer

    vfx = render_prompt("vfx_supervisor.md", app_config)
    for rule in app_config.domain.edit_rules:
        assert rule in vfx


def test_changing_the_config_changes_the_prompt(app_config):
    """The one-line-edit promise, exercised rather than asserted in prose."""
    before = render_prompt("storyboarder.md", app_config)
    assert "12" not in before

    tweaked = app_config.model_copy(deep=True)
    tweaked.domain.allowed_scene_durations = [12]
    after = render_prompt("storyboarder.md", tweaked)
    assert "12" in after


def test_unknown_placeholder_raises(tmp_path, app_config, monkeypatch):
    bad = tmp_path / "prompts"
    bad.mkdir()
    (bad / "bad.md").write_text("Durations: $not_a_real_variable", encoding="utf-8")
    monkeypatch.setattr("prompt_loader.PROMPTS_DIR", bad)

    with pytest.raises(KeyError, match="not_a_real_variable"):
        render_prompt("bad.md", app_config)


def test_preload_covers_every_configured_agent(app_config):
    rendered = preload_all_prompts(app_config)
    expected = {
        app_config.agents.scriptwriter.prompt_file,
        app_config.agents.storyboarder.prompt_file,
        app_config.agents.prompt_engineer.prompt_file,
        app_config.agents.vfx_supervisor.prompt_file,
    }
    assert set(rendered) == expected
