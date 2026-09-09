"""Loading of the system instructions in `prompts/*.md`.

The markdown files carry `$placeholders` (`$allowed_durations`, `$veo_template`,
...). They are filled from `config.prompt_variables()` with
`string.Template.substitute` - deliberately not `safe_substitute`: an unfilled
placeholder is a bug that must surface at load time, not a literal `$foo`
shipped to the model inside a system instruction.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from config import PROMPTS_DIR, AppConfig, config


def render_prompt(filename: str, app_config: AppConfig | None = None) -> str:
    """Read `prompts/<filename>` and substitute every placeholder."""
    active = app_config or config
    path = Path(PROMPTS_DIR) / filename
    raw = path.read_text(encoding="utf-8")
    try:
        return Template(raw).substitute(active.prompt_variables())
    except KeyError as exc:
        raise KeyError(
            f"Prompt '{filename}' references unknown placeholder ${exc.args[0]}. "
            f"Known placeholders: {', '.join(sorted(active.prompt_variables()))}."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"Prompt '{filename}' contains a malformed '$' sequence: {exc}. "
            "Escape a literal dollar sign as '$$'."
        ) from exc


@lru_cache(maxsize=None)
def load_prompt(filename: str) -> str:
    """Cached `render_prompt` against the process-wide config."""
    return render_prompt(filename)


def preload_all_prompts(app_config: AppConfig | None = None) -> dict[str, str]:
    """Render every configured agent prompt once, at startup.

    Turns a typo in a prompt file into a startup failure rather than a 500 on
    the third step of the funnel.
    """
    active = app_config or config
    return {
        agent.prompt_file: render_prompt(agent.prompt_file, active)
        for agent in (
            active.agents.scriptwriter,
            active.agents.storyboarder,
            active.agents.prompt_engineer,
            active.agents.vfx_supervisor,
        )
    }
