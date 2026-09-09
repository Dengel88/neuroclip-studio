"""Pydantic contracts.

Two families live here:

* *agent schemas* (``ScriptwriterOutput``, ``StoryboarderOutput``, ...) - handed
  to the LLM as ``response_schema`` and used to parse its answer;
* *HTTP request schemas* - what ``index.html`` posts. Their length and range
  limits come from ``config.yaml`` (``limits.*``, ``domain.*``), so the bounds
  advertised in the config are the bounds actually enforced.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from config import config

SHORT = config.limits.max_short_text_chars
LONG = config.limits.max_long_text_chars
# A session id is either a UUID (memory backend) or a signed state token
# (stateless backend, used on serverless). The cap bounds the second case;
# sessions.py enforces the same number when reading a token.
MAX_SESSION_ID_CHARS = 128 * 1024
_ALLOWED_DURATIONS = ", ".join(str(d) for d in config.domain.allowed_scene_durations)
_RATIOS = ", ".join(config.domain.supported_aspect_ratios)

# ==========================================
# AGENT 1: SCRIPTWRITER
# ==========================================


class ScriptwriterInput(BaseModel):
    """The brief. Posted by the browser to /api/generate-concepts."""

    video_format: str = Field(..., description=f"One of: {_RATIOS}")
    total_duration: int = Field(
        ...,
        ge=config.domain.min_total_duration,
        le=config.domain.max_total_duration,
        description="Total seconds, e.g. 30, 60",
    )
    business_goal: str = Field(..., min_length=1, max_length=SHORT)
    target_audience: str = Field(..., min_length=1, max_length=SHORT)
    topic_idea: str = Field(..., min_length=1, max_length=LONG)
    needs_voiceover: bool = Field(default=False)


class Concept(BaseModel):
    id: int = Field(..., description="Sequential id starting at 1")
    working_title: str = Field(..., description="Catchy, short title")
    logline: str = Field(..., description="1-2 sentences on the core visual and narrative idea")
    visual_style: str = Field(..., description="Specific style description suitable for AI")
    pacing: str = Field(..., description="Speed of the video")
    voiceover_tone: str = Field(
        ..., description="Tone of the voiceover. Empty string if no voiceover is required."
    )


class ScriptwriterOutput(BaseModel):
    concepts: List[Concept] = Field(
        ..., description=f"Exactly {config.domain.concepts_count} distinct video concepts."
    )


# ==========================================
# AGENT 2: STORYBOARDER
# ==========================================


class Scene(BaseModel):
    scene_number: int = Field(..., description="Sequential number of the scene (1, 2, 3, ...)")
    visual_description: str = Field(..., description="Detailed description of the shot.")
    camera_movement: str = Field(..., description="e.g. 'Slow pan right', 'Static', 'Zoom in'")
    duration: int = Field(..., description=f"Seconds. Strictly one of: {_ALLOWED_DURATIONS}.")


class StoryboarderOutput(BaseModel):
    scenes: List[Scene] = Field(..., description="All scenes making up the video, in order.")


# ==========================================
# AGENT 3: PROMPT ENGINEER
# ==========================================


class OmniPrompt(BaseModel):
    scene_number: int = Field(..., description="Matches the storyboard scene number")
    generation_type: str = Field(
        ..., description="Strictly 'text-to-video' or 'image-to-video'"
    )
    image_prompt: str = Field(
        ...,
        description="Detailed still-frame prompt when generation_type is "
        "'image-to-video'. Empty string otherwise.",
    )
    technical_prompt: str = Field(
        ...,
        description="Video prompt containing all "
        f"{len(config.domain.veo_prompt_dimensions)} labelled dimensions: "
        f"{', '.join(config.domain.veo_prompt_dimensions)}.",
    )
    omni_duration: int = Field(
        ...,
        description=f"Must equal the scene duration. Strictly one of: {_ALLOWED_DURATIONS}.",
    )
    image_url: Optional[str] = Field(
        default=None, description="Filled in by the backend, not by the model."
    )
    image_base64: Optional[str] = Field(
        default=None, description="Filled in by the backend when images.storage is 'base64'."
    )


class PromptEngineerOutput(BaseModel):
    prompts: List[OmniPrompt] = Field(..., description="One technical prompt per scene.")


# ==========================================
# AGENT 4: VFX SUPERVISOR
# ==========================================


class VideoEditInput(BaseModel):
    original_prompt: str = Field(..., min_length=1, max_length=LONG)
    user_request: str = Field(..., min_length=1, max_length=LONG)


class VideoEditOutput(BaseModel):
    optimized_edit_prompt: str = Field(
        ..., description=f"The rewritten prompt obeying all {len(config.domain.edit_rules)} rules."
    )


# ==========================================
# HTTP REQUESTS FOR THE LATER PIPELINE STAGES
# ==========================================
# session_id and concept_id travel in the body, not the query string, so the
# browser can keep posting plain JSON for every step of the funnel.


class StoryboardRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=MAX_SESSION_ID_CHARS)
    concept_id: int = Field(..., ge=0)


class PromptsRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=MAX_SESSION_ID_CHARS)
    # The browser lets the user tweak the shot descriptions before the prompt
    # stage. Those edits are merged into the server-side state, keyed by scene
    # number, so the state stays authoritative.
    scene_descriptions: Optional[Dict[int, str]] = Field(default=None)


class RegenerateSceneRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=MAX_SESSION_ID_CHARS)
    scene_number: int = Field(..., ge=1)
    note: Optional[str] = Field(
        default=None,
        max_length=SHORT,
        description="Optional instruction for the retake, e.g. 'make it wider'.",
    )
