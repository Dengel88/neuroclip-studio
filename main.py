"""HTTP layer.

Endpoints are thin: they authenticate, rate-limit, look up the session and hand
over to `pipeline.py`. All business rules live in `domain.py`, all model access
in `llm/`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, config
from errors import (
    DomainValidationError,
    InfeasibleBriefError,
    ProviderError,
    RepairExhaustedError,
)
from llm import get_llm_provider
from models import (
    PromptsRequest,
    RegenerateSceneRequest,
    ScriptwriterInput,
    StoryboardRequest,
    VideoEditInput,
    VideoEditOutput,
)
from pipeline import PipelineOrchestrator, PipelineState
from prompt_loader import load_prompt, preload_all_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("NeuroclipStudio.API")

load_dotenv()

BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER", "")
BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD", "")
_MISSING_CREDENTIALS_MESSAGE = (
    "BASIC_AUTH_USER and BASIC_AUTH_PASSWORD must be set before starting the app. "
    "Copy .env.example to .env and fill them in. "
    "Refusing to serve an endpoint with open or default credentials."
)


# ===========================================================================
# Sessions
# ===========================================================================


class SessionStore:
    """In-memory pipeline state, TTL-bounded and capacity-bounded.

    Production needs Redis - see the README. The cap matters even here: without
    it an unauthenticated flood of step-1 requests is an unbounded memory leak.
    """

    def __init__(self, ttl_seconds: int, max_sessions: int):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: Dict[str, dict] = {}

    def create(self, state: PipelineState) -> str:
        self.sweep()
        if len(self._sessions) >= self.max_sessions:
            # Evict the session closest to expiry rather than refusing service.
            oldest = min(self._sessions, key=lambda k: self._sessions[k]["expires"])
            del self._sessions[oldest]
            logger.warning("session.evicted reason=capacity max=%d", self.max_sessions)
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "state": state,
            "expires": time.monotonic() + self.ttl,
        }
        return session_id

    def get(self, session_id: str) -> PipelineState:
        session = self._sessions.get(session_id)
        if session is None or session["expires"] < time.monotonic():
            self._sessions.pop(session_id, None)
            raise HTTPException(status_code=404, detail="Session not found or expired.")
        session["expires"] = time.monotonic() + self.ttl
        return session["state"]

    def update(self, session_id: str, state: PipelineState) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["state"] = state

    def sweep(self) -> int:
        now = time.monotonic()
        expired = [k for k, v in self._sessions.items() if v["expires"] < now]
        for key in expired:
            del self._sessions[key]
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)


# ===========================================================================
# Rate limiting
# ===========================================================================


class FixedWindowLimiter:
    """Per-caller fixed-window counter.

    Deliberately simple and in-process: it protects the API-key budget of a
    single instance, not a fleet. Behind more than one worker, move it to Redis.
    """

    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self._hits: Dict[str, list[float]] = {}

    def check(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        hits = [t for t in self._hits.get(key, []) if t > cutoff]
        if len(hits) >= limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True

    def clear(self) -> None:
        self._hits.clear()


limiter = FixedWindowLimiter()
session_store = SessionStore(config.sessions.ttl_seconds, config.sessions.max_sessions)
orchestrator = PipelineOrchestrator()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(HTTPBasic())) -> str:
    if not BASIC_AUTH_USER or not BASIC_AUTH_PASSWORD:
        # Belt and braces: startup already refuses, this closes the door if the
        # app is ever mounted into another ASGI application.
        logger.error("auth.misconfigured")
        raise HTTPException(status_code=503, detail="Authentication is not configured.")
    correct_user = secrets.compare_digest(
        credentials.username.encode("utf-8"), BASIC_AUTH_USER.encode("utf-8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"), BASIC_AUTH_PASSWORD.encode("utf-8")
    )
    if not (correct_user and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def rate_limit(bucket: str, limit_attribute: str):
    """Dependency factory: one counter per bucket, per authenticated user + IP.

    The limit is read from the config on every request rather than captured
    here, so the configured number is the number actually enforced.
    """

    def dependency(request: Request, username: str = Depends(verify_credentials)) -> str:
        if not config.rate_limit.enabled:
            return username
        limit = getattr(config.rate_limit, limit_attribute)
        client = request.client.host if request.client else "unknown"
        if not limiter.check(f"{bucket}:{username}:{client}", limit):
            logger.warning("ratelimit.rejected bucket=%s user=%s", bucket, username)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: at most {limit} {bucket} requests per minute.",
                headers={"Retry-After": "60"},
            )
        return username

    return dependency


read_guard = rate_limit("read", "requests_per_minute")
generate_guard = rate_limit("generation", "generation_requests_per_minute")


# ===========================================================================
# Housekeeping
# ===========================================================================


def cleanup_images(now: Optional[float] = None) -> int:
    """Delete rendered frames older than `images.retention_seconds`."""
    directory = config.images_dir
    if config.images.storage != "static" or not directory.exists():
        return 0
    deadline = (now or time.time()) - config.images.retention_seconds
    removed = 0
    for path in directory.glob("scene_*.jpg"):
        try:
            if path.stat().st_mtime < deadline:
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("cleanup.failed path=%s error=%s", path.name, exc)
    return removed


async def _housekeeping() -> None:
    while True:
        await asyncio.sleep(config.sessions.sweep_interval_seconds)
        expired = session_store.sweep()
        if expired:
            logger.info("session.swept expired=%d live=%d", expired, len(session_store))


async def _image_housekeeping() -> None:
    while True:
        await asyncio.sleep(config.images.cleanup_interval_seconds)
        removed = cleanup_images()
        if removed:
            logger.info("cleanup.images removed=%d", removed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not BASIC_AUTH_USER or not BASIC_AUTH_PASSWORD:
        raise RuntimeError(_MISSING_CREDENTIALS_MESSAGE)
    # Turn a broken prompt placeholder into a startup failure, not a 500 later.
    preload_all_prompts()
    logger.info("startup provider=%s sessions_ttl=%ds", config.provider, config.sessions.ttl_seconds)
    tasks = [asyncio.create_task(_housekeeping())]
    if config.images.storage == "static":
        tasks.append(asyncio.create_task(_image_housekeeping()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()


app = FastAPI(title="Neuroclip Studio - AI Video Production Platform", lifespan=lifespan)

if config.images.storage == "static":
    config.images_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        f"/{config.images.directory}",
        StaticFiles(directory=config.images_dir),
        name="static",
    )


# ===========================================================================
# Error translation
# ===========================================================================


@app.exception_handler(InfeasibleBriefError)
async def _infeasible(request: Request, exc: InfeasibleBriefError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DomainValidationError)
async def _domain_invalid(request: Request, exc: DomainValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RepairExhaustedError)
async def _repair_exhausted(request: Request, exc: RepairExhaustedError):
    logger.error("repair.failed detail=%s", exc)
    return JSONResponse(
        status_code=502,
        content={
            "detail": "The model could not produce a valid result for this step. "
            f"{exc.last_error}"
        },
    )


@app.exception_handler(ProviderError)
async def _provider_failed(request: Request, exc: ProviderError):
    logger.error("provider.failed detail=%s", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The upstream model API is unavailable. Please retry shortly."},
    )


# ===========================================================================
# Endpoints
# ===========================================================================


@app.get("/", response_class=HTMLResponse)
async def read_root(username: str = Depends(read_guard)):
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": config.provider,
        "sessions": len(session_store),
    }


@app.get("/api/config")
async def public_config(username: str = Depends(read_guard)):
    """Domain constraints the browser needs, so no constant is duplicated there."""
    return {
        "allowed_scene_durations": config.domain.allowed_scene_durations,
        "supported_aspect_ratios": config.domain.supported_aspect_ratios,
        "min_total_duration": config.domain.min_total_duration,
        "max_total_duration": config.domain.max_total_duration,
    }


@app.post("/api/generate-concepts")
async def generate_concepts(
    request: ScriptwriterInput, username: str = Depends(generate_guard)
):
    state = PipelineState(**request.model_dump())
    state = await orchestrator.generate_concepts(state)
    session_id = session_store.create(state)
    return JSONResponse(
        content={"session_id": session_id, "concepts": [c.model_dump() for c in state.concepts]}
    )


@app.post("/api/generate-storyboard")
async def generate_storyboard(
    request: StoryboardRequest, username: str = Depends(generate_guard)
):
    state = session_store.get(request.session_id)
    state.selected_concept_id = request.concept_id
    state = await orchestrator.generate_storyboard(state)
    session_store.update(request.session_id, state)
    return JSONResponse(
        content={
            "session_id": request.session_id,
            "scenes": [s.model_dump() for s in state.scenes],
        }
    )


@app.post("/api/regenerate-scene")
async def regenerate_scene(
    request: RegenerateSceneRequest, username: str = Depends(generate_guard)
):
    state = session_store.get(request.session_id)
    state = await orchestrator.regenerate_scene(state, request.scene_number, request.note)
    session_store.update(request.session_id, state)
    return JSONResponse(
        content={
            "session_id": request.session_id,
            "scenes": [s.model_dump() for s in state.scenes],
        }
    )


@app.post("/api/generate-prompts")
async def generate_prompts(request: PromptsRequest, username: str = Depends(generate_guard)):
    state = session_store.get(request.session_id)
    _apply_scene_edits(state, request.scene_descriptions)
    state = await orchestrator.generate_prompts(state)
    await _attach_reference_frames(state)
    session_store.update(request.session_id, state)
    return JSONResponse(
        content={
            "session_id": request.session_id,
            "prompts": [p.model_dump() for p in state.prompts],
        }
    )


@app.post("/api/format-edit-prompt", response_model=VideoEditOutput)
async def format_edit_prompt(request: VideoEditInput, username: str = Depends(generate_guard)):
    agent = config.agents.vfx_supervisor
    provider = get_llm_provider(config.provider)
    user = (
        f"Original Prompt: {request.original_prompt}\n"
        f"User Edit Request: {request.user_request}"
    )
    return await provider.generate_structured(
        load_prompt(agent.prompt_file),
        user,
        VideoEditOutput,
        agent.model_alias,
        agent.temperature,
    )


def _apply_scene_edits(state: PipelineState, edits: Optional[Dict[int, str]]) -> None:
    """Merge the user's manual shot rewrites into the server-side storyboard."""
    if not edits:
        return
    known = {s.scene_number for s in state.scenes}
    unknown = sorted(set(edits) - known)
    if unknown:
        raise DomainValidationError(
            f"Cannot edit scene(s) {unknown}: this storyboard has scenes {sorted(known)}."
        )
    limit = config.limits.max_long_text_chars
    for scene in state.scenes:
        new_text = edits.get(scene.scene_number)
        if new_text is None or new_text == scene.visual_description:
            continue
        if len(new_text) > limit:
            raise DomainValidationError(
                f"Scene {scene.scene_number} description exceeds {limit} characters."
            )
        scene.visual_description = new_text
        state.note(f"user edited scene {scene.scene_number}")


async def _attach_reference_frames(state: PipelineState) -> None:
    """Render still frames for the image-to-video scenes.

    Storage mode comes from `images.storage`: `static` writes a JPEG and returns
    a URL (fine on a normal host, lossy on an ephemeral filesystem), `base64`
    returns the bytes inline and never touches the disk.
    """
    provider = get_llm_provider(config.provider)
    for prompt in state.prompts:
        if prompt.generation_type != "image-to-video" or not prompt.image_prompt:
            continue
        image_bytes = await provider.generate_image(prompt.image_prompt, state.video_format)
        if not image_bytes:
            continue
        if config.images.storage == "base64":
            prompt.image_base64 = base64.b64encode(image_bytes).decode("ascii")
            continue
        filename = f"scene_{prompt.scene_number}_{secrets.token_hex(4)}.jpg"
        try:
            config.images_dir.mkdir(parents=True, exist_ok=True)
            (config.images_dir / filename).write_bytes(image_bytes)
            prompt.image_url = f"/{config.images.directory}/{filename}"
        except OSError as exc:
            logger.error("image.save_failed scene=%d error=%s", prompt.scene_number, exc)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
