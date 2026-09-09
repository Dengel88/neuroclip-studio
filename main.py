"""HTTP layer.

Endpoints are thin: they rate-limit, look up the session and hand over to
`pipeline.py`. All business rules live in `domain.py`, all model access in
`llm/`, all session storage in `sessions.py`.

The demo is deliberately open - no login. That makes the rate limiter the only
thing standing between a stranger with the URL and the API budget, so it is on
by default and keyed by client IP.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
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
from sessions import MemorySessionStore, build_session_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("NeuroclipStudio.API")

load_dotenv()

SESSION_SECRET = os.getenv("SESSION_SECRET", "")


# ===========================================================================
# Rate limiting
# ===========================================================================


class FixedWindowLimiter:
    """Per-caller fixed-window counter.

    Deliberately simple and in-process: it protects the API-key budget of a
    single instance, not a fleet. Behind more than one worker or on serverless,
    each instance counts separately - see the README.
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
orchestrator = PipelineOrchestrator()


def startup_problems() -> list[str]:
    """Misconfigurations that stop the app doing useful work.

    Collected rather than raised. A deployment that dies at import returns an
    opaque 500 and tells the operator nothing; one that starts and *says* what
    is missing can be fixed without digging through platform logs.
    """
    problems: list[str] = []
    if config.sessions.backend == "stateless" and not SESSION_SECRET:
        problems.append(
            "SESSION_SECRET is not set, and sessions.backend is 'stateless' "
            "(the default on serverless hosts). Generate one with "
            '`python -c "import secrets; print(secrets.token_hex(32))"` and add '
            "it to the environment."
        )
    if config.provider == "gemini" and not any(
        os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 10)
    ):
        problems.append(
            "No GEMINI_API_KEY_1..9 found in the environment. Add at least one, "
            "or set LLM_PROVIDER=mock to run offline."
        )
    return problems


PROBLEMS = startup_problems()
# With a broken config the store cannot be built; endpoints answer 503 with the
# reason instead, and the page itself still loads.
session_store = None if PROBLEMS else build_session_store(config, SESSION_SECRET)


def require_ready() -> None:
    if PROBLEMS:
        raise HTTPException(
            status_code=503,
            detail="The deployment is not configured yet: " + " | ".join(PROBLEMS),
        )


def client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind Vercel and most proxies the socket address is the proxy, so the
    first hop of X-Forwarded-For is the real client. It is spoofable by
    definition - this is a budget guard, not an access control.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(bucket: str, limit_attribute: str):
    """Dependency factory: one counter per bucket, per client.

    The limit is read from the config on every request rather than captured
    here, so the configured number is the number actually enforced.
    """

    def dependency(request: Request) -> str:
        caller = client_key(request)
        if not config.rate_limit.enabled:
            return caller
        limit = getattr(config.rate_limit, limit_attribute)
        if not limiter.check(f"{bucket}:{caller}", limit):
            logger.warning("ratelimit.rejected bucket=%s", bucket)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: at most {limit} {bucket} requests per minute.",
                headers={"Retry-After": "60"},
            )
        return caller

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


async def _session_housekeeping() -> None:
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
    for problem in PROBLEMS:
        logger.error("startup.misconfigured %s", problem)
    # Turn a broken prompt placeholder into a startup failure, not a 500 later.
    preload_all_prompts()
    logger.info(
        "startup provider=%s sessions=%s images=%s ready=%s",
        config.provider, config.sessions.backend, config.images.storage, not PROBLEMS,
    )
    tasks = []
    if isinstance(session_store, MemorySessionStore):
        tasks.append(asyncio.create_task(_session_housekeeping()))
    if config.images.storage == "static":
        tasks.append(asyncio.create_task(_image_housekeeping()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()


app = FastAPI(title="Neuroclip Studio - AI Video Production Platform", lifespan=lifespan)

if config.images.storage == "static":
    # A read-only filesystem must degrade, not crash. The previous version
    # called os.makedirs() unguarded at import time, which is exactly how a
    # serverless deployment died with FUNCTION_INVOCATION_FAILED before the
    # first request was ever served.
    try:
        config.images_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            f"/{config.images.directory}",
            StaticFiles(directory=config.images_dir),
            name="static",
        )
    except OSError as exc:
        logger.error(
            "images.static_unavailable error=%s - falling back to inline base64. "
            "Set IMAGES_STORAGE=base64 to make this explicit.",
            exc,
        )
        config.images.storage = "base64"


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
async def read_root(caller: str = Depends(read_guard)):
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    """Deployment self-report. The first thing to curl when a host misbehaves."""
    return {
        "status": "ok" if not PROBLEMS else "misconfigured",
        "provider": config.provider,
        "sessions_backend": config.sessions.backend,
        "images_storage": config.images.storage,
        "problems": PROBLEMS,
    }


@app.get("/api/config")
async def public_config(caller: str = Depends(read_guard)):
    """Domain constraints the browser needs, so no constant is duplicated there."""
    return {
        "allowed_scene_durations": config.domain.allowed_scene_durations,
        "supported_aspect_ratios": config.domain.supported_aspect_ratios,
        "min_total_duration": config.domain.min_total_duration,
        "max_total_duration": config.domain.max_total_duration,
    }


@app.post("/api/generate-concepts")
async def generate_concepts(
    request: ScriptwriterInput,
    caller: str = Depends(generate_guard),
    _ready: None = Depends(require_ready),
):
    state = PipelineState(**request.model_dump())
    state = await orchestrator.generate_concepts(state)
    session_id = session_store.create(state)
    return JSONResponse(
        content={"session_id": session_id, "concepts": [c.model_dump() for c in state.concepts]}
    )


@app.post("/api/generate-storyboard")
async def generate_storyboard(
    request: StoryboardRequest,
    caller: str = Depends(generate_guard),
    _ready: None = Depends(require_ready),
):
    state = session_store.get(request.session_id)
    state.selected_concept_id = request.concept_id
    state = await orchestrator.generate_storyboard(state)
    session_id = session_store.update(request.session_id, state)
    return JSONResponse(
        content={"session_id": session_id, "scenes": [s.model_dump() for s in state.scenes]}
    )


@app.post("/api/regenerate-scene")
async def regenerate_scene(
    request: RegenerateSceneRequest,
    caller: str = Depends(generate_guard),
    _ready: None = Depends(require_ready),
):
    state = session_store.get(request.session_id)
    state = await orchestrator.regenerate_scene(state, request.scene_number, request.note)
    session_id = session_store.update(request.session_id, state)
    return JSONResponse(
        content={"session_id": session_id, "scenes": [s.model_dump() for s in state.scenes]}
    )


@app.post("/api/generate-prompts")
async def generate_prompts(
    request: PromptsRequest,
    caller: str = Depends(generate_guard),
    _ready: None = Depends(require_ready),
):
    state = session_store.get(request.session_id)
    _apply_scene_edits(state, request.scene_descriptions)
    state = await orchestrator.generate_prompts(state)

    # The session is saved *before* the frames are attached: a base64 JPEG is
    # hundreds of kilobytes, and with the stateless backend the state is carried
    # by the client. Images belong in the response, not in the session.
    session_id = session_store.update(request.session_id, state)

    payload = [p.model_dump() for p in state.prompts]
    frames = await _render_reference_frames(state)
    for prompt in payload:
        frame = frames.get(prompt["scene_number"])
        if frame:
            prompt.update(frame)

    return JSONResponse(content={"session_id": session_id, "prompts": payload})


@app.post("/api/format-edit-prompt", response_model=VideoEditOutput)
async def format_edit_prompt(
    request: VideoEditInput,
    caller: str = Depends(generate_guard),
    _ready: None = Depends(require_ready),
):
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


async def _render_reference_frames(state: PipelineState) -> Dict[int, dict]:
    """Render still frames for the image-to-video scenes.

    Returns `{scene_number: {"image_url": ...}}` or `{"image_base64": ...}`,
    to be merged into the response. Storage mode comes from `images.storage`:
    `static` writes a JPEG and returns a URL, `base64` returns the bytes inline
    and never touches the disk - the only option on a read-only serverless
    filesystem.
    """
    provider = get_llm_provider(config.provider)
    frames: Dict[int, dict] = {}
    for prompt in state.prompts:
        if prompt.generation_type != "image-to-video" or not prompt.image_prompt:
            continue
        image_bytes = await provider.generate_image(prompt.image_prompt, state.video_format)
        if not image_bytes:
            continue
        if config.images.storage == "base64":
            frames[prompt.scene_number] = {
                "image_base64": base64.b64encode(image_bytes).decode("ascii")
            }
            continue
        filename = f"scene_{prompt.scene_number}_{secrets.token_hex(4)}.jpg"
        try:
            config.images_dir.mkdir(parents=True, exist_ok=True)
            (config.images_dir / filename).write_bytes(image_bytes)
            frames[prompt.scene_number] = {
                "image_url": f"/{config.images.directory}/{filename}"
            }
        except OSError as exc:
            logger.error("image.save_failed scene=%d error=%s", prompt.scene_number, exc)
    return frames


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
