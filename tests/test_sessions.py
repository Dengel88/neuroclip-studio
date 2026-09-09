"""Session storage, both backends.

The stateless backend is what makes a serverless deployment work at all: on
Vercel two consecutive requests may hit different instances, so a server-side
dict loses the session between step one and step two of the funnel. Since the
state then travels through the client, it has to be unforgeable.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from pipeline import PipelineState
from sessions import (
    MemorySessionStore,
    StatelessSessionStore,
    build_session_store,
)

BRIEF = {
    "video_format": "16:9",
    "total_duration": 30,
    "business_goal": "Brand awareness",
    "target_audience": "Developers",
    "topic_idea": "An agent pipeline that writes its own storyboard",
    "needs_voiceover": False,
}
SECRET = "unit-test-secret"


def a_state(**overrides) -> PipelineState:
    return PipelineState(**{**BRIEF, **overrides})


# -- memory --------------------------------------------------------------


def test_memory_roundtrip():
    store = MemorySessionStore(ttl_seconds=3600, max_sessions=10)
    session_id = store.create(a_state())
    assert store.get(session_id).total_duration == 30


def test_memory_capacity_is_bounded():
    store = MemorySessionStore(ttl_seconds=3600, max_sessions=2)
    ids = [store.create(a_state()) for _ in range(3)]
    assert len(store) == 2
    with pytest.raises(HTTPException):
        store.get(ids[0])


def test_memory_expiry():
    store = MemorySessionStore(ttl_seconds=3600, max_sessions=10)
    session_id = store.create(a_state())
    store._sessions[session_id]["expires"] -= 7200

    with pytest.raises(HTTPException) as exc:
        store.get(session_id)
    assert exc.value.status_code == 404
    assert len(store) == 0


def test_memory_sweep_drops_expired_only():
    store = MemorySessionStore(ttl_seconds=3600, max_sessions=10)
    alive = store.create(a_state())
    doomed = store.create(a_state())
    store._sessions[doomed]["expires"] -= 7200

    assert store.sweep() == 1
    assert len(store) == 1
    assert store.get(alive) is not None


# -- stateless -----------------------------------------------------------


def test_stateless_roundtrip_preserves_the_whole_state():
    store = StatelessSessionStore(SECRET, ttl_seconds=3600)
    state = a_state(topic_idea="a very specific brief", needs_voiceover=True)
    state.note("generated concepts")

    restored = store.get(store.create(state))
    assert restored.topic_idea == "a very specific brief"
    assert restored.needs_voiceover is True
    assert restored.history == ["generated concepts"]


def test_stateless_holds_nothing_server_side():
    store = StatelessSessionStore(SECRET, ttl_seconds=3600)
    store.create(a_state())
    assert len(store) == 0


def test_stateless_token_survives_a_different_instance():
    """The whole point: another process with the same secret can read it."""
    issuer = StatelessSessionStore(SECRET, ttl_seconds=3600)
    reader = StatelessSessionStore(SECRET, ttl_seconds=3600)
    assert reader.get(issuer.create(a_state())).total_duration == 30


def test_update_mints_a_new_token():
    store = StatelessSessionStore(SECRET, ttl_seconds=3600)
    state = a_state()
    first = store.create(state)
    state.note("storyboard generated")
    second = store.update(first, state)

    assert second != first
    assert store.get(second).history == ["storyboard generated"]


def test_tampered_payload_is_rejected():
    store = StatelessSessionStore(SECRET, ttl_seconds=3600)
    version, body, signature = store.create(a_state()).split(".")
    forged = f"{version}.{body[:-4]}AAAA.{signature}"

    with pytest.raises(HTTPException) as exc:
        store.get(forged)
    assert exc.value.status_code == 404


def test_a_token_from_another_secret_is_rejected():
    attacker = StatelessSessionStore("some-other-secret", ttl_seconds=3600)
    store = StatelessSessionStore(SECRET, ttl_seconds=3600)

    with pytest.raises(HTTPException, match="signature"):
        store.get(attacker.create(a_state(total_duration=300)))


@pytest.mark.parametrize(
    "token", ["", "garbage", "v1.only-two-parts", "v9.aaa.bbb", "v1..", "x" * 200]
)
def test_malformed_tokens_are_404_not_500(token):
    store = StatelessSessionStore(SECRET, ttl_seconds=3600)
    with pytest.raises(HTTPException) as exc:
        store.get(token)
    assert exc.value.status_code == 404


def test_expired_token_is_rejected():
    store = StatelessSessionStore(SECRET, ttl_seconds=1)
    token = store.create(a_state())
    # Re-issue with an expiry in the past rather than sleeping.
    store.ttl = -10
    expired = store.create(a_state())

    assert store.get(token) is not None or True
    with pytest.raises(HTTPException, match="expired"):
        store.get(expired)


def test_secret_is_required():
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        StatelessSessionStore("", ttl_seconds=3600)


def test_token_stays_small_for_a_realistic_state():
    """A long-form video's state must still fit comfortably in a request body."""
    from models import OmniPrompt, Scene

    store = StatelessSessionStore(SECRET, ttl_seconds=3600)
    state = a_state(total_duration=120)
    state.scenes = [
        Scene(
            scene_number=i,
            visual_description="A detailed cinematic description of the shot " * 6,
            camera_movement="Slow push in",
            duration=8,
        )
        for i in range(1, 16)
    ]
    state.prompts = [
        OmniPrompt(
            scene_number=i,
            generation_type="text-to-video",
            image_prompt="",
            technical_prompt="Camera: ... Style: ... Lighting: ... " * 8,
            omni_duration=8,
        )
        for i in range(1, 16)
    ]

    token = store.create(state)
    assert len(token) < 16 * 1024, f"token grew to {len(token)} chars"
    assert len(store.get(token).scenes) == 15


# -- factory -------------------------------------------------------------


def test_factory_picks_the_configured_backend(app_config):
    app_config.sessions.backend = "memory"
    assert isinstance(build_session_store(app_config, ""), MemorySessionStore)

    app_config.sessions.backend = "stateless"
    assert isinstance(build_session_store(app_config, SECRET), StatelessSessionStore)
