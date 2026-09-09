"""HTTP contract: the shape `index.html` actually posts, auth, and limits.

Round two of the review found the funnel returning 422 on step two because the
frontend posted a body while the endpoint declared query parameters. These tests
pin the contract the browser relies on.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

import main
from config import config
from llm.mock import MockProvider

AUTH = ("test-user", "test-password")
BRIEF = {
    "video_format": "16:9",
    "total_duration": 30,
    "business_goal": "Brand awareness",
    "target_audience": "Developers",
    "topic_idea": "An agent pipeline that writes its own storyboard",
    "needs_voiceover": False,
}


@pytest.fixture
def client(monkeypatch):
    """A client wired to the offline provider, with rate limiting out of the way."""
    provider = MockProvider()
    monkeypatch.setattr(main.orchestrator, "_provider", provider)
    monkeypatch.setattr(main, "get_llm_provider", lambda *a, **k: provider)
    monkeypatch.setattr(config.rate_limit, "enabled", False)
    main.limiter.clear()
    with TestClient(main.app) as test_client:
        test_client.provider = provider
        yield test_client


def start_session(client) -> str:
    response = client.post("/api/generate-concepts", json=BRIEF, auth=AUTH)
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


# -- auth ----------------------------------------------------------------


def test_root_requires_auth(client):
    assert client.get("/").status_code == 401


def test_wrong_password_is_rejected(client):
    assert client.get("/", auth=("test-user", "nope")).status_code == 401


def test_correct_credentials_serve_the_app(client):
    response = client.get("/", auth=AUTH)
    assert response.status_code == 200
    assert "Neuroclip Studio" in response.text


def test_health_is_open(client):
    assert client.get("/api/health").json()["status"] == "ok"


# -- the funnel ----------------------------------------------------------


def test_full_funnel_over_http(client):
    session_id = start_session(client)

    concepts = client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).json()["concepts"]

    storyboard = client.post(
        "/api/generate-storyboard",
        json={"session_id": session_id, "concept_id": concepts[0]["id"]},
        auth=AUTH,
    )
    assert storyboard.status_code == 200, storyboard.text
    scenes = storyboard.json()["scenes"]
    assert sum(s["duration"] for s in scenes) == BRIEF["total_duration"]

    prompts = client.post(
        "/api/generate-prompts", json={"session_id": session_id}, auth=AUTH
    )
    assert prompts.status_code == 200, prompts.text
    payload = prompts.json()["prompts"]
    assert len(payload) == len(scenes)
    for dimension in config.domain.veo_prompt_dimensions:
        assert dimension in payload[0]["technical_prompt"]


def test_scene_edits_reach_the_server_state(client):
    session_id = start_session(client)
    concepts = client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).json()["concepts"]
    client.post(
        "/api/generate-storyboard",
        json={"session_id": session_id, "concept_id": concepts[0]["id"]},
        auth=AUTH,
    )

    response = client.post(
        "/api/generate-prompts",
        json={"session_id": session_id, "scene_descriptions": {"1": "A hand-written shot"}},
        auth=AUTH,
    )
    assert response.status_code == 200, response.text
    state = main.session_store.get(session_id)
    assert state.scenes[0].visual_description == "A hand-written shot"


def test_editing_an_unknown_scene_is_a_400(client):
    session_id = start_session(client)
    concepts = client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).json()["concepts"]
    client.post(
        "/api/generate-storyboard",
        json={"session_id": session_id, "concept_id": concepts[0]["id"]},
        auth=AUTH,
    )
    response = client.post(
        "/api/generate-prompts",
        json={"session_id": session_id, "scene_descriptions": {"99": "nope"}},
        auth=AUTH,
    )
    assert response.status_code == 400
    assert "99" in response.json()["detail"]


def test_retake_endpoint(client):
    session_id = start_session(client)
    concepts = client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).json()["concepts"]
    before = client.post(
        "/api/generate-storyboard",
        json={"session_id": session_id, "concept_id": concepts[0]["id"]},
        auth=AUTH,
    ).json()["scenes"]

    response = client.post(
        "/api/regenerate-scene",
        json={"session_id": session_id, "scene_number": 2, "note": "wider shot"},
        auth=AUTH,
    )
    assert response.status_code == 200, response.text
    after = response.json()["scenes"]
    assert len(after) == len(before)
    assert sum(s["duration"] for s in after) == BRIEF["total_duration"]


# -- input validation ----------------------------------------------------


def test_unknown_session_is_404(client):
    response = client.post(
        "/api/generate-prompts", json={"session_id": "does-not-exist"}, auth=AUTH
    )
    assert response.status_code == 404


def test_impossible_duration_is_400_not_500(client):
    response = client.post(
        "/api/generate-concepts", json={**BRIEF, "total_duration": 15}, auth=AUTH
    )
    assert response.status_code == 400
    assert "cannot be composed" in response.json()["detail"]


def test_oversized_input_is_rejected(client):
    response = client.post(
        "/api/generate-concepts",
        json={**BRIEF, "topic_idea": "x" * (config.limits.max_long_text_chars + 1)},
        auth=AUTH,
    )
    assert response.status_code == 422


def test_empty_field_is_rejected(client):
    response = client.post(
        "/api/generate-concepts", json={**BRIEF, "business_goal": ""}, auth=AUTH
    )
    assert response.status_code == 422


def test_config_endpoint_matches_the_yaml(client):
    body = client.get("/api/config", auth=AUTH).json()
    assert body["allowed_scene_durations"] == config.domain.allowed_scene_durations
    assert body["supported_aspect_ratios"] == config.domain.supported_aspect_ratios


# -- rate limiting -------------------------------------------------------


def test_generation_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(config.rate_limit, "enabled", True)
    monkeypatch.setattr(config.rate_limit, "generation_requests_per_minute", 2)
    main.limiter.clear()

    codes = [
        client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).status_code
        for _ in range(3)
    ]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    main.limiter.clear()


def test_rate_limit_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(config.rate_limit, "enabled", False)
    main.limiter.clear()
    codes = [
        client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).status_code
        for _ in range(4)
    ]
    assert set(codes) == {200}


# -- sessions ------------------------------------------------------------


def test_session_capacity_is_bounded(monkeypatch):
    store = main.SessionStore(ttl_seconds=3600, max_sessions=2)
    from pipeline import PipelineState

    ids = [store.create(PipelineState(**BRIEF)) for _ in range(3)]
    assert len(store) == 2
    with pytest.raises(Exception):
        store.get(ids[0])


def test_expired_session_is_gone():
    from fastapi import HTTPException

    from pipeline import PipelineState

    store = main.SessionStore(ttl_seconds=3600, max_sessions=10)
    session_id = store.create(PipelineState(**BRIEF))
    # Age the session past its TTL instead of moving the process clock.
    store._sessions[session_id]["expires"] -= 7200

    with pytest.raises(HTTPException) as exc:
        store.get(session_id)
    assert exc.value.status_code == 404
    assert len(store) == 0


def test_sweep_drops_expired_sessions_only():
    from pipeline import PipelineState

    store = main.SessionStore(ttl_seconds=3600, max_sessions=10)
    alive = store.create(PipelineState(**BRIEF))
    doomed = store.create(PipelineState(**BRIEF))
    store._sessions[doomed]["expires"] -= 7200

    assert store.sweep() == 1
    assert len(store) == 1
    assert store.get(alive) is not None


# -- image storage modes -------------------------------------------------


def test_base64_storage_never_touches_the_disk(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config.images, "storage", "base64")
    session_id = start_session(client)
    concepts = client.post("/api/generate-concepts", json=BRIEF, auth=AUTH).json()["concepts"]
    client.post(
        "/api/generate-storyboard",
        json={"session_id": session_id, "concept_id": concepts[0]["id"]},
        auth=AUTH,
    )
    prompts = client.post(
        "/api/generate-prompts", json={"session_id": session_id}, auth=AUTH
    ).json()["prompts"]

    inline = [p for p in prompts if p["generation_type"] == "image-to-video"]
    assert inline, "the mock router should produce at least one image-to-video scene"
    assert all(p["image_url"] is None for p in inline)
    assert all(base64.b64decode(p["image_base64"]) for p in inline)


def test_image_cleanup_removes_only_old_files(monkeypatch, tmp_path):
    monkeypatch.setattr(config.images, "storage", "static")
    monkeypatch.setattr(config.images, "directory", str(tmp_path))
    monkeypatch.setattr(config.images, "retention_seconds", 100)

    fresh = tmp_path / "scene_1_aaaa.jpg"
    stale = tmp_path / "scene_2_bbbb.jpg"
    fresh.write_bytes(b"x")
    stale.write_bytes(b"x")
    import os

    os.utime(stale, (0, 0))

    assert main.cleanup_images() == 1
    assert fresh.exists() and not stale.exists()
