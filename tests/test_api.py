"""HTTP contract: the shape `index.html` actually posts, limits, and sessions.

Round two of the review found the funnel returning 422 on step two because the
frontend posted a body while the endpoint declared query parameters. These tests
pin the contract the browser relies on, against both session backends - the
in-memory one used behind uvicorn and the stateless one required on serverless.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

import main
from config import config
from llm.mock import MockProvider

BRIEF = {
    "video_format": "16:9",
    "total_duration": 30,
    "business_goal": "Brand awareness",
    "target_audience": "Developers",
    "topic_idea": "An agent pipeline that writes its own storyboard",
    "needs_voiceover": False,
}


@pytest.fixture(params=["memory", "stateless"])
def client(request, monkeypatch):
    """A client wired to the offline provider, once per session backend."""
    from sessions import MemorySessionStore, StatelessSessionStore

    provider = MockProvider()
    monkeypatch.setattr(main.orchestrator, "_provider", provider)
    monkeypatch.setattr(main, "get_llm_provider", lambda *a, **k: provider)
    monkeypatch.setattr(config.rate_limit, "enabled", False)

    if request.param == "memory":
        store = MemorySessionStore(config.sessions.ttl_seconds, config.sessions.max_sessions)
    else:
        store = StatelessSessionStore("unit-test-secret", config.sessions.ttl_seconds)
    monkeypatch.setattr(main, "session_store", store)
    main.limiter.clear()

    with TestClient(main.app) as test_client:
        test_client.provider = provider
        test_client.backend = request.param
        yield test_client


def funnel(client, upto: str = "prompts") -> dict:
    """Walk the funnel, carrying the session id forward like the browser does."""
    response = client.post("/api/generate-concepts", json=BRIEF)
    assert response.status_code == 200, response.text
    data = response.json()
    session_id = data["session_id"]
    if upto == "concepts":
        return {"session_id": session_id, "concepts": data["concepts"]}

    response = client.post(
        "/api/generate-storyboard",
        json={"session_id": session_id, "concept_id": data["concepts"][0]["id"]},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    session_id = data["session_id"]
    if upto == "storyboard":
        return {"session_id": session_id, "scenes": data["scenes"]}

    response = client.post("/api/generate-prompts", json={"session_id": session_id})
    assert response.status_code == 200, response.text
    data = response.json()
    return {"session_id": data["session_id"], "prompts": data["prompts"]}


# -- open access ---------------------------------------------------------


def test_root_needs_no_credentials(client):
    """The demo is deliberately open - no login wall."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Neuroclip Studio" in response.text


def test_health_reports_the_active_backends(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["sessions_backend"] in ("memory", "stateless")


def test_config_endpoint_matches_the_yaml(client):
    body = client.get("/api/config").json()
    assert body["allowed_scene_durations"] == config.domain.allowed_scene_durations
    assert body["supported_aspect_ratios"] == config.domain.supported_aspect_ratios


# -- the funnel ----------------------------------------------------------


def test_full_funnel_over_http(client):
    storyboard = funnel(client, upto="storyboard")
    assert sum(s["duration"] for s in storyboard["scenes"]) == BRIEF["total_duration"]

    result = funnel(client)
    assert result["prompts"]
    for dimension in config.domain.veo_prompt_dimensions:
        assert dimension in result["prompts"][0]["technical_prompt"]


def test_every_response_carries_a_usable_session_id(client):
    """The stateless backend mints a new token per step; the browser must adopt it."""
    result = funnel(client, upto="storyboard")
    follow_up = client.post("/api/generate-prompts", json={"session_id": result["session_id"]})
    assert follow_up.status_code == 200, follow_up.text


def test_scene_edits_reach_the_server_state(client):
    storyboard = funnel(client, upto="storyboard")
    response = client.post(
        "/api/generate-prompts",
        json={
            "session_id": storyboard["session_id"],
            "scene_descriptions": {"1": "A hand-written shot"},
        },
    )
    assert response.status_code == 200, response.text

    state = main.session_store.get(response.json()["session_id"])
    assert state.scenes[0].visual_description == "A hand-written shot"


def test_editing_an_unknown_scene_is_a_400(client):
    storyboard = funnel(client, upto="storyboard")
    response = client.post(
        "/api/generate-prompts",
        json={"session_id": storyboard["session_id"], "scene_descriptions": {"99": "nope"}},
    )
    assert response.status_code == 400
    assert "99" in response.json()["detail"]


def test_retake_endpoint(client):
    storyboard = funnel(client, upto="storyboard")
    response = client.post(
        "/api/regenerate-scene",
        json={"session_id": storyboard["session_id"], "scene_number": 2, "note": "wider shot"},
    )
    assert response.status_code == 200, response.text
    after = response.json()["scenes"]
    assert len(after) == len(storyboard["scenes"])
    assert sum(s["duration"] for s in after) == BRIEF["total_duration"]


# -- input validation ----------------------------------------------------


def test_unknown_session_is_404(client):
    response = client.post("/api/generate-prompts", json={"session_id": "does-not-exist"})
    assert response.status_code == 404


def test_impossible_duration_is_400_not_500(client):
    response = client.post("/api/generate-concepts", json={**BRIEF, "total_duration": 15})
    assert response.status_code == 400
    assert "cannot be composed" in response.json()["detail"]


def test_oversized_input_is_rejected(client):
    response = client.post(
        "/api/generate-concepts",
        json={**BRIEF, "topic_idea": "x" * (config.limits.max_long_text_chars + 1)},
    )
    assert response.status_code == 422


def test_empty_field_is_rejected(client):
    response = client.post("/api/generate-concepts", json={**BRIEF, "business_goal": ""})
    assert response.status_code == 422


# -- rate limiting -------------------------------------------------------
# With no login, this is the only thing between a stranger and the API budget.


def test_generation_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(config.rate_limit, "enabled", True)
    monkeypatch.setattr(config.rate_limit, "generation_requests_per_minute", 2)
    main.limiter.clear()

    codes = [client.post("/api/generate-concepts", json=BRIEF).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    main.limiter.clear()


def test_rate_limit_is_keyed_by_forwarded_client_ip(client, monkeypatch):
    """Behind a proxy the socket address is the proxy - everyone would share a bucket."""
    monkeypatch.setattr(config.rate_limit, "enabled", True)
    monkeypatch.setattr(config.rate_limit, "generation_requests_per_minute", 1)
    main.limiter.clear()

    first = client.post(
        "/api/generate-concepts", json=BRIEF, headers={"X-Forwarded-For": "203.0.113.1"}
    )
    same_caller = client.post(
        "/api/generate-concepts", json=BRIEF, headers={"X-Forwarded-For": "203.0.113.1"}
    )
    other_caller = client.post(
        "/api/generate-concepts", json=BRIEF, headers={"X-Forwarded-For": "203.0.113.9"}
    )

    assert first.status_code == 200
    assert same_caller.status_code == 429
    assert other_caller.status_code == 200
    main.limiter.clear()


def test_rate_limit_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(config.rate_limit, "enabled", False)
    main.limiter.clear()
    codes = [client.post("/api/generate-concepts", json=BRIEF).status_code for _ in range(4)]
    assert set(codes) == {200}


# -- image storage modes -------------------------------------------------


def test_base64_storage_never_touches_the_disk(client, monkeypatch):
    monkeypatch.setattr(config.images, "storage", "base64")
    result = funnel(client)

    inline = [p for p in result["prompts"] if p["generation_type"] == "image-to-video"]
    assert inline, "the mock router should produce at least one image-to-video scene"
    assert all(p.get("image_url") is None for p in inline)
    assert all(base64.b64decode(p["image_base64"]) for p in inline)


def test_images_are_not_stored_in_the_session(client, monkeypatch):
    """A base64 JPEG inside a stateless token would blow past the size limit."""
    monkeypatch.setattr(config.images, "storage", "base64")
    result = funnel(client)

    state = main.session_store.get(result["session_id"])
    assert all(p.image_base64 is None for p in state.prompts)
    assert all(p.image_url is None for p in state.prompts)


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
