"""Pipeline session storage.

Two backends, chosen by `sessions.backend` in `config.yaml`:

* ``memory`` - a TTL- and capacity-bounded dict. Fine behind a long-lived
  process (uvicorn, Render, a container). Lost on restart, not shared between
  workers.
* ``stateless`` - the state travels with the client as a signed, compressed
  token. Required on serverless hosts such as Vercel, where consecutive
  requests may land on different instances and an in-memory dict silently
  loses the session between step one and step two of the funnel.

The stateless token is signed with HMAC-SHA256 and carries an expiry. It is
not encrypted - it holds no secrets, only the brief and the generated
storyboard - but it cannot be forged, so a client cannot hand back a state
claiming five hundred scenes to burn the API budget.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
import zlib
from typing import Dict, Protocol

from fastapi import HTTPException

from models import MAX_SESSION_ID_CHARS
from pipeline import PipelineState

logger = logging.getLogger("NeuroclipStudio.Sessions")

TOKEN_VERSION = "v1"
# Generous next to a realistic state (~5 KB compressed) and small enough that a
# crafted token cannot be used as a decompression bomb. Shared with the request
# schema so the two cannot drift apart.
MAX_TOKEN_CHARS = MAX_SESSION_ID_CHARS
MAX_STATE_BYTES = 2 * 1024 * 1024


class SessionExpired(HTTPException):
    def __init__(self, detail: str = "Session not found or expired."):
        super().__init__(status_code=404, detail=detail)


class SessionStore(Protocol):
    def create(self, state: PipelineState) -> str: ...

    def get(self, session_id: str) -> PipelineState: ...

    def update(self, session_id: str, state: PipelineState) -> str:
        """Persist `state` and return the handle to use from now on.

        The memory backend returns the same id; the stateless backend returns a
        fresh token, because the token *is* the state.
        """


# ===========================================================================
# In-memory
# ===========================================================================


class MemorySessionStore:
    """TTL- and capacity-bounded dict.

    The capacity cap is not decoration: without it an open endpoint turns
    step-one requests into an unbounded memory leak.
    """

    backend = "memory"

    def __init__(self, ttl_seconds: int, max_sessions: int):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: Dict[str, dict] = {}

    def create(self, state: PipelineState) -> str:
        self.sweep()
        if len(self._sessions) >= self.max_sessions:
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
            raise SessionExpired()
        session["expires"] = time.monotonic() + self.ttl
        return session["state"]

    def update(self, session_id: str, state: PipelineState) -> str:
        if session_id in self._sessions:
            self._sessions[session_id]["state"] = state
        return session_id

    def sweep(self) -> int:
        now = time.monotonic()
        expired = [k for k, v in self._sessions.items() if v["expires"] < now]
        for key in expired:
            del self._sessions[key]
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)


# ===========================================================================
# Stateless
# ===========================================================================


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


class StatelessSessionStore:
    """The session token carries the whole pipeline state, signed.

    Layout: ``v1.<base64url(zlib(json))>.<base64url(hmac-sha256)>``
    """

    backend = "stateless"

    def __init__(self, secret: str, ttl_seconds: int):
        if not secret:
            raise RuntimeError(
                "SESSION_SECRET must be set when sessions.backend is 'stateless'. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        self.secret = secret.encode("utf-8")
        self.ttl = ttl_seconds

    def _sign(self, payload: bytes) -> str:
        return _b64encode(hmac.new(self.secret, payload, hashlib.sha256).digest())

    def create(self, state: PipelineState) -> str:
        envelope = {"exp": int(time.time()) + self.ttl, "state": state.model_dump()}
        payload = zlib.compress(json.dumps(envelope, separators=(",", ":")).encode("utf-8"), 6)
        token = f"{TOKEN_VERSION}.{_b64encode(payload)}.{self._sign(payload)}"
        if len(token) > MAX_TOKEN_CHARS:
            raise HTTPException(
                status_code=413,
                detail="The pipeline state grew too large to carry in a session token.",
            )
        return token

    def get(self, session_id: str) -> PipelineState:
        if len(session_id) > MAX_TOKEN_CHARS:
            raise SessionExpired("Session token is malformed.")
        try:
            version, body, signature = session_id.split(".")
        except ValueError:
            raise SessionExpired("Session token is malformed.") from None
        if version != TOKEN_VERSION:
            raise SessionExpired("Session token version is not supported.")

        try:
            payload = _b64decode(body)
        except Exception:  # noqa: BLE001 - any decoding failure is the same answer
            raise SessionExpired("Session token is malformed.") from None

        # Constant-time comparison, and verified *before* decompressing anything.
        if not hmac.compare_digest(self._sign(payload), signature):
            logger.warning("session.bad_signature")
            raise SessionExpired("Session token failed its signature check.")

        try:
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(payload, MAX_STATE_BYTES)
            if decompressor.unconsumed_tail:
                raise ValueError("state payload exceeds the size limit")
            envelope = json.loads(raw)
        except Exception:  # noqa: BLE001
            raise SessionExpired("Session token could not be read.") from None

        if envelope.get("exp", 0) < time.time():
            raise SessionExpired("Session expired. Start again from the brief.")

        try:
            return PipelineState.model_validate(envelope["state"])
        except Exception:  # noqa: BLE001
            raise SessionExpired("Session state is no longer compatible.") from None

    def update(self, session_id: str, state: PipelineState) -> str:
        # The token is the state, so an update mints a new one.
        return self.create(state)

    def __len__(self) -> int:
        return 0  # nothing is held server-side


def build_session_store(config, secret: str) -> SessionStore:
    if config.sessions.backend == "stateless":
        return StatelessSessionStore(secret, config.sessions.ttl_seconds)
    return MemorySessionStore(config.sessions.ttl_seconds, config.sessions.max_sessions)
