"""The login cookie: a signed "<tg_id>.<expiry>.<hmac>" string.

Signed with SESSION_SECRET, the same secret the pipeline sessions use. It
holds nothing but the Telegram id, so it is not encrypted, only unforgeable.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

COOKIE_NAME = "nc_user"


def _sign(secret: str, body: str) -> str:
    return hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()


def issue(secret: str, tg_id: int, ttl_seconds: int, now: Optional[float] = None) -> str:
    body = f"{tg_id}.{int((now or time.time()) + ttl_seconds)}"
    return f"{body}.{_sign(secret, body)}"


def read(secret: str, token: str, now: Optional[float] = None) -> Optional[int]:
    if not secret or not token or token.count(".") != 2:
        return None
    tg_id, expiry, signature = token.split(".")
    if not hmac.compare_digest(_sign(secret, f"{tg_id}.{expiry}"), signature):
        return None
    try:
        if int(expiry) < (now or time.time()):
            return None
        return int(tg_id)
    except ValueError:
        return None
