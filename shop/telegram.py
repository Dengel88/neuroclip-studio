"""Telegram: login check and the service bot.

* Login uses the Telegram Login Widget. The widget hands the browser a signed
  set of fields; ``verify_login`` checks the signature with the bot token, so
  a client cannot claim someone else's Telegram id.
* The bot checks the subscription to the channel (``getChatMember``; the bot
  must be an admin of the channel) and sends status messages.

``TELEGRAM_API_BASE`` lets the Bot API go through a relay - the Cloudflare
worker that already fronts Gemini - when api.telegram.org is unreachable
from the server.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("NeuroclipStudio.Telegram")

LOGIN_MAX_AGE_SECONDS = 24 * 3600
SUBSCRIBED_STATUSES = {"creator", "administrator", "member", "restricted"}


class TelegramLoginError(ValueError):
    pass


def verify_login(data: Dict[str, Any], bot_token: str, now: Optional[float] = None) -> Dict[str, Any]:
    """Check a Telegram Login Widget payload; return the user fields.

    https://core.telegram.org/widgets/login#checking-authorization
    """
    if not bot_token:
        raise TelegramLoginError("Telegram login is not configured on the server.")
    received = str(data.get("hash", ""))
    fields = {k: v for k, v in data.items() if k != "hash" and v is not None}
    check_string = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hashlib.sha256(bot_token.encode("utf-8")).digest()
    expected = hmac.new(secret, check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not received or not hmac.compare_digest(expected, received):
        raise TelegramLoginError("Telegram login signature does not match.")
    try:
        auth_date = int(fields.get("auth_date", 0))
    except (TypeError, ValueError):
        raise TelegramLoginError("Telegram login has no valid auth_date.") from None
    if (now or time.time()) - auth_date > LOGIN_MAX_AGE_SECONDS:
        raise TelegramLoginError("Telegram login is too old. Log in again.")
    return {
        "id": int(fields["id"]),
        "username": str(fields.get("username", "")),
        "first_name": str(fields.get("first_name", "")),
    }


class TelegramBot:
    def __init__(self, token: str, api_base: str = "https://api.telegram.org", timeout: float = 10.0):
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "TelegramBot":
        return cls(
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
            os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org"),
        )

    @property
    def configured(self) -> bool:
        return bool(self.token)

    async def _call(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base}/bot{self.token}/{method}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {body.get('description', response.status_code)}")
        return body["result"]

    async def is_subscribed(self, channel: str, user_id: int) -> bool:
        try:
            member = await self._call("getChatMember", {"chat_id": channel, "user_id": user_id})
        except Exception as exc:  # noqa: BLE001 - "user not found" is also an error here
            logger.info("telegram.subscription_check_failed user=%s error=%s", user_id, exc)
            return False
        return member.get("status") in SUBSCRIBED_STATUSES

    async def send(self, chat_id: int | str, text: str) -> None:
        """Best effort: a failed notification never fails the request."""
        if not self.configured or not chat_id:
            return
        try:
            await self._call(
                "sendMessage",
                {"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram.send_failed chat=%s error=%s", chat_id, exc)
