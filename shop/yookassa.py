"""YooKassa (ЮKassa) payments over its REST API.

Only two calls are needed: create a payment and read one back. The webhook
body is never trusted on its own - YooKassa does not sign notifications - so
the handler reads the payment back by id with our own credentials and acts
on that answer only.

Credentials: ``YOOKASSA_SHOP_ID`` and ``YOOKASSA_SECRET_KEY``. A test shop
from the YooKassa dashboard works the same way, with test cards.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

API_BASE = "https://api.yookassa.ru/v3"


@dataclass
class Payment:
    id: str
    status: str  # pending | waiting_for_capture | succeeded | canceled
    paid: bool
    amount: str
    currency: str
    order_id: Optional[str]
    confirmation_url: Optional[str] = None

    @classmethod
    def from_api(cls, body: Dict[str, Any]) -> "Payment":
        return cls(
            id=body["id"],
            status=body.get("status", ""),
            paid=bool(body.get("paid")),
            amount=body.get("amount", {}).get("value", ""),
            currency=body.get("amount", {}).get("currency", ""),
            order_id=(body.get("metadata") or {}).get("order_id"),
            confirmation_url=(body.get("confirmation") or {}).get("confirmation_url"),
        )


def format_amount(roubles: int) -> str:
    return f"{roubles:.2f}"


class YooKassa:
    def __init__(self, shop_id: str, secret_key: str, api_base: str = API_BASE, timeout: float = 15.0):
        self.shop_id = shop_id
        self.secret_key = secret_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "YooKassa":
        return cls(os.getenv("YOOKASSA_SHOP_ID", ""), os.getenv("YOOKASSA_SECRET_KEY", ""))

    @property
    def configured(self) -> bool:
        return bool(self.shop_id and self.secret_key)

    async def create_payment(
        self,
        order_id: str,
        amount: int,
        currency: str,
        description: str,
        return_url: str,
        email: str,
        vat_code: int,
    ) -> Payment:
        body = {
            "amount": {"value": format_amount(amount), "currency": currency},
            "capture": True,
            "confirmation": {"type": "redirect", "return_url": return_url},
            "description": description[:128],
            "metadata": {"order_id": order_id},
            # 54-FZ receipt: YooKassa issues it when "Чеки от ЮKassa" is on.
            "receipt": {
                "customer": {"email": email},
                "items": [{
                    "description": description[:128],
                    "quantity": "1.00",
                    "amount": {"value": format_amount(amount), "currency": currency},
                    "vat_code": vat_code,
                    "payment_subject": "service",
                    "payment_mode": "full_payment",
                }],
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_base}/payments",
                json=body,
                auth=(self.shop_id, self.secret_key),
                # Same order -> same key, so a double click cannot pay twice.
                headers={"Idempotence-Key": str(uuid.uuid5(uuid.NAMESPACE_URL, f"order:{order_id}:{email}"))},
            )
        response.raise_for_status()
        return Payment.from_api(response.json())

    async def get_payment(self, payment_id: str) -> Payment:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.api_base}/payments/{payment_id}", auth=(self.shop_id, self.secret_key)
            )
        response.raise_for_status()
        return Payment.from_api(response.json())
