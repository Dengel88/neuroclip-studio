"""HTTP endpoints of the service, mounted under /api/shop.

Built by ``build_router`` so ``main.py`` can hand over its session store
without a circular import. External services are reached through
``ShopClients``; tests swap in fakes, nothing here touches the network on
its own.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from config import AppConfig

from . import auth
from .quips import quip
from .store import OPERATOR_STATUSES, STATUS_LABELS, Order, ShopStore
from .telegram import TelegramBot, TelegramLoginError, verify_login
from .yookassa import YooKassa, format_amount

logger = logging.getLogger("NeuroclipStudio.Shop")


@dataclass
class ShopClients:
    telegram: TelegramBot
    yookassa: YooKassa

    @classmethod
    def from_env(cls) -> "ShopClients":
        return cls(TelegramBot.from_env(), YooKassa.from_env())


class OrderRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128 * 1024)
    seconds: int = Field(..., ge=1)
    free: bool = False


class PayRequest(BaseModel):
    # Needed for the 54-FZ receipt. A plain shape check; YooKassa validates it too.
    email: str = Field(..., max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class OperatorUpdate(BaseModel):
    status: str
    message: Optional[str] = Field(default=None, max_length=1000)
    result_url: Optional[str] = Field(default=None, max_length=2000)


def _snapshot(state) -> dict:
    concept = state.selected_concept
    return {
        "brief": {
            "video_format": state.video_format,
            "total_duration": state.total_duration,
            "business_goal": state.business_goal,
            "target_audience": state.target_audience,
            "topic_idea": state.topic_idea,
            "needs_voiceover": state.needs_voiceover,
        },
        "concept": concept.model_dump() if concept else {},
        "scenes": [s.model_dump() for s in state.scenes],
        "prompts": [p.model_dump() for p in state.prompts],
    }


def build_router(
    app_config: AppConfig,
    session_store_getter: Callable[[], object],
    secret: str,
    store: ShopStore,
    clients: ShopClients,
) -> APIRouter:
    shop = app_config.shop
    router = APIRouter(prefix="/api/shop")
    admin_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
    public_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

    # -- helpers ---------------------------------------------------------------

    def current_user(request: Request) -> int:
        tg_id = auth.read(secret, request.cookies.get(auth.COOKIE_NAME, ""))
        if tg_id is None or store.get_user(tg_id) is None:
            raise HTTPException(status_code=401, detail="Войдите через Telegram.")
        return tg_id

    def own_order(order_id: str, tg_id: int) -> Order:
        order = store.get_order(order_id)
        if order is None or order.tg_id != tg_id:
            raise HTTPException(status_code=404, detail="Заказ не найден.")
        return order

    def require_admin(request: Request) -> None:
        expected = os.getenv("SHOP_ADMIN_TOKEN", "")
        given = request.headers.get("x-admin-token", "")
        if not expected or not hmac.compare_digest(expected, given):
            raise HTTPException(status_code=403, detail="Admin token required.")

    async def notify_client(order: Order, message: Optional[str] = None) -> None:
        text = f"Заказ {order.id}: {STATUS_LABELS[order.status]}.\n{quip(order.status)}"
        if message:
            text += f"\n\n{message}"
        if order.status == "delivered" and order.result_url:
            text += f"\n\nРолик: {order.result_url}"
        if public_base:
            text += f"\n\nКабинет: {public_base}/cabinet?order={order.id}"
        await clients.telegram.send(order.tg_id, text)

    async def notify_operator(order: Order) -> None:
        user = store.get_user(order.tg_id) or {}
        who = f"@{user['username']}" if user.get("username") else str(order.tg_id)
        kind = "бесплатный" if order.kind == "free" else f"оплачен {order.price} ₽"
        title = order.snapshot.get("concept", {}).get("working_title", "")
        await clients.telegram.send(
            admin_chat,
            f"Новый заказ {order.id} ({kind}): {order.seconds} с, {order.video_format}, «{title}», от {who}.",
        )

    async def settle(order: Order) -> Order:
        """Ask YooKassa about an unpaid order and queue it if the money is in."""
        if order.status != "awaiting_payment" or not order.payment_id or not clients.yookassa.configured:
            return order
        payment = await clients.yookassa.get_payment(order.payment_id)
        if (
            payment.status == "succeeded"
            and payment.order_id == order.id
            and payment.amount == format_amount(order.price)
            and payment.currency == shop.currency
        ):
            queued = store.mark_paid(order.id)
            if queued:
                logger.info("shop.paid order=%s payment=%s", order.id, payment.id)
                await notify_client(queued)
                await notify_operator(queued)
                return queued
        elif payment.status == "canceled":
            logger.info("shop.payment_canceled order=%s payment=%s", order.id, payment.id)
        return store.get_order(order.id)

    # -- public ----------------------------------------------------------------

    @router.get("/pricing")
    async def pricing():
        return {
            "currency": shop.currency,
            "tiers": [t.model_dump() for t in shop.tiers],
            "free": {"seconds": shop.free.seconds, "channel": shop.free.channel,
                     "channel_url": shop.free.channel_url},
            "telegram_bot_username": os.getenv("TELEGRAM_BOT_USERNAME", ""),
            "payments_enabled": clients.yookassa.configured,
        }

    @router.get("/quip")
    async def get_quip(stage: str = "queued"):
        return {"stage": stage, "text": quip(stage)}

    @router.post("/login")
    async def login(request: Request, response: Response):
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Expected the Telegram login fields.")
        try:
            user = verify_login(data, clients.telegram.token)
        except TelegramLoginError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from None
        store.upsert_user(user["id"], user["username"], user["first_name"])
        response.set_cookie(
            auth.COOKIE_NAME,
            auth.issue(secret, user["id"], shop.login_ttl_seconds),
            max_age=shop.login_ttl_seconds,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
        return {"user": user}

    @router.post("/logout")
    async def logout(response: Response):
        response.delete_cookie(auth.COOKIE_NAME)
        return {"ok": True}

    @router.get("/me")
    async def me(tg_id: int = Depends(current_user)):
        used = store.free_orders_used(tg_id)
        subscribed = await clients.telegram.is_subscribed(shop.free.channel, tg_id)
        return {
            "user": store.get_user(tg_id),
            "subscribed": subscribed,
            "free_left": max(0, shop.free.per_user - used),
            "channel": shop.free.channel,
            "channel_url": shop.free.channel_url,
        }

    @router.post("/orders")
    async def create_order(body: OrderRequest, tg_id: int = Depends(current_user)):
        try:
            tier = shop.tier(body.seconds)
        except KeyError:
            raise HTTPException(status_code=400, detail="Такой длительности нет в прайсе.") from None
        state = session_store_getter().get(body.session_id)
        if not state.scenes:
            raise HTTPException(status_code=400, detail="Сначала соберите раскадровку.")
        if state.total_duration != tier.storyboard_seconds:
            raise HTTPException(
                status_code=400,
                detail=f"Раскадровка рассчитана на {state.total_duration} с, а выбран тариф "
                f"{tier.seconds} с. Пересоберите раскадровку на {tier.storyboard_seconds} с.",
            )

        if body.free:
            if tier.seconds != shop.free.seconds:
                raise HTTPException(status_code=400, detail=f"Бесплатно — только ролик {shop.free.seconds} с.")
            if store.free_orders_used(tg_id) >= shop.free.per_user:
                raise HTTPException(status_code=409, detail="Бесплатный ролик уже использован.")
            if not await clients.telegram.is_subscribed(shop.free.channel, tg_id):
                raise HTTPException(
                    status_code=403,
                    detail=f"Подпишитесь на {shop.free.channel} и нажмите ещё раз.",
                )
            if store.free_orders_since(time.time() - 24 * 3600) >= shop.free.daily_cap:
                raise HTTPException(
                    status_code=429,
                    detail="Бесплатные ролики на сегодня закончились. Загляните завтра.",
                )

        order = store.create_order(
            tg_id,
            "free" if body.free else "paid",
            tier.seconds,
            0 if body.free else tier.price,
            state.video_format,
            _snapshot(state),
        )
        logger.info("shop.order_created order=%s kind=%s seconds=%d", order.id, order.kind, order.seconds)
        if order.kind == "free":
            await notify_client(order)
            await notify_operator(order)
        return order.public()

    @router.get("/orders")
    async def my_orders(tg_id: int = Depends(current_user)):
        return {"orders": [o.public() for o in store.list_orders(tg_id=tg_id)]}

    @router.get("/orders/{order_id}")
    async def get_order(order_id: str, tg_id: int = Depends(current_user)):
        order = await settle(own_order(order_id, tg_id))
        return {**order.public(), "quip": quip(order.status)}

    @router.post("/orders/{order_id}/pay")
    async def pay(order_id: str, body: PayRequest, request: Request, tg_id: int = Depends(current_user)):
        order = own_order(order_id, tg_id)
        if order.status != "awaiting_payment":
            raise HTTPException(status_code=409, detail="Этот заказ уже не ждёт оплаты.")
        if not clients.yookassa.configured:
            raise HTTPException(status_code=503, detail="Оплата ещё не подключена.")
        base = public_base or str(request.base_url).rstrip("/")
        payment = await clients.yookassa.create_payment(
            order_id=order.id,
            amount=order.price,
            currency=shop.currency,
            description=f"Видеоролик {order.seconds} с, заказ {order.id}",
            return_url=f"{base}/cabinet?order={order.id}",
            email=str(body.email),
            vat_code=shop.vat_code,
        )
        store.attach_payment(order.id, payment.id, str(body.email))
        return {"confirmation_url": payment.confirmation_url}

    @router.post("/yookassa/webhook")
    async def yookassa_webhook(request: Request):
        # The body only tells us which payment to look at; the answer we act
        # on comes from YooKassa itself (see settle()).
        try:
            body = await request.json()
            payment_id = str(body["object"]["id"])
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Not a YooKassa notification.") from None
        order = store.find_by_payment(payment_id)
        if order is not None:
            await settle(order)
        return {"ok": True}

    # -- operator --------------------------------------------------------------

    @router.get("/admin/orders", dependencies=[Depends(require_admin)])
    async def admin_orders(status: Optional[str] = None):
        return {"orders": [
            {**o.public(), "tg_id": o.tg_id, "email": o.email, "payment_id": o.payment_id,
             "snapshot": o.snapshot}
            for o in store.list_orders(status=status)
        ]}

    @router.post("/admin/orders/{order_id}", dependencies=[Depends(require_admin)])
    async def admin_update(order_id: str, body: OperatorUpdate):
        order = store.get_order(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail="Unknown order.")
        if body.status not in OPERATOR_STATUSES:
            raise HTTPException(status_code=400, detail=f"Status must be one of {list(OPERATOR_STATUSES)}.")
        if order.status == "awaiting_payment" and body.status != "cancelled":
            raise HTTPException(status_code=409, detail="Order is not paid yet.")
        if body.status == "delivered" and not (body.result_url or order.result_url):
            raise HTTPException(status_code=400, detail="A delivered order needs result_url.")
        order = store.set_status(order_id, body.status, body.message, body.result_url)
        await notify_client(order, body.message)
        return order.public()

    return router
