"""The paid service: login, free offer, payments, operator flow.

Telegram and YooKassa are replaced by in-memory fakes; the router is mounted
on a bare FastAPI app so these tests do not depend on main.py's globals.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import load_config
from models import Concept, Scene
from pipeline import PipelineState
from sessions import MemorySessionStore
from shop import auth
from shop.routes import ShopClients, build_router
from shop.store import ShopStore
from shop.yookassa import Payment

BOT_TOKEN = "123456:TEST-TOKEN"
SECRET = "shop-test-secret"


class FakeTelegram:
    def __init__(self):
        self.token = BOT_TOKEN
        self.subscribers: set[int] = set()
        self.sent: list[tuple] = []

    @property
    def configured(self):
        return True

    async def is_subscribed(self, channel, user_id):
        return user_id in self.subscribers

    async def send(self, chat_id, text):
        self.sent.append((chat_id, text))


class FakeYooKassa:
    def __init__(self):
        self.payments: dict[str, Payment] = {}
        self.created: list[dict] = []

    @property
    def configured(self):
        return True

    async def create_payment(self, **kwargs):
        self.created.append(kwargs)
        pid = f"pay-{len(self.payments) + 1}"
        self.payments[pid] = Payment(
            id=pid, status="pending", paid=False, amount=f"{kwargs['amount']:.2f}",
            currency=kwargs["currency"], order_id=kwargs["order_id"],
            confirmation_url=f"https://yoomoney.example/checkout/{pid}",
        )
        return self.payments[pid]

    async def get_payment(self, payment_id):
        return self.payments[payment_id]

    def succeed(self, payment_id, amount=None):
        p = self.payments[payment_id]
        p.status, p.paid = "succeeded", True
        if amount is not None:
            p.amount = amount


def signed_login(user_id: int, username: str = "client", auth_date: int | None = None) -> dict:
    fields = {"id": user_id, "first_name": "Ivan", "username": username,
              "auth_date": auth_date or int(time.time())}
    check = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    key = hashlib.sha256(BOT_TOKEN.encode()).digest()
    return {**fields, "hash": hmac.new(key, check.encode(), hashlib.sha256).hexdigest()}


@pytest.fixture
def shop_env(monkeypatch):
    monkeypatch.setenv("SHOP_ADMIN_TOKEN", "admin-token")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_ID", "777")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://video.example")
    cfg = load_config()
    sessions = MemorySessionStore(3600, 50)
    store = ShopStore(":memory:")
    telegram, yookassa = FakeTelegram(), FakeYooKassa()
    app = FastAPI()
    app.include_router(build_router(cfg, lambda: sessions, SECRET, store,
                                    ShopClients(telegram, yookassa)))
    client = TestClient(app)
    return {"client": client, "sessions": sessions, "store": store, "cfg": cfg,
            "telegram": telegram, "yookassa": yookassa}


def storyboard(sessions, total: int) -> str:
    durations = []
    left = total
    while left:
        step = 8 if left >= 12 or left == 8 else (6 if left in (6, 10) else 4)
        durations.append(step)
        left -= step
    state = PipelineState(
        video_format="9:16", total_duration=total, business_goal="Sales",
        target_audience="Moms", topic_idea="Coffee shop", needs_voiceover=False,
        concepts=[Concept(id=1, working_title="Утро с кофе", logline="x", visual_style="x",
                          pacing="fast", voiceover_tone="")],
        selected_concept_id=1,
        scenes=[Scene(scene_number=i + 1, visual_description="shot", camera_movement="static",
                      duration=d) for i, d in enumerate(durations)],
    )
    return sessions.create(state)


def login(env, user_id=1001, username="client"):
    response = env["client"].post("/api/shop/login", json=signed_login(user_id, username))
    assert response.status_code == 200, response.text
    return response


# -- login -------------------------------------------------------------------


def test_login_sets_a_cookie_and_me_works(shop_env):
    login(shop_env)
    me = shop_env["client"].get("/api/shop/me").json()
    assert me["user"]["tg_id"] == 1001
    assert me["subscribed"] is False and me["free_left"] == 1


def test_forged_login_is_rejected(shop_env):
    data = signed_login(1001)
    data["id"] = 2002
    assert shop_env["client"].post("/api/shop/login", json=data).status_code == 401


def test_stale_login_is_rejected(shop_env):
    data = signed_login(1001, auth_date=int(time.time()) - 3 * 24 * 3600)
    assert shop_env["client"].post("/api/shop/login", json=data).status_code == 401


def test_endpoints_need_login(shop_env):
    assert shop_env["client"].get("/api/shop/me").status_code == 401
    assert shop_env["client"].get("/api/shop/orders").status_code == 401


def test_tampered_cookie_is_ignored():
    token = auth.issue(SECRET, 1001, 60)
    assert auth.read(SECRET, token) == 1001
    assert auth.read(SECRET, token.replace("1001.", "1002.", 1)) is None
    assert auth.read(SECRET, auth.issue(SECRET, 1001, -10)) is None


# -- pricing and config -----------------------------------------------------


def test_pricing_starts_at_1000_for_15_seconds(shop_env):
    data = shop_env["client"].get("/api/shop/pricing").json()
    first = data["tiers"][0]
    assert (first["seconds"], first["price"], first["storyboard_seconds"]) == (15, 1000, 16)
    assert data["free"]["channel"] == "@SynndicateHQ"


def test_unbuildable_tier_is_a_config_error(tmp_path):
    import yaml
    from config import DEFAULT_CONFIG_PATH

    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["shop"]["tiers"][0]["storyboard_seconds"] = 15
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(Exception, match="cannot be built"):
        load_config(path)


def test_quip_endpoint(shop_env):
    data = shop_env["client"].get("/api/shop/quip", params={"stage": "generating"}).json()
    assert data["text"]


# -- free offer --------------------------------------------------------------


def test_free_video_needs_the_subscription(shop_env):
    login(shop_env)
    sid = storyboard(shop_env["sessions"], 16)
    body = {"session_id": sid, "seconds": 15, "free": True}
    assert shop_env["client"].post("/api/shop/orders", json=body).status_code == 403

    shop_env["telegram"].subscribers.add(1001)
    response = shop_env["client"].post("/api/shop/orders", json=body)
    assert response.status_code == 200, response.text
    order = response.json()
    assert order["status"] == "queued" and order["price"] == 0
    assert any(chat == "777" for chat, _ in shop_env["telegram"].sent), "operator not told"

    again = shop_env["client"].post("/api/shop/orders", json=body)
    assert again.status_code == 409


def test_free_offer_is_only_the_short_video(shop_env):
    login(shop_env)
    shop_env["telegram"].subscribers.add(1001)
    sid = storyboard(shop_env["sessions"], 30)
    response = shop_env["client"].post(
        "/api/shop/orders", json={"session_id": sid, "seconds": 30, "free": True})
    assert response.status_code == 400


def test_free_offer_has_a_daily_cap(shop_env, monkeypatch):
    monkeypatch.setattr(shop_env["cfg"].shop.free, "daily_cap", 1)
    for uid in (1, 2):
        shop_env["telegram"].subscribers.add(uid)
    login(shop_env, 1)
    sid = storyboard(shop_env["sessions"], 16)
    ok = shop_env["client"].post("/api/shop/orders", json={"session_id": sid, "seconds": 15, "free": True})
    assert ok.status_code == 200
    login(shop_env, 2)
    sid = storyboard(shop_env["sessions"], 16)
    capped = shop_env["client"].post("/api/shop/orders", json={"session_id": sid, "seconds": 15, "free": True})
    assert capped.status_code == 429


def test_storyboard_length_must_match_the_tier(shop_env):
    login(shop_env)
    sid = storyboard(shop_env["sessions"], 30)
    response = shop_env["client"].post("/api/shop/orders", json={"session_id": sid, "seconds": 60})
    assert response.status_code == 400
    assert "60" in response.json()["detail"]


# -- paid flow ---------------------------------------------------------------


def paid_order(env, seconds=30):
    login(env)
    sid = storyboard(env["sessions"], env["cfg"].shop.tier(seconds).storyboard_seconds)
    response = env["client"].post("/api/shop/orders", json={"session_id": sid, "seconds": seconds})
    assert response.status_code == 200, response.text
    return response.json()


def test_paid_order_goes_through_yookassa(shop_env):
    order = paid_order(shop_env)
    assert order["status"] == "awaiting_payment" and order["price"] == 1900

    pay = shop_env["client"].post(f"/api/shop/orders/{order['id']}/pay", json={"email": "a@b.ru"})
    assert pay.status_code == 200
    assert pay.json()["confirmation_url"].startswith("https://")
    created = shop_env["yookassa"].created[0]
    assert created["amount"] == 1900 and created["return_url"].endswith(f"order={order['id']}")

    shop_env["yookassa"].succeed("pay-1")
    hook = {"type": "notification", "event": "payment.succeeded", "object": {"id": "pay-1"}}
    assert shop_env["client"].post("/api/shop/yookassa/webhook", json=hook).status_code == 200
    assert shop_env["client"].post("/api/shop/yookassa/webhook", json=hook).status_code == 200

    state = shop_env["client"].get(f"/api/shop/orders/{order['id']}").json()
    assert state["status"] == "queued" and state["quip"]
    operator_messages = [t for chat, t in shop_env["telegram"].sent if chat == "777"]
    assert len(operator_messages) == 1, "a repeated webhook must not queue twice"


def test_webhook_body_alone_does_not_mark_paid(shop_env):
    order = paid_order(shop_env)
    shop_env["client"].post(f"/api/shop/orders/{order['id']}/pay", json={"email": "a@b.ru"})
    # The notification claims success, but YooKassa itself still says pending.
    hook = {"event": "payment.succeeded", "object": {"id": "pay-1", "status": "succeeded"}}
    shop_env["client"].post("/api/shop/yookassa/webhook", json=hook)
    assert shop_env["store"].get_order(order["id"]).status == "awaiting_payment"


def test_wrong_amount_does_not_mark_paid(shop_env):
    order = paid_order(shop_env)
    shop_env["client"].post(f"/api/shop/orders/{order['id']}/pay", json={"email": "a@b.ru"})
    shop_env["yookassa"].succeed("pay-1", amount="1.00")
    shop_env["client"].post("/api/shop/yookassa/webhook", json={"object": {"id": "pay-1"}})
    assert shop_env["store"].get_order(order["id"]).status == "awaiting_payment"


def test_bad_email_is_rejected(shop_env):
    order = paid_order(shop_env)
    response = shop_env["client"].post(f"/api/shop/orders/{order['id']}/pay", json={"email": "nope"})
    assert response.status_code == 422


def test_orders_are_private(shop_env):
    order = paid_order(shop_env)
    login(shop_env, 2002, "other")
    assert shop_env["client"].get(f"/api/shop/orders/{order['id']}").status_code == 404
    assert shop_env["client"].get("/api/shop/orders").json()["orders"] == []


# -- operator ----------------------------------------------------------------


def test_operator_moves_the_order_and_the_client_is_told(shop_env):
    order = paid_order(shop_env)
    client, url = shop_env["client"], f"/api/shop/admin/orders/{order['id']}"
    assert client.post(url, json={"status": "generating"}).status_code == 403

    headers = {"x-admin-token": "admin-token"}
    assert client.post(url, json={"status": "generating"}, headers=headers).status_code == 409

    shop_env["store"].mark_paid(order["id"])
    assert client.post(url, json={"status": "generating"}, headers=headers).status_code == 200
    assert client.post(url, json={"status": "delivered"}, headers=headers).status_code == 400
    done = client.post(url, json={"status": "delivered", "result_url": "https://cdn.example/v.mp4"},
                       headers=headers)
    assert done.status_code == 200
    assert done.json()["result_url"] == "https://cdn.example/v.mp4"
    to_client = [t for chat, t in shop_env["telegram"].sent if chat == 1001]
    assert any("https://cdn.example/v.mp4" in t for t in to_client)
    assert [h["status"] for h in done.json()["history"]] == [
        "awaiting_payment", "queued", "generating", "delivered"]


def test_admin_list_needs_the_token(shop_env):
    paid_order(shop_env)
    assert shop_env["client"].get("/api/shop/admin/orders").status_code == 403
    listed = shop_env["client"].get("/api/shop/admin/orders", headers={"x-admin-token": "admin-token"})
    assert len(listed.json()["orders"]) == 1
