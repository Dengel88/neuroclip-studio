"""Orders and clients in SQLite.

One file, one process: the service runs as a single uvicorn worker on the
VPS, so SQLite with a lock is the simplest thing that keeps data across
restarts. Every status change is written to `events` as well, which is the
audit trail an operator reads when a client asks "where is my video".
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# The order life cycle. Paid orders start at awaiting_payment, free ones at
# queued. Everything after queued is moved by the operator.
STATUSES = (
    "awaiting_payment",
    "queued",
    "generating",
    "assembling",
    "review",
    "delivered",
    "cancelled",
)
OPERATOR_STATUSES = STATUSES[1:]

STATUS_LABELS = {
    "awaiting_payment": "Ждёт оплаты",
    "queued": "В очереди",
    "generating": "Генерируем кадры",
    "assembling": "Собираем ролик",
    "review": "Проверяем качество",
    "delivered": "Готово",
    "cancelled": "Отменён",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id       INTEGER PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id          TEXT PRIMARY KEY,
    tg_id       INTEGER NOT NULL REFERENCES users(tg_id),
    kind        TEXT NOT NULL CHECK (kind IN ('paid', 'free')),
    seconds     INTEGER NOT NULL,
    price       INTEGER NOT NULL,
    status      TEXT NOT NULL,
    video_format TEXT NOT NULL,
    snapshot    TEXT NOT NULL,
    email       TEXT,
    payment_id  TEXT,
    result_url  TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    paid_at     REAL
);
CREATE INDEX IF NOT EXISTS orders_by_user ON orders(tg_id, created_at);
CREATE INDEX IF NOT EXISTS orders_by_payment ON orders(payment_id);
CREATE TABLE IF NOT EXISTS events (
    order_id    TEXT NOT NULL REFERENCES orders(id),
    at          REAL NOT NULL,
    status      TEXT NOT NULL,
    message     TEXT
);
"""


@dataclass
class Order:
    id: str
    tg_id: int
    kind: str
    seconds: int
    price: int
    status: str
    video_format: str
    snapshot: Dict[str, Any]
    email: Optional[str]
    payment_id: Optional[str]
    result_url: Optional[str]
    created_at: float
    updated_at: float
    paid_at: Optional[float]
    events: List[Dict[str, Any]] = field(default_factory=list)

    def public(self) -> Dict[str, Any]:
        """What the client sees: no payment ids, no operator notes."""
        return {
            "id": self.id,
            "kind": self.kind,
            "seconds": self.seconds,
            "price": self.price,
            "status": self.status,
            "status_label": STATUS_LABELS[self.status],
            "video_format": self.video_format,
            "title": self.snapshot.get("concept", {}).get("working_title", ""),
            "scenes": self.snapshot.get("scenes", []),
            "result_url": self.result_url if self.status == "delivered" else None,
            "created_at": self.created_at,
            "history": [
                {"at": e["at"], "status": e["status"], "label": STATUS_LABELS[e["status"]],
                 "message": e["message"]}
                for e in self.events
            ],
        }


class ShopStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # -- users ---------------------------------------------------------------

    def upsert_user(self, tg_id: int, username: str = "", first_name: str = "") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO users (tg_id, username, first_name, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tg_id) DO UPDATE SET username = excluded.username, "
                "first_name = excluded.first_name",
                (tg_id, username, first_name, time.time()),
            )
            self._db.commit()

    def get_user(self, tg_id: int) -> Optional[Dict[str, Any]]:
        row = self._db.execute("SELECT * FROM users WHERE tg_id = ?", (tg_id,)).fetchone()
        return dict(row) if row else None

    def free_orders_used(self, tg_id: int) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM orders WHERE tg_id = ? AND kind = 'free' AND status != 'cancelled'",
            (tg_id,),
        ).fetchone()
        return row[0]

    def free_orders_since(self, since: float) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM orders WHERE kind = 'free' AND created_at >= ?", (since,)
        ).fetchone()
        return row[0]

    # -- orders --------------------------------------------------------------

    def create_order(
        self,
        tg_id: int,
        kind: str,
        seconds: int,
        price: int,
        video_format: str,
        snapshot: Dict[str, Any],
    ) -> Order:
        status = "queued" if kind == "free" else "awaiting_payment"
        order_id = secrets.token_hex(5)
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO orders (id, tg_id, kind, seconds, price, status, video_format, "
                "snapshot, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (order_id, tg_id, kind, seconds, price, status, video_format,
                 json.dumps(snapshot, ensure_ascii=False), now, now),
            )
            self._db.execute(
                "INSERT INTO events (order_id, at, status, message) VALUES (?, ?, ?, ?)",
                (order_id, now, status, None),
            )
            self._db.commit()
        return self.get_order(order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        row = self._db.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if not row:
            return None
        events = [
            dict(e) for e in self._db.execute(
                "SELECT at, status, message FROM events WHERE order_id = ? ORDER BY at, rowid",
                (order_id,),
            )
        ]
        data = dict(row)
        data["snapshot"] = json.loads(data["snapshot"])
        return Order(**data, events=events)

    def find_by_payment(self, payment_id: str) -> Optional[Order]:
        row = self._db.execute(
            "SELECT id FROM orders WHERE payment_id = ?", (payment_id,)
        ).fetchone()
        return self.get_order(row["id"]) if row else None

    def list_orders(self, tg_id: Optional[int] = None, status: Optional[str] = None) -> List[Order]:
        query, args = "SELECT id FROM orders WHERE 1 = 1", []
        if tg_id is not None:
            query += " AND tg_id = ?"
            args.append(tg_id)
        if status:
            query += " AND status = ?"
            args.append(status)
        query += " ORDER BY created_at DESC LIMIT 200"
        return [self.get_order(r["id"]) for r in self._db.execute(query, args)]

    def attach_payment(self, order_id: str, payment_id: str, email: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE orders SET payment_id = ?, email = ?, updated_at = ? WHERE id = ?",
                (payment_id, email, time.time(), order_id),
            )
            self._db.commit()

    def set_status(
        self,
        order_id: str,
        status: str,
        message: Optional[str] = None,
        result_url: Optional[str] = None,
    ) -> Order:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        now = time.time()
        with self._lock:
            sets, args = ["status = ?", "updated_at = ?"], [status, now]
            if result_url is not None:
                sets.append("result_url = ?")
                args.append(result_url)
            self._db.execute(f"UPDATE orders SET {', '.join(sets)} WHERE id = ?", (*args, order_id))
            self._db.execute(
                "INSERT INTO events (order_id, at, status, message) VALUES (?, ?, ?, ?)",
                (order_id, now, status, message),
            )
            self._db.commit()
        return self.get_order(order_id)

    def mark_paid(self, order_id: str) -> Optional[Order]:
        """Move an order from awaiting_payment to queued, exactly once.

        Webhooks are retried and the return page polls too, so the same
        payment is reported more than once; only the first report moves it.
        """
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "UPDATE orders SET status = 'queued', paid_at = ?, updated_at = ? "
                "WHERE id = ? AND status = 'awaiting_payment'",
                (now, now, order_id),
            )
            if cursor.rowcount:
                self._db.execute(
                    "INSERT INTO events (order_id, at, status, message) VALUES (?, ?, 'queued', ?)",
                    (order_id, now, "Оплата получена"),
                )
            self._db.commit()
        return self.get_order(order_id) if cursor.rowcount else None
