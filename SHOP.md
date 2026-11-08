# «Создай видео»: платный сервис поверх Нейроклипа

Клиент проходит бриф → концепции → раскадровку на `/studio`, входит через Telegram и оформляет заказ:
бесплатный ролик 15 с за подписку на канал или платный через ЮKassa. Генерацию в Higgsfield и сборку
пока делаем руками; статус заказа двигает оператор, клиент видит его в `/cabinet` и в Telegram.

| Страница / API | Что делает |
| --- | --- |
| `/studio` | бриф, концепции, раскадровка, заказ (для клиентов, по-русски) |
| `/cabinet` | заказы клиента, оплата, статусы, ссылка на готовый ролик |
| `/` | старое демо для хакатонов, без изменений |
| `/api/shop/*` | прайс, вход, заказы, оплата, вебхук ЮKassa, операторские ручки |

Прайс, бесплатный оффер и канал — в `config.yaml`, секция `shop`. Тариф 15 с планируется как 16 с
(из клипов 4/6/8 с нельзя собрать 15), при сборке обрезаем до 15.

## Статусы заказа

`awaiting_payment` → `queued` → `generating` → `assembling` → `review` → `delivered` (или `cancelled`).
Бесплатный заказ сразу попадает в `queued`. Из `awaiting_payment` в `queued` переводит только
подтверждённая оплата: тело вебхука не доверяем, платёж перечитываем у ЮKassa по id и сверяем сумму.

## Запуск на VDSina (тот же сервер, поддомен video.synndicate.ru)

1. Код и окружение:
   ```bash
   git clone https://github.com/Dengel88/neuroclip-studio /var/www/fastuser/data/apps/neuroclip
   cd /var/www/fastuser/data/apps/neuroclip
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   cp .env.example .env   # заполнить, см. ниже
   ```
2. Запуск под pm2 (как AI-Штат), только на localhost:
   ```bash
   pm2 start ".venv/bin/uvicorn main:app --host 127.0.0.1 --port 8010" --name neuroclip
   pm2 save
   ```
   Один процесс: сессии раскадровки в памяти, заказы в SQLite (`data/shop.sqlite3`).
3. FASTPANEL: создать сайт `video.synndicate.ru`, выпустить Let's Encrypt, в nginx
   `location / { proxy_pass http://127.0.0.1:8010; proxy_set_header Host $host;
   proxy_set_header X-Forwarded-For $remote_addr; proxy_set_header X-Forwarded-Proto $scheme;
   proxy_read_timeout 180s; }`. DNS: A-запись `video` → 45.150.39.198.
4. Проверка: `https://video.synndicate.ru/api/health` → `"status":"ok"`, `"shop":"enabled"`.

## .env на сервере

- `GEMINI_API_KEY_1` — рабочий ключ (тот же, что у AI-Штата).
- `GEMINI_BASE_URL` — Google не отвечает на российские IP. Нужен ретранслятор нативного Gemini API
  (`/v1beta/models/...:generateContent`). Текущий воркер `ai-shtat-gemini` проксирует только
  OpenAI-совместимый `/v1` — ему нужен ещё один маршрут, иначе раскадровка с сервера не пойдёт.
- `SESSION_SECRET` — случайная строка (`python -c "import secrets; print(secrets.token_hex(32))"`).
- `PUBLIC_BASE_URL=https://video.synndicate.ru`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME` — бот из @BotFather. Боту: `/setdomain` →
  `video.synndicate.ru` (без этого не работает вход), и сделать его админом @SynndicateHQ
  (без этого не проверить подписку).
- `TELEGRAM_ADMIN_CHAT_ID` — куда слать «новый заказ» (свой chat id или группа).
- `TELEGRAM_API_BASE` — только если api.telegram.org с сервера недоступен (маршрут `/tg` воркера).
- `YOOKASSA_SHOP_ID`, `YOOKASSA_SECRET_KEY` — из кабинета ЮKassa. Сначала тестовый магазин.
- `SHOP_ADMIN_TOKEN` — случайная строка для операторских ручек.

## ЮKassa

- Кабинет → Интеграция → HTTP-уведомления: URL `https://video.synndicate.ru/api/shop/yookassa/webhook`,
  события `payment.succeeded` и `payment.canceled`.
- Включить «Чеки от ЮKassa» (54-ФЗ). Чек идёт на e-mail, который клиент вводит перед оплатой.
  Ставка НДС — `shop.vat_code` в `config.yaml` (1 = без НДС, для УСН).
- Даже если вебхук не дошёл, заказ переходит в `queued`, когда клиент возвращается в кабинет:
  страница сама спрашивает статус платежа.

## Работа оператора

Новый заказ приходит в Telegram (`TELEGRAM_ADMIN_CHAT_ID`). Раскадровка и промпты лежат в заказе:

```bash
H="X-Admin-Token: $SHOP_ADMIN_TOKEN"
curl -s -H "$H" "https://video.synndicate.ru/api/shop/admin/orders?status=queued"
# двигаем статус, клиенту уходит сообщение со статусом и шуткой
curl -s -H "$H" -H "Content-Type: application/json" \
  -d '{"status":"generating"}' https://video.synndicate.ru/api/shop/admin/orders/<id>
# выдача: ссылка на готовый файл обязательна
curl -s -H "$H" -H "Content-Type: application/json" \
  -d '{"status":"delivered","result_url":"https://.../video.mp4","message":"Спасибо за заказ!"}' \
  https://video.synndicate.ru/api/shop/admin/orders/<id>
```

## Чего пока нет

- Автогенерации в Higgsfield и автосборки — это этап 2.
- Водяной знак на бесплатном ролике ставим при сборке руками.
- Возвратов через API: отмена оплаченного заказа — возврат вручную в кабинете ЮKassa.
