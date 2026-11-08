// Shared by studio.html and cabinet.html. Plain script, no build step.

// Vercel's Python bridge rejects raw non-ASCII bodies; \uXXXX escapes decode to
// the same string. Harmless on a normal server, so the shop pages use it too.
function toAsciiJson(payload) {
    return JSON.stringify(payload).replace(
        /[\u0080-￿]/g,
        ch => '\\u' + ch.charCodeAt(0).toString(16).padStart(4, '0')
    );
}

function formatDetail(detail) {
    if (!detail) return "";
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map(e => e.msg).join('; ');
    return JSON.stringify(detail);
}

async function api(url, payload, method) {
    const options = { method: method || (payload === undefined ? 'GET' : 'POST'), credentials: 'same-origin' };
    if (payload !== undefined) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = toAsciiJson(payload);
    }
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const error = new Error(formatDetail(data.detail) || `Ошибка сервера (HTTP ${res.status})`);
        error.status = res.status;
        throw error;
    }
    return data;
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[ch]);
}

function rub(n) {
    return new Intl.NumberFormat('ru-RU').format(n) + ' ₽';
}

function lengthLabel(seconds) {
    return seconds < 60 ? `${seconds} с` : `${seconds / 60} мин`;
}

// Waiting room: a line from /api/shop/quip every few seconds while `work` runs.
async function withQuips(stage, target, work) {
    let stopped = false;
    const show = async () => {
        if (stopped) return;
        try {
            const q = await api(`/api/shop/quip?stage=${encodeURIComponent(stage)}`);
            if (!stopped) target.innerHTML = `<p class="quip">${escapeHtml(q.text)}</p>`;
        } catch (e) { /* a missing quip is not worth an error */ }
    };
    show();
    const timer = setInterval(show, 4500);
    try {
        return await work();
    } finally {
        stopped = true;
        clearInterval(timer);
    }
}

// Telegram Login Widget. The bot's domain must be set in @BotFather (/setdomain).
function mountTelegramLogin(container, botUsername, onDone) {
    if (!botUsername) {
        container.innerHTML = '<p class="muted">Вход через Telegram ещё не настроен.</p>';
        return;
    }
    window.onTelegramAuth = async (user) => {
        try {
            await api('/api/shop/login', user);
            onDone();
        } catch (e) {
            alert('Не получилось войти: ' + e.message);
        }
    };
    const script = document.createElement('script');
    script.async = true;
    script.src = 'https://telegram.org/js/telegram-widget.js?22';
    script.setAttribute('data-telegram-login', botUsername);
    script.setAttribute('data-size', 'large');
    script.setAttribute('data-radius', '10');
    script.setAttribute('data-onauth', 'onTelegramAuth(user)');
    script.setAttribute('data-request-access', 'write');
    container.innerHTML = '';
    container.appendChild(script);
}
