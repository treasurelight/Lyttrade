"""Forex signal bot for Telegram.
Combines SMC (structure, BOS/CHoCH, FVG, order blocks, liquidity sweeps,
premium/discount), EMA/RSI, news sentiment and the economic calendar.
Sends: pair, BUY/SELL, entry, SL, TP1, TP2.

NOT financial advice. Signals are probabilistic; always use proper risk management.
"""
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

import smc

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fxbot")

# ---------------- config ----------------
TD_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = str(os.getenv("TELEGRAM_CHAT_ID", ""))
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

PAIRS = [p.strip() for p in os.getenv(
    "PAIRS", "EUR/USD,GBP/USD,USD/JPY,AUD/USD,USD/CAD,USD/CHF").split(",") if p.strip()]
ENTRY_TF = os.getenv("ENTRY_TF", "15min")
HTF = os.getenv("HTF", "4h")
MIN_SCORE = float(os.getenv("MIN_SCORE", "6"))          # max possible is ~13
MIN_RR = float(os.getenv("MIN_RR", "2"))
SCAN_EVERY_MIN = int(os.getenv("SCAN_EVERY_MIN", "15"))
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "4"))
SESSION_START_UTC = int(os.getenv("SESSION_START_UTC", "7"))   # London open
SESSION_END_UTC = int(os.getenv("SESSION_END_UTC", "20"))      # NY close-ish
NEWS_BLACKOUT_MIN = int(os.getenv("NEWS_BLACKOUT_MIN", "30"))
API_PAUSE_SEC = float(os.getenv("API_PAUSE_SEC", "8"))         # free TwelveData: 8 req/min

_last_signal: dict = {}
_htf_cache: dict = {}
_news_cache = {"ts": 0, "scores": {}, "headlines": []}
_cal_cache = {"ts": 0, "events": []}


# ---------------- telegram ----------------
def tg_send(text: str, chat_id: str = None):
    """Send a Telegram message; retries on network errors and never crashes the bot."""
    for attempt in range(3):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": chat_id or TG_CHAT, "text": text, "parse_mode": "HTML"},
                timeout=40,
            )
            if not r.ok:
                log.error("telegram error: %s", r.text)
            return
        except requests.RequestException as e:
            log.warning("telegram send attempt %d failed: %s", attempt + 1, e)
            time.sleep(3)


# ---------------- data ----------------
def fetch_candles(pair: str, interval: str, size: int = 300) -> pd.DataFrame:
    time.sleep(API_PAUSE_SEC)
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": pair, "interval": interval, "outputsize": size,
                "apikey": TD_KEY, "order": "ASC", "timezone": "UTC"},
        timeout=30,
    )
    data = r.json()
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"{pair} {interval}: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.reset_index(drop=True)


def get_htf(pair: str) -> pd.DataFrame:
    cached = _htf_cache.get(pair)
    if cached and time.time() - cached[0] < 3600:
        return cached[1]
    df = fetch_candles(pair, HTF, 200)
    _htf_cache[pair] = (time.time(), df)
    return df


# ---------------- news & calendar ----------------
def get_calendar():
    """Free weekly calendar (ForexFactory mirror). Used for news blackout."""
    if time.time() - _cal_cache["ts"] < 3600:
        return _cal_cache["events"]
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20)
        _cal_cache["events"] = r.json() if r.ok else []
        _cal_cache["ts"] = time.time()
    except Exception as e:
        log.warning("calendar fetch failed: %s", e)
    return _cal_cache["events"]


def upcoming_high_impact(pair: str):
    """Return event title if a high-impact event for either currency is within the blackout window."""
    ccys = set(pair.split("/"))
    now = datetime.now(timezone.utc)
    for ev in get_calendar():
        try:
            if ev.get("impact") != "High" or ev.get("country") not in ccys:
                continue
            t = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
            if abs((t - now).total_seconds()) <= NEWS_BLACKOUT_MIN * 60:
                return f'{ev["country"]} {ev["title"]}'
        except Exception:
            continue
    return None


def get_news_scores():
    """Sentiment per currency in [-2, 2]. Needs FINNHUB + ANTHROPIC keys; otherwise neutral."""
    if time.time() - _news_cache["ts"] < 1800:
        return _news_cache["scores"], _news_cache["headlines"]
    scores, heads = {}, []
    if FINNHUB_KEY and ANTHROPIC_KEY:
        try:
            r = requests.get("https://finnhub.io/api/v1/news",
                             params={"category": "forex", "token": FINNHUB_KEY}, timeout=20)
            heads = [n["headline"] for n in r.json()[:30]]
            ccys = sorted({c for p in PAIRS for c in p.split("/")})
            prompt = (
                "You are a forex macro analyst. Based ONLY on these recent headlines, rate the "
                f"near-term directional pressure for each currency in {ccys} from -2 (strongly "
                "bearish) to 2 (strongly bullish); 0 if no relevant info. Reply with JSON only, "
                'e.g. {"USD": 1, "EUR": -1}.\n\n' + "\n".join(f"- {h}" for h in heads)
            )
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": CLAUDE_MODEL, "max_tokens": 300,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            txt = resp.json()["content"][0]["text"]
            scores = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        except Exception as e:
            log.warning("news scoring failed: %s", e)
    _news_cache.update(ts=time.time(), scores=scores, headlines=heads)
    return scores, heads


# ---------------- analysis ----------------
def fmt(pair: str, x: float) -> str:
    return f"{x:.3f}" if "JPY" in pair else f"{x:.5f}"


def pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def analyse(pair: str):
    """Returns signal dict or None."""
    ltf = fetch_candles(pair, ENTRY_TF, 300)
    htf = get_htf(pair)
    price = ltf["close"].iloc[-1]
    a = smc.atr(ltf).iloc[-1]

    buy, sell = 0.0, 0.0
    why_b, why_s = [], []

    def add(side, pts, reason):
        nonlocal buy, sell
        if side == "buy":
            buy += pts; why_b.append(reason)
        else:
            sell += pts; why_s.append(reason)

    # 1) HTF bias (weight 3)
    hsh, hsl = smc.swings(htf, 3)
    htrend = smc.trend_from_swings(hsh, hsl)
    ema50 = htf["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    if htrend == "bull" and htf["close"].iloc[-1] > ema50:
        add("buy", 3, f"{HTF} bullish structure (HH/HL) above EMA50")
    elif htrend == "bear" and htf["close"].iloc[-1] < ema50:
        add("sell", 3, f"{HTF} bearish structure (LH/LL) below EMA50")
    elif htrend == "bull":
        add("buy", 1, f"{HTF} bullish structure")
    elif htrend == "bear":
        add("sell", 1, f"{HTF} bearish structure")

    # 2) LTF structure break (weight 2)
    sh, sl = smc.swings(ltf, 3)
    brk = smc.last_break(ltf, sh, sl)
    if brk:
        side = "buy" if brk["dir"] == "up" else "sell"
        add(side, 2, f"{ENTRY_TF} {brk['type']} {brk['dir']}")

    # 3) Liquidity sweep (weight 2)
    sw = smc.liquidity_sweep(ltf, sh, sl)
    if sw:
        add("buy" if sw["dir"] == "bull" else "sell", 2,
            f"Liquidity sweep of {'lows' if sw['dir']=='bull' else 'highs'}")

    # 4) Premium / discount (weight 1)
    zone, _ = smc.premium_discount(ltf)
    add("buy" if zone == "discount" else "sell", 1, f"Price in {zone} zone")

    # 5) Retracement into FVG / order block (weight 2 each)
    last = ltf.iloc[-1]
    ob = smc.order_block(ltf, brk)
    ob_low = ob_high = None
    if ob and last["low"] <= ob["high"] and last["close"] >= ob["low"] and ob["dir"] == "bull":
        add("buy", 2, "Price reacting at bullish order block"); ob_low = ob["low"]
    elif ob and last["high"] >= ob["low"] and last["close"] <= ob["high"] and ob["dir"] == "bear":
        add("sell", 2, "Price reacting at bearish order block"); ob_high = ob["high"]
    for g in smc.fair_value_gaps(ltf):
        if g["low"] <= price <= g["high"]:
            add("buy" if g["dir"] == "bull" else "sell", 2,
                f"Price filling {g['dir']}ish FVG".replace("bullish", "bullish").replace("bearish", "bearish"))
            break

    # 6) EMA alignment + RSI (weight 1 each)
    e20 = ltf["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = ltf["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    if e20 > e50 and price > e20:
        add("buy", 1, "EMA20>EMA50 momentum")
    elif e20 < e50 and price < e20:
        add("sell", 1, "EMA20<EMA50 momentum")
    r = smc.rsi(ltf["close"]).iloc[-1]
    if r < 30:
        add("buy", 1, f"RSI oversold ({r:.0f})")
    elif r > 70:
        add("sell", 1, f"RSI overbought ({r:.0f})")

    # 7) News sentiment (weight up to 2)
    scores, _ = get_news_scores()
    base, quote = pair.split("/")
    ns = max(-2, min(2, scores.get(base, 0) - scores.get(quote, 0)))
    news_txt = "no sentiment data"
    if scores:
        news_txt = f"{base} {scores.get(base, 0):+} / {quote} {scores.get(quote, 0):+}"
        if ns > 0:
            add("buy", abs(ns), f"News flow favours {base} over {quote}")
        elif ns < 0:
            add("sell", abs(ns), f"News flow favours {quote} over {base}")

    side, score, reasons = ("BUY", buy, why_b) if buy >= sell else ("SELL", sell, why_s)
    opposing = sell if side == "BUY" else buy
    if score < MIN_SCORE or score - opposing < 3:
        return None
    # require HTF not against us
    if (side == "BUY" and htrend == "bear") or (side == "SELL" and htrend == "bull"):
        return None

    # Stop loss: beyond sweep/OB/recent swing, bounded by ATR
    recent = ltf.tail(12)
    if side == "BUY":
        sl_p = min(recent["low"].min(), ob_low or 1e9, sw["extreme"] if sw and sw["dir"] == "bull" else 1e9)
        sl_p -= 0.2 * a
        risk = price - sl_p
    else:
        sl_p = max(recent["high"].max(), ob_high or -1e9, sw["extreme"] if sw and sw["dir"] == "bear" else -1e9)
        sl_p += 0.2 * a
        risk = sl_p - price
    risk = max(risk, 0.8 * a)
    if risk > 3 * a:
        return None  # stop too wide, skip
    sl_p = price - risk if side == "BUY" else price + risk
    d = 1 if side == "BUY" else -1
    tp1, tp2 = price + d * risk * MIN_RR, price + d * risk * (MIN_RR + 1)

    return {
        "pair": pair, "side": side, "entry": price, "sl": sl_p, "tp1": tp1, "tp2": tp2,
        "risk_pips": risk / pip(pair), "score": score, "reasons": reasons, "news": news_txt,
        "htf_trend": htrend,
    }


def format_signal(s: dict) -> str:
    p = s["pair"]
    emoji = "🟢" if s["side"] == "BUY" else "🔴"
    conf = min(100, int(s["score"] / 13 * 100))
    return (
        f"{emoji} <b>{s['side']} {p}</b>\n"
        f"Entry: <code>{fmt(p, s['entry'])}</code>\n"
        f"SL: <code>{fmt(p, s['sl'])}</code> ({s['risk_pips']:.0f} pips)\n"
        f"TP1: <code>{fmt(p, s['tp1'])}</code> (1:{MIN_RR:g})\n"
        f"TP2: <code>{fmt(p, s['tp2'])}</code> (1:{MIN_RR + 1:g})\n"
        f"Confluence: {s['score']:.0f} pts (~{conf}%)\n"
        f"News: {s['news']}\n\n"
        + "\n".join(f"• {r}" for r in s["reasons"])
        + "\n\n⚠️ Not financial advice. Risk ≤1% per trade. Move SL to breakeven at TP1."
    )


def in_session() -> bool:
    now = datetime.now(timezone.utc)
    return now.weekday() < 5 and SESSION_START_UTC <= now.hour < SESSION_END_UTC


def scan(force: bool = False, reply_chat: str = None):
    if not force and not in_session():
        log.info("outside session, skipping")
        return
    sent = 0
    for pair in PAIRS:
        try:
            last = _last_signal.get(pair)
            if last and datetime.now() - last < timedelta(hours=COOLDOWN_HOURS):
                continue
            ev = upcoming_high_impact(pair)
            if ev:
                log.info("%s blocked by news: %s", pair, ev)
                continue
            sig = analyse(pair)
            if sig:
                tg_send(format_signal(sig), reply_chat)
                _last_signal[pair] = datetime.now()
                sent += 1
        except Exception as e:
            log.error("%s failed: %s", pair, e)
    if reply_chat and sent == 0:
        tg_send("Scan complete: no high-confluence setups right now.", reply_chat)


# ---------------- command polling + scheduler ----------------
def scheduler():
    while True:
        now = datetime.now(timezone.utc)
        wait = SCAN_EVERY_MIN * 60 - ((now.minute % SCAN_EVERY_MIN) * 60 + now.second) + 5
        time.sleep(wait)  # wake just after candle close
        try:
            scan()
        except Exception as e:
            log.error("scan crashed: %s", e)


def poll_commands():
    offset = None
    while True:
        try:
            r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                             params={"timeout": 30, "offset": offset}, timeout=40).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                m = u.get("message", {})
                chat = str(m.get("chat", {}).get("id", ""))
                text = (m.get("text") or "").strip().lower()
                if text:
                    log.info("telegram message from chat %s: %s", chat, text[:30])
                if chat != TG_CHAT:
                    if text.startswith("/"):
                        log.warning("ignored command from chat %s (TELEGRAM_CHAT_ID is %s)", chat, TG_CHAT)
                    continue
                if text.startswith("/start"):
                    tg_send("Forex SMC bot online.\n/scan – scan now\n/status – settings", chat)
                elif text.startswith("/scan"):
                    tg_send("Scanning… (forex market is closed on weekends, so data may be from Friday)"
                            if datetime.now(timezone.utc).weekday() >= 5 else "Scanning…", chat)
                    threading.Thread(target=scan, kwargs={"force": True, "reply_chat": chat}, daemon=True).start()
                elif text.startswith("/status"):
                    tg_send(f"Pairs: {', '.join(PAIRS)}\nTF: {HTF}→{ENTRY_TF}\nMin score: {MIN_SCORE}\n"
                            f"Session UTC: {SESSION_START_UTC}-{SESSION_END_UTC}\n"
                            f"News scoring: {'on' if FINNHUB_KEY and ANTHROPIC_KEY else 'off'}", chat)
        except Exception as e:
            log.error("poll error: %s", e)
            time.sleep(5)


class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def serve_health():
    """Render web services must bind $PORT; this tiny endpoint also works for uptime pings."""
    port = int(os.getenv("PORT", "0"))
    if port:
        HTTPServer(("0.0.0.0", port), _Health).serve_forever()


if __name__ == "__main__":
    if not (TD_KEY and TG_TOKEN and TG_CHAT):
        raise SystemExit("Set TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in .env")
    threading.Thread(target=serve_health, daemon=True).start()
    tg_send("✅ Forex SMC bot started.")
    threading.Thread(target=scheduler, daemon=True).start()
    poll_commands()
