"""Forex signal bot for Telegram.
Combines SMC (structure, BOS/CHoCH, FVG, order blocks, liquidity sweeps,
premium/discount), EMA/RSI, news sentiment and the economic calendar.
Sends: pair, BUY/SELL, entry, SL, TP1, TP2.

NOT financial advice. Signals are probabilistic; always use proper risk management.
"""
import html
import json
import logging
import os
import re
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
CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
TG_CHAT = CHAT_IDS[0] if CHAT_IDS else ""
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")                 # free: 25 requests/day
AV_REFRESH_HOURS = float(os.getenv("AV_REFRESH_HOURS", "8"))   # ~7 calls per refresh -> 21/day
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

PAIRS = [p.strip() for p in os.getenv(
    "PAIRS", "EUR/USD,GBP/USD,USD/JPY,XAU/USD,BTC/USD,ETH/USD,SPY,QQQ,USO").split(",") if p.strip()]
ENTRY_TF = os.getenv("ENTRY_TF", "15min")
HTF = os.getenv("HTF", "4h")
MIN_SCORE = float(os.getenv("MIN_SCORE", "6"))          # max possible is ~13
MIN_RR = float(os.getenv("MIN_RR", "2"))
SCAN_EVERY_MIN = int(os.getenv("SCAN_EVERY_MIN", "10"))
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "4"))
SESSION_START_UTC = int(os.getenv("SESSION_START_UTC", "7"))   # London open
SESSION_END_UTC = int(os.getenv("SESSION_END_UTC", "20"))      # NY close-ish
NEWS_BLACKOUT_MIN = int(os.getenv("NEWS_BLACKOUT_MIN", "30"))
API_PAUSE_SEC = float(os.getenv("API_PAUSE_SEC", "8"))         # free TwelveData: 8 req/min
MAX_DATA_AGE_MIN = int(os.getenv("MAX_DATA_AGE_MIN", "45"))   # reject stale candles
MIN_SL_PIPS = float(os.getenv("MIN_SL_PIPS", "10"))            # never give a tighter stop than this
WATCH_SCORE = float(os.getenv("WATCH_SCORE", "4"))            # forming setup -> "hold on" alert
WATCH_EXPIRE_HOURS = float(os.getenv("WATCH_EXPIRE_HOURS", "8"))
# --- stricter rules for crypto (it whipsaws more than forex) ---
CRYPTO_MIN_SCORE = float(os.getenv("CRYPTO_MIN_SCORE", "8"))
CRYPTO_24_7 = os.getenv("CRYPTO_24_7", "false").lower() == "true"   # default: London/NY hours only
CRYPTO_SL_BUFFER = float(os.getenv("CRYPTO_SL_BUFFER", "0.5"))      # ATRs beyond the swing
CRYPTO_SL_MIN_ATR = float(os.getenv("CRYPTO_SL_MIN_ATR", "1.5"))    # never tighter than this
CRYPTO_SL_MAX_ATR = float(os.getenv("CRYPTO_SL_MAX_ATR", "5"))
HEARTBEAT_HOURS = float(os.getenv("HEARTBEAT_HOURS", "2"))     # status message when nothing to report
EAT = timezone(timedelta(hours=3))
_last_beat = 0.0

_last_signal: dict = {}
_watch: dict = {}
_htf_cache: dict = {}
_news_cache = {"ts": 0, "scores": {}, "headlines": []}
_cal_cache = {"ts": 0, "events": []}
_av_cache = {"ts": 0, "scores": {}}


# ---------------- telegram ----------------
def tg_send(text: str, chat_id: str = None):
    """Send to one chat, or broadcast to every chat in TELEGRAM_CHAT_ID.
    Retries on network errors; if Telegram rejects the formatting, resends as plain text."""
    for target in ([chat_id] if chat_id else CHAT_IDS):
        body, mode = text, "HTML"
        for attempt in range(3):
            try:
                payload = {"chat_id": target, "text": body}
                if mode:
                    payload["parse_mode"] = mode
                r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                                  json=payload, timeout=40)
                if r.ok:
                    break
                log.error("telegram error (%s): %s", target, r.text)
                if "parse entities" in r.text and mode:
                    body, mode = html.unescape(re.sub(r"</?(b|code)>", "", text)), None
                    continue  # retry immediately as plain text
                break
            except requests.RequestException as e:
                log.warning("telegram send attempt %d failed: %s", attempt + 1, e)
                time.sleep(3)


# ---------------- data ----------------
_calls = {"day": "", "n": 0}
_quota = {"day": ""}


def count_call():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _calls["day"] != today:
        _calls.update(day=today, n=0)
    _calls["n"] += 1


def fetch_candles(pair: str, interval: str, size: int = 300) -> pd.DataFrame:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _quota["day"] == today:
        raise RuntimeError("daily data limit reached")
    time.sleep(API_PAUSE_SEC)
    count_call()
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": pair, "interval": interval, "outputsize": size,
                "apikey": TD_KEY, "order": "ASC", "timezone": "UTC"},
        timeout=30,
    )
    data = r.json()
    if data.get("status") == "error" or "values" not in data:
        msg = str(data.get("message", data))
        if data.get("code") == 429 and ("day" in msg.lower() or "daily" in msg.lower()):
            _quota["day"] = today
            tg_send("⏸ Daily data limit reached on the free Twelve Data plan. Scanning pauses until "
                    "00:00 UTC (03:00 EAT), then resumes by itself. Fewer markets or a longer scan "
                    "interval avoids this.")
        raise RuntimeError(f"{pair} {interval}: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.reset_index(drop=True)


def get_htf(pair: str) -> pd.DataFrame:
    cached = _htf_cache.get(pair)
    if cached and time.time() - cached[0] < 14400:
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
    ccys = currencies(pair)
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


def av_news_scores():
    """Free news sentiment from Alpha Vantage NEWS_SENTIMENT, one call per currency/coin.
    Averages relevance-weighted ticker sentiment (-1..1) and maps it to -2..2."""
    if time.time() - _av_cache["ts"] < AV_REFRESH_HOURS * 3600:
        return _av_cache["scores"]
    codes = set()
    for p in PAIRS:
        if "/" in p:
            for c in p.split("/"):
                if c not in METALS | OIL:
                    codes.add(c)
    scores = dict(_av_cache["scores"])
    for code in sorted(codes):
        ticker = f"CRYPTO:{code}" if code in CRYPTO else f"FOREX:{code}"
        try:
            r = requests.get("https://www.alphavantage.co/query",
                             params={"function": "NEWS_SENTIMENT", "tickers": ticker,
                                     "sort": "LATEST", "limit": 50, "apikey": AV_KEY}, timeout=30)
            feed = r.json().get("feed")
            if feed is None:
                log.warning("alpha vantage: %s", str(r.json())[:150])  # usually the daily limit
                break
            num = den = 0.0
            for art in feed:
                for ts in art.get("ticker_sentiment", []):
                    if ts.get("ticker") == ticker:
                        w = float(ts.get("relevance_score", 0))
                        num += w * float(ts.get("ticker_sentiment_score", 0))
                        den += w
            avg = num / den if den else 0.0
            scores[code] = max(-2, min(2, round(avg / 0.25)))
        except Exception as e:
            log.warning("alpha vantage %s failed: %s", ticker, e)
        time.sleep(13)  # free tier allows ~5 calls/min
    _av_cache.update(ts=time.time(), scores=scores)
    log.info("alpha vantage sentiment: %s", scores)
    return scores


def get_news_scores():
    """Sentiment per currency in [-2, 2]: Alpha Vantage (free) if keyed, else Finnhub+Claude, else neutral."""
    if AV_KEY:
        return av_news_scores(), []
    if time.time() - _news_cache["ts"] < 1800:
        return _news_cache["scores"], _news_cache["headlines"]
    scores, heads = {}, []
    if FINNHUB_KEY and ANTHROPIC_KEY:
        try:
            r = requests.get("https://finnhub.io/api/v1/news",
                             params={"category": "forex", "token": FINNHUB_KEY}, timeout=20)
            heads = [n["headline"] for n in r.json()[:30]]
            ccys = sorted({c for p in PAIRS if "/" in p for c in p.split("/")})
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
CRYPTO = {"BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LTC", "AVAX", "LINK"}
METALS = {"XAU", "XAG", "XPT", "XPD"}
OIL = {"WTI", "BRENT", "USOIL", "UKOIL"}


def asset_class(pair: str) -> str:
    if "/" not in pair:
        return "equity"  # indices / ETFs such as SPY, QQQ, USO
    base = pair.split("/")[0]
    if base in CRYPTO:
        return "crypto"
    if base in METALS:
        return "metal"
    if base in OIL:
        return "oil"
    return "forex"


def currencies(pair: str) -> set:
    return set(pair.split("/")) if "/" in pair else {"USD"}


def fmt(pair: str, x: float) -> str:
    if asset_class(pair) == "forex":
        return f"{x:.3f}" if "JPY" in pair else f"{x:.5f}"
    return f"{x:.2f}" if abs(x) >= 10 else f"{x:.4f}"


def pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def analyse(pair: str, min_score: float = None):
    """Returns {"type": "signal"|"watch"|"none", ...} or None when data is stale."""
    ltf = fetch_candles(pair, ENTRY_TF, 300)
    htf = get_htf(pair)
    age_min = (datetime.now(timezone.utc).replace(tzinfo=None) - ltf["datetime"].iloc[-1]).total_seconds() / 60
    if age_min > MAX_DATA_AGE_MIN:
        log.info("%s: latest candle is %.0f min old (market closed/stale feed), skipping", pair, age_min)
        return None
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
    base, quote = pair.split("/") if "/" in pair else (pair, "")
    is_crypto = asset_class(pair) == "crypto"
    if is_crypto:  # judge the coin on its own news, not against forex USD sentiment
        ns = max(-2, min(2, scores.get(base, 0)))
    else:
        ns = max(-2, min(2, scores.get(base, 0) - scores.get(quote, 0))) if quote else 0
    news_txt = "no sentiment data"
    if scores and is_crypto:
        news_txt = f"{base} {scores.get(base, 0):+} (coin news only)"
        if ns != 0:
            add("buy" if ns > 0 else "sell", abs(ns), f"{base} news flow is {'bullish' if ns > 0 else 'bearish'}")
    elif scores and quote:
        news_txt = f"{base} {scores.get(base, 0):+} / {quote} {scores.get(quote, 0):+}"
        if ns > 0:
            add("buy", abs(ns), f"News flow favours {base} over {quote}")
        elif ns < 0:
            add("sell", abs(ns), f"News flow favours {quote} over {base}")

    side, score, reasons = ("BUY", buy, why_b) if buy >= sell else ("SELL", sell, why_s)
    opposing = sell if side == "BUY" else buy
    need = MIN_SCORE if min_score is None else min_score
    htf_against = (side == "BUY" and htrend == "bear") or (side == "SELL" and htrend == "bull")
    cls = asset_class(pair)
    sl_buf, sl_min, sl_max = 0.2, 0.8, 3.0
    allowed = True
    if cls == "crypto":
        need = max(need, CRYPTO_MIN_SCORE)
        sl_buf, sl_min, sl_max = CRYPTO_SL_BUFFER, CRYPTO_SL_MIN_ATR, CRYPTO_SL_MAX_ATR
        # crypto must have the 4h trend AND a fresh 15min structure break pointing the same way
        allowed = (htrend == ("bull" if side == "BUY" else "bear")
                   and bool(brk) and ((brk["dir"] == "up") == (side == "BUY")))

    if score >= need and score - opposing >= 3 and not htf_against and allowed:
        # Stop loss: beyond sweep/OB/recent swing, bounded by ATR
        recent = ltf.tail(12)
        if side == "BUY":
            sl_p = min(recent["low"].min(), ob_low or 1e9, sw["extreme"] if sw and sw["dir"] == "bull" else 1e9)
            sl_p -= sl_buf * a
            risk = price - sl_p
        else:
            sl_p = max(recent["high"].max(), ob_high or -1e9, sw["extreme"] if sw and sw["dir"] == "bear" else -1e9)
            sl_p += sl_buf * a
            risk = sl_p - price
        risk = max(risk, sl_min * a)
        if risk <= sl_max * a:  # otherwise stop too wide: fall through to watch
            if cls == "forex":
                risk = max(risk, MIN_SL_PIPS * pip(pair))  # realistic minimum stop
            d = 1 if side == "BUY" else -1
            risk_txt = f"{risk / pip(pair):.0f} pips" if cls == "forex" else f"{risk:.2f} pts"
            return {
                "type": "signal", "pair": pair, "side": side, "entry": price,
                "sl": price - d * risk, "tp1": price + d * risk * MIN_RR,
                "tp2": price + d * risk * (MIN_RR + 1), "risk_txt": risk_txt, "price": price,
                "score": score, "reasons": reasons, "news": news_txt, "htf_trend": htrend,
            }

    if score >= WATCH_SCORE and score - opposing >= 2 and not htf_against:
        if side == "BUY":
            above = [p for _, p in sh if p > price]
            trig = above[-1] if above else max(ltf["high"].tail(20).max(), price + 0.5 * a)
            below = [p for _, p in sl if p < price]
            inv = below[-1] if below else price - 2 * a
        else:
            below = [p for _, p in sl if p < price]
            trig = below[-1] if below else min(ltf["low"].tail(20).min(), price - 0.5 * a)
            above = [p for _, p in sh if p > price]
            inv = above[-1] if above else price + 2 * a
        inv = min(inv, price - 0.5 * a) if side == "BUY" else max(inv, price + 0.5 * a)
        return {"type": "watch", "pair": pair, "side": side, "trigger": trig, "invalid": inv,
                "score": score, "reasons": reasons, "price": price}

    return {"type": "none", "price": price}


def format_signal(s: dict) -> str:
    p = s["pair"]
    emoji = "🟢" if s["side"] == "BUY" else "🔴"
    conf = min(100, int(s["score"] / 13 * 100))
    return (
        f"{emoji} <b>{s['side']} {p}</b>\n"
        f"Entry: <code>{fmt(p, s['entry'])}</code>\n"
        f"SL: <code>{fmt(p, s['sl'])}</code> ({s['risk_txt']})\n"
        f"TP1: <code>{fmt(p, s['tp1'])}</code> (1:{MIN_RR:g})\n"
        f"TP2: <code>{fmt(p, s['tp2'])}</code> (1:{MIN_RR + 1:g})\n"
        f"Confluence: {s['score']:.0f} pts (~{conf}%)\n"
        f"News: {html.escape(s['news'])}\n\n"
        + "\n".join(f"• {html.escape(r)}" for r in s["reasons"])
        + "\n\n⚠️ Not financial advice. Open ONE position per signal, risk ≤1%. Move SL to breakeven at TP1."
    )


def format_watch(w: dict) -> str:
    p = w["pair"]
    up = w["side"] == "BUY"
    return (
        f"👀 <b>HOLD ON: watching {w['side']} {p}</b>\n"
        "Not a trade yet. A setup is forming but not confirmed.\n\n"
        f"Price now: <code>{fmt(p, w['price'])}</code>\n"
        f"✅ I'll send the signal if a 15min candle closes {'above' if up else 'below'} "
        f"<code>{fmt(p, w['trigger'])}</code>\n"
        f"❌ Idea is dead if price closes {'below' if up else 'above'} "
        f"<code>{fmt(p, w['invalid'])}</code>\n\n"
        "Why it's on watch:\n" + "\n".join(f"• {html.escape(r)}" for r in w["reasons"])
        + "\n\nI'll message you when it confirms or fails."
    )


def in_session() -> bool:
    """Forex/metals/oil trading window (weekdays, London to NY)."""
    now = datetime.now(timezone.utc)
    return now.weekday() < 5 and SESSION_START_UTC <= now.hour < SESSION_END_UTC


def pair_in_session(pair: str) -> bool:
    cls = asset_class(pair)
    now = datetime.now(timezone.utc)
    if cls == "crypto":
        return True if CRYPTO_24_7 else in_session()  # weekday London/NY hours unless 24/7 is switched on
    if cls == "equity":  # US cash session 13:30-20:00 UTC
        return now.weekday() < 5 and 13 <= now.hour < 20
    return in_session()


def pair_market_open(pair: str) -> bool:
    cls = asset_class(pair)
    if cls == "crypto":
        return True
    if cls == "equity":
        return pair_in_session(pair)
    return market_open_now()


def market_open_now() -> bool:
    """Forex trades Sun 22:00 UTC to Fri 22:00 UTC."""
    now = datetime.now(timezone.utc)
    wd = now.weekday()
    if wd == 5 or (wd == 4 and now.hour >= 22) or (wd == 6 and now.hour < 22):
        return False
    return True


def next_scan_start() -> datetime:
    now = datetime.now(timezone.utc)
    d = now.replace(hour=SESSION_START_UTC, minute=0, second=0, microsecond=0)
    if d <= now:
        d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def status_text() -> str:
    nxt = next_scan_start()
    why = ("Forex market is closed for the weekend." if not market_open_now()
           else "Outside the London/NY trading session.")
    return (f"😴 {why} No setups to scan right now.\n"
            f"Next scan: {nxt:%a %H:%M} UTC ({nxt.astimezone(EAT):%a %H:%M} EAT).")


def watch_lines() -> str:
    if not _watch:
        return ""
    return "\n\n👀 Watching:\n" + "\n".join(
        f"• {w['side']} {p}: close {'above' if w['side'] == 'BUY' else 'below'} {fmt(p, w['trigger'])}"
        for p, w in _watch.items())


def heartbeat(msg: str):
    """Status message at most every HEARTBEAT_HOURS, so the bot is never silent."""
    global _last_beat
    if time.time() - _last_beat >= HEARTBEAT_HOURS * 3600:
        tg_send(msg + watch_lines())
        _last_beat = time.time()


def scan(force: bool = False, reply_chat: str = None):
    global _last_beat
    pairs = [p for p in PAIRS if (pair_market_open(p) if force else pair_in_session(p))]
    if not pairs:
        log.info("no markets in session, skipping")
        if not force:
            heartbeat(status_text())
        return
    sent = 0
    try:
        get_news_scores()
    except Exception as e:
        log.warning("news refresh failed: %s", e)
    for pair in pairs:
        try:
            last = _last_signal.get(pair)
            if last and datetime.now() - last < timedelta(hours=COOLDOWN_HOURS):
                continue
            ev = upcoming_high_impact(pair)
            if ev:
                log.info("%s blocked by news: %s", pair, ev)
                continue
            w = _watch.get(pair)
            res = analyse(pair, MIN_SCORE - 1 if w else None)
            if not res:
                continue  # stale data
            if res["type"] == "signal":
                head = "✅ <b>CONFIRMED</b> (was on watch)\n" if w else ""
                tg_send(head + format_signal(res), reply_chat)
                _watch.pop(pair, None)
                _last_signal[pair] = datetime.now()
                sent += 1
            elif w:
                px = res["price"]
                if (w["side"] == "BUY" and px < w["invalid"]) or (w["side"] == "SELL" and px > w["invalid"]):
                    tg_send(f"❌ {w['side']} {pair} idea invalidated (price {fmt(pair, px)}). Back to scanning.", reply_chat)
                    _watch.pop(pair, None)
                elif time.time() - w["ts"] > WATCH_EXPIRE_HOURS * 3600:
                    tg_send(f"⌛ Dropped the {w['side']} {pair} watch: nothing confirmed in {WATCH_EXPIRE_HOURS:g}h.", reply_chat)
                    _watch.pop(pair, None)
            elif res["type"] == "watch":
                _watch[pair] = {"side": res["side"], "trigger": res["trigger"],
                                "invalid": res["invalid"], "ts": time.time()}
                tg_send(format_watch(res), reply_chat)
                sent += 1
        except Exception as e:
            log.error("%s failed: %s", pair, e)
    if sent:
        _last_beat = time.time()
    elif reply_chat:
        tg_send(f"Scan complete ({len(pairs)} markets): nothing confirmed right now." + watch_lines(), reply_chat)
    else:
        heartbeat(f"🔎 Scanned {len(pairs)} markets: no confirmed setups. "
                  f"Still watching, next scan in {SCAN_EVERY_MIN} min.")


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


ALIASES = {"GOLD": "XAU/USD", "XAU": "XAU/USD", "SILVER": "XAG/USD", "BTC": "BTC/USD", "BITCOIN": "BTC/USD",
           "ETH": "ETH/USD", "OIL": "USO", "USOIL": "USO", "SPX": "SPY", "SP500": "SPY", "US500": "SPY",
           "NAS": "QQQ", "NAS100": "QQQ", "NASDAQ": "QQQ", "US100": "QQQ"}


def parse_pair(arg: str):
    a = arg.upper().replace(" ", "").replace("-", "").replace("_", "")
    if a in ALIASES:
        return ALIASES[a]
    if "/" in a:
        return a
    if len(a) == 6 and a.isalpha():
        return f"{a[:3]}/{a[3:]}"
    return a if a else None


def signal_for(pair: str, chat: str):
    """On-demand analysis of one market: signal, hold-on, or wait."""
    try:
        res = analyse(pair)
    except Exception as e:
        tg_send(f"Couldn't analyse {pair}: {str(e)[:150]}", chat)
        return
    if res is None:
        tg_send(f"⏸ {pair}: no live prices right now (market closed or data delayed). Try again when it's open.", chat)
    elif res["type"] == "signal":
        tg_send(format_signal(res), chat)
    elif res["type"] == "watch":
        tg_send(format_watch(res), chat)
    else:
        tg_send(f"⏳ <b>{pair}</b>: no clear setup yet (price {fmt(pair, res['price'])}). "
                "Hold off and wait. I'll alert you when something forms.", chat)


HELP = ("Forex SMC bot commands:\n"
        "/signal EURUSD – analyse one market now (also: gold, btc, eth, oil, nas100, spx)\n"
        "/scan – scan all markets\n/watch – setups I'm watching\n/status – settings\n"
        "/id – show this chat's ID")


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
                if not text.startswith("/"):
                    continue
                parts = text.split()
                cmd = parts[0].split("@")[0]          # "/scan@MyBot" in groups -> "/scan"
                args = " ".join(parts[1:])
                log.info("command %s from chat %s", cmd, chat)
                if cmd == "/id":                       # works anywhere, so you can find a group's ID
                    tg_send(f"This chat's ID: <code>{chat}</code>", chat)
                    continue
                if chat not in CHAT_IDS:
                    log.warning("ignored command from chat %s (allowed: %s)", chat, CHAT_IDS)
                    tg_send("This chat isn't authorised yet. Add this chat ID to TELEGRAM_CHAT_ID: "
                            f"<code>{chat}</code>", chat)
                    continue
                if cmd in ("/start", "/help"):
                    tg_send(HELP, chat)
                elif cmd == "/signal":
                    pair = parse_pair(args)
                    if not pair:
                        tg_send("Usage: /signal EURUSD  (or gold, btc, eth, oil, nas100, spx)", chat)
                    else:
                        tg_send(f"Analysing {pair}… (about 20 seconds)", chat)
                        threading.Thread(target=signal_for, args=(pair, chat), daemon=True).start()
                elif cmd == "/scan" and not any(pair_market_open(p) for p in PAIRS):
                    tg_send(status_text(), chat)
                elif cmd == "/scan":
                    tg_send("Scanning…", chat)
                    threading.Thread(target=scan, kwargs={"force": True, "reply_chat": chat}, daemon=True).start()
                elif cmd == "/watch":
                    tg_send(("👀 Watchlist" + watch_lines()) if _watch else "Nothing on watch right now.", chat)
                elif cmd == "/status":
                    tg_send(f"Pairs: {', '.join(PAIRS)}\nTF: {HTF}→{ENTRY_TF}\nMin score: {MIN_SCORE}\n"
                            f"Forex/metals session UTC: {SESSION_START_UTC}-{SESSION_END_UTC}; crypto {'24/7' if CRYPTO_24_7 else 'same hours'} (min score {CRYPTO_MIN_SCORE:g}); US ETFs 13-20\n"
                            f"Scan every {SCAN_EVERY_MIN} min · data calls today: {_calls['n']}/800\n"
                            f"News scoring: {'Alpha Vantage' if AV_KEY else ('Finnhub+Claude' if FINNHUB_KEY and ANTHROPIC_KEY else 'off (ALPHAVANTAGE_API_KEY not set)')}", chat)
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


def keep_awake():
    """Render's free plan sleeps after ~15 min without inbound HTTP traffic. Pinging our own
    public URL counts as traffic, so the bot stays up while it is running."""
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEPALIVE_URL", "")
    if not url:
        return
    while True:
        time.sleep(600)
        try:
            requests.get(url, timeout=30)
        except Exception as e:
            log.warning("keepalive ping failed: %s", e)


if __name__ == "__main__":
    if not (TD_KEY and TG_TOKEN and CHAT_IDS):
        raise SystemExit("Set TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in .env")
    threading.Thread(target=serve_health, daemon=True).start()
    threading.Thread(target=keep_awake, daemon=True).start()
    tg_send("✅ Forex SMC bot started.")
    threading.Thread(target=scheduler, daemon=True).start()
    poll_commands()
