"""Smart Money Concepts (SMC) helpers: swings, structure, BOS/CHoCH, FVG,
order blocks, liquidity sweeps, premium/discount, ATR, RSI."""
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def swings(df: pd.DataFrame, n: int = 3):
    """Fractal swing highs/lows. Returns ([(idx, price)], [(idx, price)])."""
    h, l = df["high"].values, df["low"].values
    sh, sl = [], []
    for i in range(n, len(df) - n):
        if h[i] == h[i - n : i + n + 1].max() and h[i] > h[i - 1]:
            sh.append((i, h[i]))
        if l[i] == l[i - n : i + n + 1].min() and l[i] < l[i - 1]:
            sl.append((i, l[i]))
    return sh, sl


def trend_from_swings(sh, sl) -> str:
    """'bull' for HH+HL, 'bear' for LH+LL, else 'range'."""
    if len(sh) < 2 or len(sl) < 2:
        return "range"
    hh = sh[-1][1] > sh[-2][1]
    hl = sl[-1][1] > sl[-2][1]
    if hh and hl:
        return "bull"
    if not hh and not hl:
        return "bear"
    return "range"


def last_break(df, sh, sl, n=3, lookback=15):
    """Most recent close-through of a prior swing. Returns dict or None.
    type = BOS (with trend) or CHoCH (against prior trend)."""
    close = df["close"].values
    found = None
    for i in range(max(len(df) - lookback, 5), len(df)):
        ph = [p for idx, p in sh if idx < i - n]
        pl = [p for idx, p in sl if idx < i - n]
        if not ph or not pl:
            continue
        sh_before = [(idx, p) for idx, p in sh if idx < i - n]
        sl_before = [(idx, p) for idx, p in sl if idx < i - n]
        prior_trend = trend_from_swings(sh_before, sl_before)
        if close[i] > ph[-1] and close[i - 1] <= ph[-1]:
            found = {"dir": "up", "idx": i, "level": ph[-1],
                     "type": "BOS" if prior_trend == "bull" else "CHoCH"}
        elif close[i] < pl[-1] and close[i - 1] >= pl[-1]:
            found = {"dir": "down", "idx": i, "level": pl[-1],
                     "type": "BOS" if prior_trend == "bear" else "CHoCH"}
    return found


def fair_value_gaps(df, lookback=40):
    """Unfilled 3-candle imbalances. Returns list of dicts."""
    h, l = df["high"].values, df["low"].values
    out = []
    start = max(2, len(df) - lookback)
    for i in range(start, len(df)):
        if l[i] > h[i - 2]:  # bullish FVG
            lo, hi = h[i - 2], l[i]
            if i + 1 >= len(df) or l[i + 1 :].min() > lo:
                out.append({"dir": "bull", "low": lo, "high": hi, "idx": i - 1})
        elif h[i] < l[i - 2]:  # bearish FVG
            lo, hi = h[i], l[i - 2]
            if i + 1 >= len(df) or h[i + 1 :].max() < hi:
                out.append({"dir": "bear", "low": lo, "high": hi, "idx": i - 1})
    return out


def order_block(df, brk):
    """Last opposite-colour candle before the structure break impulse."""
    if not brk:
        return None
    o, c = df["open"].values, df["close"].values
    h, l = df["high"].values, df["low"].values
    for j in range(brk["idx"] - 1, max(brk["idx"] - 15, 0), -1):
        if brk["dir"] == "up" and c[j] < o[j]:
            return {"dir": "bull", "low": l[j], "high": h[j]}
        if brk["dir"] == "down" and c[j] > o[j]:
            return {"dir": "bear", "low": l[j], "high": h[j]}
    return None


def liquidity_sweep(df, sh, sl, n=3, recent=6):
    """Wick through a prior swing that closes back inside = stop hunt."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    for i in range(len(df) - 1, len(df) - 1 - recent, -1):
        pl = [p for idx, p in sl if idx < i - n]
        ph = [p for idx, p in sh if idx < i - n]
        if pl and l[i] < pl[-1] and c[i] > pl[-1]:
            return {"dir": "bull", "level": pl[-1], "extreme": l[i]}
        if ph and h[i] > ph[-1] and c[i] < ph[-1]:
            return {"dir": "bear", "level": ph[-1], "extreme": h[i]}
    return None


def premium_discount(df, window=60):
    seg = df.tail(window)
    hi, lo = seg["high"].max(), seg["low"].min()
    mid = (hi + lo) / 2
    price = df["close"].iloc[-1]
    return ("discount" if price < mid else "premium"), mid
