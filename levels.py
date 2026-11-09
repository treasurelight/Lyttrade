"""Support / resistance key levels, built from the higher-timeframe candles:
previous day and previous week highs and lows, higher-timeframe swing points and round numbers.
Nearby levels are merged into one stronger level."""
import math

import pandas as pd

import smc

# one "big round number" step per market (None = don't use round numbers)
ROUND_STEP = {"XAU": 50.0, "BTC": 1000.0, "ETH": 100.0}


def _round_step(pair: str, cls: str):
    if cls == "forex":
        return 1.0 if "JPY" in pair else 0.01
    return ROUND_STEP.get(pair.split("/")[0]) if cls in ("metal", "crypto") else None


def _prev_period(htf: pd.DataFrame):
    """Previous completed UTC day and ISO week highs/lows."""
    d = htf.copy()
    d["date"] = d["datetime"].dt.date
    out = []
    days = sorted(d["date"].unique())
    if len(days) >= 2:
        prev = d[d["date"] == days[-2]]
        out += [(float(prev["high"].max()), 2, "PDH"), (float(prev["low"].min()), 2, "PDL")]
    iso = d["datetime"].dt.isocalendar()
    d["wk"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    weeks = sorted(d["wk"].unique())
    if len(weeks) >= 2:
        prevw = d[d["wk"] == weeks[-2]]
        out += [(float(prevw["high"].max()), 3, "PWH"), (float(prevw["low"].min()), 3, "PWL")]
    return out


def _cluster(levels, tol):
    out = []
    for p, s, lab in sorted(levels, key=lambda x: x[0]):
        if out and abs(p - out[-1]["price"]) <= tol:
            c = out[-1]
            w = c["strength"] + s
            c["price"] = (c["price"] * c["strength"] + p * s) / w
            c["strength"] = min(5, w)
            c["labels"].append(lab)
        else:
            out.append({"price": p, "strength": s, "labels": [lab]})
    for c in out:
        c["label"] = "+".join(list(dict.fromkeys(c["labels"]))[:2])
    return out


def ltf_levels(ltf: pd.DataFrame, lookback: int = 200):
    """Recent entry-timeframe swing highs/lows. A swing that price clearly turned away from
    (moved 1.5 ATR the other way within 12 candles) counts as a rejection level (strength 2),
    any other swing counts 1. Levels tested twice add up, so repeated rejections become strong."""
    a = float(smc.atr(ltf).iloc[-1])
    hi, lo = ltf["high"].values, ltf["low"].values
    sh, sl = smc.swings(ltf, 3)
    start = max(0, len(ltf) - lookback)
    out = []
    for i, p in sh:
        if i >= start:
            fut = lo[i + 1:i + 13]
            rej = len(fut) > 0 and fut.min() <= p - 1.5 * a
            out.append((float(p), 2 if rej else 1, "15m high" if rej else "15m swing"))
    for i, p in sl:
        if i >= start:
            fut = hi[i + 1:i + 13]
            rej = len(fut) > 0 and fut.max() >= p + 1.5 * a
            out.append((float(p), 2 if rej else 1, "15m low" if rej else "15m swing"))
    return out


def key_levels(pair: str, cls: str, htf: pd.DataFrame, ltf: pd.DataFrame = None):
    """Sorted list of {price, strength (1-5), label}."""
    atr_h = float(smc.atr(htf).iloc[-1])
    price = float(htf["close"].iloc[-1])
    raw = _prev_period(htf)
    sh, sl = smc.swings(htf, 3)
    raw += [(float(p), 2, "swing") for _, p in sh[-8:]]
    raw += [(float(p), 2, "swing") for _, p in sl[-8:]]
    if ltf is not None and len(ltf) > 30:
        raw += ltf_levels(ltf)
    step = _round_step(pair, cls)
    if step:
        span = 6 * atr_h
        k = math.ceil((price - span) / step)
        while k * step <= price + span:
            raw.append((k * step, 2, "round"))
            k += 1
    return _cluster(raw, 0.3 * atr_h)


def nearby(levels, price, min_strength=2):
    """Nearest resistance above and support below (each a level dict or None)."""
    ab = [L for L in levels if L["price"] > price and L["strength"] >= min_strength]
    be = [L for L in levels if L["price"] < price and L["strength"] >= min_strength]
    return (min(ab, key=lambda L: L["price"] - price) if ab else None,
            min(be, key=lambda L: price - L["price"]) if be else None)


def targets(side, entry, risk, levels, atr, rr1, rr2, floor):
    """Take-profits at rr1/rr2, pulled in so they stop just short of a strong opposing level.
    Returns {"tp1","tp2","rr1","rr2","note"} or {"blocked": level} when a level leaves less than `floor`R."""
    d = 1 if side == "BUY" else -1
    t1, t2 = entry + d * risk * rr1, entry + d * risk * rr2
    buf = 0.15 * atr
    note = ""
    ahead = sorted((L for L in levels if L["strength"] >= 2 and d * (L["price"] - entry) > 0.3 * risk),
                   key=lambda L: d * (L["price"] - entry))
    first = next((L for L in ahead if d * (L["price"] - t1) < 0), None)
    kind = "resistance" if side == "BUY" else "support"
    if first:
        n1 = first["price"] - d * buf
        if d * (n1 - entry) < floor * risk:
            return {"blocked": first}
        nxt = next((L for L in ahead if d * (L["price"] - first["price"]) > 0 and d * (L["price"] - t1) < 0), None)
        t2n = nxt["price"] - d * buf if nxt else t1
        t2 = t2n if d * (t2n - n1) >= 0.25 * risk else t1
        t1 = n1
        note = f"TP1 stops short of {kind} {first['price']:.5g} ({first['label']})"
    else:
        second = next((L for L in ahead if d * (L["price"] - t1) > 0 and d * (L["price"] - t2) < 0), None)
        if second:
            n2 = second["price"] - d * buf
            if d * (n2 - t1) >= 0.25 * risk:
                t2 = n2
                note = f"TP2 stops short of {kind} {second['price']:.5g} ({second['label']})"
    return {"tp1": t1, "tp2": t2, "rr1": d * (t1 - entry) / risk, "rr2": d * (t2 - entry) / risk, "note": note}
