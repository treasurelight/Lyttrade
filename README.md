# Forex SMC Telegram Signal Bot

Scans forex pairs (SMC structure, order blocks, FVG, liquidity sweeps, EMA/RSI, optional AI news sentiment, economic-calendar blackout) and sends BUY/SELL signals with entry, SL, TP1, TP2 to Telegram. Signals only; it never places trades. Not financial advice.

## Run
1. `cp .env.example .env` and fill in keys (never commit `.env`)
2. `pip install -r requirements.txt`
3. `python bot.py`  (Telegram: `/scan`, `/status`)

## Deploy (Render web service)
Build: `pip install -r requirements.txt` · Start: `python bot.py`
Env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVEDATA_API_KEY (+ optional FINNHUB_API_KEY, ANTHROPIC_API_KEY).
Use a paid plan: free web services sleep and the bot would stop.

## Support / resistance
Key levels (previous day/week high-low, 4h swings, round numbers) add score at support/resistance, penalise trades into a level, and shorten TPs before strong levels. Env: LEVELS_ON=true, MIN_TP1_RR=1.5, LEVEL_NEAR_ATR=1.5.

Env: CLOSED_ONLY=true (judge closed candles only), MAX_LIVE_DRIFT_ATR=1.0 (skip if price ran >1 ATR since the last closed candle). 15m swing/rejection levels now count as levels.

## Asian session
Between `ASIAN_START_UTC` (default 0) and `SESSION_START_UTC` (7) the bot also scans `ASIAN_PAIRS` (default USD/JPY, GBP/JPY, AUD/USD, XAU/USD) on weekdays. Only those pairs are scanned, to protect the 800/day data limit. Set `ASIAN_SESSION=false` to turn it off.

## Weekend crypto
`CRYPTO_ALWAYS` (default `BTC/USD`) lists crypto pairs scanned 24/7, weekends included. Add `ETH/USD` to scan it too (uses more of the 800/day data limit). `CRYPTO_24_7=true` makes all crypto 24/7.
