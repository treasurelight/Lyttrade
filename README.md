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
