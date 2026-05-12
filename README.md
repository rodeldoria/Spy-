# Spy- — crypto + SPY paper-trading signal dashboard

Streamlit UI on top of the [Monte](../Monte) engine. No automated execution —
just signals, indicators, simulated paper trades, and a vector-DB-powered
pattern explorer so you can decide for yourself when to buy a dip or fade a
pump.

## What it does

- **Live signal feed** (home page) — tails alerts produced by Monte's dip/pump
  detector across your crypto + SPY watchlist, ranked by triangulated
  confidence.
- **Crypto Watchlist** — RSI, MACD, Bollinger, ATR, regime, and Monte-Carlo
  zone for each symbol on multiple timeframes.
- **Pattern Explorer** — encode the current OHLCV window, query Chroma for
  similar historical patterns, and show the forward-return distribution of the
  neighbours.
- **Budget & Paper Portfolio** — set your budget, simulate buys/sells against
  current alerts, watch P&L on the local paper book.
- **Backtest** — replay `~/.monte/alerts.jsonl` against realised forward
  returns; see hit-rate by confidence bucket.
- **Settings** — view current watchlists, triangulation weights, alert
  thresholds.

## Quickstart

```bash
# 1. install (editable; depends on the sibling Monte checkout)
pip install -e ../Monte
pip install -e .

# 2. backfill the vector store with a year of BTC and SPY history
python -m monte.patterns.ingest BTC-USD 1h --years 1
python -m monte.patterns.ingest SPY 1d --years 5

# 3. set env (optional — defaults work without keys)
cp .env.example .env

# 4. launch the dashboard
streamlit run app/streamlit_app.py
```

## Architecture

```
Spy- (this repo)              Monte (sibling repo)
  app/streamlit_app.py  ───▶  monte.alerts.tail_alerts
  app/pages/01_*.py     ───▶  monte.data.{crypto,prices} + monte.indicators
  app/pages/02_*.py     ───▶  monte.patterns.{encoder, match, vector_store}
  app/pages/03_*.py     ───▶  monte.broker.paper_book + monte.risk.sizing
  app/pages/04_*.py     ───▶  monte.alerts.engine.tail_alerts (replay)
  app/pages/05_*.py     ───▶  monte.config.settings
```

The dashboard never executes against a real broker. The "Paper Portfolio" page
runs a local simulated book in `~/.monte/paper/` and marks-to-market against
live mid prices from Coinbase / yfinance.

The signal stack triangulates five axes — technical (RSI/MACD/BB), Monte Carlo
zone, AI sentiment, regime alignment, vector-pattern similarity — and reports
both a score and an agreement-based confidence percentage. Magnitude alone
caps confidence at 50%; cross-axis agreement unlocks the rest.
