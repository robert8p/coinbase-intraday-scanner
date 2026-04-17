# Coinbase Intraday Scanner

A port of the Russell 2000 intraday scanner adapted for Coinbase-tradeable coins.
Same architecture — FastAPI + LightGBM + React — with the structural changes
that 24/7 crypto markets require.

## What this does

Ranks ~140 liquid Coinbase USD spot pairs by the probability that price will hit
a take-profit barrier before a stop-loss barrier within a 4-hour horizon.

Six scan slots per day (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC), one
LightGBM model per slot, trained on first-passage labels from historical 15-min
candles.

## What changed from the R2K version

| R2K                              | Coinbase                                          | Why                                           |
| -------------------------------- | ------------------------------------------------- | --------------------------------------------- |
| Alpaca API (API key required)    | Coinbase Exchange public API (no auth)            | Easiest drop-in data source                   |
| 5 scan hours (11:00–15:00 ET)    | 6 scan slots (00/04/08/12/16/20 UTC)              | Crypto is 24/7; no closing bell               |
| Forced close at 15:55 ET         | Fixed 4-hour horizon per trade                    | No close → need explicit trade window         |
| 5-minute bars                    | 15-minute bars                                    | Better signal/noise over a 4h horizon         |
| ~300 Russell 2000 tickers        | ~140 curated USD pairs                            | Deduped list of what Coinbase actually trades |
| SPY as market factor             | BTC-USD as market factor                          | BTC is the crypto "market beta"               |
| GICS sectors                     | Crypto categories (L1/L2/DeFi/Meme/AI/Gaming/...) | Sector-relative features still work           |
| TP 0.95% / SL 1.50%              | TP 2.0% / SL 3.0%                                 | Crypto realized vol ≈ 2–3× equities           |
| 12 months of 5-min bars          | 6 months of 15-min bars                           | Similar sample count; fits Coinbase limits    |
| Batch `/stocks/bars`             | Per-product candles, paginated 300/request        | Coinbase hard-caps 300 candles per call       |
| Outcomes recorded 16:12 ET       | Outcomes recorded 00:30 UTC                       | 30 min after 20:00 scan's horizon ends        |

Feature count went from 33 to 32 — the only structural drop is `hours_left`
(no "end of day" in crypto). Everything else maps 1:1: momentum, VWAP
distance/slope, RSI, ATR-reach, realized vol, range expansion, ORB strength,
trend strength, BTC-relative return/momentum, category-relative return/momentum,
category breadth, gap%/gap-filled, plus 11 cross-sectional ranks.

## Environment variables

All optional. Coinbase public market-data endpoints don't need auth.

| Var                    | Default                              | Notes                        |
| ---------------------- | ------------------------------------ | ---------------------------- |
| `COINBASE_API_URL`     | `https://api.exchange.coinbase.com`  | Base URL for candles/stats   |
| `COINBASE_API_KEY`     | *(empty)*                            | Unused; reserved for future  |
| `COINBASE_API_SECRET`  | *(empty)*                            | Unused; reserved for future  |
| `TRAIN_DAYS`           | `180`                                | History window for training  |
| `PORT`                 | `10000`                              | HTTP port                    |

## Running locally

```bash
# Backend
pip install -r requirements.txt
uvicorn server:app --port 10000

# Frontend (separate terminal)
npm install
npm run dev   # opens http://localhost:5173
```

The Vite dev server proxies `/api` to `:10000`. For production, `npm run build`
emits `dist/`, which FastAPI serves directly (see the SPA fallback in `server.py`).

## Running with Docker

```bash
docker build -t coinbase-scanner .
docker run -p 10000:10000 -v $(pwd)/.data:/data coinbase-scanner
# → http://localhost:10000
```

## Deploying to Render

`render.yaml` is ready to go. Push to GitHub, connect the repo, and Render
picks it up. The 1GB disk at `/mnt/data` persists trained models, cached bars,
scans, and recorded outcomes across deploys.

## First-run workflow

1. Open the UI → **Training** tab → click **Train**.
2. First training fetches ~6 months of 15-min bars for ~140 pairs. Expect
   **20–30 minutes**. Progress bar updates live.
3. Subsequent trainings (e.g. different TP/SL) use the cached bars — about
   **3–5 minutes**.
4. Once trained, **Scanner** tab shows ranked pairs for the current UTC slot.
   Cron scans run automatically 5 min after each slot.
5. Outcomes record daily at 00:30 UTC. After ~10 days you'll have enough to
   see real top-10 win rates in the **Outcomes** tab.

## Strategy sweep

The **Training → Sweep** section runs a 3×5 grid over TP ∈ {1, 2, 3}% and
SL ∈ {1, 1.5, 2, 3, 4}%. Each cell is a full retrain. Resumable. Results
persist to `/data/sweep_results.json` so you can close the browser.

## Key API endpoints

| Endpoint                    | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| `GET /api/health`           | Server status, active TP/SL, universe    |
| `GET /api/scan/{hour}`      | Scan results for a UTC slot (cached live)|
| `POST /api/scan/{hour}/refresh` | Force a fresh scan                  |
| `POST /api/train`           | Train all 6 models (optional TP/SL body) |
| `GET /api/training/progress`| Training progress + model meta           |
| `POST /api/sweep`           | Start grid search                        |
| `GET /api/sweep/status`     | Sweep progress                           |
| `GET /api/sweep/results`    | Sweep grid results                       |
| `GET /api/outcomes/summary` | Recent recorded outcomes                 |
| `GET /api/diagnostic`       | Full diagnostic dump (downloads as JSON) |

## Notes on the universe

`PRODUCTS` in `server.py` is a curated list. Coinbase's actual listable set
changes: new listings land, a few get delisted. The scanner tolerates this —
pairs without sufficient candles are skipped. To refresh the universe, hit
`https://api.exchange.coinbase.com/products`, filter to `quote_currency="USD"`
and `status="online"`, and update the list.

## Caveats

- Coinbase's public candle endpoint is not recommended for high-frequency
  polling. The scanner hits it every 4 hours on the scheduler, which is well
  within reason, but don't aggressively refresh.
- Crypto has many liquidity regimes (weekends, Asian hours). The single-model
  approach still works but you may see slot-to-slot performance variation.
  Run the **Sweep** after ~30 days of outcomes to find TP/SL that holds up
  across slots.
- This is a research tool. No position sizing, no execution, no risk management
  beyond the labeled first-passage barriers. Size trades yourself.
