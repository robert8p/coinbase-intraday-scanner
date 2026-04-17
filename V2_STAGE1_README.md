# v2 Stage 1 — Vol-Normalized Threshold Classifier

## What's new

This stage adds a fundamentally new scanner framework alongside v1. v1 endpoints
are preserved (for backwards compatibility) but should not be used to draw
conclusions — v1's first-passage TP/SL framework was shown to be unproductive.

### New approach in one sentence

Instead of "will price hit +2% before −1%?" (v1), v2 asks "will price touch
+k × this coin's 7-day ATR at any point within H hours?" — a simpler,
volatility-adapted question.

### New artifacts

- **`v2.py`** — self-contained module with:
  - `compute_atr_fraction(bars)` → 7-day ATR as fraction of price
  - `did_touch_threshold(entry, future, k, atr_frac, horizon)` → binary label
  - `compute_features_v2(...)` → 53-feature vector
  - `add_cross_sectional_features_v2(...)` → fills rank and category features
  - `run_train_cell_v2(k, H, ...)` → trains one cell, returns metrics

- **New endpoints in `server.py`:**
  - `POST /api/v2/train` — body `{"k_atr": 1.0, "horizon_hours": 4}` → trains one cell
  - `GET /api/v2/training/progress` — live status with phase/pct/message
  - `GET /api/v2/models` — list of trained cells with summary metrics
  - `GET /api/v2/model/{k_atr}/{horizon_hours}` — full cell metadata
  - `GET /api/v2/diagnostic` — downloadable JSON of all cells (for analysis)

### 53 features, 6 groups

**A. Price action, vol-normalized (13):** ret_1b/3b/6b/12b_atr, vwap_dist/slope_atr,
trend_str_atr, momentum_accel, bb_position/width/squeeze, macd_hist/cross.

**B. Momentum & reversal (8):** rsi_14, rsi_divergence, stoch_k/d/cross_up,
williams_r, cci_20, roc_6b.

**C. Volume (7):** vol_zscore, vol_surge, obv_slope, vwap_above_fraction,
rel_volume, vol_trend, dollar_volume_rank.

**D. Structure & breakouts (8):** resistance/support_dist_atr, range_position,
orb_strength_atr, higher_highs_5b, donchian_breakout, keltner_position,
squeeze_fire.

**E. BTC & category context (10):** btc_ret_4b/12b, btc_vol, ret_vs_btc_atr,
mom_vs_btc, beta_to_btc, cat_breadth, ret_vs_cat_atr, cat_strongest/weakest.

**F. Cross-sectional ranks (5):** rank_momentum/vol_zscore/bb_position/rsi/range_position.

**G. Time (2):** scan_hour_sin, scan_hour_cos (so global model can learn slot effects).

### Training method

- **Single global model per cell** (not per-slot). Scan hour encoded as sin/cos
  feature; model learns slot effects if they're real.
- **Purged time-series split:** 60% train / embargo / 20% val / embargo / 20% test.
  Embargo = ceil(horizon_hours / 24) days to prevent label leakage since the label
  depends on future H hours.
- **Validation set** is used only for early stopping and isotonic calibration.
- **Test set is what you should trust** — it's the one neither model nor
  calibrator has seen. All reported precision/AUC metrics are on the test set.

### Metrics reported per cell

- `base_rate_test` — fraction of test-set scans where label=1. This is the
  bar to beat.
- `auc_test` — ROC-AUC on test set. 0.5 = random, 0.6 = weak signal, 0.7+ = real.
- `precision_at_threshold` — at thresholds 0.50/0.60/0.70/**0.75**/0.80/0.85/0.90,
  reports precision, count of predictions, coverage, and avg-per-day.
- `top_k_precision` — precision at top 0.1% / 0.5% / 1% / 5% of predictions.
  Useful for ranking-oriented views.
- `feature_importance` — gain-normalized feature importance.

The precision-at-0.75 metric is your primary success measure: **if >= 0.75 with
at least 1 per-day average, the cell is a candidate for the production scanner.**

## How to test (Stage 1 is curl-only — no UI)

Once deployed:

```bash
# Start training one cell
curl -X POST https://YOUR-RENDER/api/v2/train \
  -H "Content-Type: application/json" \
  -d '{"k_atr": 1.0, "horizon_hours": 4}'

# Check progress (poll every 30s)
curl https://YOUR-RENDER/api/v2/training/progress

# When done, get the full metrics for that cell
curl https://YOUR-RENDER/api/v2/model/1/4 | python3 -m json.tool

# List all trained cells (after training several)
curl https://YOUR-RENDER/api/v2/models | python3 -m json.tool

# Download full diagnostic JSON
curl https://YOUR-RENDER/api/v2/diagnostic -o v2_diag.json
```

Training reuses v1's cached bars, so the first v2 train is ~3-5 minutes if v1
was recently trained. Otherwise it'll refetch (~20-30 min).

## Suggested first run

Train a diagonal slice of the grid — 5 cells that cover the space cheaply:

```bash
# k=0.5, H=2h (low threshold, short horizon)
curl -X POST .../api/v2/train -d '{"k_atr": 0.5, "horizon_hours": 2}'
# k=1.0, H=4h (baseline, roughly matches v1's 2% / 4h)
curl -X POST .../api/v2/train -d '{"k_atr": 1.0, "horizon_hours": 4}'
# k=1.5, H=8h (medium)
curl -X POST .../api/v2/train -d '{"k_atr": 1.5, "horizon_hours": 8}'
# k=2.0, H=12h (high threshold, medium horizon)
curl -X POST .../api/v2/train -d '{"k_atr": 2.0, "horizon_hours": 12}'
# k=2.5, H=24h (high threshold, long horizon)
curl -X POST .../api/v2/train -d '{"k_atr": 2.5, "horizon_hours": 24}'
```

Wait for each to finish (`progress` endpoint) before starting the next. Then
`curl .../api/v2/diagnostic -o v2_diag.json` and share it.

## What Stage 2 adds (not in this zip)

- Full 5×5 grid sweep as a background job (not 5 curls)
- React UI with heatmap of cells
- Clickable cells show setup explainer (per-winner feature values, top contributors)
- Plain-English setup descriptions
- Ability to "pin" a cell as the production candidate

## What Stage 3 adds (not in this zip)

- Live scanner that runs hourly against the pinned cell
- Precision-thresholded output (only fire when P ≥ 0.75)
- Outcome recording for forward-validation
- Final UI with scanner + outcomes + explainer tabs
