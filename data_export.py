"""
data_export — analytical CSV presets from cached Coinbase bars.

Each preset produces a ZIP of small CSVs designed for a specific analytical
question. Uploaded to a fresh Claude conversation, the CSVs + README are
self-describing so that session can analyze without project context.

Presets:
  A — daily_overview      Universe + daily returns + cross-coin correlations
  B — hourly_features     6h cross-sectional snapshots for lead/lag analysis
  C — event_windows       Pre/post windows around large moves
  D — time_of_day         Hour-of-day / day-of-week aggregates per coin
  E — rolling_correlations  BTC correlation drift per coin per week

All numeric outputs are rounded to 4-6 decimal places for compactness.
Design: each ZIP stays under ~10 MB so Claude can process easily.
"""
import io, csv, json, zipfile, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from bisect import bisect_left
import numpy as np

log = logging.getLogger("data_export")
UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════

def _as_ts(bar):
    """Bar timestamp as string (already ISO Z-form in Coinbase bars)."""
    return bar["t"]

def _daily_aggregate(bars):
    """Collapse 15-min bars into daily OHLCV. Returns dict keyed by date (YYYY-MM-DD)."""
    if not bars: return {}
    by_date = defaultdict(lambda: {"o": None, "h": -1e18, "l": 1e18, "c": None, "v": 0.0, "n": 0})
    for b in sorted(bars, key=_as_ts):
        d = b["t"][:10]
        rec = by_date[d]
        if rec["o"] is None: rec["o"] = b["o"]
        if b["h"] > rec["h"]: rec["h"] = b["h"]
        if b["l"] < rec["l"]: rec["l"] = b["l"]
        rec["c"] = b["c"]
        rec["v"] += b["v"]
        rec["n"] += 1
    return dict(by_date)


def _pct_return(a, b):
    if a is None or b is None or a == 0: return None
    return (b - a) / a


def _writer(rows, header=None):
    """Build a CSV string from rows (list of dicts). Writes header if provided,
    else uses keys of first row."""
    if not rows and header is None:
        return ""
    buf = io.StringIO()
    if header is None:
        header = list(rows[0].keys())
    w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        # Round floats for compactness
        clean = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
        w.writerow(clean)
    return buf.getvalue()


def _rsi(closes, period=14):
    """Simple RSI; returns None if insufficient data."""
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        if ch > 0: gains.append(ch); losses.append(0)
        else: gains.append(0); losses.append(-ch)
    avg_g = sum(gains[-period:]) / period
    avg_l = sum(losses[-period:]) / period
    if avg_l == 0: return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


# ═══════════════════════════════════════════════════════════════════
# PRESET A — Daily Overview
# ═══════════════════════════════════════════════════════════════════

def build_daily_overview(intraday_bars, products, categories, benchmark):
    """
    Produces:
      universe.csv — one row per coin: category, avg_daily_volume_usd,
                     total_return_pct, realized_vol_pct, current_price, n_days
      daily_returns.csv — one row per (coin, date): return, range, volume,
                          vs_BTC (coin_return - btc_return)
      daily_correlations.csv — pairwise correlation of daily returns (long
                                format: coin_a, coin_b, correlation, n_obs)
    """
    daily_by_coin = {}   # coin -> {date: {o,h,l,c,v,n}}
    for coin in products + [benchmark]:
        bars = intraday_bars.get(coin, [])
        if not bars: continue
        daily_by_coin[coin] = _daily_aggregate(bars)

    # --- universe.csv ---
    universe_rows = []
    all_dates_seen = set()
    for coin in products:
        d = daily_by_coin.get(coin, {})
        if not d: continue
        dates = sorted(d.keys())
        all_dates_seen.update(dates)
        first_close = d[dates[0]]["c"]
        last_close = d[dates[-1]]["c"]
        total_ret = _pct_return(first_close, last_close)
        # realized vol = stdev of daily log returns, annualized
        log_rets = []
        for i in range(1, len(dates)):
            p0 = d[dates[i-1]]["c"]; p1 = d[dates[i]]["c"]
            if p0 and p1 and p0 > 0 and p1 > 0:
                log_rets.append(np.log(p1 / p0))
        vol_ann = (np.std(log_rets) * np.sqrt(365)) if log_rets else None
        avg_dv = sum(d[x]["v"] * d[x]["c"] for x in dates) / len(dates) if dates else 0
        universe_rows.append({
            "coin": coin,
            "category": categories.get(coin, "?"),
            "current_price": last_close,
            "total_return_pct": (total_ret * 100) if total_ret is not None else None,
            "realized_vol_ann_pct": (vol_ann * 100) if vol_ann is not None else None,
            "avg_daily_volume_usd": avg_dv,
            "n_days": len(dates),
            "first_date": dates[0],
            "last_date": dates[-1],
        })
    universe_rows.sort(key=lambda r: -(r.get("avg_daily_volume_usd") or 0))

    # --- daily_returns.csv ---
    # Compute BTC daily returns first for the vs_BTC column
    btc_daily = daily_by_coin.get(benchmark, {})
    btc_ret_by_date = {}
    btc_dates = sorted(btc_daily.keys())
    for i in range(1, len(btc_dates)):
        d0, d1 = btc_dates[i-1], btc_dates[i]
        btc_ret_by_date[d1] = _pct_return(btc_daily[d0]["c"], btc_daily[d1]["c"])

    returns_rows = []
    returns_by_coin = defaultdict(dict)   # coin -> date -> return (for correlation later)
    for coin in products:
        d = daily_by_coin.get(coin, {})
        dates = sorted(d.keys())
        for i in range(1, len(dates)):
            d0, d1 = dates[i-1], dates[i]
            ret = _pct_return(d[d0]["c"], d[d1]["c"])
            if ret is None: continue
            rng = (d[d1]["h"] - d[d1]["l"]) / d[d1]["c"] if d[d1]["c"] > 0 else None
            vs_btc = (ret - btc_ret_by_date[d1]) if btc_ret_by_date.get(d1) is not None else None
            returns_rows.append({
                "date": d1,
                "coin": coin,
                "return_pct": ret * 100,
                "range_pct": (rng * 100) if rng is not None else None,
                "volume_usd": d[d1]["v"] * d[d1]["c"],
                "vs_btc_pct": (vs_btc * 100) if vs_btc is not None else None,
            })
            returns_by_coin[coin][d1] = ret

    # --- daily_correlations.csv ---
    # Pairwise Pearson on daily returns, long-format. To keep size manageable
    # and useful: only compute for coin pairs where BOTH have ≥ 60 overlapping days.
    corr_rows = []
    coins_sorted = sorted(returns_by_coin.keys())
    for i, a in enumerate(coins_sorted):
        for b in coins_sorted[i+1:]:
            # Align
            dates_common = set(returns_by_coin[a].keys()) & set(returns_by_coin[b].keys())
            if len(dates_common) < 60: continue
            dates_sorted = sorted(dates_common)
            va = np.array([returns_by_coin[a][d] for d in dates_sorted])
            vb = np.array([returns_by_coin[b][d] for d in dates_sorted])
            if va.std() == 0 or vb.std() == 0: continue
            corr = float(np.corrcoef(va, vb)[0, 1])
            corr_rows.append({
                "coin_a": a,
                "coin_b": b,
                "correlation": corr,
                "n_obs": len(dates_sorted),
            })
    # Keep only meaningful pairs: |corr| > 0.2 to keep file size sane;
    # with 100 coins we'd otherwise have 4950 rows of mostly-zero correlations
    corr_rows = [r for r in corr_rows if abs(r["correlation"]) >= 0.2]
    corr_rows.sort(key=lambda r: -abs(r["correlation"]))

    readme = f"""# Preset A — Daily Overview

Generated: {datetime.now(UTC).isoformat()}
Universe size: {len(universe_rows)} coins
Date range: {min(all_dates_seen) if all_dates_seen else "?"} to {max(all_dates_seen) if all_dates_seen else "?"}

## Files

### universe.csv
One row per coin. Columns:
- coin, category
- current_price, total_return_pct (over full period), realized_vol_ann_pct
- avg_daily_volume_usd (for liquidity ranking)
- n_days, first_date, last_date

### daily_returns.csv
One row per (coin, date). Columns:
- date, coin
- return_pct (daily close-to-close return)
- range_pct (daily high-low as % of close)
- volume_usd
- vs_btc_pct (coin return minus BTC return that day)

### daily_correlations.csv
Pairwise Pearson correlations on daily returns, filtered to |corr| >= 0.2 for
size. Columns: coin_a, coin_b, correlation, n_obs. Sorted by absolute
correlation descending.

## Questions this preset answers well
- Which coins have the highest returns / volatility over this period?
- Do certain coins consistently outperform or underperform BTC?
- Which coin pairs move together? (correlation clusters)
- Which coins stand out from their category?

## Questions this preset does NOT answer
- Intraday (sub-daily) patterns → use Preset B (hourly_features) or D (time_of_day)
- What happens before/after big moves → use Preset C (event_windows)
- Changes in correlation over time → use Preset E (rolling_correlations)
"""

    return {
        "universe.csv": _writer(universe_rows),
        "daily_returns.csv": _writer(returns_rows),
        "daily_correlations.csv": _writer(corr_rows),
        "README.md": readme,
    }


# ═══════════════════════════════════════════════════════════════════
# PRESET B — Hourly Features (6h snapshots for lead/lag)
# ═══════════════════════════════════════════════════════════════════

def build_hourly_features(intraday_bars, products, categories, benchmark,
                            slot_hours=6):
    """
    For each (coin, UTC date, slot), compute a snapshot of features. Sampled to
    slot_hours (default 6h) to keep file size manageable.

    hourly_features.csv columns:
      timestamp, date, slot_hour, coin, category
      ret_1slot_pct    (return since last slot)
      ret_4slot_pct    (24h return)
      vol_8slot_pct    (48h realized vol, annualized)
      rsi_14
      vs_btc_1slot_pct (coin return minus BTC return this slot)
      distance_from_btc_zscore (z-score of recent coin-BTC return diff)
      volume_z (volume z-score vs 20-slot mean)

    Each row is a snapshot at the CLOSE of slot [date] [slot_hour].
    """
    if slot_hours not in (1, 2, 3, 4, 6, 8, 12, 24):
        raise ValueError("slot_hours must divide 24")
    bars_per_slot = slot_hours * 4  # 15-min bars per slot

    # Pre-sort BTC bars and make slot-close index for BTC returns
    btc_bars = sorted(intraday_bars.get(benchmark, []), key=_as_ts)
    btc_closes_by_slot = {}   # (date, slot) -> close
    for i, b in enumerate(btc_bars):
        d = b["t"][:10]
        hour = int(b["t"][11:13])
        slot = (hour // slot_hours) * slot_hours
        # We only store the "last bar" of each slot; use bar hour+minute
        minute = int(b["t"][14:16])
        # Approximate slot-close: last bar starting at slot_end_hour - 15min
        slot_end_min = (slot + slot_hours) * 60
        bar_start_min = hour * 60 + minute
        if bar_start_min == slot_end_min - 15:   # final 15m bar of slot
            btc_closes_by_slot[(d, slot)] = b["c"]

    # Build list of (date, slot) tuples we'll emit rows for
    all_slots = sorted(btc_closes_by_slot.keys())
    if not all_slots:
        return {"hourly_features.csv": "", "README.md": "Empty (no BTC data)"}

    # Pre-compute BTC slot-returns
    btc_slot_rets = {}
    for i in range(1, len(all_slots)):
        k0, k1 = all_slots[i-1], all_slots[i]
        btc_slot_rets[k1] = _pct_return(btc_closes_by_slot[k0], btc_closes_by_slot[k1])

    rows = []
    for coin in products:
        bars = sorted(intraday_bars.get(coin, []), key=_as_ts)
        if not bars: continue

        # Build slot-close map for this coin
        coin_slot_close = {}
        coin_slot_vol = {}   # sum of bar volumes within slot
        for b in bars:
            d = b["t"][:10]
            hour = int(b["t"][11:13])
            minute = int(b["t"][14:16])
            slot = (hour // slot_hours) * slot_hours
            key = (d, slot)
            # Accumulate volume
            coin_slot_vol[key] = coin_slot_vol.get(key, 0.0) + b["v"]
            # Slot close = last 15m bar of slot
            slot_end_min = (slot + slot_hours) * 60
            bar_start_min = hour * 60 + minute
            if bar_start_min == slot_end_min - 15:
                coin_slot_close[key] = b["c"]

        # Emit one row per slot, using sliding windows
        coin_slots = sorted(coin_slot_close.keys())
        closes = [coin_slot_close[k] for k in coin_slots]
        rolling_vols_window = []   # for volume_z
        for i, k in enumerate(coin_slots):
            close = coin_slot_close[k]
            # ret_1slot: vs prior slot
            ret_1slot = _pct_return(closes[i-1], close) if i >= 1 else None
            ret_4slot = _pct_return(closes[i-4], close) if i >= 4 else None
            # Realized vol from last 8 log returns
            if i >= 8:
                rets8 = []
                for j in range(i-7, i+1):
                    if j >= 1 and closes[j-1] > 0 and closes[j] > 0:
                        rets8.append(np.log(closes[j] / closes[j-1]))
                vol_8 = float(np.std(rets8)) * np.sqrt(365 * (24/slot_hours))
            else: vol_8 = None
            # RSI on last 15 slot closes
            rsi = _rsi(closes[max(0, i-14):i+1]) if i >= 14 else None
            # vs BTC
            btc_r = btc_slot_rets.get(k)
            vs_btc = (ret_1slot - btc_r) if (ret_1slot is not None and btc_r is not None) else None
            # Volume z-score vs 20-slot rolling mean
            vol_here = coin_slot_vol.get(k, 0.0)
            if i >= 20:
                recent_vols = [coin_slot_vol.get(coin_slots[j], 0) for j in range(i-19, i+1)]
                mean = np.mean(recent_vols); sd = np.std(recent_vols)
                volume_z = float((vol_here - mean) / sd) if sd > 0 else 0.0
            else:
                volume_z = None

            rows.append({
                "timestamp": f"{k[0]}T{k[1]:02d}:00:00Z",
                "date": k[0],
                "slot_hour": k[1],
                "coin": coin,
                "category": categories.get(coin, "?"),
                "ret_1slot_pct": (ret_1slot*100) if ret_1slot is not None else None,
                "ret_4slot_pct": (ret_4slot*100) if ret_4slot is not None else None,
                "vol_8slot_ann_pct": (vol_8*100) if vol_8 is not None else None,
                "rsi_14": rsi,
                "vs_btc_1slot_pct": (vs_btc*100) if vs_btc is not None else None,
                "volume_z": volume_z,
            })

    readme = f"""# Preset B — Hourly Features (cross-sectional snapshots)

Generated: {datetime.now(UTC).isoformat()}
Slot size: {slot_hours}h
Universe size: {len(set(r['coin'] for r in rows))} coins
Total rows: {len(rows)}

## File

### hourly_features.csv
One row per (coin, date, slot_hour). Sampled every {slot_hours} hours to keep
size manageable.

Columns:
- timestamp, date, slot_hour, coin, category
- ret_1slot_pct: return since previous slot (= {slot_hours}h return)
- ret_4slot_pct: return over last 4 slots (= {slot_hours*4}h return)
- vol_8slot_ann_pct: realized volatility over last 8 slots, annualized
- rsi_14: standard RSI on slot closes
- vs_btc_1slot_pct: coin return minus BTC return this slot
- volume_z: volume z-score vs 20-slot rolling mean (detects volume spikes)

## Questions this preset answers well
- Which coins consistently move BEFORE or AFTER BTC at this cadence?
- Pre-event signatures: sort by ret_4slot_pct and look at the rows LEADING UP to big returns
- Cross-sectional momentum: at time T, which coins are oversold vs their category?
- Volume anomalies: where is volume_z > 2.0? Do these precede moves?

## Lead/lag analysis tip
To find if coin A leads coin B: compute correlation between
A's ret_1slot_pct at time T vs B's ret_1slot_pct at time T+k slots
for k in [-4, 4]. Max correlation lag tells you the relationship.
"""

    return {
        "hourly_features.csv": _writer(rows),
        "README.md": readme,
    }


# ═══════════════════════════════════════════════════════════════════
# PRESET C — Event Windows
# ═══════════════════════════════════════════════════════════════════

def build_event_windows(intraday_bars, products, categories, benchmark,
                         min_move_pct=5.0, horizon_hours=8, pre_hours=6):
    """
    For every coin, find all windows where price moved >= min_move_pct within
    horizon_hours. For each such event, record a pre_hours window (bars BEFORE
    the move started) and the post window (bars DURING the move).

    events.csv:
      event_id, coin, category, trigger_time (time at start of window),
      peak_time, peak_return_pct, bars_to_peak, btc_return_same_window_pct

    event_features.csv:
      event_id, offset_bars (negative = pre-event), return_cum_pct (since
      event start), price_vs_event_start, volume, btc_return_cum_pct

    Output capped to 500 events max (sorted by peak_return descending) to
    keep file sizes tractable.
    """
    bars_per_hour = 4
    horizon_bars = horizon_hours * bars_per_hour
    pre_bars = pre_hours * bars_per_hour
    threshold = min_move_pct / 100.0

    btc_bars = sorted(intraday_bars.get(benchmark, []), key=_as_ts)
    btc_ts_to_close = {b["t"]: b["c"] for b in btc_bars}
    btc_ts_sorted = [b["t"] for b in btc_bars]

    def btc_return_window(start_ts, end_ts):
        if not btc_ts_sorted: return None
        i0 = bisect_left(btc_ts_sorted, start_ts)
        i1 = bisect_left(btc_ts_sorted, end_ts)
        if i0 >= len(btc_ts_sorted) or i1 >= len(btc_ts_sorted): return None
        p0 = btc_ts_to_close[btc_ts_sorted[i0]]
        p1 = btc_ts_to_close[btc_ts_sorted[i1]]
        return _pct_return(p0, p1)

    events = []
    event_id_counter = 0
    for coin in products:
        bars = sorted(intraday_bars.get(coin, []), key=_as_ts)
        n = len(bars)
        # Find non-overlapping events: when we find one, skip ahead horizon_bars
        i = 0
        while i < n - horizon_bars:
            entry_price = bars[i]["c"]
            # Search forward through horizon
            peak_high = entry_price
            peak_bar_idx = i
            for j in range(i+1, min(i + horizon_bars + 1, n)):
                if bars[j]["h"] > peak_high:
                    peak_high = bars[j]["h"]
                    peak_bar_idx = j
            peak_return = (peak_high - entry_price) / entry_price
            if peak_return >= threshold:
                event_id_counter += 1
                events.append({
                    "event_id": event_id_counter,
                    "coin": coin,
                    "category": categories.get(coin, "?"),
                    "trigger_time": bars[i]["t"],
                    "peak_time": bars[peak_bar_idx]["t"],
                    "peak_return_pct": peak_return * 100,
                    "bars_to_peak": peak_bar_idx - i,
                    "entry_price": entry_price,
                    "peak_price": peak_high,
                    "entry_bar_idx_global": i,    # used below to find the window
                })
                # Skip to end of this event's horizon so we don't get overlapping events
                i = peak_bar_idx + 1
            else:
                i += 1

    # Keep top 500 by peak return (largest first); but also ensure coverage
    # across coins — if only 500 available total, all kept
    events.sort(key=lambda e: -e["peak_return_pct"])
    events = events[:500]

    # Build feature rows for each event
    feature_rows = []
    for ev in events:
        coin = ev["coin"]
        bars = sorted(intraday_bars.get(coin, []), key=_as_ts)
        trigger_idx = ev["entry_bar_idx_global"]
        # Pre-window: [trigger_idx - pre_bars, trigger_idx)
        # Post-window: [trigger_idx, trigger_idx + horizon_bars]
        window_start = max(0, trigger_idx - pre_bars)
        window_end = min(len(bars) - 1, trigger_idx + horizon_bars)
        entry_price = ev["entry_price"]

        trigger_ts = bars[trigger_idx]["t"]
        for j in range(window_start, window_end + 1):
            offset = j - trigger_idx
            b = bars[j]
            cum_ret = _pct_return(entry_price, b["c"])
            btc_cum = btc_return_window(trigger_ts, b["t"])
            feature_rows.append({
                "event_id": ev["event_id"],
                "offset_bars": offset,
                "timestamp": b["t"],
                "return_cum_pct": (cum_ret * 100) if cum_ret is not None else None,
                "price_vs_entry_pct": (cum_ret * 100) if cum_ret is not None else None,
                "high_vs_entry_pct": ((b["h"] - entry_price) / entry_price * 100),
                "low_vs_entry_pct": ((b["l"] - entry_price) / entry_price * 100),
                "volume": b["v"],
                "btc_return_cum_pct": (btc_cum * 100) if btc_cum is not None else None,
            })

    # Clean up events.csv (remove internal index)
    for e in events: e.pop("entry_bar_idx_global", None)

    readme = f"""# Preset C — Event Windows

Generated: {datetime.now(UTC).isoformat()}
Trigger: first bar where high reaches +{min_move_pct}% within {horizon_hours}h
Window: {pre_hours}h before trigger through {horizon_hours}h after
Events captured: {len(events)} (top by peak return; capped at 500 for size)
Events are non-overlapping: once an event fires, we skip forward {horizon_hours}h
on the same coin.

## Files

### events.csv
One row per event. Columns:
- event_id (unique), coin, category
- trigger_time (UTC, start of window — when we'd have entered)
- peak_time, peak_return_pct (highest high / entry - 1)
- bars_to_peak (how many 15-min bars to reach the peak)
- entry_price, peak_price

### event_features.csv
One row per event × offset_bar. Columns:
- event_id
- offset_bars (negative = pre-event, 0 = trigger bar, positive = post-event)
- timestamp
- return_cum_pct, price_vs_entry_pct, high_vs_entry_pct, low_vs_entry_pct
  (all cumulative % from entry price)
- volume
- btc_return_cum_pct (BTC's cumulative return since trigger time)

## Questions this preset answers well
- What are common features in the {pre_hours}h before big moves?
  (filter event_features to offset_bars < 0; look for patterns)
- How are big moves correlated with BTC moves? Do they lead or lag BTC?
- Which coins have the most explosive peaks?
- How long does it take to reach the peak? (bars_to_peak distribution)

## Quick analysis starting points
1. Group events.csv by category: are certain sectors more explosive?
2. Look at the offset=0 bar's volume vs offset=-4 bar's volume: do big moves
   pre-announce themselves with volume?
3. Check btc_return_cum_pct at offset=0 — do most events happen during BTC
   rallies, or independently?
4. For events with high peak_return_pct, is bars_to_peak small (explosive) or
   large (grinding)?
"""

    return {
        "events.csv": _writer(events),
        "event_features.csv": _writer(feature_rows),
        "README.md": readme,
    }


# ═══════════════════════════════════════════════════════════════════
# PRESET D — Time-of-Day / Day-of-Week
# ═══════════════════════════════════════════════════════════════════

def build_time_of_day(intraday_bars, products, categories, benchmark):
    """
    Per-coin aggregates by hour-of-day (UTC) and day-of-week.

    hour_of_day_stats.csv:
      coin, category, hour_utc, n_samples, mean_return_pct, win_rate, std_pct
      (stats over 15-min bar returns starting at that hour)

    day_of_week_stats.csv:
      coin, category, day_of_week (0=Mon), n_samples, mean_daily_return_pct,
      win_rate, std_pct
    """
    import calendar

    hour_rows = []
    dow_rows = []

    for coin in products + [benchmark]:
        bars = sorted(intraday_bars.get(coin, []), key=_as_ts)
        if len(bars) < 100: continue
        # 1-bar return per bar
        rets_by_hour = defaultdict(list)
        for i in range(1, len(bars)):
            b = bars[i]
            prev = bars[i-1]
            if prev["c"] <= 0: continue
            r = (b["c"] - prev["c"]) / prev["c"]
            hour = int(b["t"][11:13])
            rets_by_hour[hour].append(r)
        for hr in range(24):
            rs = rets_by_hour.get(hr, [])
            if len(rs) < 5: continue
            hour_rows.append({
                "coin": coin,
                "category": categories.get(coin, "?"),
                "hour_utc": hr,
                "n_samples": len(rs),
                "mean_return_pct": float(np.mean(rs)) * 100,
                "win_rate": float(np.mean([1 if r > 0 else 0 for r in rs])),
                "std_pct": float(np.std(rs)) * 100,
            })

        # Day of week — aggregate daily returns
        daily = _daily_aggregate(bars)
        dates_sorted = sorted(daily.keys())
        dow_rets = defaultdict(list)
        for i in range(1, len(dates_sorted)):
            d0 = dates_sorted[i-1]; d1 = dates_sorted[i]
            r = _pct_return(daily[d0]["c"], daily[d1]["c"])
            if r is None: continue
            # Compute day of week
            dt = datetime.strptime(d1, "%Y-%m-%d")
            dow = dt.weekday()
            dow_rets[dow].append(r)
        for dow in range(7):
            rs = dow_rets.get(dow, [])
            if len(rs) < 3: continue
            dow_rows.append({
                "coin": coin,
                "category": categories.get(coin, "?"),
                "day_of_week": dow,
                "day_name": calendar.day_name[dow],
                "n_samples": len(rs),
                "mean_daily_return_pct": float(np.mean(rs)) * 100,
                "win_rate": float(np.mean([1 if r > 0 else 0 for r in rs])),
                "std_pct": float(np.std(rs)) * 100,
            })

    readme = f"""# Preset D — Time-of-Day / Day-of-Week Aggregates

Generated: {datetime.now(UTC).isoformat()}
Universe size: {len(set(r['coin'] for r in hour_rows))} coins

## Files

### hour_of_day_stats.csv
For each (coin, hour_utc), aggregate return stats using 15-min bar returns
where the bar STARTS at that hour. Columns:
- coin, category, hour_utc (0-23)
- n_samples (each sample = one 15-min return during that hour across all days)
- mean_return_pct, win_rate (fraction positive), std_pct

### day_of_week_stats.csv
For each (coin, day_of_week), aggregate daily-return stats. Columns:
- coin, category, day_of_week (0=Mon...6=Sun), day_name
- n_samples (distinct days)
- mean_daily_return_pct, win_rate, std_pct

## Questions this preset answers well
- Are certain hours of day systematically bullish or bearish?
- Does volatility concentrate in specific hours (US market open, Asia close)?
- Do weekends systematically differ from weekdays?
- Which coins show the strongest time-of-day effects? (= the ones whose
  mean_return_pct varies most across hours)

## Gotchas
- Mean returns on 15-min bars are NOISY; statistical significance requires
  n_samples in the hundreds + effect size > 1-2 standard errors.
- Crypto markets run 24/7 but liquidity varies. Low-liquidity hours may have
  wider bid-ask bounce, inflating apparent volatility.
- Day-of-week effects on crypto have historically been weak; expect mostly null.
"""
    return {
        "hour_of_day_stats.csv": _writer(hour_rows),
        "day_of_week_stats.csv": _writer(dow_rows),
        "README.md": readme,
    }


# ═══════════════════════════════════════════════════════════════════
# PRESET E — Rolling Correlations to BTC
# ═══════════════════════════════════════════════════════════════════

def build_rolling_correlations(intraday_bars, products, categories, benchmark,
                                 window_days=30):
    """
    Rolling 30-day Pearson correlation of daily returns vs BTC, computed weekly.

    rolling_corr_btc.csv:
      date, coin, category, corr_to_btc, n_obs (in window)
    volatility_regimes.csv:
      date, btc_vol_30d_pct, btc_vol_percentile, regime_label
      (so Claude can condition analyses on regime)
    """
    daily_by_coin = {}
    for coin in products + [benchmark]:
        bars = intraday_bars.get(coin, [])
        if bars: daily_by_coin[coin] = _daily_aggregate(bars)

    btc_daily = daily_by_coin.get(benchmark, {})
    btc_dates = sorted(btc_daily.keys())
    if len(btc_dates) < window_days + 7:
        return {"rolling_corr_btc.csv": "", "volatility_regimes.csv": "", "README.md": "Insufficient data"}

    # Daily returns
    returns_by_coin = {}
    for coin, d in daily_by_coin.items():
        dates = sorted(d.keys())
        rets = {}
        for i in range(1, len(dates)):
            r = _pct_return(d[dates[i-1]]["c"], d[dates[i]]["c"])
            if r is not None: rets[dates[i]] = r
        returns_by_coin[coin] = rets

    btc_rets = returns_by_coin.get(benchmark, {})

    # Compute once-per-week endpoints
    weekly_dates = btc_dates[window_days::7]    # step 7 days
    rows = []
    for end_date in weekly_dates:
        # Window: [end_date - window_days, end_date)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=window_days)
        window_dates = [d for d in btc_dates
                         if datetime.strptime(d, "%Y-%m-%d") >= start_dt
                         and datetime.strptime(d, "%Y-%m-%d") <= end_dt]
        if len(window_dates) < 20: continue

        btc_w = np.array([btc_rets.get(d, 0) for d in window_dates if d in btc_rets])
        if btc_w.std() == 0: continue

        for coin in products:
            coin_rets = returns_by_coin.get(coin, {})
            common = [d for d in window_dates if d in coin_rets and d in btc_rets]
            if len(common) < 20: continue
            a = np.array([coin_rets[d] for d in common])
            b = np.array([btc_rets[d] for d in common])
            if a.std() == 0 or b.std() == 0: continue
            corr = float(np.corrcoef(a, b)[0, 1])
            rows.append({
                "date": end_date,
                "coin": coin,
                "category": categories.get(coin, "?"),
                "corr_to_btc_30d": corr,
                "n_obs": len(common),
            })

    # Volatility regimes
    regime_rows = []
    btc_closes = [btc_daily[d]["c"] for d in btc_dates]
    # Daily log returns
    log_rets = []
    for i in range(1, len(btc_closes)):
        if btc_closes[i] > 0 and btc_closes[i-1] > 0:
            log_rets.append(np.log(btc_closes[i] / btc_closes[i-1]))
        else:
            log_rets.append(0.0)
    # 30-day rolling vol
    for i in range(window_days - 1, len(log_rets)):
        window = log_rets[i - window_days + 1 : i + 1]
        vol = float(np.std(window)) * np.sqrt(365)
        regime_rows.append({
            "date": btc_dates[i + 1],
            "btc_vol_30d_ann_pct": vol * 100,
        })
    # Assign percentile + regime label
    vols = [r["btc_vol_30d_ann_pct"] for r in regime_rows]
    if vols:
        for r in regime_rows:
            pct = float(np.mean([v <= r["btc_vol_30d_ann_pct"] for v in vols]))
            r["btc_vol_percentile"] = pct
            if pct <= 0.33: r["regime_label"] = "low_vol"
            elif pct <= 0.67: r["regime_label"] = "mid_vol"
            else: r["regime_label"] = "high_vol"

    readme = f"""# Preset E — Rolling Correlations + Volatility Regimes

Generated: {datetime.now(UTC).isoformat()}
Rolling window: {window_days} days
Cadence: one data point per week per coin

## Files

### rolling_corr_btc.csv
For each (coin, weekly endpoint), the {window_days}-day trailing Pearson
correlation of DAILY RETURNS vs BTC. Columns:
- date (window endpoint), coin, category
- corr_to_btc_30d (Pearson, range [-1, 1])
- n_obs (number of overlapping daily returns in the window)

### volatility_regimes.csv
Daily BTC realized volatility + regime label. Columns:
- date
- btc_vol_30d_ann_pct (30-day realized vol, annualized)
- btc_vol_percentile (empirical percentile over the full period)
- regime_label ("low_vol" if ≤33rd percentile, "mid_vol" if 33-67, "high_vol" if >67)

## Questions this preset answers well
- Which coins decoupled from BTC recently? (corr dropping close to 0 or negative)
- Which coins are consistently high-beta (corr > 0.8 throughout)?
- Do correlations rise during high-vol regimes? (crypto's classic "correlation-of-1" phenomenon)
- Regime-conditional analysis: join regime_label to any other data to filter

## Analysis tip
If you want to know "when is this coin independent vs dependent on BTC,"
plot corr_to_btc_30d over time. Big swings up = decoupling; swings to 1 =
sector-wide sell-off mode.
"""

    return {
        "rolling_corr_btc.csv": _writer(rows),
        "volatility_regimes.csv": _writer(regime_rows),
        "README.md": readme,
    }


# ═══════════════════════════════════════════════════════════════════
# ZIP BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_preset_zip(preset_name, files_dict):
    """Pack a dict of {filename: content_string} into a ZIP bytes buffer."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, content in files_dict.items():
            z.writestr(fname, content)
    buf.seek(0)
    return buf.getvalue()


# Registry
PRESET_REGISTRY = {
    "daily_overview": build_daily_overview,
    "hourly_features": build_hourly_features,
    "event_windows": build_event_windows,
    "time_of_day": build_time_of_day,
    "rolling_correlations": build_rolling_correlations,
    "capitulation_paths": lambda intraday, products, categories, benchmark:
        build_capitulation_paths(intraday, products, categories, benchmark),
}

PRESET_DESCRIPTIONS = {
    "daily_overview": "Universe + daily returns + cross-coin correlations (≈2MB)",
    "hourly_features": "6h cross-sectional snapshots per coin with returns/vol/RSI/volume_z (≈5-10MB)",
    "event_windows": "Pre/post windows for all ≥5% moves (top 500 events, ≈3-5MB)",
    "time_of_day": "Per-coin hour-of-day and day-of-week return aggregates (≈1MB)",
    "rolling_correlations": "30-day rolling correlation to BTC + BTC volatility regime labels (≈1-2MB)",
    "capitulation_paths": "15-min price paths for every 6h drop ≥10% setup — for intra-horizon TP/SL simulation (≈2-4MB)",
}


def build_capitulation_paths(intraday_bars, products, categories, benchmark):
    """
    Export preset for intra-horizon P&L simulation of the capitulation-bounce rule.

    Finds every setup where a coin's 6h return (trailing 24 × 15min bars) ≤ -10%,
    then exports the post-setup bars for 12h (48 × 15min bars).

    Output:
      setups.csv: one row per setup:
        setup_id, coin, category, setup_time, setup_price (entry),
        ret_6h_pct (actual value), date
      paths.csv: one row per (setup_id, bar_offset):
        setup_id, bar_offset (1-48 = minutes 15..720 after entry),
        timestamp, open, high, low, close, volume
    """
    setups = []
    setup_id = 0

    for coin in products:
        bars = sorted(intraday_bars.get(coin, []), key=lambda b: b["t"])
        if len(bars) < 80: continue

        # Walk bars looking for 6h drops
        # Use non-overlapping setups: once one fires, skip 48 bars ahead
        i = 24  # need 24 bars of history for 6h return
        while i < len(bars) - 48:
            close_now = bars[i]["c"]
            close_6h_ago = bars[i - 24]["c"]
            if close_6h_ago <= 0 or close_now <= 0:
                i += 1; continue
            ret_6h = (close_now / close_6h_ago - 1) * 100
            if ret_6h < -10:
                # This is a setup. Record it and the next 48 bars of path.
                setup_id += 1
                setups.append({
                    "setup_id": setup_id,
                    "coin": coin,
                    "category": categories.get(coin, "?"),
                    "setup_time": bars[i]["t"],
                    "date": bars[i]["t"][:10],
                    "setup_price": close_now,
                    "ret_6h_pct": ret_6h,
                    "_bar_idx": i,   # internal: strip before export
                })
                i += 48  # skip 12h forward, non-overlapping
            else:
                i += 1

    # Build paths.csv rows
    path_rows = []
    for s in setups:
        bars = sorted(intraday_bars.get(s["coin"], []), key=lambda b: b["t"])
        start_idx = s["_bar_idx"]
        entry = s["setup_price"]
        for offset in range(1, 49):   # bars 1..48 = next 12h
            idx = start_idx + offset
            if idx >= len(bars): break
            b = bars[idx]
            path_rows.append({
                "setup_id": s["setup_id"],
                "bar_offset": offset,
                "timestamp": b["t"],
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
                # Precomputed cumulative % from entry
                "close_vs_entry_pct": round((b["c"]/entry - 1) * 100, 4) if entry > 0 else None,
                "high_vs_entry_pct":  round((b["h"]/entry - 1) * 100, 4) if entry > 0 else None,
                "low_vs_entry_pct":   round((b["l"]/entry - 1) * 100, 4) if entry > 0 else None,
            })

    # Strip internal fields
    for s in setups: s.pop("_bar_idx", None)

    readme = f"""# Preset F — Capitulation-Bounce Setup Paths

Generated: {datetime.now(UTC).isoformat()}
Setups captured: {len(setups)}
Total path bars: {len(path_rows)}

## Files

### setups.csv
One row per capitulation-bounce setup. A setup is any bar where the coin's
trailing 6h return (24 × 15min bars) is ≤ -10%. Non-overlapping: once a setup
fires, the next 12h on that coin are skipped.

Columns:
- setup_id (1..N)
- coin, category
- setup_time (UTC timestamp when setup fires)
- date (YYYY-MM-DD)
- setup_price (bar close price at setup time = entry price)
- ret_6h_pct (actual 6h return — always ≤ -10 by definition of setup)

### paths.csv
15-minute price path for each setup's following 12 hours (48 bars).

Columns:
- setup_id
- bar_offset (1..48 = minutes 15 to 720 after entry)
- timestamp (UTC)
- open, high, low, close, volume (raw OHLCV)
- close_vs_entry_pct, high_vs_entry_pct, low_vs_entry_pct
  (cumulative % change from entry_price — precomputed for convenience)

## Intended analysis

Simulate TP/SL exit strategies with realistic intra-bar granularity:
  For each setup, walk through bars 1..48. Exit at first bar where:
    - high_vs_entry_pct ≥ TP_threshold (TP triggered at TP price)
    - low_vs_entry_pct ≤ -SL_threshold (SL triggered at SL price)
  If neither triggers, exit at bar 48 close (horizon).

The hourly_features preset can only do close-to-close checks at 6h intervals;
this preset gives true intra-bar simulation.

## Questions this preset answers

- Does TP+3%/SL-2% recover more EV than close-to-close simulation suggests?
- What fraction of "no-stop" winners would have hit TP early?
- What fraction of "no-stop" losers would have hit SL early?
- Is there an optimal TP/SL combo we missed at lower granularity?
"""
    return {
        "setups.csv": _writer(setups),
        "paths.csv": _writer(path_rows),
        "README.md": readme,
    }
