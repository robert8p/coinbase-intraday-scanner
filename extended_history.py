"""
extended_history — 365-day bar cache for out-of-sample pattern validation.

Separate from the live scanner's 180-day operational cache. Fetches on demand,
persists to disk, supports analysis queries against historical windows.

Workflow:
  1. POST /api/extended/fetch  (triggers background fetch; takes ~1 hour)
  2. GET  /api/extended/status (poll progress)
  3. GET  /api/extended/oos_pattern?rule=capitulation_bounce&window=prior_180d
     (run the analysis on the prior 180-day window once fetched)
"""
import io, csv, json, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np

log = logging.getLogger("extended_history")
UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════════
# FEATURE BUILDER (for a given date window, from raw bars)
# ═══════════════════════════════════════════════════════════════════
# Must match the hourly_features.csv schema from data_export.py so the same
# pattern definitions translate 1:1. Simplified here — only the features
# needed for the capitulation-bounce pattern:
#   - ret_1slot_pct (6h return)
#   - rsi_14 (computed on 6h slot closes)
#   - volume_z (20-slot rolling z)
#   - fwd_24h = sum of next 4 6h returns

def build_hourly_for_window(intraday_bars, products, categories, benchmark,
                              start_date, end_date, slot_hours=6):
    """
    For each (coin, 6h slot) within [start_date, end_date], compute features.

    Returns list of dicts with keys: timestamp, date, slot_hour, coin, category,
    ret_1slot_pct, rsi_14, volume_z, fwd_6h, fwd_12h, fwd_18h, fwd_24h

    Note: fwd_24h requires 4 slots of future data past end_date. Slots where
    future isn't available return None and are dropped from pattern analysis.
    """
    bars_per_slot = slot_hours * 4

    rows_by_coin = {}   # coin -> list of (timestamp, close, volume)
    for coin in products:
        bars = sorted(intraday_bars.get(coin, []), key=lambda b: b["t"])
        if not bars: continue

        slot_closes = {}
        slot_vols = defaultdict(float)
        for b in bars:
            d = b["t"][:10]
            hour = int(b["t"][11:13])
            minute = int(b["t"][14:16])
            slot = (hour // slot_hours) * slot_hours
            key = (d, slot)
            slot_vols[key] += b["v"]
            # slot close = last 15m bar of slot
            slot_end_min = (slot + slot_hours) * 60
            bar_start_min = hour * 60 + minute
            if bar_start_min == slot_end_min - 15:
                slot_closes[key] = b["c"]

        sorted_keys = sorted(slot_closes.keys())
        rows_by_coin[coin] = [(k, slot_closes[k], slot_vols[k]) for k in sorted_keys]

    rsi_period = 14

    def _rsi(closes, period=14):
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

    rows = []
    for coin, coin_rows in rows_by_coin.items():
        closes = [r[1] for r in coin_rows]
        vols = [r[2] for r in coin_rows]
        for i, (k, close, vol) in enumerate(coin_rows):
            if i < 1: continue   # need previous close for ret
            d, slot = k
            if d < start_date or d > end_date: continue   # out of window
            ret_1slot = ((close - closes[i-1]) / closes[i-1] * 100) if closes[i-1] > 0 else None

            rsi = _rsi(closes[max(0, i-14):i+1]) if i >= 14 else None

            if i >= 20:
                window = vols[i-19:i+1]
                m = np.mean(window); sd = np.std(window)
                volume_z = float((vol - m) / sd) if sd > 0 else 0.0
            else:
                volume_z = None

            # Forward 24h = next 4 slots
            fwd_24h = None
            fwd_6h = None; fwd_12h = None; fwd_18h = None
            if i + 4 < len(coin_rows):
                fc = [coin_rows[j][1] for j in range(i, i+5)]   # entry + next 4
                if all(f > 0 for f in fc):
                    # Compound via sum of log returns ≈ sum of % returns for small moves
                    r1 = (fc[1]/fc[0] - 1) * 100
                    r2 = (fc[2]/fc[0] - 1) * 100
                    r3 = (fc[3]/fc[0] - 1) * 100
                    r4 = (fc[4]/fc[0] - 1) * 100
                    fwd_6h = r1; fwd_12h = r2; fwd_18h = r3; fwd_24h = r4

            rows.append({
                "timestamp": f"{d}T{slot:02d}:00:00Z",
                "date": d,
                "slot_hour": slot,
                "coin": coin,
                "category": categories.get(coin, "?"),
                "ret_1slot_pct": ret_1slot,
                "rsi_14": rsi,
                "volume_z": volume_z,
                "fwd_6h": fwd_6h,
                "fwd_12h": fwd_12h,
                "fwd_18h": fwd_18h,
                "fwd_24h": fwd_24h,
            })
    return rows


# ═══════════════════════════════════════════════════════════════════
# PATTERN EVALUATOR
# ═══════════════════════════════════════════════════════════════════

PATTERN_DEFS = {
    "capitulation_bounce": {
        "description": "RSI<30 AND 6h return<-10%",
        "setup": lambda r: (r["rsi_14"] is not None and r["rsi_14"] < 30
                              and r["ret_1slot_pct"] is not None
                              and r["ret_1slot_pct"] < -10),
        "outcomes": {
            "fwd_24h > 0": lambda r: r["fwd_24h"] is not None and r["fwd_24h"] > 0,
            "fwd_24h > +1%": lambda r: r["fwd_24h"] is not None and r["fwd_24h"] > 1,
            "fwd_24h > +2%": lambda r: r["fwd_24h"] is not None and r["fwd_24h"] > 2,
            "fwd_24h > +3%": lambda r: r["fwd_24h"] is not None and r["fwd_24h"] > 3,
        },
    },
    # Stricter variant
    "capitulation_bounce_strict": {
        "description": "RSI<25 AND 6h return<-10%",
        "setup": lambda r: (r["rsi_14"] is not None and r["rsi_14"] < 25
                              and r["ret_1slot_pct"] is not None
                              and r["ret_1slot_pct"] < -10),
        "outcomes": {
            "fwd_24h > 0": lambda r: r["fwd_24h"] is not None and r["fwd_24h"] > 0,
            "fwd_24h > +2%": lambda r: r["fwd_24h"] is not None and r["fwd_24h"] > 2,
        },
    },
}


def evaluate_pattern(rows, pattern_name):
    """
    Run the named pattern against the feature rows. Returns summary stats +
    per-fire details for deeper analysis.
    """
    if pattern_name not in PATTERN_DEFS:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    pdef = PATTERN_DEFS[pattern_name]

    fires = [r for r in rows
              if pdef["setup"](r)
              and r.get("fwd_24h") is not None]

    if not fires:
        return {
            "pattern": pattern_name, "description": pdef["description"],
            "n_setups": 0, "no_data": True,
        }

    fwd_24h_arr = np.array([f["fwd_24h"] for f in fires])
    winners = [f for f in fires if f["fwd_24h"] > 0]
    losers = [f for f in fires if f["fwd_24h"] <= 0]

    outcome_stats = {}
    for oname, ofn in pdef["outcomes"].items():
        hits = sum(1 for f in fires if ofn(f))
        outcome_stats[oname] = {
            "n_hits": hits,
            "hit_rate": hits / len(fires),
        }

    # Category + coin breakdown
    by_cat = defaultdict(list)
    by_coin = defaultdict(list)
    for f in fires:
        by_cat[f["category"]].append(f["fwd_24h"])
        by_coin[f["coin"]].append(f["fwd_24h"])

    cat_stats = {
        cat: {
            "n": len(vals),
            "hit_rate": sum(1 for v in vals if v > 0) / len(vals),
            "mean_fwd_24h": float(np.mean(vals)),
        }
        for cat, vals in by_cat.items()
    }
    coin_stats = {
        coin: {
            "n": len(vals),
            "hit_rate": sum(1 for v in vals if v > 0) / len(vals),
            "mean_fwd_24h": float(np.mean(vals)),
        }
        for coin, vals in by_coin.items()
    }

    return {
        "pattern": pattern_name,
        "description": pdef["description"],
        "n_setups": len(fires),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "fwd_24h": {
            "mean": float(fwd_24h_arr.mean()),
            "median": float(np.median(fwd_24h_arr)),
            "std": float(fwd_24h_arr.std()),
            "min": float(fwd_24h_arr.min()),
            "max": float(fwd_24h_arr.max()),
            "pct_25": float(np.percentile(fwd_24h_arr, 25)),
            "pct_75": float(np.percentile(fwd_24h_arr, 75)),
        },
        "outcomes": outcome_stats,
        "by_category": cat_stats,
        "by_coin": coin_stats,
        "fires": fires,   # full detail for loser analysis
    }


def compare_windows(rows, pattern_name, window_splits):
    """
    Evaluate pattern across multiple time windows for OOS comparison.
    window_splits: list of (label, start_date, end_date) tuples.
    """
    results = []
    for label, start, end in window_splits:
        window_rows = [r for r in rows if start <= r["date"] <= end]
        res = evaluate_pattern(window_rows, pattern_name)
        res["window_label"] = label
        res["window_start"] = start
        res["window_end"] = end
        res["n_rows_in_window"] = len(window_rows)
        results.append(res)
    return results


# ═══════════════════════════════════════════════════════════════════
# LOSER ANALYSIS
# ═══════════════════════════════════════════════════════════════════
# Given a list of setup fires, identify what characteristics differ
# systematically between winners and losers.

def analyze_losers(fires, features_to_compare=None):
    """
    For each numeric feature, compute the distribution in winners vs losers
    and flag features where the two distributions differ meaningfully.

    fires: output of evaluate_pattern()["fires"]
    features_to_compare: list of feature names; defaults to core ones.
    """
    if not features_to_compare:
        features_to_compare = ["rsi_14", "ret_1slot_pct", "volume_z"]

    winners = [f for f in fires if f["fwd_24h"] > 0]
    losers = [f for f in fires if f["fwd_24h"] <= 0]

    if not winners or not losers:
        return {"error": "Need both winners and losers to compare", "n_winners": len(winners), "n_losers": len(losers)}

    comparisons = {}
    for feat in features_to_compare:
        w_vals = [f[feat] for f in winners if f.get(feat) is not None]
        l_vals = [f[feat] for f in losers if f.get(feat) is not None]
        if not w_vals or not l_vals: continue
        w_arr = np.array(w_vals); l_arr = np.array(l_vals)
        # Welch's t-test approximation: difference relative to pooled std
        diff = w_arr.mean() - l_arr.mean()
        pooled_std = np.sqrt((w_arr.var() + l_arr.var()) / 2) if len(w_arr) > 1 and len(l_arr) > 1 else 1.0
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        comparisons[feat] = {
            "winner_mean": float(w_arr.mean()),
            "winner_median": float(np.median(w_arr)),
            "winner_std": float(w_arr.std()),
            "loser_mean": float(l_arr.mean()),
            "loser_median": float(np.median(l_arr)),
            "loser_std": float(l_arr.std()),
            "diff": float(diff),
            "effect_size": float(effect_size),   # |effect_size| > 0.5 = noteworthy
        }

    # Category breakdown: which categories skew losers?
    cat_breakdown = {}
    for f in fires:
        c = f["category"]
        if c not in cat_breakdown:
            cat_breakdown[c] = {"n_winners": 0, "n_losers": 0}
        if f["fwd_24h"] > 0: cat_breakdown[c]["n_winners"] += 1
        else: cat_breakdown[c]["n_losers"] += 1
    for c, v in cat_breakdown.items():
        total = v["n_winners"] + v["n_losers"]
        v["total"] = total
        v["loser_rate"] = v["n_losers"] / total if total > 0 else None

    # Coin-level breakdown
    coin_breakdown = defaultdict(lambda: {"n_winners": 0, "n_losers": 0})
    for f in fires:
        if f["fwd_24h"] > 0: coin_breakdown[f["coin"]]["n_winners"] += 1
        else: coin_breakdown[f["coin"]]["n_losers"] += 1
    loser_heavy_coins = []
    for coin, v in coin_breakdown.items():
        total = v["n_winners"] + v["n_losers"]
        if total >= 2 and v["n_losers"] / total > 0.5:
            loser_heavy_coins.append({
                "coin": coin,
                "n_total": total,
                "n_losers": v["n_losers"],
                "loser_rate": v["n_losers"] / total,
            })
    loser_heavy_coins.sort(key=lambda x: -x["loser_rate"])

    return {
        "n_winners": len(winners),
        "n_losers": len(losers),
        "feature_comparisons": comparisons,
        "by_category": cat_breakdown,
        "loser_heavy_coins": loser_heavy_coins,
    }
