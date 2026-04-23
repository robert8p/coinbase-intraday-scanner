"""
v2 — Vol-normalized threshold classifier.

Goal: identify coins that have a high probability of *touching* +k*ATR within
H hours. This is different from v1's first-passage TP/SL framework:

- Label is one-sided (no stop loss): did high >= entry * (1 + k*ATR_7d)
  at any point during the horizon?
- Threshold is volatility-normalized (k*ATR_7d, not absolute %) so the model
  can't just learn "high vol coins hit high thresholds more often".
- Feature set is expanded to ~50 and includes ATR-normalized versions of
  price-action features, classical technical indicators (Bollinger, MACD,
  Stochastic, Williams %R, CCI, ROC), volume features (z-score, surge, OBV),
  structure features (support/resistance distance, range position, breakouts),
  and BTC/category context.
- Single global model per (k, H) cell, with scan_hour_sin/cos as features so
  the model can *learn* slot effects if they exist.
- Purged time-series split: train [0..T-H) / embargo [T-H..T) / val [T..end).
  Embargo prevents label leakage since labels depend on future H hours.

Stage 1 scope: label + features + single-cell train endpoint. No UI, no sweep,
no scanner output. Validated via curl only. Stage 2 builds the research UI.

v1 endpoints remain functional but should not be used to draw conclusions —
v1's first-passage framework was shown to be unproductive. We're keeping v1 on
disk purely so the deployment doesn't break during the transition.
"""
import os, json, time, math, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve

log = logging.getLogger("v2")

UTC = ZoneInfo("UTC")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
# Bar granularity matches v1 so cached bars can be reused.
CANDLE_GRANULARITY_SEC = 900               # 15-min bars
BARS_PER_HOUR = 3600 // CANDLE_GRANULARITY_SEC
BARS_PER_DAY = 24 * BARS_PER_HOUR

# Feature lookback: 48 bars = 12 hours of 15-min bars. Longer than v1's 24
# because some indicators (Bollinger period 20, MACD 26/12/9, ATR 7d) need more
# history. 48 is a compromise between indicator stability and data availability.
FEATURE_LOOKBACK_BARS = 48

# ATR window: 7 days in bar units. Matches "ATR_7d" in the label name.
ATR_WINDOW_BARS = 7 * BARS_PER_DAY          # 672

# Minimum intraday bars we need before we'll evaluate a coin for a given scan.
MIN_BARS_FOR_FEATURES = 30

# Minimum days of history for a coin to be included in training.
MIN_HISTORY_DAYS = 14

# Scan slots — same as v1. Model learns slot effects via cyclical features.
SCAN_HOURS = [0, 4, 8, 12, 16, 20]

# Purged time-series split: embargo = horizon_bars so training labels can't
# leak future information into validation. Calculated per (k, H) cell.

# ═══════════════════════════════════════════════════════════════════
# FEATURE NAMES — 51 features in 6 groups
# ═══════════════════════════════════════════════════════════════════
FEATURE_NAMES_V2 = [
    # --- A. Price action, vol-normalized (13) ---
    "ret_1b_atr","ret_3b_atr","ret_6b_atr","ret_12b_atr",
    "vwap_dist_atr","vwap_slope_atr","trend_str_atr","momentum_accel",
    "bb_position","bb_width","bb_squeeze","macd_hist","macd_cross",

    # --- B. Momentum & reversal (8) ---
    "rsi_14","rsi_divergence","stoch_k","stoch_d","stoch_cross_up",
    "williams_r","cci_20","roc_6b",

    # --- C. Volume (7) ---
    "vol_zscore","vol_surge","obv_slope","vwap_above_fraction",
    "rel_volume","vol_trend","dollar_volume_rank",

    # --- D. Structure & breakouts (8) ---
    "resistance_dist_atr","support_dist_atr","range_position",
    "orb_strength_atr","higher_highs_5b","donchian_breakout",
    "keltner_position","squeeze_fire",

    # --- E. BTC & category context (10) ---
    "btc_ret_4b","btc_ret_12b","btc_vol","ret_vs_btc_atr","mom_vs_btc",
    "beta_to_btc","cat_breadth","ret_vs_cat_atr","cat_strongest","cat_weakest",

    # --- F. Cross-sectional ranks (5) ---
    "rank_momentum","rank_vol_zscore","rank_bb_position","rank_rsi","rank_range_position",

    # --- G. Time (2) — so global model can learn slot effects if they exist ---
    "scan_hour_sin","scan_hour_cos",
]
assert len(FEATURE_NAMES_V2) == 53, f"expected 53, got {len(FEATURE_NAMES_V2)}"

# ═══════════════════════════════════════════════════════════════════
# ATR — 7-day rolling average true range, normalized to price
# ═══════════════════════════════════════════════════════════════════
def compute_atr_fraction(bars, window_bars=ATR_WINDOW_BARS):
    """
    ATR_7d as a *fraction of price* (so thresholds like '1 * ATR' are relative).
    bars: list of dicts with h, l, c keys, ascending time.
    Returns ATR/price as a float, or a floor value if insufficient history.

    We return ATR as a fraction so k*ATR is a percentage-like move. A coin
    with ATR_7d = 4% of price, at k=1.0, has a threshold of +4%. A coin with
    ATR_7d = 1% of price (e.g., BTC in calm regime), at k=1.0, threshold is
    +1%. This is the whole point: the threshold adapts to each coin.
    """
    if len(bars) < 3:
        return 0.02   # 2% floor — matches typical crypto daily ATR
    window = bars[-window_bars:] if len(bars) > window_bars else bars
    trs = []
    for i in range(1, len(window)):
        h, l, pc = window[i]["h"], window[i]["l"], window[i-1]["c"]
        tr = max(h-l, abs(h-pc), abs(l-pc))
        trs.append(tr)
    if not trs:
        return 0.02
    mean_tr = sum(trs) / len(trs)
    mean_price = sum(b["c"] for b in window) / len(window)
    atr_frac = mean_tr / mean_price if mean_price > 0 else 0.02
    # Floor at 0.3% to avoid division-blowup on ultra-stable coins. Ceiling at
    # 20% to prevent crazy-meme-coin ATR from making thresholds meaningless.
    return float(max(0.003, min(0.20, atr_frac)))

# ═══════════════════════════════════════════════════════════════════
# LABEL — did price touch entry * (1 + k * atr_frac) within horizon_bars?
# ═══════════════════════════════════════════════════════════════════
def did_touch_threshold(entry_price, future_bars, k_atr, atr_frac, horizon_bars):
    """
    Returns 1 if at any point during future_bars[:horizon_bars] the HIGH of
    any bar reaches entry_price * (1 + k_atr * atr_frac), else 0.

    Wicks count — this is the 'touched at any point' definition the user
    chose. It's the easiest label to satisfy but also the one most prone to
    volatility-detection shortcuts; vol-normalization of the threshold
    mitigates but doesn't eliminate this.
    """
    target = entry_price * (1.0 + k_atr * atr_frac)
    window = future_bars[:horizon_bars]
    for b in window:
        if b["h"] >= target:
            return 1
    return 0

# ═══════════════════════════════════════════════════════════════════
# INDICATOR PRIMITIVES — all computed from scratch, no TA-Lib dependency
# ═══════════════════════════════════════════════════════════════════
def _sma(xs, n):
    if len(xs) < n: return None
    return sum(xs[-n:]) / n

def _ema(xs, n):
    """Exponential moving average, most-recent value."""
    if len(xs) < n: return None
    alpha = 2.0 / (n + 1)
    ema = sum(xs[:n]) / n
    for x in xs[n:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema

def _std(xs):
    if len(xs) < 2: return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x-m)**2 for x in xs) / len(xs))

def _bollinger(closes, n=20, k_std=2.0):
    """Returns (mid, upper, lower, width_pct, position_in_band_[-1,1])."""
    if len(closes) < n:
        return None
    window = closes[-n:]
    mid = sum(window) / n
    sd = _std(window)
    upper, lower = mid + k_std*sd, mid - k_std*sd
    width_pct = (upper - lower) / mid if mid > 0 else 0
    # position: -1 at lower band, +1 at upper band
    last = closes[-1]
    denom = (upper - lower) / 2
    pos = (last - mid) / denom if denom > 0 else 0
    return (mid, upper, lower, width_pct, max(-2.0, min(2.0, pos)))

def _macd(closes, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram). Needs >= slow+signal bars."""
    if len(closes) < slow + signal:
        return None
    # Build EMA fast/slow series for the last (signal) bars so we can EMA the macd
    macd_series = []
    for i in range(len(closes) - signal, len(closes)):
        e_fast = _ema(closes[:i+1], fast)
        e_slow = _ema(closes[:i+1], slow)
        if e_fast is None or e_slow is None:
            return None
        macd_series.append(e_fast - e_slow)
    macd_now = macd_series[-1]
    signal_now = _ema(macd_series, signal) if len(macd_series) >= signal else (sum(macd_series)/len(macd_series))
    hist = macd_now - signal_now
    return (macd_now, signal_now, hist)

def _stochastic(bars, n=14, d_period=3):
    """Returns (%K, %D, cross_up_flag)."""
    if len(bars) < n + d_period:
        return None
    ks = []
    for i in range(len(bars) - d_period, len(bars)):
        window = bars[i-n+1:i+1] if i-n+1 >= 0 else bars[:i+1]
        hi = max(b["h"] for b in window)
        lo = min(b["l"] for b in window)
        c = bars[i]["c"]
        k = 100 * (c - lo) / (hi - lo) if hi > lo else 50.0
        ks.append(k)
    k_now = ks[-1]
    d_now = sum(ks) / len(ks)
    # cross_up: was %K <= %D 1 bar ago and is now above?
    prev_k = ks[-2] if len(ks) > 1 else k_now
    prev_d = sum(ks[:-1]) / len(ks[:-1]) if len(ks) > 1 else d_now
    cross_up = 1 if (prev_k <= prev_d and k_now > d_now) else 0
    return (k_now, d_now, cross_up)

def _williams_r(bars, n=14):
    if len(bars) < n: return None
    window = bars[-n:]
    hi = max(b["h"] for b in window)
    lo = min(b["l"] for b in window)
    c = bars[-1]["c"]
    if hi == lo: return -50.0
    return -100 * (hi - c) / (hi - lo)

def _cci(bars, n=20):
    if len(bars) < n: return None
    window = bars[-n:]
    tps = [(b["h"]+b["l"]+b["c"])/3 for b in window]
    sma = sum(tps) / n
    mad = sum(abs(tp - sma) for tp in tps) / n
    if mad == 0: return 0.0
    return (tps[-1] - sma) / (0.015 * mad)

def _rsi(bars, n=14):
    if len(bars) < n+1: return 50.0
    closes = [b["c"] for b in bars[-(n+1):]]
    gains = [max(0, closes[i]-closes[i-1]) for i in range(1, len(closes))]
    losses = [max(0, closes[i-1]-closes[i]) for i in range(1, len(closes))]
    ag = sum(gains) / n
    al = sum(losses) / n
    if al == 0: return 100.0
    rs = ag / al
    return 100 - 100/(1+rs)

def _rsi_series(bars, n=14):
    """Return the last ~10 RSI values so we can check for divergence."""
    if len(bars) < n + 10: return None
    series = []
    for i in range(n, len(bars)):
        series.append(_rsi(bars[:i+1], n))
    return series[-10:]

def _rsi_bullish_divergence(bars, n=14, window=10):
    """
    Bullish divergence: price makes lower low, RSI makes higher low.
    Returns 1 if detected in last `window` bars, 0 otherwise.
    """
    rsi_series = _rsi_series(bars, n)
    if rsi_series is None or len(bars) < window + n: return 0
    price_window = [b["l"] for b in bars[-window:]]
    # Find two local lows in price
    if len(price_window) < 5: return 0
    # Lowest and second-lowest indices, at least 3 bars apart
    sorted_idx = sorted(range(len(price_window)), key=lambda i: price_window[i])
    lo1, lo2 = sorted_idx[0], None
    for i in sorted_idx[1:]:
        if abs(i - lo1) >= 3:
            lo2 = i; break
    if lo2 is None: return 0
    # Order them
    early, late = (lo1, lo2) if lo1 < lo2 else (lo2, lo1)
    # Bullish divergence: price_late < price_early AND rsi_late > rsi_early
    if price_window[late] < price_window[early] and rsi_series[-1] > rsi_series[-(window-early)]:
        return 1
    return 0

def _obv_slope(bars, n=20):
    """OBV = cumulative volume signed by close direction. Slope normalized."""
    if len(bars) < n+1: return 0.0
    window = bars[-(n+1):]
    obv = 0.0
    obvs = [0.0]
    for i in range(1, len(window)):
        if window[i]["c"] > window[i-1]["c"]:
            obv += window[i]["v"]
        elif window[i]["c"] < window[i-1]["c"]:
            obv -= window[i]["v"]
        obvs.append(obv)
    # Slope via endpoints normalized by |max(obv)|
    denom = max(abs(max(obvs)), abs(min(obvs)), 1)
    return (obvs[-1] - obvs[0]) / denom

def _keltner_position(bars, n=20, k=1.5):
    """Position within Keltner Channels (EMA ± k*ATR). -1 at lower, +1 at upper."""
    if len(bars) < n + 1: return 0.0
    closes = [b["c"] for b in bars[-n:]]
    mid = _ema(closes, n) or (sum(closes)/len(closes))
    atr = compute_atr_fraction(bars, n) * mid
    upper, lower = mid + k*atr, mid - k*atr
    denom = (upper - lower) / 2
    if denom <= 0: return 0.0
    pos = (bars[-1]["c"] - mid) / denom
    return max(-2.0, min(2.0, pos))

# ═══════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION — takes before-scan bars + context, returns 53-dim dict
# ═══════════════════════════════════════════════════════════════════
def compute_btc_context_v2(btc_bars_before, atr_btc_frac):
    """BTC market-factor context. All values are scalars, same for every coin
    at a given scan. atr_btc_frac is BTC's own ATR for normalization."""
    if len(btc_bars_before) < 13:
        return {"btc_ret_4b":0.0,"btc_ret_12b":0.0,"btc_vol":0.0}
    closes = [b["c"] for b in btc_bars_before]
    btc_ret_4b = math.log(closes[-1]/closes[-5]) if closes[-5] > 0 else 0.0
    btc_ret_12b = math.log(closes[-1]/closes[-13]) if closes[-13] > 0 else 0.0
    # realized vol
    rets = [math.log(closes[i]/closes[i-1]) for i in range(1,len(closes)) if closes[i-1]>0]
    vol = _std(rets) * math.sqrt(BARS_PER_DAY) if len(rets) > 1 else 0
    return {"btc_ret_4b":btc_ret_4b, "btc_ret_12b":btc_ret_12b, "btc_vol":vol}

def compute_beta_to_btc(coin_bars, btc_bars, n=50):
    """Rolling beta of coin returns to BTC returns over last n bars.
    Safe to call with mismatched lengths — we align on position."""
    cb = coin_bars[-n-1:]
    bb = btc_bars[-n-1:]
    m = min(len(cb), len(bb))
    if m < 10: return 1.0
    cb, bb = cb[-m:], bb[-m:]
    cr = [math.log(cb[i]["c"]/cb[i-1]["c"]) for i in range(1,m) if cb[i-1]["c"]>0]
    br = [math.log(bb[i]["c"]/bb[i-1]["c"]) for i in range(1,m) if bb[i-1]["c"]>0]
    if len(cr) < 5 or len(br) < 5: return 1.0
    m = min(len(cr), len(br))
    cr, br = cr[-m:], br[-m:]
    var_btc = _std(br) ** 2
    if var_btc == 0: return 1.0
    mean_c, mean_b = sum(cr)/m, sum(br)/m
    cov = sum((cr[i]-mean_c)*(br[i]-mean_b) for i in range(m)) / m
    return cov / var_btc

def compute_features_v2(bars, daily_bars, current_price, open_price, scan_hour,
                         btc_bars_before=None, btc_context=None, atr_frac=None):
    """
    Compute 53-feature vector for one coin at one scan time.

    bars: intraday 15-min bars BEFORE scan time (strict), most recent last.
    daily_bars: recent daily bars for context (not used heavily in v2; we now
                compute ATR from intraday bars for consistency).
    current_price: last close before scan.
    open_price: open price of the scan-day's first bar (for intraday-ret feats).
    scan_hour: UTC hour of scan slot, in [0..23].
    btc_bars_before: BTC intraday bars up to scan time (for beta).
    btc_context: dict from compute_btc_context_v2.
    atr_frac: this coin's ATR_7d / price (precomputed for label consistency).

    Returns dict of 53 features, or None if insufficient data.
    """
    if len(bars) < MIN_BARS_FOR_FEATURES:
        return None
    if atr_frac is None:
        atr_frac = compute_atr_fraction(bars)
    atr_abs = atr_frac * current_price if current_price > 0 else 0.02

    closes = [b["c"] for b in bars]
    highs = [b["h"] for b in bars]
    lows  = [b["l"] for b in bars]
    vols  = [b["v"] for b in bars]

    # ── A. Price action, vol-normalized ──
    def _norm_ret(n):
        if len(closes) <= n or closes[-n-1] <= 0 or atr_frac <= 0: return 0.0
        r = math.log(closes[-1]/closes[-n-1])
        return r / atr_frac
    ret_1b_atr  = _norm_ret(1)
    ret_3b_atr  = _norm_ret(3)
    ret_6b_atr  = _norm_ret(6)
    ret_12b_atr = _norm_ret(12)

    # Raw percent returns (not vol-normalized) — used by threshold-based rules.
    # These are NOT in FEATURE_NAMES_V2 so they don't affect the v2 model.
    # They ARE included in the feature dict so pinned rules can reference them.
    def _raw_pct_ret(n):
        if len(closes) <= n or closes[-n-1] <= 0: return 0.0
        return (closes[-1] / closes[-n-1] - 1) * 100
    ret_6h_pct  = _raw_pct_ret(24)   # 24 × 15min = 6h
    ret_12h_pct = _raw_pct_ret(48)   # 12h
    ret_24h_pct = _raw_pct_ret(96)   # 24h (if enough bars)

    # VWAP features
    vn = sum((b["h"]+b["l"]+b["c"])/3 * b["v"] for b in bars)
    vd = sum(vols)
    vwap = vn/vd if vd > 0 else current_price
    vwap_dist_atr = (current_price - vwap) / atr_abs if atr_abs > 0 else 0
    # VWAP slope: compare vwap of first half vs last half
    half = len(bars) // 2
    if half >= 2:
        vn1 = sum((b["h"]+b["l"]+b["c"])/3 * b["v"] for b in bars[:half])
        vd1 = sum(b["v"] for b in bars[:half])
        vn2 = sum((b["h"]+b["l"]+b["c"])/3 * b["v"] for b in bars[half:])
        vd2 = sum(b["v"] for b in bars[half:])
        v1 = vn1/vd1 if vd1 > 0 else current_price
        v2 = vn2/vd2 if vd2 > 0 else current_price
        vwap_slope_atr = (v2 - v1) / atr_abs if atr_abs > 0 else 0
    else:
        vwap_slope_atr = 0

    # Trend strength: linear regression slope over last 24 bars (if available),
    # normalized by ATR (per bar)
    trend_len = min(24, len(closes))
    if trend_len >= 4:
        y = closes[-trend_len:]
        x_mean = (trend_len - 1) / 2
        y_mean = sum(y) / trend_len
        num = sum((i - x_mean) * (y[i] - y_mean) for i in range(trend_len))
        den = sum((i - x_mean)**2 for i in range(trend_len))
        slope_per_bar = num / den if den > 0 else 0
        trend_str_atr = slope_per_bar / atr_abs if atr_abs > 0 else 0
    else:
        trend_str_atr = 0

    # Momentum acceleration: 2nd derivative proxy. Compare mid-period to recent
    # momentum.
    if len(closes) >= 13:
        r_recent = math.log(closes[-1]/closes[-7]) if closes[-7] > 0 else 0
        r_earlier = math.log(closes[-7]/closes[-13]) if closes[-13] > 0 else 0
        momentum_accel = (r_recent - r_earlier) / atr_frac if atr_frac > 0 else 0
    else:
        momentum_accel = 0

    # Bollinger
    bb = _bollinger(closes, n=20, k_std=2.0)
    if bb is not None:
        _, _, _, bb_width, bb_position = bb
    else:
        bb_width, bb_position = 0, 0
    # Bollinger squeeze: is current width below 20th pct of last 100 bars?
    bb_squeeze = 0
    if len(closes) >= 100:
        widths = []
        for i in range(20, 100):
            b_hist = _bollinger(closes[-100:-100+i+1], n=20, k_std=2.0)
            if b_hist: widths.append(b_hist[3])
        if widths:
            widths.sort()
            p20 = widths[len(widths)//5]
            bb_squeeze = 1 if bb_width < p20 else 0

    # MACD
    macd = _macd(closes, fast=12, slow=26, signal=9)
    if macd is not None:
        _, _, macd_hist = macd
        # Normalize by price for cross-coin comparability
        macd_hist = macd_hist / current_price if current_price > 0 else 0
    else:
        macd_hist = 0
    # MACD cross: did hist flip from negative to positive in last 2 bars?
    macd_cross = 0
    if len(closes) >= 36:
        m1 = _macd(closes[:-1], fast=12, slow=26, signal=9)
        m2 = _macd(closes, fast=12, slow=26, signal=9)
        if m1 and m2 and m1[2] <= 0 and m2[2] > 0:
            macd_cross = 1

    # ── B. Momentum & reversal ──
    rsi_14 = _rsi(bars, n=14)
    rsi_divergence = _rsi_bullish_divergence(bars, n=14, window=10)
    stoch = _stochastic(bars, n=14, d_period=3)
    if stoch is not None:
        stoch_k, stoch_d, stoch_cross_up = stoch
    else:
        stoch_k, stoch_d, stoch_cross_up = 50, 50, 0
    williams_r = _williams_r(bars, n=14) or -50.0
    cci_20 = _cci(bars, n=20) or 0.0
    roc_6b = math.log(closes[-1]/closes[-7])*100 if len(closes) > 7 and closes[-7] > 0 else 0

    # ── C. Volume ──
    if len(vols) >= 30:
        window_v = vols[-30:]
        mean_v = sum(window_v)/len(window_v)
        sd_v = _std(window_v)
        vol_zscore = (vols[-1] - mean_v) / sd_v if sd_v > 0 else 0
    else:
        vol_zscore = 0
    vol_surge = 1 if vol_zscore > 2 else 0
    obv_slope = _obv_slope(bars, n=20)

    if len(bars) >= 12:
        vwap_above_fraction = sum(1 for b in bars[-12:] if b["c"] > vwap) / 12
    else:
        vwap_above_fraction = 0.5

    if len(vols) >= 20:
        avg20 = sum(vols[-20:])/20
        rel_volume = vols[-1]/avg20 if avg20 > 0 else 1.0
    else:
        rel_volume = 1.0

    # Volume trend: slope of volume over last 6 bars
    if len(vols) >= 6:
        y = vols[-6:]
        x_mean = 2.5
        y_mean = sum(y)/6
        num = sum((i - x_mean)*(y[i] - y_mean) for i in range(6))
        den = sum((i - x_mean)**2 for i in range(6))
        vol_trend = num/den/y_mean if den > 0 and y_mean > 0 else 0
    else:
        vol_trend = 0

    # dollar_volume_rank is computed cross-sectionally in a later pass
    dollar_volume_rank = 0.5

    # ── D. Structure & breakouts ──
    lookback_structure = min(48, len(bars))
    window = bars[-lookback_structure:]
    hi48 = max(b["h"] for b in window)
    lo48 = min(b["l"] for b in window)
    resistance_dist_atr = (hi48 - current_price) / atr_abs if atr_abs > 0 else 0
    support_dist_atr    = (current_price - lo48) / atr_abs if atr_abs > 0 else 0
    range_position = (current_price - lo48) / (hi48 - lo48) if hi48 > lo48 else 0.5

    # Opening range breakout: first 6 bars of scan day
    orb = bars[:min(6, len(bars))]
    orb_h = max(b["h"] for b in orb)
    orb_l = min(b["l"] for b in orb)
    orb_range = orb_h - orb_l
    orb_strength_atr = (current_price - orb_h) / atr_abs if atr_abs > 0 else 0
    # Note: positive = above the opening range high (bullish breakout)

    # Higher-highs count in last 5 bars
    if len(highs) >= 6:
        higher_highs_5b = sum(1 for i in range(len(highs)-5, len(highs)) if highs[i] > highs[i-1])
    else:
        higher_highs_5b = 0

    # Donchian breakout: above last-20-bar high?
    if len(highs) >= 21:
        prior_max = max(highs[-21:-1])
        donchian_breakout = 1 if current_price > prior_max else 0
    else:
        donchian_breakout = 0

    keltner_position = _keltner_position(bars, n=20, k=1.5)

    # Squeeze fire: BB squeeze state broke out? (was squeezed 3-5 bars ago AND
    # current close is outside the current BB band)
    squeeze_fire = 0
    if bb is not None and len(closes) >= 110:
        _, upper, lower, _, _ = bb
        # Check squeeze state 3 bars ago
        bb_3ago = _bollinger(closes[:-3], n=20, k_std=2.0)
        if bb_3ago:
            _, _, _, w_3ago, _ = bb_3ago
            # If was squeezed and now outside band, fire
            widths_hist = []
            for i in range(20, 100):
                b_h = _bollinger(closes[-103:-103+i+1], n=20, k_std=2.0)
                if b_h: widths_hist.append(b_h[3])
            if widths_hist:
                widths_hist.sort()
                p20 = widths_hist[len(widths_hist)//5]
                was_squeezed = w_3ago < p20
                broke_out = current_price > upper or current_price < lower
                squeeze_fire = 1 if (was_squeezed and broke_out) else 0

    # ── E. BTC & category context ──
    bc = btc_context or {"btc_ret_4b":0.0,"btc_ret_12b":0.0,"btc_vol":0.0}
    btc_ret_4b = bc["btc_ret_4b"]
    btc_ret_12b = bc["btc_ret_12b"]
    btc_vol = bc["btc_vol"]
    # ret_vs_btc_atr: coin's recent return minus BTC's, normalized by this
    # coin's ATR
    coin_ret_4b = math.log(closes[-1]/closes[-5]) if len(closes) > 5 and closes[-5] > 0 else 0
    ret_vs_btc_atr = (coin_ret_4b - btc_ret_4b) / atr_frac if atr_frac > 0 else 0
    # Coin's 3-bar momentum vs BTC's
    coin_mom = math.log(closes[-1]/closes[-4]) if len(closes) > 4 and closes[-4] > 0 else 0
    btc_mom  = math.log(btc_bars_before[-1]["c"] / btc_bars_before[-4]["c"]) \
               if btc_bars_before and len(btc_bars_before) > 4 and btc_bars_before[-4]["c"] > 0 else 0
    mom_vs_btc = coin_mom - btc_mom

    # Beta (rolling)
    beta_to_btc = 1.0
    if btc_bars_before:
        beta_to_btc = compute_beta_to_btc(bars, btc_bars_before, n=50)

    # Category-relative placeholders (filled cross-sectionally)
    cat_breadth = 0.5
    ret_vs_cat_atr = 0.0
    cat_strongest = 0
    cat_weakest = 0

    # ── F. Cross-sectional rank placeholders ──
    rank_momentum = 0.5
    rank_vol_zscore = 0.5
    rank_bb_position = 0.5
    rank_rsi = 0.5
    rank_range_position = 0.5

    # ── G. Time encoding ──
    scan_hour_sin = math.sin(2 * math.pi * scan_hour / 24)
    scan_hour_cos = math.cos(2 * math.pi * scan_hour / 24)

    return {
        # A
        "ret_1b_atr":ret_1b_atr,"ret_3b_atr":ret_3b_atr,"ret_6b_atr":ret_6b_atr,"ret_12b_atr":ret_12b_atr,
        "vwap_dist_atr":vwap_dist_atr,"vwap_slope_atr":vwap_slope_atr,"trend_str_atr":trend_str_atr,
        "momentum_accel":momentum_accel,
        "bb_position":bb_position,"bb_width":bb_width,"bb_squeeze":bb_squeeze,
        "macd_hist":macd_hist,"macd_cross":macd_cross,
        # B
        "rsi_14":rsi_14,"rsi_divergence":rsi_divergence,
        "stoch_k":stoch_k,"stoch_d":stoch_d,"stoch_cross_up":stoch_cross_up,
        "williams_r":williams_r,"cci_20":cci_20,"roc_6b":roc_6b,
        # C
        "vol_zscore":vol_zscore,"vol_surge":vol_surge,"obv_slope":obv_slope,
        "vwap_above_fraction":vwap_above_fraction,"rel_volume":rel_volume,"vol_trend":vol_trend,
        "dollar_volume_rank":dollar_volume_rank,
        # D
        "resistance_dist_atr":resistance_dist_atr,"support_dist_atr":support_dist_atr,
        "range_position":range_position,"orb_strength_atr":orb_strength_atr,
        "higher_highs_5b":higher_highs_5b,"donchian_breakout":donchian_breakout,
        "keltner_position":keltner_position,"squeeze_fire":squeeze_fire,
        # E
        "btc_ret_4b":btc_ret_4b,"btc_ret_12b":btc_ret_12b,"btc_vol":btc_vol,
        "ret_vs_btc_atr":ret_vs_btc_atr,"mom_vs_btc":mom_vs_btc,"beta_to_btc":beta_to_btc,
        "cat_breadth":cat_breadth,"ret_vs_cat_atr":ret_vs_cat_atr,
        "cat_strongest":cat_strongest,"cat_weakest":cat_weakest,
        # F
        "rank_momentum":rank_momentum,"rank_vol_zscore":rank_vol_zscore,
        "rank_bb_position":rank_bb_position,"rank_rsi":rank_rsi,
        "rank_range_position":rank_range_position,
        # G
        "scan_hour_sin":scan_hour_sin,"scan_hour_cos":scan_hour_cos,
        # H. Raw %-return features for threshold-based rule mining (NOT in FEATURE_NAMES_V2,
        # so v2 classifier won't see them; only pinned rules that explicitly reference
        # them will use them).
        "ret_6h_pct":ret_6h_pct,"ret_12h_pct":ret_12h_pct,"ret_24h_pct":ret_24h_pct,
    }

def add_cross_sectional_features_v2(features_list, cat_list, dollar_vol_list):
    """Fill the rank_* and category-relative features using the cross-section of
    coins at this scan time. Mutates features_list in place."""
    n = len(features_list)
    if n < 2: return features_list

    def pct_rank(vals):
        arr = np.array(vals)
        o = arr.argsort().argsort()
        return (o / (n-1)).astype(float)

    ranks = {
        "rank_momentum":       pct_rank([f["ret_6b_atr"]      for f in features_list]),
        "rank_vol_zscore":     pct_rank([f["vol_zscore"]      for f in features_list]),
        "rank_bb_position":    pct_rank([f["bb_position"]     for f in features_list]),
        "rank_rsi":            pct_rank([50 - abs(f["rsi_14"] - 55) for f in features_list]),
        "rank_range_position": pct_rank([f["range_position"]  for f in features_list]),
    }
    dv_ranks = pct_rank(dollar_vol_list)

    # Category aggregates
    cat_indices = defaultdict(list)
    for i, c in enumerate(cat_list):
        cat_indices[c].append(i)

    for i in range(n):
        for k, v in ranks.items():
            features_list[i][k] = float(v[i])
        features_list[i]["dollar_volume_rank"] = float(dv_ranks[i])

        c = cat_list[i]
        peers = cat_indices[c]
        if len(peers) >= 2:
            peer_rets = [features_list[j]["ret_6b_atr"] for j in peers if j != i]
            if peer_rets:
                features_list[i]["ret_vs_cat_atr"] = features_list[i]["ret_6b_atr"] - np.mean(peer_rets)
            positive = sum(1 for j in peers if features_list[j]["ret_6b_atr"] > 0)
            features_list[i]["cat_breadth"] = positive / len(peers)
            # strongest/weakest in category
            cat_rets = [(j, features_list[j]["ret_6b_atr"]) for j in peers]
            cat_rets.sort(key=lambda x: x[1])
            features_list[i]["cat_weakest"]   = 1 if cat_rets[0][0] == i else 0
            features_list[i]["cat_strongest"] = 1 if cat_rets[-1][0] == i else 0

    return features_list

def feat_to_arr_v2(f):
    return np.array([f.get(n, 0) for n in FEATURE_NAMES_V2])

# ═══════════════════════════════════════════════════════════════════
# TRAINING — single (k_atr, horizon_hours) cell
# ═══════════════════════════════════════════════════════════════════
def run_train_cell_v2(k_atr, horizon_hours, intraday_bars, daily_bars,
                       products, categories, benchmark, model_dir,
                       scan_hours=SCAN_HOURS, progress_cb=None):
    """
    Train a single (k_atr, horizon_hours) model.

    intraday_bars: dict of product -> list of 15-min bars, ascending time
    daily_bars: dict of product -> list of daily bars
    products: list of product codes to include
    categories: dict of product -> category string
    benchmark: product code for market-factor features (e.g., "BTC-USD")
    model_dir: pathlib.Path where to store model + metadata
    scan_hours: list of UTC hours that count as "scan times" for label generation
    progress_cb(pct, msg): optional callback for progress updates

    Saves:
      {model_dir}/v2_k{k_atr}_h{horizon_hours}.txt         — LightGBM model
      {model_dir}/v2_k{k_atr}_h{horizon_hours}_cal.pkl     — isotonic calibrator
      {model_dir}/v2_k{k_atr}_h{horizon_hours}_meta.json   — metrics/importance

    Returns meta dict.
    """
    def prog(p, m):
        if progress_cb: progress_cb(p, m)
        log.info(f"[v2 train k={k_atr} h={horizon_hours}] {p}% — {m}")

    horizon_bars = int(horizon_hours * BARS_PER_HOUR)

    # ── Group bars by product+date ──
    prog(5, "Grouping bars by date...")
    by_td = defaultdict(lambda: defaultdict(list))
    for product in list(products) + [benchmark]:
        for b in intraday_bars.get(product, []):
            by_td[product][b["t"][:10]].append(b)

    all_dates = sorted(set(d for t in by_td for d in by_td[t]))
    if len(all_dates) < MIN_HISTORY_DAYS:
        raise ValueError(f"Only {len(all_dates)} days of data, need >= {MIN_HISTORY_DAYS}")

    prog(10, f"Building {len(all_dates)} × {len(scan_hours)} scan points...")
    # For each coin we precompute a flat sorted list of ALL its intraday bars
    # (across all days) for fast slicing around each scan point.
    flat_by_product = {}
    for product in list(products) + [benchmark]:
        bars = intraday_bars.get(product, [])
        flat_by_product[product] = sorted(bars, key=lambda b: b["t"])

    # Precompute BTC bar index for fast slicing
    btc_flat = flat_by_product.get(benchmark, [])
    if len(btc_flat) < 50:
        raise ValueError(f"Benchmark {benchmark} has only {len(btc_flat)} bars")

    # ── Build feature rows ──
    rows = []
    n_dates = len(all_dates)
    for di, date in enumerate(all_dates):
        for scan_hour in scan_hours:
            scan_min = scan_hour * 60

            # ── BTC context up to scan time on this date ──
            # We need the BTC bar timestamp string prefix to be <= date + scan_min
            # Simpler: find all BTC bars with date==this and minute<scan_min
            btc_day = by_td[benchmark].get(date, [])
            btc_before_today = []
            for b in btc_day:
                try:
                    dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(UTC)
                    if dt.hour * 60 + dt.minute < scan_min:
                        btc_before_today.append(b)
                except:
                    continue
            # Also include earlier days' full bars for longer context (beta, ATR)
            btc_earlier = []
            for d2 in all_dates:
                if d2 >= date: break
                btc_earlier.extend(by_td[benchmark].get(d2, []))
            btc_history = btc_earlier + btc_before_today
            if len(btc_history) < 50:
                continue

            btc_atr = compute_atr_fraction(btc_history)
            btc_ctx = compute_btc_context_v2(btc_history[-48:], btc_atr)

            # ── Per-coin loop ──
            date_feats, date_meta, date_cats, date_dv = [], [], [], []

            for product in products:
                day_bars = by_td[product].get(date, [])
                if len(day_bars) < 4: continue

                # Split day bars into before/after scan minute
                before_today, after_today = [], []
                for b in day_bars:
                    try:
                        dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(UTC)
                        bm = dt.hour * 60 + dt.minute
                        if bm < scan_min: before_today.append(b)
                        else: after_today.append(b)
                    except:
                        continue

                # Get historical bars (earlier days) for lookback
                earlier = []
                for d2 in all_dates:
                    if d2 >= date: break
                    earlier.extend(by_td[product].get(d2, []))
                history = earlier + before_today
                if len(history) < MIN_BARS_FOR_FEATURES: continue

                # Use last FEATURE_LOOKBACK_BARS for feature computation
                feat_bars = history[-FEATURE_LOOKBACK_BARS:]

                # ATR is computed from the longer history for stability
                atr_frac = compute_atr_fraction(history)

                # Entry price = open of the first bar AFTER scan time (next day's
                # midnight bar if scan is 20:00). Need at least horizon_bars after.
                # Collect after-bars across this day and subsequent days.
                after_bars = list(after_today)
                if len(after_bars) < horizon_bars + 1:
                    # Pull from subsequent days
                    try:
                        idx_today = all_dates.index(date)
                    except:
                        idx_today = -1
                    if idx_today >= 0:
                        for di2 in range(idx_today + 1, len(all_dates)):
                            if len(after_bars) >= horizon_bars + 1: break
                            after_bars.extend(by_td[product].get(all_dates[di2], []))
                if len(after_bars) < horizon_bars + 1: continue

                entry_price = after_bars[0]["o"]
                current_price = feat_bars[-1]["c"]
                open_price = day_bars[0]["o"]

                feat = compute_features_v2(
                    feat_bars, daily_bars.get(product, []),
                    current_price, open_price, scan_hour,
                    btc_bars_before=btc_history[-50:], btc_context=btc_ctx,
                    atr_frac=atr_frac,
                )
                if feat is None: continue

                # Label: did price touch +k*ATR within horizon_bars?
                label = did_touch_threshold(
                    entry_price, after_bars[1:], k_atr, atr_frac, horizon_bars,
                )

                date_feats.append(feat)
                date_meta.append({
                    "product":product, "date":date, "scan_hour":scan_hour,
                    "label":label, "entry_price":entry_price, "atr_frac":atr_frac,
                })
                date_cats.append(categories.get(product, "?"))
                date_dv.append(sum(b["v"]*b["c"] for b in feat_bars[-20:]))

            if len(date_feats) >= 3:
                add_cross_sectional_features_v2(date_feats, date_cats, date_dv)
                for j in range(len(date_feats)):
                    date_feats[j]["label"] = date_meta[j]["label"]
                    date_feats[j]["date"] = date_meta[j]["date"]
                    date_feats[j]["scan_hour"] = date_meta[j]["scan_hour"]
                    date_feats[j]["product"] = date_meta[j]["product"]
                    rows.append(date_feats[j])

        if (di + 1) % 20 == 0:
            prog(10 + int((di / n_dates) * 60),
                 f"Processed {di+1}/{n_dates} days ({len(rows)} rows so far)...")

    if len(rows) < 500:
        raise ValueError(f"Only {len(rows)} training rows, too few")

    prog(75, f"Building DataFrame from {len(rows)} rows...")
    df = pd.DataFrame(rows)
    dates_sorted = sorted(df["date"].unique())

    # ── Purged time-series split ──
    # Train: first 60% of dates
    # Embargo: next horizon_hours / 24 days (fraction of a day)
    # Val:   next 20%
    # Test:  final 20%
    # We use the time-ordered split (NOT random) because labels depend on future
    # H hours. Random split would leak info across the split boundary.
    n_days = len(dates_sorted)
    embargo_days = max(1, int(math.ceil(horizon_hours / 24)))
    train_end = int(n_days * 0.6)
    val_start = train_end + embargo_days
    val_end   = val_start + int(n_days * 0.2)
    test_start = val_end + embargo_days

    if test_start >= n_days - 2:
        # Not enough data for embargo; reduce train share
        train_end = max(1, int(n_days * 0.55))
        val_start = train_end + embargo_days
        val_end = val_start + max(1, int(n_days * 0.2))
        test_start = val_end + embargo_days

    train_dates = set(dates_sorted[:train_end])
    val_dates   = set(dates_sorted[val_start:val_end])
    test_dates  = set(dates_sorted[test_start:])
    log.info(f"Split: train {len(train_dates)}d / val {len(val_dates)}d / "
             f"test {len(test_dates)}d (embargo {embargo_days}d)")

    train_df = df[df["date"].isin(train_dates)]
    val_df   = df[df["date"].isin(val_dates)]
    test_df  = df[df["date"].isin(test_dates)]

    if len(train_df) < 200 or len(val_df) < 50 or len(test_df) < 50:
        raise ValueError(f"Splits too small: train {len(train_df)} val {len(val_df)} test {len(test_df)}")

    X_tr = train_df[FEATURE_NAMES_V2].values; y_tr = train_df["label"].values
    X_va = val_df[FEATURE_NAMES_V2].values;   y_va = val_df["label"].values
    X_te = test_df[FEATURE_NAMES_V2].values;  y_te = test_df["label"].values

    base_tr = float(y_tr.mean())
    base_va = float(y_va.mean())
    base_te = float(y_te.mean())
    log.info(f"Base rates — train {base_tr:.3f}, val {base_va:.3f}, test {base_te:.3f}")

    prog(85, "Training LightGBM...")
    ts = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES_V2)
    vs = lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES_V2, reference=ts)
    params = {
        "objective":"binary","metric":"binary_logloss",
        "boosting_type":"gbdt","num_leaves":15,"learning_rate":0.03,
        "feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":5,
        "min_child_samples":50,"lambda_l2":1.0,"verbose":-1,
    }
    model = lgb.train(params, ts, num_boost_round=800, valid_sets=[vs],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    # Calibrate on val, evaluate on test
    raw_va = model.predict(X_va)
    raw_te = model.predict(X_te)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    cal.fit(raw_va, y_va)
    cal_te = cal.predict(raw_te)

    auc_te = roc_auc_score(y_te, raw_te) if len(set(y_te)) > 1 else 0.5

    # Precision at various thresholds (on TEST set — the honest one)
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    precision_at = {}
    for t in thresholds:
        mask = cal_te >= t
        n_pos = int(mask.sum())
        precision = float(y_te[mask].mean()) if n_pos > 0 else 0.0
        # Predictions per day at this threshold
        tmp = test_df.copy(); tmp["p"] = cal_te
        per_day = tmp[tmp["p"] >= t].groupby("date").size().mean() if n_pos > 0 else 0.0
        precision_at[f"{t:.2f}"] = {
            "precision": round(precision, 4),
            "n_predictions": n_pos,
            "coverage_pct": round(100 * n_pos / len(y_te), 2),
            "avg_per_day": round(float(per_day), 2) if n_pos > 0 else 0.0,
        }

    # Also: precision at TOP K% of predictions
    top_k_precision = {}
    for pct in [0.1, 0.5, 1.0, 5.0]:
        k_count = max(1, int(len(cal_te) * pct / 100))
        topk_idx = np.argsort(-cal_te)[:k_count]
        top_k_precision[f"top_{pct}_pct"] = {
            "precision": round(float(y_te[topk_idx].mean()), 4),
            "n": k_count,
        }

    # Feature importance
    imp = dict(zip(FEATURE_NAMES_V2, model.feature_importance("gain").tolist()))
    ti = sum(imp.values()) or 1
    imp_norm = {k: round(v/ti, 4) for k, v in imp.items()}

    # Save artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    cell_key = f"k{k_atr:g}_h{horizon_hours}"
    model_path = model_dir / f"v2_{cell_key}.txt"
    cal_path   = model_dir / f"v2_{cell_key}_cal.pkl"
    meta_path  = model_dir / f"v2_{cell_key}_meta.json"
    model.save_model(str(model_path))
    cal_path.write_bytes(pickle.dumps(cal))

    meta = {
        "k_atr": float(k_atr),
        "horizon_hours": int(horizon_hours),
        "horizon_bars": horizon_bars,
        "trained_at": datetime.now(UTC).isoformat(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_dates": len(train_dates),
        "val_dates": len(val_dates),
        "test_dates": len(test_dates),
        "embargo_days": embargo_days,
        "base_rate_train": round(base_tr, 4),
        "base_rate_val": round(base_va, 4),
        "base_rate_test": round(base_te, 4),
        "auc_test": round(float(auc_te), 4),
        "precision_at_threshold": precision_at,
        "top_k_precision": top_k_precision,
        "best_iteration": int(model.best_iteration or 0),
        "feature_importance": imp_norm,
        "model_path": str(model_path.name),
        "cal_path": str(cal_path.name),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    prog(100, "Done.")
    log.info(f"[v2 k={k_atr} h={horizon_hours}] AUC {auc_te:.3f}, "
             f"base_rate_test {base_te:.3f}, "
             f"prec@0.75 {precision_at['0.75']['precision']} "
             f"(n={precision_at['0.75']['n_predictions']})")
    return meta

# ═══════════════════════════════════════════════════════════════════
# ROW BUILDER for rule mining — features + multi-horizon ABSOLUTE% labels
# ═══════════════════════════════════════════════════════════════════

def did_touch_absolute(entry_price, future_bars, pct_threshold, horizon_bars):
    """
    Returns 1 if any bar's HIGH in future_bars[:horizon_bars] reaches
    entry_price * (1 + pct_threshold). Wicks count.

    Differs from did_touch_threshold: uses a fixed absolute percentage
    (e.g., 0.02 = +2%), NOT k*ATR. This is what the rule-mining framework
    uses because the user prefers a concrete, tradeable target.
    """
    target = entry_price * (1.0 + pct_threshold)
    window = future_bars[:horizon_bars]
    for b in window:
        if b["h"] >= target:
            return 1
    return 0


def build_rows_for_rule_mining(intraday_bars, daily_bars, products, categories,
                                benchmark, scan_hours, horizon_hours_list,
                                pct_threshold, progress_cb=None):
    """
    Build a DataFrame-ready list of rows for rule mining.

    Each row = one (date, scan_hour, coin) sample, containing:
      • all 53 features from compute_features_v2()
      • 'label_{H}h' columns: binary, 1 if coin hit +pct_threshold within H hours
      • 'date', 'scan_hour', 'product' for splitting and display
      • 'is_train', 'is_val', 'is_test': purged time-series split indicators
        (embargo = ceil(max_horizon / 24) days)

    Returns: (rows, split_info) where split_info has train/val/test date sets.
    """
    def prog(p, m):
        if progress_cb: progress_cb(p, m)
        log.info(f"[row build] {p}% — {m}")

    max_horizon_hours = max(horizon_hours_list)
    max_horizon_bars = int(max_horizon_hours * BARS_PER_HOUR)
    horizon_bars_map = {H: int(H * BARS_PER_HOUR) for H in horizon_hours_list}

    prog(5, "Grouping bars by date...")
    by_td = defaultdict(lambda: defaultdict(list))
    for product in list(products) + [benchmark]:
        for b in intraday_bars.get(product, []):
            by_td[product][b["t"][:10]].append(b)

    all_dates = sorted(set(d for t in by_td for d in by_td[t]))
    if len(all_dates) < MIN_HISTORY_DAYS:
        raise ValueError(f"Only {len(all_dates)} days of data, need >= {MIN_HISTORY_DAYS}")

    prog(10, f"Building rows across {len(all_dates)} dates × {len(scan_hours)} slots...")
    rows = []
    n_dates = len(all_dates)

    for di, date in enumerate(all_dates):
        for scan_hour in scan_hours:
            scan_min = scan_hour * 60

            # BTC context up to scan time on this date
            btc_day = by_td[benchmark].get(date, [])
            btc_before_today = []
            for b in btc_day:
                try:
                    dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(UTC)
                    if dt.hour * 60 + dt.minute < scan_min:
                        btc_before_today.append(b)
                except:
                    continue
            btc_earlier = []
            for d2 in all_dates:
                if d2 >= date: break
                btc_earlier.extend(by_td[benchmark].get(d2, []))
            btc_history = btc_earlier + btc_before_today
            if len(btc_history) < 50: continue
            btc_atr = compute_atr_fraction(btc_history)
            btc_ctx = compute_btc_context_v2(btc_history[-48:], btc_atr)

            date_feats, date_meta, date_cats, date_dv = [], [], [], []

            for product in products:
                day_bars = by_td[product].get(date, [])
                if len(day_bars) < 4: continue

                before_today, after_today = [], []
                for b in day_bars:
                    try:
                        dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(UTC)
                        bm = dt.hour * 60 + dt.minute
                        if bm < scan_min: before_today.append(b)
                        else: after_today.append(b)
                    except: continue

                earlier = []
                for d2 in all_dates:
                    if d2 >= date: break
                    earlier.extend(by_td[product].get(d2, []))
                history = earlier + before_today
                if len(history) < MIN_BARS_FOR_FEATURES: continue

                feat_bars = history[-FEATURE_LOOKBACK_BARS:]
                atr_frac = compute_atr_fraction(history)

                # Need enough future bars to evaluate the LONGEST horizon label
                after_bars = list(after_today)
                if len(after_bars) < max_horizon_bars + 1:
                    try:
                        idx_today = all_dates.index(date)
                    except:
                        idx_today = -1
                    if idx_today >= 0:
                        for di2 in range(idx_today + 1, len(all_dates)):
                            if len(after_bars) >= max_horizon_bars + 1: break
                            after_bars.extend(by_td[product].get(all_dates[di2], []))
                if len(after_bars) < max_horizon_bars + 1: continue

                entry_price = after_bars[0]["o"]
                current_price = feat_bars[-1]["c"]
                open_price = day_bars[0]["o"]

                feat = compute_features_v2(
                    feat_bars, daily_bars.get(product, []),
                    current_price, open_price, scan_hour,
                    btc_bars_before=btc_history[-50:], btc_context=btc_ctx,
                    atr_frac=atr_frac,
                )
                if feat is None: continue

                # Compute labels for each horizon
                for H in horizon_hours_list:
                    feat[f"label_{H}h"] = did_touch_absolute(
                        entry_price, after_bars[1:], pct_threshold, horizon_bars_map[H])

                date_feats.append(feat)
                date_meta.append({"product": product, "date": date, "scan_hour": scan_hour,
                                  "entry_price": entry_price, "atr_frac": atr_frac})
                date_cats.append(categories.get(product, "?"))
                date_dv.append(sum(b["v"] * b["c"] for b in feat_bars[-20:]))

            # Require at least 3 coins per date for cross-sectional features
            # to be meaningful. In production (~130 coins) this easily clears.
            if len(date_feats) >= 3:
                add_cross_sectional_features_v2(date_feats, date_cats, date_dv)
                for j in range(len(date_feats)):
                    for mk, mv in date_meta[j].items():
                        date_feats[j][mk] = mv
                    rows.append(date_feats[j])

        if (di + 1) % 20 == 0:
            prog(10 + int((di / n_dates) * 60),
                 f"Processed {di+1}/{n_dates} dates ({len(rows)} rows)...")

    if len(rows) < 500:
        raise ValueError(f"Only {len(rows)} rows, need >= 500")

    # Purged time-series split using MAX horizon for embargo
    prog(80, "Computing train/val/test split...")
    dates_sorted = sorted(set(r["date"] for r in rows))
    n_days = len(dates_sorted)
    embargo_days = max(1, int(math.ceil(max_horizon_hours / 24)))
    train_end = int(n_days * 0.6)
    val_start = train_end + embargo_days
    val_end = val_start + int(n_days * 0.2)
    test_start = val_end + embargo_days
    if test_start >= n_days - 2:
        train_end = max(1, int(n_days * 0.55))
        val_start = train_end + embargo_days
        val_end = val_start + max(1, int(n_days * 0.2))
        test_start = val_end + embargo_days

    train_dates = set(dates_sorted[:train_end])
    val_dates = set(dates_sorted[val_start:val_end])
    test_dates = set(dates_sorted[test_start:])

    for r in rows:
        r["is_train"] = r["date"] in train_dates
        r["is_val"] = r["date"] in val_dates
        r["is_test"] = r["date"] in test_dates

    prog(100, f"Built {len(rows)} rows. "
              f"Split: train {len(train_dates)}d / val {len(val_dates)}d / test {len(test_dates)}d "
              f"(embargo {embargo_days}d)")

    split_info = {
        "n_train": sum(1 for r in rows if r["is_train"]),
        "n_val": sum(1 for r in rows if r["is_val"]),
        "n_test": sum(1 for r in rows if r["is_test"]),
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
        "embargo_days": embargo_days,
        "base_rates": {
            f"label_{H}h": {
                "train": sum(r[f"label_{H}h"] for r in rows if r["is_train"]) / max(1, sum(1 for r in rows if r["is_train"])),
                "val":   sum(r[f"label_{H}h"] for r in rows if r["is_val"])   / max(1, sum(1 for r in rows if r["is_val"])),
                "test":  sum(r[f"label_{H}h"] for r in rows if r["is_test"])  / max(1, sum(1 for r in rows if r["is_test"])),
            }
            for H in horizon_hours_list
        },
    }

    return rows, split_info
