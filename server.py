import os, json, time, math, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import httpx
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("scanner")

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Coinbase port
# ═══════════════════════════════════════════════════════════════════
# Coinbase Exchange public market-data endpoints require no auth, but the
# Advanced Trade endpoints do. We only read public candles & stats here, so
# credentials are optional and kept purely so users can swap in an API proxy
# or higher-rate plan if they want to. Absent creds → we still run.
CB_API_URL    = os.environ.get("COINBASE_API_URL", "https://api.exchange.coinbase.com")
CB_KEY        = os.environ.get("COINBASE_API_KEY", "")
CB_SECRET     = os.environ.get("COINBASE_API_SECRET", "")
DATA_DIR      = Path("/data") if Path("/data").exists() else Path(__file__).parent / ".data"
MODEL_DIR     = DATA_DIR / "models"
OUTCOME_DIR   = DATA_DIR / "outcomes"
SCAN_DIR      = DATA_DIR / "scans"
PORT          = int(os.environ.get("PORT", 10000))

# Crypto is 24/7 — we use UTC, not ET. Scans run every 4 hours at :05.
# Each "scan slot" is a UTC hour; the model learns whether the *next 4 hours*
# hit TP before SL. This mirrors the R2K version's intraday horizon structure.
UTC = ZoneInfo("UTC")
SCAN_HOURS = [0, 4, 8, 12, 16, 20]   # 6 slots / day, every 4h UTC
HORIZON_HOURS = 4                    # prediction window
TP_PCT = 0.020   # +2.0% take profit
SL_PCT = 0.010   # -1.0% stop loss (2:1 reward:risk, 33% break-even)
                 # Chosen via sweep: best combination of edge (+8.56%) and PnL
                 # (+0.019%) across all 5 UTC slots, with a reasonable 33%
                 # break-even bar that the model can actually clear.

# Candle granularity: 15min (900s). 4h horizon = 16 bars of lookahead max.
# Feature lookback: ~24 bars (6 hours) before the scan point.
CANDLE_GRANULARITY = 900             # seconds (15 min)
BARS_PER_HOUR = 3600 // CANDLE_GRANULARITY  # 4
HORIZON_BARS = HORIZON_HOURS * BARS_PER_HOUR   # 16
FEATURE_LOOKBACK_BARS = 24           # 6 hours of 15m bars before scan
DAILY_LOOKBACK = 10                  # daily bars for ATR / ADV

# Universe: ~140 liquid Coinbase-tradeable spot pairs quoted in USD.
# Grouped by crypto "sector" (category) — the analog of GICS sectors.
# Stablecoins are excluded (no edge in a $1 peg). Names where Coinbase returns
# insufficient history will be skipped at scan time.
PRODUCTS = [
    # Layer 1 / Majors
    "BTC-USD","ETH-USD","SOL-USD","AVAX-USD","ADA-USD","DOT-USD","ATOM-USD",
    "NEAR-USD","ALGO-USD","XTZ-USD","EGLD-USD","HBAR-USD","ICP-USD","FLOW-USD",
    "KAVA-USD","MINA-USD","ROSE-USD","SUI-USD","APT-USD","SEI-USD","TIA-USD",
    "INJ-USD","KAS-USD","TON-USD","FTM-USD","TRX-USD","XLM-USD","XRP-USD",
    # Layer 2 / Scaling
    "MATIC-USD","ARB-USD","OP-USD","IMX-USD","STRK-USD","MANTA-USD",
    # DeFi
    "UNI-USD","AAVE-USD","MKR-USD","COMP-USD","SNX-USD","CRV-USD","LDO-USD",
    "SUSHI-USD","BAL-USD","YFI-USD","1INCH-USD","DYDX-USD","GMX-USD","PENDLE-USD",
    "RPL-USD","FXS-USD","ENA-USD","ONDO-USD",
    # Meme
    "DOGE-USD","SHIB-USD","PEPE-USD","WIF-USD","BONK-USD","FLOKI-USD","MOG-USD",
    "BRETT-USD","MEW-USD","POPCAT-USD",
    # AI
    "FET-USD","RNDR-USD","TAO-USD","GRT-USD","AIOZ-USD","AKT-USD","OCEAN-USD",
    "NMR-USD","ARKM-USD","IO-USD","WLD-USD",
    # Gaming / Metaverse
    "AXS-USD","SAND-USD","MANA-USD","APE-USD","GALA-USD","CHZ-USD","IMX-USD",
    "ENJ-USD","ILV-USD","BEAM-USD","PRIME-USD","PIXEL-USD","GODS-USD",
    # Infrastructure / Oracles
    "LINK-USD","BAND-USD","API3-USD","TRB-USD","PYTH-USD","FIL-USD","AR-USD",
    "STORJ-USD","ANKR-USD","POKT-USD",
    # Privacy
    "XMR-USD","ZEC-USD","DASH-USD",
    # Exchange / CEX
    "BNB-USD","CRO-USD","LEO-USD","OKB-USD","KCS-USD",
    # RWA / Real World Assets
    "ONDO-USD","MKR-USD","PENDLE-USD","CFG-USD","TRAC-USD",
    # Other majors / legacy
    "LTC-USD","BCH-USD","ETC-USD","ZEC-USD","EOS-USD","IOTA-USD","NEO-USD",
    # Newer / rotational
    "JUP-USD","JTO-USD","W-USD","ZETA-USD","DYM-USD","ALT-USD","PORTAL-USD",
    "METIS-USD","MAGIC-USD","BLUR-USD","AUDIO-USD","MASK-USD","ENS-USD",
    "LPT-USD","SKL-USD","CTSI-USD","1INCH-USD","LRC-USD","LOOM-USD"
]
# Dedupe while preserving order
seen = set()
PRODUCTS = [p for p in PRODUCTS if not (p in seen or seen.add(p))]

# Approximate category map. Pairs not in the dict fall back to "?" and still work.
CATEGORIES = {
    # L1 / Majors
    "BTC-USD":"L1","ETH-USD":"L1","SOL-USD":"L1","AVAX-USD":"L1","ADA-USD":"L1",
    "DOT-USD":"L1","ATOM-USD":"L1","NEAR-USD":"L1","ALGO-USD":"L1","XTZ-USD":"L1",
    "EGLD-USD":"L1","HBAR-USD":"L1","ICP-USD":"L1","FLOW-USD":"L1","KAVA-USD":"L1",
    "MINA-USD":"L1","ROSE-USD":"L1","SUI-USD":"L1","APT-USD":"L1","SEI-USD":"L1",
    "TIA-USD":"L1","INJ-USD":"L1","KAS-USD":"L1","TON-USD":"L1","FTM-USD":"L1",
    "TRX-USD":"L1","XLM-USD":"L1","XRP-USD":"L1",
    # L2
    "MATIC-USD":"L2","ARB-USD":"L2","OP-USD":"L2","IMX-USD":"L2","STRK-USD":"L2",
    "MANTA-USD":"L2",
    # DeFi
    "UNI-USD":"DeFi","AAVE-USD":"DeFi","MKR-USD":"DeFi","COMP-USD":"DeFi",
    "SNX-USD":"DeFi","CRV-USD":"DeFi","LDO-USD":"DeFi","SUSHI-USD":"DeFi",
    "BAL-USD":"DeFi","YFI-USD":"DeFi","1INCH-USD":"DeFi","DYDX-USD":"DeFi",
    "GMX-USD":"DeFi","PENDLE-USD":"DeFi","RPL-USD":"DeFi","FXS-USD":"DeFi",
    "ENA-USD":"DeFi","ONDO-USD":"DeFi","CFG-USD":"DeFi",
    # Meme
    "DOGE-USD":"Meme","SHIB-USD":"Meme","PEPE-USD":"Meme","WIF-USD":"Meme",
    "BONK-USD":"Meme","FLOKI-USD":"Meme","MOG-USD":"Meme","BRETT-USD":"Meme",
    "MEW-USD":"Meme","POPCAT-USD":"Meme",
    # AI
    "FET-USD":"AI","RNDR-USD":"AI","TAO-USD":"AI","GRT-USD":"AI","AIOZ-USD":"AI",
    "AKT-USD":"AI","OCEAN-USD":"AI","NMR-USD":"AI","ARKM-USD":"AI","IO-USD":"AI",
    "WLD-USD":"AI",
    # Gaming
    "AXS-USD":"Gaming","SAND-USD":"Gaming","MANA-USD":"Gaming","APE-USD":"Gaming",
    "GALA-USD":"Gaming","CHZ-USD":"Gaming","ENJ-USD":"Gaming","ILV-USD":"Gaming",
    "BEAM-USD":"Gaming","PRIME-USD":"Gaming","PIXEL-USD":"Gaming","GODS-USD":"Gaming",
    "MAGIC-USD":"Gaming",
    # Infra / Oracles
    "LINK-USD":"Infra","BAND-USD":"Infra","API3-USD":"Infra","TRB-USD":"Infra",
    "PYTH-USD":"Infra","FIL-USD":"Infra","AR-USD":"Infra","STORJ-USD":"Infra",
    "ANKR-USD":"Infra","POKT-USD":"Infra","TRAC-USD":"Infra",
    # Privacy
    "XMR-USD":"Privacy","ZEC-USD":"Privacy","DASH-USD":"Privacy",
    # CEX
    "BNB-USD":"CEX","CRO-USD":"CEX","LEO-USD":"CEX","OKB-USD":"CEX","KCS-USD":"CEX",
    # Legacy
    "LTC-USD":"Legacy","BCH-USD":"Legacy","ETC-USD":"Legacy","EOS-USD":"Legacy",
    "IOTA-USD":"Legacy","NEO-USD":"Legacy",
    # Rotational / new
    "JUP-USD":"New","JTO-USD":"New","W-USD":"New","ZETA-USD":"New","DYM-USD":"New",
    "ALT-USD":"New","PORTAL-USD":"New","METIS-USD":"New","BLUR-USD":"DeFi",
    "AUDIO-USD":"Infra","MASK-USD":"DeFi","ENS-USD":"Infra","LPT-USD":"Infra",
    "SKL-USD":"Infra","CTSI-USD":"Infra","LRC-USD":"L2","LOOM-USD":"AI"
}

# Benchmark product: BTC plays the role SPY played in the R2K version —
# the "market factor" that every other coin is measured against.
BENCHMARK = "BTC-USD"

FEATURE_NAMES = [
    # Bar-data features (11) — same shape as R2K minus hours_left (24/7 market)
    "momentum","ret_from_open","rel_volume","vwap_dist","vwap_slope",
    "orb_strength","atr_reach","realized_vol","trend_str","rsi","range_expansion",
    # Benchmark-relative features (5) — BTC plays SPY's role
    "btc_ret","ret_vs_btc","btc_momentum","mom_vs_btc","btc_vol",
    # Category-relative features (3) — how this coin moves vs its peers
    "ret_vs_cat","cat_breadth","mom_vs_cat",
    # Gap features (2) — crypto gaps are meaningful at 4h bar boundaries
    "gap_pct","gap_filled",
    # Cross-sectional ranks (11)
    "rank_momentum","rank_ret","rank_volume","rank_vwap","rank_slope",
    "rank_orb","rank_atr_inv","rank_vol","rank_trend","rank_rsi","rank_range"
]

for d in [DATA_DIR, MODEL_DIR, OUTCOME_DIR, SCAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════
models = {}
calibrators = {}
model_meta = {}
last_scans = {}
training_in_progress = False
training_progress = {"phase":"idle","pct":0,"message":""}
sweep_in_progress = False
sweep_progress = {"phase":"idle","current":0,"total":0,"message":"","currentTP":None,"currentSL":None}

STATUS_PATH = DATA_DIR / "status.json"
SWEEP_RESULTS_PATH = DATA_DIR / "sweep_results.json"

def load_sweep_results():
    try: return json.loads(SWEEP_RESULTS_PATH.read_text())
    except: return {"grid":[],"startedAt":None,"completedAt":None}
def save_sweep_results(r): SWEEP_RESULTS_PATH.write_text(json.dumps(r,indent=2))
def load_status():
    try: return json.loads(STATUS_PATH.read_text())
    except: return {"trained":False,"trainDate":None,"outcomeDays":0,"daysSinceRetrain":0}
def save_status(s): STATUS_PATH.write_text(json.dumps(s,indent=2))
status = load_status()

def load_models():
    global models, calibrators, model_meta
    for h in SCAN_HOURS:
        mp = MODEL_DIR / f"model_{h}.txt"
        cp = MODEL_DIR / f"calibrator_{h}.pkl"
        mtp = MODEL_DIR / f"meta_{h}.json"
        if mp.exists():
            models[h] = lgb.Booster(model_file=str(mp))
            log.info(f"Loaded model {h:02d}:00 UTC")
        if cp.exists():
            calibrators[h] = pickle.loads(cp.read_bytes())
        if mtp.exists():
            model_meta[h] = json.loads(mtp.read_text())
load_models()

LAST_SCAN_PATH = DATA_DIR / "last_scans.json"
try: last_scans = json.loads(LAST_SCAN_PATH.read_text())
except: last_scans = {}

# ═══════════════════════════════════════════════════════════════════
# TIME / COINBASE HELPERS
# ═══════════════════════════════════════════════════════════════════
def now_utc(): return datetime.now(UTC)
def today_utc(): return now_utc().strftime("%Y-%m-%d")
def hour_utc(): return now_utc().hour
def market_open():
    # Crypto trades 24/7. We return True unless we want to artificially gate scans.
    return True
def has_creds():
    # Public endpoints don't require credentials. Always "true" for the purposes
    # of enabling the scan schedule.
    return True

def sleep_ms(ms): time.sleep(ms/1000.0)
def chunk(a, n):
    o = []
    for i in range(0, len(a), n): o.append(a[i:i+n])
    return o

def cb_client():
    # No auth needed for public endpoints; we keep the interface identical to
    # alpaca_client() so the rest of the code mirrors the original.
    headers = {"User-Agent":"coinbase-intraday-scanner/1.0","Accept":"application/json"}
    return httpx.Client(base_url=CB_API_URL, headers=headers, timeout=30.0)

def _candles_to_dicts(raw, granularity_sec):
    """
    Coinbase returns candles as [time, low, high, open, close, volume] with
    time = unix seconds at the *start* of the bucket, newest first.
    We normalize to Alpaca-style dicts {t, o, h, l, c, v} ordered ascending
    so the downstream feature code doesn't change.
    """
    out = []
    for row in raw:
        if len(row) < 6: continue
        ts, lo, hi, op, cl, vol = row[0], row[1], row[2], row[3], row[4], row[5]
        iso = datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00","Z")
        out.append({"t":iso,"o":op,"h":hi,"l":lo,"c":cl,"v":vol})
    out.sort(key=lambda b: b["t"])
    return out

def fetch_candles_for_product(client, product, start_dt, end_dt, granularity_sec):
    """
    Fetch candles for ONE product between start_dt and end_dt.
    Coinbase caps each request at 300 candles, so we page through by
    shifting the window forward in 300-bucket chunks.
    """
    results = []
    window_sec = 300 * granularity_sec  # max 300 candles per request
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(seconds=window_sec), end_dt)
        params = {
            "granularity": granularity_sec,
            "start": cur.isoformat().replace("+00:00","Z"),
            "end": nxt.isoformat().replace("+00:00","Z"),
        }
        for attempt in range(5):
            r = client.get(f"/products/{product}/candles", params=params)
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            if r.status_code == 404:
                # Product may be delisted / unavailable → give up on this one
                return results
            if r.status_code >= 500:
                time.sleep(1.0 + attempt)
                continue
            try:
                raw = r.json()
            except:
                raw = []
            if isinstance(raw, dict) and raw.get("message"):
                # e.g. {"message":"NotFound"} → skip
                return results
            results.extend(_candles_to_dicts(raw, granularity_sec))
            break
        cur = nxt
        time.sleep(0.15)  # light throttle → ~6 req/s max
    # Dedup (Coinbase sometimes returns overlap at window edges)
    seen_ts = set()
    uniq = []
    for b in results:
        if b["t"] in seen_ts: continue
        seen_ts.add(b["t"])
        uniq.append(b)
    return uniq

def fetch_candles_bulk(client, products, start_dt, end_dt, granularity_sec):
    """Fetch candles for many products. Serial, but with short sleeps."""
    out = {}
    total = len(products)
    for i, p in enumerate(products):
        try:
            bars = fetch_candles_for_product(client, p, start_dt, end_dt, granularity_sec)
            if bars: out[p] = bars
        except Exception as e:
            log.warning(f"candles {p}: {e}")
        if (i+1) % 20 == 0:
            log.info(f"  fetched {i+1}/{total} products")
    return out

def fetch_stats(client, products):
    """24h stats per product → used for current-snapshot price in live scan."""
    stats = {}
    for p in products:
        try:
            r = client.get(f"/products/{p}/stats")
            if r.status_code == 200:
                stats[p] = r.json()
        except Exception as e:
            log.debug(f"stats {p}: {e}")
        time.sleep(0.05)
    return stats

def fetch_ticker(client, product):
    try:
        r = client.get(f"/products/{product}/ticker")
        if r.status_code == 200: return r.json()
    except: pass
    return None

def bar_to_utc_minutes(b):
    """Convert bar timestamp to UTC minutes-since-midnight."""
    try:
        dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(UTC)
        return dt.hour * 60 + dt.minute
    except:
        return None

# ═══════════════════════════════════════════════════════════════════
# FIRST-PASSAGE LABEL: does price hit TP before SL within HORIZON_BARS?
# ═══════════════════════════════════════════════════════════════════
def compute_trade_outcome(entry_price, bars_after_entry, tp_pct=None, sl_pct=None,
                          horizon_bars=None):
    """
    Walk bars in order. Check each bar against TP/SL barriers. Force-close at
    horizon. Returns (outcome, pnl_pct, exit_reason).

    Crypto version: no 15:55 force-close; we cap at horizon_bars (default 16
    fifteen-minute bars = 4 hours).
    """
    if tp_pct is None: tp_pct = TP_PCT
    if sl_pct is None: sl_pct = SL_PCT
    if horizon_bars is None: horizon_bars = HORIZON_BARS

    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    window = bars_after_entry[:horizon_bars]
    for b in window:
        hit_tp = b["h"] >= tp_price
        hit_sl = b["l"] <= sl_price
        if hit_tp and hit_sl:
            # Both barriers in same bar — disambiguate by bar open
            if b["o"] >= entry_price:
                return (1, round(tp_pct * 100, 3), "tp")
            else:
                return (0, round(-sl_pct * 100, 3), "sl")
        elif hit_tp:
            return (1, round(tp_pct * 100, 3), "tp")
        elif hit_sl:
            return (0, round(-sl_pct * 100, 3), "sl")

    # Horizon reached, no barrier hit → mark-to-close at last bar
    if window:
        pnl = (window[-1]["c"] - entry_price) / entry_price
        return (1 if pnl > 0 else 0, round(pnl * 100, 3), "horizon")
    return (0, 0.0, "no_data")

# ═══════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════
def compute_btc_context(btc_bars):
    """
    Market-factor features computed from BTC bars up to scan time.
    Same for every coin at a given scan — describes the market environment.
    """
    if len(btc_bars) < 3:
        return {"btc_ret":0,"btc_momentum":0,"btc_vol":0}
    btc_open = btc_bars[0]["o"]
    btc_current = btc_bars[-1]["c"]
    btc_ret = (btc_current - btc_open) / btc_open if btc_open > 0 else 0

    tail = btc_bars[-3:]
    btc_momentum = (tail[-1]["c"] - tail[0]["o"]) / tail[0]["o"] if tail[0]["o"] > 0 else 0

    btc_rets = [math.log(btc_bars[i]["c"]/btc_bars[i-1]["c"])
                for i in range(1, len(btc_bars)) if btc_bars[i-1]["c"] > 0]
    # Annualization-agnostic: we just want the std as a volatility proxy.
    # Scale by sqrt(bars-per-day) for dimensional consistency with R2K version.
    bars_per_day = 24 * BARS_PER_HOUR
    btc_vol = np.std(btc_rets) * math.sqrt(bars_per_day) if len(btc_rets) > 1 else 0

    return {"btc_ret":btc_ret, "btc_momentum":btc_momentum, "btc_vol":btc_vol}

def compute_features(bars, daily_bars, current_price, open_price, scan_hour,
                     btc_context=None, prev_close=None):
    """
    Per-coin features. Mirrors the R2K compute_features exactly in spirit —
    the only structural changes: BTC replaces SPY, and there's no "hours_left"
    since the horizon is a fixed HORIZON_HOURS window.

    'bars' = FEATURE_LOOKBACK_BARS of 15-min candles strictly BEFORE scan time
    'daily_bars' = recent daily candles for ATR / ADV
    'current_price' = last close at feature time
    'open_price' = first open of the scan's "day" (00:00 UTC candle)
    'scan_hour' = UTC hour of the scan slot (0/4/8/12/16/20)
    """
    if len(bars) < 15: return None

    # ─── Original bar-data features ──────────────────────────────
    tail = bars[-3:]
    momentum = (tail[-1]["c"] - tail[0]["o"]) / tail[0]["o"] if tail[0]["o"] > 0 else 0
    ret_from_open = (current_price - open_price) / open_price if open_price > 0 else 0

    avg_bv = sum(b["v"] for b in bars) / len(bars)
    rel_volume = 1.0
    if daily_bars and len(daily_bars) >= 2:
        adv = sum(d["v"] for d in daily_bars[-5:]) / min(5, len(daily_bars))
        bars_per_day = 24 * BARS_PER_HOUR
        exp = adv / bars_per_day
        if exp > 0: rel_volume = avg_bv / exp

    vn = sum((b["h"]+b["l"]+b["c"])/3 * b["v"] for b in bars)
    vd = sum(b["v"] for b in bars)
    vwap = vn/vd if vd > 0 else current_price
    vwap_dist = (current_price - vwap) / vwap if vwap > 0 else 0

    vwap_slope = 0.0
    if len(bars) >= 6:
        t = len(bars)//3
        n1 = sum((b["h"]+b["l"]+b["c"])/3*b["v"] for b in bars[:t])
        d1 = sum(b["v"] for b in bars[:t])
        n2 = sum((b["h"]+b["l"]+b["c"])/3*b["v"] for b in bars[:t*2])
        d2 = sum(b["v"] for b in bars[:t*2])
        v1, v2 = (n1/d1 if d1>0 else current_price), (n2/d2 if d2>0 else current_price)
        vwap_slope = (v2-v1)/v1 if v1 > 0 else 0

    # ORB: "opening range" = first N bars of the scan's UTC day
    orb = bars[:min(6,len(bars))]
    orb_h, orb_l = max(b["h"] for b in orb), min(b["l"] for b in orb)
    orb_range = orb_h - orb_l
    orb_strength = (current_price - orb_h)/orb_range if orb_range > 0 else 0

    # ATR proxy from daily bars; fall back to 2% of price (crypto baseline)
    atr = current_price * 0.02
    if daily_bars and len(daily_bars) >= 5:
        trs = [max(daily_bars[i]["h"]-daily_bars[i]["l"],
                    abs(daily_bars[i]["h"]-daily_bars[i-1]["c"]),
                    abs(daily_bars[i]["l"]-daily_bars[i-1]["c"]))
               for i in range(1, len(daily_bars))]
        atr = np.mean(trs[-5:])
    target = current_price * TP_PCT
    # atr_reach: how many ATRs the TP is away, scaled by horizon. Lower = easier.
    atr_scaled = atr * math.sqrt(HORIZON_HOURS/24.0) if atr > 0 else current_price*0.01
    atr_reach = target/atr_scaled if atr_scaled > 0 else 2.0

    rets = [math.log(bars[i]["c"]/bars[i-1]["c"]) for i in range(1,len(bars)) if bars[i-1]["c"]>0]
    bars_per_day = 24 * BARS_PER_HOUR
    realized_vol = np.std(rets) * math.sqrt(bars_per_day) if len(rets) > 1 else 0

    trend_str = 0.0
    if len(bars) >= 10:
        half = len(bars)//2
        trend_str = (np.mean([b["c"] for b in bars[-half:]]) / np.mean([b["c"] for b in bars[:half]]) - 1)

    rsi = 50.0
    if len(bars) >= 15:
        gains = [max(0, bars[i]["c"]-bars[i-1]["c"]) for i in range(len(bars)-14, len(bars))]
        losses = [max(0, bars[i-1]["c"]-bars[i]["c"]) for i in range(len(bars)-14, len(bars))]
        ag, al = np.mean(gains), np.mean(losses)
        rsi = 100 - (100/(1+ag/al)) if al > 0 else 100

    last_r = (bars[-1]["h"]-bars[-1]["l"])/bars[-1]["c"] if bars[-1]["c"]>0 else 0
    avg_r = np.mean([(b["h"]-b["l"])/b["c"] for b in bars[-10:] if b["c"]>0]) or 1
    range_expansion = last_r/avg_r if avg_r > 0 else 1

    # ─── BTC-relative features ───────────────────────────────────
    bc = btc_context or {"btc_ret":0,"btc_momentum":0,"btc_vol":0}
    btc_ret = bc["btc_ret"]
    ret_vs_btc = ret_from_open - btc_ret
    btc_momentum_val = bc["btc_momentum"]
    mom_vs_btc = momentum - btc_momentum_val
    btc_vol = bc["btc_vol"]

    # ─── Gap features — overnight analog: difference between scan's opening
    # candle and the prior day's close. In crypto this still captures
    # regime-change-at-UTC-midnight behavior.
    gap_pct = 0.0
    gap_filled = 0
    if prev_close and prev_close > 0:
        gap_pct = (open_price - prev_close) / prev_close
        if gap_pct > 0:
            gap_filled = 1 if min(b["l"] for b in bars) <= prev_close else 0
        elif gap_pct < 0:
            gap_filled = 1 if max(b["h"] for b in bars) >= prev_close else 0

    # ─── Category-relative placeholders (filled by add_category_relative) ──
    ret_vs_cat = 0.0
    cat_breadth = 0.5
    mom_vs_cat = 0.0

    return {
        "momentum":momentum,"ret_from_open":ret_from_open,"rel_volume":rel_volume,
        "vwap_dist":vwap_dist,"vwap_slope":vwap_slope,"orb_strength":orb_strength,
        "atr_reach":atr_reach,"realized_vol":realized_vol,"trend_str":trend_str,
        "rsi":rsi,"range_expansion":range_expansion,
        # BTC-relative
        "btc_ret":btc_ret,"ret_vs_btc":ret_vs_btc,
        "btc_momentum":btc_momentum_val,"mom_vs_btc":mom_vs_btc,"btc_vol":btc_vol,
        # Category-relative (placeholders)
        "ret_vs_cat":ret_vs_cat,"cat_breadth":cat_breadth,"mom_vs_cat":mom_vs_cat,
        # Gap
        "gap_pct":gap_pct,"gap_filled":gap_filled
    }

def add_ranks(features_list):
    n = len(features_list)
    if n < 2: return features_list
    def pr(vals):
        arr = np.array(vals); o = arr.argsort().argsort()
        return o / (n-1)
    ranks = {
        "rank_momentum": pr([f["momentum"] for f in features_list]),
        "rank_ret":      pr([f["ret_from_open"] for f in features_list]),
        "rank_volume":   pr([f["rel_volume"] for f in features_list]),
        "rank_vwap":     pr([f["vwap_dist"] for f in features_list]),
        "rank_slope":    pr([f["vwap_slope"] for f in features_list]),
        "rank_orb":      pr([f["orb_strength"] for f in features_list]),
        "rank_atr_inv":  pr([-f["atr_reach"] for f in features_list]),
        "rank_vol":      pr([f["realized_vol"] for f in features_list]),
        "rank_trend":    pr([f["trend_str"] for f in features_list]),
        "rank_rsi":      pr([50-abs(f["rsi"]-55) for f in features_list]),
        "rank_range":    pr([f["range_expansion"] for f in features_list]),
    }
    for i in range(n):
        for k, v in ranks.items(): features_list[i][k] = float(v[i])
    return features_list

def add_category_relative(features_list, cat_list):
    """
    Category-relative features: how each coin compares to its category peers.
    Direct analog of R2K's add_sector_relative.
    """
    n = len(features_list)
    if n < 2: return features_list

    cat_indices = defaultdict(list)
    for i, c in enumerate(cat_list):
        cat_indices[c].append(i)

    for i in range(n):
        c = cat_list[i]
        peers = cat_indices[c]
        if len(peers) < 2: continue

        peer_rets = [features_list[j]["ret_from_open"] for j in peers if j != i]
        if peer_rets:
            features_list[i]["ret_vs_cat"] = features_list[i]["ret_from_open"] - np.mean(peer_rets)

        positive = sum(1 for j in peers if features_list[j]["ret_from_open"] > 0)
        features_list[i]["cat_breadth"] = positive / len(peers)

        peer_moms = [features_list[j]["momentum"] for j in peers if j != i]
        if peer_moms:
            features_list[i]["mom_vs_cat"] = features_list[i]["momentum"] - np.mean(peer_moms)

    return features_list

def feat_to_arr(f):
    return np.array([f.get(n, 0) for n in FEATURE_NAMES])

# ═══════════════════════════════════════════════════════════════════
# TRAINING — FIRST-PASSAGE LABELS
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BARS_DAILY_CACHE = CACHE_DIR / "bars_daily.pkl"
BARS_INTRADAY_CACHE = CACHE_DIR / "bars_intraday.pkl"
CACHE_MAX_AGE_HOURS = 24

def cache_age_hours(path):
    if not path.exists(): return 999
    return (time.time() - path.stat().st_mtime) / 3600

# Training window: crypto moves a lot; 180 days of 15m bars is ~17k bars per
# product, which fits comfortably even across ~140 products. Adjust if needed.
TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", 180))

def run_training(tp_pct=None, sl_pct=None):
    global models, calibrators, model_meta, training_in_progress, training_progress, status
    if training_in_progress: return
    training_in_progress = True
    training_progress = {"phase":"starting","pct":0,"message":"Starting..."}

    use_tp = tp_pct if tp_pct is not None else TP_PCT
    use_sl = sl_pct if sl_pct is not None else SL_PCT

    try:
        daily_age = cache_age_hours(BARS_DAILY_CACHE)
        intra_age = cache_age_hours(BARS_INTRADAY_CACHE)
        cache_fresh = daily_age < CACHE_MAX_AGE_HOURS and intra_age < CACHE_MAX_AGE_HOURS

        # Include BENCHMARK (BTC) in fetch list for market-factor features
        fetch_products = list(set(PRODUCTS + [BENCHMARK]))

        if cache_fresh:
            training_progress = {"phase":"loading_cache","pct":5,
                "message":f"Loading cached bars (age {intra_age:.1f}h)..."}
            log.info(f"Using cached bars (daily age {daily_age:.1f}h, intraday age {intra_age:.1f}h)")
            daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
            intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())
        else:
            client = cb_client()
            end_dt = now_utc().replace(minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=TRAIN_DAYS)

            training_progress = {"phase":"fetch_daily","pct":3,
                "message":f"Fetching {TRAIN_DAYS}d daily bars (incl BTC)..."}
            daily_bars = fetch_candles_bulk(client, fetch_products, start_dt, end_dt, 86400)

            training_progress = {"phase":"fetch_intraday","pct":8,
                "message":f"Fetching {TRAIN_DAYS}d of 15-min bars (incl BTC)..."}
            intraday = fetch_candles_bulk(client, fetch_products, start_dt, end_dt, CANDLE_GRANULARITY)
            client.close()

            training_progress = {"phase":"caching","pct":44,"message":"Caching bars to disk..."}
            BARS_DAILY_CACHE.write_bytes(pickle.dumps(daily_bars))
            BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(intraday))
            log.info(f"Cached {sum(len(v) for v in intraday.values())} intraday bars to disk")

        training_progress = {"phase":"grouping","pct":45,"message":"Grouping bars by date..."}
        # by_td[product][YYYY-MM-DD] = list of bars on that UTC day
        by_td = defaultdict(lambda: defaultdict(list))
        for product in fetch_products:
            for b in intraday.get(product, []):
                by_td[product][b["t"][:10]].append(b)

        all_dates = sorted(set(d for t in by_td for d in by_td[t]))
        log.info(f"Training: {len(all_dates)} dates, TP={use_tp*100:.2f}% / SL={use_sl*100:.2f}%, "
                 f"horizon={HORIZON_HOURS}h, {len(fetch_products)} products (+BTC benchmark)")

        training_progress = {"phase":"features","pct":50,
            "message":f"Computing features + BTC/category context (TP {use_tp*100:.2f}% / SL {use_sl*100:.2f}%)..."}
        rows_per_hour = defaultdict(list)

        for di, date in enumerate(all_dates):
            # All UTC-intraday bars for this day, all products
            # (precompute a flat per-product sorted list for this date)
            for scan_hour in SCAN_HOURS:
                scan_min = scan_hour * 60

                # ── BTC context at this scan point ──
                btc_day = by_td[BENCHMARK].get(date, [])
                btc_before = [b for b in btc_day if (bar_to_utc_minutes(b) or 0) < scan_min]
                btc_ctx = compute_btc_context(btc_before)

                date_features, date_meta, date_cats = [], [], []

                for product in PRODUCTS:
                    day_bars = by_td[product].get(date, [])
                    if len(day_bars) < 12: continue

                    # All bars on this date AND the next HORIZON_HOURS of the
                    # following date (since a 20:00 scan at hour=20 has its
                    # horizon cross midnight). We grab "after" bars from both
                    # today's remaining bars plus tomorrow's early bars.
                    tmr_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                    tmr_bars = by_td[product].get(tmr_date, [])

                    before, after_today = [], []
                    for b in day_bars:
                        bm = bar_to_utc_minutes(b)
                        if bm is None: continue
                        if bm < scan_min: before.append(b)
                        else: after_today.append(b)

                    after = after_today + tmr_bars
                    # Lookback window: last FEATURE_LOOKBACK_BARS of `before`
                    before = before[-FEATURE_LOOKBACK_BARS:]
                    if len(before) < 15 or len(after) < 2: continue

                    entry_price = after[0]["o"]
                    feature_price = before[-1]["c"]
                    open_price = day_bars[0]["o"]
                    daily_up_to = [d for d in daily_bars.get(product,[]) if d["t"][:10] < date][-DAILY_LOOKBACK:]
                    prev_close = daily_up_to[-1]["c"] if daily_up_to else None

                    feat = compute_features(before, daily_up_to, feature_price, open_price, scan_hour,
                                            btc_context=btc_ctx, prev_close=prev_close)
                    if feat is None: continue

                    outcome, pnl, reason = compute_trade_outcome(
                        entry_price, after[1:], tp_pct=use_tp, sl_pct=use_sl)

                    date_features.append(feat)
                    date_meta.append({"product":product,"label":outcome,"pnl":pnl,"reason":reason,"date":date})
                    date_cats.append(CATEGORIES.get(product,"?"))

                if len(date_features) >= 10:
                    add_ranks(date_features)
                    add_category_relative(date_features, date_cats)
                    for j in range(len(date_features)):
                        date_features[j]["label"] = date_meta[j]["label"]
                        date_features[j]["date"] = date_meta[j]["date"]
                        date_features[j]["pnl"] = date_meta[j]["pnl"]
                        date_features[j]["reason"] = date_meta[j]["reason"]
                        rows_per_hour[scan_hour].append(date_features[j])

            if (di+1) % 20 == 0:
                training_progress = {"phase":"features","pct":50+int((di/len(all_dates))*35),
                    "message":f"Processed {di+1}/{len(all_dates)} days..."}

        training_progress = {"phase":"training","pct":87,"message":"Training LightGBM models..."}
        new_models, new_cals, new_meta = {}, {}, {}

        for h in SCAN_HOURS:
            rows = rows_per_hour[h]
            if len(rows) < 200:
                log.warning(f"{h:02d}:00 UTC only {len(rows)} samples, skip"); continue

            df = pd.DataFrame(rows)
            dates = sorted(df["date"].unique())
            # Random date-level split (not time-ordered) to reduce regime bias.
            # Seeded per-slot so results are reproducible and each slot gets a
            # different validation set. NOTE: this can leak info if adjacent-day
            # outcomes are correlated; validate against recorded outcomes once
            # you have 2+ weeks of live data before trusting the metrics.
            rng = np.random.default_rng(42 + h)
            val_size = max(20, int(len(dates) * 0.2))
            val_dates = set(rng.choice(dates, size=val_size, replace=False).tolist())
            train_dates = set(dates) - val_dates

            train_df = df[df["date"].isin(train_dates)]
            val_df = df[df["date"].isin(val_dates)]

            X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["label"].values
            X_va, y_va = val_df[FEATURE_NAMES].values, val_df["label"].values

            win_rate_train = y_tr.mean()
            win_rate_val = y_va.mean()
            log.info(f"{h:02d}:00 UTC — train {len(train_df)} (WR {win_rate_train:.3f}), "
                     f"val {len(val_df)} (WR {win_rate_val:.3f})")

            ts = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES)
            vs = lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES, reference=ts)

            params = {
                "objective":"binary","metric":"binary_logloss",
                "boosting_type":"gbdt",
                "num_leaves":15,         # was 31 — simpler model resists regime shift
                "learning_rate":0.03,    # was 0.05 — slower = more trees before overfit
                "feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":5,
                "min_child_samples":50,  # was 20 — demand stronger evidence per leaf
                "lambda_l2":1.0,         # new — L2 regularization for noisy labels
                "verbose":-1
            }
            model = lgb.train(params, ts, num_boost_round=800, valid_sets=[vs],
                              callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
                                                 # ^^ patience 30 -> 80: a few bad
                                                 # rounds won't kill training

            val_probs = model.predict(X_va)
            auc = roc_auc_score(y_va, val_probs) if len(set(y_va)) > 1 else 0

            val_df = val_df.copy()
            val_df["prob"] = val_probs

            # Top-10 metrics per validation day
            p10_list, pnl10_list = [], []
            for d in val_dates:
                day = val_df[val_df["date"]==d].nlargest(10,"prob")
                if len(day) >= 10:
                    p10_list.append(day["label"].mean())
                    pnl10_list.append(day["pnl"].mean())
            avg_p10 = np.mean(p10_list) if p10_list else 0
            avg_pnl10 = np.mean(pnl10_list) if pnl10_list else 0

            breakeven_p = use_sl / (use_sl + use_tp)

            def ev_at(thresh):
                subset = val_df[val_df["prob"] >= thresh]
                if len(subset) == 0: return 0, 0
                return subset["pnl"].mean(), len(subset)
            ev_at_be, n_at_be = ev_at(breakeven_p)
            ev_at_be5, n_at_be5 = ev_at(breakeven_p + 0.05)
            ev_at_50 = val_df[val_df["prob"]>=0.5]["pnl"].mean() if len(val_df[val_df["prob"]>=0.5])>0 else 0
            n_above_50 = len(val_df[val_df["prob"]>=0.5])
            ev_at_55 = val_df[val_df["prob"]>=0.55]["pnl"].mean() if len(val_df[val_df["prob"]>=0.55])>0 else 0
            n_above_55 = len(val_df[val_df["prob"]>=0.55])

            val_reasons = val_df["reason"].value_counts().to_dict() if "reason" in val_df.columns else {}

            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.95)
            cal.fit(val_probs, y_va)

            imp = dict(zip(FEATURE_NAMES, model.feature_importance("gain").tolist()))
            ti = sum(imp.values()) or 1
            imp = {k: round(v/ti, 4) for k,v in imp.items()}

            model.save_model(str(MODEL_DIR / f"model_{h}.txt"))
            (MODEL_DIR / f"calibrator_{h}.pkl").write_bytes(pickle.dumps(cal))

            meta = {
                "scan_hour":h,
                "train_samples":len(train_df),"val_samples":len(val_df),
                "train_dates":len(train_dates),"val_dates":len(val_dates),
                "train_win_rate":round(float(win_rate_train),4),
                "val_win_rate":round(float(win_rate_val),4),
                "auc":round(auc,4),
                "avg_win_rate_top10":round(float(avg_p10),4),
                "avg_pnl_top10":round(float(avg_pnl10),3),
                "ev_above_50pct":round(float(ev_at_50),3),
                "n_above_50pct":int(n_above_50),
                "ev_above_55pct":round(float(ev_at_55),3),
                "n_above_55pct":int(n_above_55),
                "breakeven_threshold":round(breakeven_p,3),
                "ev_above_breakeven":round(float(ev_at_be),3),
                "n_above_breakeven":int(n_at_be),
                "ev_above_breakeven_plus5":round(float(ev_at_be5),3),
                "n_above_breakeven_plus5":int(n_at_be5),
                "val_exit_reasons":val_reasons,
                "importance":imp,
                "trained_at":datetime.now(UTC).isoformat(),
                "best_iteration":model.best_iteration,
                "tp_pct":use_tp*100, "sl_pct":use_sl*100,
                "horizon_hours":HORIZON_HOURS
            }
            (MODEL_DIR / f"meta_{h}.json").write_text(json.dumps(meta, indent=2))

            new_models[h] = model
            new_cals[h] = cal
            new_meta[h] = meta
            log.info(f"{h:02d}:00 UTC — AUC {auc:.3f}, Top10 WR {avg_p10:.3f} "
                     f"(base {win_rate_val:.3f}), Top10 PnL {avg_pnl10:.3f}%, EV@50%: {ev_at_50:.3f}%")

        models.update(new_models)
        calibrators.update(new_cals)
        model_meta.update(new_meta)
        status["trained"] = True
        status["trainDate"] = datetime.now(UTC).isoformat()
        status["daysSinceRetrain"] = 0
        status["activeTP"] = use_tp * 100
        status["activeSL"] = use_sl * 100
        save_status(status)

        training_progress = {"phase":"done","pct":100,
            "message":f"Done. {len(new_models)} models trained (TP {use_tp*100:.2f}% / SL {use_sl*100:.2f}%)."}
        log.info(f"Training complete. Active TP/SL: {use_tp*100:.2f}% / {use_sl*100:.2f}%")

    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        training_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        training_in_progress = False

# ═══════════════════════════════════════════════════════════════════
# SWEEP: grid search over TP/SL combinations
# ═══════════════════════════════════════════════════════════════════
# Crypto ranges: wider than R2K because moves are larger.
SWEEP_TP_VALUES = [1.0, 2.0, 3.0]          # percent
SWEEP_SL_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0]  # percent

def summarize_models_for_sweep(tp, sl):
    summary = {
        "tp_pct": tp, "sl_pct": sl,
        "breakeven": round(sl / (sl + tp) * 100, 2),
        "hours": {},
        "avg_top10_wr": None, "avg_top10_pnl": None, "avg_auc": None,
        "avg_base_wr": None, "avg_edge": None,
        "completedAt": datetime.now(UTC).isoformat()
    }
    top10_wrs, top10_pnls, aucs, base_wrs = [], [], [], []
    for h in SCAN_HOURS:
        m = model_meta.get(h)
        if not m: continue
        wr10 = m.get("avg_win_rate_top10", 0) * 100
        pnl10 = m.get("avg_pnl_top10", 0)
        auc = m.get("auc", 0)
        base = m.get("val_win_rate", 0) * 100
        edge = wr10 - summary["breakeven"]
        summary["hours"][str(h)] = {
            "top10_wr": round(wr10, 2),
            "top10_pnl": round(pnl10, 3),
            "auc": round(auc, 4),
            "base_wr": round(base, 2),
            "edge": round(edge, 2)
        }
        top10_wrs.append(wr10); top10_pnls.append(pnl10)
        aucs.append(auc); base_wrs.append(base)

    if top10_wrs:
        summary["avg_top10_wr"] = round(float(np.mean(top10_wrs)), 2)
        summary["avg_top10_pnl"] = round(float(np.mean(top10_pnls)), 3)
        summary["avg_auc"] = round(float(np.mean(aucs)), 4)
        summary["avg_base_wr"] = round(float(np.mean(base_wrs)), 2)
        summary["avg_edge"] = round(summary["avg_top10_wr"] - summary["breakeven"], 2)
    return summary

def run_sweep(resume=True):
    global sweep_in_progress, sweep_progress, training_in_progress
    if sweep_in_progress: return
    if training_in_progress: return
    sweep_in_progress = True

    grid_cells = [(tp, sl) for tp in SWEEP_TP_VALUES for sl in SWEEP_SL_VALUES]
    total = len(grid_cells)

    existing = load_sweep_results() if resume else {"grid":[],"startedAt":None,"completedAt":None}
    completed_keys = {f"{r['tp_pct']}_{r['sl_pct']}" for r in existing.get("grid",[])}
    if not existing.get("startedAt") or not resume:
        existing = {"grid":[], "startedAt":datetime.now(UTC).isoformat(), "completedAt":None,
                    "gridShape":{"tpValues":SWEEP_TP_VALUES, "slValues":SWEEP_SL_VALUES}}
        completed_keys = set()

    log.info(f"Sweep: {total} cells, {len(completed_keys)} already complete, {total-len(completed_keys)} to run")

    try:
        for idx, (tp, sl) in enumerate(grid_cells):
            key = f"{tp}_{sl}"
            if key in completed_keys:
                log.info(f"Sweep cell {idx+1}/{total}: TP {tp}% / SL {sl}% — skip (cached)")
                continue

            sweep_progress = {
                "phase":"running","current":idx+1,"total":total,
                "currentTP":tp,"currentSL":sl,
                "message":f"Cell {idx+1}/{total}: TP {tp}% / SL {sl}% (break-even {sl/(sl+tp)*100:.1f}%)"
            }

            log.info(f"Sweep cell {idx+1}/{total}: starting TP={tp}% SL={sl}%")
            run_training(tp_pct=tp/100.0, sl_pct=sl/100.0)

            cell_summary = summarize_models_for_sweep(tp, sl)
            existing["grid"].append(cell_summary)
            save_sweep_results(existing)
            log.info(f"Sweep cell {idx+1}/{total} done: avg top10 WR {cell_summary['avg_top10_wr']}%, "
                     f"breakeven {cell_summary['breakeven']}%, edge {cell_summary['avg_edge']}%")

        existing["completedAt"] = datetime.now(UTC).isoformat()
        save_sweep_results(existing)
        sweep_progress = {"phase":"done","current":total,"total":total,
            "message":f"Sweep complete. {total} cells evaluated.",
            "currentTP":None,"currentSL":None}
        log.info("Sweep complete.")

    except Exception as e:
        log.error(f"Sweep failed: {e}", exc_info=True)
        sweep_progress = {"phase":"error","current":0,"total":total,"message":str(e),
                         "currentTP":None,"currentSL":None}
    finally:
        sweep_in_progress = False

# ═══════════════════════════════════════════════════════════════════
# LIVE SCAN
# ═══════════════════════════════════════════════════════════════════
def run_live_scan(scan_hour):
    if scan_hour not in models: raise ValueError(f"No model for {scan_hour:02d}:00 UTC")
    t0 = time.time()
    client = cb_client()

    # Fetch enough history to cover FEATURE_LOOKBACK_BARS before the scan hour,
    # which means we need bars from the scan day's 00:00 UTC up to now.
    now = now_utc()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # If the scan is for a future hour today, we use whatever bars exist up to "now".
    # If it's for the current hour, same thing. This is a live snapshot.
    start_dt = day_start
    end_dt = now

    fetch_products = list(set(PRODUCTS + [BENCHMARK]))
    training_progress_msg = f"Fetching {len(fetch_products)} products @ 15m bars..."
    log.info(training_progress_msg)
    intra = fetch_candles_bulk(client, fetch_products, start_dt, end_dt, CANDLE_GRANULARITY)

    # 24h stats for current price (fallback to last candle close)
    stats = fetch_stats(client, PRODUCTS)

    # Daily bars for ATR / ADV — 20 days back
    daily_start = (now - timedelta(days=20)).replace(hour=0, minute=0, second=0, microsecond=0)
    daily = fetch_candles_bulk(client, fetch_products, daily_start, now, 86400)
    client.close()

    btc_bars = intra.get(BENCHMARK, [])
    btc_ctx = compute_btc_context(btc_bars)

    raw_feats, coin_info, coin_cats = [], [], []
    for product in PRODUCTS:
        bars = intra.get(product, [])
        if len(bars) < 15: continue
        # Current price: prefer 24h stats "last", fall back to last bar close
        s = stats.get(product, {})
        try: cp = float(s.get("last")) if s.get("last") else bars[-1]["c"]
        except: cp = bars[-1]["c"]
        op = bars[0]["o"]
        dl = daily.get(product, [])
        prev_close = dl[-2]["c"] if len(dl) >= 2 else (dl[-1]["c"] if dl else None)

        feat = compute_features(bars, dl, cp, op, scan_hour,
                                btc_context=btc_ctx, prev_close=prev_close)
        if feat is None: continue
        raw_feats.append(feat)
        coin_info.append({"product":product,"category":CATEGORIES.get(product,"?"),
                          "price":cp,"open":op})
        coin_cats.append(CATEGORIES.get(product,"?"))

    if len(raw_feats) < 5: raise ValueError(f"Only {len(raw_feats)} coins with enough data")
    add_ranks(raw_feats)
    add_category_relative(raw_feats, coin_cats)

    meta = model_meta.get(scan_hour, {})
    active_tp = (meta.get("tp_pct", TP_PCT*100)) / 100
    active_sl = (meta.get("sl_pct", SL_PCT*100)) / 100

    X = np.array([feat_to_arr(f) for f in raw_feats])
    raw_probs = models[scan_hour].predict(X)
    cal_probs = calibrators[scan_hour].predict(raw_probs) if scan_hour in calibrators else raw_probs

    results = []
    for i in range(len(raw_feats)):
        ci, rf = coin_info[i], raw_feats[i]
        wp = float(cal_probs[i])
        ev = (wp * active_tp - (1 - wp) * active_sl) * 100
        # Price decimals adapt to price magnitude (BTC at 60k → 2dp, MOG at 0.000001 → 8dp)
        p = ci["price"]
        if p >= 100:       price_str = f"{p:,.2f}"
        elif p >= 1:       price_str = f"{p:.4f}"
        elif p >= 0.01:    price_str = f"{p:.5f}"
        else:              price_str = f"{p:.8f}"
        results.append({
            "rank":0,"product":ci["product"],"ticker":ci["product"],  # ticker alias for UI compat
            "category":ci["category"],"sector":ci["category"],         # sector alias
            "price":price_str,
            "changeFromOpen":f"{((ci['price']-ci['open'])/ci['open']*100):.2f}",
            "winProb":round(wp,4),
            "ev":round(ev,3),
            "rawScore":round(float(raw_probs[i]),4),
            "features":{
                "momentum":f"{rf['momentum']:.4f}","relVolume":f"{rf['rel_volume']:.2f}",
                "vwapDist":f"{rf['vwap_dist']*100:.2f}","vwapSlope":f"{rf['vwap_slope']:.4f}",
                "orbStrength":f"{rf['orb_strength']:.3f}","atrReach":f"{rf['atr_reach']:.2f}",
                "realizedVol":f"{rf['realized_vol']:.4f}","trendStr":f"{rf['trend_str']:.4f}",
                "rsi":f"{rf['rsi']:.1f}",
                "retVsBtc":f"{rf['ret_vs_btc']*100:.2f}","retVsCat":f"{rf['ret_vs_cat']*100:.2f}",
                "catBreadth":f"{rf['cat_breadth']:.2f}","gapPct":f"{rf['gap_pct']*100:.2f}"
            }
        })

    results.sort(key=lambda x: x["winProb"], reverse=True)
    for i,r in enumerate(results): r["rank"] = i+1

    elapsed = int((time.time()-t0)*1000)

    scan_result = {
        "data":results,"timestamp":datetime.now(UTC).isoformat(),"source":"live",
        "elapsed":elapsed,"scanHour":scan_hour,
        "modelAUC":meta.get("auc"),"modelWR10":meta.get("avg_win_rate_top10"),
        "modelPnL10":meta.get("avg_pnl_top10"),
        "scoreRange":{"min":results[-1]["rawScore"],"max":results[0]["rawScore"]} if results else None,
        "tp_pct":active_tp*100,"sl_pct":active_sl*100,
        "breakeven":round(active_sl/(active_sl+active_tp)*100,1),
        "horizonHours":HORIZON_HOURS
    }

    sp = SCAN_DIR / f"{today_utc()}.json"
    try: saved = json.loads(sp.read_text())
    except: saved = {}
    saved[str(scan_hour)] = results
    sp.write_text(json.dumps(saved))

    last_scans[str(scan_hour)] = scan_result
    LAST_SCAN_PATH.write_text(json.dumps(last_scans, default=str))

    log.info(f"Scan {scan_hour:02d}:00 UTC: {len(results)} coins, {elapsed}ms, "
             f"top5 EV: {[r['ev'] for r in results[:5]]}")
    return scan_result

# ═══════════════════════════════════════════════════════════════════
# OUTCOME RECORDING — FIRST-PASSAGE
# ═══════════════════════════════════════════════════════════════════
def record_outcomes():
    """Record the realized outcomes of today's scans. Run 4h after the LAST
    scan slot of the day (so the final horizon has elapsed). The R2K
    version ran at 16:12 ET; we run at 00:30 UTC (30 min after the 20:00 UTC
    scan's 4h horizon closes at 00:00 UTC of the next day)."""
    today = today_utc()
    # Record outcomes for YESTERDAY's scans (since the 20:00 scan's horizon
    # extends past midnight UTC). Check if today's outcomes exist first.
    yday = (now_utc() - timedelta(days=1)).strftime("%Y-%m-%d")
    out_path = OUTCOME_DIR / f"{yday}.json"
    if out_path.exists(): log.info(f"Outcomes {yday} already done."); return

    log.info(f"Recording outcomes {yday} (first-passage, {HORIZON_HOURS}h horizon)...")
    client = cb_client()
    try:
        # Fetch intraday bars covering yesterday through end of today's early UTC hours
        start_dt = datetime.strptime(yday,"%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1, hours=HORIZON_HOURS+1)
        all_bars = fetch_candles_bulk(client, PRODUCTS, start_dt, end_dt, CANDLE_GRANULARITY)
        client.close()
    except Exception as e:
        log.error(f"Outcome fetch: {e}"); client.close(); return

    sp = SCAN_DIR / f"{yday}.json"
    try: yday_scans = json.loads(sp.read_text())
    except: yday_scans = {}

    outcomes = {}
    for h in SCAN_HOURS:
        outcomes[str(h)] = []
        scan_min = h * 60
        for product in PRODUCTS:
            bars = all_bars.get(product, [])
            if len(bars) < 6: continue
            # Filter to this scan date's bars after the scan minute + HORIZON bars after
            before, after = [], []
            for b in bars:
                bdate = b["t"][:10]
                bm = bar_to_utc_minutes(b)
                if bm is None: continue
                if bdate == yday:
                    if bm < scan_min: before.append(b)
                    else: after.append(b)
                elif bdate > yday:
                    after.append(b)
            if not before or len(after) < 2: continue

            entry_price = after[0]["o"]
            outcome, pnl, reason = compute_trade_outcome(entry_price, after[1:])

            raw_score = None
            scanned = yday_scans.get(str(h), [])
            for s in scanned:
                if s.get("product") == product or s.get("ticker") == product:
                    raw_score = s.get("rawScore")
                    break

            outcomes[str(h)].append({
                "ticker":product,"entryPrice":entry_price,"outcome":outcome,
                "pnl":pnl,"reason":reason,"rawScore":raw_score
            })

    out_path.write_text(json.dumps({"date":yday,"outcomes":outcomes,
        "tp_pct":TP_PCT*100,"sl_pct":SL_PCT*100,"horizonHours":HORIZON_HOURS,
        "recordedAt":datetime.now(UTC).isoformat()}, indent=2))

    n_files = len(list(OUTCOME_DIR.glob("*.json")))
    status["outcomeDays"] = n_files
    status["daysSinceRetrain"] = status.get("daysSinceRetrain",0) + 1
    save_status(status)
    log.info(f"Outcomes saved. {n_files} days total.")

# ═══════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════
app = FastAPI()

@app.get("/api/health")
def health():
    active_tp = status.get("activeTP", TP_PCT*100)
    active_sl = status.get("activeSL", SL_PCT*100)
    return {
        "status":"ok","hasCredentials":has_creds(),"marketOpen":market_open(),
        "currentHourUTC":hour_utc(),"currentHourET":hour_utc(),  # alias for UI compat
        "trained":status.get("trained",False),"trainDate":status.get("trainDate"),
        "outcomeDays":status.get("outcomeDays",0),
        "daysSinceRetrain":status.get("daysSinceRetrain",0),
        "modelsLoaded":list(models.keys()),
        "hasLastScan":bool(last_scans),
        "lastScanHours":list(last_scans.keys()),
        "tp_pct":active_tp,"sl_pct":active_sl,
        "breakeven":round(active_sl/(active_sl+active_tp)*100,1) if active_tp>0 else 50.0,
        "horizonHours":HORIZON_HOURS,
        "universeSize":len(PRODUCTS),
        "benchmark":BENCHMARK
    }

@app.get("/api/scan/{hour}")
def get_scan(hour: int):
    if hour not in SCAN_HOURS: return JSONResponse({"error":"Invalid"},400)
    if hour in models:
        try: return run_live_scan(hour)
        except Exception as e: log.error(f"Scan: {e}")
    cached = last_scans.get(str(hour))
    if cached: return {**cached,"source":"cached"}
    return {"data":[],"source":"offline","timestamp":datetime.now(UTC).isoformat(),
            "message":"Train model first, then scan."}

@app.post("/api/scan/{hour}/refresh")
def refresh(hour: int):
    if hour not in SCAN_HOURS: return JSONResponse({"error":"Invalid"},400)
    if hour not in models: return JSONResponse({"error":"No model"},400)
    return run_live_scan(hour)

class TrainRequest(BaseModel):
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None

@app.post("/api/train")
def trigger_train(bg: BackgroundTasks, req: Optional[TrainRequest] = None):
    if training_in_progress: return {"status":"already_running"}
    tp_pct = req.tp_pct if req and req.tp_pct is not None else None
    sl_pct = req.sl_pct if req and req.sl_pct is not None else None
    tp = tp_pct / 100.0 if tp_pct is not None else None
    sl = sl_pct / 100.0 if sl_pct is not None else None
    if tp is not None and not (0.002 <= tp <= 0.10): return JSONResponse({"error":"tp_pct must be 0.2-10.0"},400)
    if sl is not None and not (0.002 <= sl <= 0.10): return JSONResponse({"error":"sl_pct must be 0.2-10.0"},400)
    bg.add_task(run_training, tp, sl)
    return {"status":"started","tp_pct":tp_pct or TP_PCT*100,"sl_pct":sl_pct or SL_PCT*100}

@app.post("/api/cache/clear")
def clear_cache():
    if training_in_progress: return JSONResponse({"error":"Cannot clear during training"},400)
    deleted = []
    for f in [BARS_DAILY_CACHE, BARS_INTRADAY_CACHE]:
        if f.exists():
            f.unlink()
            deleted.append(f.name)
    return {"status":"ok","deleted":deleted}

@app.get("/api/cache/status")
def cache_status():
    return {
        "daily": {"exists":BARS_DAILY_CACHE.exists(),"age_hours":round(cache_age_hours(BARS_DAILY_CACHE),1)},
        "intraday": {"exists":BARS_INTRADAY_CACHE.exists(),"age_hours":round(cache_age_hours(BARS_INTRADAY_CACHE),1)},
        "max_age_hours":CACHE_MAX_AGE_HOURS
    }

@app.get("/api/training/progress")
def progress():
    return {"inProgress":training_in_progress,**training_progress,
            "meta":{str(h):model_meta[h] for h in model_meta}}

@app.post("/api/sweep")
def trigger_sweep(bg: BackgroundTasks):
    if sweep_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress; wait for it to finish"},400)
    bg.add_task(run_sweep, True)
    total = len(SWEEP_TP_VALUES) * len(SWEEP_SL_VALUES)
    return {"status":"started","total_cells":total,
            "grid":{"tp":SWEEP_TP_VALUES,"sl":SWEEP_SL_VALUES}}

@app.post("/api/sweep/reset")
def reset_sweep():
    if sweep_in_progress: return JSONResponse({"error":"Cannot reset during sweep"},400)
    if SWEEP_RESULTS_PATH.exists(): SWEEP_RESULTS_PATH.unlink()
    return {"status":"ok"}

@app.get("/api/sweep/status")
def sweep_status():
    return {"inProgress":sweep_in_progress, **sweep_progress,
            "grid":{"tp":SWEEP_TP_VALUES,"sl":SWEEP_SL_VALUES}}

@app.get("/api/sweep/results")
def sweep_results():
    return load_sweep_results()

@app.get("/api/outcomes/summary")
def outcome_summary():
    files = sorted(OUTCOME_DIR.glob("*.json"))
    if not files: return {"totalDays":0,"recent":[]}
    recent = []
    for f in files[-20:]:
        try: d = json.loads(f.read_text())
        except: continue
        hs = {}
        for h in SCAN_HOURS:
            entries = d.get("outcomes",{}).get(str(h),[])
            scored = sorted([e for e in entries if e.get("rawScore") is not None], key=lambda e:-e["rawScore"])
            top10 = scored[:10]
            wins = sum(1 for e in top10 if e["outcome"]==1)
            avg_pnl = np.mean([e["pnl"] for e in top10]) if top10 else 0
            base_wr = np.mean([e["outcome"] for e in entries]) if entries else 0
            reasons = {}
            for e in entries:
                r = e.get("reason","?")
                reasons[r] = reasons.get(r,0)+1
            hs[str(h)] = {"total":len(entries),"top10wins":wins,
                      "top10pnl":round(avg_pnl,3),"baseWR":round(base_wr*100,1),
                      "reasons":reasons}
        recent.append({"date":d["date"],"hours":hs})
    return {"totalDays":len(files),"recent":recent}

@app.get("/api/diagnostic")
def diagnostic():
    outcome_files = sorted(OUTCOME_DIR.glob("*.json"))
    outcomes = []
    for f in outcome_files[-20:]:
        try: d = json.loads(f.read_text())
        except: continue
        hd = {}
        for h in SCAN_HOURS:
            entries = d.get("outcomes",{}).get(str(h),[])
            scored = sorted([e for e in entries if e.get("rawScore") is not None], key=lambda e:-e["rawScore"])
            t10 = scored[:10]
            hd[str(h)] = {
                "totalStocks":len(entries),
                "baseWinRate":round(np.mean([e["outcome"] for e in entries])*100,1) if entries else None,
                "top10":[{"ticker":e["ticker"],"score":e["rawScore"],"outcome":e["outcome"],"pnl":e["pnl"],"reason":e["reason"]} for e in t10],
                "top10wins":sum(1 for e in t10 if e["outcome"]==1),
                "top10pnl":round(np.mean([e["pnl"] for e in t10]),3) if t10 else 0,
                "reasons":{r:sum(1 for e in entries if e.get("reason")==r) for r in set(e.get("reason","?") for e in entries)}
            }
        outcomes.append({"date":d["date"],"hours":hd})

    scans = {}
    for h_str, scan in last_scans.items():
        scans[h_str] = {
            "timestamp":scan.get("timestamp"),"source":scan.get("source"),
            "scoreRange":scan.get("scoreRange"),
            "top20":(scan.get("data") or [])[:20]
        }

    return JSONResponse({
        "_type":"coinbase_scanner_diagnostic","_version":"1.0_crypto_first_passage",
        "generatedAt":datetime.now(UTC).isoformat(),
        "strategy":{"tp_pct":TP_PCT*100,"sl_pct":SL_PCT*100,
                    "horizon_hours":HORIZON_HOURS,"entry_delay":"1 bar"},
        "server":{
            "hasCredentials":has_creds(),"marketOpen":market_open(),"currentHourUTC":hour_utc(),
            "trained":status.get("trained",False),"trainDate":status.get("trainDate"),
            "outcomeDays":status.get("outcomeDays",0),"daysSinceRetrain":status.get("daysSinceRetrain",0)
        },
        "modelMeta":{str(h):model_meta[h] for h in model_meta},
        "lastScans":scans,
        "outcomes":outcomes,
        "outcomeSummary":{"totalDays":len(outcome_files),
            "dateRange":{"first":outcome_files[0].stem,"last":outcome_files[-1].stem} if outcome_files else None}
    }, headers={"Content-Disposition":f'attachment; filename="coinbase_diagnostic_{today_utc()}.json"'})

# ═══════════════════════════════════════════════════════════════════
# v2 — Vol-normalized threshold classifier (Stage 1)
# ═══════════════════════════════════════════════════════════════════
# v1 endpoints above remain for backward-compatibility only and should not be
# used to draw conclusions. The productive work lives under /api/v2/ below.
import v2 as v2_module

V2_MODEL_DIR = DATA_DIR / "models_v2"
V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)

v2_training_in_progress = False
v2_training_progress = {"phase":"idle","pct":0,"message":"","cell":None}

def _v2_progress_cb(pct, msg):
    global v2_training_progress
    v2_training_progress = {**v2_training_progress, "pct":pct, "message":msg}

def _v2_run_train(k_atr, horizon_hours):
    """Background task wrapper for v2 cell training."""
    global v2_training_in_progress, v2_training_progress
    if v2_training_in_progress:
        log.warning("v2 training already in progress, skipping")
        return
    v2_training_in_progress = True
    v2_training_progress = {"phase":"starting","pct":0,
        "message":f"Starting cell k={k_atr}, H={horizon_hours}h",
        "cell":{"k_atr":k_atr,"horizon_hours":horizon_hours}}
    try:
        # Reuse v1's cached intraday & daily bars if available
        daily_age = cache_age_hours(BARS_DAILY_CACHE)
        intra_age = cache_age_hours(BARS_INTRADAY_CACHE)
        cache_fresh = daily_age < CACHE_MAX_AGE_HOURS and intra_age < CACHE_MAX_AGE_HOURS
        if cache_fresh:
            _v2_progress_cb(2, f"Loading cached bars (age {intra_age:.1f}h)...")
            log.info(f"v2 using cached bars (daily {daily_age:.1f}h, intra {intra_age:.1f}h)")
            daily = pickle.loads(BARS_DAILY_CACHE.read_bytes())
            intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())
        else:
            _v2_progress_cb(2, "No fresh cache — fetching from Coinbase...")
            log.info("v2 fetching fresh bars")
            client = cb_client()
            end_dt = now_utc().replace(minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=TRAIN_DAYS)
            fetch_products = list(set(PRODUCTS + [BENCHMARK]))
            daily = fetch_candles_bulk(client, fetch_products, start_dt, end_dt, 86400)
            _v2_progress_cb(4, f"Fetching {TRAIN_DAYS}d of 15-min bars...")
            intraday = fetch_candles_bulk(
                client, fetch_products, start_dt, end_dt, CANDLE_GRANULARITY)
            client.close()
            BARS_DAILY_CACHE.write_bytes(pickle.dumps(daily))
            BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(intraday))
            log.info(f"v2 cached {sum(len(v) for v in intraday.values())} intraday bars")

        v2_training_progress["phase"] = "training"
        meta = v2_module.run_train_cell_v2(
            k_atr=k_atr,
            horizon_hours=horizon_hours,
            intraday_bars=intraday,
            daily_bars=daily,
            products=PRODUCTS,
            categories=CATEGORIES,
            benchmark=BENCHMARK,
            model_dir=V2_MODEL_DIR,
            scan_hours=SCAN_HOURS,
            progress_cb=_v2_progress_cb,
        )
        v2_training_progress = {"phase":"done","pct":100,
            "message":f"Done. k={k_atr}, H={horizon_hours}h — "
                      f"AUC {meta['auc_test']}, "
                      f"prec@0.75 {meta['precision_at_threshold']['0.75']['precision']} "
                      f"(n={meta['precision_at_threshold']['0.75']['n_predictions']})",
            "cell":{"k_atr":k_atr,"horizon_hours":horizon_hours},
            "meta": meta}
    except Exception as e:
        log.exception(f"v2 training failed: {e}")
        v2_training_progress = {"phase":"error","pct":0,
            "message":f"Error: {e}",
            "cell":{"k_atr":k_atr,"horizon_hours":horizon_hours}}
    finally:
        v2_training_in_progress = False

class V2TrainRequest(BaseModel):
    k_atr: float
    horizon_hours: int

@app.post("/api/v2/train")
def v2_trigger_train(bg: BackgroundTasks, req: V2TrainRequest):
    """Train one (k_atr, horizon_hours) cell.
    Body: {"k_atr": 1.0, "horizon_hours": 4}"""
    if v2_training_in_progress:
        return JSONResponse({"status":"already_running",
            "progress":v2_training_progress}, 409)
    if not (0.1 <= req.k_atr <= 10.0):
        return JSONResponse({"error":"k_atr must be 0.1-10.0"}, 400)
    if not (1 <= req.horizon_hours <= 72):
        return JSONResponse({"error":"horizon_hours must be 1-72"}, 400)
    bg.add_task(_v2_run_train, req.k_atr, req.horizon_hours)
    return {"status":"started","k_atr":req.k_atr,"horizon_hours":req.horizon_hours}

@app.get("/api/v2/training/progress")
def v2_get_progress():
    return {"inProgress": v2_training_in_progress, **v2_training_progress}

@app.get("/api/v2/models")
def v2_list_models():
    """List all trained v2 cells with their summary metrics."""
    cells = []
    for meta_file in sorted(V2_MODEL_DIR.glob("v2_*_meta.json")):
        try:
            m = json.loads(meta_file.read_text())
            cells.append({
                "k_atr": m.get("k_atr"),
                "horizon_hours": m.get("horizon_hours"),
                "trained_at": m.get("trained_at"),
                "auc_test": m.get("auc_test"),
                "base_rate_test": m.get("base_rate_test"),
                "precision_at_0_75": m.get("precision_at_threshold", {}).get("0.75", {}).get("precision"),
                "n_at_0_75": m.get("precision_at_threshold", {}).get("0.75", {}).get("n_predictions"),
                "per_day_at_0_75": m.get("precision_at_threshold", {}).get("0.75", {}).get("avg_per_day"),
                "best_iteration": m.get("best_iteration"),
            })
        except Exception as e:
            log.warning(f"v2 models list: skipping {meta_file.name}: {e}")
    return {"models": cells, "count": len(cells)}

@app.get("/api/v2/model/{k_atr}/{horizon_hours}")
def v2_get_model(k_atr: float, horizon_hours: int):
    """Full metadata for one cell."""
    cell_key = f"k{k_atr:g}_h{horizon_hours}"
    meta_file = V2_MODEL_DIR / f"v2_{cell_key}_meta.json"
    if not meta_file.exists():
        return JSONResponse({"error":"no such cell","cell_key":cell_key}, 404)
    return JSONResponse(json.loads(meta_file.read_text()))

@app.get("/api/v2/diagnostic")
def v2_diagnostic():
    """All v2 models + their full metadata in one downloadable JSON."""
    cells = {}
    for meta_file in sorted(V2_MODEL_DIR.glob("v2_*_meta.json")):
        try:
            m = json.loads(meta_file.read_text())
            cell_key = f"k{m['k_atr']:g}_h{m['horizon_hours']}"
            cells[cell_key] = m
        except:
            continue
    return JSONResponse({
        "_type": "coinbase_scanner_v2_diagnostic",
        "_version": "v2.0_stage1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "universe_size": len(PRODUCTS),
        "benchmark": BENCHMARK,
        "n_features": len(v2_module.FEATURE_NAMES_V2),
        "feature_names": v2_module.FEATURE_NAMES_V2,
        "cells": cells,
    }, headers={"Content-Disposition":
        f'attachment; filename="coinbase_v2_diagnostic_{today_utc()}.json"'})

# ═══════════════════════════════════════════════════════════════════
# v2 STAGE 2 — Rule mining endpoints
# ═══════════════════════════════════════════════════════════════════
import v2_rules as v2_rules_module

V2_RULES_DIR = DATA_DIR / "rules_v2"
V2_RULES_DIR.mkdir(parents=True, exist_ok=True)

v2_mining_in_progress = False
v2_mining_progress = {"phase":"idle","pct":0,"message":"","params":None}
v2_mining_last_result = None  # Last finished mining run summary

def _v2_rules_progress_cb(pct, msg):
    global v2_mining_progress
    v2_mining_progress = {**v2_mining_progress, "pct":pct, "message":msg}

def _v2_run_mining(threshold_pct, horizon_hours_list, methods,
                     min_precision, min_support, min_lift):
    """
    Background task: mine rules across specified horizons for one threshold.

    Implementation notes:
    - All horizons share the SAME feature matrix; only the label column changes.
      build_rows_for_rule_mining() emits one dict per (date, scan_hour, coin) row
      with columns label_4h, label_6h, label_8h (one per horizon), plus
      is_train/is_val/is_test split indicators.
    - Each horizon gets its own rule catalog mined independently on its label
      (three parallel universes, per user choice).
    - After mining each horizon, every rule is cross-horizon retested against
      the OTHER horizons' labels on the SAME rows.
    """
    global v2_mining_in_progress, v2_mining_progress, v2_mining_last_result

    if v2_mining_in_progress:
        log.warning("v2 mining already in progress, skipping")
        return
    v2_mining_in_progress = True
    v2_mining_progress = {"phase":"starting","pct":0,
        "message":"Starting rule mining...",
        "params":{"threshold_pct":threshold_pct,
                   "horizon_hours":horizon_hours_list,
                   "methods":list(methods)}}

    try:
        # ── Step 1: Load bars (reuse v1 cache if fresh) ──
        daily_age = cache_age_hours(BARS_DAILY_CACHE)
        intra_age = cache_age_hours(BARS_INTRADAY_CACHE)
        if not (daily_age < CACHE_MAX_AGE_HOURS and intra_age < CACHE_MAX_AGE_HOURS):
            _v2_rules_progress_cb(2, "Fetching fresh bars from Coinbase...")
            client = cb_client()
            end_dt = now_utc().replace(minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=TRAIN_DAYS)
            fetch_products = list(set(PRODUCTS + [BENCHMARK]))
            daily = fetch_candles_bulk(client, fetch_products, start_dt, end_dt, 86400)
            intraday = fetch_candles_bulk(
                client, fetch_products, start_dt, end_dt, CANDLE_GRANULARITY)
            client.close()
            BARS_DAILY_CACHE.write_bytes(pickle.dumps(daily))
            BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(intraday))
        else:
            _v2_rules_progress_cb(2, f"Loading cached bars (age {intra_age:.1f}h)...")
            daily = pickle.loads(BARS_DAILY_CACHE.read_bytes())
            intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        # ── Step 2: Build ONE row dataset with labels for all horizons ──
        def row_progress(p, m):
            # Rows phase: 5%..40% of overall progress
            _v2_rules_progress_cb(5 + int(p * 0.35), f"[rows] {m}")

        rows, split_info = v2_module.build_rows_for_rule_mining(
            intraday_bars=intraday,
            daily_bars=daily,
            products=PRODUCTS,
            categories=CATEGORIES,
            benchmark=BENCHMARK,
            scan_hours=SCAN_HOURS,
            horizon_hours_list=horizon_hours_list,
            pct_threshold=threshold_pct,
            progress_cb=row_progress,
        )
        log.info(f"Rows built: {len(rows)} total. "
                 f"train={split_info['n_train']} val={split_info['n_val']} test={split_info['n_test']}")

        # ── Step 3: Mine rules per horizon ──
        # feature_names to mine over: all v2 features EXCEPT those in EXCLUDE_FROM_MINING.
        # We pass the full list and v2_rules filters inside build_binned_dataframe.
        feature_names = list(v2_module.FEATURE_NAMES_V2)

        all_catalogs = []
        n_horizons = len(horizon_hours_list)
        for hi, hh in enumerate(horizon_hours_list):
            def mine_progress(p, m, _hi=hi, _hh=hh):
                base = 40 + int((_hi / n_horizons) * 55)
                scale = 55 / n_horizons
                _v2_rules_progress_cb(base + int(p * scale / 100),
                                       f"[h={_hh}] {m}")

            winner_col = f"label_{hh}h"
            _v2_rules_progress_cb(40 + int((hi / n_horizons) * 55),
                                   f"[h={hh}] Mining rules...")

            result = v2_rules_module.mine_all_for_cell(
                rows=rows,
                feature_names=feature_names,
                winner_column=winner_col,
                split_cols={"train": "is_train", "val": "is_val", "test": "is_test"},
                min_precision=min_precision,
                min_support_frac=max(0.001, min_support / max(1, split_info["n_train"])),
                min_lift=min_lift,
                methods=tuple(methods),
                progress_cb=mine_progress,
            )

            # Cross-horizon retest: evaluate every rule against every OTHER horizon
            # using the same rows but a different label column. Build the binned
            # feature matrix ONCE (identical across horizons) and reuse it for
            # all rules and all other-horizon labels.
            import pandas as _pd
            import numpy as _np
            df_for_retest = _pd.DataFrame(rows)
            binned_df_rt, _, _ = v2_rules_module.build_binned_dataframe(
                df_for_retest, feature_names,
                df_for_retest["is_train"].astype(bool).values)
            test_mask_rt = df_for_retest["is_test"].astype(bool).values
            for r in result["rules"]:
                r["cross_horizon"] = {}
                mask = v2_rules_module.rule_to_mask(r, binned_df_rt)
                for hh_other in horizon_hours_list:
                    if hh_other == hh: continue
                    other_col = f"label_{hh_other}h"
                    labels_other = df_for_retest[other_col].astype(int).values
                    ev = v2_rules_module.evaluate_rule(mask & test_mask_rt, labels_other)
                    base_rate_other = float(labels_other[test_mask_rt].mean()) \
                                      if test_mask_rt.sum() > 0 else 0.0
                    r["cross_horizon"][f"h_{hh_other}"] = {
                        "precision": round(ev["precision"], 4),
                        "support": ev["support"],
                        "lift_vs_base": round(ev["precision"] - base_rate_other, 4),
                        "base_rate": round(base_rate_other, 4),
                    }

            # Persist catalog to disk
            threshold_bps = int(round(threshold_pct * 10000))
            catalog_path = V2_RULES_DIR / f"rules_t{threshold_bps}_h{hh}.json"
            catalog_data = {
                "threshold_pct": threshold_pct,
                "horizon_hours": hh,
                "label_column": winner_col,
                "mined_at": datetime.now(UTC).isoformat(),
                "methods": list(methods),
                "min_precision": min_precision,
                "min_support": min_support,
                "min_lift": min_lift,
                "base_rates": result["base_rates"],
                "bin_edges": result["bin_edges"],
                "bin_labels": result["bin_labels"],
                "stats": result["stats"],
                "split_info": split_info,
                "rules": result["rules"],
            }
            catalog_path.write_text(json.dumps(catalog_data, indent=2, default=str))
            log.info(f"[h={hh}] Saved {catalog_path.name} ({len(result['rules'])} rules)")

            all_catalogs.append({
                "threshold_pct": threshold_pct,
                "threshold_bps": threshold_bps,
                "horizon_hours": hh,
                "rule_count": len(result["rules"]),
                "filename": catalog_path.name,
                "base_rate_test": round(result["base_rates"]["test"], 4),
            })

        v2_mining_last_result = {
            "threshold_pct": threshold_pct,
            "horizons": horizon_hours_list,
            "catalogs": all_catalogs,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        v2_mining_progress = {"phase":"done","pct":100,
            "message":f"Done. {sum(c['rule_count'] for c in all_catalogs)} rules across "
                      f"{len(all_catalogs)} horizon(s).",
            "params":v2_mining_progress.get("params"),
            "result": v2_mining_last_result}

    except Exception as e:
        log.exception(f"v2 mining failed: {e}")
        v2_mining_progress = {"phase":"error","pct":0,
            "message":f"Error: {e}",
            "params":v2_mining_progress.get("params")}
    finally:
        v2_mining_in_progress = False


class V2MineRequest(BaseModel):
    threshold_pct: float = 0.02        # e.g. 0.02 for +2%
    horizon_hours: list = [4, 6, 8]
    methods: list = ["univariate", "tree", "apriori"]
    min_precision: float = 0.65
    min_support: int = 30
    min_lift: float = 0.03

@app.post("/api/v2/mine_rules")
def v2_mine_rules(bg: BackgroundTasks, req: V2MineRequest):
    if v2_mining_in_progress:
        return JSONResponse({"status":"already_running",
            "progress":v2_mining_progress}, 409)
    if v2_training_in_progress:
        return JSONResponse({"error":"v2 training in progress — wait for it"}, 409)

    # Validation
    if not (0.005 <= req.threshold_pct <= 0.20):
        return JSONResponse({"error":"threshold_pct must be 0.005-0.20"}, 400)
    if not req.horizon_hours or not all(1 <= h <= 72 for h in req.horizon_hours):
        return JSONResponse({"error":"horizon_hours must be list of 1-72"}, 400)
    valid_methods = {"univariate","tree","apriori"}
    if not req.methods or not set(req.methods).issubset(valid_methods):
        return JSONResponse({"error":f"methods must be subset of {valid_methods}"}, 400)
    if not (0.3 <= req.min_precision <= 0.99):
        return JSONResponse({"error":"min_precision must be 0.3-0.99"}, 400)
    if not (1 <= req.min_support <= 10000):
        return JSONResponse({"error":"min_support must be 1-10000"}, 400)
    if not (0.0 <= req.min_lift <= 0.5):
        return JSONResponse({"error":"min_lift must be 0.0-0.5"}, 400)

    bg.add_task(_v2_run_mining,
                req.threshold_pct, req.horizon_hours, req.methods,
                req.min_precision, req.min_support, req.min_lift)
    return {"status":"started",
            "threshold_pct":req.threshold_pct,
            "horizon_hours":req.horizon_hours,
            "methods":req.methods}

@app.get("/api/v2/mine_rules/progress")
def v2_mining_progress_get():
    return {"inProgress": v2_mining_in_progress, **v2_mining_progress}

@app.get("/api/v2/rules/catalogs")
def v2_list_catalogs():
    """List all rule catalogs (one per threshold,horizon pair)."""
    catalogs = []
    for p in sorted(V2_RULES_DIR.glob("rules_t*_h*.json")):
        try:
            d = json.loads(p.read_text())
            catalogs.append({
                "threshold_pct": d.get("threshold_pct"),
                "threshold_bps": int(round(d.get("threshold_pct", 0) * 10000)),
                "horizon_hours": d.get("horizon_hours"),
                "rule_count": len(d.get("rules", [])),
                "mined_at": d.get("mined_at"),
                "filename": p.name,
                "base_rate_test": round(d.get("base_rates", {}).get("test", 0), 4),
                "methods": d.get("methods", []),
            })
        except Exception as e:
            log.warning(f"catalog list: skipping {p.name}: {e}")
    return {"catalogs": catalogs}

@app.get("/api/v2/rules/catalog/{threshold_bps}/{horizon_hours}")
def v2_get_catalog(threshold_bps: int, horizon_hours: int):
    """
    Get a full catalog. threshold_bps = threshold in basis-points*10
    (e.g. 200 = 2.00%, 150 = 1.50%). Integer URL keys avoid float routing
    issues. Returns everything: rules, bin_edges, bin_labels, stats.
    """
    path = V2_RULES_DIR / f"rules_t{threshold_bps}_h{horizon_hours}.json"
    if not path.exists():
        return JSONResponse({"error":"no catalog for this threshold/horizon",
                              "path": path.name}, 404)
    try:
        return JSONResponse(json.loads(path.read_text()))
    except Exception as e:
        return JSONResponse({"error":f"catalog read failed: {e}"}, 500)

@app.get("/api/v2/rules/rule/{threshold_bps}/{horizon_hours}/{rule_id}")
def v2_get_rule(threshold_bps: int, horizon_hours: int, rule_id: str):
    """Full detail of a single rule by its ID."""
    path = V2_RULES_DIR / f"rules_t{threshold_bps}_h{horizon_hours}.json"
    if not path.exists():
        return JSONResponse({"error":"no catalog"}, 404)
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return JSONResponse({"error":f"catalog read failed: {e}"}, 500)
    for r in d.get("rules", []):
        if r.get("id") == rule_id:
            # Attach the bin_labels map so the UI can render conditions nicely
            return JSONResponse({"rule": r, "bin_labels": d.get("bin_labels", {}),
                                  "bin_edges": d.get("bin_edges", {}),
                                  "base_rates": d.get("base_rates", {}),
                                  "threshold_pct": d.get("threshold_pct"),
                                  "horizon_hours": d.get("horizon_hours")})
    return JSONResponse({"error":"rule not found","rule_id":rule_id}, 404)

@app.delete("/api/v2/rules/catalog/{threshold_bps}/{horizon_hours}")
def v2_delete_catalog(threshold_bps: int, horizon_hours: int):
    """Delete a single catalog file — useful when you want to re-mine fresh."""
    path = V2_RULES_DIR / f"rules_t{threshold_bps}_h{horizon_hours}.json"
    if not path.exists():
        return JSONResponse({"error":"no such catalog"}, 404)
    path.unlink()
    return {"deleted": path.name}

@app.get("/api/v2/rules/diagnostic")
def v2_rules_diagnostic():
    """All catalogs in one downloadable JSON."""
    catalogs = []
    for p in sorted(V2_RULES_DIR.glob("rules_t*_h*.json")):
        try:
            catalogs.append(json.loads(p.read_text()))
        except Exception as e:
            log.warning(f"rules diagnostic: skipping {p.name}: {e}")
    return JSONResponse({
        "_type": "coinbase_scanner_v2_rules_diagnostic",
        "_version": "v2.0_stage2",
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_size": len(PRODUCTS),
        "benchmark": BENCHMARK,
        "n_features": len(v2_module.FEATURE_NAMES_V2),
        "feature_names": v2_module.FEATURE_NAMES_V2,
        "catalogs": catalogs,
    }, headers={"Content-Disposition":
        f'attachment; filename="coinbase_v2_rules_diagnostic_{today_utc()}.json"'})

@app.get("/api/v2/rules/debug_disqualifier/{threshold_bps}/{horizon_hours}")
def v2_debug_disqualifier(threshold_bps: int, horizon_hours: int, rule_index: int = 0):
    """
    Diagnostic endpoint: runs disqualifier_analysis step by step on one rule of
    an existing catalog and returns every intermediate value so we can see
    where it's returning empty. Also calls the real function at the end and
    reports how many disqualifiers it found.

    Usage: curl /api/v2/rules/debug_disqualifier/200/8
           curl /api/v2/rules/debug_disqualifier/200/8?rule_index=0
    """
    path = V2_RULES_DIR / f"rules_t{threshold_bps}_h{horizon_hours}.json"
    if not path.exists():
        return JSONResponse({"error": f"no catalog at {path.name}"}, 404)

    try:
        cat = json.loads(path.read_text())
        rules = cat.get("rules", [])
        if rule_index >= len(rules):
            return JSONResponse({"error": f"rule_index {rule_index} out of range ({len(rules)} rules)"}, 400)
        rule = rules[rule_index]

        # Reload cached bars
        if not BARS_INTRADAY_CACHE.exists() or not BARS_DAILY_CACHE.exists():
            return JSONResponse({"error": "no cached bars on disk"}, 500)
        intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())
        daily = pickle.loads(BARS_DAILY_CACHE.read_bytes())

        # Rebuild rows for this horizon only
        threshold_pct = threshold_bps / 10000.0
        rows, split_info = v2_module.build_rows_for_rule_mining(
            intraday_bars=intraday, daily_bars=daily,
            products=PRODUCTS, categories=CATEGORIES, benchmark=BENCHMARK,
            scan_hours=SCAN_HOURS, horizon_hours_list=[horizon_hours],
            pct_threshold=threshold_pct,
        )

        import pandas as pd
        df = pd.DataFrame(rows)
        feature_names = list(v2_module.FEATURE_NAMES_V2)
        binned_df, _, _ = v2_rules_module.build_binned_dataframe(
            df, feature_names, df["is_train"].astype(bool).values)

        label_col = f"label_{horizon_hours}h"
        labels = df[label_col].astype(int).values
        train_mask = df["is_train"].astype(bool).values

        # Step through disqualifier logic
        rule_mask = v2_rules_module.rule_to_mask(rule, binned_df)
        combined_mask = rule_mask & train_mask
        fires_labels = labels[combined_mask]
        fires_df = df[combined_mask].reset_index(drop=True)
        tp_mask = fires_labels == 1
        fp_mask = fires_labels == 0
        n_tp = int(tp_mask.sum())
        n_fp = int(fp_mask.sum())

        # Test one arbitrary continuous feature not in the rule
        rule_feats = {c["feature"] for c in rule["conditions"]}
        cont_features = [f for f in feature_names
                          if f not in v2_rules_module.BINARY_FEATURES
                          and f not in v2_rules_module.EXCLUDE_FROM_MINING
                          and f not in rule_feats]

        sample_results = []
        for test_feat in cont_features[:5]:
            if test_feat not in fires_df.columns:
                sample_results.append({"feature": test_feat, "status": "NOT_IN_COLUMNS"})
                continue
            try:
                tp_vals = fires_df[test_feat].iloc[tp_mask].dropna()
                fp_vals = fires_df[test_feat].iloc[fp_mask].dropna()
                sample_results.append({
                    "feature": test_feat,
                    "tp_count": len(tp_vals),
                    "fp_count": len(fp_vals),
                    "tp_mean": float(tp_vals.mean()) if len(tp_vals) else None,
                    "fp_mean": float(fp_vals.mean()) if len(fp_vals) else None,
                    "tp_first_3": tp_vals.head(3).tolist() if len(tp_vals) else [],
                    "fp_first_3": fp_vals.head(3).tolist() if len(fp_vals) else [],
                })
            except Exception as e:
                sample_results.append({"feature": test_feat, "error": str(e)})

        # Actually call the real function
        actual = v2_rules_module.disqualifier_analysis(
            rule, binned_df, df, labels, train_mask, cont_features)

        return JSONResponse({
            "rule": {
                "id": rule.get("id"),
                "english": rule.get("english"),
                "conditions": rule.get("conditions"),
                "train_support_claimed": rule["train"]["support"],
                "train_precision_claimed": rule["train"]["precision"],
            },
            "rebuilt_data": {
                "n_rows_rebuilt": len(df),
                "n_train": int(train_mask.sum()),
                "label_mean": float(labels.mean()),
                "n_feature_cols_binned": len(binned_df.columns),
                "fires_df_dtypes_sample": {f: str(fires_df[f].dtype) for f in list(fires_df.columns)[:5]},
            },
            "rule_application": {
                "rule_fires_total": int(rule_mask.sum()),
                "rule_fires_on_train": int(combined_mask.sum()),
                "tp_count": n_tp,
                "fp_count": n_fp,
                "early_exit_lt_20": int(combined_mask.sum()) < 20,
                "early_exit_lt_5_each": n_tp < 5 or n_fp < 5,
            },
            "candidate_features": {
                "total_continuous": len(cont_features),
                "first_5_tested": sample_results,
            },
            "disqualifier_function_result": {
                "count_returned": len(actual),
                "first_3": actual[:3] if actual else [],
            },
        })
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, 500)

# ═══════════════════════════════════════════════════════════════════
# v2 STAGE 3 — Live scanner and outcome recording
# ═══════════════════════════════════════════════════════════════════
# The Stage 3 flow:
#   1. User pins rules from validated catalogs (with optional disqualifier)
#   2. A scheduled job runs every 4 hours: compute v2 features for all coins,
#      evaluate pinned rules, persist fires to disk
#   3. Another scheduled job runs hourly: for each unresolved fire whose horizon
#      has elapsed, fetch post-bars and determine hit/miss
#   4. Dashboard: live precision per rule vs validation precision
#
# Live scan uses FRESH Coinbase data (not cached), because the point is to
# validate forward outcomes. It takes ~20-30 sec since it only fetches enough
# history to compute features (~48 bars = 12h × 15min) per coin.

import v2_live as v2_live_module

V2_LIVE_DIR = DATA_DIR / "live_v2"
V2_LIVE_DIR.mkdir(parents=True, exist_ok=True)

# Lock to prevent concurrent live scans (they'd duplicate fires)
v2_live_scan_in_progress = False
v2_live_scan_last_result = None

def _v2_compute_live_features(intraday_by_product, daily_by_product,
                                scan_time):
    """
    For each coin in PRODUCTS, compute the 53 v2 features using the most
    recent bars. Returns {product: {feature: value}} and {product: entry_price}.

    Matches how v2 features were computed during mining, except:
      - scan_hour is derived from scan_time.hour
      - "now" is the scan time; all bars with timestamp < scan_time are history
      - The entry_price for outcome tracking is the most recent bar's close
        (the price the user would realistically enter at)
    """
    btc_bars = sorted(intraday_by_product.get(BENCHMARK, []), key=lambda b: b["t"])
    if len(btc_bars) < 50:
        log.error(f"live scan: insufficient BTC bars ({len(btc_bars)})")
        return {}, {}
    # Keep only bars BEFORE scan_time (strict).
    scan_ts = scan_time.isoformat().replace("+00:00", "Z")
    btc_bars = [b for b in btc_bars if b["t"] < scan_ts]
    if len(btc_bars) < 50:
        log.error(f"live scan: insufficient BTC bars after filter ({len(btc_bars)})")
        return {}, {}

    btc_atr = v2_module.compute_atr_fraction(btc_bars)
    btc_ctx = v2_module.compute_btc_context_v2(btc_bars[-48:], btc_atr)

    scan_hour = scan_time.hour

    # Collect features per coin, preparing for cross-sectional ranks
    date_feats, date_meta, date_cats, date_dv = [], [], [], []
    entry_prices = {}

    for product in PRODUCTS:
        bars = sorted(intraday_by_product.get(product, []), key=lambda b: b["t"])
        bars_before = [b for b in bars if b["t"] < scan_ts]
        if len(bars_before) < v2_module.MIN_BARS_FOR_FEATURES:
            continue
        feat_bars = bars_before[-v2_module.FEATURE_LOOKBACK_BARS:]
        atr_frac = v2_module.compute_atr_fraction(bars_before)
        current_price = feat_bars[-1]["c"]
        # Open of the scan day's first bar
        scan_date = scan_time.date().isoformat()
        today_bars = [b for b in bars_before if b["t"][:10] == scan_date]
        open_price = today_bars[0]["o"] if today_bars else current_price

        feat = v2_module.compute_features_v2(
            feat_bars, daily_by_product.get(product, []),
            current_price, open_price, scan_hour,
            btc_bars_before=btc_bars[-50:], btc_context=btc_ctx,
            atr_frac=atr_frac,
        )
        if feat is None: continue
        date_feats.append(feat)
        date_meta.append({"product": product})
        date_cats.append(CATEGORIES.get(product, "?"))
        date_dv.append(sum(b["v"] * b["c"] for b in feat_bars[-20:]))
        entry_prices[product] = float(current_price)

    if len(date_feats) < 3:
        log.warning(f"live scan: only {len(date_feats)} coins with usable features")
        return {}, {}

    v2_module.add_cross_sectional_features_v2(date_feats, date_cats, date_dv)

    features_by_coin = {}
    for i, meta in enumerate(date_meta):
        features_by_coin[meta["product"]] = date_feats[i]
    return features_by_coin, entry_prices


def _v2_get_post_bars(product, start_dt, n_bars):
    """
    For outcome recording: fetch intraday bars for `product` starting at
    start_dt, as many as needed to cover n_bars (15-min bars).
    Called by v2_live.record_outcomes after a fire's horizon has elapsed.

    Strategy: use cached intraday bars if fresh enough; otherwise fetch just
    for this one product over the needed time window.
    """
    try:
        # Try cache first — it might already have what we need
        intra_age = cache_age_hours(BARS_INTRADAY_CACHE)
        if intra_age < CACHE_MAX_AGE_HOURS:
            cached = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())
            bars = cached.get(product, [])
            start_ts = start_dt.isoformat().replace("+00:00", "Z")
            bars_after = sorted([b for b in bars if b["t"] > start_ts],
                                 key=lambda b: b["t"])
            if len(bars_after) >= n_bars:
                return bars_after[:n_bars]
        # Fall through to fresh fetch
        client = cb_client()
        # Fetch window: start_dt to start_dt + n_bars * 15min + 1h buffer
        end_dt = start_dt + timedelta(minutes=n_bars * 15 + 60)
        bars = fetch_candles_for_product(client, product, start_dt, end_dt,
                                          CANDLE_GRANULARITY)
        client.close()
        start_ts = start_dt.isoformat().replace("+00:00", "Z")
        bars_after = sorted([b for b in bars if b["t"] > start_ts],
                             key=lambda b: b["t"])
        return bars_after[:n_bars]
    except Exception as e:
        log.warning(f"_v2_get_post_bars({product}): {e}")
        return []


def _v2_run_live_scan():
    """Top-level live-scan function called by scheduler or manual trigger."""
    global v2_live_scan_in_progress, v2_live_scan_last_result
    if v2_live_scan_in_progress:
        log.info("live scan skipped: already in progress")
        return {"status": "skipped", "reason": "already_in_progress"}

    pinned = v2_live_module.load_pinned_rules(V2_LIVE_DIR)
    if not pinned:
        log.info("live scan skipped: no pinned rules")
        return {"status": "skipped", "reason": "no_pinned_rules"}

    v2_live_scan_in_progress = True
    t0 = time.time()
    try:
        log.info(f"v2 live scan: starting with {len(pinned)} pinned rules")
        scan_time = now_utc().replace(second=0, microsecond=0)

        client = cb_client()
        # Fetch intraday bars — need FEATURE_LOOKBACK_BARS = 48 plus enough
        # BTC context history (~50 more). Fetch last 3 days to be safe.
        end_dt = scan_time
        start_dt = end_dt - timedelta(days=3)
        fetch_products = list(set(PRODUCTS + [BENCHMARK]))
        intraday = fetch_candles_bulk(client, fetch_products, start_dt, end_dt,
                                        CANDLE_GRANULARITY)
        # Daily bars for the daily_bars arg — 20 days
        daily_start = end_dt - timedelta(days=20)
        daily = fetch_candles_bulk(client, fetch_products, daily_start, end_dt,
                                     86400)
        client.close()

        features_by_coin, entry_prices = _v2_compute_live_features(
            intraday, daily, scan_time)
        if not features_by_coin:
            v2_live_scan_last_result = {
                "status": "no_features",
                "scan_time": scan_time.isoformat(),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            return v2_live_scan_last_result

        fires = v2_live_module.scan_live(
            V2_LIVE_DIR, features_by_coin, scan_time=scan_time,
            entry_prices=entry_prices)

        elapsed = round(time.time() - t0, 1)
        result = {
            "status": "ok",
            "scan_time": scan_time.isoformat(),
            "n_coins_evaluated": len(features_by_coin),
            "n_pinned_rules": len(pinned),
            "n_fires": len(fires),
            "fires_by_pin": {pin["pin_id"]: sum(1 for f in fires if f["pin_id"] == pin["pin_id"])
                              for pin in pinned},
            "elapsed_sec": elapsed,
        }
        v2_live_scan_last_result = result
        log.info(f"v2 live scan done: {len(fires)} fires in {elapsed}s")
        return result
    except Exception as e:
        log.exception(f"v2 live scan failed: {e}")
        v2_live_scan_last_result = {"status": "error", "error": str(e)}
        return v2_live_scan_last_result
    finally:
        v2_live_scan_in_progress = False


def _v2_run_record_outcomes():
    """Top-level outcome recorder called by scheduler or manual trigger."""
    try:
        result = v2_live_module.record_outcomes(
            V2_LIVE_DIR, _v2_get_post_bars, bars_per_hour=v2_module.BARS_PER_HOUR)
        log.info(f"v2 record_outcomes: {result}")
        return result
    except Exception as e:
        log.exception(f"v2 record_outcomes failed: {e}")
        return {"error": str(e)}


# ═══ Endpoints ═══

class V2PinRuleRequest(BaseModel):
    threshold_bps: int
    horizon_hours: int
    rule_id: str
    disqualifier: dict = None   # optional: {feature, condition, thresh, direction}

@app.post("/api/v2/live/pin")
def v2_live_pin(req: V2PinRuleRequest):
    """Pin a rule from a catalog to the live scanner."""
    path = V2_RULES_DIR / f"rules_t{req.threshold_bps}_h{req.horizon_hours}.json"
    if not path.exists():
        return JSONResponse({"error": f"no catalog at {path.name}"}, 404)
    try:
        cat = json.loads(path.read_text())
        pinned = v2_live_module.pin_rule(V2_LIVE_DIR, cat, req.rule_id,
                                           disqualifier=req.disqualifier)
        return {"status": "pinned", "pin": pinned}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 404)
    except Exception as e:
        log.exception(f"pin failed: {e}")
        return JSONResponse({"error": str(e)}, 500)

@app.delete("/api/v2/live/pin/{pin_id}")
def v2_live_unpin(pin_id: str):
    ok = v2_live_module.unpin_rule(V2_LIVE_DIR, pin_id)
    return {"status": "unpinned" if ok else "not_found", "pin_id": pin_id}

@app.get("/api/v2/live/pinned")
def v2_live_list_pinned():
    return {"rules": v2_live_module.load_pinned_rules(V2_LIVE_DIR)}

@app.post("/api/v2/live/scan")
def v2_live_scan_now(bg: BackgroundTasks):
    """Trigger a live scan manually. Runs in the background."""
    if v2_live_scan_in_progress:
        return JSONResponse({"status": "already_in_progress"}, 409)
    bg.add_task(_v2_run_live_scan)
    return {"status": "started"}

@app.get("/api/v2/live/scan/status")
def v2_live_scan_status():
    return {
        "inProgress": v2_live_scan_in_progress,
        "lastResult": v2_live_scan_last_result,
    }

@app.post("/api/v2/live/record_outcomes")
def v2_live_record_outcomes_now(bg: BackgroundTasks):
    """Trigger outcome resolution manually (usually automatic)."""
    bg.add_task(_v2_run_record_outcomes)
    return {"status": "started"}

@app.get("/api/v2/live/stats")
def v2_live_stats(window_days: int = None):
    """Aggregated live stats per rule."""
    return v2_live_module.aggregate_stats(V2_LIVE_DIR, window_days=window_days)

@app.get("/api/v2/live/fires")
def v2_live_fires(limit: int = 100, pin_id: str = None, only_unresolved: bool = False):
    return {"fires": v2_live_module.list_recent_fires(
        V2_LIVE_DIR, limit=limit, pin_id=pin_id, only_unresolved=only_unresolved)}

@app.get("/api/v2/live/diagnostic")
def v2_live_diagnostic():
    """Everything at once, for download."""
    pinned = v2_live_module.load_pinned_rules(V2_LIVE_DIR)
    stats = v2_live_module.aggregate_stats(V2_LIVE_DIR)
    fires = v2_live_module.list_recent_fires(V2_LIVE_DIR, limit=1000)
    return JSONResponse({
        "_type": "coinbase_scanner_v2_live_diagnostic",
        "_version": "v2.0_stage3",
        "generated_at": datetime.now(UTC).isoformat(),
        "pinned_rules": pinned,
        "stats": stats,
        "recent_fires": fires,
    }, headers={"Content-Disposition":
        f'attachment; filename="coinbase_v2_live_diagnostic_{today_utc()}.json"'})

# SPA fallback
dist_path = Path(__file__).parent / "dist"
if dist_path.exists():
    app.mount("/assets", StaticFiles(directory=dist_path/"assets"), name="assets")
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        fp = dist_path / full_path
        if fp.is_file(): return FileResponse(fp)
        return FileResponse(dist_path / "index.html")

# ═══════════════════════════════════════════════════════════════════
# SCHEDULER — crypto is 24/7 so we run every day, every 4h
# ═══════════════════════════════════════════════════════════════════
scheduler = BackgroundScheduler(timezone=UTC)
def cron_scan():
    h = hour_utc()
    if h in SCAN_HOURS and h in models:
        try: run_live_scan(h)
        except Exception as e: log.error(f"Cron scan: {e}")
# Scan runs 5 min after each slot hour to ensure the slot's opening candle is closed
scheduler.add_job(cron_scan, "cron", hour=",".join(str(h) for h in SCAN_HOURS), minute=5)
# Record outcomes at 00:30 UTC daily — 30min after the 20:00 scan's 4h horizon ends
scheduler.add_job(record_outcomes, "cron", hour=0, minute=30)

# v2 Stage 3 schedule:
# Live scan at the same 6 slots as v1 (4h cadence), at :06 (1 min after v1's :05)
# to avoid clobbering the single-threaded bar fetch.
def v2_cron_live_scan():
    try: _v2_run_live_scan()
    except Exception as e: log.error(f"v2 cron live scan: {e}")
scheduler.add_job(v2_cron_live_scan, "cron",
                   hour=",".join(str(h) for h in SCAN_HOURS), minute=6)
# Outcome resolution runs hourly — cheap (just reads JSONL, pulls bars for
# fires whose horizons have elapsed)
def v2_cron_record_outcomes():
    try: _v2_run_record_outcomes()
    except Exception as e: log.error(f"v2 cron record outcomes: {e}")
scheduler.add_job(v2_cron_record_outcomes, "cron", minute=15)   # every hour at :15

scheduler.start()
log.info(f"Scheduler: v1 scans {SCAN_HOURS} UTC :05, v1 outcomes 00:30 UTC, "
         f"v2 live scans :06, v2 outcomes hourly at :15")
