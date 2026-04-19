"""
v2_live — Stage 3 live scanner and outcome recording for rule-based trading.

The purpose of this module is to answer ONE question: do the rules that look
good in the backtest actually work in real life? It does this by:

1. Letting the user "pin" rules from validated catalogs
2. Evaluating those pinned rules against fresh Coinbase data on a schedule
3. Recording every "fire" (rule triggered on coin X at time T)
4. After the rule's horizon elapses, checking whether +2% was actually hit
5. Surfacing rolling live precision per rule, comparable to validation precision

Key design principles:
  • Append-only JSONL storage for fires and outcomes. Durable, simple, easy to
    back up, easy to export.
  • Every fire is traceable: full rule id, conditions, feature values at fire
    time, entry price. This means we can later debug why a rule fired.
  • Outcomes are resolved via a second pass that pulls post-bars and checks the
    absolute +pct threshold. This is independent of the fire-recording pass, so
    a crash between them doesn't lose data.
  • Pinned rules include a snapshot of their conditions AND disqualifiers.
    That way, even if the catalog JSON is later deleted or overwritten, the
    pinned rule still works.
  • NO NEW FEATURES: rules evaluated using the same bin_edges used during
    mining. If the catalog changes edges, results will diverge. We store the
    edges with the pinned rule to prevent drift.

STORAGE LAYOUT (relative to V2_LIVE_DIR on disk):
  pinned_rules.json           — {rules: [{rule_id, english, conditions,
                                           bin_edges, disqualifier?,
                                           threshold_pct, horizon_hours,
                                           validation_precision, source_catalog, pinned_at}]}
  fires.jsonl                 — one JSON per fire. Fields:
                                  fire_id (uuid4 hex)
                                  rule_id
                                  scan_time (ISO UTC)
                                  product
                                  entry_price
                                  horizon_hours
                                  threshold_pct
                                  matched_conditions (the rule that fired)
                                  feature_values (all 53)
                                  disqualifier_applied (if any; bool + condition)
                                  resolved (bool, set True by outcome recorder)
                                  outcome? (added by recorder: hit/miss, max_pct, time_to_hit)
  outcomes.jsonl              — redundant copy of just the outcomes, for
                                 faster aggregation queries. Fields:
                                  fire_id, rule_id, scan_time, resolve_time,
                                  hit (bool), max_pct, time_to_hit_bars (None if miss)
"""
import os, json, uuid, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("v2_live")
UTC = ZoneInfo("UTC")


# ═══════════════════════════════════════════════════════════════════
# FILE HELPERS — JSONL append, safe read
# ═══════════════════════════════════════════════════════════════════

def _jsonl_append(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file, creating it if needed.
    Uses default=str to handle datetime and numpy types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _jsonl_read(path: Path, limit: int = None, filter_fn=None) -> list:
    """Read a JSONL file into a list of dicts. Optional filter_fn(record) ->
    bool keeps only matching records. limit caps total returned (most-recent
    first if limit is set, though file order is insertion order)."""
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"skipping corrupt line in {path.name}: {e}")
                continue
            if filter_fn is None or filter_fn(rec):
                out.append(rec)
    if limit is not None and limit < len(out):
        out = out[-limit:]
    return out


def _jsonl_update(path: Path, predicate, updater) -> int:
    """Rewrite the JSONL file, applying updater(record) to every record
    matching predicate(record). Returns count of records updated.
    predicate/updater can mutate the dict; updater returns the new dict (or
    None to delete). Not-matching records are preserved unchanged."""
    if not path.exists():
        return 0
    records = _jsonl_read(path)
    updated = 0
    new_records = []
    for rec in records:
        if predicate(rec):
            new_rec = updater(rec)
            if new_rec is not None:
                new_records.append(new_rec)
            updated += 1
        else:
            new_records.append(rec)
    # Atomic rewrite
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in new_records:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.replace(path)
    return updated


# ═══════════════════════════════════════════════════════════════════
# PINNED RULES — JSON dict at a known path
# ═══════════════════════════════════════════════════════════════════

def pinned_rules_path(live_dir: Path) -> Path:
    return live_dir / "pinned_rules.json"


def load_pinned_rules(live_dir: Path) -> list:
    p = pinned_rules_path(live_dir)
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
        return d.get("rules", [])
    except Exception as e:
        log.error(f"load_pinned_rules: {e}")
        return []


def save_pinned_rules(live_dir: Path, rules: list) -> None:
    p = pinned_rules_path(live_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rules": rules}, default=str, indent=2))


def pin_rule(live_dir: Path, catalog_dict: dict, rule_id: str,
              disqualifier: dict = None) -> dict:
    """
    Pin a rule from a catalog. If disqualifier is provided, it becomes part of
    the pinned rule and will be applied at evaluation time.

    disqualifier format: {'feature': str, 'condition': str (readable),
                          'thresh': float, 'direction': 'exclude_if_greater'|'exclude_if_less'}

    Returns the pinned rule dict.
    """
    rule = None
    for r in catalog_dict.get("rules", []):
        if r.get("id") == rule_id:
            rule = r
            break
    if rule is None:
        raise ValueError(f"Rule {rule_id} not in catalog")

    # Combine pin-id from rule_id + disqualifier-signature so user can pin
    # both the refined and the unrefined version of the same rule
    pin_id_suffix = f"_dq_{disqualifier['feature']}_{disqualifier['direction']}" if disqualifier else ""
    pin_id = f"{rule_id}{pin_id_suffix}"

    pinned = {
        "pin_id": pin_id,
        "rule_id": rule_id,
        "english": rule.get("english", ""),
        "conditions": rule["conditions"],
        "bin_edges": catalog_dict.get("bin_edges", {}),
        "bin_labels": catalog_dict.get("bin_labels", {}),
        "threshold_pct": catalog_dict["threshold_pct"],
        "horizon_hours": catalog_dict["horizon_hours"],
        "validation_precision": rule.get("test", {}).get("precision"),
        "validation_support": rule.get("test", {}).get("support"),
        "validation_lift": rule.get("test", {}).get("lift_vs_base"),
        "validation_base_rate": catalog_dict.get("base_rates", {}).get("test"),
        "source_catalog": {
            "threshold_bps": int(round(catalog_dict["threshold_pct"] * 10000)),
            "horizon_hours": catalog_dict["horizon_hours"],
        },
        "disqualifier": disqualifier,  # may be None
        "pinned_at": datetime.now(UTC).isoformat(),
    }
    # Insert or replace
    current = load_pinned_rules(live_dir)
    current = [r for r in current if r["pin_id"] != pin_id]
    current.append(pinned)
    save_pinned_rules(live_dir, current)
    return pinned


def unpin_rule(live_dir: Path, pin_id: str) -> bool:
    current = load_pinned_rules(live_dir)
    new = [r for r in current if r["pin_id"] != pin_id]
    if len(new) == len(current):
        return False
    save_pinned_rules(live_dir, new)
    return True


# ═══════════════════════════════════════════════════════════════════
# RULE EVALUATION — check if a rule fires on a feature row
# ═══════════════════════════════════════════════════════════════════

def _assign_bin(val, edges):
    """Same logic as v2_rules.assign_bin but inlined to avoid circular import."""
    if val is None:
        return -1
    try:
        v = float(val)
    except (TypeError, ValueError):
        return -1
    if not np.isfinite(v):
        return -1
    if edges is None:
        # Binary feature — clamp to {0, 1}
        return int(max(0, min(1, int(round(v)))))
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def evaluate_pinned_rule(pinned: dict, features: dict) -> dict:
    """
    Check whether a pinned rule fires on a feature vector. Returns:
      {"fires": bool,
       "matched_conditions": [list of conditions that matched],
       "disqualifier_applied": bool,
       "reason": str}

    Uses bin_edges stored with the pinned rule — not current catalog's — so
    results are stable even if the source catalog is re-mined.
    """
    bin_edges = pinned.get("bin_edges", {})

    # Step 1: all primary conditions must hold
    for cond in pinned["conditions"]:
        feat = cond["feature"]
        expected_bins = cond["bins"]
        if feat not in features:
            return {"fires": False, "reason": f"feature_missing:{feat}",
                    "matched_conditions": [], "disqualifier_applied": False}
        edges = bin_edges.get(feat)
        # Convert edges list back to numpy array for consistency
        edges_arr = np.array(edges) if edges is not None else None
        actual_bin = _assign_bin(features[feat], edges_arr)
        if actual_bin not in expected_bins:
            return {"fires": False, "reason": f"bin_mismatch:{feat}",
                    "matched_conditions": [], "disqualifier_applied": False}

    # Step 2: if disqualifier is set, check it
    disq = pinned.get("disqualifier")
    disqualifier_excluded = False
    if disq:
        feat = disq["feature"]
        thresh = disq["thresh"]
        direction = disq["direction"]
        if feat in features:
            try:
                v = float(features[feat])
                if np.isfinite(v):
                    # exclude_if_greater: threshold >X -> exclude, so remaining is x <= thresh
                    # In the original analysis: "condition: '<= thresh'" means we KEEP things <= thresh.
                    # But the stored direction semantics are:
                    #   "exclude_if_greater" => exclude when v > thresh => KEEP v <= thresh
                    #   "exclude_if_less"    => exclude when v < thresh => KEEP v >= thresh
                    if direction == "exclude_if_greater" and v > thresh:
                        disqualifier_excluded = True
                    elif direction == "exclude_if_less" and v < thresh:
                        disqualifier_excluded = True
            except (TypeError, ValueError):
                pass

    if disqualifier_excluded:
        return {"fires": False, "reason": "disqualifier_excluded",
                "matched_conditions": pinned["conditions"],
                "disqualifier_applied": True}

    return {"fires": True, "reason": "all_conditions_met",
            "matched_conditions": pinned["conditions"],
            "disqualifier_applied": disq is not None}


# ═══════════════════════════════════════════════════════════════════
# LIVE SCAN — run all pinned rules against current feature snapshot
# ═══════════════════════════════════════════════════════════════════

def scan_live(live_dir: Path, features_by_coin: dict,
               scan_time: datetime = None,
               entry_prices: dict = None) -> list:
    """
    Evaluate all pinned rules against a snapshot of features for each coin.
    Returns a list of fire records (already persisted to fires.jsonl).

    features_by_coin: dict { product: {feature: value, ...} }
    entry_prices: dict { product: float } - price at scan time used as entry reference
    scan_time: defaults to now

    Persists: each fire → fires.jsonl (with resolved=False)
    """
    if scan_time is None:
        scan_time = datetime.now(UTC)
    entry_prices = entry_prices or {}

    pinned = load_pinned_rules(live_dir)
    if not pinned:
        return []

    fires = []
    for product, features in features_by_coin.items():
        for rule in pinned:
            result = evaluate_pinned_rule(rule, features)
            if not result["fires"]:
                continue
            fire = {
                "fire_id": uuid.uuid4().hex,
                "pin_id": rule["pin_id"],
                "rule_id": rule["rule_id"],
                "scan_time": scan_time.isoformat(),
                "product": product,
                "entry_price": entry_prices.get(product),
                "horizon_hours": rule["horizon_hours"],
                "threshold_pct": rule["threshold_pct"],
                "disqualifier_applied": result["disqualifier_applied"],
                "feature_values": features,
                "resolved": False,
                "outcome": None,
            }
            _jsonl_append(live_dir / "fires.jsonl", fire)
            fires.append(fire)
    log.info(f"scan_live: {len(fires)} fires across {len(pinned)} pinned rules "
             f"and {len(features_by_coin)} coins")
    return fires


# ═══════════════════════════════════════════════════════════════════
# OUTCOME RECORDING — resolve fires whose horizon has elapsed
# ═══════════════════════════════════════════════════════════════════

def compute_outcome(entry_price: float, future_bars: list,
                     threshold_pct: float, horizon_bars: int) -> dict:
    """
    Check if price hit entry_price * (1 + threshold_pct) within horizon_bars
    of future bars. Returns:
      {"hit": bool, "max_pct": float, "time_to_hit_bars": int|None,
       "final_pct": float, "bars_used": int}
    """
    target = entry_price * (1.0 + threshold_pct)
    window = future_bars[:horizon_bars]
    if not window:
        return {"hit": False, "max_pct": 0.0, "time_to_hit_bars": None,
                "final_pct": 0.0, "bars_used": 0}

    max_high = entry_price
    time_to_hit = None
    for i, b in enumerate(window):
        if b["h"] > max_high:
            max_high = b["h"]
        if b["h"] >= target and time_to_hit is None:
            time_to_hit = i + 1

    final_close = window[-1]["c"]
    max_pct = (max_high - entry_price) / entry_price
    final_pct = (final_close - entry_price) / entry_price
    return {
        "hit": time_to_hit is not None,
        "max_pct": round(float(max_pct), 4),
        "time_to_hit_bars": time_to_hit,
        "final_pct": round(float(final_pct), 4),
        "bars_used": len(window),
    }


def record_outcomes(live_dir: Path, get_post_bars_fn, now: datetime = None,
                     bars_per_hour: int = 4) -> dict:
    """
    For each unresolved fire whose horizon has elapsed, pull post-fire bars
    and determine whether the coin hit +threshold_pct in horizon_hours.

    get_post_bars_fn(product, start_dt, n_bars) -> list of bar dicts
      expected bar dict: {"t": iso str, "o", "h", "l", "c", "v"}

    now: defaults to UTC now. Fires with scan_time + horizon_hours > now are
    still pending and skipped.

    Returns: {"resolved": int, "pending": int, "errors": int}
    """
    if now is None:
        now = datetime.now(UTC)
    fires_path = live_dir / "fires.jsonl"
    if not fires_path.exists():
        return {"resolved": 0, "pending": 0, "errors": 0}

    resolved_count = 0
    pending_count = 0
    error_count = 0

    # Index fires — we'll rewrite the file atomically
    def predicate(rec): return not rec.get("resolved", False)

    def updater(rec):
        nonlocal resolved_count, pending_count, error_count
        try:
            scan_time = datetime.fromisoformat(rec["scan_time"].replace("Z", "+00:00"))
            if scan_time.tzinfo is None:
                scan_time = scan_time.replace(tzinfo=UTC)
            elapsed = (now - scan_time).total_seconds() / 3600
            if elapsed < rec["horizon_hours"]:
                pending_count += 1
                return rec   # still pending, unchanged
            # Horizon has passed — pull bars and resolve
            horizon_bars = int(rec["horizon_hours"] * bars_per_hour)
            entry_price = rec.get("entry_price")
            if entry_price is None or entry_price <= 0:
                # Can't resolve — mark as unresolvable
                rec["resolved"] = True
                rec["outcome"] = {"error": "missing_entry_price"}
                error_count += 1
                return rec
            post_bars = get_post_bars_fn(rec["product"], scan_time, horizon_bars)
            if not post_bars:
                rec["resolved"] = True
                rec["outcome"] = {"error": "no_post_bars"}
                error_count += 1
                return rec
            outcome = compute_outcome(
                entry_price, post_bars, rec["threshold_pct"], horizon_bars)
            rec["resolved"] = True
            rec["outcome"] = outcome
            rec["resolved_at"] = now.isoformat()
            # Also append to outcomes.jsonl for fast queries
            _jsonl_append(live_dir / "outcomes.jsonl", {
                "fire_id": rec["fire_id"],
                "pin_id": rec["pin_id"],
                "rule_id": rec["rule_id"],
                "product": rec["product"],
                "scan_time": rec["scan_time"],
                "resolved_at": rec["resolved_at"],
                "hit": outcome.get("hit", False),
                "max_pct": outcome.get("max_pct"),
                "time_to_hit_bars": outcome.get("time_to_hit_bars"),
                "final_pct": outcome.get("final_pct"),
            })
            resolved_count += 1
            return rec
        except Exception as e:
            log.exception(f"error resolving fire {rec.get('fire_id')}: {e}")
            error_count += 1
            rec["resolved"] = True
            rec["outcome"] = {"error": str(e)}
            return rec

    _jsonl_update(fires_path, predicate, updater)
    log.info(f"record_outcomes: resolved={resolved_count} pending={pending_count} "
             f"errors={error_count}")
    return {"resolved": resolved_count, "pending": pending_count, "errors": error_count}


# ═══════════════════════════════════════════════════════════════════
# AGGREGATION — live precision per rule, rolling windows
# ═══════════════════════════════════════════════════════════════════

def aggregate_stats(live_dir: Path, window_days: int = None) -> dict:
    """
    Aggregate live outcomes into per-rule statistics.

    Returns: {
      "by_rule": {pin_id: {n_fires, n_resolved, n_hit, precision,
                            avg_max_pct, avg_final_pct, ...}},
      "totals": {...},
      "window_days": window_days,
    }
    """
    outcomes = _jsonl_read(live_dir / "outcomes.jsonl")

    if window_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        outcomes = [o for o in outcomes
                     if datetime.fromisoformat(o["scan_time"].replace("Z","+00:00")).astimezone(UTC) >= cutoff]

    # Load pending fires too (unresolved)
    pending_fires = _jsonl_read(
        live_dir / "fires.jsonl",
        filter_fn=lambda r: not r.get("resolved", False))

    by_rule = defaultdict(lambda: {
        "pin_id": None, "rule_id": None,
        "n_fires_resolved": 0, "n_hits": 0,
        "n_fires_pending": 0,
        "sum_max_pct": 0.0, "sum_final_pct": 0.0,
        "times_to_hit_bars": [],
    })

    for o in outcomes:
        k = o["pin_id"]
        b = by_rule[k]
        b["pin_id"] = k
        b["rule_id"] = o["rule_id"]
        b["n_fires_resolved"] += 1
        if o.get("hit"):
            b["n_hits"] += 1
            if o.get("time_to_hit_bars") is not None:
                b["times_to_hit_bars"].append(o["time_to_hit_bars"])
        if o.get("max_pct") is not None:
            b["sum_max_pct"] += o["max_pct"]
        if o.get("final_pct") is not None:
            b["sum_final_pct"] += o["final_pct"]

    for f in pending_fires:
        k = f["pin_id"]
        if k not in by_rule:
            by_rule[k]["pin_id"] = k
            by_rule[k]["rule_id"] = f["rule_id"]
        by_rule[k]["n_fires_pending"] += 1

    # Augment with validation data from pinned rules
    pinned_lookup = {r["pin_id"]: r for r in load_pinned_rules(live_dir)}

    out = {}
    for k, b in by_rule.items():
        n_res = b["n_fires_resolved"]
        live_precision = b["n_hits"] / n_res if n_res > 0 else None
        pinned = pinned_lookup.get(k, {})
        val_precision = pinned.get("validation_precision")
        val_base = pinned.get("validation_base_rate")
        precision_delta = None
        if live_precision is not None and val_precision is not None:
            precision_delta = live_precision - val_precision
        out[k] = {
            "pin_id": k,
            "rule_id": b["rule_id"],
            "english": pinned.get("english", ""),
            "has_disqualifier": pinned.get("disqualifier") is not None,
            "n_fires_resolved": n_res,
            "n_fires_pending": b["n_fires_pending"],
            "n_hits": b["n_hits"],
            "live_precision": round(live_precision, 4) if live_precision is not None else None,
            "validation_precision": val_precision,
            "validation_support": pinned.get("validation_support"),
            "validation_base_rate": val_base,
            "precision_delta": round(precision_delta, 4) if precision_delta is not None else None,
            "avg_max_pct": round(b["sum_max_pct"] / n_res, 4) if n_res > 0 else None,
            "avg_final_pct": round(b["sum_final_pct"] / n_res, 4) if n_res > 0 else None,
            "avg_time_to_hit_bars": round(
                sum(b["times_to_hit_bars"]) / len(b["times_to_hit_bars"]), 2)
                if b["times_to_hit_bars"] else None,
            "threshold_pct": pinned.get("threshold_pct"),
            "horizon_hours": pinned.get("horizon_hours"),
            "pinned_at": pinned.get("pinned_at"),
        }

    totals = {
        "n_fires_resolved": sum(b["n_fires_resolved"] for b in by_rule.values()),
        "n_fires_pending": sum(b["n_fires_pending"] for b in by_rule.values()),
        "n_hits": sum(b["n_hits"] for b in by_rule.values()),
        "n_rules_active": len(out),
    }
    if totals["n_fires_resolved"] > 0:
        totals["overall_live_precision"] = round(
            totals["n_hits"] / totals["n_fires_resolved"], 4)

    return {"by_rule": out, "totals": totals, "window_days": window_days}


def list_recent_fires(live_dir: Path, limit: int = 100,
                       pin_id: str = None, only_unresolved: bool = False) -> list:
    """List recent fires with optional filters."""
    def flt(r):
        if pin_id and r.get("pin_id") != pin_id:
            return False
        if only_unresolved and r.get("resolved"):
            return False
        return True
    return _jsonl_read(live_dir / "fires.jsonl", limit=limit, filter_fn=flt)


# ═══════════════════════════════════════════════════════════════════
# BACKTEST — evaluate pinned rules against historical rows
# ═══════════════════════════════════════════════════════════════════

def run_backtest(live_dir, rows, horizon_hours_list, pinned_rules,
                  backtest_id=None, progress_cb=None, start_date=None,
                  end_date=None):
    """
    For each pinned rule, evaluate it against every row in `rows` (inclusive of
    start_date..end_date if provided). For each fire, use the row's label_{H}h
    column to determine hit/miss (already computed during row build).

    Returns the backtest report (also persists to disk under
    {live_dir}/backtests/{backtest_id}.json).

    Args:
      rows: output of v2.build_rows_for_rule_mining — each row has features +
            label_4h/6h/8h + is_train/is_val/is_test + date + product + scan_hour.
      horizon_hours_list: horizons present in the rows (e.g. [4,6,8])
      pinned_rules: list of pinned rule dicts (from load_pinned_rules)
      start_date/end_date: ISO dates (YYYY-MM-DD) to filter rows by. If None,
        use all rows. Inclusive.

    Report structure:
      {
        "backtest_id": str,
        "generated_at": ISO,
        "date_range": {start, end, n_rows_in_range},
        "n_rules": int,
        "per_rule": [
          {
            "pin_id": str, "english": str, "conditions": [...],
            "disqualifier": {...} or None,
            "threshold_pct": float, "horizon_hours": int,
            "validation_precision": float,   # from the pinned rule
            "overall": {n_fires, n_hits, precision, hit_rate_by_coin, base_rate_in_range},
            "by_split": {
              "train": {n_fires, n_hits, precision},
              "val":   {...},
              "test":  {...},
            },
            "top_fire_coins": [(coin, hits, fires)...]  # top 20 coins by fire count
          }, ...
        ],
      }
    """
    import pandas as pd
    if backtest_id is None:
        backtest_id = f"bt_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    def prog(p, m):
        if progress_cb: progress_cb(p, m)
        log.info(f"[backtest] {p}% — {m}")

    # Filter rows by date range if specified
    if start_date or end_date:
        def in_range(r):
            d = r.get("date", "")
            if start_date and d < start_date: return False
            if end_date and d > end_date: return False
            return True
        rows = [r for r in rows if in_range(r)]
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("No rows in date range")

    prog(5, f"Building DataFrame from {n_rows} rows...")
    df = pd.DataFrame(rows)

    # For each pinned rule, we need to evaluate its conditions against each row.
    # But the rule's conditions use BIN IDs (specific to the bin_edges stored
    # with the pinned rule). We need to (a) use the pinned rule's own bin_edges,
    # (b) assign each row's raw feature values to bins using those edges, (c)
    # check if the row's bins match the rule's required bins.
    prog(10, "Evaluating rules against rows...")

    per_rule = []
    n_pinned = len(pinned_rules)
    for pi, pin in enumerate(pinned_rules):
        bin_edges = pin.get("bin_edges", {})
        horizon_hours = pin["horizon_hours"]
        label_col = f"label_{horizon_hours}h"
        if label_col not in df.columns:
            log.warning(f"Pin {pin['pin_id']}: no label_{horizon_hours}h column, skipping")
            continue

        # Build a fire mask by evaluating rule conditions against each row.
        # Vectorized where possible: for each condition's feature, assign bins
        # to the whole column, then AND across conditions.
        import numpy as np
        fire_mask = np.ones(len(df), dtype=bool)
        all_feats_present = True
        for cond in pin["conditions"]:
            feat = cond["feature"]
            expected_bins = set(cond["bins"])
            if feat not in df.columns:
                all_feats_present = False
                break
            edges = bin_edges.get(feat)
            edges_arr = np.array(edges) if edges is not None else None
            # Assign bins for this column
            col_vals = df[feat].values
            if edges_arr is None:
                # Binary — clamp to 0/1
                col_bins = np.where(np.isfinite(col_vals),
                                      np.clip(col_vals.astype(int), 0, 1), -1)
            else:
                # np.searchsorted gives us the insertion position — which equals
                # the bin id (0..len(edges)) using the same semantics as
                # assign_bin(): val < edges[0] → bin 0, val >= edges[-1] → bin len(edges)
                col_bins = np.searchsorted(edges_arr, col_vals, side="right")
                # side='right' gives correct semantics: edges=[30,45], val=30 → pos 1 (bin 1),
                # val=29 → pos 0 (bin 0), val=45 → pos 2 (bin 2). This matches v2_rules'
                # "for i, e in enumerate(edges): if val < e: return i" logic.
                # But handle NaN/inf: searchsorted treats NaN oddly, replace with -1
                col_bins = np.where(np.isfinite(col_vals), col_bins, -1)
            # Check if each row's bin is in expected_bins
            match = np.array([int(b) in expected_bins for b in col_bins])
            fire_mask &= match

        if not all_feats_present:
            per_rule.append({
                "pin_id": pin["pin_id"], "english": pin["english"],
                "error": "feature_missing_in_rows",
            })
            continue

        # Apply disqualifier if set
        disq = pin.get("disqualifier")
        disq_exclude_mask = np.zeros(len(df), dtype=bool)
        if disq:
            feat = disq["feature"]
            thresh = disq["thresh"]
            direction = disq["direction"]
            if feat in df.columns:
                col_vals = df[feat].values
                finite = np.isfinite(col_vals)
                if direction == "exclude_if_greater":
                    disq_exclude_mask = finite & (col_vals > thresh)
                elif direction == "exclude_if_less":
                    disq_exclude_mask = finite & (col_vals < thresh)
            fire_mask &= ~disq_exclude_mask

        labels = df[label_col].values
        n_fires = int(fire_mask.sum())
        n_hits = int(labels[fire_mask].sum()) if n_fires > 0 else 0

        if n_fires == 0:
            per_rule.append({
                "pin_id": pin["pin_id"],
                "english": pin["english"],
                "conditions": pin["conditions"],
                "disqualifier": disq,
                "threshold_pct": pin["threshold_pct"],
                "horizon_hours": horizon_hours,
                "validation_precision": pin.get("validation_precision"),
                "overall": {"n_fires": 0, "n_hits": 0, "precision": None},
                "by_split": {},
                "top_fire_coins": [],
            })
            prog(10 + int((pi + 1) / n_pinned * 85),
                 f"Rule {pi+1}/{n_pinned}: 0 fires")
            continue

        # Precision per split
        by_split = {}
        for split in ("train", "val", "test"):
            split_col = f"is_{split}"
            if split_col not in df.columns: continue
            split_mask = df[split_col].astype(bool).values
            combined = fire_mask & split_mask
            sn = int(combined.sum())
            sh = int(labels[combined].sum()) if sn > 0 else 0
            by_split[split] = {
                "n_fires": sn,
                "n_hits": sh,
                "precision": round(sh / sn, 4) if sn > 0 else None,
                "base_rate": round(float(labels[split_mask].mean()), 4) if split_mask.sum() > 0 else None,
            }

        # Top-firing coins
        fired_rows_df = df[fire_mask]
        fire_outcomes = labels[fire_mask]
        coin_stats = {}
        for i, (_, r) in enumerate(fired_rows_df.iterrows()):
            coin = r["product"]
            if coin not in coin_stats:
                coin_stats[coin] = {"fires": 0, "hits": 0}
            coin_stats[coin]["fires"] += 1
            if fire_outcomes[i]:
                coin_stats[coin]["hits"] += 1
        top_coins = sorted(
            [(c, s["hits"], s["fires"]) for c, s in coin_stats.items()],
            key=lambda x: -x[2])[:20]

        base_rate_in_range = float(labels.mean())

        per_rule.append({
            "pin_id": pin["pin_id"],
            "english": pin["english"],
            "conditions": pin["conditions"],
            "disqualifier": disq,
            "threshold_pct": pin["threshold_pct"],
            "horizon_hours": horizon_hours,
            "validation_precision": pin.get("validation_precision"),
            "overall": {
                "n_fires": n_fires,
                "n_hits": n_hits,
                "precision": round(n_hits / n_fires, 4),
                "base_rate_in_range": round(base_rate_in_range, 4),
                "lift_vs_base": round(n_hits / n_fires - base_rate_in_range, 4),
                "n_distinct_coins": len(coin_stats),
            },
            "by_split": by_split,
            "top_fire_coins": top_coins,
        })
        prog(10 + int((pi + 1) / n_pinned * 85),
             f"Rule {pi+1}/{n_pinned}: {n_fires} fires, {n_hits} hits")

    prog(97, "Saving report...")
    # Actual date range observed in rows
    all_dates = [r.get("date", "") for r in rows if r.get("date")]
    date_range = {
        "requested_start": start_date,
        "requested_end": end_date,
        "actual_first": min(all_dates) if all_dates else None,
        "actual_last": max(all_dates) if all_dates else None,
        "n_rows": n_rows,
        "n_distinct_dates": len(set(all_dates)) if all_dates else 0,
    }
    report = {
        "_type": "coinbase_scanner_v2_backtest",
        "backtest_id": backtest_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "date_range": date_range,
        "n_pinned_rules": len(pinned_rules),
        "n_rules_evaluated": len(per_rule),
        "per_rule": per_rule,
    }
    out_dir = live_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{backtest_id}.json").write_text(json.dumps(report, default=str, indent=2))
    prog(100, f"Done. Report saved as {backtest_id}.json")
    return report


def list_backtests(live_dir):
    """Return summaries of all saved backtest runs."""
    out_dir = live_dir / "backtests"
    if not out_dir.exists(): return []
    summaries = []
    for p in sorted(out_dir.glob("bt_*.json"), reverse=True):
        try:
            r = json.loads(p.read_text())
            summaries.append({
                "backtest_id": r["backtest_id"],
                "generated_at": r["generated_at"],
                "n_rules_evaluated": r["n_rules_evaluated"],
                "date_range": r.get("date_range"),
            })
        except Exception as e:
            log.warning(f"list_backtests: skipping {p.name}: {e}")
    return summaries


def load_backtest(live_dir, backtest_id):
    """Load a specific backtest report."""
    p = live_dir / "backtests" / f"{backtest_id}.json"
    if not p.exists(): return None
    return json.loads(p.read_text())
