"""
v2_rules — Rule mining for setup discovery.

Goal: find describable setups (conditions on technical features) that precede
winners (price touched +X% within H hours) at high precision, and identify
disqualifiers that distinguish winners from false-positive losers.

Three mining methods run in parallel:
  1. Univariate screening — for each feature, find value ranges where winners
     cluster more densely than losers.
  2. Decision trees with precision-optimized leaves — each leaf is a multi-
     feature rule; keep leaves with high precision.
  3. Frequent pattern (Apriori-style) mining — discretize features into bins,
     treat each snapshot as a set of items, find item combinations that occur
     disproportionately in winners.

All rules are validated on a held-out test set. Rules are then scored by test
precision, test lift (precision minus base rate), and coverage (fires per day).

DESIGN DECISIONS (documented because they matter):
  • Absolute +2% threshold (not vol-normalized k*ATR). User chose this for
    tradability. Introduces high-vol bias — model will naturally favor
    volatile coins. Rules will surface this as a feature-in-the-rule (e.g.,
    "when realized_vol > 0.03"), which is interpretable and actionable.
  • Three parallel universes per horizon (4h, 6h, 8h). Each horizon produces
    its own rule catalog. No cross-horizon consistency required.
  • Features binned via training-set quantiles (not fixed values). Each
    feature gets 5 bins: very-low, low, med, high, very-high. Binary features
    stay as {0, 1}.
  • Precision-first: min_precision and min_support are the two filters.
    Default 0.65 and 0.005 (= 0.5% of training samples).
  • No overfit protection within rule discovery. Overfitting is caught by
    train/test precision comparison — rules with train-test gap >0.15 are
    flagged as likely overfit.
  • Disqualifier analysis: for each rule, look at the losers it fires on and
    find features where those losers differ from the winners the rule fires
    on. The disqualifier becomes a refinement.

NUMERICAL SAFETY:
  • All feature arrays cleaned of NaN/Inf before binning.
  • Quantiles computed on training data only (no leakage).
  • Rules with support < min_support_count (absolute floor of 20) are dropped.
"""
import os, json, time, math, logging, pickle, hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo
from itertools import combinations

import numpy as np
import pandas as pd

log = logging.getLogger("v2_rules")
UTC = ZoneInfo("UTC")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

# Absolute threshold for "winner" label (user choice).
WINNER_THRESHOLD_PCT = 0.02   # +2%

# Horizons mined in parallel. Each gets its own rule catalog.
MINING_HORIZONS_HOURS = [4, 6, 8]

# Binning
N_BINS_CONTINUOUS = 5   # very-low, low, med, high, very-high
BIN_LABELS = ["VL", "L", "M", "H", "VH"]

# Features that are naturally binary (0/1) — keep as-is, no binning
BINARY_FEATURES = {
    "bb_squeeze", "macd_cross", "rsi_divergence", "stoch_cross_up",
    "vol_surge", "higher_highs_5b", "donchian_breakout", "squeeze_fire",
    "cat_strongest", "cat_weakest",
}

# Features we deliberately EXCLUDE from binning (too noisy / too ranked / redundant)
# Keep scan_hour_sin/cos for rules but don't mine categorical rules on them.
EXCLUDE_FROM_MINING = {
    "scan_hour_sin", "scan_hour_cos",   # encoded time, hard to interpret as a rule
    "dollar_volume_rank",                # already cross-sectional
    "rank_momentum", "rank_vol_zscore", "rank_bb_position",
    "rank_rsi", "rank_range_position",   # cross-sectional ranks
}

# Minimum support (absolute count of training samples) for any rule
MIN_SUPPORT_COUNT_FLOOR = 20

# Default mining thresholds (user-tunable at runtime)
DEFAULT_MIN_PRECISION = 0.65
DEFAULT_MIN_SUPPORT_FRAC = 0.005   # 0.5% of training samples
DEFAULT_MIN_LIFT = 0.05            # 5pp lift over base rate

# Decision tree mining
TREE_MAX_DEPTH = 4
TREE_MIN_SAMPLES_LEAF = 50
TREE_N_TREES = 12   # ensemble of trees with randomized feature subsets

# Apriori pattern mining
APRIORI_MAX_ITEMSET_SIZE = 3   # up to 3-feature rules
APRIORI_MIN_ITEMSET_SUPPORT = 0.02   # item must appear in 2%+ of samples to be a candidate

# Overfitting detection
OVERFIT_TRAIN_TEST_GAP = 0.15

# ═══════════════════════════════════════════════════════════════════
# BIN COMPUTATION — quantile-based on training data
# ═══════════════════════════════════════════════════════════════════

def compute_bin_edges(train_values, n_bins=N_BINS_CONTINUOUS):
    """
    Compute quantile bin edges from training data.
    Returns array of (n_bins-1) edges. Deduplicated and sorted.
    Falls back to fewer bins if the feature has many identical values.
    """
    arr = np.array(train_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < n_bins * 10:
        return None
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(arr, quantiles)
    # Deduplicate (common for features with lots of zeros)
    edges = np.unique(np.round(edges, 8))
    if len(edges) < 1:
        return None
    return edges


def assign_bin(val, edges):
    """Given a value and bin edges, return integer bin id in [0, len(edges)]."""
    if val is None or not np.isfinite(val):
        return -1   # missing bucket
    for i, e in enumerate(edges):
        if val < e:
            return i
    return len(edges)


def build_binned_dataframe(df, feature_names, train_mask):
    """
    Given a DataFrame with continuous feature columns and a mask indicating
    training rows, return:
      • binned_df: DataFrame with '{feature}_bin' columns (integer bin ids)
      • bin_edges: dict feature -> np.array of edges used for each feature
      • bin_labels: dict feature -> list of human-readable bin labels
    Binary features pass through unchanged.
    """
    train_df = df[train_mask]
    binned = {}
    edges_map = {}
    labels_map = {}

    for feat in feature_names:
        if feat in EXCLUDE_FROM_MINING:
            continue
        if feat in BINARY_FEATURES:
            # Binary — keep as-is
            vals = df[feat].fillna(0).astype(int).clip(0, 1).values
            binned[feat] = vals
            edges_map[feat] = None
            labels_map[feat] = ["0", "1"]
            continue

        train_vals = train_df[feat].values
        edges = compute_bin_edges(train_vals)
        if edges is None:
            # Too-uniform feature, skip
            continue
        # Assign bins to all rows
        all_vals = df[feat].values
        bin_ids = np.array([assign_bin(v, edges) for v in all_vals])
        binned[feat] = bin_ids
        edges_map[feat] = edges
        # Make human-readable labels for each bin
        n_bins = len(edges) + 1
        lbls = []
        for i in range(n_bins):
            if i == 0:
                lbls.append(f"<{edges[0]:.3g}")
            elif i == n_bins - 1:
                lbls.append(f">={edges[-1]:.3g}")
            else:
                lbls.append(f"[{edges[i-1]:.3g},{edges[i]:.3g})")
        labels_map[feat] = lbls

    binned_df = pd.DataFrame(binned, index=df.index)
    return binned_df, edges_map, labels_map


# ═══════════════════════════════════════════════════════════════════
# RULE EVALUATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

def evaluate_rule(mask, labels):
    """
    Given a boolean mask of samples the rule fires on, and binary labels,
    return: precision (winner fraction of fires), support (count of fires),
    and recall (fraction of winners captured). Safe when mask empty.
    """
    n = int(mask.sum())
    if n == 0:
        return {"precision": 0.0, "support": 0, "recall": 0.0, "lift_vs_base": 0.0}
    fired_labels = labels[mask]
    n_winners_fired = int(fired_labels.sum())
    precision = n_winners_fired / n
    total_winners = int(labels.sum()) or 1
    recall = n_winners_fired / total_winners
    base_rate = float(labels.mean())
    return {
        "precision": float(precision),
        "support": n,
        "recall": float(recall),
        "lift_vs_base": float(precision - base_rate),
    }


def rule_to_mask(rule, binned_df):
    """
    Given a rule definition (dict with conditions) and a binned DataFrame,
    return a boolean mask of rows that satisfy all conditions.
    Conditions format: [{"feature": str, "bins": [int, ...]}, ...]
    A row satisfies the condition if its bin id is in the 'bins' list.
    """
    if not rule.get("conditions"):
        return np.zeros(len(binned_df), dtype=bool)
    masks = []
    for cond in rule["conditions"]:
        feat = cond["feature"]
        bins = cond["bins"]
        if feat not in binned_df.columns:
            return np.zeros(len(binned_df), dtype=bool)
        mask = binned_df[feat].isin(bins).values
        masks.append(mask)
    return np.all(masks, axis=0) if masks else np.zeros(len(binned_df), dtype=bool)


def rule_to_english(rule, bin_labels_map):
    """Render a rule as human-readable English."""
    parts = []
    for cond in rule.get("conditions", []):
        feat = cond["feature"]
        bins = cond["bins"]
        labels = bin_labels_map.get(feat, [])
        if feat in BINARY_FEATURES:
            # Binary — "bb_squeeze=1" or "bb_squeeze=0"
            vals = sorted(set(bins))
            if len(vals) == 1:
                parts.append(f"{feat}={vals[0]}")
            else:
                parts.append(f"{feat} in {{{', '.join(str(v) for v in vals)}}}")
        else:
            # Continuous — use bin labels
            bin_strs = [labels[b] if 0 <= b < len(labels) else f"bin{b}" for b in sorted(set(bins))]
            if len(bin_strs) == len(labels):
                parts.append(f"{feat}: any")
            else:
                parts.append(f"{feat} ∈ {{{', '.join(bin_strs)}}}")
    return " AND ".join(parts) if parts else "(empty rule)"


# ═══════════════════════════════════════════════════════════════════
# MINING METHOD 1: UNIVARIATE SCREENING
# ═══════════════════════════════════════════════════════════════════

def mine_univariate(binned_df, labels, train_mask, min_precision, min_support_count, min_lift):
    """
    For each feature and each bin id, compute precision/support/lift.
    Keep single-bin rules that clear the thresholds on training data.
    Also try 2-bin combinations per feature (e.g., VH OR H).
    """
    log.info("univariate mining...")
    rules = []
    train_labels = labels[train_mask]
    base_rate = float(train_labels.mean())

    for feat in binned_df.columns:
        bin_vals = binned_df[feat].values[train_mask]
        unique_bins = sorted(set(int(b) for b in bin_vals if b >= 0))
        if len(unique_bins) < 2:
            continue

        # 1-bin rules
        for b in unique_bins:
            mask = (bin_vals == b)
            if mask.sum() < min_support_count:
                continue
            n_winners = int(train_labels[mask].sum())
            n_total = int(mask.sum())
            prec = n_winners / n_total
            lift = prec - base_rate
            if prec >= min_precision and lift >= min_lift:
                rules.append({
                    "method": "univariate",
                    "conditions": [{"feature": feat, "bins": [b]}],
                })

        # 2-bin (OR) rules within the same feature — e.g., "H or VH"
        for combo in combinations(unique_bins, 2):
            mask = np.isin(bin_vals, combo)
            if mask.sum() < min_support_count:
                continue
            n_winners = int(train_labels[mask].sum())
            n_total = int(mask.sum())
            prec = n_winners / n_total
            lift = prec - base_rate
            if prec >= min_precision and lift >= min_lift:
                rules.append({
                    "method": "univariate",
                    "conditions": [{"feature": feat, "bins": list(combo)}],
                })
    log.info(f"  univariate: {len(rules)} raw rules")
    return rules


# ═══════════════════════════════════════════════════════════════════
# MINING METHOD 2: PRECISION-OPTIMIZED DECISION TREES
# ═══════════════════════════════════════════════════════════════════

def _tree_leaves_as_rules(feat_mat, feat_names, labels, max_depth,
                          min_leaf, min_precision, min_support_count, min_lift,
                          base_rate, rng):
    """
    Grow a precision-optimized tree and return its leaf conditions as rules.
    Uses a greedy splitter that picks splits maximizing (precision * sqrt(support)).
    This is a custom tree (not sklearn) because we want direct rule extraction
    and no overhead of fitting full classifiers.
    """
    # Each node represented as: (indices_of_rows, list_of_conditions_so_far)
    stack = [(np.arange(len(labels)), [])]
    leaf_conditions = []
    n_nodes = 0
    node_limit = 2 ** (max_depth + 1)

    while stack and n_nodes < node_limit:
        n_nodes += 1
        idx, conds = stack.pop()
        if len(idx) < min_leaf * 2:
            # Too small to split — emit as leaf
            if len(idx) >= min_support_count:
                lbls = labels[idx]
                prec = float(lbls.mean())
                if prec >= min_precision and (prec - base_rate) >= min_lift:
                    leaf_conditions.append({"conditions": conds[:], "node_support": len(idx),
                                            "node_precision": prec})
            continue

        if len(conds) >= max_depth:
            # At max depth — emit as leaf
            if len(idx) >= min_support_count:
                lbls = labels[idx]
                prec = float(lbls.mean())
                if prec >= min_precision and (prec - base_rate) >= min_lift:
                    leaf_conditions.append({"conditions": conds[:], "node_support": len(idx),
                                            "node_precision": prec})
            continue

        # Find best split across a random feature subset
        n_features = feat_mat.shape[1]
        subset_size = max(5, int(math.sqrt(n_features)))
        feature_subset = rng.choice(n_features, size=min(subset_size, n_features), replace=False)

        best_score = -1.0
        best_split = None   # (feat_idx, left_bins, right_bins)

        for f_idx in feature_subset:
            feat_name = feat_names[f_idx]
            # Skip features already used in the path (avoid redundant splits)
            if any(c["feature"] == feat_name for c in conds):
                continue
            node_vals = feat_mat[idx, f_idx]
            unique = sorted(set(int(v) for v in node_vals if v >= 0))
            if len(unique) < 2:
                continue

            # Try splitting at each unique value: left = {bins < split}, right = {bins >= split}
            for split_bin in unique[1:]:
                left_mask = node_vals < split_bin
                right_mask = ~left_mask
                nl, nr = int(left_mask.sum()), int(right_mask.sum())
                if nl < min_leaf or nr < min_leaf:
                    continue
                left_prec = float(labels[idx[left_mask]].mean())
                right_prec = float(labels[idx[right_mask]].mean())
                # Score: pick the side with higher precision, weight by sqrt(support)
                candidates = [
                    (left_prec, nl, [b for b in unique if b < split_bin],
                     [b for b in unique if b >= split_bin], "left"),
                    (right_prec, nr, [b for b in unique if b >= split_bin],
                     [b for b in unique if b < split_bin], "right"),
                ]
                for prec, support, chosen_bins, other_bins, side in candidates:
                    # Reward high precision + reasonable support
                    score = (prec - base_rate) * math.sqrt(support)
                    if score > best_score:
                        best_score = score
                        best_split = (f_idx, feat_name, chosen_bins, other_bins, side)

        if best_split is None:
            # No good split — emit as leaf
            if len(idx) >= min_support_count:
                lbls = labels[idx]
                prec = float(lbls.mean())
                if prec >= min_precision and (prec - base_rate) >= min_lift:
                    leaf_conditions.append({"conditions": conds[:], "node_support": len(idx),
                                            "node_precision": prec})
            continue

        f_idx, feat_name, chosen_bins, other_bins, _ = best_split
        chosen_mask = np.isin(feat_mat[idx, f_idx], chosen_bins)

        # Recurse on chosen-side only (precision-optimized = follow the good side)
        new_conds = conds + [{"feature": feat_name, "bins": chosen_bins}]
        stack.append((idx[chosen_mask], new_conds))
        # Also keep the other side — it might have its own high-precision sub-region
        other_conds = conds + [{"feature": feat_name, "bins": other_bins}]
        stack.append((idx[~chosen_mask], other_conds))

    return leaf_conditions


def mine_trees(binned_df, labels, train_mask, min_precision, min_support_count, min_lift, n_trees=TREE_N_TREES):
    """
    Grow `n_trees` precision-optimized decision trees. Each uses a random subset
    of features. Collect all leaves that pass the threshold as candidate rules.
    """
    log.info(f"decision tree mining ({n_trees} trees, max depth {TREE_MAX_DEPTH})...")
    feat_names = list(binned_df.columns)
    # Train-only slice
    feat_mat_train = binned_df.values[train_mask]
    train_labels = labels[train_mask]
    base_rate = float(train_labels.mean())

    all_rules = []
    rng = np.random.default_rng(42)
    for t in range(n_trees):
        tree_rng = np.random.default_rng(42 + t)
        leaves = _tree_leaves_as_rules(
            feat_mat_train, feat_names, train_labels,
            max_depth=TREE_MAX_DEPTH,
            min_leaf=TREE_MIN_SAMPLES_LEAF,
            min_precision=min_precision,
            min_support_count=min_support_count,
            min_lift=min_lift,
            base_rate=base_rate,
            rng=tree_rng,
        )
        for leaf in leaves:
            # Only keep leaves with at least 1 condition (empty leaf = base rate, useless)
            if leaf["conditions"]:
                all_rules.append({
                    "method": "tree",
                    "conditions": leaf["conditions"],
                    "tree_id": t,
                })
    log.info(f"  tree: {len(all_rules)} raw rules (pre-dedup)")
    return all_rules


# ═══════════════════════════════════════════════════════════════════
# MINING METHOD 3: FREQUENT PATTERN (Apriori-style)
# ═══════════════════════════════════════════════════════════════════

def mine_apriori(binned_df, labels, train_mask, min_precision, min_support_count, min_lift,
                 max_itemset_size=APRIORI_MAX_ITEMSET_SIZE,
                 min_item_support=APRIORI_MIN_ITEMSET_SUPPORT):
    """
    Apriori-style frequent pattern mining.

    Step 1: Enumerate all (feature, bin_id) "items". Keep items that appear in
    at least min_item_support of training samples.
    Step 2: Build 2-itemsets from frequent 1-items. Prune by support.
    Step 3: Extend to 3-itemsets. Prune by support.
    At each stage evaluate precision on training data and keep rules that pass.
    """
    log.info(f"apriori mining (max size {max_itemset_size}, "
             f"min item support {min_item_support:.2%})...")

    n_train = int(train_mask.sum())
    min_item_count = max(MIN_SUPPORT_COUNT_FLOOR, int(min_item_support * n_train))
    train_labels = labels[train_mask]
    base_rate = float(train_labels.mean())

    # Step 1: frequent 1-items (feature, bin) pairs.
    frequent_items = []   # list of (feature_name, bin_id, mask_on_full_data, support_count_train)
    for feat in binned_df.columns:
        bin_vals = binned_df[feat].values
        for b in sorted(set(int(v) for v in bin_vals[train_mask] if v >= 0)):
            item_mask = (bin_vals == b)
            train_support = int(item_mask[train_mask].sum())
            if train_support >= min_item_count:
                frequent_items.append((feat, b, item_mask, train_support))
    log.info(f"  1-items: {len(frequent_items)} frequent")

    rules = []
    # Evaluate 1-items as rules
    for (feat, b, mask, support) in frequent_items:
        fired_labels = train_labels[mask[train_mask]]
        n_fires = len(fired_labels)
        if n_fires == 0: continue
        prec = float(fired_labels.mean())
        lift = prec - base_rate
        if prec >= min_precision and lift >= min_lift and n_fires >= min_support_count:
            rules.append({
                "method": "apriori",
                "conditions": [{"feature": feat, "bins": [b]}],
            })

    if max_itemset_size < 2:
        log.info(f"  apriori: {len(rules)} raw rules")
        return rules

    # Step 2: 2-itemsets (pairs from different features)
    pair_count = 0
    for i in range(len(frequent_items)):
        for j in range(i+1, len(frequent_items)):
            feat_i, bin_i, mask_i, _ = frequent_items[i]
            feat_j, bin_j, mask_j, _ = frequent_items[j]
            if feat_i == feat_j:
                continue   # same feature — skip (already covered by univariate OR-rules)
            combined_mask = mask_i & mask_j
            train_fires = combined_mask[train_mask]
            n_fires = int(train_fires.sum())
            if n_fires < min_item_count:
                continue
            pair_count += 1
            fired_labels = train_labels[train_fires]
            prec = float(fired_labels.mean())
            lift = prec - base_rate
            if prec >= min_precision and lift >= min_lift and n_fires >= min_support_count:
                rules.append({
                    "method": "apriori",
                    "conditions": [
                        {"feature": feat_i, "bins": [bin_i]},
                        {"feature": feat_j, "bins": [bin_j]},
                    ],
                })
    log.info(f"  2-items: {pair_count} pairs evaluated, "
             f"{len([r for r in rules if len(r['conditions'])==2])} rules")

    if max_itemset_size < 3:
        log.info(f"  apriori: {len(rules)} raw rules")
        return rules

    # Step 3: 3-itemsets — cap the number we try to keep runtime bounded.
    # We only extend frequent 2-itemsets by adding one more frequent 1-item.
    pair_rules = [r for r in rules if len(r["conditions"]) == 2]
    triple_count = 0
    seen_triples = set()
    for pair_rule in pair_rules[:500]:   # cap on pair-seeds
        pair_feats = {c["feature"] for c in pair_rule["conditions"]}
        pair_mask = np.ones(len(binned_df), dtype=bool)
        for c in pair_rule["conditions"]:
            pair_mask &= (binned_df[c["feature"]].values == c["bins"][0])
        for (feat_k, bin_k, mask_k, _) in frequent_items:
            if feat_k in pair_feats:
                continue
            key = tuple(sorted([(c["feature"], c["bins"][0]) for c in pair_rule["conditions"]] + [(feat_k, bin_k)]))
            if key in seen_triples:
                continue
            seen_triples.add(key)
            combined = pair_mask & mask_k
            train_fires = combined[train_mask]
            n_fires = int(train_fires.sum())
            if n_fires < min_item_count:
                continue
            triple_count += 1
            fired_labels = train_labels[train_fires]
            prec = float(fired_labels.mean())
            lift = prec - base_rate
            if prec >= min_precision and lift >= min_lift and n_fires >= min_support_count:
                rules.append({
                    "method": "apriori",
                    "conditions": [
                        {"feature": c["feature"], "bins": c["bins"]} for c in pair_rule["conditions"]
                    ] + [{"feature": feat_k, "bins": [bin_k]}],
                })
    log.info(f"  3-items: {triple_count} triples evaluated, "
             f"{len([r for r in rules if len(r['conditions'])==3])} rules")

    log.info(f"apriori: {len(rules)} raw rules total")
    return rules


# ═══════════════════════════════════════════════════════════════════
# DEDUPLICATION — same rule discovered by multiple methods
# ═══════════════════════════════════════════════════════════════════

def _rule_signature(rule):
    """Canonical string for rule deduplication. Ignores method & metadata."""
    parts = []
    for c in sorted(rule["conditions"], key=lambda x: x["feature"]):
        bins_str = ",".join(str(b) for b in sorted(c["bins"]))
        parts.append(f"{c['feature']}:{bins_str}")
    return "|".join(parts)


def dedupe_rules(rules):
    """
    Collapse duplicate rules (same conditions, possibly found by multiple methods).
    Preserves a 'methods' list so we know which method(s) found each rule.
    """
    by_sig = {}
    for r in rules:
        sig = _rule_signature(r)
        if sig not in by_sig:
            by_sig[sig] = {
                "conditions": r["conditions"],
                "methods": [r["method"]],
                "signature": sig,
            }
        else:
            if r["method"] not in by_sig[sig]["methods"]:
                by_sig[sig]["methods"].append(r["method"])
    return list(by_sig.values())


# ═══════════════════════════════════════════════════════════════════
# RULE VALIDATION — train/val/test precision per rule
# ═══════════════════════════════════════════════════════════════════

def validate_rules(rules, binned_df, labels, train_mask, val_mask, test_mask, base_rates):
    """
    For each deduped rule, evaluate precision/support/lift on train/val/test
    splits. Annotate with the train-test gap (overfitting signal).
    Filter out rules that score 0 fires on test (can't be evaluated).
    """
    log.info(f"validating {len(rules)} rules on train/val/test...")
    out = []
    for i, r in enumerate(rules):
        mask = rule_to_mask(r, binned_df)
        tr = evaluate_rule(mask & train_mask, labels)
        va = evaluate_rule(mask & val_mask, labels)
        te = evaluate_rule(mask & test_mask, labels)
        # Drop rules with no test fires — they can't be validated.
        if te["support"] < 5:
            continue
        # Compute lift relative to each split's base rate
        tr["lift_vs_base"] = tr["precision"] - base_rates["train"]
        va["lift_vs_base"] = va["precision"] - base_rates["val"]
        te["lift_vs_base"] = te["precision"] - base_rates["test"]
        overfit_flag = (tr["precision"] - te["precision"]) > OVERFIT_TRAIN_TEST_GAP

        out.append({
            **r,
            "train": tr, "val": va, "test": te,
            "train_test_gap": round(tr["precision"] - te["precision"], 4),
            "overfit_flag": overfit_flag,
        })
    log.info(f"  {len(out)} rules passed (had test fires)")
    return out


# ═══════════════════════════════════════════════════════════════════
# DISQUALIFIER ANALYSIS — compare winners vs false-positive losers
# ═══════════════════════════════════════════════════════════════════

def disqualifier_analysis(rule, binned_df, raw_df, labels, train_mask, feature_names_continuous):
    """
    For a given rule, pull out training-set fires. Split into:
      • true positives (winners the rule fires on)
      • false positives (losers the rule fires on)
    For each continuous feature, compute the distribution in each group and
    find features where the two groups differ meaningfully. The top differentiators
    are candidate 'disqualifiers': conditions that, when added, would exclude the
    false positives without hurting the true positives.

    Returns a list of up to 5 disqualifier candidates ranked by how much they'd
    improve precision.
    """
    mask = rule_to_mask(rule, binned_df) & train_mask
    if mask.sum() < 20:
        return []

    fires_labels = labels[mask]
    fires_df = raw_df[mask]
    tp_mask = fires_labels == 1
    fp_mask = fires_labels == 0

    n_tp = int(tp_mask.sum())
    n_fp = int(fp_mask.sum())
    if n_tp < 5 or n_fp < 5:
        return []

    candidates = []
    for feat in feature_names_continuous:
        if feat not in fires_df.columns:
            continue
        if feat in [c["feature"] for c in rule["conditions"]]:
            continue   # don't propose the same feature as a disqualifier
        tp_vals = fires_df.loc[tp_mask, feat].dropna()
        fp_vals = fires_df.loc[fp_mask, feat].dropna()
        if len(tp_vals) < 5 or len(fp_vals) < 5:
            continue
        # Try several candidate thresholds: the 25/50/75th percentile of combined dist
        all_vals = np.concatenate([tp_vals.values, fp_vals.values])
        candidate_thresholds = np.quantile(all_vals, [0.25, 0.5, 0.75])

        for thresh in candidate_thresholds:
            for direction in (">", "<"):
                if direction == ">":
                    excluded_tp = int((tp_vals > thresh).sum())
                    excluded_fp = int((fp_vals > thresh).sum())
                else:
                    excluded_tp = int((tp_vals < thresh).sum())
                    excluded_fp = int((fp_vals < thresh).sum())

                remaining_tp = n_tp - excluded_tp
                remaining_fp = n_fp - excluded_fp
                if remaining_tp + remaining_fp < 10:
                    continue
                new_precision = remaining_tp / (remaining_tp + remaining_fp)
                old_precision = n_tp / (n_tp + n_fp)
                precision_gain = new_precision - old_precision
                # Only meaningful if we remove more FPs than TPs
                if excluded_fp > excluded_tp and precision_gain > 0.03:
                    candidates.append({
                        "feature": feat,
                        "condition": f"{feat} {'<=' if direction == '>' else '>='} {thresh:.4g}",
                        "thresh": float(thresh),
                        "direction": "exclude_if_greater" if direction == ">" else "exclude_if_less",
                        "excluded_tp": excluded_tp,
                        "excluded_fp": excluded_fp,
                        "new_precision_train": round(float(new_precision), 4),
                        "precision_gain_train": round(float(precision_gain), 4),
                        "fp_excluded_per_tp_excluded": round(excluded_fp / max(1, excluded_tp), 2),
                    })

    # Rank by precision gain
    candidates.sort(key=lambda c: -c["precision_gain_train"])
    return candidates[:5]


# ═══════════════════════════════════════════════════════════════════
# LIVE RULE EVALUATION — check which rules fire on current scan data
# ═══════════════════════════════════════════════════════════════════

def evaluate_live(rule, live_binned_row):
    """
    Given a rule and a single live row of binned features, return True if
    all conditions are satisfied.
    """
    for cond in rule.get("conditions", []):
        feat = cond["feature"]
        if feat not in live_binned_row:
            return False
        if int(live_binned_row[feat]) not in cond["bins"]:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATION — the function the server calls
# ═══════════════════════════════════════════════════════════════════

def mine_all_for_cell(rows, feature_names, winner_column, split_cols,
                     min_precision=DEFAULT_MIN_PRECISION,
                     min_support_frac=DEFAULT_MIN_SUPPORT_FRAC,
                     min_lift=DEFAULT_MIN_LIFT,
                     methods=("univariate", "tree", "apriori"),
                     progress_cb=None):
    """
    Main entry point for rule mining on one horizon.

    rows: list of dicts, each containing all feature values plus 'label'
          (0/1 for winner) plus the train/val/test split indicator columns.
    feature_names: list of feature names to mine over (subset of FEATURE_NAMES_V2)
    winner_column: name of the label column (e.g., 'label_4h')
    split_cols: dict {'train': col_name, 'val': col_name, 'test': col_name}
                where each column contains boolean split membership

    Returns: {
        'rules': [list of validated rules with train/val/test metrics],
        'base_rates': {'train': ..., 'val': ..., 'test': ...},
        'bin_edges': {feature: edges},
        'bin_labels': {feature: labels},
        'stats': {...counts per method, timings...},
    }
    """
    def prog(p, m):
        if progress_cb: progress_cb(p, m)
        log.info(f"[rule mining] {p}% — {m}")

    if len(rows) < 500:
        raise ValueError(f"Too few rows to mine: {len(rows)}")

    prog(5, f"Building DataFrame from {len(rows)} rows...")
    df = pd.DataFrame(rows)
    labels = df[winner_column].astype(int).values
    train_mask = df[split_cols["train"]].astype(bool).values
    val_mask   = df[split_cols["val"]].astype(bool).values
    test_mask  = df[split_cols["test"]].astype(bool).values

    base_rates = {
        "train": float(labels[train_mask].mean()),
        "val":   float(labels[val_mask].mean()) if val_mask.sum() > 0 else 0.0,
        "test":  float(labels[test_mask].mean()) if test_mask.sum() > 0 else 0.0,
    }

    # Filter out excluded features
    minable = [f for f in feature_names if f not in EXCLUDE_FROM_MINING and f in df.columns]

    # Min support in absolute terms
    n_train = int(train_mask.sum())
    min_support_count = max(MIN_SUPPORT_COUNT_FLOOR, int(min_support_frac * n_train))

    prog(10, f"Binning {len(minable)} features...")
    binned_df, bin_edges, bin_labels_map = build_binned_dataframe(df, minable, train_mask)
    log.info(f"Binned {len(binned_df.columns)} features, base_rate={base_rates['train']:.3f}, "
             f"min_support_count={min_support_count}")

    all_rules = []
    method_counts = {}

    if "univariate" in methods:
        prog(20, "Method 1: univariate screening...")
        u_rules = mine_univariate(binned_df, labels, train_mask,
                                   min_precision, min_support_count, min_lift)
        all_rules.extend(u_rules)
        method_counts["univariate"] = len(u_rules)

    if "tree" in methods:
        prog(40, "Method 2: decision trees...")
        t_rules = mine_trees(binned_df, labels, train_mask,
                             min_precision, min_support_count, min_lift)
        all_rules.extend(t_rules)
        method_counts["tree"] = len(t_rules)

    if "apriori" in methods:
        prog(60, "Method 3: apriori pattern mining...")
        a_rules = mine_apriori(binned_df, labels, train_mask,
                                min_precision, min_support_count, min_lift)
        all_rules.extend(a_rules)
        method_counts["apriori"] = len(a_rules)

    prog(80, f"Deduplicating {len(all_rules)} raw rules...")
    deduped = dedupe_rules(all_rules)
    log.info(f"After dedupe: {len(deduped)} unique rules")

    prog(85, "Validating rules on held-out splits...")
    validated = validate_rules(deduped, binned_df, labels, train_mask, val_mask, test_mask, base_rates)

    prog(90, "Disqualifier analysis on top rules...")
    # Expensive — only run on top 50 rules by test precision
    validated.sort(key=lambda r: -r["test"]["precision"])
    continuous_features = [f for f in feature_names if f not in BINARY_FEATURES and f not in EXCLUDE_FROM_MINING]
    for r in validated[:50]:
        r["disqualifiers"] = disqualifier_analysis(
            r, binned_df, df, labels, train_mask, continuous_features)
    # Rules beyond top 50 get empty disqualifiers
    for r in validated[50:]:
        r["disqualifiers"] = []

    # Annotate each rule with an English description and a stable ID
    for r in validated:
        r["english"] = rule_to_english(r, bin_labels_map)
        r["id"] = hashlib.md5(r["signature"].encode()).hexdigest()[:12]

    # Serialize bin_edges for JSON (numpy arrays -> lists)
    edges_json = {}
    for f, e in bin_edges.items():
        edges_json[f] = e.tolist() if e is not None else None

    prog(100, f"Done. {len(validated)} validated rules.")
    return {
        "rules": validated,
        "base_rates": base_rates,
        "bin_edges": edges_json,
        "bin_labels": bin_labels_map,
        "stats": {
            "n_rows": len(rows),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_features_mined": len(binned_df.columns),
            "raw_rules_by_method": method_counts,
            "n_deduped": len(deduped),
            "n_validated": len(validated),
            "min_precision": min_precision,
            "min_support_frac": min_support_frac,
            "min_support_count": min_support_count,
            "min_lift": min_lift,
        },
    }
