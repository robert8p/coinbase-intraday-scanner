# v2 Stage 2 — Rule Mining

## What's new vs Stage 1

Stage 1 (v2 Classifier) trained a LightGBM model per (k, H) cell and ranked coins
by its predicted probability. We found that the signal was narrow — great lift
at k=1/h=4 but it didn't replicate to neighboring cells, suggesting more
statistical artifact than robust pattern.

Stage 2 (Rule Mining) changes the framing entirely. Instead of asking "rank all
coins by some opaque score," it asks: "find describable technical setups that
precede winners at high precision, and identify what distinguishes winners from
false-positive losers."

## How it works

1. **Build labeled dataset** — for every historical (date, scan_hour, coin), compute
   all 53 v2 features AND binary labels for whether the coin touched absolute
   +X% within each of several horizons (4h, 6h, 8h by default).
2. **Mine rules** across three methods in parallel:
   - **Univariate** — single-feature bin rules (e.g., `trend_str_atr ∈ {H, VH}`)
   - **Decision trees** — precision-optimized tree ensemble; each leaf is a rule
   - **Apriori** — frequent feature-bin combinations (up to 3-way)
3. **Validate on held-out splits** — every rule gets train/val/test precision. Rules
   with train-test gap > 15pp are flagged as likely overfit.
4. **Cross-horizon retest** — each rule mined on horizon H is automatically
   re-evaluated against OTHER horizons' labels on the same test rows. A rule
   that works at 4h but fails at 6h and 8h is suspicious.
5. **Disqualifier analysis** — for each top rule, look at its false-positive
   losers on training data. For each continuous feature (not already in the
   rule), find thresholds that exclude more FPs than TPs. Output: "adding
   `beta_to_btc > 0.7` would exclude 23 FPs while only excluding 4 TPs,
   raising precision from 68% to 82%."

## UI

New **Rules** tab (now the default landing page). Sections:

1. **Mine Rules panel** — threshold (default +2%), horizons (default 4h/6h/8h),
   method checkboxes, min_precision/min_support/min_lift sliders, Mine button.
2. **Mining Progress** — live progress bar with phase/percentage.
3. **Catalogs list** — one chip per (threshold, horizon) pair with rule count
   and test base rate. Click a chip to load its rules. × button to delete.
4. **Rules table** — sortable columns: methods, conditions summary, train P,
   test P, test N, lift, train-test gap, cross-horizon precision badges.
   Paginated at 25 per page. Click any row to see full detail.
5. **Rule detail** — full English condition statement, train/val/test metrics
   table, cross-horizon retest table, full conditions with bin labels, and
   disqualifier analysis table.

## Endpoints

- `POST /api/v2/mine_rules` — start mining; body: `{threshold_pct, horizon_hours[], methods[], min_precision, min_support, min_lift}`
- `GET /api/v2/mine_rules/progress` — live progress
- `GET /api/v2/rules/catalogs` — list all catalogs
- `GET /api/v2/rules/catalog/{threshold_bps}/{horizon_hours}` — full catalog
- `GET /api/v2/rules/rule/{threshold_bps}/{horizon_hours}/{rule_id}` — rule detail with bin_labels
- `DELETE /api/v2/rules/catalog/{threshold_bps}/{horizon_hours}` — delete a catalog
- `GET /api/v2/rules/diagnostic` — download all catalogs as JSON

Note: `threshold_bps` is `threshold_pct * 10000` as an integer (e.g., 2% = 200).
This avoids URL float routing issues.

## Files added or changed

- **`v2.py`** — added `did_touch_absolute()` and `build_rows_for_rule_mining()` at end
- **`v2_rules.py`** — new (884 lines). All mining logic, validation, disqualifier analysis, cross-horizon retest.
- **`server.py`** — Stage 2 section (`_v2_run_mining`, 6 new endpoints) under the existing Stage 1 endpoints
- **`src/App.jsx`** — new `RulesTab` component (620+ lines)

## How to use

1. Deploy this zip (push to GitHub; Render auto-deploys)
2. Open the app — the **Rules** tab is the default landing page
3. Confirm defaults (threshold 2%, horizons 4h/6h/8h, all 3 methods) or adjust
4. Click **⛏ Mine Rules** — takes 5-15 min depending on horizons
5. Watch the progress bar
6. When done, click one of the catalog chips to view its rules
7. Sort the table by "Test P" (descending) — green cells are precision ≥ 70%
8. Click any row to see full detail and disqualifier analysis

## What to expect

**Honest framing:** based on v2 Stage 1 results, the underlying signal in
Coinbase intraday data is narrow. Rule mining might surface more interpretable
setups than ML ranking did, OR it might confirm that no robust setups exist
at these thresholds/horizons. Either outcome is useful.

**Good outcome:** several rules show test precision ≥ 70%, lift ≥ +10pp,
cross-horizon retest green on at least one other horizon, train-test gap
< 10pp.

**Mixed outcome:** rules show lift but test precision is in the 55-65% range.
Disqualifier analysis may help push some of them higher — but at this point
you'd want to forward-test live, not keep tuning.

**Bad outcome:** zero rules pass the filters, or everything is flagged as
overfit. Confirms v2 Stage 1 finding that the signal is very narrow.

## What's NOT in this stage

- **Live signals** — "which rules are firing right now on live data." Deferred
  because we need validated rules first. Once you confirm 1-2 rules that hold
  up, we add a live-signals panel that surfaces any coin matching those rules
  at the latest scan time.
- **Forward outcome recording** — the Outcomes (v1) tab still works for v1's
  top-10 tracking. Outcomes for rule firings need a Stage 3 build.
