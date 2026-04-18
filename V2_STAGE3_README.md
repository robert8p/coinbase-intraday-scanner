# v2 Stage 3 — Live Scanner & Outcome Recording

## What this is

Stage 3 is the empirical test. You've spent weeks building mining infrastructure
that produced rules with +27pp lift in validation. Stage 3 answers: does that
hold up on real, forward data?

Each pinned rule is evaluated against fresh Coinbase data every 4 hours. Every
fire is recorded. After each rule's horizon elapses, an outcome is recorded
automatically (was +X% hit?). A dashboard compares live precision to validation
precision. After 2-3 weeks you'll know whether any rule actually works.

## What's new in this stage

### Backend

**`v2_live.py`** (588 lines) — live scanner + outcome recorder module:
- `pin_rule(...)` / `unpin_rule(...)` — manage active rules
- `evaluate_pinned_rule(...)` — check if a rule fires on a feature vector
- `scan_live(...)` — evaluate all pinned rules against current features, persist fires
- `compute_outcome(...)` — given entry + future bars, check if +X% was hit in H hours
- `record_outcomes(...)` — resolve all pending fires whose horizons have elapsed
- `aggregate_stats(...)` — per-rule live precision, rolling windows, delta vs validation

**Storage layout (JSONL — append-only, durable):**
- `/data/live_v2/pinned_rules.json` — active rules with conditions + disqualifiers + bin edges
- `/data/live_v2/fires.jsonl` — one JSON per rule fire
- `/data/live_v2/outcomes.jsonl` — one JSON per resolved outcome (fast aggregation)

**New endpoints (9):**
- `POST /api/v2/live/pin` — pin a rule (optionally with disqualifier)
- `DELETE /api/v2/live/pin/{pin_id}` — unpin
- `GET /api/v2/live/pinned` — list pinned rules
- `POST /api/v2/live/scan` — trigger a live scan now
- `GET /api/v2/live/scan/status` — current scan state + last result
- `POST /api/v2/live/record_outcomes` — trigger outcome resolution now
- `GET /api/v2/live/stats` — aggregated stats per rule
- `GET /api/v2/live/fires` — recent fires (filterable by pin_id)
- `GET /api/v2/live/diagnostic` — download everything

**Scheduled jobs:**
- Live scan runs every 4h at :06 (1min after v1 scans at :05 to not collide)
- Outcome recorder runs hourly at :15

### Frontend

New **Live** tab (default landing page):
- Pinned rules table with columns: rule English, validation precision, live
  precision, delta, hits/resolved, pending, avg max %, unpin button
- Color-coded deltas: green ≤5pp below validation (healthy), yellow 5-15pp
  (warning), red >15pp (signal likely collapsed)
- Manual "Run scan now" and "Resolve outcomes now" buttons
- Recent fires table with pin_id filter (click any pinned rule row to filter)
- "Pin to Live" + "Pin with top DQ" buttons added to Rules tab rule detail view

## How to use

### Day 1: Pin rules

1. Open the **Rules** tab
2. Click a catalog chip (e.g., `+2.0% / 8h`) to load its rules
3. Click the top rule to expand detail view
4. Click **📌 Pin to Live** (green button next to method badges) to pin
   the rule as-is
5. Optionally click **📌 Pin + top DQ** (orange button) to pin the rule WITH
   its top disqualifier applied — this is a separate pin from the unrefined
   one, so you can track both
6. Repeat for 2-4 rules you want to forward-test

### Day 1-14: Let it run

- Every 4h the scheduler runs a live scan
- Fires accumulate in the Live tab (filter by clicking a pinned rule row)
- Hourly, outcomes are resolved as horizons elapse
- The "Live P" and "Δ" columns populate as resolved outcomes accumulate

### Day 14+: Interpret

For each pinned rule:
- **Δ between −5 and +5pp** → rule is real, forward performance matches validation.
  Pick the best and scale up.
- **Δ between −15 and −5pp** → moderate degradation. Could be noise, could be
  regime drift. Need more samples.
- **Δ < −15pp** → signal has collapsed. The validation measurement was not
  predictive. Important negative result.

With ~10 fires/day × 4 pinned rules × 14 days ≈ 500+ resolved fires, we'll
have statistically meaningful samples per rule.

## What's NOT in Stage 3

- **Automated trading** — this is reporting only. The scanner tells you what's
  firing; you decide whether to act.
- **Execution quality modeling** — entry price is the close of the last 15-min
  bar before scan. Real execution would include spread + slippage; actual P&L
  will lag reported precision.
- **Paper-trading P&L** — we record whether +X% was hit; we don't model stop
  losses, partial exits, or position sizing. A rule with 60% precision isn't
  automatically profitable without understanding the losing tail.
- **Signal push notifications** — you need to actively check the UI.

## Files added/changed

- **`v2_live.py`** — NEW (588 lines)
- **`server.py`** — Stage 3 section added (~270 lines) + scheduler wiring
- **`src/App.jsx`** — NEW `LiveTab` component (~350 lines) + pin buttons in
  RulesTab detail view + tab registration

## Tests

8 unit tests in v2_live.py cover pin/unpin, rule evaluation (with disqualifiers,
NaN, missing features), scan_live, compute_outcome, record_outcomes (with
pending-skip), aggregate_stats, fire filtering.

End-to-end integration test: mock catalog → pin → scan 4 coins → resolve
outcomes → verify per-rule stats. Plus a dedicated test for disqualifier
direction semantics (exclude_if_greater vs exclude_if_less vs boundary values).

All tests pass locally before this zip shipped.

## Deployment notes

Same as all prior stages:
1. Push zip contents to GitHub
2. Render auto-deploys (~8 min)
3. If old code keeps running: Render Shell → `kill 1` → wait 90s
4. Default tab is now **Live**; the Rules tab is still available for mining

Nothing breaks v1 or Stage 1/2 functionality. Scanner (v1), Training (v1),
Outcomes (v1), v2 Classifier, and Rules tabs all remain.
