# Next steps (tomorrow's queue)

_Last updated: 2026-05-15_

Read `docs/v4_phase2_summary.md` first for full context. This file is the operational plan.

**Hard rule (carried from Phase 2 lesson):** every new pipeline must use the V4-style `lgb.Dataset` pattern — fresh `lgb.Dataset(...)` inside each config loop, no explicit `.construct()`, no `free_raw_data` flag, `del ds_train, ds_val, model; gc.collect()` after each iter. Otherwise the binning seed reverts to default `data_random_seed=1` and results drift ~−0.0024 NDCG@5. See summary §4.

**Anchor invariant:** any new run that includes `label_gain="0,1,15"` with `seed=456` on the full V4 feature set should produce val NDCG@5 ≈ **0.42191**. If it doesn't, stop and diagnose before drawing conclusions (cf. `pipelines/phase2_anchor_check.py`).

**Selection metric reminder:** the local→Kaggle gap matters more than peak local val NDCG@5. V4 ensemble's gap is −0.00491; Phase 2 single's gap was −0.00619. Anything that *closes the gap* — better generalization, less overfit on val — is a real win, even if local NDCG@5 only moves a little.

---

## Phase 3 — Weighting / IPW sweep (RUN FIRST)

**Script:** `pipelines/phase3_weighting.py` (already scaffolded, **not yet run**)

**Configs (14 total):** 7 weighting variants × 2 label_gains.

| weighting variant | semantics |
|---|---|
| `ipw_default` | V4 default. IPW on non-random rows, clipped [0.1, 10.0]. |
| `no_ipw` | All weights = 1. Baseline. |
| `ipw_positive` | IPW only on `click_bool=1` rows. Unclicked = 1.0. |
| `ipw_clip3` | IPW clipped [0.1, 3.0]. |
| `ipw_clip5` | IPW clipped [0.1, 5.0]. |
| `rand_up_1.5` | IPW default × 1.5 on `random_bool=1` rows. |
| `rand_up_2.0` | IPW default × 2.0 on `random_bool=1` rows. |

Label gains: `0,2,15` (Phase 2 winner) and `0,1,15` (V4 bal15 anchor).

**Run (from project root):**
```
uv run --no-sync python -m pipelines.phase3_weighting 2>&1 | tee logs/phase3_$(date +%Y%m%d_%H%M%S).log
```
Expected wall-clock: ~3–4 min × 14 = ~45–60 min.

**Aggregate after completion:**
```
uv run --no-sync python scripts/aggregate_results.py \
    --run-id phase3_weighting --phase P3 \
    --change-summary "Weighting / IPW sweep (7 variants × 2 label_gains)"
```

**Decision gate (P3 → P4):**
- If a variant beats the V4 ensemble's `local→Kaggle` gap proxy (i.e., val NDCG@5 ≥ 0.42191 AND `mean_booked_rank` improves vs `ipw_default_lg_0_1_15`), pin it as the new weighting default.
- If `no_ipw` competes with `ipw_default` (Δ < 0.0005), prefer `no_ipw` — fewer assumptions, simpler.
- Pure raw NDCG@5 maxima are *directional only*. Do not submit a single model. Roll the winning weighting into the V4 ensemble retrain at the end.

---

## Phase 4 — Feature pruning

**Script:** _not yet written_. Will be `pipelines/phase4_features.py`.

**Inputs from this session:**
- `experiment_logs/feature_audit.csv` (143 features, with `decision` column).
- Specifically flagged:
  - **`prop_avg_position`** — `decision=TEST_DROP`. Gain rank #7 (~513k), but encodes position bias even with OOF k-fold. Top suspect for closing the local→Kaggle gap.
  - **Cross-key TEs** marked `TEST_HIGHER_SMOOTHING` (handled in Phase 5):
    - `prop_dest_book_rate` (#2 by gain, ~829k)
    - `site_dest_book_rate`
    - `prop_site_book_rate` (#17 by gain, ~261k)
    - `cpair_book_rate`
    - `site_country_book_rate`
  - **Zero-gain candidate:** none currently — Phase 2 importances all > 0.

**Plan (4 ablations):**

| ablation | what it removes | comparison |
|---|---|---|
| `drop_prop_avg_position` | `prop_avg_position` only | vs Phase 3 winner |
| `drop_position_derived` | `prop_avg_position` (currently the only position-derived feature in features.py) | same as above; will diverge if we add more in future |
| `top100_only` | keep only top-100 features by Phase 2 mean gain | tests whether bottom-43 features are noise |
| `top80_only` | keep only top-80 | aggressive pruning |

Use the winning weighting + label_gain from Phase 3. 1 run per ablation = 4 single-model runs.

**Decision gate (P4 → P5):**
- If `drop_prop_avg_position` improves or matches val NDCG@5, drop it from `src/features.py` (or gate behind a flag) and use the reduced feature set in P5+.
- If `top100_only` matches or beats full, switch to top-100 going forward — speeds up every later run.

---

## Phase 5 — Target encoding variants

**Script:** _not yet written_. Will be `pipelines/phase5_te.py`.

**Goal:** reduce the local→Kaggle gap on cross-key TEs without losing the gain (~1.1M combined across the 5 cross-TEs).

**Sweeps:**

1. **Smoothing `prior_weight` (alpha)** for the 5 cross-key TEs. Current: `prop_dest_book=10, site_dest_book=15, prop_site_book=15, cpair_book=20, site_country_book=20`. Try `2×` and `4×` (e.g., `prop_dest_book=40` to suppress rare-pair overfit).
2. **Count thresholds**: only encode if group has ≥ N occurrences in train; else fall back to global / single-key TE.
3. **Fallback hierarchy**: e.g., `prop_dest` → fallback to `prop_id` TE → fallback to `dest_id` TE → fallback to global.
4. **TE rank within query**: take each TE and compute its rank within `srch_id` — relative signal, less drift.

Use the winning weighting + features from P4.

**Decision gate (P5 → P6):**
- If higher smoothing reduces overfit (val drops slightly but `mean_booked_rank` improves or train→val gap narrows), pin it.
- TE-rank-within-query is the most likely structural win — propose adding to `src/features.py` `listwise_features()`.

---

## Phase 6 — Hard-negative diagnostics

**Script:** use `notebooks/04_model_diagnostics.ipynb` (framework already exists; needs to run against the post-P5 booster).

**Outputs:**
- Recall@K curve. Where does the booked hotel fall in the ranking?
- Hard-negative analysis: which non-booked hotels does the model rank above the booked hotel? What features differentiate them?
- Easy-wins-missed: queries where the booked hotel has top features but the model still ranks it < 5.
- Segment failure analysis: NDCG@5 by booking_window, length_of_stay, query size, is_family, cold-start prop_id presence.
- Fairness baselines: NDCG@5 by site_id, by prop_country, by random_bool.

**Decision gate (P6 → ensemble):**
- Identify 2–3 segment failures. Decide whether they motivate a targeted feature (e.g., overpriced penalty, family×price, cold-start fallback).
- If yes: implement in a separate `src/features_v6.py` (don't modify `features.py` core). Re-run P3 winner with new features as a sanity check.
- If no: skip to ensemble.

---

## Phase 7 — Ensemble rebuild

**Script:** new — call it `pipelines/v5_ensemble.py` (don't confuse with the deleted old `run_v5.py`).

**Plan:**
- Start from the V4 ensemble template (8 members).
- Replace each member's `label_gain` and weighting with the P3 winner(s).
- Replace the feature set with the P4 winner.
- Optionally add 1–2 new members trained with `dart` boosting or `xgboost` ranking for diversity.
- Generate one Kaggle submission. **This is the actual production candidate.**

**Sanity check before submitting:**
- Re-run the anchor: a `seed=456, label_gain="0,1,15"` single model with V4 features should still hit 0.42191 ± 0.0002.
- Ensemble val NDCG@5 must beat V4's 0.42512 by a margin that justifies the assumption it'll beat 0.42021 on Kaggle. Without a margin, do not submit — V4 is still production.

---

## Order of operations (1-line summary)

```
Phase 3  (pipelines/phase3_weighting.py) — done in 1 hour, aggregate, pick winner
Phase 4  (pipelines/phase4_features.py)  — write + run, ~30 min, pick winner
Phase 5  (pipelines/phase5_te.py)        — write + run, ~1 hour, pick winner
Phase 6  (notebooks/04_model_diagnostics.ipynb on P5 best) — analysis, no training
Phase 7  (pipelines/v5_ensemble.py)      — final ensemble + Kaggle submission
```

Invocation pattern is always `python -m pipelines.<name>` from the project root.

## Things explicitly NOT to do (carried from constraints)

- ❌ No new submissions until Phase 7.
- ❌ No long single-model runs targeting val NDCG@5 maxima.
- ❌ No core changes to `src/features.py` until P5 / P6 decisions are made — additions go in separate `_v6` files.
- ❌ No ensembling until P3–P5 winners are stable.
- ❌ Never pre-construct `lgb.Dataset` outside the per-config loop — see anchor invariant above.
