# V4 + Phase 2 — narrative & lessons

_Last updated: 2026-05-15 (after v4.2_submit Kaggle result)_

This document is the load-bearing context if the chat history is lost. Read it before opening Phase 3.

---

## 1 — Where we stand

| Run | Val NDCG@5 | Kaggle public | Notes |
|---|---|---|---|
| **V4 ensemble** (`run_v4.py`) | 0.42512 | **0.42021** | 8-model rank-averaged ensemble. Current Kaggle leader. |
| V4 best single (lambdarank_bal15) | 0.42191 | _(not submitted alone)_ | seed=456, label_gain="0,1,15". The Phase 2 anchor. |
| Phase 2 winner `lg_0_2_15` (`run_phase2_submit.py` → v4.2) | 0.42258 | **0.41639** | Δ vs V4 ensemble: val +0.00067 / Kaggle **−0.00382** → overfit on val. |

**Verdict**: V4 ensemble (Kaggle = 0.42021) remains the production reference. Single-model variants underperform on Kaggle.

---

## 2 — What V4 introduced (commit `8bbfd5d`)

`run_v4.py` is a 4-stage staged pipeline with checkpointing in `models/v4/` (gitignored). Now also writes the canonical artifact layout under `models/v4_stage3/` + `artifacts/v4_stage3/` via `src/artifacts.py` helpers.

**Stage 3 ensemble** — 8 LightGBM members:

| Member | seed | label_gain (remapped {0,1,2}) | Special |
|---|---|---|---|
| lambdarank_base | 42 | 0,1,31 | default IPW |
| lambdarank_click3 | 123 | 0,3,31 | upweight clicks |
| lambdarank_bal15 | 456 | 0,1,15 | balanced gain — **this is the anchor** |
| lambdarank_book50 | 789 | 0,1,50 | num_leaves=300, lr=0.05 |
| lambdarank_noipw | 2024 | 0,1,31 | no IPW (weight=1) |
| rank_xendcg | 314 | n/a | rank_xendcg objective, num_leaves=350 |
| lambdarank_randup | 555 | 0,2,25 | random_bool=1 rows ×2 weight |
| booking_clf | 666 | n/a | binary booking_bool, AUC |

Combined via NDCG-proportional rank averaging (`v4_rank`). `BASE_PARAMS`:
- objective=lambdarank, lr=0.03, num_leaves=400, min_child=50, subsample=0.7, colsample=0.6, reg_alpha=0.1, reg_lambda=1.0.

**Features built by `src/features.py:build_features()`**: 143 columns. Groups: raw 22, listwise 44, interaction 14, competitor 12, price 8, TE singles 7, quality 6, TE cross 5, missing-flag 5, temporal 4, visitor 4, hotel-agg price 3, hotel-agg count 3, hotel-agg dest 3, hotel-agg position 1, TE count 1, TE book-given-click 1. KFold OOF (`seed=42`, 5 folds) for all target-derived features in `hotel_aggregates()`.

**IPW**: `compute_position_propensity()` fits propensity from `random_bool=1` rows. `compute_sample_weights()` applies `max_prop/propensity[position]` to non-random rows, sets random rows = 1.0, clips to `[0.1, 10.0]`.

**Validation split**: `split_val(val_frac=0.1, random_state=42)` — 179,816 train / 19,979 val srch_ids (no overlap).

---

## 3 — Phase 2: label-gain sweep around the bal15 anchor

**Goal**: tune `label_gain` for the single-model anchor, looking for a setting that beats `0.42191`.

**Configs** (all with seed=456, IPW default, full V4 feature set):
- `lg_0_1_10`, `lg_0_1_12`, `lg_0_1_15` (anchor reproduction), `lg_0_1_18`, `lg_0_1_20`, `lg_0_2_15`.

**First run (TAINTED)**: launched 2026-05-15 ~21:34. All 6 results landed −0.0024 below the anchor:

| config | ndcg5 (tainted) | best_iter |
|---|---|---|
| lg_0_1_10 | 0.42096 | 453 |
| lg_0_1_12 | 0.42154 | 554 |
| lg_0_1_15 | **0.41951** ← reproduction failed | 421 |
| lg_0_1_18 | 0.41960 | 338 |
| lg_0_1_20 | 0.42130 | 447 |
| lg_0_2_15 | 0.42063 | 442 |

Rows preserved as `exp_id = P2_INVALID_*` (`kept=NO`) in `experiment_logs/experiment_tracker.csv`.

---

## 4 — The LightGBM Dataset binning bug

**Symptom**: `lg_0_1_15` reproduction gave 0.41951 instead of V4 anchor 0.42191 (Δ = −0.00240), with `best_iter=421` (truncated; V4 was ~600).

**Root cause** (confirmed empirically): the tainted `run_phase2.py` created a single `lgb.Dataset(..., free_raw_data=False)` outside the loop and called `ds_train.construct()` + `ds_val.construct()` explicitly to share binning across all 6 configs.

When you construct a `lgb.Dataset` without passing `params=`, `ds.params is None`. Explicit `.construct()` then bins the histograms with LightGBM defaults — most importantly `data_random_seed=1` (not the training `seed=456`). The training params passed later via `lgb.train(params, ds_train)` only reach `ds.params` via `_update_params`, but **after** the bins are already fixed.

V4 stage 3 never pre-constructs — `lgb.train(params, ds_train)` constructs lazily with `seed=456` in scope. So V4's bins were sampled with seed 456; the tainted Phase 2's bins were sampled with seed 1. Different bin edges on 143 continuous features → different splits → ~ −0.0024 NDCG drop.

**Verification** (`run_phase2_anchor_check.py`, `run_id=phase2_anchor_check`): fresh `lgb.Dataset` inside the loop, no `.construct()`, no `free_raw_data`. Identical config to `lg_0_1_15`. Result: NDCG@5 = **0.42191** — bit-for-bit anchor reproduction. Δ = 0.00000. `best_iter=632`. **Hypothesis confirmed.**

**Fix** (committed): `run_phase2.py` now follows V4's pattern exactly — fresh `lgb.Dataset` inside each config iteration, no explicit construct, `del ds_train, ds_val, model; gc.collect()` per iter.

---

## 5 — Phase 2 corrected results

Launched 2026-05-15 ~22:21 with the V4-style pattern. All 6 configs use seed=456, IPW default, 143 features.

| config | label_gain | val NDCG@5 | val Recall@5 | best_iter |
|---|---|---|---|---|
| lg_0_1_10 | 0,1,10 | 0.42192 | 0.6415 | 647 |
| lg_0_1_12 | 0,1,12 | 0.41870 | 0.6373 | 240 |
| lg_0_1_15 (anchor) | 0,1,15 | **0.42191** ← reproduces V4 | 0.6408 | 632 |
| lg_0_1_18 | 0,1,18 | 0.42229 | 0.6418 | 315 |
| lg_0_1_20 | 0,1,20 | 0.42116 | 0.6399 | 389 |
| **lg_0_2_15** ★ | **0,2,15** | **0.42258** | 0.6412 | 470 |

`lg_0_1_15` reproducing 0.42191 with the corrected pattern is the determinism check that the anchor is now stable.

---

## 6 — v4.2 submission (lg_0_2_15)

`run_phase2_submit.py` retrained `lg_0_2_15` (label_gain="0,2,15", seed=456, IPW default) on **full train**, 470 rounds (the val-determined `best_iter`), no validation set, no early stopping. Then featurized test (`agg_source=train_raw`) and predicted.

- Submission: `submissions/submission_phase2_best_20260515_225726.csv` (4,959,183 rows, 199,549 srch_ids).
- Local val NDCG@5: 0.42258. Kaggle public: **0.41639**.
- Local→Kaggle gap: **−0.00619** (much wider than V4 ensemble's local→Kaggle gap of −0.00491).
- Artifacts: `artifacts/v4.2_submit/` (renamed from `phase2_best_submit`); model in `models/v4.2_submit/model_full.txt`; external copy at `/home/ubuntu/experiment_artifacts/v4.2_submit/`.

**Conclusion**: single-model `lg_0_2_15` is worse on Kaggle than V4 ensemble. Two compounding effects:
1. **Val overfit**: 0.42258 was likely a noisy local maximum (Phase 2 best was 0.0007 above the anchor; not robust).
2. **Loss of ensemble diversity**: V4's 8 members hedge against test distribution shift; a single model can't.

---

## 7 — What this means for next experiments

1. **Stop optimizing a single model in isolation.** Local val NDCG@5 ≥ 0.42258 is no longer a useful signal on its own. Either move to ensemble-level evaluation, or treat single-model val deltas as directional only.

2. **The local→Kaggle gap is the real metric.** V4 ensemble has a gap of −0.00491; v4.2 single has −0.00619. **Closing the gap matters more than peak local val NDCG@5.** Anything that improves generalization (more aggressive smoothing on TEs, dropping prop_avg_position, IPW alternatives, hard-negative reweighting) should be measured by *reducing the gap*, not by val NDCG@5.

3. **Phase 3+ keep producing single-model val numbers**, but interpret them with these caveats — directional, not selection.

4. **Final submissions should remain ensembles.** Phase 2 lesson: the v4.2 single-model submission was a stress-test, not a candidate path to beating V4 on Kaggle.

5. **Determinism guard rails**: any new pipeline must use `lgb.Dataset` per-config (V4-style), never pre-construct, and verify the `lg_0_1_15` anchor reproduces 0.42191 before drawing conclusions.

---

## 8 — File map (current)

```
DataMiningAssignment2/
├── README.md                       # top-level overview + how to run
├── pipelines/                      # executable training scripts (invoke via `python -m`)
│   ├── v4_ensemble.py              # 8-model V4 ensemble pipeline (production reference)
│   ├── v3_baseline.py              # V3 single-model baseline (historical)
│   ├── phase2_labelgain.py         # corrected label-gain sweep
│   ├── phase2_anchor_check.py      # single-config diagnostic (V4-style pattern verification)
│   ├── phase2_submit.py            # v4.2 submission generator
│   └── phase3_weighting.py         # weighting sweep (scaffolded, not yet run)
├── scripts/aggregate_results.py    # promote per-run artifact rows → experiment_logs/
├── src/                            # data_loader, features, evaluate, submission, artifacts, config
├── notebooks/01..04                # EDA + diagnostics
├── docs/
│   ├── v4_phase2_summary.md        # this file
│   ├── next_steps.md               # operational plan (Phase 3 → 7)
│   └── archive/                    # SNAPSHOT.md, EDA_PLAN.md (historical)
├── experiment_logs/
│   ├── experiment_tracker.csv      # one row per exp (V4_*, P2_*, P2_INVALID_*, V4.2_SUBMIT)
│   ├── model_results.csv           # one row per single model
│   ├── ensemble_results.csv        # one row per ensemble
│   └── feature_audit.csv           # 143 rows: gain, split, risk labels, decision
├── artifacts/
│   ├── phase2_labelgain/           # 6 corrected Phase 2 models (metadata; .npy gitignored)
│   ├── phase2_anchor_check/        # 1 verification model
│   └── v4.2_submit/                # final v4.2 submission artifacts
├── models/                         # boosters .txt (gitignored)
├── submissions/                    # final CSVs (gitignored)
└── logs/                           # training logs (gitignored)
```

External archive of submission artifacts: `/home/ubuntu/experiment_artifacts/v4.2_submit/`.

---

## 9 — Open items (for next session, before Phase 3)

- The 5 cross-key TEs (`prop_dest_book_rate`, `site_dest_book_rate`, `prop_site_book_rate`, `cpair_book_rate`, `site_country_book_rate`) are marked `TEST_HIGHER_SMOOTHING` in `feature_audit.csv` — they're high-gain but high overfit-risk; sweep `prior_weight` in Phase 5.
- `prop_avg_position` is `TEST_DROP` — gain rank #7 but encodes position bias even with OOF.
- Phase 3 prepared as `run_phase3_weighting.py` (not yet run).
- Read `next_steps.md` for tomorrow's ordered queue.
