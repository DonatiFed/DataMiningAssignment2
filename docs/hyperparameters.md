# Hyperparameter Tuning

Tuning is organized as **phased sweeps**, each varying one axis at a time and measured against the [anchor](validation.md). The defining episode was a `Dataset` binning bug that silently invalidated an entire sweep &mdash; the reason determinism rules now precede every run.

## Overview

- **Shared base configuration** &mdash; one `BASE_PARAMS` block underlies every LambdaRank member; only the swept axis changes.
- **Label gain** is the primary tuning axis &mdash; how aggressively to weight bookings vs clicks in the `{0,1,5}` relevance mapping.
- **The binning bug** &mdash; a pre-constructed `lgb.Dataset` shifted every Phase 2 result &minus;0.0024 NDCG@5; caught by the anchor, fixed, re-run.
- **Weighting / IPW sweep** &mdash; 14 configs scaffolded in `pipelines/phase3_weighting.py`, not yet run.
- All single-model sweep numbers are **directional only** &mdash; the ±0.0037 noise floor forbids selecting submissions on them.

---

## Base configuration

Every LambdaRank member shares this base config; ensemble members override only the noted fields ([models.md](models.md)).

```
objective      = lambdarank      metric        = ndcg        eval_at = [5]
learning_rate  = 0.03            num_leaves    = 400         max_depth = -1
min_child_samples = 50           subsample     = 0.7         colsample_bytree = 0.6
reg_alpha      = 0.1             reg_lambda    = 1.0
```

| Parameter | Choice | Rationale |
|---|---|---|
| `num_leaves` | 400 | Large &mdash; 143 features with strong interactions reward a high-capacity tree; regularization controls overfit instead of depth limits. |
| `learning_rate` | 0.03 | Low, paired with early stopping on val NDCG@5 &mdash; best iteration typically 300&ndash;650 rounds. |
| `subsample` / `colsample_bytree` | 0.7 / 0.6 | Row and column subsampling decorrelate trees and act as the main overfit control. |
| `reg_alpha` / `reg_lambda` | 0.1 / 1.0 | L1/L2 penalties; L2-weighted because the goal is shrinkage, not feature elimination. |
| `min_child_samples` | 50 | Prevents leaves splitting on rare ID combinations &mdash; relevant given sparse cross-entity features. |

---

## Label-gain sweep (Phase 2)

`label_gain` maps relevance grades `{0,1,5}` to the gain values LambdaRank uses when computing pairwise &Delta;NDCG. The grade-2 slot is the remapped click level; the third value controls how much a booking outweighs a click. The sweep tuned this around the `bal15` anchor, searching for a single-model setting beating `0.42191`.

**Configs** (all `seed=456`, IPW default, full 143-feature set): `lg_0_1_10`, `lg_0_1_12`, `lg_0_1_15` (anchor reproduction), `lg_0_1_18`, `lg_0_1_20`, `lg_0_2_15`.

**Corrected results** (`pipelines/phase2_labelgain.py`, V4-style `Dataset` pattern):

| Config | label_gain | val NDCG@5 | Recall@5 | best_iter |
|---|---|---|---|---|
| `lg_0_1_10` | 0,1,10 | 0.42192 | 0.6415 | 647 |
| `lg_0_1_12` | 0,1,12 | 0.41870 | 0.6373 | 240 |
| `lg_0_1_15` (anchor) | 0,1,15 | 0.42191 | 0.6408 | 632 |
| `lg_0_1_18` | 0,1,18 | 0.42229 | 0.6418 | 315 |
| `lg_0_1_20` | 0,1,20 | 0.42116 | 0.6399 | 389 |
| **`lg_0_2_15`** &#9733; | **0,2,15** | **0.42258** | 0.6412 | 470 |

`lg_0_1_15` reproducing **0.42191** exactly is the determinism check that the anchor is stable. The nominal winner `lg_0_2_15` beats the anchor by only **+0.00067** &mdash; well inside the ±0.0037 noise floor, so the result is directional, not decisive. The v4.2 submission built from it underperformed on Kaggle ([models.md](models.md)).

---

## The LightGBM binning bug

The first Phase 2 run (2026-05-15) produced six results all landing **~0.0024 below** the anchor &mdash; including the `lg_0_1_15` reproduction itself, which gave 0.41951 instead of 0.42191:

| Config | Tainted run | Corrected run | &Delta; |
|---|---|---|---|
| `lg_0_1_10` | 0.42096 | 0.42192 | +0.00096 |
| `lg_0_1_12` | 0.42154 | 0.41870 | &mdash; |
| `lg_0_1_15` | 0.41951 | 0.42191 | +0.00240 |
| `lg_0_1_18` | 0.41960 | 0.42229 | +0.00269 |
| `lg_0_1_20` | 0.42130 | 0.42116 | &mdash; |
| `lg_0_2_15` | 0.42063 | 0.42258 | +0.00195 |

**Root cause.** The tainted pipeline created a single `lgb.Dataset(..., free_raw_data=False)` *outside* the config loop and called `.construct()` explicitly to share binning across all six configs. A `Dataset` constructed without `params=` has `ds.params is None`; explicit `.construct()` then bins the 143 continuous features with LightGBM defaults &mdash; including `data_random_seed=1` instead of the training `seed=456`. Training params passed later via `lgb.train()` reach `ds.params` only *after* the bins are fixed. Different bin edges &rarr; different splits &rarr; ~&minus;0.0024 NDCG@5.

**Verification.** `pipelines/phase2_anchor_check.py` reran the `lg_0_1_15` config with a fresh `Dataset` inside the loop, no `.construct()`, no `free_raw_data`. Result: NDCG@5 = **0.42191**, `best_iter=632` &mdash; bit-for-bit anchor reproduction. Hypothesis confirmed.

**Fix.** Every pipeline now builds a fresh `lgb.Dataset` inside each config iteration, never pre-constructs, and runs `del ds_train, ds_val, model; gc.collect()` per iteration. The full determinism rules are in [validation.md](validation.md).

The tainted rows are preserved as `P2_INVALID_*` (`kept=NO`) in `experiment_logs/experiment_tracker.csv` rather than deleted &mdash; a record of the failure mode.

---

## Weighting / IPW sweep (Phase 3)

Scaffolded in `pipelines/phase3_weighting.py` &mdash; **14 configs** (7 weighting variants × 2 label gains), not yet run.

| Weighting variant | Semantics |
|---|---|
| `ipw_default` | V4 default &mdash; IPW on non-random rows, clipped `[0.1, 10.0]` |
| `no_ipw` | All weights = 1 (baseline) |
| `ipw_positive` | IPW only on `click_bool=1` rows; unclicked = 1.0 |
| `ipw_clip3` | IPW clipped `[0.1, 3.0]` |
| `ipw_clip5` | IPW clipped `[0.1, 5.0]` |
| `rand_up_1.5` | IPW default ×1.5 on `random_bool=1` rows |
| `rand_up_2.0` | IPW default ×2.0 on `random_bool=1` rows |

Label gains swept: `0,2,15` (Phase 2 winner) and `0,1,15` (the anchor).

**Decision gate.** Pin a variant as the new weighting default only if it beats the anchor *and* improves `mean_booked_rank`. If `no_ipw` competes within &Delta;<0.0005, prefer it &mdash; fewer assumptions. The winning weighting rolls into the final ensemble retrain, not a standalone submission.

---

## Tuning philosophy

- **Sweep one axis at a time**, always against the anchor &mdash; a moving anchor means the result is non-reproducible, not informative.
- **Directional, not decisive.** Single-model deltas below the ±0.0037 noise floor ([validation.md](validation.md)) are noise. No submission is ever selected on a single-model sweep number.
- **Optimize the gap, not the peak.** A config that closes the local&rarr;Kaggle gap beats one with higher raw val NDCG@5.
- **Winners roll into the ensemble** &mdash; sweep outputs are ingredients for the final V4-style ensemble rebuild, never end products themselves.
