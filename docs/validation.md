# Validation

The validation strategy answers one question: **does a local score predict the Kaggle score?** For a dataset full of leakage traps, this is not a formality &mdash; it is what makes every other measurement trustworthy.

## Overview

- **Search-level 90/10 holdout** &mdash; split by `srch_id`, ~19,979 validation searches (~495K rows).
- **Adversarial validation** confirms train and test are well-matched (AUC `0.524`, barely above chance).
- **Seed robustness** sets the noise floor: single-model NDCG@5 varies `±0.0037` across split seeds &mdash; differences smaller than that are noise.
- **Determinism rules** &mdash; a LightGBM `Dataset` binning bug worth &minus;0.0024 NDCG@5 means every pipeline must follow a fixed construction pattern and reproduce a known **anchor**.
- **Selection metric** &mdash; the local&rarr;Kaggle gap matters more than peak validation NDCG@5.

---

## Split design

A random **90/10 holdout split by `srch_id`** (`split_val(val_frac=0.1, random_state=42)`): all rows of a given search stay together in train or val, never split across both. Result: 179,816 train / 19,979 val searches, no overlap.

**Why search-level.** The same reasoning as OOF target encoding ([feature-engineering.md](feature-engineering.md)) &mdash; a row-level split would place hotels from the same search on both sides, letting within-query statistics computed on training rows leak information about val rows in that query.

**Why not temporal.** The dataset spans 2012-11 to 2013-06 in *both* train and test, with random assignment. A temporal split would misrepresent the actual train/test relationship: test queries are drawn from the same period, not the future.

---

## Does local track Kaggle?

**Adversarial validation.** A binary classifier trained to distinguish train rows from test rows scores **AUC = 0.524** &mdash; barely above chance, confirming train and test are well-matched distributions.

**Observed local vs Kaggle:**

| Model | Local NDCG@5 | Kaggle NDCG@5 | Gap |
|---|---|---|---|
| V1 (leaky) | 0.468 | 0.38208 | **&minus;0.086** (leak exposed) |
| V3 (honest) | 0.412&ndash;0.417 | 0.41392 | ~0 |
| V4 (ensemble) | 0.42512 | 0.42021 | &minus;0.00491 |

V3's local and Kaggle scores agree within rounding error. V1's large gap was the *signal* that same-row target aggregates were leaking &mdash; local scored high because the model had near-direct access to the label during training. Once leakage was removed, local became a reliable predictor of Kaggle.

---

## Seed robustness

How stable is a single NDCG@5 number? The V3 model was retrained on five different 90/10 split seeds:

| Seed | NDCG@5 |
|---|---|
| 42 | 0.42171 |
| 123 | 0.41423 |
| 456 | 0.41152 |
| 789 | 0.41650 |
| 2024 | 0.41568 |
| **Mean ± std** | **0.4159 ± 0.0037** |

**The std of 0.0037 is the detection threshold.** Score differences smaller than ~0.004 between two configurations are noise, not signal. This is why ensemble members within ~0.003 of each other should never be ranked on point estimates alone, and why single-model sweep results are treated as *directional only*.

---

## Determinism rules

A reproducibility anchor caught a subtle non-determinism bug; the rules below are mandatory for every new pipeline.

**The LightGBM `Dataset` binning bug.** A `lgb.Dataset` constructed without explicit `params=` has `ds.params is None`. Calling `ds.construct()` explicitly then bins the histograms with LightGBM defaults &mdash; including `data_random_seed=1`, **not** the training `seed`. Training params passed later via `lgb.train()` arrive *after* the bins are already fixed. Different bin edges on 143 continuous features &rarr; different splits &rarr; **~&minus;0.0024 NDCG@5** drift. Full diagnosis in [hyperparameters.md](hyperparameters.md).

**The rules:**

- Build a **fresh `lgb.Dataset` inside each config loop** &mdash; never pre-construct one outside it.
- **Never call `Dataset.construct()` explicitly.** Let `lgb.train(params, ds)` construct lazily, with the training `seed` in scope.
- Do not set `free_raw_data`. After each iteration: `del ds_train, ds_val, model; gc.collect()`.

**The anchor invariant.** Any run that includes `label_gain="0,1,15"`, `seed=456`, IPW default, and the full 143-feature set **must** reproduce **val NDCG@5 = 0.42191**. If it does not, stop and diagnose before drawing any conclusion. Verification script: `pipelines/phase2_anchor_check.py`.

---

## Selection metric

Peak validation NDCG@5 is **not** the objective. After the Phase 2 single-model submission (v4.2) scored 0.42258 locally but only 0.41639 on Kaggle &mdash; a &minus;0.00619 gap, wider than the V4 ensemble's &minus;0.00491 &mdash; the selection rule was fixed:

- **The local&rarr;Kaggle gap is the real metric.** Anything that *closes the gap* (more aggressive TE smoothing, dropping position-derived features, IPW alternatives) is a real win, even if local NDCG@5 barely moves.
- Single-model validation deltas are **directional only**, never a basis for selecting a submission.
- **Final submissions are always ensembles** &mdash; a single model cannot hedge against test distribution shift. See [models.md](models.md).
