# Insights

Methodological and technical takeaways from this project. Each lesson is
documented with the evidence that produced it.

## 1. Local NDCG@5 is a treacherous metric — temporal validation is mandatory

V5 had higher random-validation NDCG@5 than V4 (0.42633 vs 0.42512) but
lower Kaggle NDCG@5 (0.41943 vs 0.42021). This is the classic
random-split-overfitting trap: the random split places test rows
arbitrarily across the time axis, so each fold sees roughly the same
distribution as the training data. The Kaggle test set, however, is
strictly future-dated relative to training, so a model that overfits the
training distribution looks great on random val and bad on Kaggle.

The fix is **temporal validation**: hold out the last N days of training
data as the validation set, train on everything before. From V6 onward
we used cutoff 2013-05-21, with 39,959 searches after the cutoff as val.

Temporal validation is much stricter than random validation. V6 LOO-9
scored 0.40896 on temporal val but 0.42004 on Kaggle (a gap of +0.011).
This gap appears because temporal val is *closer to* the Kaggle test
distribution, but still not identical — see Lesson 4 below.

**Evidence:** `docs/v4_phase2_summary.md`,
`diagnostics/v5_gap_*/README.md`,
`diagnostics/temporal_validation_*/split_meta.json`.

## 2. Train/test drift is the dominant problem on this dataset

V5 added 12 target-encoding features, including 4 cross-key TEs
(`site_book_rate`, `country_book_rate`, `site_country_book_rate`,
`cpair_book_rate`). The V10 adversarial classifier later identified
these EXACT features as the four most discriminative train-vs-test
features, with holdout AUC = 1.0 (perfect separability).

This means the model can MEMORIZE properties of training target-encoded
values that do not match the test distribution. Adding these features
made V5 worse on Kaggle even though they were standard k-fold
out-of-fold features (no in-fold leakage).

The drift is **structural** — it lives in the per-key TE values
themselves, not in feature scales or simple distribution shifts.

**Evidence:** `diagnostics/v5_gap_20260516_101321/`,
`diagnostics/adv_reweight_batch_*/adv_feature_importance_eval.csv`.

## 3. Sample reweighting cannot fix structural drift

V10 attempted to correct the drift by training an adversarial classifier
(label train=0 vs test=1) and weighting train rows by the importance
ratio `P(test|x) / (1 − P(test|x))`. This is a standard
domain-adaptation technique.

**Result:** Kaggle NDCG@5 = 0.41903, a regression of −0.00118 vs V4
(0.42021). The reweighting de-emphasized the rows whose target-encoded
features matched the test distribution LEAST, but the model still
NEEDED those features to rank well. Penalizing their importance hurt
overall predictive quality.

**Implication:** the right tool for this kind of drift is **feature
engineering that produces non-drifting alternatives**, not sample
reweighting. CP (`prop_click_rate_pos_adj_s40_oof`) and DS
(`prop_dest_book_rate_safe`) were attempts at this. Both had clean
drift (`|Δμ|/σ < 0.02`), but their predictive gain was too small to
overcome V4 on Kaggle alone.

**Evidence:** `diagnostics/adv_reweight_batch_*/README.md`.

## 4. The local→Kaggle gap shrinks with diversity but never closes

| version | local NDCG@5 (temporal) | Kaggle NDCG@5 | gap |
|---|---:|---:|---:|
| V6 LOO-9 | 0.40896 | 0.42004 | +0.01108 |
| V9 overnight | 0.40971 | 0.42012 | +0.01041 |
| V11 MEGA-BAG | ≈0.4090 | 0.42003 | +0.0110 |

Adding diversifiers and reducing V6 weight slightly compresses the gap,
but the local improvements translate to much smaller Kaggle improvements.
The +0.00075 local improvement (V9 vs V6 base) became only +0.00008 on
Kaggle. **The drift absorbed roughly 89% of any local improvement.**

This is the operational reason 0.426 was unreachable: even if we found
a +0.005 local gain (large by our standards), it would translate to
about +0.0005 Kaggle = 0.4207, far from 0.426.

## 5. In-model feature stacking has a hard ceiling at this dataset size

Three weighted-ensemble combination experiments produced anti-additive
results when adding multiple positive single-feature models inside one
V4_ANCHOR LightGBM configuration:

- `prop_click_posadj + te_rank`: each individually positive, combined
  −0.00160 vs V4_ANCHOR
- `prop_click_posadj + price_dest`: combined −0.00087
- `prop_click_posadj + prop_dest_safe`: combined +0.00065 (still
  below either alone)

With LightGBM's default `colsample_bytree=1.0`, the tree builder picks
the locally-best feature at each split. Adding correlated alternatives
changes split selection in ways that hurt validation. The bottleneck is
GBDT capacity at this dataset size with this feature count.

**Resolution:** rank-average ensembling outside the model recovers most
of the lost additivity. An ensemble of two single-feature models often
outperforms the in-model stack of the same features.

**Evidence:** `diagnostics/eval_variants/results.csv` (rows containing
"posadj_plus_*").

## 6. Ensembling has a sweet-spot V6 weight around 0.80

Across multiple weighted-ensemble grids, the V6 weight that maximized
Kaggle score was consistently 0.80–0.90, never below 0.50:

| configuration | V6 weight | Kaggle |
|---|---:|---:|
| V6 LOO-9 alone | 1.00 | 0.42004 |
| V6 + 4 diversifiers @ 0.05 each (V9) | 0.80 | 0.42012 |
| V6 + 6 diversifiers @ 0.042 each (V11 SAFE-PUSH) | 0.75 | 0.41995 |
| 23 models equal weight (V11 MEGA-BAG) | 0.39 | 0.42003 |
| V6 adv + 4 diversifiers @ 0.125 each (V10) | 0.50 | 0.41903 |

Lowering V6 weight below 0.75 consistently hurts on Kaggle even when
local NDCG@5 stays similar. The diversifiers are individually too weak
to carry their portion of the ensemble.

## 7. Different objectives provide more diversity than different label_gains

`rank_xendcg_regularized` (LightGBM with the `rank_xendcg` objective)
and the CatBoost `YetiRank` models gave the largest ensemble lifts when
added at small weights. Adding more LambdaRank variants with different
`label_gain` values (V6 already had four) gave diminishing returns.

**Evidence:**
`diagnostics/structural_batch_20260516_212040/ensemble_results.csv` (top
entries dominated by `rank_xendcg_regularized` and the two CatBoost
rankers).

## 8. Fault-tolerant pipelines pay off

The codebase contains several multi-hour training pipelines. Every
long-running pipeline is designed to be:

- **Resumable**: skip any model whose artifact (model file + val_pred
  `.npy`) already exists
- **Per-model exception-isolated**: one model's failure does not kill
  the batch
- **Atomic writes**: every CSV/JSON written via `tmp` + rename
- **Errors directory**: per-failure `ERROR_<id>.txt`, plus `FATAL.txt`
  on outer crash
- **Side-effect-free imports**: directory creation moved inside `main()`
  so importing a pipeline module does not create empty stubs

These patterns proved valuable on multiple occasions. The V6 pipeline
crashed during its in-pipeline submission step (a `best_iter = −1`
reload bug); `scripts/overnight_submit_best.py` recovered the
submission without losing the V6 training results. The V9 pipeline
crashed in its README writer; the README was regenerated from saved
CSV files without re-running 2.5 hours of training.

## 9. The Kaggle submission header is `srch_id,prop_id` (lowercase)

For the Expedia Personalized Hotel Search competition, Kaggle expects
the lowercase column names matching `data/submission_sample.csv`. The
V6 pipeline initially wrote `SearchId,PropertyId` (capitalized,
following common Pandas defaults) and required a post-hoc header fix.

Both V6 and V9 submission writers now explicitly emit lowercase
headers. Any future submission writer should validate against the
sample file.

## 10. CatBoost YetiRank is genuinely diverse from LightGBM LambdaRank

In ensemble experiments, the most consistent positive diversifiers
were CatBoost rankers (`cb_rank_C_deeper`, `cb_rank_A`). They ranked
below V6 individually (around 0.404 local) but added small positive
Kaggle signal as ensemble members at weights 0.05–0.075 each. This
held across multiple ensemble configurations.

**Implication:** future ensemble work on similar problems should
include CatBoost rank as a default diversifier alongside LightGBM.

## What was not attempted

If the timeline had been longer, the next moves in order of expected
leverage would be:

1. **Neural network listwise model** (PyTorch ListNet or approxNDCG
   loss) as a third model class. Estimated implementation cost:
   4–6 hours. Could plausibly add +0.002 to +0.005 Kaggle as an
   ensemble member.

2. **Two-stage propensity model**: train a base model on
   `random_bool=1` data only (truly unbiased), then fine-tune on the
   full training set using the base model as a regularizer.
   Theoretically grounded; could close part of the train→test drift
   gap.

3. **Hard-negative mining as features**: train V6, identify property
   IDs that V6 consistently mis-ranks, encode this signal as new
   features. A failure-driven feature engineering approach.

4. **Seed bagging**: 10–20 seeds per V6 member with strong
   stochasticity (`feature_fraction`, `bagging_fraction`). Boring but
   historically effective for variance reduction in tabular GBDT
   pipelines.

See `docs/next_steps.md` for a more detailed forward queue.
