# V5 Ensemble Submission — v5_ensemble_submit

_Generated 2026-05-16 09:47:45_

## Members (7) — equal-weight rank averaging within srch_id

| id  | val NDCG@5 | best_iter | label_gain | weighting   | feature_filter | row_filter      | params override |
|-----|------------|-----------|------------|-------------|----------------|-----------------|------------------|
| B3  | 0.42396    |       326 | 0,2,15     | ipw_clip3   | -              | -               | - |
| F13 | 0.42253    |       500 | 0,3,15     | ipw_default | -              | positive_q_only | - |
| D4  | 0.42228    |       523 | 0,2,15     | ipw_default | top_120        | -               | - |
| A1  | 0.42205    |       385 | 0,2,12     | ipw_default | -              | -               | - |
| B10 | 0.42198    |       358 | 0,2,20     | no_ipw      | -              | -               | - |
| E4  | 0.42291    |       518 | 0,2,15     | ipw_default | -              | -               | {'num_leaves': 400, 'min_child_samples': 100} |
| E3  | 0.42185    |       551 | 0,2,15     | ipw_default | -              | -               | {'num_leaves': 512, 'min_child_samples': 100, 'reg_lambda': 2.0} |

## Baselines

| Reference | NDCG@5 |
|-----------|--------|
| Anchor single (V4 bal15) | 0.42191 |
| V4 ensemble local         | 0.42512 |
| V4 ensemble Kaggle public | 0.42021 |
| Ensemble val NDCG@5 (E2_top5 + E4 + E3, from val-only search) | ~0.42622+ (val) |

## Outputs

- Submission CSV: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/submissions/submission_v5_ensemble_20260516_094741.csv`
- Submission copy: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_ensemble_submit/submission.csv`
- Submission rows: 4,959,183
- Unique srch_ids: 199,549
- Models: `models/v5_ensemble_submit/model_<id>.txt` (one per member)
- Per-member test predictions: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_ensemble_submit/test_pred_<id>.npy`
- Final rank average: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_ensemble_submit/ensemble_test_rank_avg.npy`
- Run config: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_ensemble_submit/ensemble_config.json`

## Wall-clock

- Total: 31.8 min

## Method

For each member m: train LightGBM on FULL train with its exact overnight config
(label_gain, weighting, feature_filter, row_filter, param_overrides, seed=456),
for `best_iter` rounds (no early stopping, no validation). Predict each test row
to get continuous score `p_m`. Within each `srch_id`, rank rows by `p_m` descending
to get integer ranks `r_m` (1=best). The ensemble score for each row is
`mean(r_1, …, r_7)`. Sort rows ascending by ensemble score within srch_id;
emit `(srch_id, prop_id)`.
