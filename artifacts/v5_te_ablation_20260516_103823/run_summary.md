# V5 TE-ablation — v5_te_ablation_20260516_103823

_Generated 2026-05-16 11:09:45_

## Hypothesis
V5 lost on Kaggle (0.41943 vs V4 0.42021, Δ=−0.00078) because high-drift single-key
target-encoding features (`country_book_rate`, `site_book_rate`, `site_country_book_rate`,
`cpair_book_rate`) shifted distributions between train and test (adversarial AUC=1.0).

## Setup

| | Value |
|---|---|
| Members | B3, F13, D4, A1, B10, E4, E3 |
| Method | equal-weight rank averaging within srch_id |
| Dropped features | `country_book_rate, site_book_rate, site_country_book_rate, cpair_book_rate` |
| Feature count (before) | 143 |
| Feature count (after, full set) | 139 |
| D4 feature count (top_120 ∩ post-drop) | 116 |
| seeds / label_gains / weighting / params / best_iter | identical to v5_ensemble_submit |

## Baselines

| Reference | Kaggle public | Local val |
|---|---|---|
| V4 ensemble | 0.42021 | 0.42512 |
| V5 ensemble | 0.41943 | 0.42633 |
| **V5 TE-ablation (this run)** | _pending Kaggle upload_ | _not computed (no val split)_ |

## Outputs

- Submission CSV: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/submissions/submission_v5_te_ablation_20260516_103823.csv`
- Submission copy: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_te_ablation_20260516_103823/submission.csv`
- Submission rows: 4,959,183 · unique srch_ids: 199,549
- Models: `models/v5_te_ablation_20260516_103823/model_<id>.txt`
- Per-member test predictions: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_te_ablation_20260516_103823/test_pred_<id>.npy`
- Final rank average: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_te_ablation_20260516_103823/ensemble_test_rank_avg.npy`
- Ablation config: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/artifacts/v5_te_ablation_20260516_103823/ablation_config.json`

## Interpretation rubric

| New Kaggle | Δ vs V5 (0.41943) | Δ vs V4 (0.42021) | Conclusion |
|---|---|---|---|
| ≥ 0.4210 | ≥ +0.0016 | ≥ +0.0008 | **TE drift confirmed**. Beats V4. Promote ablation to v6 base; sweep TE smoothing next. |
| 0.4200–0.4210 | +0.0006 to +0.0016 | −0.0002 to +0.0008 | TE drift partially confirmed. Closes most of the gap. Worth more TE work. |
| 0.4194–0.4200 | 0 to +0.0006 | small loss | Marginal — TE drift is one of multiple factors. Run `--temporal` next. |
| < 0.4194 | < 0 | unchanged | TE drift NOT the dominant factor. Don't blame TEs; revisit `--temporal` + position-bias actions. |

## Wall-clock

Total: 31.4 min
