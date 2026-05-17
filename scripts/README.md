# scripts/

One-off utilities, diagnostics, and recovery scripts. Unlike `pipelines/`,
each script is standalone — not part of a sequential workflow.

Run from the project root:

```bash
uv run python scripts/<name>.py [args]
```

## Diagnostic scripts

| file | what it does |
|---|---|
| `diagnose_v5_gap.py` | V4 ↔ V5 Kaggle-gap forensics. Adversarial validation (train-vs-test classifier), per-feature drift, slice-level NDCG decomposition. Produced the finding that V5 raw cross-key TEs had adversarial AUC = 1.0. |
| `eda_summary.py` | Five EDA tables for the report: click/book rate per position, per month/week, V5 TE feature importance, `prop_id` frequency p25/p50/p75. Writes `diagnostics/eda_summary/`. |
| `eda_dest_click_rate.py` | Investigates `dest_click_rate` as a query-context feature. Bucketing into low/mid/high, booked-hotel feature profiles per bucket, NDCG@5 per bucket. Motivated several V7 interaction feature attempts. |
| `failure_pattern_analysis.py` | V6 booked-vs-top-wrong analysis: for each search where V6 ranked the booked hotel below position 1, compares the booked hotel against V6's predicted top-1. Drove the V7 feature design. |
| `temporal_rescore_overnight.py` | Rescore the 85 V4-era boosters on temporal validation. Confirmed that those predictions cannot be reused for temporal ensembles (random↔temporal split overlap implies leakage). |
| `te_safe_single.py` | Single-key TE drift safety experiment. Used during the V5 → V5.2 ablation. |

## Ensemble & analysis sandboxes

| file | what it does |
|---|---|
| `ensemble_rank_avg.py` | Quick rank-average sandbox over saved boosters. Used pre-V6 to compute the 3-model V4 + CP + DS = 0.40679 benchmark. |
| `ensemble_search.py` | Combinator search over ensemble member subsets. Older tool, predates the V6 LOO approach. |
| `ensemble_normalization_search.py` | Paper-inspired ensemble normalization sweep (rank-average, global z-score, query-wise z-score, two 50/50 blends). 345 ensemble tests across three weight grids. Highest local NDCG@5 found = 0.41005 with the `blend_rank_global_z` method, but never validated on Kaggle. |
| `build_two_final_submissions.py` | V11 final two submissions (no retraining, pure rank-average of saved test predictions). Produces SAFE-PUSH (V6 @ 0.75 + 6 diversifiers) and MEGA-BAG (23 models equal weight). |

## Recovery scripts

| file | what it does |
|---|---|
| `overnight_submit_best.py` | V9 submission recovery. Reads outputs from `overnight_final_batch.py`, finds the best deployable ensemble, retrains the two CatBoost models on full train (a `best_iter = −1` reload bug had crashed the in-pipeline submission step), produces the V9 submission CSV. |
| `overnight_xgb_rescue.py` | V9 XGBoost rescue. The 5 XGB models in `overnight_final_batch.py` failed on `inf in input data`. This script cleans `±inf → NaN` so XGBoost treats them as missing (the same behavior LightGBM and CatBoost have by default), retrains the 5 XGB models, and updates ensemble results. |

## Operational utility

| file | what it does |
|---|---|
| `aggregate_results.py` | Promote per-run artifacts from `artifacts/<run>/` into the master trackers in `experiment_logs/`. Run after any pipeline that writes a new artifact directory. Idempotent. |

## Output conventions

- Diagnostic scripts write to `diagnostics/<task>/<TS>/` with a `README.md`
  inside.
- Ensemble sandboxes write to `diagnostics/<task>_<TS>/` with CSVs and a
  summary README.
- Recovery scripts write to the same output directory as the pipeline they
  recover (so artifacts stay co-located with the original run).
