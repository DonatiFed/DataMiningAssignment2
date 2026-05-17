# pipelines/

End-to-end runnable workflows. One file = one entry point. Run from the
project root:

```bash
uv run python pipelines/<name>.py [args]
```

## Active pipelines

| file | version | purpose |
|---|---|---|
| `temporal_validation.py` | infra | Build temporal split (cutoff 2013-05-21), train V4_ANCHOR + B3 on temporal val and on a random-control split. Established the V6+ validation contract. |
| `evaluate_variant.py` | infra | Single-feature variant harness on temporal val. Cached parquet features, ~4 min per variant. Decision rule REJECT / HOLD / KEEP / STRONG_KEEP based on Δ vs V4_ANCHOR temporal (0.40401). Used during V6 feature engineering. |
| `overnight_experiments.py` | V4 | Bulk LightGBM grid over `label_gain`, `num_leaves`, `learning_rate`, weighting. Random split (val_frac=0.1, seed=456). Slow (~10 hr full). Produced the 85 V4-era boosters. |
| `v5.py` | V5 | V5 production submission: 7-member rank-average with V4 + V5 cross-key TE feature set. Submission: `submission_v5_ensemble_*.csv` → Kaggle 0.41943. |
| `v5_2.py` | V5.2 | V5 minus the 4 high-drift cross-key TEs. |
| `v6.py` | V6 | ★ V6 main pipeline. Trains 10 diverse temporal-clean members on temporal_train, evaluates each on temporal_val, builds 6 rank-average ensembles + LOO on the best. Conditionally builds a submission. Resumable, per-member try/except, atomic writes. |
| `v6_submit.py` | V6 | V6 submission builder. Reads V6 outputs, retrains LOO-best 9 members on full train (using `current_iteration()` as `best_iter`), predicts on test, rank-averages, writes Kaggle CSV. Resumable. |
| `phase7_batch.py` | V7 | Five failure-pattern-driven feature variants on temporal val. 4 HOLD, 1 REJECT. |
| `phase7_weighted_batch.py` | V7 | Weighted-ensemble grid testing of the V7 feature models with V6 LOO-9 as the base. Confirmed no weight scheme helped on temporal val. |
| `structural_batch.py` | V8 | Structural diversity batch: 13 models with different loss functions, regularization, hyperparameters. First positive single-model ensemble lift (`rank_xendcg_regularized` at weight 0.10 → +0.00037 vs V6 LOO-9). |
| `overnight_final_batch.py` | V9 | Large 24-model diversity batch (LightGBM + XGBoost + CatBoost + binary classifiers). 5 XGBoost models failed on `inf in input`. Best ensemble Kaggle 0.42012. |
| `adversarial_reweight_batch.py` | V10 | Adversarial sample reweighting. Trains a train-vs-test binary classifier, weights train rows by importance ratio, retrains V6 members. Adversarial AUC = 1.0 confirmed extreme drift. Submission Kaggle 0.41903 (regression). |

## Legacy pipelines

`legacy/` holds earlier pipelines kept for archaeology and for the V4
anchor invariant. None are part of the V6+ workflow but they document the
journey from V3 baseline through V4 production.

| file | era |
|---|---|
| `v3_baseline.py` | initial single-model LambdaRank baseline (~0.40 local) |
| `v4_ensemble.py` | V4 production ensemble (Kaggle 0.42021, the winning submission) |
| `phase2_anchor_check.py` | anchor invariant verification (random val NDCG@5 = 0.42191) |
| `phase2_labelgain.py` | `label_gain` sweep (6 configurations) |
| `phase2_submit.py` | generated `submission_phase2_best_*.csv` |
| `phase3_weighting.py` | weighting sweep scaffold (14 configurations, never run) |

## Import dependency graph

```
temporal_validation.py
  ↑ ↑ ↑ ↑ ↑
  │ │ │ │ └─ evaluate_variant.py
  │ │ │ │       ↑
  │ │ │ └─── v6.py / v6_submit.py
  │ │ └───── phase7_batch.py / phase7_weighted_batch.py
  │ └─────── structural_batch.py
  └───────── overnight_final_batch.py
             adversarial_reweight_batch.py
```

`legacy/*` and `v5*.py` are standalone (import only from `src/`).
