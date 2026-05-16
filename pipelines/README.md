# pipelines/

End-to-end runnable workflows. One file = one entry point.

Run from the project root:
```bash
uv run python pipelines/<name>.py [args]
```

## Active

| file | what it does |
|---|---|
| `temporal_validation.py` | Build the temporal split (cutoff 2013-05-21), train V4_ANCHOR + B3 on temporal + random-control. Establishes the V6 validation contract. Writes `diagnostics/temporal_validation_<TS>/`. |
| `evaluate_variant.py` | Single-feature variant harness on temporal val. Cached parquet features, ~4 min per variant. Decision rule REJECT/HOLD/KEEP based on Δ vs V4_ANCHOR_TEMPORAL (0.40401). Used during V6 feature engineering. |
| `overnight_experiments.py` | Bulk LightGBM grid over `label_gain`, `num_leaves`, `learning_rate`, weighting variants. Random split (val_frac=0.1, seed=456). Slow (~10 hr full). Produced the 85 V4 boosters. |
| `v5.py` | V5 production submission: 7-member rank-average ensemble using the full V4 + V5 TE feature set. Generated `submissions/submission_v5_ensemble_*.csv`. |
| `v5_2.py` | V5.2 ablation: V5 minus the 4 high-drift cross-key TEs (`country_book_rate`, `site_book_rate`, `site_country_book_rate`, `cpair_book_rate`). |
| `v6.py` | ★ V6 main pipeline. Trains 10 diverse temporal-clean members on temporal_train, evaluates each on temporal_val, builds 6 rank-average ensembles + above-median selection + LOO on the best. Conditionally builds a submission. Resumable; per-member try/except; incremental saves. |
| `v6_submit.py` | V6 submission builder. Reads the v6 outputs, retrains the LOO-best 9 members on full train (with `current_iteration()` as `best_iter`), predicts on test, rank-averages, writes Kaggle CSV. Resumable. |

## Legacy

`legacy/` holds earlier pipelines kept for archaeology and the V4 anchor
invariant. None are part of the V6 workflow.

| file | era |
|---|---|
| `v3_baseline.py` | initial LambdaRank baseline (~0.40 local) |
| `v4_ensemble.py` | V4 production ensemble |
| `phase2_anchor_check.py` | anchor invariant verification (val NDCG@5 = 0.42191 on random split) |
| `phase2_labelgain.py` | `label_gain` sweep (6 configs) |
| `phase2_submit.py` | generated `submissions/submission_phase2_best_*.csv` |
| `phase3_weighting.py` | weighting sweep scaffold (14 configs, never run) |

## Cross-references

`evaluate_variant.py`, `v6.py`, and `v6_submit.py` all import helpers
from `temporal_validation.py` (split, IPW, metrics) and from each other.
The full import graph:

```
temporal_validation.py
  ↑ ↑ ↑
  | | └── evaluate_variant.py ──┐
  | |                            ↓
  | └──────────────────── v6.py
  |                              ↑
  └──── v6_submit.py ────────────┘
```

`legacy/*` and `v5*.py` are standalone (import only from `src/`).
