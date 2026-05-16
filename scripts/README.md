# scripts/

One-off utilities and diagnostics. Unlike `pipelines/`, scripts here are
not part of a sequential workflow — each is a standalone tool.

Run from the project root:
```bash
uv run python scripts/<name>.py [args]
```

| file | what it does |
|---|---|
| `aggregate_results.py` | Promote per-run artifacts from `artifacts/<run>/` into the master trackers in `experiment_logs/`. Run after any pipeline that writes a new artifact dir. |
| `diagnose_v5_gap.py` | V4↔V5 Kaggle-gap forensics. Adversarial validation (train vs test classifier), per-feature drift, slice-level NDCG decomposition. Produced the finding that V5 raw TEs had adversarial AUC = 1.0. |
| `eda_summary.py` | 5 EDA tables for the report: click/book rate per position, per month/week, V5 TE feature importance, prop_id frequency p25/p50/p75. Writes `diagnostics/eda_summary/`. |
| `eda_dest_click_rate.py` | Investigates `dest_click_rate` as a query-context feature. Buckets searches into low/mid/high, profiles booked-hotel features per bucket, scores model NDCG@5 per bucket. Motivated the V6 interaction-feature experiments. |
| `ensemble_rank_avg.py` | Quick rank-average sandbox over saved boosters. Used pre-V6 to compute the 3-model V4+CP+DS = 0.40679 benchmark. |
| `ensemble_search.py` | Combinator search over ensemble member subsets. Older tool, predates the V6 LOO approach. |
| `temporal_rescore_overnight.py` | Rescore the 85 overnight V4-style boosters on temporal val. Confirms the overnight predictions cannot be reused for temporal ensembles (random↔temporal data overlap → leakage). |
| `te_safe_single.py` | Single-key TE drift safety experiment. Used during the V5→V5.2 ablation. |

## Notes

- `aggregate_results.py` reads from `artifacts/` and writes to
  `experiment_logs/`. It's idempotent; rerun is safe.
- Diagnostic scripts (`diagnose_v5_gap.py`, `temporal_rescore_overnight.py`)
  write to `diagnostics/<task>/`.
