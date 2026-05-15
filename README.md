# Expedia Hotel Search Ranking — DM Assignment 2

Learning-to-rank on the Expedia Personalized Hotel Search Kaggle dataset. Metric: **NDCG@5**.

## Status

| Model | Val NDCG@5 | Kaggle public |
|---|---|---|
| **V4 ensemble** (production reference) | 0.42512 | **0.42021** |
| V4 best single (lambdarank_bal15, anchor) | 0.42191 | — |
| v4.2 single (lg_0_2_15) | 0.42258 | 0.41639 (worse — single overfits) |

See `docs/v4_phase2_summary.md` for the full narrative and `docs/next_steps.md` for the ordered queue.

## Repo layout

```
.
├── README.md                # this file
├── pyproject.toml + uv.lock + requirements.txt + setup_vm.sh
│
├── src/                     # importable library
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py          # build_features (143 features) + IPW + KFold TE
│   ├── evaluate.py
│   ├── submission.py
│   └── artifacts.py         # per-run artifact saving helpers
│
├── pipelines/               # executable training scripts (run via `python -m`)
│   ├── v3_baseline.py
│   ├── v4_ensemble.py       # ★ current Kaggle reference
│   ├── phase2_labelgain.py  # label_gain sweep (6 configs)
│   ├── phase2_anchor_check.py # single-config V4 reproduction (diagnostic)
│   ├── phase2_submit.py     # generated the v4.2 submission
│   └── phase3_weighting.py  # weighting sweep (14 configs, NOT YET RUN)
│
├── scripts/
│   └── aggregate_results.py # promote per-run artifacts → experiment_logs/
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_target_and_position.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_diagnostics.ipynb
│
├── docs/
│   ├── v4_phase2_summary.md  # narrative: V4, Phase 2, the binning bug, v4.2 result
│   ├── next_steps.md         # ordered Phase 3 → 7 queue
│   └── archive/
│       ├── SNAPSHOT.md       # superseded by v4_phase2_summary.md
│       └── EDA_PLAN.md       # historical EDA roadmap
│
├── experiment_logs/          # tracker CSVs (data, not narrative)
│   ├── experiment_tracker.csv  # one row per exp
│   ├── model_results.csv       # one row per single model
│   ├── ensemble_results.csv    # one row per ensemble
│   └── feature_audit.csv       # 143 features risk-labeled
│
├── artifacts/                # per-run outputs (json/csv tracked; .npy/.parquet/submission*.csv gitignored)
│   ├── phase2_labelgain/
│   ├── phase2_anchor_check/
│   └── v4.2_submit/
│
└── (gitignored) models/ submissions/ logs/ data/ .venv/
```

## How to run

All pipelines are executed as Python modules from the project root, so `src/` resolves on `sys.path`:

```bash
# V4 ensemble (current Kaggle reference)
uv run --no-sync python -m pipelines.v4_ensemble

# Phase 2 label-gain sweep
uv run --no-sync python -m pipelines.phase2_labelgain

# Phase 3 weighting sweep (scaffolded, not yet run)
uv run --no-sync python -m pipelines.phase3_weighting

# Aggregate a finished run into the master trackers
uv run --no-sync python scripts/aggregate_results.py \
    --run-id phase3_weighting --phase P3 \
    --change-summary "Weighting sweep (7 variants × 2 label_gains)"
```

Run from the project root every time — `python -m` requires cwd on `sys.path` for `from src.X import …` to resolve.

## Anchor invariant

Any new pipeline that includes `label_gain="0,1,15"`, `seed=456`, IPW default, and the full V4 feature set **must** reproduce val NDCG@5 = 0.42191. If it doesn't, the LightGBM `Dataset` was pre-constructed before `lgb.train` could propagate the training seed to bin sampling — see `docs/v4_phase2_summary.md` §4. Verification script: `pipelines/phase2_anchor_check.py`.

## Setup

```bash
./setup_vm.sh
```
Installs `uv` if missing, creates the venv, and runs a smoke import.

## Key constraints (carried into future phases)

- **No `lgb.Dataset.construct()` outside the per-config loop.** Each iteration builds a fresh Dataset; let `lgb.train` construct lazily so `seed=456` reaches bin sampling.
- **Selection metric is the local→Kaggle gap, not peak val NDCG@5.** V4 ensemble's gap is −0.00491; v4.2 single was −0.00619. Phase 3+ should aim to close the gap.
- **No new submissions until Phase 7** (rebuild ensemble after weighting + features stabilize).
