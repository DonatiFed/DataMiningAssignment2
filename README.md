# Expedia Hotel Search Ranking — DM Assignment 2

Learning-to-rank on the Expedia Personalized Hotel Search Kaggle dataset.
**Metric: NDCG@5.** Listwise ranking inside each `srch_id`.

## Headline results

| version | local NDCG@5 | Kaggle public | notes |
|---|---:|---:|---|
| V3 baseline | 0.401 | — | initial GBDT, naive features |
| V4 ensemble | 0.42512 | **0.42021** | label_gain sweep + IPW + 143 features |
| V5 ensemble | 0.42633 | 0.41943 | added high-drift cross-key TEs → lost on Kaggle |
| V5.2 ablation | — | — | dropped drift-heavy TEs (submitted, see artifacts) |
| **V6 ensemble** (latest) | **0.40896** | **0.42004** | 9-member rank-average, clean drift |

V6 local validation runs on **temporal split** (anchor 0.40401),
not the random split V4/V5 used — so V6 local numbers are not directly
comparable to V4/V5 local. See `docs/results.md` for the full timeline.

V6 effectively matched V4 on Kaggle (0.42004 vs 0.42021). The +0.00217 local
gain over the quick 3-model rank-average did not translate. Top of the
leaderboard remains ~0.46 — see `CHANGELOG.md` for the gap analysis.

## Quick start

```bash
./setup_vm.sh                 # one-time: install uv + venv + smoke import

# Train V6 ensemble on temporal split (10 members, ~30 min)
uv run python pipelines/v6.py

# Build a Kaggle submission from V6 results (9 LOO-best members, ~30 min)
uv run python pipelines/v6_submit.py

# Single-feature variant evaluation (used during V6 feature engineering)
uv run python pipelines/evaluate_variant.py --variant prop_click_rate_pos_adj_s40_oof
```

Run from the project root — `pipelines.*` imports need `cwd` on `sys.path`.

## Repo layout

```
.
├── README.md                # this file
├── CHANGELOG.md             # version timeline + key decisions
├── pyproject.toml / requirements.txt / uv.lock / setup_vm.sh
├── .gitignore
│
├── data/                    # raw competition CSVs (gitignored)
│
├── src/                     # importable library
│   ├── config.py            # data paths, ID/label column lists
│   ├── data_loader.py       # load_train / load_test / split helpers
│   ├── features.py          # build_features (143-feature pipeline), IPW, k-fold TE
│   ├── evaluate.py          # NDCG@5 helpers
│   ├── submission.py        # CSV writer
│   └── artifacts.py         # per-run artifact saving helpers
│
├── pipelines/               # end-to-end runnable workflows
│   ├── temporal_validation.py   # temporal split + V4 anchor reproduction
│   ├── evaluate_variant.py      # single-feature variant harness
│   ├── overnight_experiments.py # bulk LightGBM grid (random split, V4)
│   ├── v5.py                    # V5 ensemble submission (was v5_ensemble_submit)
│   ├── v5_2.py                  # V5.2 TE-ablation (was v5_te_ablation)
│   ├── v6.py                    # ★ V6 ensemble training + LOO + conditional submit
│   ├── v6_submit.py             # V6 submission builder (resumable retrain)
│   └── legacy/                  # earlier pipelines (V3 baseline, V4, Phase 2, Phase 3)
│
├── scripts/                 # diagnostics & one-off utilities
│   ├── aggregate_results.py     # promote per-run artifacts → experiment_logs/
│   ├── diagnose_v5_gap.py       # V4↔V5 Kaggle-gap forensics
│   ├── eda_summary.py           # 5 EDA tables for the report
│   ├── eda_dest_click_rate.py   # dest_click_rate as query-context feature
│   ├── ensemble_rank_avg.py     # quick rank-average sandbox
│   ├── ensemble_search.py       # ensemble-member combinator search
│   ├── temporal_rescore_overnight.py # rescore overnight boosters on temporal val
│   └── te_safe_single.py        # single-key TE drift safety experiment
│
├── notebooks/               # EDA (executed)
│   ├── 01_data_overview.ipynb
│   ├── 02_target_and_position.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_diagnostics.ipynb
│
├── docs/                    # documentation
│   ├── architecture.md          # code structure + data flow
│   ├── results.md               # NDCG@5 progression across versions
│   ├── v4_phase2_summary.md     # V4 + Phase 2 narrative
│   ├── next_steps.md            # forward queue
│   └── archive/                 # superseded planning docs
│
├── experiment_logs/         # aggregated CSV trackers
├── artifacts/               # per-run outputs (JSON/CSV tracked, binaries gitignored)
├── diagnostics/             # local analysis + cached features
│   └── eval_variants/
│       └── base_features_temporal_*.parquet  ← critical V6 feature cache
├── models/                  # saved LightGBM boosters (gitignored, large)
├── submissions/             # Kaggle submission CSVs (gitignored)
└── logs/                    # session logs (gitignored)
```

## Architecture in one paragraph

`src/` is the importable library: feature engineering (143 base features +
optional V6 add-ons), data loaders, evaluation, submission writers.
`pipelines/` holds runnable workflows — each is an entry point (`python
pipelines/<name>.py`). `scripts/` holds one-off diagnostics. Validation
uses a **temporal split** (cutoff 2013-05-21) as of V6; V4/V5 used random
splits and have a documented local→Kaggle drift gap. See
`docs/architecture.md` for the full data-flow diagram.

## V6 reproducibility

The cached V6 features live in `diagnostics/eval_variants/base_features_temporal_*.parquet`.
They are NOT in git (too large) but `pipelines/v6.py` will rebuild them
on first run from `data/training_set_VU_DM.csv`.

Two V6-specific feature builders live in `pipelines/evaluate_variant.py`:

- `_pos_adj_oof_te` — k-fold OOF position-adjusted click TE (**CP** member)
- `_prop_dest_book_rate_safe` — 3-way fallback smoothed (prop, dest) booking rate (**DS** member)

Both are clean-drift, leak-safe, and verified on temporal val.
See `docs/architecture.md` §Features for the full mechanic.

## Key invariants

- **Temporal split** is the validation contract going forward (anchor: V4_ANCHOR
  temporal NDCG@5 = 0.40401). Random-split scores are not comparable.
- **No `lgb.Dataset.construct()`** — let `lgb.train` construct lazily so the seed
  propagates correctly to bin sampling. (V4 anchor bug, documented in
  `docs/v4_phase2_summary.md`.)
- **Kaggle submission header MUST be `srch_id,prop_id`** lowercase — matches
  `data/submission_sample.csv`. V6 had a post-hoc fix; both `v6.py` and
  `v6_submit.py` now emit the correct header.
- **CP and DS are separate ensemble members.** In-model stacking is consistently
  anti-additive at this feature count (see `docs/results.md` §LOO).

## Status

V6 submission completed and uploaded. Kaggle score 0.42004 — within noise of V4
(0.42021), drift gap closed only at local level. Next investigation: hard-negative
mining and failure-driven features. See `docs/next_steps.md`.

## Setup

```bash
./setup_vm.sh
```
Installs `uv` if missing, creates the venv, runs a smoke import.
