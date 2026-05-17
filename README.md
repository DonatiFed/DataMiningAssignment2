# Expedia Hotel Search Ranking

LightGBM-based learning-to-rank on the Expedia Personalized Hotel Search
Kaggle dataset. The model sorts hotels within each search query so that
the booked hotel appears as high as possible. Metric: **NDCG@5**.

## Results

| version | Kaggle public NDCG@5 |
|---|---:|
| **V4 ensemble** ★ | **0.42021** |
| V9 overnight diversity | 0.42012 |
| V6 LOO-9 temporal | 0.42004 |
| V11 mega-bag (23-model equal rank-avg) | 0.42003 |
| V11 safe-push | 0.41995 |
| V5 cross-key TEs | 0.41943 |
| V10 adversarial reweight | 0.41903 |
| V4.2 Phase 2 | 0.41639 |

★ selected for the Kaggle private leaderboard. Full scoreboard with
reproducibility commands → [`docs/final_kaggle_results.md`](docs/final_kaggle_results.md).

## Quick start

```bash
./setup_vm.sh                                       # one-time setup (uv + venv)
uv run python pipelines/legacy/v4_ensemble.py       # reproduce V4 (best)
uv run python pipelines/v6.py                       # reproduce V6 temporal ensemble
uv run python pipelines/v6_submit.py                # build the V6 submission CSV
```

All commands run from the project root.

## Repository

```
src/              library — features, IPW, k-fold TE, NDCG helpers
pipelines/        end-to-end runnable workflows (V5 → V10)
pipelines/legacy/ V3, V4, Phase 2/3 archive
scripts/          diagnostic & recovery scripts
notebooks/        EDA notebooks (executed)
docs/             documentation (see below)
diagnostics/      per-experiment outputs (CSV / JSON / README per run)
artifacts/        per-run JSON / CSV metadata
submissions/      Kaggle CSVs (gitignored) + per-submission READMEs
```

## Documentation

| file | what it covers |
|---|---|
| [`docs/journey.md`](docs/journey.md) | chronological narrative V3 → V11 with what was learned at each step |
| [`docs/final_kaggle_results.md`](docs/final_kaggle_results.md) | authoritative Kaggle scoreboard + reproducibility commands |
| [`docs/lessons_learned.md`](docs/lessons_learned.md) | methodological and technical takeaways with supporting evidence |
| [`docs/architecture.md`](docs/architecture.md) | code structure and data flow |
| [`docs/results.md`](docs/results.md) | experiment-level NDCG@5 results (variants, ensembles) |
| [`docs/next_steps.md`](docs/next_steps.md) | unattempted ideas and recommended forward queue |
| [`docs/v4_phase2_summary.md`](docs/v4_phase2_summary.md) | detailed V4 + Phase 2 narrative |
| [`CHANGELOG.md`](CHANGELOG.md) | version-by-version changelog |
| [`pipelines/README.md`](pipelines/README.md) | index of runnable pipelines |
| [`scripts/README.md`](scripts/README.md) | index of standalone scripts |
| [`submissions/README.md`](submissions/README.md) | submission inventory |

## Key findings (short)

- Train/test distribution **drift is severe** (adversarial AUC = 1.0 on
  cross-key target-encoding features).
- **Local NDCG@5 is a treacherous metric.** Random splits overfit; the
  temporal split is closer to Kaggle but still leaves a +0.011 gap.
- **Sample reweighting cannot fix structural drift.** V10 attempted this
  and lost −0.00118 on Kaggle.
- **V4's LambdaRank ensemble is near the practical ceiling** for this
  dataset with LightGBM and the 143-feature pipeline. Six subsequent
  versions could not surpass it on Kaggle.
- **Different objectives provide more ensemble diversity than different
  `label_gain` values.** `rank_xendcg` and CatBoost `YetiRank` were the
  most useful diversifiers.

Full discussion: [`docs/lessons_learned.md`](docs/lessons_learned.md).

## Setup

```bash
./setup_vm.sh
```
Installs `uv` if missing, creates the virtual environment, runs a smoke
import test.
