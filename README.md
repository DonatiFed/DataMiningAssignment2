<h1 align="center">expedia-rank</h1>

<p align="center">
  <strong>Learning-to-rank pipeline for the Expedia Personalized Hotel Search Kaggle competition</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/lightgbm-4.6-success.svg" alt="LightGBM 4.6">
  <img src="https://img.shields.io/badge/xgboost-3.2-orange.svg" alt="XGBoost 3.2">
  <img src="https://img.shields.io/badge/catboost-1.2-yellow.svg" alt="CatBoost 1.2">
  <img src="https://img.shields.io/badge/metric-NDCG%405-blueviolet.svg" alt="NDCG@5">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#results">Results</a> &middot;
  <a href="docs/journey.md">Journey</a> &middot;
  <a href="docs/architecture.md">Architecture</a> &middot;
  <a href="docs/lessons_learned.md">Insights</a> &middot;
  <a href="docs/final_kaggle_results.md">Kaggle Scoreboard</a>
</p>

<p align="center">
  Submitted as <strong>team_80</strong> for the
  <a href="https://www.kaggle.com/competitions/dmt-2026-2nd-assignment">DMT 2026 — 2nd Assignment</a> Kaggle competition
</p>

---

End-to-end learning-to-rank system that predicts the ordering of hotels
within each search query so the booked hotel appears as high as possible.
Eleven model versions (V3 → V11) were explored across feature
engineering, ensembling, drift correction, and multi-framework
diversity, all evaluated on a strict temporal validation split.

```
raw CSVs  -->  feature engineering  -->  LightGBM rank  -->  ensemble  -->  submission
   (5M)        (143 features +          (LambdaRank +      (rank-avg     (Kaggle CSV)
                target encoding +         rank_xendcg +     within
                position bias)            CatBoost +        srch_id)
                                          XGBoost)
```

## Highlights

- **Eleven model versions** spanning baseline (V3) through V4 ensemble
  (the best Kaggle submission, 0.42021), V5 cross-key TEs, V6 temporal
  LOO-9, V7 failure-driven features, V8 structural diversity, V9 large
  multi-framework batch, V10 adversarial reweighting, and V11 final
  ensemble bets
- **Temporal validation** — strict date-based train/val split
  (cutoff 2013-05-21), not random splits, because random splits silently
  overfit the train distribution
- **Adversarial drift analysis** — confirmed AUC = 1.0 between train
  and test on cross-key target-encoding features, explaining V5's
  Kaggle regression
- **Multi-framework ensembling** — LightGBM LambdaRank + rank_xendcg,
  CatBoost YetiRank, and XGBoost rank tested together
- **Fault-tolerant pipelines** — per-model `try / except`, atomic CSV
  writes, resumable training, `FATAL.txt` on outer crash

## Quick Start

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/),
the competition CSV files in `data/`.

```bash
git clone <repo>
cd DataMiningAssignment2
./setup_vm.sh                                       # installs uv + venv + smoke import
```

```bash
# Reproduce V4 (the best Kaggle submission, 0.42021)
uv run python pipelines/legacy/v4_ensemble.py

# Reproduce V6 LOO-9 (temporal-clean baseline)
uv run python pipelines/v6.py
uv run python pipelines/v6_submit.py
```

Run from the project root. Full reproducibility commands per submission:
[Kaggle Scoreboard](docs/final_kaggle_results.md).

## Results

| version | Kaggle public NDCG@5 | Δ vs V4 |
|---|---:|---:|
| **V4 ensemble** | **0.42021** | — |
| V9 overnight diversity | 0.42012 | −0.00009 |
| V6 LOO-9 temporal | 0.42004 | −0.00017 |
| V11 mega-bag (23 models equal) | 0.42003 | −0.00018 |
| V11 safe-push | 0.41995 | −0.00026 |
| V5 cross-key TEs | 0.41943 | −0.00078 |
| V10 adversarial reweight | 0.41903 | −0.00118 |
| V4.2 Phase 2 | 0.41639 | −0.00382 |

V4 ensemble remained the highest-scoring submission across all attempted
versions. Per-version narrative: [Journey](docs/journey.md).

## Project Structure

```
expedia-rank/
├── data/                            # raw CSVs (gitignored)
├── src/                             # importable library
│   ├── features.py                  #   build_features (143-col pipeline)
│   ├── data_loader.py / evaluate.py / submission.py / config.py
│   └── artifacts.py
├── pipelines/                       # runnable end-to-end workflows (V5 → V10)
│   ├── v6.py / v6_submit.py         #   ★ V6 temporal-clean ensemble
│   ├── overnight_final_batch.py     #   V9 large diversity batch
│   ├── adversarial_reweight_batch.py # V10 adversarial reweighting
│   └── legacy/                      #   V3 / V4 / Phase 2-3 archive
├── scripts/                         # diagnostic + recovery scripts
├── notebooks/                       # executed EDA notebooks (01–04)
├── docs/                            # documentation (see below)
├── diagnostics/                     # per-experiment outputs
├── artifacts/                       # per-run JSON / CSV metadata
└── submissions/                     # Kaggle CSVs (gitignored, READMEs tracked)
```

## Documentation

| Guide | Description |
|---|---|
| **[Journey](docs/journey.md)** | Chronological narrative V3 → V11 |
| **[Final Kaggle Results](docs/final_kaggle_results.md)** | Authoritative scoreboard + reproducibility commands |
| **[Lessons Learned](docs/lessons_learned.md)** | Methodological and technical takeaways |
| **[Architecture](docs/architecture.md)** | Code structure, data flow, validation contracts |
| **[Results](docs/results.md)** | Experiment-level NDCG@5 tables |
| **[Next Steps](docs/next_steps.md)** | Unattempted ideas, ordered by expected leverage |
| **[V4 Phase 2 Summary](docs/v4_phase2_summary.md)** | Detailed V4 + Phase 2 narrative |
| **[Changelog](CHANGELOG.md)** | Version-by-version changelog |
| [`pipelines/README.md`](pipelines/README.md) | Index of runnable pipelines |
| [`scripts/README.md`](scripts/README.md) | Index of standalone scripts |
| [`submissions/README.md`](submissions/README.md) | Submission inventory |

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Primary model | [LightGBM](https://lightgbm.readthedocs.io/) 4.6 (LambdaRank + rank_xendcg) |
| Ensemble members | [XGBoost](https://xgboost.readthedocs.io/) 3.2 (rank:ndcg), [CatBoost](https://catboost.ai/) 1.2 (YetiRank) |
| Data | pandas, numpy, pyarrow (parquet caching) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Notebooks | Jupyter (EDA only) |

## Key Findings (short)

- **Train/test distribution drift is severe** — adversarial classifier
  scores AUC = 1.0 on cross-key target-encoding features
- **Local NDCG@5 is a treacherous metric** — random splits overfit;
  temporal split is closer to Kaggle but still leaves a +0.011 gap
- **Sample reweighting cannot fix structural drift** — V10 attempted
  this and lost −0.00118 on Kaggle
- **V4's LambdaRank ensemble is near the practical ceiling** for this
  dataset with LightGBM and the 143-feature pipeline
- **Different objectives provide more ensemble diversity than different
  `label_gain` values** — `rank_xendcg` and CatBoost `YetiRank` were
  the most useful diversifiers

Full discussion: [Lessons Learned](docs/lessons_learned.md).

## License

Academic project for the Data Mining Techniques course (Assignment 2).
Code released as-is for educational reference.

---

<p align="center">
  <sub>Submitted as team_80 ·
  <a href="https://www.kaggle.com/competitions/dmt-2026-2nd-assignment">DMT 2026 — 2nd Assignment</a></sub>
</p>
