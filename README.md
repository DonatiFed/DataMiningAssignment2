<h1 align="center">Expedia Hotel Search Ranking</h1>

<p align="center">
  <strong>Learning-to-rank on the Expedia Personalized Hotel Search dataset &mdash; optimizing NDCG@5</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776ab.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LightGBM-4.0+-2ca02c.svg" alt="LightGBM 4.0+">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-f89939.svg?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-2.0+-150458.svg?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/project-academic-blue.svg" alt="Academic">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="docs/eda.md">EDA</a> &middot;
  <a href="docs/feature-engineering.md">Features</a> &middot;
  <a href="docs/validation.md">Validation</a> &middot;
  <a href="docs/models.md">Models</a> &middot;
  <a href="docs/hyperparameters.md">Tuning</a> &middot;
  <a href="docs/results.md">Results</a> &middot;
  <a href="docs/roadmap.md">Roadmap</a>
</p>

---

A learning-to-rank pipeline for the [Expedia Personalized Hotel Search](https://www.kaggle.com/c/expedia-personalized-sort) Kaggle dataset. For each hotel search, the model scores every candidate hotel and sorts them so that **booked** and **clicked** hotels land in the top positions. The scoring metric is **NDCG@5**, and because a booking carries 5&times; the weight of a click, the task is almost entirely about getting the booked hotel into the top 5.

The dataset is hard for three specific reasons: **95.5% of rows carry zero relevance**, training labels are confounded by **display-position bias**, and naive target-derived features **leak the label** unless built with out-of-fold encoding. This project addresses all three explicitly.

```
train.csv  -->   EDA    -->  feature build   -->  LightGBM LambdaRank  -->   ensemble   -->  NDCG@5
(~10M rows)   (notebooks)    (143 features,       (8 diverse members,      (rank-average   submission
                             OOF TE + IPW)        IPW position debias)        blend)
```

## Key Results

- **V4 ensemble &mdash; Kaggle NDCG@5 `0.42021`** (validation `0.42512`), an 8-model rank-averaged LightGBM ensemble. Current production reference.
- **Beats Expedia's own production ranking** (`0.3967` on non-random rows) by **~0.024 NDCG@5**.
- **Honest validation.** After moving all target-derived features to out-of-fold encoding, the local&harr;Kaggle gap collapsed from **&minus;0.086** (leaky V1) to **~0** (V3 onward).
- **143 engineered features** across 9 groups &mdash; within-query listwise ranks, out-of-fold target encoding, and Inverse Propensity Weighting to correct display-position bias.
- **Reproducibility anchor.** A `seed=456, label_gain=0,1,15` single model must reproduce val NDCG@5 `0.42191`; this caught a LightGBM `Dataset` binning bug worth &minus;0.0024 NDCG@5.

> Full evaluation, diagnostics, and fairness analysis: **[Results](docs/results.md)**

## Quick Start

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and `train.csv` / `test.csv` placed in `data/` (gitignored, ~1.2 GB each).

```bash
# Set up the environment (installs uv if missing, creates the venv, smoke-imports src/)
./setup_vm.sh
```

All pipelines run as Python modules **from the project root**, so `src/` resolves on `sys.path`:

```bash
# V4 ensemble — current Kaggle reference (8-model staged pipeline)
uv run --no-sync python -m pipelines.v4_ensemble

# V3 single-model baseline
uv run --no-sync python -m pipelines.v3_baseline

# Phase 2 label-gain sweep (6 configs)
uv run --no-sync python -m pipelines.phase2_labelgain
```

```bash
# Promote a finished run's per-run artifacts into the master experiment trackers
uv run --no-sync python scripts/aggregate_results.py \
    --run-id phase3_weighting --phase P3 \
    --change-summary "Weighting / IPW sweep (7 variants x 2 label_gains)"
```

> Full setup, data placement, and the determinism rules every new pipeline must follow: **[Validation Guide](docs/validation.md)**

## Pipeline Map

| Stage | Doc | Description |
|---|---|---|
| **Exploratory analysis** | [eda.md](docs/eda.md) | Target structure, position bias, missing-value signal, single-feature baselines |
| **Feature engineering** | [feature-engineering.md](docs/feature-engineering.md) | 143 features across 9 groups; out-of-fold leak protection |
| **Validation** | [docs/validation.md](docs/validation.md) | Search-level holdout, adversarial validation, seed-robustness, determinism rules |
| **Model development** | [models.md](docs/models.md) | V1&rarr;V4 progression, why ensemble, the 8-member diversity design |
| **Hyperparameter tuning** | [hyperparameters.md](docs/hyperparameters.md) | Base config, label-gain sweep, the LightGBM binning bug |
| **Results & diagnostics** | [results.md](docs/results.md) | Final metrics, retrieval/error analysis, segment & fairness breakdown |
| **Roadmap** | [roadmap.md](docs/roadmap.md) | Gap analysis, V6 plan, risk register |

## Project Structure

```
Assignment2/
├── README.md                  # this file
├── pyproject.toml + uv.lock + requirements.txt + setup_vm.sh
│
├── src/                       # importable library
│   ├── config.py              #   paths, NON_FEATURE_COLS
│   ├── data_loader.py         #   load_train/test, make_target, split_val
│   ├── features.py            #   build_features() — 143 features + IPW + KFold TE
│   ├── evaluate.py            #   NDCG@k, recall@k
│   ├── submission.py          #   write submission CSV
│   └── artifacts.py           #   per-run artifact saving helpers
│
├── pipelines/                 # executable training scripts (run via `python -m`)
│   ├── v3_baseline.py         #   V3 single LambdaRank baseline
│   ├── v4_ensemble.py         #   ★ V4 8-model ensemble — Kaggle reference
│   ├── phase2_labelgain.py    #   label_gain sweep (6 configs)
│   ├── phase2_anchor_check.py #   single-config V4 reproduction (diagnostic)
│   ├── phase2_submit.py       #   v4.2 submission generator
│   └── phase3_weighting.py    #   weighting / IPW sweep (scaffolded)
│
├── scripts/
│   └── aggregate_results.py   # promote per-run artifacts → experiment_logs/
│
├── notebooks/
│   ├── 01_data_overview.ipynb       # schema, missing values, ID overlap
│   ├── 02_target_and_position.ipynb # position bias, propensity fitting
│   ├── 03_feature_analysis.ipynb    # feature signal, baselines, fairness
│   └── 04_model_diagnostics.ipynb   # full diagnostic suite
│
├── docs/                      # documentation (this guide set + working notes)
├── experiment_logs/           # tracker CSVs — one row per experiment / model / ensemble
├── artifacts/                 # per-run outputs (json/csv tracked; .npy/.parquet gitignored)
└── (gitignored) data/ models/ submissions/ logs/ .venv/
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Ranking model | [LightGBM](https://lightgbm.readthedocs.io/) 4.0+ (LambdaRank, rank_xendcg) |
| Data / numerics | [pandas](https://pandas.pydata.org/) 2.0+, [NumPy](https://numpy.org/) 1.24+, [PyArrow](https://arrow.apache.org/) 15+ |
| ML utilities | [scikit-learn](https://scikit-learn.org/) 1.3+ (splits, metrics, adversarial validation) |
| Exploration | [Jupyter](https://jupyter.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) |
| Environment | [uv](https://docs.astral.sh/uv/) (locked via `uv.lock`) |
| Experiment tracking | CSV trackers in `experiment_logs/` + per-run JSON/CSV artifacts |

## License

Academic coursework &mdash; Data Mining Assignment 2. Not licensed for redistribution. The underlying dataset is governed by the [Kaggle competition rules](https://www.kaggle.com/c/expedia-personalized-sort/rules).
