# Expedia Hotel Search Ranking — Data Mining Techniques Assignment 2

Learning-to-rank on the Expedia Personalized Hotel Search Kaggle dataset.
**Metric: NDCG@5.** The model must sort the hotels shown for each search
query (`srch_id`) so that the booked hotel appears as high as possible.

## Headline result

**Kaggle public NDCG@5 = 0.42021 (V4 ensemble).**

V4 was the best Kaggle submission across all attempted versions. Six
subsequent versions (V5 through V11) were explored to try to surpass it
but did not improve the score on the Kaggle leaderboard. The full
journey, including what was learned from each version, is documented in
`docs/journey.md`.

## Kaggle public scoreboard (all uploaded submissions)

| version | submission | Kaggle public NDCG@5 |
|---|---|---:|
| **V4 ensemble** ★ | `submissions/submission_v4_20260515_151132.csv` | **0.42021** |
| V9 overnight diversity | `submission_overnight_best_deployable_20260517_095946.csv` | 0.42012 |
| V6 LOO-9 (temporal split) | `submission_v6_loo9_20260516_184304.csv` | 0.42004 |
| V11 MEGA-BAG (23-model equal rank-avg) | `submission_FINAL_megabag_25equal_*.csv` | 0.42003 |
| V11 SAFE-PUSH | `submission_FINAL_safepush_v75_6div_*.csv` | 0.41995 |
| V5 ensemble (added cross-key TEs) | `submission_v5_ensemble_*.csv` | 0.41943 |
| V10 adversarial reweighting | `submission_adv_reweight_*.csv` | 0.41903 |
| V4.2 (Phase 2 single model) | `submission_phase2_best_*.csv` | 0.41639 |

★ = selected as one of two final submissions for the private leaderboard.
Top of the public leaderboard at the time: approximately 0.46.

## Repository layout

```
.
├── README.md                          ← this file
├── CHANGELOG.md                       ← version-by-version changelog
├── pyproject.toml / requirements.txt / uv.lock / setup_vm.sh
│
├── data/                              raw CSVs (gitignored)
├── notebooks/                         EDA notebooks (executed)
│
├── src/                               importable library
│   ├── config.py                       data paths, column lists
│   ├── data_loader.py                  load_train / load_test / split_val
│   ├── features.py                     143-feature pipeline + IPW + k-fold TE
│   ├── evaluate.py                     NDCG@5 helpers
│   ├── submission.py                   CSV writer
│   └── artifacts.py                    per-run artifact saving
│
├── pipelines/                         end-to-end runnable workflows
│   ├── README.md
│   ├── temporal_validation.py          temporal split + V4 anchor on temporal val
│   ├── evaluate_variant.py             single-feature variant harness
│   ├── overnight_experiments.py        bulk LightGBM grid (random split)
│   ├── v5.py / v5_2.py                 V5 (cross-key TEs) and V5.2 (drift-TE ablation)
│   ├── v6.py / v6_submit.py            ★ V6 temporal-clean ensemble + submission
│   ├── phase7_batch.py                 V7 — failure-pattern feature variants
│   ├── phase7_weighted_batch.py        V7 — weighted-ensemble tuning of V7 features
│   ├── structural_batch.py             V8 — structural diversity (objectives, regularization)
│   ├── overnight_final_batch.py        V9 — large diversity batch (LGBM + XGB + CatBoost)
│   ├── adversarial_reweight_batch.py   V10 — adversarial sample reweighting
│   └── legacy/                         V3, V4, Phase 2/3 earlier pipelines
│
├── scripts/                           one-off diagnostic and recovery scripts
│   ├── README.md
│   ├── diagnose_v5_gap.py              V4↔V5 Kaggle gap forensics (found adv AUC = 1.0)
│   ├── eda_summary.py                  5 EDA tables for the report
│   ├── eda_dest_click_rate.py          dest_click_rate as a query-context feature
│   ├── failure_pattern_analysis.py     V6 booked-vs-top-wrong analysis (drove V7)
│   ├── ensemble_rank_avg.py            quick rank-average sandbox
│   ├── ensemble_search.py              ensemble member-subset combinator search
│   ├── ensemble_normalization_search.py paper-inspired ensemble normalization sweep
│   ├── temporal_rescore_overnight.py   rescore overnight V4 boosters on temporal val
│   ├── te_safe_single.py               single-key TE drift safety experiment
│   ├── aggregate_results.py            promote per-run artifacts → experiment_logs
│   ├── overnight_submit_best.py        V9 submission recovery (after pipeline crash)
│   ├── overnight_xgb_rescue.py         V9 XGB retraining with NaN/Inf cleanup
│   └── build_two_final_submissions.py  V11 final 2 submissions (no retraining)
│
├── docs/                              ← documentation
│   ├── journey.md                      ★ chronological narrative V3 → V11
│   ├── final_kaggle_results.md         ★ authoritative scoreboard + reproducibility
│   ├── lessons_learned.md              ★ methodological + technical takeaways
│   ├── architecture.md                 code structure + data flow
│   ├── results.md                      experiment-level NDCG@5 results
│   ├── next_steps.md                   forward queue (unattempted ideas)
│   ├── v4_phase2_summary.md            V4 + Phase 2 narrative
│   └── archive/                        superseded planning docs
│
├── experiment_logs/                   aggregated CSV trackers
├── artifacts/                         per-run outputs (JSON/CSV tracked, binaries gitignored)
├── diagnostics/                       analyses + cached features per experiment
├── models/                            saved boosters (gitignored)
├── submissions/                       Kaggle CSVs (gitignored, see scoreboard above)
└── logs/                              session logs (gitignored)
```

## Quick start

```bash
./setup_vm.sh              # one-time: install uv + venv + smoke import

# Reproduce the winning V4 ensemble
uv run python pipelines/legacy/v4_ensemble.py

# Reproduce V6 LOO-9 (temporal-clean baseline)
uv run python pipelines/v6.py
uv run python pipelines/v6_submit.py

# Test a single feature variant
uv run python pipelines/evaluate_variant.py --variant prop_click_rate_pos_adj_s40_oof
```

Always run from the project root (Python module imports require `cwd` on
`sys.path`).

## What to read first

If you are reading this repository for the first time:

1. **`docs/journey.md`** — the chronological story of how the model
   evolved. Best starting point.
2. **`docs/final_kaggle_results.md`** — definitive scoreboard with
   reproducibility commands.
3. **`docs/lessons_learned.md`** — methodological and technical
   takeaways with supporting evidence.
4. **`docs/architecture.md`** — code structure and data flow.

## Key technical findings

A short version of `docs/lessons_learned.md`:

- **Train/test distribution drift is the dominant problem** on this
  dataset. The V5 cross-key target-encoding features had adversarial AUC
  = 1.0 against the test set, meaning a binary classifier could perfectly
  distinguish train from test rows. The V10 adversarial classifier
  independently rediscovered V5's failure mode and confirmed the same
  features as the drift drivers.
- **Sample reweighting (importance ratios) cannot fix structural drift.**
  V10 attempted this and lost −0.00118 on Kaggle. The drift is in
  features the model must use to rank well; penalizing their importance
  hurts generalization.
- **Local NDCG@5 is a treacherous metric.** Random validation overfits
  the train distribution; temporal validation is closer to Kaggle but
  still leaves a +0.011 local→Kaggle gap. Roughly 89% of any local
  improvement was absorbed by drift on the path from local to Kaggle.
- **V4's LambdaRank ensemble with `label_gain` sweep is near the
  practical ceiling** for this dataset using LightGBM and the
  143-feature engineering pipeline. Six subsequent versions and roughly
  50 trained models could not surpass it on Kaggle.
- **Different objectives (`rank_xendcg`, CatBoost YetiRank) provide more
  diversity than different `label_gain` values.** This was the single
  most consistent ensemble-improvement direction.

## Key invariants (carried into any new pipeline)

- **Temporal split** is the validation contract going forward. Anchor:
  V4_ANCHOR temporal NDCG@5 = 0.40401. Reference:
  `pipelines/temporal_validation.py`.
- **No `lgb.Dataset.construct()` outside the per-configuration loop** —
  let `lgb.train` construct lazily so `seed=456` propagates correctly to
  bin sampling. See `docs/v4_phase2_summary.md` for the original bug.
- **Kaggle submission CSV header must be `srch_id,prop_id` (lowercase)** —
  matches `data/submission_sample.csv`.
- **CP and DS are separate ensemble members.** In-model feature stacking
  is consistently anti-additive at this feature count (proven in
  `diagnostics/eval_variants/results.csv`).
- **Any new target-encoding feature requires an adversarial-AUC check**
  on the generated feature distribution between train and test (proxy:
  temporal val).

## Setup

```bash
./setup_vm.sh
```
Installs `uv` if missing, creates the virtual environment, runs a smoke
import test.
