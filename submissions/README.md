# submissions/

This directory contains all Kaggle submission CSVs produced during the
project. The CSVs themselves are gitignored (each is ~60 MB); this README
documents what each one is and what it scored on the Kaggle public
leaderboard.

Submitted as **team_80** for the
[DMT 2026 — 2nd Assignment](https://www.kaggle.com/competitions/dmt-2026-2nd-assignment)
competition.

## Headline submissions

The two highest-scoring uploads:

1. **`submission_v4_20260515_151132.csv`** — V4 ensemble — Kaggle public 0.42021 ★
2. **`submission_overnight_best_deployable_20260517_095946.csv`** — V9 overnight — Kaggle public 0.42012

## Full submission inventory

| upload order | file | version | local NDCG@5 | Kaggle public | reproduces from |
|---:|---|---|---:|---:|---|
| 1 | `submission_v4_20260515_151132.csv` ★ | V4 | 0.42512 (random val) | **0.42021** | `pipelines/legacy/v4_ensemble.py` |
| 2 | `submission_phase2_best_20260515_225726.csv` | V4.2 | 0.42258 (random val) | 0.41639 | `pipelines/legacy/phase2_submit.py` |
| 3 | `submission_v5_ensemble_20260516_094741.csv` | V5 | 0.42633 (random val) | 0.41943 | `pipelines/v5.py` |
| 4 | `submission_v5_te_ablation_20260516_103823.csv` | V5.2 | — | (not uploaded) | `pipelines/v5_2.py` |
| 5 | `submission_v6_loo9_20260516_184304.csv` | V6 | 0.40896 (temporal val) | 0.42004 | `pipelines/v6.py` + `pipelines/v6_submit.py` |
| 6 | `submission_overnight_best_conservative_20260517_022323.csv` | V9 | 0.40943 (temporal val) | (not uploaded) | `pipelines/overnight_final_batch.py` |
| 7 | `submission_overnight_best_deployable_20260517_095946.csv` ★ | V9 | 0.40971 (temporal val) | 0.42012 | `pipelines/overnight_final_batch.py` + `scripts/overnight_submit_best.py` |
| 8 | `submission_adv_reweight_20260517_111219.csv` | V10 | 0.40997 (temporal val) | 0.41903 | `pipelines/adversarial_reweight_batch.py` |
| 9 | `submission_FINAL_safepush_v75_6div_20260517_123507.csv` | V11a | ≈ 0.4099 (estimated) | 0.41995 | `scripts/build_two_final_submissions.py` |
| 10 | `submission_FINAL_megabag_25equal_20260517_123507.csv` | V11b | ≈ 0.4090 (estimated) | 0.42003 | `scripts/build_two_final_submissions.py` |

★ = highest-scoring submission, used as the team's primary upload.

## Per-submission READMEs

For the submissions that have their own `*_README.md` next to the CSV in
this directory, those READMEs contain:
- Composition (members and weights)
- Local validation score
- Validation checks performed
- Risk notes

These per-submission READMEs are not gitignored.

## Validation format

All submissions in this directory follow the Kaggle Expedia format:
- header: `srch_id,prop_id` (lowercase, matching
  `data/submission_sample.csv`)
- rows: 4,959,183 (one per test row, sorted by `srch_id` then by
  predicted relevance descending within each `srch_id`)
- unique `srch_id`: 199,549 (matches `submission_sample.csv`)
- no duplicates, no NaN values

`scripts/build_two_final_submissions.py` and
`pipelines/v6_submit.py` both apply this validation programmatically
before writing.
