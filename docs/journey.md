# Journey — from baseline to final submission

This document describes the chronological evolution of the model on this
Kaggle Expedia Personalized Hotel Search dataset (NDCG@5 metric). Each
labeled version (V3 → V11) is a coherent modelling cycle with its own
validation strategy, conclusions, and follow-up.

## Reading guide for non-specialists

A few terms used throughout this document:

- **Learning-to-rank**: instead of predicting a number, the model learns to
  sort items so the best one is at the top. NDCG@5 measures whether the
  top 5 items contain the booked hotel.
- **LambdaRank / rank_xendcg**: two listwise loss functions used by
  LightGBM. Both train the model to optimize the ranking of items inside
  each search query.
- **Target encoding (TE)**: replace a category (e.g., `prop_id`) with the
  average outcome (e.g., booking rate) observed for that category in
  training data. Powerful but prone to leakage and drift.
- **Adversarial validation**: train a separate classifier to distinguish
  train rows from test rows. If it succeeds easily, train and test are
  distributionally different — a problem for any model.
- **Local NDCG / Kaggle NDCG**: local NDCG is measured on a held-out slice
  of the training data. Kaggle NDCG is measured on the true test set after
  submission. They can disagree.

## Headline

**Best Kaggle public NDCG@5 = 0.42021**, achieved by the V4 ensemble.
Six subsequent versions (V5 → V11) were attempted with the aim of improving
the score. All underperformed V4 on Kaggle by margins ranging from −0.0001
to −0.0038, even though several scored higher on local validation.

The remainder of this document explains why each version was attempted,
what was learned, and how that informed the next move.

---

## V3 — baseline (≈ 0.40 local)

Initial single-model LightGBM LambdaRank pipeline using a 143-feature
engineering suite (`src/features.py:build_features`): k-fold target
encoding for `prop_id` and `srch_destination_id`, inverse-propensity
weighting for position bias, listwise within-query features
(price/star/review ranks, price spread, etc.). Standard configuration with
`label_gain="0,1,15"`.

**Outcome:** about 0.40 local NDCG@5 on random validation. Established the
feature pipeline that all subsequent versions inherited.

**Code:** `pipelines/legacy/v3_baseline.py`.

---

## V4 — production reference (Kaggle 0.42021) ★

A multi-configuration LambdaRank ensemble: several models trained with
different `label_gain` settings (`0,1,15`, `0,2,15`, `0,3,15`, `0,1,30`,
`0,2,25`, …), each with default inverse-propensity weighting and
`seed=456`. Final score = rank-average of all members within each
`srch_id`.

**A critical reproducibility bug surfaced during V4:** pre-constructing
`lgb.Dataset` outside the per-configuration loop silently suppressed
`seed=456` propagation to the bin-sampling step, breaking reproducibility
by roughly −0.0024 NDCG@5. The fix is to let `lgb.train` construct the
dataset lazily inside the loop. This pattern became the **V4 anchor
invariant** that every subsequent pipeline had to respect.

**Outcome:** Kaggle public NDCG@5 = **0.42021** (local random-val
ensemble 0.42512, anchor single model 0.42191). This score remained the
production reference for the rest of the project.

**Code:** `pipelines/legacy/v4_ensemble.py`,
`pipelines/legacy/phase2_anchor_check.py`.
**Detailed narrative:** `docs/v4_phase2_summary.md`.

---

## V5 — adding cross-key target encodings (Kaggle 0.41943, regression)

**Hypothesis:** more aggressive target encoding (especially cross-key
combinations) would improve V4. Added 12 TE features, of which 4 were
**cross-key** (combining two categorical IDs):

- `prop_dest_book_rate` (prop_id × srch_destination_id)
- `site_dest_book_rate` (site_id × srch_destination_id)
- `prop_site_book_rate` (prop_id × site_id)
- `cpair_book_rate` (visitor_country × prop_country)
- `site_country_book_rate` (site_id × prop_country)

**Result:** local random-val improved to 0.42633 (+0.00121 over V4) but
**Kaggle regressed** to 0.41943 (−0.00078 vs V4).

**Forensic analysis** (`scripts/diagnose_v5_gap.py`) ran an adversarial
classifier (train vs test, binary objective) on the V5 feature set. The
classifier achieved **AUC = 1.0** — it could perfectly distinguish a
training row from a test row using only these features. The TE values had
a radically different distribution between train and Kaggle test, so the
model memorized the training distribution and generalized poorly.

**This finding shaped every subsequent decision.** Train→test drift,
specifically in cross-key TE features, became the dominant problem to
manage.

**Code:** `pipelines/v5.py`, `scripts/diagnose_v5_gap.py`.

---

## V5.2 — drift-TE ablation

**Hypothesis:** drop the 4 high-drift cross-key TEs from V5, keep the
rest. Submitted but did not substantially beat V4.

**Code:** `pipelines/v5_2.py`.

---

## V6 — clean temporal ensemble (Kaggle 0.42004)

The validation strategy was changed from random-val (misleading per V5) to
**temporal validation**: training data sorted by `date_time`, cutoff at
2013-05-21, with the most recent 39,959 searches held out as val. This
mimics the Kaggle test set (also future-dated).

A 10-member diverse ensemble was trained:

- 4 LambdaRank variants with different `label_gain`
- One with no inverse-propensity weighting, one with up-weighting of
  random-display rows
- One with the `rank_xendcg` objective (different loss function)
- One binary classifier on `booking_bool` (turned out harmful, see below)
- **CP** = `prop_click_rate_pos_adj_s40_oof` — a new feature: 5-fold
  out-of-fold position-adjusted click target encoding. Single-feature gain
  +0.00132 local
- **DS** = `prop_dest_book_rate_safe` — another new feature: smoothed
  (prop_id, srch_destination_id) booking rate with a three-way fallback
  (per-prop, per-dest, global). Lowest train/val drift in the entire
  project (`|Δμ|/σ = 0.004`)

A leave-one-out analysis on the 10-member ensemble showed the binary
classifier was actively HURTING the ensemble (dropping it improved the
score by +0.00056). The final V6 LOO-9 ensemble = 9-member rank-average.

**Local temporal NDCG@5 = 0.40896. Kaggle NDCG@5 = 0.42004**, −0.0002 vs V4.

The local→Kaggle gap was much larger than for V4 (+0.011 here vs ~−0.005
for V4 ensemble on random val). This is because temporal val is stricter
than random val.

**Code:** `pipelines/v6.py`, `pipelines/v6_submit.py`.

---

## V7 — failure-pattern driven features

A diagnostic analysis (`diagnostics/failure_patterns/patterns.md`) was
run on V6's mistakes. Findings:

- 76% of booked searches had V6 mis-rank the booked hotel
- 50% of those mis-rankings put the booked hotel at rank 2–5
  ("near misses")
- Booked hotels were consistently MORE expensive than the model's
  top-wrong pick (+$16 vs prop history on average)
- Booked hotels were LESS popular than the model's top-wrong pick
  (~half the historical booking rate)

Five features were designed to target these patterns:

1. `price_premium_vs_prop_hist_x_short_window`
2. `is_long_window_x_top_quartile_price`
3. `prop_rare_x_long_trip`
4. `brand_x_domestic`
5. `query_difficulty_index` = `log(candidate_count) × (1 − dest_click_rate)`

**Result:** 4 of 5 features got HOLD (small positive local gain on top of
V4_ANCHOR), 1 was REJECTED (`price_premium` had HIGH_DRIFT). However,
every weighted-ensemble combination with V6 LOO-9 HURT performance: even
at weight 0.05, the new feature models diluted V6 (their individual
NDCG@5 around 0.404 was much weaker than V6 ensemble at 0.409).

**Conclusion:** single-feature models added on top of V6 cannot help
unless they are individually competitive with V6 alone. None were.

**Code:** `pipelines/phase7_batch.py`,
`pipelines/phase7_weighted_batch.py`.

---

## V8 — structural diversity (first real ensemble gain, +0.00037 local)

Feature engineering was paused. Instead, 13 models with structural
variation (different loss functions, regularization profiles, and
hyperparameters) were trained:

- 4 label_gain variants
- 3 weighting variants
- 2 regularized variants (more leaves, stronger L2)
- 2 objective variants (`rank_xendcg`, binary classifier)
- 2 with extra features (CP_regularized, DS_regularized)

**Result:** `v6 + rank_xendcg_regularized @ 0.10` reached **temporal
NDCG@5 = 0.40933** (+0.00037 vs V6 LOO-9). This was the first positive
single-model addition in the project. Leave-one-out confirmed that
`rank_xendcg_regularized` was the load-bearing addition (removing it
cost −0.00038).

**Why rank_xendcg helps:** it uses a different objective from V6's 9
LambdaRank members, so it adds ranking signal that V6 didn't already have.

**Code:** `pipelines/structural_batch.py`.

---

## V9 — full diversity batch (Kaggle 0.42012)

A larger 24-model batch: 5 rank_xendcg seeds + 1 conservative xendcg + 3
XGBoost rankers + 3 CatBoost rankers + 3 binary classifiers + 9 LightGBM
regularized seeds. Runtime ≈ 2.5 hours.

**5 XGBoost models failed** with `inf in input data` (LightGBM and
CatBoost handle infinite values natively; XGBoost requires either explicit
`missing=np.nan` or pre-cleaning). A rescue script was written for a
follow-up run.

**Best ensemble:** `v6 + 4 diversifiers @ 0.05 each` (cb_rank_C_deeper,
cb_rank_A, xendcg_conservative, xendcg_reg_seed42) = local temporal
0.40971 → **Kaggle 0.42012**. Slightly above V6 LOO-9 (0.42004) but still
below V4 (0.42021).

A critical observation: the local→Kaggle ratio compressed. The +0.00075
local improvement over V6 became only +0.00008 on Kaggle — about 89% of
the local gain was absorbed by the train/test drift gap.

**Code:** `pipelines/overnight_final_batch.py`,
`scripts/overnight_submit_best.py` (a recovery script after an early
crash at `best_iter=-1` for reused models).

---

## V10 — adversarial reweighting (Kaggle 0.41903, regression)

**Hypothesis:** if drift is the bottleneck, correct it directly. A binary
classifier was trained to distinguish train vs test rows. Each training
row was then weighted by its importance ratio
`P(test|x) / (1 − P(test|x))` — a standard domain-adaptation technique.
V6 members were retrained with these weights.

**Diagnostics:**
- Adversarial AUC = **1.0** (holdout) — confirming extreme drift
- The 5 most drift-discriminative features were:
  `site_book_rate`, `country_book_rate`, `site_country_book_rate`,
  `cpair_book_rate`, `site_id` — exactly the features V5 had added that
  caused V5 to fail. The adversarial classifier independently rediscovered
  V5's failure mode.

**Result:** the adversarially-reweighted V6 alone scored WORSE than V6
LOO-9 (local 0.40818 vs 0.40896) — first warning sign. The best Phase A
ensemble (adv-V6 @ 0.50 + 4 diversifiers @ 0.125 each) reached 0.40997
local, but **Kaggle = 0.41903**, a −0.00118 regression vs V4.

**Why sample reweighting failed here:** the drift is in features the
model MUST use to rank well. Down-weighting their importance penalizes
their predictive power and hurts generalization. The right tool for this
drift is feature engineering that creates non-drifting alternatives
(CP and DS were attempts at this), not sample reweighting.

**Code:** `pipelines/adversarial_reweight_batch.py`.

---

## V11 — final two submissions (Kaggle 0.41995 and 0.42003)

With two submission slots remaining and no clear path to surpass V4, two
distinct philosophies were tested using only saved test predictions
(no retraining):

**SAFE-PUSH** = `V6 @ 0.75 + 6 diversifiers @ 0.0417 each` — extension
of the V9 winner with two more diversifiers and slightly lower V6 weight.
**Kaggle = 0.41995**, below V4 by 0.00026. Confirmed that V6 @ 0.80 is
the sweet spot; lowering it hurts.

**MEGA-BAG** = 23 trained models, equal rank-average, V6 effective
weight 9/23 = 39%. Pure variance reduction via massive diversity.
**Kaggle = 0.42003**, better than expected (model diversity DID add
signal even with V6 backbone diluted), still below V4 by 0.00018.

**Code:** `scripts/build_two_final_submissions.py`.

---

## Final Kaggle scoreboard

| version | submission CSV | Kaggle public NDCG@5 | Δ vs V4 |
|---|---|---:|---:|
| **V4 ensemble** ★ | `submission_v4_20260515_151132.csv` | **0.42021** | — |
| V9 overnight | `submission_overnight_best_deployable_20260517_095946.csv` | 0.42012 | −0.00009 |
| V6 LOO-9 | `submission_v6_loo9_20260516_184304.csv` | 0.42004 | −0.00017 |
| V11 MEGA-BAG | `submission_FINAL_megabag_25equal_20260517_123507.csv` | 0.42003 | −0.00018 |
| V11 SAFE-PUSH | `submission_FINAL_safepush_v75_6div_20260517_123507.csv` | 0.41995 | −0.00026 |
| V5 | `submission_v5_ensemble_20260516_094741.csv` | 0.41943 | −0.00078 |
| V10 adv reweight | `submission_adv_reweight_20260517_111219.csv` | 0.41903 | −0.00118 |
| V4.2 (Phase 2) | `submission_phase2_best_20260515_225726.csv` | 0.41639 | −0.00382 |

**Submitted as team_80** for the
[DMT 2026 — 2nd Assignment](https://www.kaggle.com/competitions/dmt-2026-2nd-assignment)
competition. **Top of public leaderboard:** approximately 0.46. Gap from
V4: +0.04.

---

## What was confirmed

1. **The V4 LambdaRank ensemble with `label_gain` sweep is near the
   ceiling for this dataset** when using LightGBM and the 143-feature
   engineering pipeline. Six subsequent versions and roughly 50 trained
   models could not surpass it on Kaggle.

2. **Train/test drift is structural, not correctable by sample
   reweighting.** Adversarial AUC = 1.0; the drift is in target-encoding
   features (especially site/country cross-keys). Reweighting penalizes
   their importance and HURTS generalization.

3. **Local NDCG@5 is a poor predictor of Kaggle NDCG@5** when local val
   approaches the test distribution boundary. V6 LOO-9 gained +0.00075
   local over V6 base but only +0.00008 Kaggle. Drift absorbed roughly
   89% of local gains.

4. **Model diversity (different objectives, regularization, frameworks)
   DID add signal** — even with V6 effective weight at 39% in the
   MEGA-BAG submission, the Kaggle score 0.42003 was within noise of V6
   alone. But even maximum diversity could not push above V4.

5. **The path to a 0.43+ Kaggle score requires structurally different
   approaches not attempted here** — listwise neural ranking, two-stage
   propensity models, or feature engineering that targets the specific
   drift mechanisms found by adversarial analysis.

For lessons learned, see `lessons_learned.md`.
For the authoritative Kaggle scoreboard, see `final_kaggle_results.md`.
For the forward queue of unattempted ideas, see `next_steps.md`.
