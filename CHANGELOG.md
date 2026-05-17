# Changelog

Version-by-version changelog for the Expedia Personalized Hotel Search
Kaggle project. Each version is a coherent modelling effort with its own
validation strategy. See `docs/journey.md` for the full narrative and
`docs/final_kaggle_results.md` for the authoritative scoreboard.

---

## V11 — final two submissions (Kaggle 0.41995 and 0.42003)

With two Kaggle submission slots remaining and no clear path to surpass
V4, two ensemble strategies were tested using only saved test predictions
(no retraining):

**V11a — SAFE-PUSH:** `V6 LOO-9 @ 0.75 + 6 diversifiers @ 0.0417 each`
(cb_rank_C_deeper, cb_rank_A, xendcg_conservative, xendcg_reg_seed42,
xendcg_reg_seed123, xendcg_reg_seed456). Extension of the V9 winner with
two additional diversifiers and slightly reduced V6 weight.
**Kaggle 0.41995** (Δ vs V4 = −0.00026).

**V11b — MEGA-BAG:** 23 trained models, equal rank-average within
`srch_id`. V6 effective weight 9/23 = 39%. Pure diversity bet with no
preferred backbone. **Kaggle 0.42003** (Δ vs V4 = −0.00018). Slightly
better than expected — the diversity DID add signal even with V6
backbone diluted.

**Code:** `scripts/build_two_final_submissions.py`.

---

## V10 — adversarial reweighting (Kaggle 0.41903, regression)

Trained a binary classifier to distinguish train rows from test rows
(adversarial validation pattern). Adversarial holdout AUC = **1.0** —
perfect separability, confirming the extreme drift first surfaced in V5.
The top five discriminative features were `site_book_rate`,
`country_book_rate`, `site_country_book_rate`, `cpair_book_rate`, and
`site_id`. These are the exact features V5 had added that caused V5 to
fail; the adversarial classifier independently rediscovered V5's failure
mode.

Train rows were then weighted by the importance ratio
`P(test|x) / (1 − P(test|x))` (sqrt-clipped for stability), and seven V6
members were retrained with these weights.

**Result:** the adversarially-reweighted V6 alone scored *worse* than the
original V6 LOO-9 (local 0.40818 vs 0.40896). The best ensemble
(adv-V6 @ 0.50 + 4 diversifiers @ 0.125 each) reached local 0.40997, but
**Kaggle = 0.41903** — a −0.00118 regression vs V4 (0.42021).

**Conclusion:** sample reweighting cannot fix structural drift. The drift
is in features the model MUST use to rank well; penalizing their
importance hurts generalization. The right tool for this kind of drift is
feature engineering that produces non-drifting alternatives.

**Code:** `pipelines/adversarial_reweight_batch.py`.

---

## V9 — full diversity batch (Kaggle 0.42012)

A 24-model batch combining LambdaRank, rank_xendcg, XGBoost ranker,
CatBoost YetiRank, and binary classifiers across multiple seeds and
regularization profiles.

**5 XGBoost models failed** with `inf in input data` (LightGBM and
CatBoost handle infinite values natively; XGBoost requires explicit
`missing=np.nan`). A rescue script was written for a follow-up run
(`scripts/overnight_xgb_rescue.py`).

**Best ensemble:** `V6 @ 0.80 + cb_rank_C_deeper + cb_rank_A +
xendcg_conservative + xendcg_reg_seed42` each at 0.05 = local 0.40971 →
**Kaggle 0.42012** (Δ vs V4 = −0.00009, slightly above V6 LOO-9 at 0.42004).

**Critical observation:** the local→Kaggle ratio compressed sharply. The
+0.00075 local improvement over V6 became only +0.00008 on Kaggle — the
drift gap absorbed roughly 89% of the local gain.

**Code:** `pipelines/overnight_final_batch.py`,
`scripts/overnight_submit_best.py` (recovery after a `best_iter = −1`
reload bug in the in-pipeline submission step).

---

## V8 — structural diversity (first real ensemble gain, +0.00037 local)

Feature engineering was paused. 13 models with structural variation
(different loss functions, regularization profiles, hyperparameters) were
trained:

- 4 LambdaRank variants with different `label_gain`
- 3 weighting variants (`no_ipw`, `ipw_clip3`, `random_upweight`)
- 2 regularized variants (more leaves, stronger L2)
- 2 objective variants (`rank_xendcg_regularized`,
  `booking_clf_calibrated`)
- 2 with extra features (`CP_regularized`, `DS_regularized`)

**Result:** `V6 + rank_xendcg_regularized @ 0.10` reached temporal NDCG@5
= 0.40933 (+0.00037 vs V6 LOO-9). The first positive single-model
addition in the project. Leave-one-out confirmed that
`rank_xendcg_regularized` was the load-bearing addition (removing it cost
−0.00038).

**Why this worked:** `rank_xendcg` uses a different ranking objective
from V6's nine LambdaRank members, so it added independent signal that
V6 did not already have.

**Code:** `pipelines/structural_batch.py`.

---

## V7 — failure-pattern driven features

A diagnostic analysis on V6's mistakes
(`diagnostics/failure_patterns/patterns.md`) found that 76% of booked
searches had V6 mis-rank the booked hotel, with 50% of those mis-rankings
placing the booked hotel at rank 2–5 ("near misses"). Booked hotels were
consistently MORE expensive and LESS popular than V6's top-wrong pick.

Five features were designed to target these patterns:

1. `price_premium_vs_prop_hist_x_short_window`
2. `is_long_window_x_top_quartile_price`
3. `prop_rare_x_long_trip`
4. `brand_x_domestic`
5. `query_difficulty_index` = `log(candidate_count) × (1 − dest_click_rate)`

**Result:** 4 of 5 features got HOLD (small local gain on top of
V4_ANCHOR), 1 was REJECTED (`price_premium` had HIGH_DRIFT). However,
every weighted-ensemble combination with V6 LOO-9 *hurt* performance
(even at weight 0.05, the new feature models diluted V6 because their
individual NDCG@5 of around 0.404 was much weaker than V6 ensemble's
0.409).

**Conclusion:** single-feature models added on top of V6 cannot help
unless they are individually competitive with V6 alone.

**Code:** `pipelines/phase7_batch.py`,
`pipelines/phase7_weighted_batch.py`.

---

## V6 — clean temporal ensemble (Kaggle 0.42004) ★ deployed

The validation strategy was changed from random split (overfits per V5)
to **temporal split**: cutoff 2013-05-21, 159,836 training searches /
39,959 validation searches. A 10-member diverse ensemble was trained,
then trimmed via leave-one-out (the binary booking classifier was
actively harmful, so dropped).

Final V6 LOO-9 = 9-member rank-average. Two new features (CP and DS, both
clean-drift, leak-safe) were introduced.

**Local temporal NDCG@5 = 0.40896. Kaggle NDCG@5 = 0.42004**
(−0.0002 vs V4 = 0.42021).

**Important fixes baked in:**
- Kaggle submission header MUST be `srch_id,prop_id` lowercase (matching
  `data/submission_sample.csv`).
- LightGBM `Booster.best_iteration` returns −1 after `save_model` /
  reload; use `current_iteration()` instead.
- Test set has no `position` column; the `_pos_adj_oof_te` helper assert
  was relaxed to only require `prop_id` on the val/test side.

**Code:** `pipelines/v6.py`, `pipelines/v6_submit.py`.

---

## V5.2 — TE-ablation submission

Same ensemble shape as V5 but with the four high-drift cross-key TEs
(`country_book_rate`, `site_book_rate`, `site_country_book_rate`,
`cpair_book_rate`) dropped. Hypothesis: V5 lost on Kaggle because of
adversarial AUC = 1.0 on the raw cross-key TE features; removing them
should close the gap. Submitted; did not substantially beat V4.

**Code:** `pipelines/v5_2.py`.

---

## V5 — adding cross-key TEs (Kaggle 0.41943, regression)

Local random val improved to 0.42633 (+0.00121 over V4) but Kaggle
regressed to 0.41943 (−0.00078 vs V4 = 0.42021).

**Forensic analysis** (`scripts/diagnose_v5_gap.py`) ran an adversarial
classifier on the V5 feature set. The classifier scored AUC = 1.0 — a
classifier could perfectly distinguish a training row from a test row
using only these features. The TE distribution was radically different
between train and Kaggle test, causing the model to overfit train and
generalize poorly. V5.2 was the corrective ablation.

**Code:** `pipelines/v5.py`.

---

## V4 — production reference (Kaggle 0.42021) ★ deployed best

Local random-val ensemble NDCG@5 = 0.42512. Kaggle public NDCG@5 =
**0.42021** — the highest Kaggle score in the project.

V4 established:
- The 143-feature engineering pipeline (`src/features.py:build_features`).
- IPW position-bias correction.
- LambdaRank with `label_gain="0,1,15"`, `seed=456`.
- Multi-configuration ensemble across `label_gain` and `learning_rate`
  variants (the "Phase 2" sweep).

V4 also surfaced the **bin-sampling seed bug**: pre-constructing
`lgb.Dataset` outside the per-configuration loop suppressed `seed=456`
propagation to bin sampling. The fix is to let `lgb.train` construct the
dataset lazily. The V4 anchor invariant
(`pipelines/legacy/phase2_anchor_check.py`) verifies this still
reproduces random-val NDCG@5 = 0.42191.

**Code:** `pipelines/legacy/v4_ensemble.py`,
`pipelines/legacy/phase2_*.py`.
**Narrative:** `docs/v4_phase2_summary.md`.

---

## V3 — baseline

Initial single-model LambdaRank pipeline. About 0.40 local. Code:
`pipelines/legacy/v3_baseline.py`. Superseded by V4.

---

## V1–V2 — exploratory

Pre-V3 exploration phase. Not preserved as runnable code; see
`notebooks/01_data_overview.ipynb` for the EDA that fed into V3.

---

## What was not attempted

See `docs/next_steps.md` for the forward queue. Top candidates if a
longer timeline were available:

1. Neural network listwise model (PyTorch ListNet or approxNDCG) as a
   third model class
2. Two-stage propensity model: train base model on `random_bool=1` data
   only, then fine-tune on full train using the base model as a
   regularizer
3. Hard-negative mining as features
4. Heterogeneous base learners and seed bagging
