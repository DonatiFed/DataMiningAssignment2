# Changelog

NDCG@5 on the Expedia Kaggle competition. Each version is a coherent
modelling effort with its own validation strategy. See `docs/results.md`
for the full per-experiment table.

## V6 — 2026-05-16 — 0.42004 Kaggle (≈ V4 parity)

**Local NDCG@5:** 0.40896 on temporal val (LOO-best 9-member rank-average).
**Kaggle:** 0.42004 (vs V4 0.42021 = −0.00017, essentially flat).

**Members (9, LOO-best):** `lambdarank_base` (lg=0,1,31), `lambdarank_click3`
(lg=0,3,31), `lambdarank_bal15` (= V4_ANCHOR, lg=0,1,15), `lambdarank_book50`
(lg=0,1,50), `lambdarank_noipw`, `rank_xendcg`, `lambdarank_randup` (lg=0,2,25 +
random×2 weight), `CP` (= V4 + `prop_click_rate_pos_adj_s40_oof`), `DS`
(= V4 + `prop_dest_book_rate_safe`).
**Dropped:** `booking_clf` — LOO showed it hurt the ensemble by 0.00056.

**New features added during this cycle:**
- **CP** `prop_click_rate_pos_adj_s40_oof` — 5-fold OOF position-adjusted click
  target encoding. Best single-feature gain of the session: +0.00132 local.
- **DS** `prop_dest_book_rate_safe` — smoothed `(prop_id, srch_destination_id)`
  booking rate with 3-way fallback (per-prop, per-dest, global). Lowest train/val
  drift of any positive variant (`|Δμ|/σ = 0.004`).

**Key findings:**
1. In-model feature stacking has hit a ceiling at this dataset size with
   default `colsample_bytree=1.0`. Three combinations of two positive features
   (click_posadj + te_rank, click_posadj + price_dest, click_posadj +
   prop_dest_safe) were all anti-additive. **Rank-average ensembling
   recovered the additive gain** (e.g. CP+DS in-model gave +0.00065, but
   CP+DS rank-average gave +0.00250).
2. **Local→Kaggle correlation is weak.** Local +0.00217 over the quick
   3-model V4+CP+DS rank-average translated to ~0 on Kaggle. Drift is the
   prime suspect: most members inherit V4's feature set, which had a
   documented train/test distribution gap.
3. **Drop sub-anchor members from ensembles only after LOO.** `booking_clf`
   (binary classifier on `booking_bool`) scored 0.387 solo (well below anchor
   0.404) but the LOO confirmed it was actively *hurting* the ensemble — not
   just contributing nothing.

**Code/infra additions:**
- `pipelines/temporal_validation.py` — temporal split harness.
- `pipelines/evaluate_variant.py` — per-feature variant evaluation harness
  (12 variants tested; 1 KEEP, 4 HOLDs, 7 REJECTs).
- `pipelines/v6.py` — 10-member training pipeline with per-member try/except,
  resumable saves, and LOO on the best ensemble.
- `pipelines/v6_submit.py` — submission builder, resumable (reuses any
  already-completed FULL retrains).
- `scripts/eda_dest_click_rate.py` — EDA on `dest_click_rate` as query context.
- `scripts/temporal_rescore_overnight.py` — rescored the 85 overnight
  boosters on temporal val (confirmed leakage from random→full split overlap).

**Bugs found & fixed:**
- LightGBM `Booster.best_iteration` returns −1 after `save_model`/`load`
  reload; use `current_iteration()` instead.
- Kaggle submission header must be lowercase `srch_id,prop_id` (the v6.py
  first version emitted `SearchId,PropertyId` and the resulting CSV was fixed
  post-hoc; both v6 scripts now emit the correct header).
- Test set has no `position` column; `_pos_adj_oof_te` assert relaxed to
  only require `prop_id` on the val/test side.

---

## V5.2 — 2026-05-16 — TE-ablation submission

Same ensemble shape as V5 but with the 4 high-drift cross-key TEs dropped
(`country_book_rate`, `site_book_rate`, `site_country_book_rate`,
`cpair_book_rate`). Hypothesis: V5 lost on Kaggle because adversarial
AUC = 1.0 on the raw TE features; dropping them should close the gap.

Submitted, awaiting Kaggle public score. Code: `pipelines/v5_2.py`.

---

## V5 — 2026-05-15 — 0.41943 Kaggle (regression)

**Local NDCG@5:** 0.42633 (random split) — best local of any version.
**Kaggle:** 0.41943 (vs V4 0.42021 = **−0.00078**).

V5 added 12 target-encoding features (raw `prop_click_rate`, `prop_book_rate`,
`prop_rel_rate`, plus 5 single-key and 4 cross-key book-rate TEs, all k-fold
OOF on train and full-train aggregate on val). Locally V5 beat V4 by +0.00121
but on Kaggle it lost.

**Forensics (`scripts/diagnose_v5_gap.py`):** the raw TEs had adversarial
AUC = 1.0 — a classifier trained to distinguish train vs test rows scored
perfectly on the TE features alone. The TE distribution shifted dramatically
between train and Kaggle test. V5.2 was the corrective ablation.

Code: `pipelines/v5.py`.

---

## V4 — 2026-05-13 → 2026-05-15 — 0.42021 Kaggle (production reference)

**Local NDCG@5:** 0.42512 (ensemble, random val), 0.42191 (anchor single).
**Kaggle:** 0.42021.

V4 is the production reference. It established:
- The 143-feature engineering pipeline (`src/features.py:build_features`).
- IPW position-bias correction (random vs non-random subset).
- LambdaRank with `label_gain=0,1,15`, `seed=456`.
- Multi-config ensemble across `label_gain` and `learning_rate` variants
  (the "Phase 2" sweep).

V4 also documented the **bin-sampling seed bug**: pre-constructing
`lgb.Dataset` outside the per-config loop suppressed `seed=456` propagation
to bin sampling, silently breaking reproducibility. The fix is to let
`lgb.train` construct lazily. The anchor invariant
(`pipelines/legacy/phase2_anchor_check.py`) verifies this still reproduces
val NDCG@5 = 0.42191 on the random split.

Code: `pipelines/legacy/v4_ensemble.py`, `pipelines/legacy/phase2_*.py`.
See `docs/v4_phase2_summary.md` for the full narrative.

---

## V3 — earlier — baseline

Initial single-model LambdaRank pipeline. ~0.40 local. Code:
`pipelines/legacy/v3_baseline.py`. Superseded by V4.

---

## V1–V2 — exploratory

Pre-V3 exploration phase. Not preserved as runnable code; see
`notebooks/01_data_overview.ipynb` for the EDA that fed into V3.

---

## Forward queue

See `docs/next_steps.md`. Top candidates before more feature engineering:

1. **Hard-negative mining / failure-driven features** — train a model,
   identify the prop_ids it consistently mis-ranks high, encode that
   signal as features for the next iteration.
2. **Adversarial sample reweighting** — V5 had adversarial AUC = 1.0;
   weighting train rows to match test's covariate distribution may close
   the local→Kaggle gap that V6 also showed.
3. **Heterogeneous base learners** — XGBoost rank or CatBoost listwise as
   additional ensemble members for genuine model-class diversity.
4. **Loss-side position bias** — propensity-weighted listwise loss, or
   two-stage random→non-random training, instead of features+IPW.
