# Roadmap

Current best is **Kaggle NDCG@5 `0.42021`**; the target is **&ge; 0.430**. This guide lays out how to close the ~0.010 gap, the risks carried into future work, and the open items for the report.

## Overview

- **Gap to close** &mdash; ~0.010 NDCG@5 between the V4 ensemble and the 0.430 target.
- **Two structural causes** &mdash; feature **cold-start** (31% of test destinations are unseen) and **ensemble correlation** (all eight V4 members are LightGBM).
- **V6 plan** &mdash; train+test catalog stats, SVD latent features, and an XGBoost member, ordered by effort-to-lift.
- The operational phase-by-phase queue lives in `docs/next_steps.md` (a working document, not part of this guide set).

---

## Gap analysis

| | NDCG@5 |
|---|---|
| Current best (V4 ensemble) | 0.42021 |
| Target | 0.430 |
| **Gap** | **~0.010** |

The gap is small enough that no single change closes it; it needs the combined effect of new feature signal plus algorithm diversity.

---

## Prioritized ideas

| # | Idea | Expected lift | Effort |
|---|---|---|---|
| 1 | Add XGBoost / CatBoost ranker to the ensemble | +0.003&ndash;0.008 | M |
| 2 | Train+test catalog stats (non-target aggregates) | +0.002&ndash;0.005 | S |
| 3 | SVD on the (dest × prop) click matrix &rarr; latent features | +0.002&ndash;0.005 | M |
| 4 | Stacking &mdash; learned blender over OOF base predictions | +0.002&ndash;0.005 | M |
| 5 | Hard-negative reweighting | +0.002&ndash;0.003 | M |
| 6 | Adversarial-aware feature pruning | +0.000&ndash;0.002 | S |
| 7 | Further label-gain tuning | +0.000&ndash;0.002 | S |

**1 &mdash; XGBoost / CatBoost member.** All eight V4 members are LightGBM; different seeds and objectives add some diversity, but the gradient-boosting implementation is identical, so residual errors are correlated. A genuinely different algorithm (XGBoost's exact greedy splits, CatBoost's symmetric trees + ordered boosting) errs on different queries. Rank-blending pays off precisely when members disagree &mdash; highest expected lift.

**2 &mdash; train+test catalog stats.** 6% of test props and 31% of test destinations are unseen in training; their target-encoded features fall back to the global prior. **Non-target** statistics (counts, mean/std price per `prop_id` / `dest_id`) can safely include test rows &mdash; they encode no label &mdash; cutting destination cold-start from 31% toward zero.

**3 &mdash; SVD latent features.** Current collaborative signal is limited to marginal rates per entity. Matrix factorization on the (destination × hotel) click matrix yields latent vectors capturing co-click structure; new destinations are placed in the latent space via hotel membership. 8&ndash;16 dimensions as extra features. Also satisfies the report's recommender-systems requirement.

**4 &mdash; stacking.** V4 blends with fixed NDCG-proportional weights. A small LightGBM ranker trained on OOF base-model predictions can learn which members to trust per query type &mdash; strictly better than fixed weights when members have heterogeneous strengths.

**5 &mdash; hard-negative reweighting.** [results.md](results.md) identifies 2,150 "easy win" failures where the booked hotel was objectively superior on 3+ dimensions. Upweighting these misranked queries in a retrain turns the diagnostic into a training-time correction.

**6 &mdash; adversarial-aware pruning.** Adversarial AUC is 0.524 &mdash; near chance &mdash; so leverage is low. Capping the few drifting raw features (`srch_booking_window`, `orig_destination_distance`) trims the small residual drift.

**7 &mdash; label-gain tuning.** Already explored in Phase 2 ([hyperparameters.md](hyperparameters.md)); minimal incremental effect. Lowest priority.

---

## V6 plan

Ordered by effort-to-lift ratio given the time before the Kaggle deadline:

| Step | What | Why first | Est. time |
|---|---|---|---|
| 1 | Catalog stats from train+test | Small effort; directly fixes the 31% destination cold-start | 1&ndash;2 h |
| 2 | SVD latent features (8 dims) | Medium effort; new signal type + satisfies the RecSys requirement | 2&ndash;3 h |
| 3 | Add XGBoost ranker as a 9th ensemble member | Highest expected lift; runs through the existing V4 gating pipeline | 3&ndash;4 h |
| 4 | Re-run the full ensemble with new features + XGBoost, submit | | 1&ndash;2 h |
| 5 *(if time)* | Hard-negative upweighting retrain | Marginal; only if steps 1&ndash;4 are complete | 2&ndash;3 h |

Steps 1&ndash;2 produce features that feed **every** ensemble member, multiplying their value; step 3 adds algorithm diversity. Together they address the two largest gaps: feature cold-start and ensemble homogeneity.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **30.7% of searches have no booking** | LambdaRank produces zero booking gradient on these queries | `lambdarank_click3` (3× click weight) and the pointwise `booking_clf` member; no full fix without more data |
| **31% of test destinations unseen** | Destination target-encoded features fall back to a weak global prior | V6 step 1 (train+test catalog stats); V6 step 2 (SVD places new destinations in latent space) |
| **`prop_location_score2` is part quality, part position** | Spearman(position, score2) = &minus;0.26 &mdash; value confounded with Expedia's prior ordering | Retained &mdash; it is the strongest booked/non-booked separator (d = 0.359) &mdash; but flagged as a confound |
| **`price_usd` carries inconsistent tax/fee variance** | `gross_bookings / (price × nights)` centers at 1.15, not 1.0 | Use price features only as **within-query relative** comparisons (same tax regime); avoid cross-search absolute price |
| **Visitor history covers only ~5% of rows** | Visitor-match features inactive for 95% of queries | `has_visitor_history` flag captures the missingness; underlying values give marginal lift on the small covered subset |

---

## Open report items

- **Bias mitigation not implemented.** [results.md](results.md) records model-side fairness gaps (branded vs independent NDCG gap 0.115; high-star vs low-star 0.157). The data-side baselines partly explain them, but no counterfactual analysis or mitigation step exists yet &mdash; required for the report's Task 5.
- **RecSys technique.** The report requires a recommender-systems method; the planned approach is the SVD latent features of V6 step 2.

**Report task mapping:**

| Report task | Status | Reference |
|---|---|---|
| Dataset & EDA | Done | [eda.md](eda.md), [validation.md](validation.md) |
| Feature engineering | Done | [feature-engineering.md](feature-engineering.md) |
| Model & results | Done | [models.md](models.md), [results.md](results.md) |
| RecSys technique | Planned (SVD, V6 step 2) | this guide |
| Bias / fairness | Data-side done; model mitigation missing | [eda.md](eda.md), [results.md](results.md) |
