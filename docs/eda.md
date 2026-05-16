# Exploratory Data Analysis

EDA for this project is not a column tour &mdash; it is a set of **ranking-specific diagnostics** whose conclusions drive every downstream decision. The analysis lives in three notebooks (`01_data_overview`, `02_target_and_position`, `03_feature_analysis`); this guide is the distilled result.

## Overview

Four findings shape the entire pipeline:

- **Extreme imbalance** &mdash; 95.5% of rows carry zero relevance; 30.7% of searches contribute no booking signal at all.
- **Position bias** &mdash; a hotel at rank 1 gets ~9.6&times; the click exposure of one at rank 40, independent of quality. Corrected with **Inverse Propensity Weighting**; `position` is excluded from features.
- **Informative missingness** &mdash; whether a column is missing predicts booking (up to 2&times; rate gap). Encoded as **binary flags** rather than imputed.
- **Within-query signal** &mdash; only features that *vary across hotels in the same search* carry ranking signal. This split (Tier 1 vs Tier 2) drives feature engineering.

Each section below ends with where the finding is acted on.

---

## Target structure

Relevance is defined as `5 * booking_bool + click_bool * (1 - booking_bool)`, giving grades `{0, 1, 5}`. NDCG@5 rewards placing high-relevance hotels in the top 5; because a booking weighs 5&times; a click, the metric is **almost entirely driven by where the booked hotel lands**.

- Click rate: **4.47%** &mdash; Booking rate: **2.79%** &rarr; 95.5% of rows are grade 0.
- **Every booking implies a click** (0 exceptions, verified by crosstab).
- **At most one booking per search**: 69.3% of searches have exactly one booking, 30.7% have only clicks, 0% have neither.

**Why this matters.** LambdaRank only computes gradients over pairs where one item outranks another in relevance. The 30.7% of searches with no booking therefore contribute **only click-level signal** &mdash; a hard limit on sample efficiency. The test set additionally withholds `booking_bool`, `click_bool`, `position`, `random_bool`, and `gross_bookings_usd`; none can be used as features.

> Acted on in [models.md](models.md): the `lambdarank_click3` ensemble member assigns 3&times; weight to clicks to recover gradient from no-booking queries; `booking_clf` sidesteps the issue with a pointwise binary objective.

---

## Position bias

This is the central obstacle in the dataset. Training labels are **observational**: hotels displayed higher receive more clicks and bookings purely from exposure.

| Position | Click rate | Booking rate |
|---|---|---|
| 1 | 19.25% | 14.10% |
| 10 | 4.35% | 2.55% |
| 40 | 1.52% | 1.52% |

**Quantifying the bias.** Using only `random_bool=1` rows (hotels shown in random order, unconfounded by quality), a propensity model is fit against click rate by position:

```
propensity(pos) = 0.6146 / (pos + 3.5352)
```

The position 1 / position 40 propensity ratio is **9.6&times;** &mdash; a rank-1 hotel receives nearly ten times the click exposure of a rank-40 hotel, regardless of quality.

**Signal vs noise.** Booking rate on `random_bool=1` rows vs `random_bool=0` rows is **0.53% vs 3.74%** (a 7&times; gap). This confirms Expedia's algorithm already pre-selects for quality. On random rows, quality&ndash;position correlations vanish (Spearman &asymp; 0); on non-random rows, `prop_location_score2` correlates &minus;0.26 with position.

**Decision &mdash; IPW.** Non-random rows are reweighted by `weight = max(propensity) / propensity(pos)`, clipped to `[0.1, 10]`. Random rows are already unbiased and keep weight 1.0.

**Decision &mdash; exclude `position` from features.** A 10% ablation gives NDCG@5 **0.530 with position vs 0.499 without** &mdash; but the gap is illusory: `position` does not exist in test. Including it would replicate Expedia's ranking rather than discover quality signal.

> Acted on in [feature-engineering.md](feature-engineering.md) (IPW weights) and [hyperparameters.md](hyperparameters.md) (the `no_ipw` / `ipw_*` weighting sweep).

---

## Missing values

Several columns are severely incomplete:

| Column | Missing rate |
|---|---|
| Competitor columns (`comp1`&ndash;`8`) | 55&ndash;97% per column |
| `srch_query_affinity_score` | ~94% |
| Visitor history columns | ~95% |
| `orig_destination_distance` | ~32% |
| `prop_location_score2` | ~22% |

**Missingness is informative, not just noise.** Hotels with `prop_location_score2` present have click rate 5.18% / booking rate 0.59%, vs 2.99% / 0.25% when missing &mdash; a **2&times; booking-rate difference**. The missing indicator itself carries signal.

**Missingness is stable train&rarr;test** (max drift 0.3pp across all columns), so using it directly as a feature introduces no distribution shift.

**Decisions:**
- Add binary flags &mdash; `has_location_score2`, `has_visitor_history`, `has_query_affinity`, `has_distance`, `has_historical_price` &mdash; instead of imputing.
- Competitor columns are structurally sparse (the `comp{i}_rate` / `comp{i}_inv` group co-misses by competitor index). Retaining 24 sparse columns adds noise &rarr; aggregate into summaries (`comp_rate_sum`, `comp_cheaper_count`, `comp_rate_advantage`, ...) and drop the raw columns.

> Acted on in [feature-engineering.md](feature-engineering.md): groups `missing_flags` and `competitor_features`.

---

## Feature signal

Feature&ndash;target analysis is run on `random_bool=1` rows to avoid position confounding.

**Tier 1 &mdash; within-query variance.** These vary across hotels within the same search and carry direct ranking signal: `price_usd`, `prop_starrating`, `prop_review_score`, `prop_location_score1/2`, `prop_brand_bool`, competitor aggregates, `prop_log_historical_price`. A model can use them directly to separate hotels in a query.

**Tier 2 &mdash; constant within query.** Search-level parameters (`srch_adults_count`, `srch_destination_id`, `site_id`, visitor country) are identical for all hotels in a search &rarr; **zero ranking gradient as raw features**. Useful only via interactions with Tier 1 features.

**The booked / clicked / ignored price pattern.** Mean price by outcome:

| Outcome | Mean price |
|---|---|
| Booked | $260 |
| Clicked-only | $344 |
| Ignored | $252 |

Users **click expensive hotels but book cheaper ones**. This is the strongest actionable signal for separating grade 5 from grade 1, and it motivates the within-query price-rank features (`price_rank`, `price_vs_median`, `is_cheapest`).

**Key correlations** (|&rho;| > 0.5): `prop_starrating` &harr; `price_usd` (0.55), `prop_location_score1` &harr; `score2` (0.53), `prop_log_historical_price` &harr; `price_usd` (0.57). None severe enough to drop a member &mdash; all retained.

> Acted on in [feature-engineering.md](feature-engineering.md): Tier 1 used directly, Tier 2 only as interactions; the `listwise_features` group is the ranking backbone.

---

## Single-feature baselines

Rule-based rankers computed on the full training set, as orientation for what the model must beat:

| Ranker | NDCG@5 |
|---|---|
| Highest review score | 0.1610 |
| Best `prop_location_score1` | 0.1769 |
| Highest stars | 0.1918 |
| Cheapest first | 0.2049 |
| Best value (`star / log(price)`) | 0.2486 |
| Expedia's original position (non-random rows) | **0.3967** |
| V3 LambdaRank | **0.4170** |

Expedia's own ranker is a very strong baseline &mdash; no single raw feature comes close. The model beats it by ~0.02 NDCG@5, which reflects the task's difficulty: Expedia already incorporates quality signals that can only be approximated from logged data.

---

## Data-side fairness baselines

Raw data rates before any model is applied &mdash; the reference point for the report's bias analysis (Task 5):

| Group | Click rate | Book rate |
|---|---|---|
| Family (children > 0) | 4.78% | 2.98% |
| No children | 4.40% | 2.70% |
| Domestic search | 4.43% | 2.86% |
| International search | 4.58% | 2.62% |
| Branded hotel | 4.54% | 2.92% |
| Independent hotel | 4.39% | 2.51% |
| Low-star (0&ndash;2) | 3.28% | 2.02% |
| High-star (4&ndash;5) | 5.20% | 3.11% |

Gaps are modest at the data level. Whether the **model** amplifies or reduces them is a separate question &mdash; see the model-side fairness analysis in [results.md](results.md).
