# Results & Diagnostics

The headline number is **Kaggle NDCG@5 `0.42021`** (V4 ensemble). This guide goes past the single number into *where* the model wins, *where* it fails, and *why*.

## Overview

- **V4 ensemble** &mdash; Kaggle `0.42021`, validation `0.42512`. The production reference.
- **Retrieval** &mdash; the booked hotel lands in the top 5 in **64.16%** of searches; most misses are near-misses (rank 6&ndash;10).
- **Dominant failure mode** &mdash; the model **over-weights quality** (stars, reviews, location) and **under-weights price**, picking quality hotels users skipped for cheaper ones.
- **Largest segment gap** &mdash; NDCG@5 `0.448` on non-random rows vs `0.362` on random rows; the latter is the model's true marginal value over random ordering.

> Diagnostics (sections below the scoreboard) are run on a **fresh V3 retrain** against a 10% holdout (19,979 searches, 495K rows, NDCG@5 0.42171), via `notebooks/04_model_diagnostics.ipynb`. They characterise the V3 single-model configuration, not the V4 ensemble.

---

## Final scoreboard

| Model | Val NDCG@5 | Kaggle NDCG@5 | Local&rarr;Kaggle gap |
|---|---|---|---|
| **V4 ensemble** (production reference) | 0.42512 | **0.42021** | &minus;0.00491 |
| V4 best single (`lambdarank_bal15`, anchor) | 0.42191 | &mdash; | &mdash; |
| v4.2 single (`lg_0_2_15`) | 0.42258 | 0.41639 | &minus;0.00619 |

The v4.2 single model scores *higher* on validation but *lower* on Kaggle &mdash; the ensemble's narrower gap is the reason it remains the production reference ([models.md](models.md)).

---

## Retrieval performance

| Metric | Value |
|---|---|
| Booking Recall@1 | 25.34% |
| Booking Recall@3 | 49.88% |
| Booking Recall@5 | 64.16% |
| Booking Recall@10 | 82.22% |
| Click Recall@5 | 58.30% |
| Mean booked-hotel rank | 5.9 |

The model places the booked hotel in the top 5 in **64.16%** of searches with a booking. Of the 35.84% failures:

| Failure rank | Share |
|---|---|
| Rank 6&ndash;10 | 50.4% |
| Rank 11&ndash;20 | 37.0% |
| Rank 21+ | 12.6% |

Most failures are **near-misses** &mdash; the model is directionally correct but not precise enough in the top 5.

**Score separation** by true relevance grade:

| Grade | Mean score | Std |
|---|---|---|
| 0 (ignored) | &minus;3.364 | 1.142 |
| 1 (click-only) | &minus;2.542 | 1.092 |
| 5 (booked) | &minus;2.179 | 1.067 |

Grades separate in the right direction, but the standard deviations (~1.1) dwarf the mean gaps (0.36 between grades 1 and 5). This is the fundamental limit: within a query, the booked hotel is not always distinguishable from clicked hotels on observable features alone.

---

## Winner vs loser analysis

Within each search with a booking, booked vs non-booked hotels (Cohen's d):

| Feature | Booked mean | Non-booked mean | Cohen's d |
|---|---|---|---|
| `prop_location_score2` | 0.1915 | 0.1292 | **0.359** |
| `promotion_flag` | 0.303 | 0.210 | 0.214 |
| `prop_review_score` | 3.939 | 3.798 | 0.154 |
| `prop_starrating` | 3.317 | 3.166 | 0.152 |
| `comp_rate_advantage` | 0.137 | 0.037 | 0.110 |
| `prop_brand_bool` | 0.667 | 0.652 | 0.032 |
| `price_usd` | 158.1 | 167.5 | &minus;0.007 |

`prop_location_score2` is the strongest single separator (d = 0.359) despite 22% missingness. Booked hotels are cheaper, higher-star, better-reviewed, more often promoted and branded. The booked hotel's within-query price percentile is 0.592 &mdash; cheaper than ~59% of alternatives in its search.

---

## Hard-negative analysis

For each search with a booking, the model's **top-ranked non-booked hotel** (its "wrong pick") vs the actual booked hotel. A positive difference means the model over-weights that feature:

| Feature | Booked | Wrong pick | Difference | Pattern |
|---|---|---|---|---|
| `prop_starrating` | 3.317 | 3.452 | +0.135 | over-weighted |
| `promotion_flag` | 0.303 | 0.386 | +0.083 | over-weighted |
| `prop_review_score` | 3.939 | 4.016 | +0.077 | over-weighted |
| `comp_rate_advantage` | 0.137 | 0.191 | +0.053 | over-weighted |
| `prop_location_score2` | 0.192 | 0.221 | +0.029 | over-weighted |
| `price_usd` | 158.1 | 150.3 | **&minus;7.8** | prefers cheaper |

The model systematically favors higher-star, better-reviewed, better-located hotels &mdash; even when users booked a slightly lower-quality but cheaper option. The pattern **intensifies in failure cases** (booked rank > 5): the price gap widens to &minus;$24.8 and the star gap to +0.32. The model's quality bias is the primary driver of hard failures.

**Easy wins missed.** Of 4,911 failure cases (booked hotel ranked outside the top 5):

- 54.4% of booked hotels were cheaper than the query median
- 54.5% had higher star rating than the query mean
- 61.7% had better review score than the query mean
- 62.3% had better location score than the query mean
- **43.8% (2,150 cases) were superior on 3+ of these dimensions**

Nearly half the failures involve an objectively-better hotel. The model is not failing for lack of signal &mdash; it fails to reconcile multiple **conflicting** signals (a hotel cheaper and better-reviewed but slightly lower-star than the wrong pick).

---

## Segment performance

| Segment | NDCG@5 | Success rate |
|---|---|---|
| `random_bool=0` | 0.4480 | 63.97% |
| `random_bool=1` | 0.3623 | 67.46% |
| Domestic | 0.4301 | 65.04% |
| International | 0.4056 | 62.51% |
| Family | 0.4406 | 65.69% |
| No children | 0.4156 | 63.67% |
| Small group (<15 hotels) | &mdash; | 86.20% |
| Large group (30+ hotels) | &mdash; | 56.42% |

The **`random_bool` gap (0.448 vs 0.362)** is the most significant finding. On non-random rows the model partially replicates Expedia's already-quality-sorted ordering and gets credit for it; on random rows it must rank from raw features alone. The 0.362 figure is the model's true marginal value in an unbiased context.

The **group-size effect** is strong: success rate falls from 86.2% (small groups) to 56.4% (large groups). More hotels means more price/quality variance and more near-ties &mdash; harder regardless of model quality.

---

## Popularity bias

Within-query rank and booking recall by property frequency in training:

| Popularity bucket | Mean within-query rank pct | Exposure@5 | Booking Recall@5 |
|---|---|---|---|
| 1&ndash;5 impressions (cold-start) | 0.603 | 0.222 | **77.3%** |
| 6&ndash;20 | 0.579 | 0.201 | 65.8% |
| 21&ndash;50 | 0.545 | 0.198 | 66.3% |
| 51&ndash;100 | 0.519 | 0.195 | 61.8% |
| 100+ (popular) | 0.496 | 0.205 | 63.4% |

Cold-start properties have the *highest* recall@5 (77.3%) &mdash; counterintuitive, but explained by group size: rare properties tend to appear in smaller searches with fewer alternatives. The model does **not** suppress cold-start properties; their target-encoded features fall back to the global prior, a reasonable estimate.

---

## Model-side fairness

| Dimension | Group A | NDCG@5 | Group B | NDCG@5 | Gap |
|---|---|---|---|---|---|
| Family | Family | 0.4406 | No children | 0.4156 | 0.025 |
| Geography | Domestic | 0.4301 | International | 0.4056 | 0.025 |
| Brand | Branded | 0.3372 | Independent | 0.2177 | **0.115** |
| Star tier | High-star (4&ndash;5) | 0.2922 | Low-star (0&ndash;2) | 0.1354 | **0.157** |

The branded/independent and high-star/low-star gaps are large. They **partly reflect the data-side baselines** ([eda.md](eda.md)) &mdash; branded and high-star hotels have higher intrinsic booking rates &mdash; rather than purely model-induced disparity. Disentangling the two requires a counterfactual analysis not yet implemented; it is the open item for the report's bias task, tracked in [roadmap.md](roadmap.md).
