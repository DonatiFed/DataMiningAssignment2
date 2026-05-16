# Project Snapshot — Expedia Hotel Search Ranking (DMT Assignment 2)

---

## 1. Overview

**What we're solving.** For each Expedia hotel search, rank the candidate hotels so that booked and clicked hotels appear at the top. The model scores each (search, hotel) pair; at inference time, hotels within each search are sorted by that score. Ranking is always within-query — nothing is compared across searches.

**Labels and metric.** Relevance is `5·booking_bool + click_bool·(1−booking_bool)` → grades `{0, 1, 5}`. NDCG@5 rewards placing higher-relevance hotels in the top 5 positions. Because booking carries 5× the weight of a click, the metric is almost entirely driven by where the **booked hotel** lands. The test set withholds `booking_bool`, `click_bool`, `position`, `random_bool`, and `gross_bookings_usd` — none of these can be used as features.

**Where we stand.**

| Model | Kaggle NDCG@5 | Local NDCG@5 | Notes |
|---|---|---|---|
| Expedia's own algorithm | — | 0.3967 | Measured on non-random rows; the ceiling for a naive replication |
| V1 | 0.38208 | 0.468 | Leaked target aggregates — local score is meaningless |
| V2 | 0.39149 | — | Incremental fix |
| V3 — single LambdaRank | 0.41392 | 0.412–0.417 | First honest model: no leakage, full feature set |
| **V4 — 8-model ensemble** | **0.42021** | unknown | Current best submission |
| **Target** | **≥ 0.430** | | Gap to close before 2026-05-17 |

---

## 2. EDA Findings

### 2.1 Target structure

Click rate is 4.47%, booking rate 2.79% — 95.5% of rows carry zero relevance. Two properties verified by crosstab:
- Every booking implies a click (0 exceptions).
- At most 1 booking per search: 69.3% of searches have exactly one booking, 30.7% have only clicks, 0% have neither.

**Why this matters.** The extreme imbalance (grade 5 appears in ~2.8% of rows) means most queries provide zero booking gradient. LambdaRank handles this by construction — it only computes gradients over pairs where one item has higher relevance than the other. But 30.7% of queries contribute only click-level signal, which directly limits sample efficiency. This is why the `lambdarank_click3` ensemble member assigns 3× weight to clicks — it extracts more gradient from those queries.

---

### 2.2 Position bias

This is the central obstacle in the dataset. Training labels are observational: hotels displayed higher receive more clicks and bookings purely due to exposure. The effect is large:

| Position | Click rate | Booking rate |
|---|---|---|
| 1 | 19.25% | 14.10% |
| 10 | 4.35% | 2.55% |
| 40 | 1.52% | 1.52% |

**Quantifying the bias.** Using only `random_bool=1` rows (hotels shown in random order, unconfounded by quality), we fit a propensity model:

```
propensity(pos) = 0.6146 / (pos + 3.5352)
```

The position 1 / position 40 propensity ratio is **9.6×** — a hotel at rank 1 receives nearly 10× the click exposure of a hotel at rank 40, regardless of quality.

**Confirming the signal vs noise split.** Booking rates on `random_bool=1` rows vs `random_bool=0` rows: **0.53% vs 3.74%** (7× gap). This proves Expedia's algorithm is already doing something sensible — non-random rows are pre-selected for quality. On random rows, quality correlations with position vanish (Spearman ≈ 0), while on non-random rows `prop_location_score2` correlates at −0.26 with position.

**Decision: IPW.** We use Inverse Propensity Weighting during training. Rows at low positions are upweighted to correct for reduced exposure: `weight = max(propensity) / propensity(pos)`, clipped to [0.1, 10]. Applied only to `random_bool=0` rows (random rows are already unbiased). The `lambdarank_noipw` ensemble member exists specifically to test whether IPW helps at the model level — its presence in the ensemble is an implicit A/B.

**Decision: exclude position from features.** Ablation on a 10% sample: NDCG@5 **0.530 with position vs 0.499 without**. The gap is illusory — position does not exist in test. Including it would produce a model that replicates Expedia's ranking rather than discovering genuine quality signals.

---

### 2.3 Missing values

Several columns have severe missingness:

| Column | Missing rate |
|---|---|
| Competitor columns (comp1–8) | 55–97% per column |
| `srch_query_affinity_score` | ~94% |
| Visitor history columns | ~95% |
| `orig_destination_distance` | ~32% |
| `prop_location_score2` | ~22% |

**Missingness is informative, not just noise.** Hotels with `prop_location_score2` present have click rate 5.18% and booking rate 0.59%, vs 2.99% / 0.25% when missing — a 2× booking rate difference. This means the missing indicator itself carries signal.

**Missingness is stable train → test** (max drift 0.3pp across all columns), so using it directly as a feature does not introduce distribution shift. Decision: add `has_location_score2`, `has_visitor_history`, `has_query_affinity`, `has_distance`, `has_historical_price` as binary features rather than imputing.

**Competitor columns are structurally sparse.** Individual `comp{i}_rate` / `comp{i}_inv` columns are missing together by competitor index. Retaining 24 sparse columns adds noise. Decision: aggregate into `comp_rate_sum`, `comp_cheaper_count`, `comp_rate_advantage`, etc., and drop the raw columns.

---

### 2.4 Feature signal

Analysis run on `random_bool=1` rows to avoid position confounding.

**Tier 1 — within-query variance.** These features vary across hotels within the same search and carry direct ranking signal: `price_usd`, `prop_starrating`, `prop_review_score`, `prop_location_score1/2`, `prop_brand_bool`, competitor aggregates, `prop_log_historical_price`. A model can use them directly to distinguish hotels within a query.

**Tier 2 — constant within query.** Search-level parameters (`srch_adults_count`, `srch_destination_id`, `site_id`, visitor country) are identical for all hotels in a search. Used raw, they produce zero ranking gradient. Decision: use them only via interactions with Tier 1 features — e.g., `price_per_person`, `price_per_night`, `is_domestic`, `is_family`.

**The booked/clicked/ignored price pattern.** Mean prices by outcome: booked **$260**, clicked-only **$344**, ignored **$252**. Users click expensive hotels but book cheaper ones. This is the strongest actionable signal for separating grade 5 from grade 1, and it drove the emphasis on within-query price rank features (`price_rank`, `price_vs_median`, `is_cheapest`).

**Key correlations** (|ρ| > 0.5): `prop_starrating ↔ price_usd` (0.55), `prop_location_score1 ↔ score2` (0.53), `prop_log_historical_price ↔ price_usd` (0.57). None severe enough to drop either member; both retained.

---

### 2.5 Single-feature baselines

Computed on the full training set as orientation for what the model needs to beat:

| Ranker | NDCG@5 |
|---|---|
| Cheapest first | 0.2049 |
| Best value (`star / log(price)`) | 0.2486 |
| Highest stars | 0.1918 |
| Best `prop_location_score1` | 0.1769 |
| Highest review score | 0.1610 |
| Expedia's original position (non-random rows) | 0.3967 |
| **V3 LambdaRank** | **0.4170** |

Expedia's own ranker is a very strong baseline. Our model beats it by ~0.02 NDCG@5, which reflects the difficulty of the task: Expedia's algorithm already incorporates quality signals we can only approximate from the logged data.

---

### 2.6 Data-side fairness baselines

Recorded as the reference point for Task 5 of the report. These are raw data rates before any model is applied.

| Group | Click rate | Book rate |
|---|---|---|
| Family (children > 0) | 4.78% | 2.98% |
| No children | 4.40% | 2.70% |
| Domestic search | 4.43% | 2.86% |
| International search | 4.58% | 2.62% |
| Branded hotel | 4.54% | 2.92% |
| Independent hotel | 4.39% | 2.51% |
| Low-star (0–2) | 3.28% | 2.02% |
| High-star (4–5) | 5.20% | 3.11% |

Gaps are modest at the data level. The model-side fairness numbers (whether the model amplifies or reduces these gaps) are in Section 6.

---

## 3. Feature Engineering

~80 features across 9 groups. The design follows directly from the EDA findings in Section 2: Tier 1 features are used directly, Tier 2 features are only useful as interactions, and all target-derived features use k-fold OOF to prevent leakage.

---

### 3.1 Temporal (`temporal_features`)

`month`, `hour`, `dayofweek`, `is_weekend_search`.

Captures seasonality and time-of-day search patterns. These are Tier 2 (constant within a query) but useful as context for the model to weight other features differently — e.g., last-minute weekend searches may have different price sensitivity.

---

### 3.2 Missing-value flags (`missing_flags`)

`has_location_score2`, `has_visitor_history`, `has_query_affinity`, `has_distance`, `has_historical_price`.

Directly motivated by Section 2.3: missingness correlates with booking rate (2× difference for `prop_location_score2`). These flags carry signal independently of the underlying value, and are stable across train/test so they do not introduce drift.

---

### 3.3 Price features (`price_features`)

`price_diff_from_hist`, `price_ratio_to_hist`, `price_per_night`, `total_cost`, `price_per_person`, `price_per_room`, `log_price`, `log_distance`.

Raw `price_usd` is a Tier 1 feature but it conflates several effects. Breaking it down: `price_per_night` normalizes for stay length, `price_per_person` normalizes for group size, `price_diff/ratio_to_hist` captures whether the hotel is currently discounted relative to its own historical price. `log_price` is included because price effects are multiplicative, not additive.

---

### 3.4 Visitor match features (`visitor_match_features`)

`star_diff` (visitor historical star − hotel star), `abs_star_diff`, `price_diff_from_visitor_hist`, `price_ratio_to_visitor_hist`.

Only meaningful for the 5% of rows with visitor history. These capture personalisation — whether the hotel matches the user's past preferences. The `has_visitor_history` flag (Section 3.2) tells the model when to trust these values.

---

### 3.5 Competitor aggregates (`competitor_features`)

Raw `comp{1..8}_rate`, `_inv`, `_rate_percent_diff` (24 columns) → dropped after aggregation.

Decision motivated by Section 2.3: individual competitor columns are 55–97% missing and co-missing by competitor index, making them noisy. Aggregated into: `comp_rate_sum`, `comp_rate_count`, `comp_cheaper_count`, `comp_more_expensive_count`, `comp_rate_advantage`, `comp_inv_count`, `comp_no_inv_count`, `comp_rate_pct_mean/min/max`. These summaries are denser and directly express competitive positioning — whether competitors are generally cheaper/unavailable, and by how much.

---

### 3.6 Quality interactions (`quality_features`)

`value_score = prop_starrating / log1p(price_usd)`, `star_review_product`, `location_total`, `is_domestic`, `starrating_is_zero`, `review_is_zero`.

Tier 2 search features (site, visitor country) become useful here via `is_domestic`. `starrating_is_zero` and `review_is_zero` separate "truly zero" from missing — important because LightGBM would otherwise conflate imputed zeros with genuine zeroes.

---

### 3.7 Within-query / listwise features (`listwise_features`)

**The ranking backbone.** Motivated by the core insight in Section 2.4: what matters is a hotel's position relative to alternatives in the same search, not its absolute value.

**Ranks** (also normalized as `_norm = rank / group_size`):
`price_rank`, `starrating_rank`, `review_rank`, `location1_rank`, `location2_rank`, `location_total_rank`, `value_score_rank`, `price_per_star_rank`, `comp_advantage_rank`.

**Deltas vs query statistics** — how far each hotel deviates from the search's distribution:
`price_vs_mean/median/min/max`, `price_z_score`, `log_price_z_score`, `star_vs_mean`, `review_vs_mean`, `loc1_vs_mean`, `loc2_vs_mean`, `price_to_min_ratio`.

**Binary flags and composite scores:**
`is_cheapest`, `is_most_expensive`, `is_best_star`, `is_best_review`, `is_best_location1`, `n_best_flags` (count of flags), `quality_rank_avg` (average rank across quality dims), `value_gap = quality_rank_avg − price_rank_norm` (how much better/worse the hotel's quality rank is relative to its price rank — captures "good deals").

**Query-level context** (constant within query, useful for the model to gauge the search landscape):
`query_hotel_count`, `query_price_std`, `query_star_mean`, `query_price_mean`.

---

### 3.8 Cross-feature interactions (`interaction_features`)

`price_per_star`, `price_per_night_per_star`, `star_x_brand`, `query_affinity_exp = exp(clip(srch_query_affinity_score, max=0))`, `distance_x_international`, `promo_x_cheap`, `is_last_minute (≤1d)`, `is_short_window (2–7d)`, `is_long_window (>30d)`, `is_family`, `total_guests`, `price_per_guest`, `is_discounted (price/hist < 0.9)`, `is_overpriced (price/hist > 1.2)`.

These mostly convert Tier 2 search parameters into features with within-query variance, by multiplying them against Tier 1 hotel attributes. `query_affinity_exp` clips at 0 before exponentiation because the score is log-scaled and has no natural zero — negative values dominate otherwise.

---

### 3.9 Hotel aggregates — target encoding (`hotel_aggregates`)

Aggregated statistics that capture historical performance of hotels, destinations, and their combinations. These are the main source of collaborative signal (what users have done with this hotel/destination in the past).

**Single-entity encodings** (Bayesian smoothing with prior weight α):

| Group | Target | Feature | α |
|---|---|---|---|
| `prop_id` | click / booking / relevance | `prop_click_rate`, `prop_book_rate`, `prop_rel_rate` | 30 |
| `srch_destination_id` | click / booking | `dest_click_rate`, `dest_book_rate` | 30 |
| `prop_country_id` | booking | `country_book_rate` | 50 |
| `site_id` | booking | `site_book_rate` | 50 |

Higher α for coarser groupings (country, site) because those groups are large and the global prior is a better estimate — shrinking aggressively toward the mean avoids overfitting on large but noisy buckets.

**Cross-entity encodings** (lower α because cross groups are sparse — 33.2% of `prop×dest` pairs unseen in test):

| Group | Feature | α |
|---|---|---|
| `prop_id × srch_destination_id` | `prop_dest_book_rate` | 10 |
| `site_id × srch_destination_id` | `site_dest_book_rate` | 15 |
| `prop_id × site_id` | `prop_site_book_rate` | 15 |
| `visitor_country × prop_country` | `cpair_book_rate` | 20 |
| `site_id × prop_country` | `site_country_book_rate` | 20 |

**Booking-given-click:** `(book_sum + 10·global_bgc) / (click_sum + 10)` per `prop_id` → `prop_book_given_click`. Separates "hotels users click but don't book" from "hotels users click and book", which maps directly onto the grade 1 vs grade 5 distinction.

**Property price statistics:** `prop_mean_price`, `prop_std_price`, `price_vs_prop_mean`, `prop_price_zscore` — captures whether the current listing price is typical for this hotel or an anomaly.

**Average display position** (non-random only): `prop_avg_position` — Expedia's implicit quality score. Derived from `random_bool=0` rows specifically because position bias is the signal here: if Expedia consistently shows a hotel early, it's because their algorithm ranks it highly. Using random rows would dilute this.

**Destination-relative features:** `dest_mean_price/star/review`, `price_vs_dest_mean`, `star_vs_dest_mean`, `review_vs_dest_mean` — how the hotel compares to the average hotel at that destination.

---

### 3.10 Leak protection

All target-derived features (Section 3.9) use **5-fold OOF with folds split by `srch_id`**. Grouping by `srch_id` ensures all rows from the same search go to the same fold — if we split at the row level, a hotel from search X in the training fold would see the booking outcome of another hotel in search X that ended up in the held-out fold. This is a subtle but real leak.

Val and test rows receive features computed from the training source only (a separate `agg_source` argument). An assertion verifies no `srch_id` spans two folds before any training run.

This is why V1's local NDCG@5 of 0.468 was meaningless: it used same-row target statistics, making the features trivially predictive of the label they were derived from.

---

## 4. Validation Strategy

**Split design.** A random 90/10 holdout split by `srch_id`: all rows belonging to a given search stay together in either train or val, never split across both. The val set contains ~19,979 searches (~495K rows).

The split must be search-level for the same reason as OOF target encoding — a row-level split would place hotels from the same search on both sides, allowing within-query statistics computed on training rows to leak information about val rows in that same query.

The split is **not temporal** (the dataset covers 2012-11 to 2013-06 in both train and test with a random assignment). A temporal split would misrepresent the actual train/test relationship, where test queries are drawn from the same period, not from the future.

**Does local track Kaggle?** Adversarial validation (train a binary classifier to distinguish train vs test rows) gives AUC = **0.524** — barely above chance, confirming that train and test are well-matched distributions. In practice:

| Model | Local NDCG@5 | Kaggle NDCG@5 | Gap |
|---|---|---|---|
| V1 (leaky) | 0.468 | 0.38208 | −0.086 (leak exposed) |
| V3 (honest) | 0.412–0.417 | 0.41392 | ~0 |
| V4 (ensemble) | unknown | 0.42021 | — |

V3's local and Kaggle scores are within rounding error. V1's large gap was the signal that same-row target aggregates were leaking — local scored high because the model had near-direct access to the label during training.

**Robustness check.** 5 random seeds for the 90/10 split on the V3 model:

| Seed | NDCG@5 |
|---|---|
| 42 | 0.42171 |
| 123 | 0.41423 |
| 456 | 0.41152 |
| 789 | 0.41650 |
| 2024 | 0.41568 |
| **Mean ± std** | **0.4159 ± 0.0037** |

The standard deviation of 0.004 sets the detection threshold: score differences smaller than this between configurations are noise, not signal. This is why V4 ensemble members with NDCG@5 within ~0.003 of each other should not be compared purely on point estimates.

---

## 5. Model Development

### V1 → local 0.468 / Kaggle 0.38208

First end-to-end pipeline. Used LambdaRank with basic hotel and search features plus target aggregates (click rate, booking rate per `prop_id`) computed on the full training set including each row's own label.

The **0.086 local-Kaggle gap** was the diagnostic. A gap this large is the fingerprint of label leakage: the model learned to look up the answer rather than generalize. Specifically, the same-row target aggregates encoded the booking/click outcome of the current row into its own feature, making the label trivially recoverable at training time but useless at test time.

**Decision:** move all target-derived aggregates to k-fold OOF (Section 3.10).

---

### V2 → Kaggle 0.39149

Addressed the leak. The Kaggle score jumped from 0.382 to 0.391 — confirming that the gap in V1 was entirely due to leakage and not model quality. No local score was tracked at this stage.

---

### V3 → local 0.412–0.417 / Kaggle 0.41392

The first honest model. Three changes from V2:

1. **Full OOF target encoding** (Sections 3.9–3.10) — all aggregates computed fold-by-fold, with val and test receiving features from training source only.
2. **IPW** (Section 2.2) — non-random rows reweighted by inverse propensity to correct for display-position bias.
3. **Full feature set** — the complete ~80-feature catalog from Section 3, including all listwise within-query features.

LightGBM LambdaRank with the following base configuration:
```
objective=lambdarank, metric=ndcg, eval_at=[5]
learning_rate=0.03, num_leaves=400, max_depth=-1
min_child_samples=50, subsample=0.7, colsample_bytree=0.6
reg_alpha=0.1, reg_lambda=1.0
```

The local–Kaggle gap dropped to ~0, confirming the validation strategy (Section 4) is reliable. V3 becomes **the reference baseline**: any new configuration must beat 0.412 locally before being considered.

---

### V4 → Kaggle 0.42021

**Why ensemble.** The robustness check (Section 4) shows single-model variance of ±0.004 NDCG@5 across seeds. A single V3 model trained on a different 10% holdout could score anywhere from 0.411 to 0.422. Ensembling reduces this variance and, when members make different errors, improves the mean.

**Why 8 diverse members, not 8 copies.** If all members share the same objective and hyperparameters, they produce correlated predictions — ensembling correlated models gives diminishing returns. Diversity was built along four axes:

| Axis | Members | Rationale |
|---|---|---|
| Label gain (`{0,1,5}` mapping) | `lambdarank_base` (0,1,31), `lambdarank_click3` (0,3,31), `lambdarank_bal15` (0,1,15), `lambdarank_book50` (0,1,50) | Tests how aggressively to weight bookings vs clicks. `click3` recovers gradient from the 30.7% of queries with no booking. `book50` over-bets on bookings. |
| Loss function | `rank_xendcg` | Different NDCG surrogate; errors are not perfectly correlated with LambdaRank. |
| IPW on/off | `lambdarank_noipw` | If IPW hurts in some query subsets, this member partially corrects for it. |
| Training data weighting | `lambdarank_randup` (random_bool=1 rows ×2) | Upweights the unbiased subset, trading some non-random signal for cleaner labels. |
| Objective type | `booking_clf` (binary cross-entropy on `booking_bool`) | Pointwise classifier — fundamentally different from listwise. Adds signal on the grade 5 vs {0,1} distinction. |

**Blending.** Each member produces raw scores that are not comparable across members (different objectives, different score scales). Direct averaging would be dominated by the members with the widest score range. Instead: for each member, scores are converted to within-query percentile ranks before averaging, weighted by each member's validation NDCG@5. A simple unweighted average is also computed; whichever scores higher on validation is used for the test submission. The actual choice is not recoverable (checkpoints are gitignored).

**Gating logic.** The pipeline runs in four staged checkpoints to prevent submitting a regression:
1. 10% sample sanity check — abort if NDCG@5 < 0.28.
2. Full data, single model — gate on NDCG@5 ≥ 0.412 (V3 baseline).
3. Full ensemble — keep only above-median members.
4. Retrain on full training set at best iteration, generate test submission.

**Result.** Kaggle score improved from 0.41392 (V3) to **0.42021** (+0.006), near the top of the single-seed variance range — meaning the ensemble reliably picks up the upper end of what a single model can achieve, and in some configurations exceeds it through error decorrelation.

---

## 6. Model Diagnostics

All diagnostics run on a fresh V3 retrain (notebook 04) against a 10% held-out val set: **19,979 searches, 495K rows, NDCG@5 = 0.42171**. The framework retrains V3 from scratch if no saved model is present, so these numbers reflect the same configuration as Section 5 V3, not V4.

---

### 6.1 Retrieval performance

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
| Rank 6–10 | 50.4% |
| Rank 11–20 | 37.0% |
| Rank 21+ | 12.6% |

Most failures are near-misses (rank 6–10), not catastrophic misrankings — the model is directionally correct but not precise enough in the top 5.

**Score separation.** Predicted score distributions by relevance grade:

| Grade | Mean score | Std |
|---|---|---|
| 0 (ignored) | −3.364 | 1.142 |
| 1 (click-only) | −2.542 | 1.092 |
| 5 (booked) | −2.179 | 1.067 |

The model separates grades in the right direction but with substantial overlap — standard deviations (~1.1) are much larger than the mean gaps (0.36 between grade 1 and 5, 0.82 between grade 0 and 5). This is the fundamental limit: within a query, the booked hotel is not always distinguishable from clicked hotels on observable features alone.

---

### 6.2 Winner vs loser analysis

Within each search that has a booking, booked hotels vs non-booked hotels (Cohen's d):

| Feature | Booked mean | Non-booked mean | Cohen's d |
|---|---|---|---|
| `prop_location_score2` | 0.1915 | 0.1292 | **0.359** |
| `promotion_flag` | 0.303 | 0.210 | 0.214 |
| `prop_review_score` | 3.939 | 3.798 | 0.154 |
| `prop_starrating` | 3.317 | 3.166 | 0.152 |
| `comp_rate_advantage` | 0.137 | 0.037 | 0.110 |
| `prop_brand_bool` | 0.667 | 0.652 | 0.032 |
| `price_usd` | 158.1 | 167.5 | −0.007 |

`prop_location_score2` has the strongest separation (d=0.359) despite 22% missingness. Booked hotels are cheaper (−$9 mean), higher star, better reviewed, more often on promotion, and more often branded. The within-query percentile for the booked hotel on price is 0.592 — it is cheaper than ~59% of alternatives in its search.

---

### 6.3 Hard-negative analysis

For each search with a booking, the model's top-ranked non-booked hotel (the "wrong pick") is compared to the actual booked hotel. A positive difference means the model's wrong pick scores higher on that feature — i.e., the model **over-weights** it:

| Feature | Booked | Wrong pick | Difference | Pattern |
|---|---|---|---|---|
| `prop_starrating` | 3.317 | 3.452 | +0.135 | over-weighted |
| `promotion_flag` | 0.303 | 0.386 | +0.083 | over-weighted |
| `prop_review_score` | 3.939 | 4.016 | +0.077 | over-weighted |
| `prop_location_score2` | 0.192 | 0.221 | +0.029 | over-weighted |
| `comp_rate_advantage` | 0.137 | 0.191 | +0.053 | over-weighted |
| `price_usd` | 158.1 | 150.3 | **−7.8** | prefers cheaper |

The model systematically favors hotels with higher stars, better reviews, and better location scores — even when users actually booked a slightly lower-quality but cheaper hotel. This pattern intensifies in failure cases (booked rank > 5): the price gap widens to −$24.8 and the star gap to +0.32, suggesting the model's quality bias is the primary driver of hard failures.

**Easy wins missed.** 4,911 failure cases where the booked hotel ranked outside the top 5. Of these:

- 54.4% of booked hotels were cheaper than the query median
- 54.5% had higher star rating than the query mean
- 61.7% had better review score than the query mean
- 62.3% had better location score than the query mean
- **43.8% were superior on 3 or more of the above dimensions (2,150 cases)**

Nearly half the failures involve a hotel that is objectively better on most quality dimensions. The model is not failing because the signal is absent — it is failing because it cannot reconcile multiple conflicting signals (e.g., a hotel that is cheaper, better reviewed, but slightly lower star than the wrong pick).

---

### 6.4 Segment performance

| Segment | NDCG@5 | Success rate | Notes |
|---|---|---|---|
| `random_bool=0` | 0.4480 | 63.97% | Non-random rows: model benefits from quality-position correlation |
| `random_bool=1` | 0.3623 | 67.46% | Random rows: no position signal, harder to rank |
| Domestic | 0.4301 | 65.04% | |
| International | 0.4056 | 62.51% | |
| Family | 0.4406 | 65.69% | |
| No children | 0.4156 | 63.67% | |
| Short stay | 0.4364 | — | |
| Last-minute | 0.4358 | — | |
| Small group (<15 hotels) | — | **86.20%** | Fewer alternatives → easier to rank |
| Large group (30+ hotels) | — | 56.42% | More alternatives → harder, more noise |

The **random_bool NDCG gap (0.448 vs 0.362)** is the most significant finding. On non-random rows, the model partially replicates Expedia's ordering (which is already quality-sorted) and gets credit for it. On random rows, it must rank from raw features alone — the 0.086 gap is the model's true marginal value over random ordering in an unbiased context.

The **group size effect** is strong: success rate drops from 86.2% (small groups) to 56.4% (large groups). Large groups have more hotels to rank, more price/quality variance, and more near-ties — harder regardless of model quality.

---

### 6.5 Popularity bias

Within-query rank by property frequency in training:

| Popularity bucket | Mean within-query rank pct | Exposure@1 | Exposure@5 | Booking Recall@5 |
|---|---|---|---|---|
| 1–5 impressions (cold-start) | 0.603 | 0.042 | 0.222 | **77.3%** |
| 6–20 | 0.579 | 0.037 | 0.201 | 65.8% |
| 21–50 | 0.545 | 0.038 | 0.198 | 66.3% |
| 51–100 | 0.519 | 0.038 | 0.195 | 61.8% |
| 100+ (popular) | 0.496 | 0.043 | 0.205 | 63.4% |

Cold-start properties (≤5 impressions) have the highest booking recall@5 (77.3%) and the highest mean within-query rank percentile (0.60). This seems counterintuitive but is explained by group size: cold-start properties tend to appear in smaller searches with fewer alternatives, making them easier to rank correctly. The model does not appear to be suppressing cold-start properties — their target encoding features fall back to the global prior, which is a reasonable estimate.

---

### 6.6 Model-side fairness

| Dimension | Group A | NDCG@5 | Recall@5 | Mean rank | Group B | NDCG@5 | Recall@5 | Mean rank |
|---|---|---|---|---|---|---|---|---|
| Family | Family | 0.4406 | 0.657 | 5.7 | No children | 0.4156 | 0.637 | 6.0 |
| Geography | Domestic | 0.4301 | 0.650 | 5.8 | International | 0.4056 | 0.625 | 6.1 |
| Brand | Branded | 0.3372 | 0.654 | 5.8 | Independent | 0.2177 | 0.616 | 6.2 |
| Star tier | High-star (4–5) | 0.2922 | 0.685 | 5.4 | Low-star (0–2) | 0.1354 | 0.559 | 7.2 |

The branded/independent and high-star/low-star gaps are large in NDCG@5 terms (0.115 and 0.157 respectively). However, these partly reflect the data-side baselines from Section 2.6 — branded hotels have higher intrinsic booking rates — rather than purely model-induced disparity. Disentangling the two requires a counterfactual analysis not yet implemented. This is the open item for Task 5 of the report.

---

## 7. Gap Analysis & V6 Roadmap

Current best: **0.42021**. Target: **≥ 0.430**. Gap: ~0.010 NDCG@5.

---

### 7.1 Prioritized ideas

**1. Add XGBoost or CatBoost ranker to ensemble**
Expected lift: +0.003–0.008 | Effort: M

All 8 V4 members are LightGBM. Different seeds and objectives introduce some diversity, but the underlying gradient boosting implementation is identical — the residual errors are correlated. A genuinely different algorithm (XGBoost with its exact greedy splits, or CatBoost with its symmetric trees and ordered boosting) makes errors on different queries. Rank-blending pays off precisely when members disagree, and the current ensemble likely has a ceiling from algorithm homogeneity. Highest expected lift of any single change.

**2. Include train+test for catalog statistics (non-target aggregates)**
Expected lift: +0.002–0.005 | Effort: S

The ID overlap analysis shows 6% of test props and 31% of test destinations are unseen in training. For these, `prop_book_rate`, `dest_book_rate`, and cross-entity encodings fall back to the global prior — the features are essentially uninformative. Non-target statistics (counts, mean price, price std per `prop_id` and `dest_id`) can safely include test rows because they do not encode the label. This reduces cold-start fall-through for the most important group (destination) from 31% to near zero.

**3. SVD on (dest_id × prop_id) click-rate matrix → latent features**
Expected lift: +0.002–0.005 | Effort: M

The current collaborative signal is limited to marginal rates per entity (Section 3.9). A matrix factorization on the (destination × hotel) click matrix produces latent vectors that capture which hotels are co-clicked across similar destinations. This provides a true collaborative filtering signal — hotels that tend to be chosen together in similar destinations get similar embeddings. 8–16 latent dimensions as additional features. Also directly satisfies the assignment's requirement to incorporate a recommender-systems technique.

**4. Stacking — learned blender over OOF base-model predictions**
Expected lift: +0.002–0.005 | Effort: M

V4 uses fixed NDCG-proportional weights for blending. A small LightGBM ranker trained on OOF base-model predictions (+ a few raw features for context) can learn which members to trust on which query types. For example, `booking_clf` may be more reliable on queries with strong price signal; `lambdarank_randup` may be better on random-exposure subsets. A learned blender does strictly better than fixed weights when members have heterogeneous strengths across query types.

**5. Hard-negative reweighting**
Expected lift: +0.002–0.003 | Effort: M

Section 6.3 identifies 2,150 "easy win" failures — booked hotel objectively superior on 3+ quality dimensions but ranked outside the top 5. These queries represent cases where the model's decision boundary is clearly wrong and additional training signal would help. The idea: run V4 on training data, identify misranked queries where the booked hotel was clearly superior, and upweight those queries in the retrain. This turns the diagnostic output into a training-time correction.

**6. Adversarial-aware feature pruning**
Expected lift: +0.000–0.002 | Effort: S

Adversarial validation AUC is 0.524 — close to chance overall, but the top drifting features are `srch_destination_id`, `srch_booking_window`, `orig_destination_distance`, and `prop_location_score2`. Re-encoding raw IDs (already done via target encoding) and capping the raw distance/window values reduces the small remaining drift. Low leverage because AUC is already near 0.5.

**7. Label-gain tuning (V5)**
Expected lift: +0.000–0.002 | Effort: S

Already explored in V5 — different `{0,1,x}` mappings beyond the range covered by V4 members. Minimal incremental effect. Lowest priority.

---

### 7.2 V6 plan

Ordered by effort-to-lift ratio given the remaining time before the Kaggle deadline:

| Step | What | Why first | Est. time |
|---|---|---|---|
| 1 | Catalog stats from train+test | Small effort, directly fixes the 31% destination cold-start gap | 1–2 h |
| 2 | SVD latent features (8 dims) | Medium effort, new signal type + satisfies RecSys report requirement | 2–3 h |
| 3 | Add XGBoost ranker as 9th ensemble member | Highest expected lift; runs through existing V4 gating pipeline | 3–4 h |
| 4 | Re-run full V4 pipeline with new features + XGBoost member, submit | | 1–2 h |
| 5 *(if time)* | Hard-negative upweighting retrain | Marginal, only if steps 1–4 are complete | 2–3 h |

Steps 1 and 2 produce new features that feed into all ensemble members, multiplying their value. Step 3 adds algorithm diversity. Together these three changes address the two largest identified gaps: feature cold-start (31% of test destinations) and ensemble correlation (all-LightGBM V4).

---

## 8. Risk Register

**30.7% of searches have no booking.**
LambdaRank produces zero booking gradient on these queries — only click-level pairs contribute. This is a fundamental data limitation, not a modelling choice. Mitigated partially by the `lambdarank_click3` member (3× click weight) and the `booking_clf` member (pointwise binary target on `booking_bool` skips this problem entirely). No full resolution exists without additional data.

**31% of test destinations are unseen.**
`srch_destination_id` has 31% new IDs in test. All destination target-encoded features (`dest_book_rate`, `dest_click_rate`, cross-entity encodings) fall back to the global prior for these rows. The global prior is a weak signal — it assigns the same score to every hotel regardless of destination. Adding train+test for non-target aggregates (V6 Step 1) partially addresses this; the SVD approach (V6 Step 2) handles it structurally by placing new destinations in the latent space via their hotel membership.

**`prop_location_score2` is both a quality signal and a position-encoded feature.**
On non-random rows, Spearman(position, `prop_location_score2`) = −0.26 — Expedia consistently shows hotels with better location scores higher. The feature is informative for ranking but its value is partially confounded with Expedia's prior ordering. It is retained because it is the strongest single separator of booked vs non-booked (Cohen's d = 0.359, Section 6.2), but its 22% missingness means the model sees a truncated version of the signal.

**`price_usd` includes tax/fee variance.**
The ratio `gross_bookings_usd / (price × length_of_stay)` centers around 1.15, not 1.0, indicating `price_usd` is per-night but incorporates some tax/fee component inconsistently. The `total_cost` and `price_per_night` features still work correctly as **relative** comparisons within a query (all hotels in the same search face the same tax regime), but absolute price comparisons across searches are unreliable.

**Visitor history covers only 5% of rows.**
`visitor_hist_starrating` and `visitor_hist_adr_usd` are missing for 95% of rows. The visitor match features (Section 3.4) are effectively inactive for the vast majority of queries. The `has_visitor_history` flag captures the missingness signal; the underlying values contribute marginal lift only for the small subset where they exist.

---

## 9. Project Structure

```
Assignment2/
├── data/                        # train.csv, test.csv (~1.2 GB each, gitignored)
├── notebooks/
│   ├── 01_data_overview.ipynb   # Dataset statistics, missing values, ID overlap
│   ├── 02_target_and_position.ipynb  # Position bias, propensity fitting, label structure
│   ├── 03_feature_analysis.ipynb     # Feature signal, baselines, fairness baselines
│   └── 04_model_diagnostics.ipynb   # Full diagnostic suite (Section 6)
├── src/
│   ├── config.py                # Paths, NON_FEATURE_COLS
│   ├── data_loader.py           # load_train/test, make_target, split_val
│   ├── features.py              # ~80 engineered features (Section 3)
│   ├── evaluate.py              # NDCG@k, recall@k
│   └── submission.py            # Write submission CSV
├── run_baseline.py              # V3: single LambdaRank
├── run_v4.py                    # V4: staged 8-model ensemble (current best)
├── run_v5.py                    # V5: experimental, not submitted
├── models/                      # Gitignored — saved models + JSON checkpoints
└── submissions/                 # Gitignored — submission CSVs
```

**To reproduce V3:** `python run_baseline.py` — trains from scratch, writes submission.

**To reproduce V4:** `python run_v4.py` — runs the 4-stage gated pipeline. Resumes from cached `models/v4/*.json` checkpoints if present; retrains from scratch otherwise (~2–3 h on a laptop).

**To rerun diagnostics:** Open `notebooks/04_model_diagnostics.ipynb`. If `models/v3_gbdt.txt` is absent (gitignored), the notebook retrains V3 automatically before running the diagnostic cells.

**Dependencies:** `pandas≥2.0`, `numpy≥1.24`, `scikit-learn≥1.3`, `lightgbm≥4.0`, `matplotlib≥3.7`, `seaborn≥0.12`.

---

## 10. Open Items

**V5 not submitted.** `run_v5.py` exists but was a label-gain tuning experiment. Section 7.1 rates this as the lowest-leverage idea; no submission is planned unless V6 underperforms expectations.

**Bias mitigation not implemented.** Section 6.6 records model-side fairness gaps (branded vs independent: NDCG gap 0.115; high-star vs low-star: 0.157). The data-side baselines (Section 2.6) partially explain these gaps, but no counterfactual analysis or mitigation step has been implemented. This is required for Task 5 of the report (15/100 points). Needs to be addressed before the report deadline (2026-05-24).

**Report not started.** LNCS template, max 14 pages + 2-page process report. The report requires a recommender-systems technique (planned: SVD latent features, V6 Step 2) and a fairness/bias analysis (Task 5). Both are currently gaps.

**Report task mapping:**

| Report task | Status | Where in SNAPSHOT |
|---|---|---|
| Dataset & EDA | Done | Sections 2, 4 |
| Feature engineering | Done | Section 3 |
| Model & results | Done | Sections 5, 6 |
| RecSys technique | Planned (SVD, V6) | Section 7.1 |
| Bias/fairness | Data-side done, model mitigation missing | Sections 2.6, 6.6 |
