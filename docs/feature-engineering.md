# Feature Engineering

`src/features.py:build_features()` produces **143 columns** across 10 groups. The design follows directly from the [EDA findings](eda.md): Tier 1 features are used directly, Tier 2 features are only useful as interactions, and every target-derived feature is built with out-of-fold encoding to prevent leakage.

## Overview

- **143 features**, entry point `build_features()` &mdash; deterministic given a fixed `agg_source`.
- **Design principle.** Within-query variance is everything: a feature that is constant across hotels in a search carries zero ranking signal as a raw column.
- **Leak protection.** All target-derived features use 5-fold out-of-fold (OOF) encoding split by `srch_id`. Val and test rows receive features computed from a separate training source only.
- **Position-bias correction.** Inverse Propensity Weighting (IPW) is applied as per-row sample weights, not as a feature.

## Feature group summary

| Group | Count | Purpose |
|---|---|---|
| Raw pass-through | 22 | Tier 1 hotel/search attributes used as-is |
| Temporal | 4 | Seasonality and time-of-day search context |
| Missing-value flags | 5 | Binary indicators &mdash; missingness is informative |
| Price | 8 | Decompose `price_usd` into normalized, comparable views |
| Visitor match | 4 | Personalisation for the ~5% of rows with visitor history |
| Competitor aggregates | 12 | Dense summaries of 24 sparse raw competitor columns |
| Quality interactions | 6 | Composite value/quality scores |
| Listwise / within-query | 44 | **The ranking backbone** &mdash; ranks and deltas vs query stats |
| Cross-feature interactions | 14 | Convert Tier 2 search params into within-query-varying features |
| Hotel aggregates (target encoding) | 24 | Collaborative signal &mdash; historical hotel/destination performance |

The four group-level subsections below cover the design rationale; the listwise and target-encoding groups get expanded treatment because they carry most of the signal.

---

## Temporal, missing flags, price, visitor match

**Temporal** (4) &mdash; `month`, `hour`, `dayofweek`, `is_weekend_search`. Tier 2 (constant within a query) but useful as context the model can use to weight other features differently &mdash; e.g. last-minute weekend searches may be more price-sensitive.

**Missing-value flags** (5) &mdash; `has_location_score2`, `has_visitor_history`, `has_query_affinity`, `has_distance`, `has_historical_price`. Directly motivated by the EDA finding that missingness correlates with booking rate (2&times; gap for `prop_location_score2`). These flags carry signal independent of the underlying value and are stable train&rarr;test, so they introduce no drift.

**Price** (8) &mdash; `price_diff_from_hist`, `price_ratio_to_hist`, `price_per_night`, `total_cost`, `price_per_person`, `price_per_room`, `log_price`, `log_distance`. Raw `price_usd` is Tier 1 but conflates several effects: `price_per_night` normalizes for stay length, `price_per_person` for group size, `price_ratio_to_hist` captures whether the hotel is discounted vs its own history. `log_price` is included because price effects are multiplicative.

**Visitor match** (4) &mdash; `star_diff`, `abs_star_diff`, `price_diff_from_visitor_hist`, `price_ratio_to_visitor_hist`. Meaningful only for the ~5% of rows with visitor history; they capture whether the hotel matches the user's past preferences. The `has_visitor_history` flag tells the model when to trust them.

---

## Competitor aggregates & quality interactions

**Competitor aggregates** (12). The raw `comp{1..8}_rate`, `_inv`, `_rate_percent_diff` columns (24 total) are 55&ndash;97% missing and co-miss by competitor index. They are dropped after aggregation into denser summaries: `comp_rate_sum`, `comp_rate_count`, `comp_cheaper_count`, `comp_more_expensive_count`, `comp_rate_advantage`, `comp_inv_count`, `comp_no_inv_count`, `comp_rate_pct_mean/min/max`. These directly express competitive positioning &mdash; whether competitors are generally cheaper or unavailable, and by how much.

**Quality interactions** (6) &mdash; `value_score = prop_starrating / log1p(price_usd)`, `star_review_product`, `location_total`, `is_domestic`, `starrating_is_zero`, `review_is_zero`. `is_domestic` turns the Tier 2 site/country pair into a usable signal. `starrating_is_zero` / `review_is_zero` separate a *genuine* zero from a missing-imputed zero &mdash; without them LightGBM conflates the two.

---

## Listwise / within-query features (the ranking backbone)

The 44 listwise features are the most important group. They are motivated by the core EDA insight: **what matters is a hotel's standing relative to the alternatives in the same search**, not its absolute value.

**Ranks** (each also normalized as `_norm = rank / group_size`):
`price_rank`, `starrating_rank`, `review_rank`, `location1_rank`, `location2_rank`, `location_total_rank`, `value_score_rank`, `price_per_star_rank`, `comp_advantage_rank`.

**Deltas vs query statistics** &mdash; how far a hotel deviates from its search's distribution:
`price_vs_mean/median/min/max`, `price_z_score`, `log_price_z_score`, `star_vs_mean`, `review_vs_mean`, `loc1_vs_mean`, `loc2_vs_mean`, `price_to_min_ratio`.

**Binary flags and composites:**
`is_cheapest`, `is_most_expensive`, `is_best_star`, `is_best_review`, `is_best_location1`, `n_best_flags` (count of flags), `quality_rank_avg` (mean rank across quality dimensions), `value_gap = quality_rank_avg - price_rank_norm` &mdash; how much better the hotel's quality rank is relative to its price rank, capturing "good deals".

**Query-level context** &mdash; constant within a query, but lets the model gauge the search landscape:
`query_hotel_count`, `query_price_std`, `query_star_mean`, `query_price_mean`.

---

## Cross-feature interactions

The 14 interaction features mostly convert Tier 2 search parameters into within-query-varying signals by multiplying them against Tier 1 hotel attributes:

`price_per_star`, `price_per_night_per_star`, `star_x_brand`, `query_affinity_exp`, `distance_x_international`, `promo_x_cheap`, `is_last_minute` (&le;1d), `is_short_window` (2&ndash;7d), `is_long_window` (>30d), `is_family`, `total_guests`, `price_per_guest`, `is_discounted` (price/hist < 0.9), `is_overpriced` (price/hist > 1.2).

`query_affinity_exp = exp(clip(srch_query_affinity_score, max=0))` clips at 0 before exponentiation because the affinity score is log-scaled with no natural zero &mdash; negative values would otherwise dominate.

---

## Hotel aggregates &mdash; target encoding

The 24 target-encoded features are the main source of **collaborative signal**: what users have historically done with a hotel, destination, or their combination. All use **Bayesian smoothing** with a prior weight &alpha;.

**Single-entity encodings** (7) &mdash; higher &alpha; for coarser groupings, because those groups are large and the global prior is the better estimate:

| Group | Features | &alpha; |
|---|---|---|
| `prop_id` | `prop_click_rate`, `prop_book_rate`, `prop_rel_rate` | 30 |
| `srch_destination_id` | `dest_click_rate`, `dest_book_rate` | 30 |
| `prop_country_id` | `country_book_rate` | 50 |
| `site_id` | `site_book_rate` | 50 |

**Cross-entity encodings** (5) &mdash; lower &alpha; because cross groups are sparse (33.2% of `prop&times;dest` pairs are unseen in test):

| Group | Feature | &alpha; |
|---|---|---|
| `prop_id × srch_destination_id` | `prop_dest_book_rate` | 10 |
| `site_id × srch_destination_id` | `site_dest_book_rate` | 15 |
| `prop_id × site_id` | `prop_site_book_rate` | 15 |
| `visitor_country × prop_country` | `cpair_book_rate` | 20 |
| `site_id × prop_country` | `site_country_book_rate` | 20 |

**Booking-given-click** (1) &mdash; `prop_book_given_click = (book_sum + 10*global_bgc) / (click_sum + 10)` per `prop_id`. Separates hotels users *click but don't book* from those they *click and book* &mdash; mapping directly onto the grade 1 vs grade 5 distinction.

**Property price statistics** (3) &mdash; `prop_mean_price`, `prop_std_price`, `price_vs_prop_mean` (`prop_price_zscore`): is the current listing price typical for this hotel or an anomaly?

**Average display position** (1) &mdash; `prop_avg_position`, computed on `random_bool=0` rows only. Here position bias *is* the signal: if Expedia consistently shows a hotel early, its algorithm ranks it highly. (Flagged `TEST_DROP` in `feature_audit.csv` &mdash; high gain but encodes position bias even with OOF; see [roadmap.md](roadmap.md).)

**Destination-relative features** (3) &mdash; `price_vs_dest_mean`, `star_vs_dest_mean`, `review_vs_dest_mean`: how the hotel compares to the average hotel at its destination.

---

## Leak protection

All target-derived features above use **5-fold OOF encoding with folds split by `srch_id`** (`seed=42`).

- Grouping by `srch_id` ensures every row of a search lands in the same fold. A row-level split would let a hotel in the training fold see the booking outcome of another hotel from *the same search* held out in another fold &mdash; a subtle but real leak.
- Val and test rows receive features computed from the **training source only**, passed via a separate `agg_source` argument.
- An assertion verifies no `srch_id` spans two folds before any training run.

This is why **V1's local NDCG@5 of 0.468 was meaningless**: it used same-row target statistics, making each feature trivially predictive of the label it was derived from. Moving to OOF (V2&rarr;V3) collapsed the local&harr;Kaggle gap from &minus;0.086 to ~0 &mdash; see [models.md](models.md) and [validation.md](validation.md).

---

## Inverse Propensity Weighting

IPW is not a feature &mdash; it is a per-row **sample weight** applied during training:

- `compute_position_propensity()` fits the propensity curve `0.6146 / (pos + 3.5352)` from `random_bool=1` rows.
- `compute_sample_weights()` applies `weight = max_propensity / propensity(position)` to `random_bool=0` rows, sets `random_bool=1` rows to weight 1.0, and clips to `[0.1, 10.0]`.

This upweights hotels shown at low positions to correct their reduced exposure. The clip range and the IPW-on/off decision are themselves swept in [hyperparameters.md](hyperparameters.md).
