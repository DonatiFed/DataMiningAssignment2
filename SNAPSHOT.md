# Project Snapshot — Expedia Hotel Search Ranking (DMT Assignment 2)

**Goal:** Predict per-search hotel ranking. Metric: **NDCG@5**. Relevance grades: 5 (booked) > 1 (clicked) > 0 (ignored).

**Status as of 2026-05-15:**
- **Best Kaggle public score: 0.42021** (V4 ensemble submission)
- **Target for next iteration: ≥ 0.43**
- **Kaggle deadline: 2026-05-17 23:55** (2 days)
- **Report deadline: 2026-05-24 23:59**

---

## 1. Dataset

| | Train | Test |
|---|---|---|
| Rows | 4,958,347 | 4,959,183 |
| Columns | 54 | 50 (no position/click_bool/booking_bool/gross_bookings_usd) |
| Unique searches | 199,795 | ~199K |
| Unique hotels | 129,113 | 129,438 |
| Date range | 2012-11-01 → 2013-06-30 | same range (random split, not temporal) |
| Group size (hotels/search) | mean 24.8, median 29, range 5–38 | mean 24.9, identical distribution |
| random_bool=1 share | ~29.6% | 29.7% |
| `srch_id × prop_id` duplicates | 0 | 0 |

**ID overlap train↔test:**
- prop_id: 94.0% overlap (7,773 test props unseen, 6.0%)
- srch_destination_id: 69% overlap (5,588 unseen, **31%**)
- prop_country_id, site_id: 100% overlap
- prop_id × srch_destination_id: **33.2% unseen** (cross-entity is much sparser)
- prop_id × site_id: **37.1% unseen**

**Adversarial validation AUC = 0.524** → train and test are well-matched, local validation will track Kaggle.

---

## 2. EDA Findings (Notebooks 01–03)

### 2.1 Target structure (notebook 02)
- Click rate **4.47%**, booking rate **2.79%**. Extreme class imbalance (95.5% relevance=0).
- **Booking always implies click** (verified by crosstab — 0 bookings without clicks).
- **Max 1 booking per search.** 69.3% of searches have exactly 1 booking; 30.7% have only clicks; **0%** have no clicks.
- Relevance label: `5*booking_bool + click_bool*(1-booking_bool)` → {0,1,5}.

### 2.2 Position bias (notebook 02) — **central finding**
- Click rate at position 1 = **19.25%**, position 10 = 4.35%, position 40 = 1.52%.
- Booking rate at position 1 = **14.10%**, position 10 = 2.55%, position 40 = 1.52%.
- **Fitted propensity from random_bool=1 data:** `propensity = 0.6146 / (position + 3.5352)`.
- Position 1/40 propensity ratio ≈ **9.6×**.
- Random vs non-random click rates: random 4.66%, non-random 4.40% (close, since both still suffer exposure bias).
- Random vs non-random **booking rates**: random **0.53%**, non-random **3.74%** → 7× gap proves Expedia's algorithm works.
- Spearman(position, quality) on **non-random**: prop_location_score2 = −0.26, prop_starrating = −0.14, price = −0.06 → quality leaks through position in non-random rows.
- Spearman(position, quality) on **random**: all near 0 (sanity check passes).
- **Ablation:** training with vs without position on 10% sample → 0.5302 vs 0.4992. Position is excluded from features (it doesn't exist in test); the gap is illusory.

### 2.3 Missing values (notebook 01)
- Heavy missingness: competitor columns (55–97% per comp), `srch_query_affinity_score` (~94%), visitor history (~95%), `orig_destination_distance` (~32%), `prop_location_score2` (~22%).
- **Missingness is informative**: prop_location_score2 missing → click 2.99% / book 0.25% vs present → 5.18% / 0.59%. → `has_*` binary flags carry signal.
- Missingness rates are **very stable train→test** (max drift 0.3pp). → can use missingness directly as feature.
- Comp columns are missing together (co-occurrence by competitor index) — aggregates work better than individual columns.

### 2.4 Feature signal (notebook 03, on random_bool=1 to avoid position confound)
- **Strongest within-query ranking signals (Tier 1, high within-query variance):** price_usd, prop_starrating, prop_review_score, prop_location_score1/2, prop_brand_bool, competitor aggregates, prop_log_historical_price.
- **Tier 2 (constant within query, raw use is zero gradient):** all srch_* params, visitor country, site_id. Only useful via **interactions** with Tier 1 (price_per_night, price_per_person, is_domestic, etc.).
- Cheapest hotel in a query: book rate **0.71%** vs 30th-cheapest: **0.15%** → `price_rank` is huge.
- Highly correlated feature pairs (|ρ| > 0.5): prop_starrating ↔ price (0.55), prop_location_score1 ↔ score2 (0.53), prop_log_historical_price ↔ price (0.57). Not severe — both members retained.
- Booked vs clicked-only vs ignored mean prices: **$260 / $344 / $252**. Clicked-only is the most expensive — users click expensive hotels but **don't book** them. → Price separation between booked and clicked-only is the biggest actionable signal (since booking=5× relevance).

### 2.5 Baseline NDCG@5 (notebook 03)
| Single-feature ranker | NDCG@5 |
|---|---|
| Cheapest first | 0.2049 |
| Best value (star / log(price)) | 0.2486 |
| Promotion first | 0.2004 |
| Highest stars | 0.1918 |
| Best location1 | 0.1769 |
| Highest review | 0.1610 |
| **Expedia's original position** (non-random rows) | **0.3967** |
| V3 LambdaRank (our model) | 0.4170 |

→ The Expedia ranker is itself a very strong baseline; ML model only beats it by ~0.02.

### 2.6 Visitor history & geography
- Only **5.0%** of rows have visitor history. Those users have higher booking rate (3.56% vs 2.73%).
- **63.4% of searches are domestic** (visitor country = prop country).
- Domestic book rate 2.86% vs international 2.62% → small gap.

### 2.7 Data-side fairness baselines (for Task 5 in report)
| Group | n | Click rate | Book rate |
|---|---|---|---|
| Family (kids > 0) | 112,757 | 4.78% | 2.98% |
| No children | 382,296 | 4.40% | 2.70% |
| Domestic | 313,708 | 4.43% | 2.86% |
| International | 181,345 | 4.58% | 2.62% |
| Branded | 313,143 | 4.54% | 2.92% |
| Independent | 181,910 | 4.39% | 2.51% |
| Low-star (0–2) | 107,786 | 3.28% | 2.02% |
| High-star (4–5) | 193,299 | 5.20% | 3.11% |

### 2.8 Popularity / cold start
- Property frequency distribution is heavy-tailed (max 2,357 impressions, median 12, 25%-ile = 4).
- Booking rate is **flat** across popularity buckets (2.77–2.89%) → no booking-rate bias by popularity per se, but rare props have noisier aggregates → cold-start affects feature reliability, not the underlying rate.

---

## 3. Model Diagnostics (Notebook 04 — V3 model, NDCG@5 = 0.417 local)

The diagnostics framework is built but is run interactively on the V3 model. Notebook content (what gets computed):

- **Recall@K**: % of booked hotels in top K predicted positions; click recall@5.
- **Booked-rank distribution**: where do the actual bookings end up?
- **Winner vs loser comparison**: feature means for booked vs non-booked within each search-with-booking, with Cohen's d and within-query percentile.
- **Hard-negative analysis (most actionable)**: for each booking, compare the booked hotel to the model's top-ranked **non-booked** hotel. Positive `wrong - booked` differences on a feature → model over-weights that feature; negative → under-weights.
- **Easy-wins missed**: failures where the booked hotel was superior on 3+ quality dimensions (cheaper than median, higher star than mean, etc.) but ranked > 5.
- **Segment failure analysis**: success rate by random_bool, domestic/international, family vs not, popularity bucket, group size.
- **Query difficulty**: NDCG@5 by group size, price IQR, star variety, #promoted.
- **Segment NDCG@5**: random_bool, domestic, family, short stay, last-minute.
- **Model popularity bias**: within-query rank (NOT raw scores — only comparable within query) by popularity bucket; exposure@1 and @5.
- **Model fairness**: NDCG@5, recall@5, mean booked rank per group (family/domestic/branded/star-tier).
- **Feature importance**: gain + split (top 25).
- **Robustness**: 5-seed split of 90/10 → mean and std of NDCG@5.
- **Score separation**: predicted-score distribution by relevance (boxplot + density overlay).

**Note:** outputs are not committed in the current notebook (it executes against a saved `models/v3_gbdt.txt` that is gitignored). The framework is ready; run after each new model to update.

---

## 4. Pipelines and Kaggle Submissions

| Version | Local NDCG@5 | Kaggle Public | Description |
|---|---|---|---|
| **V1** | 0.468 | 0.38 | Basic features + leaky aggregates (same-row target stats). Big overfit gap. |
| **V2** | — | — | Iteration (see git history). |
| **V3** (`run_baseline.py`) | 0.412–0.417 | — | K-fold OOF target encoding + IPW + tuned single LambdaRank. **The honest single-model baseline.** |
| **V4** (`run_v4.py`) | (cached in `models/v4/`) | **0.42021** | V3 features + 8-model diverse ensemble + rank-blended scoring. **Current best.** |
| **V5** (`run_v5.py`) | — | not yet submitted | V4 features + extended label_gain grid + DART. Probably marginal gain over V4 since no new features. |

### V4 ensemble configurations (`run_v4.py:289`)

Eight LambdaRank/booster members, all LightGBM, blended by **percentile rank averaged with weight = NDCG@5 of each member** (or simple average — whichever scored better on val):

| Name | Seed | Objective | label_gain {0,1,5}→{0,1,2} | Other |
|---|---|---|---|---|
| lambdarank_base | 42 | lambdarank | 0,1,31 | default |
| lambdarank_click3 | 123 | lambdarank | 0,3,31 | boosts click value 3× |
| lambdarank_bal15 | 456 | lambdarank | 0,1,15 | reduces booking premium |
| lambdarank_book50 | 789 | lambdarank | 0,1,50 | aggressive booking focus, 300 leaves, lr=0.05 |
| lambdarank_noipw | 2024 | lambdarank | 0,1,31 | no IPW reweighting |
| rank_xendcg | 314 | rank_xendcg | n/a | 350 leaves |
| lambdarank_randup | 555 | lambdarank | 0,2,25 | random_bool=1 rows weighted ×2 |
| booking_clf | 666 | binary (booking_bool) | n/a | listwise replaced by point-wise classifier |

### V4 model gating logic (`run_v4.py:595`)
- Stage 1: 10% sample sanity check; abort if NDCG@5 < 0.28.
- Stage 2: full data, single model; gate on NDCG@5 ≥ V3_baseline (0.412).
- Stage 3: full ensemble (8 models); keep above-median members.
- Stage 4: retrain on full training set at best_iter, predict on test, write submission.

### Shared LightGBM params (`run_v4.py:117`)
```
objective=lambdarank, metric=ndcg, eval_at=[5]
learning_rate=0.03, num_leaves=400, max_depth=-1
min_child_samples=50, subsample=0.7, colsample_bytree=0.6
reg_alpha=0.1, reg_lambda=1.0
verbose=-1, seed=42
```

---

## 5. Feature Catalog (~80 features, `src/features.py`)

Forbidden (leaked / unavailable in test): `position`, `click_bool`, `booking_bool`, `gross_bookings_usd`, `random_bool`, `relevance`, `date_time`.

### 5.1 Temporal (`temporal_features`)
`month`, `hour`, `dayofweek`, `is_weekend_search`.

### 5.2 Missing-value flags (`missing_flags`)
`has_location_score2`, `has_visitor_history`, `has_query_affinity`, `has_distance`, `has_historical_price`.

### 5.3 Price (`price_features`)
`price_diff_from_hist`, `price_ratio_to_hist`, `price_per_night`, `total_cost`, `price_per_person`, `price_per_room`, `log_price`, `log_distance`.

### 5.4 Visitor match (`visitor_match_features`)
`star_diff = hist_star - prop_star`, `abs_star_diff`, `price_diff_from_visitor_hist`, `price_ratio_to_visitor_hist`.

### 5.5 Competitor aggregates (`competitor_features`)
Raw `comp{1..8}_rate / _inv / _rate_percent_diff` are **dropped** after aggregation into:
`comp_rate_sum`, `comp_rate_count`, `comp_cheaper_count`, `comp_more_expensive_count`, `comp_rate_advantage`, `comp_inv_count`, `comp_no_inv_count`, `comp_rate_pct_mean/min/max`.

### 5.6 Hotel quality interactions (`quality_features`)
`value_score = prop_starrating / log1p(price_usd)`, `star_review_product`, `location_total`, `is_domestic`, `starrating_is_zero`, `review_is_zero`.

### 5.7 Within-query / listwise (`listwise_features`) — **the ranking backbone**
Ranks (with `_norm` = rank / group_size):
`price_rank`, `starrating_rank`, `review_rank`, `location1_rank`, `location2_rank`, `location_total_rank`, `value_score_rank`, `price_per_star_rank`, `comp_advantage_rank`.

Deltas vs query stats:
`price_vs_mean/median/min/max`, `price_z_score`, `log_price_z_score`, `star_vs_mean`, `review_vs_mean`, `loc1_vs_mean`, `loc2_vs_mean`, `price_to_min_ratio`.

Percentiles & flags:
`price_percentile`, `star_percentile`, `review_percentile`, `is_cheapest`, `is_most_expensive`, `is_best_star`, `is_best_review`, `is_best_location1`, `n_best_flags` (sum of above), `quality_rank_avg`, `value_gap = quality_rank_avg - price_rank_norm`.

Query-level stats (constant within query but useful for the model to gauge the search context):
`query_hotel_count`, `query_price_std`, `query_star_mean`, `query_price_mean`.

### 5.8 Cross-feature interactions (`interaction_features`)
`price_per_star`, `price_per_night_per_star`, `star_x_brand`, `query_affinity_exp = exp(clip(srch_query_affinity_score, max=0))`, `distance_x_international`, `promo_x_cheap`, `is_last_minute (≤1d)`, `is_short_window (2–7d)`, `is_long_window (>30d)`, `is_family`, `total_guests`, `price_per_guest`, `is_discounted (price/hist < 0.9)`, `is_overpriced (price/hist > 1.2)`.

### 5.9 Hotel aggregates (`hotel_aggregates`) — **k-fold OOF on train, source-based on val/test**

**Single-entity target encodings** (Bayesian smoothing, `prior_weight` = α):
| group_col | target | output | α |
|---|---|---|---|
| prop_id | click_bool | `prop_click_rate` | 30 |
| prop_id | booking_bool | `prop_book_rate` | 30 |
| prop_id | relevance | `prop_rel_rate` | 30 |
| srch_destination_id | click_bool | `dest_click_rate` | 30 |
| srch_destination_id | booking_bool | `dest_book_rate` | 30 |
| prop_country_id | booking_bool | `country_book_rate` | 50 |
| site_id | booking_bool | `site_book_rate` | 50 |

**Cross-entity target encodings:**
| group | output | α |
|---|---|---|
| prop_id × srch_destination_id | `prop_dest_book_rate` | 10 |
| site_id × srch_destination_id | `site_dest_book_rate` | 15 |
| prop_id × site_id | `prop_site_book_rate` | 15 |
| visitor_country × prop_country | `cpair_book_rate` | 20 |
| site_id × prop_country | `site_country_book_rate` | 20 |

**Booking-given-click**: `(book_sum + 10*global_bgc) / (click_sum + 10)` per prop_id → `prop_book_given_click`, with side-effect feature `prop_click_count`.

**Entity counts**: `prop_count`, `dest_count`, `log_prop_count`.

**Property price stats**: `prop_mean_price`, `prop_std_price`, `price_vs_prop_mean`, `prop_price_zscore`.

**Property avg position (non-random only)**: `prop_avg_position` — captures Expedia's belief about a hotel's quality, derived only from non-random rows so position bias is the signal.

**Destination-relative**: `dest_mean_price/star/review`, `price_vs_dest_mean`, `star_vs_dest_mean`, `review_vs_dest_mean`.

### 5.10 Leak protection (critical)
- All target-derived features use **k-fold OOF by `srch_id`** during training (a search's rows all go to the same fold).
- Val/test get features computed from the **training source** only (separate `agg_source` argument).
- Asserted: `srch_id.groupby(fold).nunique() == 1` (no leakage).

### 5.11 IPW (Inverse Propensity Weighting)
- Propensity learned from `random_bool=1` rows: `propensity(position) = click_rate(position)`.
- Weights = `max(propensity) / propensity(position)`, clipped to [0.1, 10].
- Only applied to `random_bool=0` rows (random rows are already unbiased).
- Used as LightGBM `weight` parameter.

---

## 6. Project Layout

```
DataMiningAssignment2/
├── EDA_PLAN.md            # 4-notebook EDA roadmap
├── data/                  # train/test CSVs (~1.2GB each)
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_target_and_position.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_diagnostics.ipynb
├── src/
│   ├── config.py          # paths, NON_FEATURE_COLS
│   ├── data_loader.py     # load_train/test, make_target, split_val
│   ├── features.py        # 80+ engineered features
│   ├── evaluate.py        # NDCG@k
│   └── submission.py      # write submission CSV
├── run_baseline.py        # V3 single LambdaRank
├── run_v4.py              # V4 staged 8-model ensemble (current best, 0.42021)
├── run_v5.py              # V5 (script-only, not submitted)
├── models/                # (gitignored) saved models + JSON checkpoints
└── submissions/           # (gitignored) submission CSVs
```

`requirements.txt`: pandas≥2.0, numpy≥1.24, scikit-learn≥1.3, lightgbm≥4.0, matplotlib≥3.7, seaborn≥0.12.

---

## 7. Gap Analysis — How to Push Past 0.43

| Idea | Expected lift | Effort | Why |
|---|---|---|---|
| **Add XGBoost or CatBoost ranker** to ensemble | +0.003–0.008 | M | All 8 V4 members are LightGBM. Same algorithm → correlated errors. Real algorithm diversity uncorrelates errors and rank-blending pays off. |
| **Compute catalog stats on train+test** (counts, mean-price, std-price per prop_id and dest_id) | +0.002–0.005 | S | Test data exposes prop_id and dest_id; non-target stats can include it. Reduces 6% prop and 31% dest cold-start fall-through. |
| **Matrix factorization / SVD** on (visitor_country × prop_id) or (dest_id × prop_id) click-rate matrix → emit 8–16 latent dims as features | +0.002–0.005 | M | Provides a true collaborative signal. **Also satisfies the report requirement to use a recommender-systems technique** (the assignment explicitly asks for one). |
| **Stacking** with a small LightGBM ranker on OOF base-model predictions + a few raw features | +0.002–0.005 | M | Current rank-blend uses fixed NDCG-proportional weights; a learned blender does strictly better when models are diverse. |
| **Hard-negative reweighting** (run V4 to identify misranked booked queries, upweight them in retrain) | +0.002–0.003 | M | Notebook 4 already exposes the hard-negative analysis; turn the finding into a training-time weight. |
| **Adversarial-aware feature pruning** (drop high-AUC drift features) | +0.000–0.002 | S | AUC is 0.524 so drift is small, but srch_destination_id, srch_booking_window, orig_destination_distance, prop_location_score2 are the top drifters — try re-encoding or excluding raw IDs. |
| **More label-gain tuning** (V5) | +0.000–0.002 | S | Already tried in V5; minimal effect over V4. Lowest leverage. |

**Recommended V6 plan for 2-day window:**
1. Catalog stats from train+test (1–2 h work).
2. Add XGBoost ranker as 9th ensemble member (3–4 h).
3. SVD on (dest_id × prop_id) click matrix → 8 latent features (2–3 h, and **gives the report its required RecSys technique**).
4. Optional: retrain V4 ensemble with hard-negative upweighting if time permits.
5. Re-run the V4 pipeline with the new features + new member; submit.

---

## 8. Risk Register

- **30.7% of searches have no booking** → no booking gradient on those queries. Already known; LambdaRank handles it but it limits sample efficiency. Mitigated by `lambdarank_click3` member.
- **31% of test destinations are unseen** → `dest_*` target-encoded features fall back to global prior for 1/3 of the test rows. Adding train+test to count-based aggregates helps.
- **`prop_location_score2` correlates strongly with position** (Spearman −0.26 on non-random). It is a quality signal but also a position-encoded one — its missingness (22%) is informative on its own.
- **gross_bookings_usd / (price × LOS) ratio centers around 1.15, not 1.0** → price_usd is per-night but includes some tax/fee variance. The `total_cost` feature still works as a relative magnitude.
- **Visitor history covers only 5% of rows** → visitor_hist_* features are missing for 95% of rows; the `has_visitor_history` flag is more useful than the values themselves.
- **The notebooks use a local dev machine** (`/var/folders/ph/...` paths in tracebacks) → outputs were generated locally; the snapshot above reflects the last interactive run.

---

## 9. Open Items

- V5 script exists (`run_v5.py`) but no Kaggle submission yet — likely lower priority since it's the same features as V4.
- `models/v3_gbdt.txt` is gitignored; notebook 04 retrains on the fly if absent (slower).
- No saved V4 stage-result JSONs committed (also gitignored); to re-inspect V4 metrics, re-run `python run_v4.py` (the staged pipeline resumes from cached `models/v4/*.json`).
- Bias mitigation (Task 5 of assignment) is sketched in notebook 03 and 04 but no mitigation step has been implemented — needed for the report (15 of 100 points).
- Report itself (LNCS template, max 14 pages + process report 2 pages) not started.
