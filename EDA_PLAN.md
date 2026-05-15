# EDA Plan — Expedia Hotel Search Ranking (Complete)

## Context

Our V3 LambdaRank model achieves **0.417 local NDCG@5** (up from 0.38 Kaggle on V1). The competition leader is at **0.42**. Before iterating further, we need EDA that goes beyond "understanding columns" and into **ranking-specific diagnostics**: why the booked hotel wins inside a search, where the model fails in the top 5, and which segments or generalization cases break.

## Architecture: 4 Notebooks

| Notebook | Focus | Priority |
|----------|-------|----------|
| `02_target_and_position.ipynb` | Target, position bias, random_bool, clicked-vs-booked, no-positive queries | BUILD FIRST |
| `01_data_overview.ipynb` | Schema, missing values, distributions, train/test drift, cold-start, adversarial validation | SECOND |
| `03_feature_analysis.ipynb` | Correlations, interactions, within-query, segments, data-side fairness | THIRD |
| `04_model_diagnostics.ipynb` | Error analysis, top-5 diagnostics, winner-vs-loser, hard negatives, query difficulty, robustness, model-side fairness | FOURTH (requires trained model) |

**Sampling strategy:** Full data for summary stats and tables. 10% search-level sample (~500k rows) for plots and heavy computations. 1% for scatter plots. Always sample by `srch_id` to preserve within-query structure.

**Submission tracking table** (update after every Kaggle upload):

| Version | Local Split | Local NDCG@5 | Kaggle Public | Feature Set | Notes |
|---------|-------------|-------------|---------------|-------------|-------|
| V1 | random 10% | 0.468 | 0.38 | basic + leaked aggs | aggregate leakage |
| V3 | random 10% | 0.417 | TBD | k-fold TE + IPW + GBDT | honest score |

---

## Notebook 2: `02_target_and_position.ipynb` — Target Analysis & Position Bias (BUILD FIRST)

### 2.1 Target Distribution
- Click/booking/neither counts and percentages
- Verify booking implies click (do all booking_bool=1 rows also have click_bool=1?)
- Relevance label distribution (0 / 1 / 5)
- Clicks per search distribution, bookings per search distribution, % searches with zero clicks

**What this checks:** The class balance and label structure. If bookings are extremely rare, the model must learn from a tiny positive signal.

### 2.2 No-Positive Query Analysis
- % searches with no click and no booking (all-zero relevance)
- % searches with click but no booking
- % searches with at least one booking
- For all-zero searches: are they systematically different? (mean price, star, group size, random_bool distribution)
- LambdaRank note: groups with all-zero labels provide zero gradient for learning relative order. Quantify what fraction of training signal comes from positive queries only

**What this checks:** How much of the training data is actually informative for ranking. If 60% of searches have no positive signal, the model learns from a minority of the data.

### 2.3 Duplicate (srch_id, prop_id) Sanity Check
- Count duplicates of (srch_id, prop_id) in train and test
- Expected: zero. If nonzero, submission ranking or grouping logic can break

**What this checks:** Data integrity assumption that each property appears at most once per search.

### 2.4 Position Bias (Full Dataset)
- Click rate and booking rate by position (1-40) — line plot
- Mean relevance by position
- Position statistics table: at positions 1/5/10/20/30/40 show click rate, book rate, mean price, mean starrating

**What this checks:** How much position dominates user behavior. Users click what they see first — if position 1 has 10x the click rate of position 20, raw click data is heavily confounded.

### 2.5 Random vs Non-Random Analysis (THE critical section)
- Random vs non-random split: row counts and overall click/booking rates
- **Overall rate comparison:** Do random_bool=1 and random_bool=0 have different click/booking rates? Quantify the gap. This determines whether to train on all data, random-only, or all data with IPW reweighting
- **2-panel position bias plot:** Click rate by position for random_bool=1 vs random_bool=0. Same for booking rate. NOTE: For random_bool=1, click/book rate by position will still decay — users still see higher positions first, so there is pure exposure bias. What should be flat is hotel **quality** by position (star, review, location, price should not correlate with position in random data). For random_bool=0, click/book rate reflects BOTH exposure bias AND Expedia's quality-based ranking
- Position bias quantification from random data: fit propensity curve from random_bool=1 click rate by position — this isolates pure exposure bias for IPW computation
- **Leakage check:** Spearman correlation of position with quality features (star, review, location, price) — separate for random vs non-random. In random data these should be ~0. In non-random data they'll be strong because position encodes Expedia's ranking
- Hotel quality by position (4-panel): mean starrating/review/location/price by position, two lines per panel (random vs non-random). Random lines should be flat; non-random lines should show quality decay with position

**What this checks:** Distinguishes two sources of position-dependent click rates: (1) exposure bias (users see top positions first — present in both random and non-random) and (2) quality sorting (Expedia ranks better hotels higher — only in non-random). The random_bool=1 position curve isolates pure exposure bias for IPW.

### 2.6 Position as Feature — Decision Analysis
- Quick ablation: train LambdaRank with vs without position on 10% sample, compare NDCG@5
- Inverse propensity weights from random data position curve — show weight distribution
- Discuss debiased training: IPW vs training-on-random-only vs position-as-feature tradeoffs

**What this checks:** Whether including position helps or hurts generalization. Position isn't in test data, so any signal it provides must come through other correlated features.

### 2.7 Target vs Key Features (Unbiased — random_bool=1 only)
- Click/book rate by: prop_starrating, prop_review_score, promotion_flag, prop_brand_bool, price quintile
- Each split by random_bool for comparison

**What this checks:** True causal feature-target relationships, unconfounded by Expedia's ranking.

### 2.8 Clicked-but-not-Booked Analysis

Separate all rows into three groups:
- **Booked** (booking_bool=1) — relevance 5
- **Clicked-only** (click_bool=1, booking_bool=0) — relevance 1
- **Ignored** (click_bool=0, booking_bool=0) — relevance 0

For each group, compute mean values of: price_usd, prop_starrating, prop_review_score, prop_location_score1, prop_location_score2, promotion_flag, prop_brand_bool, orig_destination_distance.

Output comparison table: feature | booked_mean | clicked_mean | ignored_mean.

Since booking has 5x the relevance of click, features that separate booked from clicked-only are more valuable than features that separate clicked from ignored.

**What this checks:** Whether the features that attract clicks are the same ones that close bookings. If not, the model needs to prioritize booking-predictive features.

### 2.9 gross_bookings_usd Analysis (keep brief)
- Scatter vs price_usd * length_of_stay — confirms price semantics
- R-squared of the relationship

**What this checks:** Whether price_usd is per-night or total stay price.

---

## Notebook 1: `01_data_overview.ipynb` — Structure, Missing Values, Distributions, Outliers

### 1.1 Dataset Shape & Schema
- Full column inventory table: name, dtype, nunique, missing%, min/max/mean/median/std, column group
- Train vs Test schema comparison: shared columns, test-only missing columns (position, click_bool, booking_bool, gross_bookings_usd)

**What this checks:** Basic data shape.

### 1.2 Train/Test Temporal & Structural Checks
- **Temporal split check:** Plot date_time distributions for train AND test (overlapping histograms). Is the test set from a later period or randomly interleaved?
- **Test set random_bool distribution:** Does the test set have random_bool? What's its distribution?
- **Repeat searches:** Are there duplicate srch_ids or near-duplicate searches (same visitor_location_country_id + srch_destination_id + date range)?
- **Train/test group size drift:** Compare candidate hotels per srch_id — mean, median, p95, distribution overlay. If test has larger groups, top-5 ranking gets harder

**What this checks:** Whether temporal features and aggregates generalize. If test is from a later period, training aggregates may be stale.

### 1.3 Cold-Start & ID Overlap Analysis

Train/test overlap for:
- `prop_id` — what % of test properties never appeared in training?
- `srch_destination_id` — what % of test destinations are unseen?
- `prop_id x srch_destination_id` — cross-feature overlap (much lower than individual)
- `prop_id x site_id` — same property on different sites
- `prop_country_id x visitor_location_country_id` — country-pair overlap
- `site_id x srch_destination_id` — site-destination overlap

For test: report coverage % and unseen rate only (no target labels available in test).
For validation: compute click/book rate for entities seen vs unseen in the training fold.

**What this checks:** Where aggregate features (target encoding, prop_click_rate) have values vs. where they fall back to global priors. If 30% of test prop_ids are unseen, target-encoded features are useless for those rows.

### 1.4 Missing Value Analysis
- Missing value bar chart sorted descending
- Missing value co-occurrence matrix (clustermap) — do comp columns go missing together? Does visitor_hist_starrating always co-occur with visitor_hist_adr_usd?
- **Informative missingness test:** For each column with >1% missing, compute click/booking rate when present vs missing, with chi-squared p-values
- Competitor missingness breakdown: per competitor index (1-8) availability rate

**What this checks:** Whether missingness is random or informative. If click rate differs when a feature is present vs absent, the missingness pattern is a useful feature.

### 1.5 Missingness as Signal by Segment
For key sparse features (srch_query_affinity_score, orig_destination_distance, visitor_hist_starrating, comp1-8), compute missingness rate segmented by:
- site_id
- prop_country_id / visitor_location_country_id
- random_bool
- property popularity bucket (impression count)
- **Train vs test** — is missingness rate different in test?

**What this checks:** Whether missing data is systematically biased. If srch_query_affinity_score is only available for certain sites, `has_query_affinity` proxies for site identity.

### 1.6 Continuous Feature Distributions (keep focused)
- Histogram grid for key continuous features, log y-axis where skewed
- Price deep-dive: distribution with percentiles, price by starrating boxplot
- `srch_query_affinity_score` distribution with click/book density overlay

**What this checks:** Distribution shapes for feature engineering (log transform? binning? clipping?).

### 1.7 Categorical/Discrete Feature Distributions
- Bar charts: prop_starrating, prop_review_score, prop_brand_bool, promotion_flag, random_bool
- Search parameter distributions: length_of_stay, booking_window, adults, children, rooms

**What this checks:** Category frequencies and imbalances.

### 1.8 Cardinality & ID Analysis
- Hotels per search distribution — is group size constant (~25) or variable?
- Property frequency distribution (log-log) — popular vs rare hotels
- Top countries/sites/destinations by volume

**What this checks:** Group structure for LambdaRank and entity frequency for aggregate features.

### 1.9 Outlier Detection
- IQR-based outlier summary table for key continuous features
- Price outlier investigation: price vs starrating scatter colored by booking

**What this checks:** Whether extreme values need clipping.

### 1.10 Temporal Patterns
- Click/booking rate by hour of day, day of week, month
- Average price by month

**What this checks:** Seasonality for temporal features.

### 1.11 Adversarial Validation
Train a LightGBM classifier to distinguish train rows from test rows (label 0=train, 1=test).

Report:
- AUC (>0.5 means distributional shift exists)
- Top 10 features by importance (features that drift most)
- Drift by missingness patterns and categorical IDs

**What this checks:** Whether local validation will match Kaggle. Top-importance features are the ones most likely to cause generalization failures.

### 1.12 Price Semantics by Site/Country
- Mean/median price_usd by site_id and by prop_country_id
- gross_bookings_usd vs price_usd * length_of_stay segmented by top 5 sites

**What this checks:** Whether price is per-night or per-stay and whether it's consistent across markets.

---

## Notebook 3: `03_feature_analysis.ipynb` — Correlations, Interactions, Within-Query Analysis

### 3.1 Feature-Target Correlations
- Spearman correlation of every raw feature with click_bool and booking_bool (on random_bool=1 data for unbiased estimates)
- Sorted bar chart of absolute correlations
- Top 10 features: mean target by decile bin — shows relationship shape (linear? U-shaped?)

**What this checks:** Which raw features have the strongest signal, unconfounded by position bias.

### 3.2 Inter-Feature Correlation Matrix (keep focused)
- Focused heatmap: property features only (star, review, brand, location1, location2, hist_price, price) — ~10x10
- Search parameter correlation matrix — ~8x8

**What this checks:** Redundant features (correlated pairs >0.9 may be droppable).

### 3.3 Property Features Deep Dive
- prop_location_score1 and score2 vs click/book rate (binned line plots)
- prop_log_historical_price: what does 0 mean? click/book rate for zero vs non-zero
- Star rating x review score 2D heatmap with booking rate overlay
- Brand effect by star rating

**What this checks:** Non-linear relationships and interactions within property-quality features.

### 3.4 Price Analysis
- Price vs booking rate (50 quantile bins) — non-linear relationship shape
- Price rank within query vs booking rate — are cheapest options preferred?
- **Price x star interaction heatmap:** price_quintile x starrating, fill = booking rate
- Promotion flag effectiveness: controlled comparison within price bands

**What this checks:** How price interacts with quality. Price rank within query may matter more than absolute price.

### 3.5 Visitor History Features
- Users with vs without history: click/book rates
- Star rating match: booking rate by (visitor_hist_star - prop_star)
- Price match: booking rate by (price_usd / visitor_hist_adr_usd) bins

**What this checks:** Whether visitor preference matching predicts booking.

### 3.6 Competitor Features Analysis
- Per-competitor summary: availability rate, rate distribution (-1/0/1), mean pct_diff
- Competitive advantage: booking rate by n_competitors_more_expensive
- Inventory effect: booking rate by n_competitors_out_of_stock
- Rate percent diff vs booking rate (binned)

**What this checks:** Whether competitive pricing and inventory advantage drive bookings.

### 3.7 Competitor Feature Reliability
- Competitor availability rate by site_id, prop_country_id, price tier, prop_brand_bool
- Is competitor data more available for popular/commercial hotels?

**What this checks:** Whether `comp_rate_count` proxies for hotel popularity rather than actual competitive position.

### 3.8 Geographic & Cross-Country Patterns
- Top 20 property and visitor countries by volume with booking rate overlay
- **Domestic vs international:** is_domestic flag, click/book rate comparison
- orig_destination_distance vs booking rate (log-binned)
- Site-level analysis: click/book rate by site_id

**What this checks:** Geographic booking patterns.

### 3.9 Within-Query Variance Analysis (Most Important for LambdaRank)

LambdaRank ranks *within* a query — features constant within a query provide zero ranking signal as raw features.

- **Variance ratio table:** For each feature, within-query variance / total variance. Sort by ratio. Split into 3 tiers:
  - **Tier 1 (high within-query variance):** The ranking signal. e.g., price_usd, prop_starrating, prop_review_score, prop_location_score1/2
  - **Tier 2 (zero within-query variance):** Constant within query. Useless as raw features BUT useful as interaction terms. e.g., srch_length_of_stay, srch_booking_window, srch_adults_count
  - **Tier 3 (low but non-zero):** Partially varying. e.g., prop_country_id
- **Explicit interaction feature proposals** for each Tier 2 feature
- Within-query price spread distribution (std, range, IQR per query)
- Within-query star rating variety
- Within-query Spearman correlation of Tier 1 features with relevance — most direct measure of ranking signal

**What this checks:** Which features actually help rank within a single search.

### 3.10 Segmented Feature Importance
Compute feature-target Spearman correlations separately for:
- random_bool = 1 vs 0
- Domestic vs international
- Short stay (1-2) vs long stay (5+)
- Last-minute (booking_window < 3) vs long-lead (> 21)
- Solo/couple vs family (children > 0)
- Known properties (100+ impressions) vs cold-start

Output: heatmap rows=features, columns=segments, fill=Spearman with booking_bool.

**What this checks:** Whether different features matter for different segments. Informs interaction features.

### 3.11 Popularity Bias Analysis (Data Side)
- Booking rate by property impression count bucket (1-10, 10-50, 50-200, 200+)
- Click rate by property impression count bucket

**What this checks:** Whether frequent hotels dominate bookings. Relevant for required bias analysis in report.

### 3.12 Fairness & Bias Diagnosis (Data Side Only)

Identify representation/rate bias in the **data** for candidate dimensions:
- Family vs solo/couple: volume split, click/book rate comparison
- Domestic vs international: volume split, click/book rate comparison
- Branded vs independent hotels: volume, click/book rate
- Low-star (0-2) vs high-star (4-5): click/book rate
- Rare vs popular properties: volume, click/book rate

NOTE: Model-side fairness metrics (group NDCG@5, recall@5, mean booked rank) belong in Notebook 4, since they require model predictions.

**What this checks:** Systematic disadvantages for certain groups in the data. Required for report Task 5.

### 3.13 Interaction Effects & Creative Features
- Value score: prop_starrating / log1p(price_usd) — bin and plot booking rate
- Booking window segments (last-minute/short/medium/long): which features matter most?
- Family vs solo/couple: booking pattern differences
- Calendar interactions: booking_window x month, saturday_night x destination type

**What this checks:** Feature interactions to engineer as new columns.

### 3.14 Sanity-Check Baseline Rankings
Compare simple rule-based rankings on validation set:
- **Cheapest first** — rank by price_usd ascending
- **Highest stars first** — rank by prop_starrating descending
- **Highest review first** — rank by prop_review_score descending
- **Best location first** — rank by prop_location_score1 descending
- **Promotion first** — promoted hotels first, then by price
- **Best value first** — rank by prop_starrating / log1p(price_usd) descending
- **Expedia original position** — rank by position ascending (train only, non-random)

For each, compute NDCG@5.

**What this checks:** Context for model performance. Also reveals which single signals are strongest.

### 3.15 Summary & Feature Engineering Roadmap
- Actionable table: finding | evidence | proposed feature | priority | already implemented?
- Current pipeline gap analysis vs EDA findings
- Train vs test distribution comparison for top 15 features — flag distributional shifts

**What this checks:** Translates EDA into action items.

---

## Notebook 4: `04_model_diagnostics.ipynb` — Error Analysis & Ranking Diagnostics

*Requires a trained model. Load V3 model from `models/v3_gbdt.txt` and predict on validation set.*

### 4.1 Overall Model Performance
- Validation NDCG@5 (0.417)
- Booking Recall@1, Recall@3, Recall@5: what % of booked hotels are in top 1/3/5?
- Click Recall@5
- Mean booked-hotel rank
- % searches where booked hotel is rank 1
- Distribution of booked-hotel rank (histogram)
- Performance only on positive queries (searches with at least one click/booking)

**What this checks:** Model quality beyond a single NDCG number.

### 4.2 Winner-vs-Loser Within-Query Comparison (CRITICAL)

For each search with a booking, compare the **booked property** vs **non-booked properties**:

| Feature | Booked Mean | Non-Booked Mean | Booked Pctile Within Query | Effect Size |
|---------|-------------|-----------------|---------------------------|-------------|
| price_usd | | | | |
| prop_starrating | | | | |
| prop_review_score | | | | |
| prop_location_score1 | | | | |
| prop_location_score2 | | | | |
| promotion_flag | | | | |
| prop_brand_bool | | | | |
| comp_rate_advantage | | | | |
| orig_destination_distance | | | | |

Also answer:
- Was booked hotel cheaper than query median? (% of time)
- Higher star rating than query mean?
- On promotion?
- Better competitor pricing?
- Closer match to visitor history?

**What this checks:** What makes the winner win. If booked hotels are almost always cheaper and higher-star, price_rank and star_vs_mean should be top features.

### 4.3 Hard-Negative Analysis (NEW — Most Actionable)

For each search with a booking, compare the **booked hotel** vs the **model's top-ranked non-booked hotel** (the "hard negative" — the one the model incorrectly preferred):

| Feature | True Booked | Model Top Wrong | Difference | Pattern |
|---------|-------------|-----------------|------------|---------|
| price_usd | | | | |
| prop_starrating | | | | |
| prop_review_score | | | | |
| prop_location_score1 | | | | |
| prop_location_score2 | | | | |
| promotion_flag | | | | |
| comp_rate_advantage | | | | |
| prop_click_rate | | | | |
| prop_book_rate | | | | |

This directly tells you which features the model is **over-weighting** (hard negative scores higher on those) and which it's **under-weighting** (booked hotel is better but model doesn't care).

**What this checks:** The most actionable error signal. If the model consistently picks a cheaper but lower-quality hotel over the booked one, it's over-weighting price.

### 4.4 "Easy Win Missed" Analysis

For failed searches (booked hotel ranked outside top 5), identify cases where the booked hotel was obviously superior:
- Booked hotel cheaper than median AND higher star than median
- Booked hotel promoted AND good location score
- Booked hotel strong on 3+ quality dimensions

If the model misses these "easy" cases, feature interactions or ranks are fundamentally broken.

**What this checks:** Whether failures are due to genuinely hard queries or systematic model weaknesses.

### 4.5 Model Error Analysis — Where Does the Model Fail?

For searches where model ranks booked hotel **outside top 5**:
- How many such searches? (% of searches with bookings)
- Booked hotel rank distribution for failures (6-10 vs 10-20 vs 20+)
- Compare failed vs successful searches by:
  - random_bool distribution
  - destination frequency (popular vs rare)
  - site_id
  - Family vs solo/couple
  - Domestic vs international
  - Price spread within query
  - Whether booked hotel is cold-start
  - Number of candidates in search

**What this checks:** Systematic error patterns for targeted feature engineering.

### 4.6 Query Difficulty Analysis

Per-search difficulty metrics:
- Number of candidate hotels
- Price range / IQR within query
- Star-rating variety (distinct values)
- Number of promoted/branded hotels
- Whether query has obvious dominant hotel
- Entropy/diversity of candidates

Then: NDCG@5 by difficulty bucket.

**What this checks:** Which searches are inherently easy vs hard.

### 4.7 NDCG@5 by Segment

Compute NDCG@5 separately for:
- random_bool = 1 vs 0
- Domestic vs international
- Short vs long stay
- Last-minute vs long booking window
- Solo/couple vs family
- Top 5 sites vs rest
- Cold-start properties present vs all-known
- Small query groups (<15) vs large (>25)

Output: segment, n_searches, NDCG@5, mean_booked_rank.

**What this checks:** Where the model is strong vs weak. Weak segments are improvement opportunities.

### 4.8 Popularity Bias Analysis (Model Side)

Use **within-query rank** and **within-query percentile** (NOT raw predicted scores, which are not comparable across queries):
- Mean within-query rank for rare vs common properties
- Mean within-query percentile for rare vs common properties
- Exposure@1 / Exposure@3 / Exposure@5 by popularity bucket (how often does a hotel from this bucket appear in top K?)
- Booked-property rank by popularity bucket

**What this checks:** Whether the model systematically pushes rare properties down. Uses within-query metrics because LambdaRank scores are only meaningful within the same query.

### 4.9 Fairness Metrics (Model Side)

For each bias dimension identified in Notebook 3 (3.12):
- Group-level NDCG@5
- Booking Recall@5
- Mean booked-hotel within-query rank

Compare across groups and flag significant disparities.

**What this checks:** Whether the model underserves certain groups. Required for report Task 5 (bias detection, mitigation, evaluation).

### 4.10 Feature Importance Analysis (keep lean)
- LightGBM gain-based importance (top 30)
- LightGBM split-based importance (top 30) — compare with gain

Skip: SHAP (too slow), partial dependence (too many plots). Can add later if time permits.

**What this checks:** Which features the model actually uses vs which have high correlation.

### 4.11 Robustness Across Validation Splits

To assess whether 0.417 NDCG@5 is stable:
- **5 random seeds:** 90/10 split with seeds 42/123/456/789/2024 — report mean and std of NDCG@5
- **Temporal split:** Last N% of searches by date as validation
- **Destination-grouped split:** Hold out searches for 10% of destinations
- **Cold-property split:** Hold out searches with properties seen <5 times
- **random_bool stratified:** NDCG@5 on random vs non-random validation queries separately

**What this checks:** Whether the model generalizes or is overfit to a particular split.

### 4.12 Prediction Score Analysis (keep brief)
- Distribution of predicted scores by true relevance (0/1/5) — boxplot
- Score separation: can the model distinguish booked from clicked from ignored?

**What this checks:** Whether the model produces well-separated scores.

---

## Verification Checklist

After creating/updating all 4 notebooks:
1. Run each notebook end-to-end
2. Confirm all plots render, no errors
3. Read Key Findings to extract actionable insights
4. Use findings to update `src/features.py` and retrain
5. Cross-reference with report outline (bias section, feature engineering, model selection)
