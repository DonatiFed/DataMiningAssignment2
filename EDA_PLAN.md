# EDA Plan — Expedia Hotel Search Ranking

## Context

Our baseline LambdaRank model scored **0.38 on Kaggle** vs 0.468 local NDCG@5 — the gap is likely from aggregate feature leakage and weak feature engineering. Before improving the pipeline, we need a thorough understanding of every column, every distribution, every correlation, and every bias in the data. This EDA will directly inform what features to build, what to fix, and what to drop.

## Architecture: 3 Notebooks

**Execution order: Notebook 2 first** — position bias and random_bool decisions shape everything downstream (training data selection, propensity weighting, feature inclusion). Then Notebook 1 for the full data picture, then Notebook 3 for feature-target analysis.

**Sampling strategy:** Full data for summary stats and tables. 10% search-level sample (~500k rows) for plots and heavy computations. 1% for scatter plots. Always sample by `srch_id` to preserve within-query structure.

Each notebook starts with a shared setup cell (imports, display settings, data loading).

---

## Notebook 2: `02_target_and_position.ipynb` — Target Analysis & Position Bias (BUILD FIRST)

### 2.1 Target Distribution
- Click/booking/neither counts and percentages
- Verify booking implies click
- Relevance label distribution (0/1/5)
- Clicks per search distribution, bookings per search distribution, % searches with zero clicks

### 2.2 Position Bias (Full Dataset)
- Click rate and booking rate by position (1-40) — line plot
- Mean relevance by position
- Position statistics table: at positions 1/5/10/20/30/40 show click rate, book rate, mean price, mean starrating

### 2.3 Random vs Non-Random Analysis (THE critical section)
- Random vs non-random split counts and rates
- **Overall rate comparison:** Do random_bool=1 and random_bool=0 have different click/booking rates overall? If so, the two subsets represent fundamentally different user experiences. This determines whether training on all data, random-only, or all data with higher weight on random rows is best. Quantify the difference and discuss implications.
- **2-panel position bias plot:** click rate by position for random_bool=1 (should be ~flat) vs random_bool=0 (steep decay). Same for booking rate
- Position bias quantification from random data: fit propensity curve
- **Leakage check:** Spearman correlation of position with quality features (star, review, location, price) — separate for random vs non-random. In non-random data, position encodes Expedia's ranking algorithm
- Hotel quality by position (4-panel): mean starrating/review/location/price by position, two lines per panel (random vs non-random)

### 2.4 Position as Feature — Decision Analysis
- Quick ablation: train LambdaRank with vs without position on 10% sample, compare NDCG@5
- Inverse propensity weights from random data position curve — show weight distribution and discuss debiased training

### 2.5 Target vs Key Features (Unbiased — random_bool=1 only)
- Click/book rate by: prop_starrating, prop_review_score, promotion_flag, prop_brand_bool, price quintile
- Each split by random_bool for comparison

### 2.6 gross_bookings_usd Analysis
- Distribution for booked rows
- Scatter vs price_usd * length_of_stay — confirms price semantics
- R-squared of the relationship

---

## Notebook 1: `01_data_overview.ipynb` — Structure, Missing Values, Distributions, Outliers

### 1.1 Dataset Shape & Schema
- Full column inventory table: name, dtype, nunique, missing%, min/max/mean/median/std, column group
- Train vs Test schema comparison: shared columns distribution shift check, test-only missing columns

### 1.2 Train/Test Temporal & Structural Checks
- **Temporal split check:** Plot date_time distributions for train AND test (overlapping histograms or KDE). Is the test set from a later time period or randomly interleaved? This determines whether temporal features and aggregate stats will generalize or suffer from temporal drift.
- **Test set random_bool distribution:** The test set doesn't have position, but does it have random_bool? Check its distribution. If test is all non-random (or all random), that changes modeling strategy.
- **Train/test prop_id overlap:** What % of test properties never appeared in training? (critical for aggregate features that rely on train prop_id stats)
- **Repeat searches:** Are there duplicate srch_ids or near-duplicate searches (same visitor_location_country_id + same srch_destination_id + same date range, different srch_id)? Could enable session-level features.

### 1.3 Missing Value Analysis
- Missing value bar chart sorted descending
- Missing value co-occurrence matrix (clustermap of missingness correlation) — do comp columns go missing together? Does visitor_hist_starrating always co-occur with visitor_hist_adr_usd?
- **Informative missingness test:** For each column with >1% missing, compute click rate and booking rate when present vs missing, with chi-squared p-values. This answers: should we create `is_missing_X` flags?
- Competitor missingness breakdown: per competitor index (1-8) availability rate, plus `n_comp_available` distribution and its correlation with booking

### 1.4 Continuous Feature Distributions
- Histogram grid (4x4) for all continuous features, log y-axis where skewed
- Price deep-dive (3 panels): distribution with percentiles marked, price by starrating boxplot, extreme outlier analysis (>99th pct)
- `srch_query_affinity_score` distribution (6.5% present) with click/book density overlay
- `orig_destination_distance` distribution (log x-axis)

### 1.5 Categorical/Discrete Feature Distributions
- Bar charts: prop_starrating, prop_review_score, prop_brand_bool, promotion_flag, random_bool, srch_saturday_night_bool
- Search parameter distributions: length_of_stay, booking_window, adults, children, rooms

### 1.6 Cardinality & ID Analysis
- Hotels per search distribution — is group size constant (~25) or variable?
- Property frequency distribution (log-log) — popular vs rare hotels
- Top countries/sites/destinations by volume

### 1.7 Outlier Detection
- IQR-based outlier summary table for all continuous features
- Price outlier investigation: price vs starrating scatter colored by booking, price vs length_of_stay, price_per_night distribution
- gross_bookings_usd vs price_usd * length_of_stay scatter — confirms whether price is per-night or total

### 1.8 Temporal Patterns
- Searches per day time series with 7-day rolling average
- Click/booking rate by hour of day, day of week, month
- Average price by month

---

## Notebook 3: `03_feature_analysis.ipynb` — Correlations, Interactions, Within-Query Analysis

### 3.1 Feature-Target Correlations (All Features)
- Spearman correlation of every feature with click_bool and booking_bool (on random_bool=1 data for unbiased estimates)
- Sorted bar chart of absolute correlations
- Point-biserial correlation for binary features
- Top 10 features: mean target by decile bin — shows relationship shape (linear? U-shaped?)

### 3.2 Inter-Feature Correlation Matrix
- Full correlation heatmap (~15x15 core features, excluding sparse comp columns)
- Focused heatmap: property features only (star, review, brand, location1, location2, hist_price, price)
- Search parameter correlation matrix

### 3.3 Property Features Deep Dive
- prop_location_score1 and score2 vs click/book rate (binned line plots)
- prop_log_historical_price investigation: what does 0 mean? click/book rate for zero vs non-zero
- Star rating x review score 2D heatmap with booking rate overlay
- Brand effect by star rating — does branding matter more for lower-star hotels?

### 3.4 Price Analysis
- Price vs booking rate (50 quantile bins) — non-linear relationship shape
- Price per night by star rating boxplot — identify anomalous combinations
- Price rank within query vs booking rate — are cheapest options preferred?
- **Price x star interaction heatmap:** price_quintile x starrating, fill = booking rate. Does a cheap 4-star massively outperform an expensive 3-star?
- Promotion flag effectiveness: controlled comparison within price bands

### 3.5 Visitor History Features
- Users with vs without history: click/book rates, site distribution
- Star rating match: booking rate by (visitor_hist_star - prop_star)
- Price match: booking rate by (price_usd / visitor_hist_adr_usd) bins
- Joint star+price match 2D heatmap

### 3.6 Competitor Features Analysis
- Per-competitor summary: availability rate, rate distribution (-1/0/1), mean pct_diff
- Competitive advantage: booking rate by n_competitors_more_expensive
- Inventory effect: booking rate by n_competitors_out_of_stock
- Rate percent diff vs booking rate (binned)
- Are all 8 competitors equally important? Per-competitor click/book rates at each rate value

### 3.7 Geographic & Cross-Country Patterns
- Top 20 property and visitor countries by volume with booking rate overlay
- **Domestic vs international:** is_domestic flag, click/book rate comparison
- orig_destination_distance vs booking rate (log-binned)
- Site-level analysis: click/book rate by site_id

### 3.8 Within-Query Variance Analysis (EXPANDED — Most Important for LambdaRank)

This is the most important section for the model. LambdaRank ranks *within* a query — it can only use features that vary across hotels in the same search. Features constant within a query (like srch_length_of_stay) contribute zero ranking signal as raw features.

- **Variance ratio table:** For each feature, compute within-query variance / total variance. Sort by ratio. Split into 3 tiers:
  - **Tier 1 (high within-query variance):** These ARE the ranking signal. e.g., price_usd, prop_starrating, prop_review_score, prop_location_score1/2
  - **Tier 2 (zero within-query variance):** Constant within query. Useless as raw features BUT become useful as **interaction terms**. Flag explicitly: e.g., srch_length_of_stay → useful as price_per_night = price_usd / srch_length_of_stay; srch_booking_window → useful for segmenting price sensitivity; srch_adults_count → useful for per-person cost
  - **Tier 3 (low but non-zero):** Partially varying. e.g., prop_country_id might vary within a query if cross-border destinations
- **Explicit interaction feature proposals** for each Tier 2 feature: what Tier 1 feature should it be combined with, and why?
- Within-query price spread distribution (std, range, IQR per query)
- Within-query star rating variety (number of distinct values per query)
- Within-query correlation: for Tier 1 features, compute the average within-query Spearman correlation with relevance. This is the most direct measure of "does this feature help rank within a query?"

### 3.9 Interaction Effects & Creative Features
- Value score: prop_starrating / log1p(price_usd) — bin and plot booking rate
- Booking window segments (last-minute/short/medium/long): which features matter most in each?
- Family vs solo/couple: booking pattern differences
- Feature-target correlation heatmap by booking_window segment — do correlations change?

### 3.10 Summary & Feature Engineering Roadmap
- Actionable table: finding → proposed feature → priority → already implemented?
- Current pipeline gap analysis vs EDA findings
- Train vs test distribution comparison for top 15 features (overlapping histograms) — flag distributional shifts

---

## Verification

After creating all 3 notebooks:
1. Run each notebook end-to-end (`Kernel → Restart & Run All`)
2. Confirm all plots render, no errors
3. Read the Key Findings cells to extract actionable insights for feature engineering
4. Use findings to update `src/features.py` and retrain
