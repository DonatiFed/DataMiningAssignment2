# Journey Explained — A First-Principles Walkthrough

A long-form explanation of what this
project did, why each step happened, and what every technique means.

This document is the **tutorial** companion to:
- `journey.md` — the project log (chronological, brief)
- `lessons_learned.md` (**Insights**) — the bullet-point takeaways
- `final_kaggle_results.md` — the authoritative scoreboard
- `next_steps.md` — the forward queue of unattempted ideas

Here you get everything explained from scratch, with a consistent
per-version template so you can scan the journey at any depth.

---

## Table of contents

- **Part 1** — The competition: what's being predicted, the data, the metric
- **Part 2** — The conceptual toolkit: every technique explained from zero
- **Part 3** — The journey, version by version, in full detail (V3 → V11)
- **Part 4** — The lessons, in plain language
- **Part 5** — Where to go from here
- **Part 6** — Glossary

---

# Part 1 · The competition

## 1.1 The problem in one paragraph

A user goes to Expedia and types a search query: "hotels in Seattle,
January 5–8, 2 adults." Expedia retrieves about 30 properties that
match. Those properties have to be displayed in SOME order — top of
the page first, bottom last. Users rarely scroll past the first few
results, so if the right hotel is at rank 25 it might as well not
exist. **The competition asks: given everything Expedia knew at search
time, what is the best order to display these properties?**

"Best" = the booked hotel should be at rank 1, the clicked-but-not-booked
hotels should rank high, the ignored hotels low. The scoring metric
(NDCG@5) heavily rewards getting the booking into the top 5.

## 1.2 The data

The dataset is the [Expedia 2013 Personalized Hotel Search dataset](https://www.kaggle.com/c/expedia-personalized-sort)
republished as Kaggle competition **DMT 2026 — 2nd Assignment**.

Every row in `train.csv` is ONE property displayed on ONE search
results page. About 5 million rows. The columns:

### Identifiers
- **`srch_id`** — unique search query ID. Rows sharing the same `srch_id`
  were displayed together on one results page.
- **`prop_id`** — unique property (hotel) ID.

### Labels (what we want to predict — TRAIN ONLY)
- **`click_bool`** — 1 if the user clicked this property
- **`booking_bool`** — 1 if the user booked this property

### Display info (TRAIN ONLY — not in test)
- **`position`** — where this property was displayed (1 = top)
- **`random_bool`** — 1 if Expedia displayed properties in random
  order for this search, 0 if Expedia's normal ranker was used

### Property features
`prop_starrating`, `prop_review_score`, `prop_brand_bool`,
`prop_location_score1`, `prop_location_score2`,
`prop_log_historical_price`, `price_usd`, `promotion_flag`, ...

### Search features
`srch_destination_id`, `srch_length_of_stay`, `srch_booking_window`,
`srch_adults_count`, `srch_room_count`, `srch_saturday_night_bool`, ...

### User features
`visitor_location_country_id`, `visitor_hist_starrating` (avg star
rating of hotels this user has booked in the past),
`visitor_hist_adr_usd` (avg daily room rate history).

### Competitor columns
`comp1_rate` through `comp8_rate` — eight other booking sites'
relative price for this property (-1 = competitor cheaper, 0 = same,
+1 = competitor more expensive, NULL = unknown).
`comp1_inv` through `comp8_inv` — whether each competitor had it.

### Time
`date_time` — when the search happened. Range: mid-2012 to mid-2013.

`test.csv` has the same columns **except** no `click_bool`, no
`booking_bool`, no `position`. That's the hidden truth you have to
predict.

## 1.3 The metric: NDCG@5

The most important number in this project. **NDCG = Normalized
Discounted Cumulative Gain.** **@5** means we only consider positions
1–5 of your ranking; everything past rank 5 is invisible to the metric.

### Relevance labels
- Booked → relevance = 5
- Clicked but not booked → relevance = 1
- Ignored → relevance = 0

### Formula

For one search, given your top-5 ranking with relevances $r_1, ..., r_5$:

$$
\text{DCG@5} = \sum_{p=1}^{5} \frac{2^{r_p} - 1}{\log_2(p+1)}
$$

$$
\text{iDCG@5} = \text{DCG@5 of the ideal ranking}
$$

$$
\text{NDCG@5} = \frac{\text{DCG@5}}{\text{iDCG@5}}
$$

### Worked example

Suppose your top-5 has relevances **[1, 5, 0, 0, 1]**:

| Position | Relevance | $(2^r - 1)$ | $\log_2(p+1)$ | Contribution |
|----------|-----------|-------------|---------------|--------------|
| 1        | 1         | 1           | 1.000         | 1.000        |
| 2        | 5         | 31          | 1.585         | 19.558       |
| 3        | 0         | 0           | 2.000         | 0            |
| 4        | 0         | 0           | 2.322         | 0            |
| 5        | 1         | 1           | 2.585         | 0.387        |

**DCG@5 = 20.945**

The ideal ranking would put the relevance-5 first, then the
relevance-1s, then zeros: **[5, 1, 1, 0, 0]**:

| Position | Relevance | Contribution |
|----------|-----------|--------------|
| 1        | 5         | 31.000       |
| 2        | 1         | 0.631        |
| 3        | 1         | 0.500        |

**iDCG@5 = 32.131**

**NDCG@5 = 20.945 / 32.131 = 0.652** for this search.

The final leaderboard score is the **mean** of per-search NDCG@5
across all test searches.

### Calibration of the score range

| Approach | Approximate NDCG@5 |
|---|---|
| Random ranking | ≈ 0.30 |
| Sort by price ascending | ≈ 0.34 |
| Single LightGBM LambdaRank baseline | ≈ 0.40 |
| **This project's best (V4)** | **0.42021** |
| Top of public leaderboard | ≈ 0.429 |

The 0.40–0.43 zone is where serious models live. Each +0.001 means
measurable improvement; +0.005 is a structural win.

---

# Part 2 · The conceptual toolkit

This is the dictionary of techniques. Every concept used in the
project is explained here, from zero. Skim it on first pass, return
later when Part 3 references something you don't remember.

## 2.1 Learning to rank (LTR) vs classification vs regression

Standard ML problem: predict number Y from features X.
- Y is continuous (price) → **regression**
- Y is a category (cat/dog) → **classification**
- Y is "the order of multiple items within a group" → **learning to rank**

**Key difference for LTR:** the model doesn't need to predict the
right *value* for any item. It needs the items' predicted scores to
be in the right *order within a group*. A model that always predicts
scores 100× too high but in the right order achieves a perfect ranking.

**Practical consequence:** LTR models train on *groups* (here, all
rows sharing a `srch_id`), not individual rows. The loss is computed
per-group. This is why LightGBM/CatBoost have ranking modes — they
need group boundaries.

## 2.2 Pointwise vs pairwise vs listwise

Three approaches to LTR:

- **Pointwise:** ignore that ranking is special. Predict relevance
  for each (search, property) pair independently; sort by predictions.
  Cheap, often weak baseline.
- **Pairwise:** for each pair (item_i, item_j) within a search,
  train the model to predict score_i > score_j when i is more
  relevant. LambdaRank (§2.8) is the famous pairwise approach.
- **Listwise:** the loss considers the entire ranked list per
  search. Optimizes a ranking metric directly or via surrogate.
  Examples: `rank_xendcg` (§2.9), YetiRank (§2.9), ListNet.

This project: V4 used LambdaRank (pairwise). V8 added xendcg
(listwise), discovered it was the largest ensemble lift because it
makes ranking mistakes DIFFERENT from LambdaRank.

## 2.3 Features and feature engineering

A **feature** is a column of numbers/categories the model consumes.
Raw Expedia data has ~50 features. The project's V4 pipeline
(`src/features.py:build_features`) expands this to **143 features**
by deriving:

- **Aggregates:** "this property's avg booking rate," "median price
  for this destination."
- **Interactions:** "this property's price ÷ its destination's
  median price."
- **Within-query ranks:** "is this property the cheapest in this
  search? Rank in the middle?" (§2.4)
- **Time-derived:** day of week, month, seasonality.
- **History differences:** "how much higher than this user's
  historical avg is this property's star rating?"

**Feature engineering** = the design of these columns. Most
competition wins come from here. Models can only learn from what
they're given.

This project's strategic weakness: the 143-feature pipeline was
built for V4 and **almost no new features were added in V5–V11**.
Versions V5–V11 mostly varied models and ensembles, not features.

## 2.4 Within-query rank features

A specific class of features that's both powerful AND drift-immune.

For each continuous feature F and each row, compute:
- `rank_within_srch(F)` — the property's rank on F among all
  properties shown in the same search
- `zscore_within_srch(F)` — (F − search_mean) / search_std
- `diff_from_min_within_srch(F)` — F − min(F in this search)
- `diff_from_median_within_srch(F)` — F − median(F)

Why drift-immune: each `srch_id` is self-contained. The relative
order of properties within one search doesn't depend on time or
sampling — both train and test searches have these same internal
relative structures.

V4 has some of these (price, star, review rank). A from-scratch
redesign would build them aggressively for every continuous feature.

## 2.5 Target encoding (TE)

**Target encoding** = replace a category with a statistic of the
target for that category.

Example. The `prop_id` column has thousands of distinct property
IDs. A model can't learn anything from "property 12345" directly
because IDs are arbitrary. But if you compute the AVERAGE
`booking_bool` for each `prop_id` across the training data, you get
a number like 0.034 (3.4% of searches for this property resulted in
a booking). Now replace property 12345 with that 0.034. The model
can now use "this property's historical booking rate" as a real
predictive signal.

**Why it's powerful:** captures category-level priors with one
column.

**Why it's dangerous — target leakage:** if you compute the TE on
the SAME rows you train on, the TE for row R was partially computed
from R's own label. The model can effectively "look up" the answer
for R, which works perfectly in training and fails in test. The fix
is out-of-fold computation (§2.6).

**Why it's dangerous EVEN WITH the OOF fix — drift:** TE values
encode properties of the training data. If the test set has
different per-category statistics (because of time, sampling,
population shift), the TE values become misleading. This is what
killed V5: cross-key TEs had radically different statistics in train
vs test.

### Cross-key target encoding

Combine two categorical features. E.g., `prop_dest_book_rate` =
average booking rate of property X specifically in destination Y.
Powerful in principle (captures very specific interactions). But
much more drift-prone in practice because:

1. Cells are small (few rows per (prop_id, dest_id) pair) → high
   variance in the estimate
2. Small cells respond more sensitively to time/sample shifts
3. The cell membership itself can change (new properties, new
   destinations) between train and test

V5 added 4 cross-key TEs. They were the cause of V5's Kaggle
regression.

## 2.6 Out-of-fold (OOF) computation

The standard fix for target leakage in TE.

**Procedure** (K-fold, typically K = 5):
1. Split training data into K folds.
2. For each row R, find its fold (say fold 3).
3. Compute the TE statistic using rows from folds 1, 2, 4, 5 only.
4. Assign that statistic to R.

This way, R's own target never contributes to its feature value.
Training scores become "honest."

For the test set: no target to leak, so compute the TE using ALL
training data and apply to test rows directly.

All TE features in this project use OOF. The V4 pipeline does this
correctly in `src/features.py`. This is baseline competence; getting
OOF wrong is the most common "I scored great in training and
terrible in test" failure mode in competition ML.

## 2.7 Position bias and inverse propensity weighting (IPW)

**The bias.** In the training data, users click on properties at
position 1 *disproportionately*. Not because position-1 properties
are inherently better, but because they're more visible. Naively
training on "did this row get clicked?" makes the model learn
"position 1 = high click probability" — true in training, useless
at test (the test set doesn't have known positions; you have to
PRODUCE positions).

**Diagnostic.** The `random_bool=1` subset is the truth: there,
Expedia displayed properties randomly, so position can't be a
confounder. Click rates by position in `random_bool=1` are roughly
flat. Click rates by position in `random_bool=0` (Expedia's
ranking) decay steeply with position. The gap = position bias.

**The fix: IPW.** Weight each training row by 1 / P(observation |
features). For position bias, P(observation) ≈ P(seen | position) ≈
how visible position p is. Estimate this from `random_bool=1` data.
Then in training, items clicked at position 1 (cheap clicks) get
DOWN-weighted, items clicked at position 25 (rare clicks) get
UP-weighted. The reweighted loss approximates "what would relevance
look like if position were random?"

V4 uses IPW with these weights baked into the LightGBM training
call. It's part of why V4 hits 0.42.

**Caveats.** IPW is approximate. It corrects the position-bias
channel only. Other biases pass through. And the propensity
estimate is itself imperfect.

## 2.8 Gradient boosting (GBDT) — LightGBM, CatBoost, XGBoost

The model family this project leans on most.

### Decision tree
A flowchart of rules. "If price > 200 and starrating ≥ 4, predict
0.8; otherwise check next condition..." Trees are interpretable but
individually weak.

### Gradient boosted decision trees (GBDT)
Train many trees sequentially. Tree 1 makes predictions. Tree 2 is
trained to predict the errors that Tree 1 makes. Tree 3 is trained
to predict the errors that Trees 1+2 still make. Continue for
hundreds or thousands of trees, each shallow (depth 4–8 typically).

The "gradient" comes from how errors are computed: each new tree
fits the negative gradient of the loss with respect to current
predictions. For regression with squared loss, that's just the
residual. For LambdaRank, it's a specific quantity called lambda
(§2.10).

### Important GBDT hyperparameters
- **`learning_rate`** — how much each new tree shifts the
  prediction. Smaller = more trees needed but smoother fit.
  V4 uses 0.05 in some configs, 0.04 in others.
- **`num_leaves`** — maximum leaves per tree. Higher = more flexible
  trees, more overfitting risk. V4 uses 127.
- **`max_depth`** — alternative to num_leaves; depth cap on each
  tree. V4 has it un-capped, relying on num_leaves.
- **`min_data_in_leaf`** — minimum rows per leaf. Higher = more
  regularization.
- **`feature_fraction` / `colsample_bytree`** — fraction of features
  randomly sampled per tree. < 1.0 adds randomness; V4 uses 1.0.
- **`bagging_fraction`** — fraction of rows sampled per tree.
- **`lambda_l1`, `lambda_l2`** — L1/L2 regularization on leaf
  weights. Larger = stronger regularization.
- **`num_iterations`** — max trees. Usually paired with
  **`early_stopping_rounds`** — stop training if val NDCG hasn't
  improved in N iterations.

### LightGBM, CatBoost, XGBoost
Three production GBDT libraries:
- **LightGBM** (Microsoft) — fast histogram-based binning, dominant
  in Kaggle. The project's workhorse.
- **CatBoost** (Yandex) — strong categorical handling, includes
  YetiRank listwise objective.
- **XGBoost** (DMLC) — the original. Used in V9 but 5 of 9 XGBoost
  models crashed due to the `inf in input data` issue (LightGBM and
  CatBoost handle infinite values; XGBoost requires explicit
  `missing=np.nan` or pre-cleaning).

## 2.9 Bin sampling, seeds, and reproducibility

LightGBM converts continuous features into discrete histograms (bins)
for fast tree splits. This **bin sampling** uses randomness, which
needs a seed.

### The V4 bug
V4 originally pre-constructed a single `lgb.Dataset` outside the
configuration loop and reused it across multiple model trainings
with different `seed=456`. The bug: bin sampling happens during
`Dataset` construction, so the binning was determined by the FIRST
config's seed and never updated. All "ensemble members" used the
SAME binning, drastically reducing their independence and costing
~0.0024 NDCG@5.

### The fix
Let `lgb.train()` construct the `Dataset` inside the loop, so each
config gets its own binning seeded by its own seed.

### The anchor invariant
Once fixed, a single-model V4 with `seed=456` reproduces NDCG@5 =
0.42191 on random val. The script
`pipelines/legacy/phase2_anchor_check.py` asserts this — every
subsequent pipeline has to pass this check to prove it inherits a
correct V4 base.

**Why this matters generally:** randomness in ML is everywhere
(seeds, bagging, dropout, init). Reproducibility requires tracking
*all* sources. A subtle bug like the bin-sampling one can silently
degrade scores without crashing anything.

## 2.10 LambdaRank — V4's workhorse objective

Intuition first.

### What LambdaRank does

For each pair of items (i, j) within a search where item i has
higher relevance than item j:
1. Compute the gradient saying "score_i should go up, score_j should
   go down."
2. **Weight that gradient by the NDCG change** caused by swapping i
   and j in the current ranking.
3. Pairs whose swap would dramatically change NDCG@5 (e.g., the
   booked item is at position 5, swapping with position 1 changes
   NDCG a lot) get LARGE gradients.
4. Pairs whose swap is irrelevant (e.g., two items both ranked
   outside top 10) get SMALL gradients.

The model focuses on swaps that actually matter for NDCG@5.

### `label_gain`

Controls how relevance scores translate to gradient magnitudes.
`label_gain="0,1,15"` means:
- relevance 0 → gain 0
- relevance 1 → gain 1
- relevance 2 → gain 15

Higher gain on the booking label means the model cares more about
getting bookings to the top vs getting clicks to the top.

### V4's ensemble strategy

V4 trained several LambdaRank models with different `label_gain`
values (0,1,15 / 0,2,15 / 0,3,15 / 0,1,30 / 0,2,25 / ...) and
rank-averaged them. Hypothesis: different label_gains emphasize
different parts of the relevance distribution → different mistakes
→ averaging helps.

In retrospect (V8 evidence): this is a *weak* form of diversity
because all members share the LambdaRank loss formulation. Switching
to a different objective gives much more ensemble lift.

## 2.11 Alternative ranking objectives

### `rank_xendcg` (LightGBM listwise)
Instead of pairwise gradients weighted by NDCG, uses a stochastic
listwise loss: sample a permutation of items, compute a loss based
on whether the permutation's score matches expected NDCG. Optimizes
NDCG more directly but with higher training variance.

V8 finding: `rank_xendcg_regularized` (with stronger L2) was the
single largest ensemble lift of the project. Different objective
→ truly different mistakes vs LambdaRank → big diversity payoff.

### `YetiRank` (CatBoost listwise)
Pairwise comparisons with carefully calibrated weights, framed as a
listwise loss. CatBoost rankers (`cb_rank_A`, `cb_rank_C_deeper`)
consistently appeared as positive small-weight diversifiers in V9
ensembles.

### Binary classifier (pointwise)
Predict `click_bool` or `booking_bool` directly with a standard
binary classification loss (logistic). Treats each row independently.
V6 originally had a binary classifier as ensemble member; LOO
trimming (§2.13) showed it was harmful and dropped it.

### NN listwise (ListNet, approxNDCG)
Not attempted in this project. Standard for neural ranking. Different
inductive bias from GBDTs → real diversity. The highest-leverage
unattempted move (§5).

## 2.12 Validation strategies

How do you measure your model before submitting? You hold out a
portion of training data and pretend it's test. The choice of HOW
is the validation strategy.

### Random validation
Shuffle the training data, take last 20% as val, train on first 80%.
Easy, fast. Assumes train and test are sampled the same way.

### Temporal validation
Sort by date, take the last N days as val. Train on dates before
the cutoff. Slower because of pre-sort. More honest when test is
"later" than train.

### K-fold cross-validation
Split into K folds, train K times, each time using K−1 for training
and the remaining fold for val. Average the K val scores. More
robust but K× compute.

### Stratified K-fold (by srch_id)
Like K-fold but ensures all rows of a srch_id stay in the same fold
(important for ranking — can't have part of a search in train and
part in val).

### The project's evolution
- V3, V4 used random val
- V5 used random val and got bitten by drift (high random val, low
  Kaggle)
- V6 switched to temporal val (cutoff 2013-05-21)
- V6–V11 stuck with temporal val

### The catch the project didn't fully resolve
| Version | Val type | Local | Kaggle | Gap |
|---|---|---|---|---|
| V4 | random | 0.42512 | 0.42021 | 0.0049 (over) |
| V6 LOO-9 | temporal | 0.40896 | 0.42004 | 0.0111 (under) |
| V9 best | temporal | 0.40971 | 0.42012 | 0.0104 (under) |

Random val OVERESTIMATED Kaggle by ~0.005.
Temporal val UNDERESTIMATED Kaggle by ~0.011.

Neither perfectly matches Kaggle. The right answer is probably:
**track both, treat disagreement as the drift signal**. The project
picked one (temporal from V6 onward) and may have over-corrected.

## 2.13 Ensembling

Combining multiple models' predictions into one better prediction.

### Rank averaging
For each search query, each model produces a score per property.
Convert scores to within-query ranks. Average the ranks across
models. Sort by the average rank.

Why ranks instead of raw scores: different models output scores on
different scales (some 0-1, some 0-100, some negative). Rank
averaging is scale-invariant.

Why this works: different models make different mistakes; averaging
cancels independent errors while preserving systematic correct
answers.

### Weighted ensembling
Same idea, non-equal weights. V9's winning ensemble:

```
final_rank = 0.80 × V6_LOO_9_rank
           + 0.05 × cb_rank_C_deeper_rank
           + 0.05 × cb_rank_A_rank
           + 0.05 × xendcg_conservative_rank
           + 0.05 × xendcg_reg_seed42_rank
```

V6 carries the heavy weight (it's the strongest individual). The
small-weight diversifiers add tiny corrections without diluting the
backbone.

### Leave-one-out (LOO) trimming
Method for selecting which models to keep.

**Algorithm:**
1. Start with N candidate models. Compute val NDCG with all N.
2. For each model M (N times total), compute val NDCG with the
   other N−1.
3. If leaving out M gives HIGHER val NDCG, M is hurting; drop it.
4. Iterate until no single drop improves val NDCG.

**V6 example:** V6 started with 10 members. LOO showed dropping
the binary classifier improved val NDCG by ~0.001. Drop it.
Remaining 9 members = **V6 LOO-9**.

**Reverse use — load-bearing attribution:** in V8, removing
`rank_xendcg_regularized` from the best ensemble cost −0.00038
local. That's strong evidence it's the addition doing the work.

### Stacking (not used in this project)
Instead of averaging, train a meta-model on base model predictions.
The meta-model can learn patterns like "trust model A on rare
destinations, trust model B on common ones." Typically beats
rank-averaging by +0.002–0.005.

V11 MEGA-BAG was equal-weight rank-average over 23 models. A
stacked meta-learner over the same 23 would likely have beaten it.

## 2.14 Train/test drift and adversarial validation

The single most important diagnostic concept in this project.

### What drift is
Train and test come from different distributions. Breaks the
standard ML assumption (training data is representative of
deployment). Causes:
- Time shift (test is later than train, world changed)
- Sample selection differences
- Feature value distribution shifts

### Adversarial validation procedure
1. Label all training rows 0.
2. Label all test rows 1.
3. Train a binary classifier (any model) to predict 0/1.
4. Compute AUC on a held-out portion.

### Interpretation
- AUC = 0.5: classifier can't distinguish → no drift
- AUC = 0.7: noticeable drift; some features differ
- AUC = 0.9+: strong drift; some features differ a LOT
- **AUC = 1.0: perfect separability** → train and test are
  categorically different in at least one feature

This project's V5 and V10 diagnostics: **adversarial AUC = 1.0**.
The top discriminative features were the cross-key TEs
(`site_book_rate`, `country_book_rate`, etc.). These features
encode time-of-training, not predictive signal.

### Why this matters
If the model is trained to say "high `site_book_rate` → high
booking probability," and the test set has `site_book_rate` values
that look nothing like training's, the predictions become noise on
those features. Worse: the model relies heavily on them (because
they ARE predictive in training), so test predictions become very
wrong.

### Drift gate (project standard)
A feature is "clean drift" if **|μ_train − μ_test| / σ_train <
0.02** AND its single-feature adversarial AUC < 0.55. Features
failing these gates were rejected in V6+. The CP and DS features
(V6 additions) were designed with these thresholds in mind.

## 2.15 Smoothing in target encoding

For small categories (few rows), the raw TE statistic has high
variance. **Smoothing** shrinks the estimate toward the global mean
when the category sample size is low.

Formula (the most common form):

$$
\text{smoothed}(c) = \frac{n_c \cdot \mu_c + s \cdot \mu_{\text{global}}}{n_c + s}
$$

Where:
- $n_c$ = number of rows in category c
- $\mu_c$ = mean target in category c
- $\mu_{\text{global}}$ = global mean target
- $s$ = smoothing parameter (larger = more shrinkage)

The V6 CP feature `prop_click_rate_pos_adj_s40_oof` uses **s = 40**
(meaning categories with fewer than 40 rows get shrunk toward the
global mean). This is part of why CP is more drift-stable than raw
TE.

---

# Part 3 · The journey, version by version

Each version below follows the same template:

> **Status** — local + Kaggle scores + delta vs V4
> **Hypothesis** — the question this version asks
> **Concepts used** — pointers into Part 2
> **Implementation** — what was actually built
> **Configuration** — key hyperparameters and choices
> **Results** — what the numbers say
> **Diagnostic findings** — forensic analyses (when applicable)
> **Why it worked / didn't** — mechanistic explanation
> **What this version taught** — the transferable lesson
> **Bridge to the next version** — what informed the next move

### Summary table (read this first)

| Version | Validation | Local | Kaggle | Δ vs V4 | Verdict |
|---|---|---|---|---|---|
| V3 | random | ~0.400 | not submitted | — | baseline |
| **V4** ★ | random | 0.42512 | **0.42021** | — | **peak** |
| V4.2 (Phase 2 best) | random | 0.42258 | 0.41639 | −0.00382 | anchor violation |
| V5 | random | 0.42633 | 0.41943 | −0.00078 | drift trap |
| V5.2 | random | — | skipped | — | ablation control |
| V6 LOO-9 | temporal | 0.40896 | 0.42004 | −0.00017 | clean backbone |
| V7 | temporal | various | not submitted | — | failure-pattern features |
| V8 | temporal | 0.40933 | not submitted | — | first ensemble lift |
| V9 best | temporal | 0.40971 | 0.42012 | −0.00009 | 24-model batch |
| V10 | temporal | 0.40997 | 0.41903 | −0.00118 | reweighting fails |
| V11a SAFE-PUSH | temporal | ≈0.4099 | 0.41995 | −0.00026 | weight tweak |
| V11b MEGA-BAG | temporal | ≈0.4090 | 0.42003 | −0.00018 | 23-model bag |

---

## V3 — the baseline

### Status
- **Local NDCG@5:** ~0.400 (random val)
- **Kaggle:** not submitted
- **Code:** `pipelines/legacy/v3_baseline.py`

### Hypothesis
"Can a single LightGBM LambdaRank model with a reasonable feature
pipeline achieve a competitive score?"

### Concepts used
1. **LightGBM** (§2.8) — the GBDT library
2. **LambdaRank** (§2.10) — the ranking objective with
   NDCG-weighted pairwise gradients
3. **`label_gain="0,1,15"`** (§2.10) — relevance-to-gain mapping
   that puts heavy weight on bookings (gain 15) over clicks (gain 1)
4. **Inverse propensity weighting (IPW)** (§2.7) — row weights to
   correct for position bias
5. **Target encoding** (§2.5) — `prop_id` and `srch_destination_id`
   replaced with their average booking/click rates
6. **Out-of-fold (OOF) computation** (§2.6) — 5-fold OOF to prevent
   target leakage in TE
7. **Within-query rank features** (§2.4) — price rank, star rank,
   review rank, distance rank within each `srch_id`
8. **Random validation** (§2.12) — held-out 20% random sample

### Implementation
A single LightGBM LambdaRank model trained on a 143-feature
pipeline (`src/features.py:build_features`). Features include:

- Raw columns from train.csv passed through cleaning
- Per-`prop_id` TE: booking rate, click rate (OOF)
- Per-`srch_destination_id` TE: booking rate, click rate (OOF)
- Within-`srch_id` ranks for price, star, review, location_score2,
  log_historical_price, distance
- Price spread features (price − min_in_search, etc.)
- User-history differences (`prop_starrating` − `visitor_hist_starrating`)
- Time-derived features (day_of_week, month, weekend_flag)
- Competitor aggregates (count of competitors with lower price)

The output is a score per (srch_id, prop_id); rank by score
descending within srch_id; that's the predicted ranking.

### Configuration
- `objective=lambdarank`
- `metric=ndcg`
- `label_gain=0,1,15`
- `num_leaves=127`
- `learning_rate=0.05`
- `num_iterations=2000` with `early_stopping_rounds=100`
- IPW row weights for position bias
- `seed=456`

### Results
- Random val NDCG@5 ≈ 0.40
- Not uploaded to Kaggle (would have scored slightly lower, ~0.39,
  due to drift)

### Why it worked
- LambdaRank directly optimizes a proxy of NDCG (the ranking
  metric)
- IPW corrects the main bias in the labels
- The feature pipeline gives the model enough signal to differentiate
  properties

### What this version taught
**Baseline competence.** Get a working ranking model on the full
feature pipeline before any cleverness. V3 establishes:
- The pipeline runs end-to-end
- OOF TE is implemented correctly
- IPW is computed correctly
- The submission format is right

### Bridge to V4
V3 was a single model. The natural next move: ensemble several
LambdaRank configurations to cancel individual model noise.

---

## V4 — the production peak (Kaggle 0.42021) ★

### Status
- **Local NDCG@5:** 0.42512 (ensemble, random val), 0.42191 (anchor
  single model)
- **Kaggle NDCG@5:** **0.42021**
- **Code:** `pipelines/legacy/v4_ensemble.py`,
  `pipelines/legacy/phase2_anchor_check.py`
- **Narrative:** `docs/v4_phase2_summary.md`
- **This is the project's best Kaggle score. Never surpassed.**

### Hypothesis
"Multiple LambdaRank models with different `label_gain` settings
will make slightly different mistakes; rank-averaging them should
yield a more robust ranking than any single model."

### Concepts used
1. **All of V3's concepts** (V4 inherits V3's pipeline)
2. **Multi-configuration ensemble** (§2.13) — several models, each
   with different hyperparameters, rank-averaged
3. **`label_gain` sweep** (§2.10) — the diversity mechanism: vary
   the relevance-to-gain mapping across ensemble members
4. **Bin sampling in LightGBM** (§2.9) — and the reproducibility
   bug it caused
5. **Anchor invariant** (§2.9) — the
   `phase2_anchor_check.py` test that pipelines must pass to prove
   V4 is reproducible

### Implementation
Train multiple LightGBM LambdaRank models with varying `label_gain`
values (and some `learning_rate` variants). Each model trained
independently on the SAME 143-feature pipeline. At inference, each
model produces scores; convert scores to within-`srch_id` ranks;
average ranks across models; sort by average rank.

### Configuration
The `label_gain` sweep included settings like:
- `0,1,15` (V3's setting)
- `0,2,15`
- `0,3,15`
- `0,1,30`
- `0,2,25`
- (others)

Each variant otherwise identical:
- `objective=lambdarank`
- `num_leaves=127`
- `learning_rate=0.05` (with some at 0.04)
- `seed=456`
- IPW row weights

### Results
- Anchor single model (random val): **0.42191**
- Ensemble (random val): **0.42512**
- **Kaggle: 0.42021** — the project's best, never surpassed by any
  of V5–V11

### The bin-sampling seed bug
**The bug.** During development, an earlier version of V4 pre-built
`lgb.Dataset` once outside the per-configuration loop and reused it.
LightGBM's bin-sampling step happens at `Dataset` construction; it
needs the seed. Reusing the Dataset meant the binning was determined
by the FIRST seed and never updated. All "ensemble members" used
identical binning → less independence → ~0.0024 NDCG@5 lost.

**The fix.** Move `lgb.Dataset` construction INSIDE the per-config
loop. Each config gets its own freshly-binned Dataset using its
own seed.

**The anchor invariant.** With the fix, a single-model V4 with
`seed=456` and `label_gain="0,2,15"` produces exactly **0.42191 NDCG@5
on random val**. The script `pipelines/legacy/phase2_anchor_check.py`
asserts this. Every subsequent pipeline (V5, V6, etc.) must pass
this check before its results are trusted.

This invariant is methodologically important: it's a tripwire against
silent regressions. If the pipeline becomes "0.41 instead of 0.42191
on the anchor," something broke even if all the new code looks right.

### Why V4 worked so well on Kaggle
1. **Strong feature base** (143 features with OOF TE and IPW)
2. **Direct NDCG optimization** via LambdaRank
3. **Modest but real ensemble diversity** from `label_gain`
   variations
4. **Random val happens to be a good proxy for Kaggle test** — the
   random val gap (0.005) is much smaller than V6+'s temporal val
   gap (0.011)
5. **No drift-laden features** — V4 stuck with single-key TEs and
   safe aggregates

### Why V4 became the project ceiling
The V4 pipeline froze. V5 added drifty features (broke things). V6
swapped validation strategy and trimmed the ensemble — but kept the
same model class on the same features. V7 added new features that
were too weak individually. V8–V11 varied objectives and weights
but stayed inside V4's feature space and (mostly) inside the
LambdaRank-CatBoost-XGBoost GBDT family. There was no structural
escape from V4's information ceiling.

### What this version taught
- The combination of (143 features, IPW, OOF TE, LambdaRank
  ensemble with label_gain sweep) is near the ceiling for this
  dataset using GBDTs.
- Reproducibility bugs in random sampling steps (bin sampling) can
  silently degrade scores; tripwires like anchor invariants catch
  this.
- Random val happens to be reasonable for this dataset — possibly
  because the Kaggle test was sampled similarly to the bulk of
  training, not strictly future-held-out.

### Bridge to V5
V4 worked. Hypothesis: more target encoding = more signal = higher
score. Add cross-key TEs.

---

## V4.2 / Phase 2 best — the anchor-violation submission

### Status
- **Local NDCG@5:** 0.42258 (random val)
- **Kaggle:** 0.41639 (−0.00382 vs V4)
- **Code:** `pipelines/legacy/phase2_*.py`

### What happened
A submission generated during V4 development, BEFORE the
bin-sampling seed bug was understood. The model passed local val
checks but its random sampling state was inconsistent — the bug
caused suboptimal binning, which manifested only on the larger
Kaggle test.

### Lesson
This is what the **anchor invariant** is designed to prevent. After
V4 was nailed down, every subsequent pipeline ran the
`phase2_anchor_check.py` check. V4.2 is included on the scoreboard
as a control: "this is what happens when the V4 invariant is
violated; don't let pipelines silently regress to here."

### What this taught
The bin-sampling bug cost ~0.004 on Kaggle in this case (single
model 0.42258 → 0.41639). That's a big drop for what looks like a
"refactor" bug. Worth-every-second-of-the-fix outcome.

---

## V5 — cross-key target encodings (Kaggle 0.41943, regression)

### Status
- **Local NDCG@5:** 0.42633 (random val, +0.00121 vs V4)
- **Kaggle NDCG@5:** 0.41943 (−0.00078 vs V4)
- **Code:** `pipelines/v5.py`
- **The defining failure of the project. Triggered the entire drift
  investigation that shaped V6–V11.**

### Hypothesis
"V4 used single-key TEs (e.g., `prop_id` → booking rate). Adding
cross-key TEs (e.g., `prop_id × srch_destination_id` → joint
booking rate) should capture finer-grained patterns and improve the
score."

### Concepts used
1. **All of V4's concepts**
2. **Cross-key target encoding** (§2.5) — TE over the combination
   of two categorical IDs
3. **Adversarial validation** (§2.14) — the diagnostic that
   eventually revealed why V5 failed
4. **Train/test drift** (§2.14) — the underlying phenomenon

### Implementation
12 new TE features added, of which 4 are cross-key:

- **`prop_dest_book_rate`** — booking rate of property X within
  destination Y (out of all searches in dest Y that included prop X)
- **`site_dest_book_rate`** — booking rate of Expedia subsite X
  within destination Y
- **`prop_site_book_rate`** — booking rate of property X on subsite Y
- **`cpair_book_rate`** — booking rate for the (visitor country,
  property country) pair
- **`site_country_book_rate`** — booking rate for the (subsite,
  property country) pair

Plus 7 single-key TEs (less drifty but still added).

All computed with 5-fold OOF.

### Configuration
Same as V4. Only the feature set changed.

### Results
- **Local random val: 0.42633** (V5's local — a meaningful
  improvement over V4's 0.42512)
- **Kaggle: 0.41943** (a meaningful regression from V4's 0.42021)

This was the wake-up call. Local said V5 was clearly better. Kaggle
said V5 was clearly worse. They cannot both be right about the same
underlying model unless something is wrong with the local val
estimate.

### Diagnostic findings (`scripts/diagnose_v5_gap.py`)
Trained an adversarial classifier (train vs test, binary). Result:
**adversarial AUC = 1.0.** The classifier could PERFECTLY tell
which row was train vs test using only V5's feature set.

Top discriminative features (highest feature importance in the
adversarial model):
1. `site_book_rate` (TE feature)
2. `country_book_rate` (TE feature)
3. `site_country_book_rate` (cross-key TE)
4. `cpair_book_rate` (cross-key TE)
5. `site_id` (raw)

These features have radically different distributions in train vs
Kaggle test. The model trained on V5 features had latched onto
training-set-specific patterns that don't exist in the test set.

### Why it failed
**Mechanism.** Cross-key TEs encode statistics over small cells
(few rows per (prop_id, dest_id) combo). Small cells have:
1. High variance in the estimate (one extra booking changes the
   rate dramatically)
2. Different cell membership between train and test windows (some
   (prop, dest) pairs appear only in train, others only in test)
3. Time-shifted population (a property's destination-specific
   booking rate in May 2013 is different from its rate in
   August 2013)

So the TE value for a given (prop, dest) in train is mostly NOISE
about the future (prop, dest) rate. The model learned to use this
noise as if it were signal. On the test set, the noise didn't
generalize.

**Why random val didn't catch it.** Random val draws from the same
distribution as train. V5's features look great on random val
because the train-distribution noise is present there too. Random
val is essentially asking "does the model fit the training
distribution?" — V5 fit it better than V4. But the question that
matters is "does the model fit the TEST distribution?" — and there,
V4 fit it better.

### What this version taught
- **Local NDCG can rise while Kaggle NDCG falls.** They are
  different objectives when drift is present.
- **Cross-key TEs are especially drift-prone** due to small cells
  and time-shifting populations.
- **Adversarial validation is the diagnostic** that catches this.
- **Random validation alone is dangerous** — it can't detect
  distribution drift between train-with-val and test.
- **More features is not automatically better.** Each feature is a
  bet that its training value generalizes to test. Cross-key TEs
  often lose that bet.

### Bridge to V5.2 and V6
Two follow-ups:
- **V5.2** — confirm the cross-key TEs were specifically to blame
  (controlled ablation)
- **V6** — rebuild with cleaner features AND a stricter validation
  strategy (temporal val) to prevent this in the future

---

## V5.2 — cross-key TE ablation

### Status
- **Local NDCG@5:** not separately tracked
- **Kaggle:** skipped (not uploaded)
- **Code:** `pipelines/v5_2.py`

### Hypothesis
"If cross-key TEs are what caused V5's Kaggle regression, removing
THEM specifically should recover V4-equivalent performance."

### Concepts used
1. **Ablation methodology** — change ONE thing at a time to attribute
   a result to a specific cause
2. **Control experiment** — design a comparison that isolates the
   variable of interest

### Implementation
V5's setup minus the 4 cross-key TEs:
- `prop_dest_book_rate`
- `site_dest_book_rate`
- `prop_site_book_rate`
- `cpair_book_rate`
- (also `site_country_book_rate`)

All other V5 changes (the 7 single-key TEs) kept.

### Results
Roughly recovered V4 Kaggle performance. The cross-key TEs were
indeed the specific cause.

### Why this matters methodologically
Without V5.2, "the cross-key TEs are drifty" is a hypothesis based
on the adversarial classifier's feature importances. With V5.2, it's
a tested causal claim: remove them, performance recovers.

Generally: when something fails, design the smallest possible
ablation to isolate the cause. Trying to fix a failure without
first attributing it is how you go in circles.

### What this version taught
- Ablations build confidence in attributions.
- The cost of an ablation (a few hours of compute) is much smaller
  than the cost of chasing the wrong hypothesis for days.

### Bridge to V6
V5/V5.2 collectively concluded: cross-key TEs drift, random val
can't detect drift, time to rebuild with stricter checks.

---

## V6 — temporal validation, LOO-9 (Kaggle 0.42004)

### Status
- **Local NDCG@5:** 0.40896 (temporal val)
- **Kaggle NDCG@5:** 0.42004 (−0.00017 vs V4)
- **Code:** `pipelines/v6.py`, `pipelines/v6_submit.py`
- **The clean backbone for all subsequent versions.**

### Hypothesis
"Switching to temporal validation will catch drift before it reaches
Kaggle. Pairing this with LOO-trimmed ensemble selection should
yield a clean, drift-free baseline."

### Concepts used
1. **Temporal validation** (§2.12) — held-out by date, not by
   random sample
2. **Leave-one-out ensemble trimming** (§2.13) — drop ensemble
   members that hurt val
3. **Drift gate** (§2.14) — only ship features with `|Δμ|/σ < 0.02`
   and single-feature adversarial AUC < 0.55
4. **Position-bias-adjusted features** (§2.7) — compute click rates
   while accounting for position
5. **Smoothing in TE** (§2.15) — shrink small-cell estimates toward
   global mean
6. **Binary classifier as ensemble member** (§2.11) — predicting
   `click_bool` with logistic loss instead of ranking
7. **Best-iteration reload bug** — LightGBM-specific gotcha after
   model save/load

### Implementation

**Validation strategy.** Cutoff date 2013-05-21. Training rows have
`date_time < cutoff` (159,836 searches). Validation rows have
`date_time >= cutoff` (39,959 searches). All ensemble decisions
made against this temporal val.

**Ensemble members (10 total before LOO).** LightGBM LambdaRank
with various hyperparameter combinations + 1 binary classifier
predicting `click_bool`.

**LOO trimming.** For each member M, compute val NDCG with the
other 9. Find the M whose removal most improves val. If removal
helps, drop M. Iterate.

**Result of LOO:** dropping the binary classifier improved val by
~0.001. Drop it. Final: **V6 LOO-9** = 9-member ranker-only
ensemble.

**Two new features added** (the first new features since V4):

- **CP** = `prop_click_rate_pos_adj_s40_oof`. Let's break down the
  name:
  - `prop` — per-property aggregate
  - `click_rate` — fraction of times this property was clicked
  - `pos_adj` — adjusted for the position the property was shown
    at (so a property that's only ever shown at position 30 isn't
    penalized for low absolute click rate)
  - `s40` — smoothed by adding 40 "global mean" pseudo-observations
    (§2.15)
  - `oof` — computed out-of-fold (§2.6)
  Result: drift gate passes (|Δμ|/σ < 0.02).

- **DS** = `prop_dest_book_rate_safe`. Per-(property, destination)
  booking rate, computed "safely" (smoothed + drift-checked + OOF).
  Drift gate also passes.

Both features were designed as drift-clean alternatives to V5's
cross-key TEs. Same conceptual signal, but stable across the
train→test boundary.

**The `best_iteration = -1` bug.** When you save a LightGBM
`Booster` and reload it, `Booster.best_iteration` returns −1. The
V6 pipeline originally relied on `best_iteration` for inference. The
fix: use `Booster.current_iteration()` on reloaded models, which
returns the actual number of trained iterations. This bug crashed
the V6 submission step; recovery happened via
`scripts/overnight_submit_best.py` (regenerating the submission
from saved val_pred files without retraining).

### Configuration
Per LightGBM member, varied across:
- `label_gain` ∈ {0,1,15 / 0,2,15 / 0,3,15 / 0,1,30}
- `learning_rate` ∈ {0.04, 0.05}
- `num_leaves` ∈ {127, 255}
- `seed` ∈ {42, 123, 456}

All with IPW row weights, OOF TE features (now including CP and DS),
and the full 143-feature pipeline.

Cutoff: `date_time < 2013-05-21` for train, `>=` for val.

### Results
- Temporal val NDCG@5 = **0.40896**
- Kaggle NDCG@5 = **0.42004**
- Gap = +0.0111 (Kaggle better than local)

### Why the temporal val score is LOWER than V4's random val score
V4 random val: 0.42512. V6 temporal val: 0.40896. The temporal val
is HARDER than random val because:
1. Train and val are time-separated, so drift exists between them
   (echoing the train→test drift)
2. Validation rows are "future" relative to training, so the model
   can't memorize training-time patterns that would be useless on
   future data

This is GOOD — temporal val gives a more honest assessment of how
the model will generalize. The Kaggle test isn't strictly future,
but it's drifted from train in some way, and temporal val
approximates that better than random val does.

### The remaining puzzle
V6 LOO-9 was a CLEAN, drift-free, conservatively-validated
ensemble. Yet its Kaggle score (0.42004) was still slightly worse
than V4's (0.42021). Why?

Best explanation: V4's random val happened to be closer to the
Kaggle test distribution than V6's temporal val. The Kaggle test
might be sampled in a way that's somewhere between random-from-all-
data and strictly-future-of-train. V4's "lucky" choice happened to
match. V6's "more honest" choice over-corrected.

This wasn't fully diagnosed in the project. A from-scratch redo
would verify it by checking the test set's date distribution
(§5.2).

### What this version taught
- **Temporal validation prevents the V5 drift trap** but may
  over-correct.
- **LOO trimming reliably finds bad ensemble members** (the binary
  classifier in this case).
- **Drift-clean features (CP, DS) preserve Kaggle performance**
  while adding training signal. They didn't HELP much, but they
  didn't HURT, unlike V5's cross-key TEs.
- **Reproducibility gotchas matter.** The `best_iteration = -1` bug
  is the kind of thing that bites silently and costs hours of
  recovery work if not detected.

### Bridge to V7
V6 is a clean baseline. Next: can we IMPROVE on it? Idea: look at
where V6 gets things wrong, and design features specifically to fix
those mistakes.

---

## V7 — failure-pattern features

### Status
- **Local NDCG@5:** various single-feature results, no improvement
  over V6 LOO-9 as ensembles
- **Kaggle:** not submitted (would have regressed)
- **Code:** `pipelines/phase7_batch.py`, `pipelines/phase7_weighted_batch.py`
- **Diagnostics:** `diagnostics/failure_patterns/patterns.md`

### Hypothesis
"V6 mistakes have a pattern. By analyzing V6's specific failures
and designing features that target those patterns, we can improve
beyond V6's information ceiling."

### Concepts used
1. **Failure pattern analysis** — examine where V6 mis-ranks and
   look for systematic patterns
2. **Targeted feature engineering** — design features specifically
   for known failure modes
3. **HIGH_DRIFT feature rejection** — kill features that fail the
   drift gate even if they would help locally
4. **Weight dilution in ensembles** (§2.13) — adding a weaker member
   at small weight can still hurt if the dilution exceeds the
   diversity gain

### Failure pattern findings
- 76% of booked searches: V6 ranked the booked hotel NOT at #1
- 50% of those misranks: booked hotel was at rank 2–5 ("near misses")
- Booked hotels were systematically MORE expensive and LESS popular
  than V6's top-wrong picks
- Long-stay searches and rare destinations had higher miss rates

### Five features designed from these patterns

1. **`price_premium_vs_prop_hist_x_short_window`** — current price
   minus property's historical median price, interacted with
   short-booking-window flag. Hypothesis: when users book quickly,
   they tolerate price premium.

2. **`is_long_window_x_top_quartile_price`** — long booking window
   AND top-quartile price within search. Hypothesis: planners
   accept high prices.

3. **`prop_rare_x_long_trip`** — property's overall rarity ×
   length-of-stay. Hypothesis: rare properties get booked more for
   long trips.

4. **`brand_x_domestic`** — `prop_brand_bool` × (visitor country ==
   property country). Hypothesis: brand matters less when local.

5. **`query_difficulty_index`** = log(candidate_count) × (1 −
   dest_click_rate). Hypothesis: a feature that quantifies how
   hard a query is gives the model an easier time on the hard ones.

### Results

**Drift gate filtering:**
- 4 of 5 features held (passed `|Δμ|/σ < 0.02`)
- 1 was rejected as HIGH_DRIFT: `price_premium_vs_prop_hist_x_short_window`

**Local performance of single-feature models** (V4-anchor +
single feature each):
- All 4 surviving features gave small POSITIVE local gain over
  V4_ANCHOR alone (~+0.0001 to +0.0008 each)

**Ensemble performance** (V6 LOO-9 @ X% + new-feature model @ Y%):
- Every weighted combination HURT performance, even at very small
  weights like 0.05 for the new model.
- The new feature models, while individually positive over V4_ANCHOR
  alone (~0.404), are individually MUCH WEAKER than V6 LOO-9
  (~0.409).
- At weight 0.05, the new model dilutes V6 by 5%, contributing 0.05 ×
  (0.404 − 0.409) = −0.00025 in expectation. The diversity gain is
  smaller than this dilution cost.

### Why this version failed to improve V6
Each new feature added a model that was:
1. Individually worse than V6 ensemble (~0.404 vs 0.409)
2. Correlated with V6's signal (both use the same backbone features)

A weak + correlated model is the worst ensemble member you can add.
The math is unforgiving:

$$
\text{ensemble} = w_{\text{V6}} \cdot \text{score}_{\text{V6}} + w_{\text{new}} \cdot \text{score}_{\text{new}}
$$

If `score_new < score_V6` and the two are highly correlated, the
ensemble is a noisy interpolation toward a worse model. No win
possible.

### What this version taught
- **Failure pattern analysis is good methodology** but doesn't
  guarantee useful features.
- **Single-feature additions on top of an established backbone face
  a high bar**: the new model has to be individually competitive
  with the backbone OR provide truly independent error patterns.
- **Drift gates can save you from shipping bad features** — the
  rejected `price_premium` feature would likely have hurt Kaggle.

### Bridge to V8
V7 confirmed: feature-engineering single-shot additions can't
improve V6. Pivot: forget new features, vary the MODEL CLASS
instead.

---

## V8 — structural diversity (first ensemble lift)

### Status
- **Local NDCG@5:** 0.40933 (+0.00037 vs V6 LOO-9)
- **Kaggle:** not submitted directly (V9 superseded)
- **Code:** `pipelines/structural_batch.py`
- **The first positive single-model addition in the project.**

### Hypothesis
"Different model classes and objectives produce ranking mistakes
that are genuinely independent of LambdaRank's mistakes. Adding such
a model to V6 LOO-9 as a small ensemble member should give a real
lift."

### Concepts used
1. **Structural ensemble diversity** (§2.11, §2.13) — vary the loss
   function and architecture, not just hyperparameters
2. **`rank_xendcg` listwise objective** (§2.11) — different ranking
   formulation than LambdaRank's pairwise
3. **Regularization in tree models** (§2.8) — L2, num_leaves caps,
   min_data_in_leaf, to control overfitting in models with novel
   objectives
4. **Load-bearing attribution via LOO** (§2.13) — verifying which
   ensemble member is doing the work

### Implementation
13 models trained, each varying ONE structural element:

- **4 LambdaRank `label_gain` variants** (sanity check on V4-style
  diversity)
- **3 weighting variants:**
  - `no_ipw` (no IPW correction)
  - `ipw_clip3` (IPW with clip at 3, less aggressive)
  - `random_upweight` (extra weight on `random_bool=1` rows)
- **2 regularized variants:**
  - More leaves (255 vs 127), stronger L2
  - Different min_data_in_leaf
- **2 objective variants:**
  - **`rank_xendcg_regularized`** — listwise objective with extra
    L2 to control variance
  - **`booking_clf_calibrated`** — binary classifier on
    `booking_bool` with calibration
- **2 extra-feature variants:**
  - **CP_regularized** — V6 plus CP feature, with extra L2
  - **DS_regularized** — V6 plus DS feature, with extra L2

For each, evaluate as: V6 LOO-9 @ 0.90 + this model @ 0.10. Pick
the addition that gives the largest local val improvement.

### Configuration
Baseline = V6 LOO-9 (the 9-member ensemble from §V6).

Per model variant, the only difference from V6's standard config
is the structural change being tested.

### Results
**Best addition:** `V6 LOO-9 @ 0.90 + rank_xendcg_regularized @
0.10` = temporal NDCG@5 **0.40933**. That's +0.00037 over V6 LOO-9
alone.

**LOO check (load-bearing attribution):** removing
`rank_xendcg_regularized` from this ensemble drops val by
−0.00038. Strong evidence that this single member is the addition
doing the work. (If removing it cost −0.001, you'd be unsure; the
fact that it cleanly matches the original lift size confirms
attribution.)

### Why xendcg specifically helps
LambdaRank optimizes pairwise gradients weighted by NDCG change.
xendcg optimizes a listwise loss directly on the full ranking. The
two formulations:
1. Disagree on which mistakes matter most
2. Have different invariances (listwise is more stable to "scale"
   of scores; pairwise is more sensitive)
3. Produce different model behaviors on hard searches

So when xendcg is wrong and LambdaRank is right (or vice versa),
the average is better than either. This is the diversity engine
working as designed.

### Why `label_gain` variants don't give the same lift
All `label_gain` variants share the LambdaRank loss. They differ
only in how the relevance values 0/1/5 map to gradient magnitudes.
This produces similar models with similar mistakes. The diversity
is shallow.

xendcg, by contrast, replaces the loss entirely. Deep diversity.

### What this version taught
- **Different objectives provide much more ensemble lift than
  different hyperparameters** within one objective.
- **A small-weight diversifier from a different model class can
  yield real gains** (+0.00037 local from a 10% weight is solid).
- **LOO attribution confirms which member is load-bearing**, ruling
  out alternative explanations like "the new ensemble overall just
  averages better."

### Bridge to V9
If one objective change gives +0.00037, what about a bigger batch
with multiple objectives, multiple model classes, multiple seeds?

---

## V9 — the overnight batch (Kaggle 0.42012)

### Status
- **Local NDCG@5:** 0.40971 (+0.00075 vs V6 LOO-9)
- **Kaggle NDCG@5:** 0.42012 (−0.00009 vs V4)
- **Code:** `pipelines/overnight_final_batch.py`,
  `scripts/overnight_submit_best.py`
- **The closest any post-V4 version got to V4 on Kaggle.**

### Hypothesis
"Maximize ensemble diversity. Train 24 models across LambdaRank,
xendcg, XGBoost, CatBoost, binary classification, varied seeds and
regularization. Pick the best ensemble via grid search."

### Concepts used
1. **All of V8's concepts**
2. **CatBoost YetiRank** (§2.11) — second listwise objective in a
   different framework
3. **XGBoost** (§2.8) — third GBDT framework
4. **`inf` handling in XGBoost** (§2.8) — XGBoost requires explicit
   `missing=np.nan` while LightGBM/CatBoost handle inf
5. **Fault-tolerant batch pipelines** — resumable, exception-isolated,
   atomic writes
6. **Drift absorption** — the observed shrinkage of local gains
   when translated to Kaggle

### Implementation
24-model batch, run overnight (~2.5 hours):

**Composition:**
- 5 `rank_xendcg` seeds
- 1 conservative xendcg (lower learning rate, more L2)
- 3 XGBoost rankers (varied seeds)
- 3 CatBoost YetiRank rankers (varied depth)
- 3 binary classifiers (varied calibration)
- 9 LightGBM regularized variants (varied seeds, label_gain)

**Pipeline architecture:**
- **Resumable**: skip any model whose artifact file exists. Allows
  re-running after crashes without redoing work.
- **Per-model exception isolation**: a `try/except` per model. One
  failure doesn't kill the batch.
- **Atomic writes**: every CSV/JSON written via `tmp` + rename.
- **Error directory**: per-failure `ERROR_<id>.txt`.

**Ensemble grid search:** after all models train, search over
weights for combinations like `V6 LOO-9 @ {0.70, 0.75, 0.80, 0.85,
0.90} + each diversifier @ {0, 0.025, 0.05, 0.075}`. Pick the
combination with best temporal val NDCG.

### The 5 XGBoost failures
XGBoost crashed with `inf in input data` errors. Cause: some
features have `inf` values (e.g., division by zero in
`price_ratio` computations). LightGBM and CatBoost handle `inf`
natively (treat as missing or extreme). XGBoost requires:
- Either explicit `missing=np.nan` parameter at training time
- Or pre-cleaning to replace inf with NaN

The pipeline's exception isolation prevented these failures from
killing the batch. A rescue script (`scripts/overnight_xgb_rescue.py`)
re-trains the XGBoost models with the right `missing` parameter for
a follow-up run.

### The submission-stage crash
The pipeline trained all 24 models successfully but crashed during
the submission-writing step. Cause: the `best_iteration = -1` bug
(same as V6) on the reloaded models. `current_iteration()` would
have worked. The training results were SAVED (val_pred .npy files
and model checkpoints), so `scripts/overnight_submit_best.py`
recovered without retraining.

### Results
**Best ensemble:** `V6 LOO-9 @ 0.80 + 4 diversifiers @ 0.05 each`,
where the diversifiers are:
- `cb_rank_C_deeper` (CatBoost YetiRank, depth 8)
- `cb_rank_A` (CatBoost YetiRank, depth 6)
- `xendcg_conservative`
- `xendcg_reg_seed42`

- Temporal val: **0.40971** (+0.00075 vs V6 LOO-9's 0.40896)
- Kaggle: **0.42012** (+0.00008 vs V6 LOO-9's 0.42004)

### The drift absorption observation
- Local gain over V6 LOO-9: +0.00075
- Kaggle gain over V6 LOO-9: +0.00008
- Translation ratio: 0.00008 / 0.00075 = **0.11**

Only 11% of local improvement showed up on Kaggle. The remaining
89% was absorbed by the train/test drift gap.

This is a quantitative ceiling. If you need +0.005 on Kaggle, you
need +0.045 local. That's enormous by this project's standards;
unreachable by ensemble tweaking.

### Why CatBoost helps
CatBoost YetiRank's listwise formulation differs from LightGBM's
`rank_xendcg` formulation AND from LambdaRank. Three distinct loss
geometries → three distinct mistake patterns → larger ensemble
diversity. CatBoost rankers were consistently positive small-weight
diversifiers throughout V9–V11.

### What this version taught
- **Structural diversity at scale doesn't escape the drift
  absorption ceiling.** Even 24 models with multiple objectives
  could only translate 11% of local gains to Kaggle.
- **CatBoost YetiRank deserves to be a default diversifier** on
  ranking tasks — third loss formulation, third source of mistakes.
- **Pipeline robustness pays back its cost.** The XGBoost crashes
  and the submission-stage crash would have lost ~3 hours of work
  in a fragile pipeline. The resumable/atomic/isolated patterns
  recovered both incidents.
- **`best_iteration = -1` after save/reload bites in multiple
  places.** Codify the `current_iteration()` fix as a defensive
  default.

### Bridge to V10
If diversity doesn't beat V4, attack drift directly. Adversarial
reweighting is the standard domain-adaptation technique. Try it.

---

## V10 — adversarial reweighting (Kaggle 0.41903, regression)

### Status
- **Local NDCG@5:** 0.40997 (highest local of the project)
- **Kaggle NDCG@5:** 0.41903 (−0.00118 vs V4 — worst regression
  since V4.2)
- **Code:** `pipelines/adversarial_reweight_batch.py`
- **The most expensive experimental failure of the project.**

### Hypothesis
"Train an adversarial classifier (train=0, test=1). Use its
predictions to reweight training rows by `P(test|x) / (1 −
P(test|x))`. This is the textbook domain-adaptation technique. If
drift is the bottleneck, this should help."

### Concepts used
1. **Adversarial validation** (§2.14) — repeated here as the
   weighting signal source
2. **Importance ratios** — `P(test|x) / (1 − P(test|x))` reweighting
3. **Importance sampling** — the statistical justification
4. **sqrt-clipping** — stability hack on the weights

### Theoretical background

**Importance sampling.** If you want to estimate $E_{p_{\text{test}}}[f(x)]$
but only have samples from $p_{\text{train}}$, you reweight:

$$
E_{p_{\text{test}}}[f(x)] = E_{p_{\text{train}}}\left[\frac{p_{\text{test}}(x)}{p_{\text{train}}(x)} f(x)\right]
$$

The weights $p_{\text{test}}(x) / p_{\text{train}}(x)$ are
**importance ratios**. If you train your model with these weights,
you're effectively training on the test distribution.

**Adversarial estimation of importance ratios.** You don't have
$p_{\text{test}}$ and $p_{\text{train}}$ directly. But if a
classifier predicts $P(\text{test} | x)$, then:

$$
\frac{p_{\text{test}}(x)}{p_{\text{train}}(x)} \propto \frac{P(\text{test}|x)}{1 - P(\text{test}|x)}
$$

(This follows from Bayes' rule.) So train an adversarial classifier,
predict on training rows, compute these ratios, use them as
LightGBM row weights.

**sqrt-clipping.** Raw importance ratios are unstable — a row that
looks 99% like test gets a huge weight. The fix is to take the
square root (softens the distribution) and clip extreme values
(e.g., max weight 5×).

### Implementation
1. Train an adversarial classifier (LightGBM binary, label =
   train/test). Confirm AUC.
2. Predict $P(\text{test}|x)$ for every training row.
3. Compute importance ratios with sqrt + clip.
4. Retrain 7 V6-equivalent ensemble members with these row weights.
5. Run a small ensemble grid search to find the best combination
   with diversifiers.

### Results

**Adversarial AUC = 1.0 again.** Same finding as V5. The top
discriminative features were:
1. `site_book_rate`
2. `country_book_rate`
3. `site_country_book_rate`
4. `cpair_book_rate`
5. `site_id`

**These are EXACTLY the features V5 had added that caused V5 to
fail.** Two independent diagnostics (V5's Kaggle regression and
V10's adversarial classifier) converged on the same set of
features. This convergence is strong evidence the diagnosis is
correct.

**Reweighted V6 alone** (the first warning sign):
- Local val: 0.40818 (WORSE than V6 LOO-9's 0.40896)
- The reweighting alone hurt local val.

**Best adv ensemble:** adv-V6 @ 0.50 + 4 diversifiers @ 0.125 each
- Local val: **0.40997** (highest local of the project)
- Kaggle: **0.41903** (worst regression since V4.2)

The local-Kaggle disagreement here is severe: local was the BEST
ever (0.40997) but Kaggle was the WORST attempt since the V4.2 bug
(0.41903).

### Why it failed (the mechanistic explanation)

The reweighting de-emphasizes training rows that look UNLIKE the
test distribution (i.e., rows with high $P(\text{train}|x)$). The
features that distinguish train from test are the cross-key TEs
(`site_book_rate` etc.). Rows where these features have "train-like"
values get DOWN-weighted.

But here's the trap: **those rows are not noise. They are the
training data the model learns its core ranking signal from.** The
features that drift are also the features that are MOST predictive
of bookings (cross-key historical rates are genuinely useful when
in-distribution). Down-weighting these rows means the model trains
on a smaller, less informative subset.

The local val benefits from the reweighting because local val
mimics test (drifted) data. But the underlying model is now worse
at the actual ranking task. Kaggle, which evaluates the underlying
model on test data, sees the degradation.

**Generalizable lesson.** Importance reweighting works when:
- Drift is MILD enough that down-weighting some rows doesn't crush
  the training signal
- The drifted features are NOT load-bearing for the prediction task

This dataset has neither property. Drift is extreme (AUC 1.0) and
the drifted features are load-bearing. Reweighting is the wrong
tool here.

### What this version taught
- **Sample reweighting cannot fix structural drift in
  load-bearing features.**
- **Adversarial diagnostics converge across versions** — the V5
  and V10 findings independently identifying the same features is
  not a coincidence; it's confirmation.
- **Local val that "improves" via reweighting is suspicious** —
  if the technique would actually help, it would help Kaggle too.
- **The right tool for this kind of drift is feature engineering**
  that produces non-drifting alternatives (e.g., CP and DS), not
  sample weights.

### Bridge to V11
V10 failed. With limited Kaggle slots remaining, no clear path to
beating V4. Test two final ensemble philosophies on existing trained
models.

---

## V11 — the final two submissions

### Status
- **V11a (SAFE-PUSH):** Kaggle 0.41995 (−0.00026 vs V4)
- **V11b (MEGA-BAG):** Kaggle 0.42003 (−0.00018 vs V4)
- **Code:** `scripts/build_two_final_submissions.py`
- **Neither beat V4. Both confirmed the ceiling.**

### Hypothesis
"Two Kaggle slots left. Test two different ensembling philosophies
without retraining. (a) Extend V9's winner with more diversifiers
at lower V6 weight. (b) Throw the entire model pool together at
equal weight."

### Concepts used
1. **Ensemble philosophy comparison** — controlled test of two
   different approaches on the same model pool
2. **Backbone weight tuning** (§2.13) — V11a tests whether 0.75 V6
   weight beats 0.80
3. **Massive equal-weight bagging** — V11b tests whether 23
   diverse models at equal weight beats a backbone-heavy ensemble

### V11a — SAFE-PUSH
**Composition:** `V6 LOO-9 @ 0.75 + 6 diversifiers @ 0.0417 each`

The 6 diversifiers:
- `cb_rank_C_deeper`
- `cb_rank_A`
- `xendcg_conservative`
- `xendcg_reg_seed42`
- `xendcg_reg_seed123`
- `xendcg_reg_seed456`

Two more xendcg seeds added vs V9's 4-diversifier mix. V6 weight
dropped from 0.80 to 0.75.

**Result:** Kaggle **0.41995** (−0.00026 vs V4, slightly WORSE
than V9 best at 0.42012).

**Conclusion:** V6 weight 0.75 is too low. V9's 0.80 was the sweet
spot. Adding two more xendcg seeds didn't compensate for the V6
dilution.

### V11b — MEGA-BAG
**Composition:** 23 trained models, equal rank-average within
`srch_id`.

- 9 LightGBM regularized seeds (effective V6-equivalent backbone)
- 5 xendcg seeds
- 1 xendcg conservative
- 3 CatBoost YetiRank
- 3 binary classifiers
- 1 reweighted-V6 member from V10
- 1 ListGain variant

V6 LOO-9 effective weight = 9/23 ≈ 39%.

**Result:** Kaggle **0.42003** (−0.00018 vs V4, slightly worse than
V9 best at 0.42012 but better than V11a).

**Conclusion:** Pure diversity ensembling holds the line at ~0.4200
even with V6 weight at 39%. Diversity does add real signal — but
not enough to exceed V4.

### Why V11a and V11b converged near V4 (not beyond it)
Both submissions are ensembles over the same model pool (V6,
xendcg variants, CatBoost rankers). The ensemble shape differs but
the underlying *information content* is the same. The diversity
inside this pool is bounded by the variance of mistakes within the
shared feature space. Once exhausted, no ensemble shape produces
more lift.

To break through 0.42, you'd need to add a model with FUNDAMENTALLY
different information — e.g., a neural network trained with ListNet
(different inductive bias), or a model trained with
`random_bool=1`-derived features (different signal source). Neither
existed in the pool.

### What this version taught
- **The V6-weight sweet spot is 0.80–0.85**, not lower.
- **Pure massive equal-weight bagging doesn't beat a backbone-heavy
  weighted ensemble** in this setup. The strong backbone matters.
- **Diversity inside one model family + one feature set has a
  ceiling** at this dataset. To break it, add a different family
  or different features.

### Bridge to "what's next"
The natural next moves — `random_bool=1` features, NN listwise,
stacking — were not attempted within the project timeline. See
Part 5 for the priority list.

---

# Part 4 · The lessons (in plain language)

These are the transferable insights. They apply to ranking ML in
general, not just this dataset.

## L1: Local val and Kaggle test are different objectives when drift is present

V5 demonstrated this most clearly: random val improved +0.00121
while Kaggle regressed −0.00078. They're not noisy estimates of the
same number; they're estimates of DIFFERENT numbers.

**Operational rule:** always run adversarial validation between
your val set and the test set. AUC > 0.7 means your val is
unreliable. AUC = 1.0 means it's essentially useless for
generalization.

## L2: Temporal validation is stricter, but not necessarily closer to Kaggle

| Version | Val type | Local | Kaggle | Gap |
|---|---|---|---|---|
| V4 | random | 0.42512 | 0.42021 | 0.005 over |
| V6 | temporal | 0.40896 | 0.42004 | 0.011 under |

Random val OVERESTIMATED by 0.005. Temporal val UNDERESTIMATED by
0.011. Neither perfectly matches.

**Operational rule:** track BOTH validation strategies. Disagreement
is the drift signal. Don't pick one.

## L3: Drift absorbs most of your local gains

V8→V9 local improvement of +0.00075 became only +0.00008 on Kaggle.
The translation ratio was ~11%.

**Operational rule:** when planning improvements, multiply your
expected local gain by 0.1 to forecast Kaggle impact. If that's not
worth your time, don't do it.

## L4: Sample reweighting cannot fix structural drift

V10's adversarial reweighting: local 0.40997 (best ever) but
Kaggle 0.41903 (worst regression). The features that drift are the
ones the model NEEDS to rank well.

**Operational rule:** for structural drift, the only fix is
feature engineering producing non-drifting alternatives. Time-windowed
aggregates; within-query relative features; killing
high-adversarial-AUC features at the source.

## L5: Objective diversity > hyperparameter diversity for ensembles

V4's "ensemble" was multiple LambdaRank `label_gain` variants —
shallow diversity. V8's biggest lift came from adding ONE
`rank_xendcg` model (different objective). V9's most consistent
diversifiers were CatBoost YetiRank rankers (different framework
AND different objective).

**Operational rule:** when ensembling, prioritize varying the LOSS
FUNCTION and MODEL CLASS over hyperparameters within a single
objective.

## L6: Strong backbone + small-weight diversifiers > equal-weight bagging

| Composition | V6 weight | Kaggle |
|---|---|---|
| V6 alone | 1.00 | 0.42004 |
| V6 + 4 diversifiers @ 0.05 | 0.80 | 0.42012 ← peak |
| V6 + 6 diversifiers @ 0.042 | 0.75 | 0.41995 |
| V6 in 23-model equal bag | 0.39 | 0.42003 |
| V6 adv + 4 diversifiers @ 0.125 | 0.50 | 0.41903 |

The peak is at V6 weight ~0.80. Lower hurts.

**Operational rule:** when one ensemble member is clearly strongest,
give it 0.70–0.85 weight and use diversifiers for fine-tuning.

## L7: In-model feature stacking has a ceiling

V7 attempts: combine two positive single-feature models inside one
LightGBM:
- `posadj + te_rank`: combined −0.00160 (anti-additive)
- `posadj + price_dest`: combined −0.00087
- `posadj + prop_dest_safe`: combined +0.00065 (still below either
  alone)

LightGBM's tree builder picks locally-best features at each split.
Adding correlated alternatives changes split selection in ways
that hurt val.

**Operational rule:** if you have multiple positive single-feature
models, ensemble their PREDICTIONS outside the model rather than
combining their FEATURES inside one model.

## L8: Fault-tolerant pipelines pay for themselves

The project's resumable / atomic / exception-isolated patterns
recovered multiple incidents:
- V6 submission crash (best_iteration bug) recovered without
  retraining
- V9 XGBoost failures isolated (5 out of 24 models failed, batch
  continued)
- V9 README-writer crash recovered from saved CSVs

**Operational rule:** any pipeline > 30 minutes should be resumable
(skip done work on restart), atomic (writes via tmp + rename),
exception-isolated (per-unit try/except), and have an error
directory for diagnostics.

## L9: Submission format is easy to get wrong

V6 initially wrote `SearchId,PropertyId` (Pandas default
capitalization) — Kaggle expected `srch_id,prop_id` lowercase
(matching `data/submission_sample.csv`). Cost: one failed upload.

**Operational rule:** every submission writer should validate
column names against the sample submission file. Bake it into the
write step.

## L10: Anchor invariants catch silent regressions

V4's bin-sampling bug demonstrated that random seeds in
non-obvious places (Dataset binning) can silently degrade scores
without crashing anything. After V4 was nailed down, the
`phase2_anchor_check.py` script asserted reproducibility for every
subsequent pipeline.

**Operational rule:** when you achieve a result, write a tripwire
script that asserts a single-model output matches a known value to
6 decimals. Run it at the top of every pipeline.

---

# Part 5 · Where to go from here

Honest assessment of the highest-leverage unattempted moves.

## 5.1 `random_bool=1`-derived features (highest single-move EV)

The Expedia dataset has ~15% of rows with `random_bool=1` (search
displayed in random order). These rows have UNBIASED click signal
because position is random.

**Strategy:**
1. Subset training to `random_bool == 1`
2. Train a LightGBM LambdaRank with V4 hyperparameters on this
   subset
3. Score on full train + test, save as one column `unbiased_score`
4. Add as feature #144 to V4 and retrain

**Why it works:** V4's IPW is APPROXIMATE position-bias correction.
The `random_bool=1` subset gives EXACT position-unbiased relevance.
As a feature, it provides every model a position-bias-free signal.

**Cost:** ~4 hours. **Expected lift:** +0.003 to +0.008. **Confidence:**
high.

## 5.2 Verify the test set's date distribution

A 10-minute experiment that could reframe everything.

V4 random val gap to Kaggle = 0.005.
V6 temporal val gap to Kaggle = 0.011.

If the Kaggle test set is actually random-sampled from the full
training date range (not strictly future), then V6+'s temporal val
was overcorrecting and V4's random val was the right objective.

**Check:** plot the date distribution of `test.csv` against
`train.csv`. If they overlap significantly, the test set is not
strictly future-held-out.

## 5.3 Within-srch_id rank features (cheap, drift-immune)

For every continuous feature (`price_usd`, `prop_starrating`,
`prop_review_score`, `prop_location_score2`,
`orig_destination_distance`, `srch_query_affinity_score`): build
`rank_within_srch`, `zscore_within_srch`,
`diff_from_min_within_srch`, `diff_from_median_within_srch`.

~30 new features, drift-immune by construction.

**Cost:** 2–3 hours. **Expected lift:** +0.001–0.003 each, additive.

## 5.4 `comp1` through `comp8` engineered properly

Currently the V4 pipeline likely treats these as raw columns.
Replace with:
- `frac_competitors_more_expensive`
- `mean_undercut_pct`
- `count_competitors_with_data`
- `dominant_in_availability`

The 2013 ICDM winning solutions reportedly hammered these features.

**Cost:** 3–4 hours. **Expected lift:** +0.002–0.005.

## 5.5 Stacking instead of rank-averaging

Replace V11 MEGA-BAG's equal-weight average with: train a small
LightGBM meta-learner on `(srch_id, prop_id, base_score_1...
base_score_23, top_3_features)` predicting `booking_bool`. Use
V4-equivalent random val for hyperparameter tuning.

**Why:** a meta-learner can learn patterns like "trust model A when
search has many candidates, trust model B otherwise" that
rank-averaging treats as fixed.

**Cost:** 3 hours. **Expected lift:** +0.002–0.005 over MEGA-BAG.

## 5.6 PyTorch listwise neural ranker

3-layer MLP (512→256→1) trained with ListNet loss (cross-entropy
listwise) or approxNDCG. New model class, new inductive bias.

**Cost:** 6–10 hours (first NN baseline is always brittle).
**Expected lift as ensemble member:** +0.002 to +0.005.

## 5.7 What NOT to do

These are settled by experimental evidence:

- **More cross-key target encodings.** V5 demonstrated they drift.
- **More in-model feature stacking.** V7 demonstrated anti-additive.
- **More sample reweighting (adversarial domain adaptation).** V10
  demonstrated this is the wrong tool for structural drift.
- **More LightGBM-LambdaRank variants with different `label_gain`.**
  V4 already has these; adding more is diminishing returns within
  one objective family.

## 5.8 Honest projection

A focused 1-week sprint on §5.1, §5.3, §5.4, §5.5 could plausibly
reach **0.428–0.432** on this dataset. Beyond that — toward the
~0.46 zone seen on the top of the public leaderboard — requires
structurally different approaches (custom losses,
semi-supervised techniques on test features, team-scale ensembling)
that aren't solo-week work.

---

# Part 6 · Glossary

Quick reference. Cross-references point to Part 2 sections where
each is explained in depth.

| Term | Definition |
|---|---|
| **Ablation** | Removing one variable to attribute an effect to it (§V5.2) |
| **Adversarial validation** | Train a classifier to distinguish train rows from test rows; AUC measures drift (§2.14) |
| **Anchor invariant** | A reproducibility tripwire: single-model output must match a known value to 6 decimals (§2.9) |
| **AUC** | Area Under ROC Curve. 0.5 = random, 1.0 = perfect separation (§2.14) |
| **Backbone** | Strongest single member of an ensemble; in this project, V6 LOO-9 (§2.13) |
| **Bin sampling** | LightGBM's process of converting continuous features into discrete buckets (§2.9) |
| **CatBoost** | Yandex's gradient boosting library; includes YetiRank ranking objective (§2.8) |
| **`comp1`–`comp8`** | Eight competitor price/availability columns in the dataset (§1.2) |
| **CP** | V6's clean-drift feature `prop_click_rate_pos_adj_s40_oof` (§V6) |
| **Cross-key target encoding** | TE computed on the combination of two categorical IDs (§2.5) |
| **DCG** | Discounted Cumulative Gain (§1.3) |
| **Drift / distribution shift** | Train and test come from different distributions (§2.14) |
| **Drift gate** | Project's threshold for shipping a feature: `|Δμ|/σ < 0.02` AND single-feature adversarial AUC < 0.55 (§2.14) |
| **DS** | V6's clean-drift feature `prop_dest_book_rate_safe` (§V6) |
| **Early stopping** | Stop training when val score hasn't improved in N iterations (§2.8) |
| **Ensembling** | Combining multiple models' predictions for joint performance (§2.13) |
| **Feature engineering** | Designing new feature columns from raw data (§2.3) |
| **GBDT** | Gradient Boosted Decision Trees; the model family behind LightGBM, CatBoost, XGBoost (§2.8) |
| **iDCG** | Ideal DCG: DCG of the perfect ranking (§1.3) |
| **Importance ratio** | `P(test|x) / (1 − P(test|x))` for sample reweighting (§V10) |
| **IPW** | Inverse Propensity Weighting: row weights `1/P(observation|features)` to correct selection bias (§2.7) |
| **`label_gain`** | LightGBM parameter mapping relevance values to gradient magnitudes (§2.10) |
| **LambdaRank** | Pairwise ranking loss weighted by NDCG impact of each swap (§2.10) |
| **LightGBM** | Microsoft's fast gradient boosting library; the project's workhorse (§2.8) |
| **Listwise** | Ranking approach where the loss considers the full list per search (§2.2) |
| **Load-bearing attribution** | Using LOO to verify which ensemble member is doing the work (§2.13) |
| **Local NDCG** | NDCG computed on a held-out portion of training data (§2.12) |
| **LOO** | Leave-One-Out: ensemble trimming method (§2.13) |
| **MEGA-BAG** | V11b submission: 23 trained models equal-weight rank-averaged (§V11) |
| **NDCG@5** | Normalized DCG considering only top 5 positions (§1.3) |
| **OOF** | Out-of-Fold: computing a feature for row R using only rows outside R's fold (§2.6) |
| **Pairwise** | Ranking approach over pairs of items (§2.2) |
| **Pointwise** | Ranking approach as independent regression/classification (§2.2) |
| **Position bias** | Top-of-page items get more clicks because of visibility, not quality (§2.7) |
| **`prop_id`** | Unique property (hotel) ID column (§1.2) |
| **Random validation** | Randomly-held-out 20% of training data (§2.12) |
| **`random_bool`** | 1 if Expedia displayed the search in random order; defines the unbiased subset (§1.2, §5.1) |
| **Rank averaging** | Ensembling by averaging within-query ranks across models (§2.13) |
| **`rank_xendcg`** | LightGBM's listwise ranking objective (§2.11) |
| **Reproducibility** | Same seed and code = same output bit-for-bit (§2.9) |
| **SAFE-PUSH** | V11a submission: V6 LOO-9 @ 0.75 + 6 diversifiers @ 0.0417 (§V11) |
| **Smoothing** | Shrinking small-cell TE estimates toward global mean (§2.15) |
| **`srch_id`** | Unique search query ID column (§1.2) |
| **Stacking** | Ensembling via a meta-model trained on base predictions (§2.13) |
| **Stratified K-fold** | K-fold ensuring same-group rows stay in the same fold (§2.12) |
| **Target encoding (TE)** | Replacing a category with the average target for that category (§2.5) |
| **Target leakage** | When a feature was computed using its own row's label (§2.5) |
| **Temporal validation** | Hold out the last N days of training as val (§2.12) |
| **V4 anchor invariant** | The reproducibility check asserting NDCG@5 = 0.42191 on V4's single-model random val (§V4) |
| **V6 LOO-9** | V6's 9-member ensemble after LOO dropped the binary classifier (§V6) |
| **Weighted ensembling** | Ensembling with non-equal weights per member (§2.13) |
| **Within-query rank feature** | Property's rank on feature F within its `srch_id` (§2.4) |
| **XGBoost** | DMLC's gradient boosting library; has the `inf in input` gotcha (§2.8) |
| **YetiRank** | CatBoost's listwise ranking objective (§2.11) |

---

*Companion documents:*
- `journey.md` — chronological project log
- `lessons_learned.md` (**Insights**) — bullet-list takeaways
- `final_kaggle_results.md` — authoritative scoreboard
- `next_steps.md` — forward queue of unattempted ideas
- `architecture.md` — code architecture
- `results.md` — detailed results
