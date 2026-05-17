# Next steps — what would be tried with a longer timeline

This document describes ideas that were not attempted within the project
timeline, ordered by expected impact. It is intended for a hypothetical
continuation of the work, or as a guide for similar projects.

## State at project end

- Best Kaggle public NDCG@5: **0.42021** (V4 ensemble).
- Top of public leaderboard: approximately 0.46. Gap from our best: +0.04.
- Validation strategy: temporal split (cutoff 2013-05-21).
- Most useful diagnostic: adversarial validation (AUC = 1.0) confirmed
  that train and test distributions are radically different, primarily
  in cross-key target-encoding features.

The remainder of this document discusses what would be tried next if the
work continued.

## What is NOT worth pursuing

These have been ruled out by direct experimental evidence:

- **More in-model feature stacking** on the V4_ANCHOR configuration.
  Three combinations of two positive features were anti-additive (see
  `lessons_learned.md` §5).
- **Sample-importance reweighting (adversarial domain adaptation).** V10
  attempted this and lost −0.00118 on Kaggle. The drift is in features
  the model must use; penalizing them hurts generalization.
- **More cross-key target encodings.** V5's cross-key TEs had
  adversarial AUC = 1.0 — they are the drift problem, not a solution.
- **Single-feature additions to V6 LOO-9 as ensemble members.** Their
  individual NDCG@5 (around 0.404) is too far below V6's ensemble level
  (0.409). Any positive weight dilutes V6 more than it adds.

## Promising directions

In order of expected leverage × confidence:

### 1. Neural network listwise model (NEW model class)

PyTorch implementation of ListNet (cross-entropy listwise loss) or
approximate-NDCG loss. Architecture: a 3-layer MLP (512 → 256 → 1) with
softmax over the candidates within each search query.

**Why this should help:**
- Different model class from V6's nine LambdaRank members and the V9
  CatBoost rankers.
- Different inductive bias may capture ranking signal that GBDT misses.
- In Kaggle Expedia 2013 post-mortems, several top teams used either
  RankNet/ListNet or stacked GBDT + NN.

**Estimated cost:** 4–6 hours implementation + 1–2 hours training on
CPU. Estimated Kaggle lift as an ensemble member: +0.002 to +0.005.

**Risk:** medium. NN training on tabular data is historically more
fragile than GBDT; getting a working baseline that beats V6 LOO-9 on
local can take several iterations.

### 2. Two-stage propensity model

Train a base model on `random_bool = 1` rows only (rows displayed in
random order, so the relevance signal is unbiased by Expedia's prior
ranking). Then fine-tune on the full training set, using the base
model's predictions as a regularization signal (e.g., as a feature, or
via distillation).

**Why this should help:**
- The base model learns from truly unbiased data, so it cannot pick up
  the position-bias-induced shortcuts that even IPW cannot fully remove.
- The fine-tuning step lets the model use the full data while the base
  model anchors it to unbiased predictions.

**Estimated cost:** 6–8 hours implementation + training. Estimated
Kaggle lift: +0.003 to +0.008 if the train→test drift is partially
position-bias-induced.

**Risk:** medium-high. Implementation complexity is higher than NN
listwise.

### 3. Hard-negative mining as features

Train V6, predict on val, identify property IDs that V6 consistently
mis-ranks (e.g., predicted top-3 but actual relevance = 0, occurring in
> N% of past queries with high price/star/etc). Encode these signals as
new features for the next training cycle.

**Why this should help:**
- The features are derived from V6's specific failure modes, so they
  add information V6 doesn't already have.
- Unlike the failure-pattern features in V7 (which were aggregate
  hypotheses about WHAT V6 mis-ranks), this approach uses V6's actual
  errors as labels.

**Estimated cost:** 3–4 hours for the feature derivation + 1 hour for
retraining. Estimated Kaggle lift: +0.002 to +0.005 if the mis-rank
patterns are consistent.

**Risk:** medium. Could overfit to V6's specific mistakes if the new
features don't generalize.

### 4. Heterogeneous base learners + seed bagging

Add more diversifier seeds and frameworks: 10 seeds of
`rank_xendcg_regularized`, 10 seeds of CatBoost YetiRank, all bagged
with strong stochasticity (`feature_fraction = 0.6`, `bagging_fraction
= 0.7`, different `bagging_freq`). Rank-average all seeds within each
framework, then ensemble.

**Why this should help:**
- Seed bagging is the most reliable variance-reduction technique for
  tabular GBDT.
- Combined with the framework diversity (LightGBM rank_xendcg +
  CatBoost YetiRank), the resulting bag is genuinely independent of
  V6's LambdaRank backbone.

**Estimated cost:** 6–10 hours of training time across the seeds.
Estimated Kaggle lift: +0.001 to +0.003.

**Risk:** low. Even if the lift is small, the cost is mostly compute
time, not engineering.

### 5. Robust temporal validation

The current local→Kaggle gap (+0.011) is large and noisy. A more
reliable local proxy would reduce wasted Kaggle submissions.

Possible improvements:
- Multi-seed temporal: split by `srch_id` randomly within the
  post-cutoff window, take the multi-seed mean local NDCG@5.
- Time-block cross-validation: roll the cutoff forward in monthly
  steps, use the mean across folds.
- Adversarial-weighted val: weight val rows by their estimated
  `P(test|x)` to emphasize test-like rows.

**Why this should help:**
- A more reliable local metric is not a direct Kaggle improvement,
  but it makes every other experiment more efficient.

**Estimated cost:** 2–3 hours implementation. No direct Kaggle gain;
this is infrastructure.

## Operational invariants to preserve

If continuing this work, the following invariants should be carried
forward:

- **Temporal split** as the primary validation contract. Random split
  proved misleading.
- **No `lgb.Dataset.construct()`** outside the per-configuration loop
  (see `docs/architecture.md` §V4 anchor invariant).
- **Kaggle submission CSV header must be `srch_id,prop_id`** lowercase.
- **CP and DS as separate ensemble members.** In-model feature stacking
  is consistently anti-additive at this dataset size and feature count.
- **Adversarial AUC check** on every new target-encoding feature
  before adoption. If train→test AUC > 0.7 for that feature alone, it
  is a drift candidate and likely to fail on Kaggle.
- **Sub-anchor models must pass LOO** before joining an ensemble. A
  solo NDCG below V4_ANCHOR does NOT disqualify a model; a harmful LOO
  contribution does.
