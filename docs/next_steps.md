# Next steps

_Last updated: 2026-05-16, post-V6._

V6 Kaggle = **0.42004** (vs V4 0.42021, essentially flat). Local +0.00217
did not translate. Top leaderboard ~0.46 → gap ~0.04. Feature engineering
on the existing pipeline has hit a ceiling (see `CHANGELOG.md` and
`docs/results.md` for the proof).

This file is the forward queue. Read `CHANGELOG.md` and `docs/results.md`
first for context.

---

## What's NOT worth doing next

These have been ruled out by data, not speculation:

- **More in-model feature stacking on V4_ANCHOR config.** Three combinations
  of two positive features were all anti-additive (`docs/results.md`).
  The bottleneck is GBDT split selection with high feature correlation
  at this dataset size.
- **More single-key TE variants.** Of `prop_click_rate_pos_adj_s40_oof`,
  `prop_book_rate_pos_adj_s40_oof`, `prop_rel_rate_pos_adj_s40_oof`, only
  the click version improved (+0.00132 local). Book/rel were noise.
- **`booking_clf` as an ensemble member.** Confirmed by LOO to actively hurt.
- **Cross-key TEs without drift control.** V5 had adversarial AUC = 1.0;
  V5.2 ablation dropped the worst offenders. Any new cross-key TE must
  pass an adversarial-AUC check (`scripts/diagnose_v5_gap.py` pattern).

---

## Top candidates, ranked by expected leverage × confidence

### 1. Hard-negative mining / failure-driven features

Train V6, predict on val, identify the `prop_id`s the model consistently
mis-ranks (e.g. predicted top-5 but actual relevance = 0, with high
frequency). Encode those signals as features for the next iteration:

- "How often was this prop_id a hard negative in past queries?"
- "How often did this prop_id displace the actual booked hotel?"

This is *failure-driven*: the feature targets exactly the rows the model
is currently getting wrong. Mechanic similar to boosting but at the
feature-engineering layer. Expected local gain: +0.002 to +0.005 if the
mis-rank pattern is consistent.

Sketch: produce a per-`prop_id` "mis-rank frequency" feature from V6
predictions on temporal_val. Verify drift on temporal_val before testing
on Kaggle.

### 2. Adversarial sample reweighting

V5's adversarial AUC was 1.0 — a simple classifier could perfectly tell
train vs test apart. V6 had cleaner features but local→Kaggle gap was
still wider than the temporal proxy suggested.

Train an adversarial classifier (train vs test, on shared columns).
Weight each train row by `P(row is test) / (1 - P(row is test))`.
Retrain V6 with these weights. Local NDCG@5 may *drop*, but Kaggle
should improve.

Expected gain: +0.003 to +0.008 Kaggle. Cheap (one classifier + one retrain).

### 3. Heterogeneous base learners in the ensemble

V6's 9 members are all LightGBM with slight config variations. Adding
genuine model-class diversity may help more than another LGBM variant:

- XGBoost rank (`xgboost.train` with `rank:pairwise`)
- CatBoost with `YetiRank`
- A small NN listwise ranker (e.g. ListNet)

Each becomes a separate ensemble member; combine via the existing
rank-average mechanic in `pipelines/v6_submit.py`. Expected gain:
+0.003 to +0.010 Kaggle.

### 4. Loss-side position-bias handling

Current pipeline uses IPW as a sample weight. Alternatives:

- **Propensity-weighted LambdaRank**: scale per-pair losses by inverse
  propensity, not per-row.
- **Two-stage**: train V0 on random_bool=1 only (truly unbiased); then
  fine-tune on full data using V0 as a regulariser.

Higher engineering cost (custom LightGBM objective or two trainings).
Expected gain: +0.005 to +0.015 if executed well, but risky.

### 5. Robust temporal validation

The local→Kaggle correlation is still weak with temporal val. Possible
improvements:

- Multi-seed temporal: split by `srch_id` randomly within the
  post-cutoff window, take the multi-seed mean.
- Time-block cross-validation: roll the cutoff forward in monthly steps,
  use the mean as the local proxy.
- Adversarial-weighted val: weight val rows by their `P(test)` estimate.

This won't *improve* Kaggle directly; it gives a more reliable selection
signal so we stop wasting runs on features that overfit local.

---

## Operational invariants (carry forward)

- **Temporal split is the validation contract.** Anchor: V4_ANCHOR
  temporal NDCG@5 = 0.40401. Reference: `pipelines/temporal_validation.py`.
- **No `lgb.Dataset.construct()`** outside the per-config loop. See
  `docs/architecture.md` §V4 anchor invariant.
- **Submission CSV header is `srch_id,prop_id`** lowercase. Verified in
  `pipelines/v6.py` and `pipelines/v6_submit.py`.
- **CP and DS are separate ensemble members.** Never combined inside a
  single model (proven anti-additive in V6).
- **Any new TE feature requires an adversarial-AUC check** on the
  generated feature distribution between train and Kaggle test (proxy: val).
- **Sub-anchor members must pass LOO** before joining an ensemble.
  Solo NDCG < anchor is NOT a disqualifier; harmful LOO contribution IS.
