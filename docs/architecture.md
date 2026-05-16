# Architecture

How the repo is wired. Read once; refer back when something moves.

## Conceptual layout

```
┌─────────────────────────────────────────────────────────────────┐
│                        data/ (raw CSVs)                         │
│                training_set_VU_DM.csv (4.96M rows)              │
│                test_set_VU_DM.csv     (4.96M rows)              │
└────────────────────────┬────────────────────────────────────────┘
                         │ load_train / load_test
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                            src/                                 │
│  data_loader.py  →  load_train, load_test, split_val,           │
│                     make_target (relevance = 5·book + click)    │
│  features.py     →  build_features (143-col pipeline) +          │
│                     IPW + kfold TE helpers                      │
│  evaluate.py     →  ndcg_at_k, evaluate_ndcg                    │
│  submission.py   →  CSV writer (srch_id,prop_id header)         │
│  artifacts.py    →  per-run artifact saving helpers             │
│  config.py       →  paths, ID/label column lists                │
└────────────────────────┬────────────────────────────────────────┘
                         │ build_features
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       feature matrix                            │
│  143 base cols (V4 standard) + optional V6 add-ons (CP, DS)     │
└────────────────────────┬────────────────────────────────────────┘
                         │ lgb.Dataset(group=…) + label_gain
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       pipelines/                                │
│  temporal_validation.py  →  build temporal split (cutoff        │
│                              2013-05-21), anchor V4_ANCHOR      │
│                              + B3 on temporal/random control    │
│  evaluate_variant.py     →  single-feature variant harness      │
│                              (cached parquet features, fast)    │
│  overnight_experiments.py → bulk LGBM grid on random split (V4) │
│  v5.py                   →  V5 7-member ensemble + Kaggle sub   │
│  v5_2.py                 →  V5 minus drift-TEs (ablation)       │
│  v6.py                   →  10-member ensemble + LOO + sub gate │
│  v6_submit.py            →  V6 submission builder (resumable)   │
│  legacy/                 →  V3 baseline, V4 ensemble, Phase 2/3 │
└────────────────────────┬────────────────────────────────────────┘
                         │ train + predict
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  models/             saved LightGBM boosters (gitignored)       │
│  submissions/        Kaggle CSVs (gitignored)                   │
│  artifacts/          per-run JSON + CSV (small files tracked)   │
│  diagnostics/        analyses + cached features                 │
└─────────────────────────────────────────────────────────────────┘
```

## Validation contracts

There are **two** validation contracts in the repo. They are NOT
interchangeable.

### Random val (V4 / V5 era)

- 10% of `srch_id`s held out randomly with `random_state=456`.
- Reference: `src.data_loader.split_val`.
- Anchor: V4_ANCHOR config (lg=0,1,15, IPW default) → **0.42191** NDCG@5.
- Used by: `pipelines/legacy/*`, `pipelines/v5.py`, `pipelines/v5_2.py`,
  `pipelines/overnight_experiments.py`.
- Known issue: local→Kaggle correlation is weak. V5 won locally
  (0.42633 > V4's 0.42512) but lost on Kaggle (0.41943 < 0.42021).

### Temporal val (V6 era)

- Cutoff date: **2013-05-21 16:55:42**. Searches after this date go to val.
- Train: 159,836 searches / 3.98M rows. Val: 39,959 searches / 978K rows.
- Reference: `pipelines.temporal_validation.temporal_split`.
- Split is asserted leak-free (`leakage_assertions_passed: true` in
  `diagnostics/temporal_validation_20260516_113831/split_meta.json`).
- Anchor: V4_ANCHOR config retrained on temporal_train → **0.40401** NDCG@5.
- Used by: `pipelines/v6.py`, `pipelines/v6_submit.py`,
  `pipelines/evaluate_variant.py`, all V6 work.

The temporal split is intended to be a better proxy for Kaggle's held-out
test (which is also future-dated relative to training). V6 still showed a
non-trivial gap (local +0.00217 → Kaggle ~0), so the proxy is closer but
not perfect.

## Features

### Base 143 features (V4 standard)

Built by `src.features.build_features` in this order:
1. `temporal_features` — month, hour, dayofweek, is_weekend
2. `missing_flags` — informative null indicators
3. `price_features` — log price, price/night, price/person, ratios
4. `visitor_match_features` — star_diff, price_ratio_to_visitor_hist
5. `competitor_features` — comp_rate_count, comp_cheaper_count, etc.
   (raw `comp{i}_*` columns then dropped)
6. `quality_features` — value_score, location_total, is_domestic
7. `listwise_features` — within-srch_id ranks: price_rank_norm,
   star_rank_norm, location2_rank_norm, etc.
8. `interaction_features` — price_per_star, distance_x_international,
   booking-window segments, family flags, is_discounted
9. `hotel_aggregates` — k-fold OOF target encodings:
   - Single-key: `prop_click_rate`, `prop_book_rate`, `prop_rel_rate`,
     `dest_click_rate`, `dest_book_rate`, `country_book_rate`, `site_book_rate`
   - Cross-key: `prop_dest_book_rate`, `site_dest_book_rate`, `prop_site_book_rate`,
     `cpair_book_rate`, `site_country_book_rate`
   - Property-level: `prop_book_given_click`, `prop_click_count`,
     `prop_count`, `prop_mean_price`, `prop_std_price`, `prop_avg_position`,
     destination-relative deltas

The full feature_cols.json is preserved in `artifacts/overnight_20260516_003443/`.

### V6 add-ons

Two leak-safe features added during the V6 cycle, implemented as helper
functions in `pipelines/evaluate_variant.py`:

**CP — `prop_click_rate_pos_adj_s40_oof`** (`_pos_adj_oof_te`)
- Position-adjusted click rate per `prop_id`.
- `global_click_rate = mean(click_bool)` on train.
- `position_click_rate[p] = mean(click_bool | position=p)` on train.
- `exposure_weight = clip(global / position_click_rate, 0.2, 3.0)` —
  downweights top positions, upweights buried positions.
- Per `prop_id`: `weighted_click_sum = Σ exposure_weight · click_bool`,
  `weighted_exposure_sum = Σ exposure_weight`.
- Smoothed: `(weighted_click_sum + 40·global) / (weighted_exposure_sum + 40)`.
- 5-fold OOF by `srch_id` on train side; full-train aggregate on val/test.
- Local NDCG@5 gain: **+0.00132** (KEEP). Drift `|Δμ|/σ = 0.018`.

**DS — `prop_dest_book_rate_safe`** (`_prop_dest_book_rate_safe`)
- Smoothed `(prop_id, srch_destination_id)` booking rate with a 3-way fallback
  for thin pairs:
  - `prop_rate` from per-`prop_id` aggregate
  - `dest_rate` from per-`srch_destination_id` aggregate
  - `global_rate` overall
- `fallback_rate = 0.5·prop_rate + 0.3·dest_rate + 0.2·global_rate`.
- Feature = `(pair_book + 40·fallback_rate) / (pair_count + 40)`.
- 5-fold OOF on train; full-train aggregate on val/test.
- Median pair has only 2 observations — strong motivation for smoothing.
- Local NDCG@5 gain: **+0.00094** (HOLD). Drift `|Δμ|/σ = 0.004` (cleanest of session).

## V4 anchor invariant

Any pipeline that uses lg=0,1,15, seed=456, IPW default, and the full V4
feature set on the random split **must** reproduce val NDCG@5 = 0.42191.
If it doesn't, the `lgb.Dataset` was likely pre-constructed before
`lgb.train` could propagate `seed=456` to bin sampling. The fix is to let
`lgb.train` construct the dataset lazily. Verification script:
`pipelines/legacy/phase2_anchor_check.py`.

## Cached features

`diagnostics/eval_variants/base_features_temporal_{train,val}.parquet`
holds the temporal-split 143-feature matrices, plus the `relevance`,
`click_bool`, `booking_bool`, `position`, `srch_id`, `random_bool`
columns needed for labels/groups/weights. Building these from scratch
takes ~5–10 min; loading from parquet is <1 sec.

`pipelines/evaluate_variant.py` and `pipelines/v6.py` both check the
parquet sidecar metadata (`base_features_meta.json`) for the expected
row/search counts before reusing. A mismatch triggers a rebuild.

## Side-effect hygiene

`v6.py` and `v6_submit.py` create their per-run output directories
**inside `main()`**, not at module-load time, so importing these modules
(as `v6_submit.py` does) does not create empty stub dirs.

## Submission format

Kaggle Expedia expects the lowercase header `srch_id,prop_id` matching
`data/submission_sample.csv`. Both `pipelines/v6.py` and
`pipelines/v6_submit.py` emit this format. Any future submission writer
should use the helper in `src/submission.py` or replicate its column-name
contract.
