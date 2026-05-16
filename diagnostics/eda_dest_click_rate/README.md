# EDA — `dest_click_rate` as a query-context feature

**Source:** temporal_val (978,308 rows / 39,959 searches).
**Predictions:** KEEP model `model_prop_click_rate_pos_adj_s40_oof_temporal.txt`
(overall temporal NDCG@5 = 0.40533).

## Bucketing

Per-search `dest_click_rate` tertiles:
- **low**   < 0.03993
- **mid**   0.03993 – 0.04498
- **high**  ≥ 0.04498

## 1. bucket_summary.csv

```
bucket  n_searches  n_rows  book_rate/srch  cand_count  price_spread  NDCG@5  Recall@5    MBR
   low      13,316 376,627          0.7394       28.28        279.13  0.3800    0.5813   6.88
   mid      13,323 337,181          0.6745       25.31      2,563.17  0.3835    0.5892   6.65
  high      13,320 264,500          0.6861       19.86      1,004.32  0.4525    0.6901   5.15
```

**Headline gaps:**

- **NDCG@5 is +0.073 higher in `high` vs `low` (0.453 vs 0.380)** — the model
  performs 19% better on popular-destination queries. The `low` bucket is
  where most of the model's error budget sits.
- `candidate_count` is **42% smaller in `high`** (20 vs 28) → high-engagement
  queries are narrower, fewer candidates to rank, easier intrinsically.
- `book_rate_per_row` is **+32% in `high`** (3.5% vs 2.6%) → bookings concentrate
  in popular destinations.

## 2. booked_hotel_profile_by_bucket.csv

For the booked hotel (the winner) in each bucket:

```
bucket  n_booked  price_rank_norm_mean  price_vs_mean_mean  location2_rank_norm  review_rank_norm  value_score_rank_norm  quality_rank_avg
   low     9,846                0.4449             −23.63               0.327              0.357                  0.397             0.351
   mid     8,987                0.4342              +7.75               0.355              0.360                  0.415             0.364
  high     9,139                0.4655             +70.60               0.332              0.365                  0.406             0.360
```

**Dominant shift: `price_vs_mean` swings ~94 USD between `low` and `high`.**

- In **low** buckets the winner is, on average, **$23 below** the query's mean
  price. Users in lesser-known destinations book *cheaper-than-average*.
- In **high** buckets the winner is, on average, **+$71 above** the query mean.
  Users in popular destinations are willing to pay a premium.
- Rank-based features (price_rank_norm, location2_rank_norm, review_rank_norm,
  value_score_rank_norm, quality_rank_avg) move very little — within 0.01–0.04
  across buckets. **The shift is in absolute price, not rank.**

The current model already has `price_vs_mean` and `dest_click_rate` as separate
features, but must learn the interaction through tree depth. An explicit
interaction would give it directly.

## 3. segment_summary.csv

Highlights from the cross-tabulation (3 segmentations × 3 buckets):

```
segmentation     segment       bucket  n_searches  book_rate/srch  NDCG@5  Recall@5    MBR
candidate_count   low            low        2,276          0.7201  0.5249    0.7743   3.79
candidate_count   low           high        7,168          0.6999  0.5319    0.7883   3.71
candidate_count  high            low        8,774          0.7496  0.3448    0.5360   7.73
candidate_count  high            mid        5,895          0.6909  0.3261    0.5102   8.06
candidate_count  high           high        2,154          0.6773  0.3285    0.5284   7.81
length_of_stay   los_3plus       low        3,658          0.6159  0.3550    0.5566   7.11
length_of_stay   los_3plus      high        4,143          0.5443  0.4191    0.6554   5.67
```

- **Hardest segment: `candidate_count` = high × any `dest_click_bucket` → NDCG ≈ 0.33.**
  Roughly a third of val searches; carries most of the headroom.
- **Easiest: `candidate_count` = low → NDCG > 0.52 regardless of bucket.**
- `domestic` flag has only a small effect across buckets (∼0.38 vs 0.40 in low).
- Long-stay queries (`los_3plus`) lose ~0.04 NDCG vs short stays across all
  buckets — long-stay is independently harder.

## Recommended interaction features (max 2)

Both motivated by the actual shifts above, in priority order:

### 1. `price_vs_mean × dest_click_rate` (recommended)

Direct multiplicative interaction:

```python
df["price_vs_mean_x_dest_click"] = df["price_vs_mean"] * df["dest_click_rate"]
```

**Why:** the booked-hotel `price_vs_mean` swings 94 USD between low and high
buckets — the largest cross-bucket shift in the data. The model currently has
to learn this through tree depth on two separate features. An explicit product
gives a single feature whose sign and magnitude encode "premium-acceptable in
high-engagement destinations, penalty in low-engagement". Expected leverage:
this is the lowest-hanging fruit from this EDA.

Optional sign-asymmetric variant (probably stronger since the effect flips sign):

```python
df["price_above_dest_premium"] = (df["price_vs_mean"].clip(lower=0)
                                  * df["dest_click_rate"])
df["price_below_dest_discount"] = (-df["price_vs_mean"].clip(upper=0)
                                   * (1 - df["dest_click_rate"]))
```

### 2. `query_difficulty` = `candidate_count / dest_click_rate`

```python
df["query_difficulty"] = df["query_hotel_count"] / (df["dest_click_rate"] + 1e-3)
```

(`query_hotel_count` already exists as the candidate-count feature.)

**Why:** the hardest val segment is "high candidate_count × low dest_click"
(NDCG ≈ 0.33), the easiest is "low candidate_count × high dest_click"
(NDCG ≈ 0.53) — a 0.20 NDCG spread driven by the joint, not by either alone.
A single ratio feature lets the model flag "long-tail destination, many
hotels to disambiguate" and adjust its split structure for that regime.

## Not recommended (ruled out from the data)

- `dest_click × review_rank_norm` / `dest_click × value_score_rank_norm` — the
  rank-feature shifts across buckets are < 0.02 (within sampling noise);
  GBDT can already capture these through tree depth without help.
- `domestic × dest_click` — domestic effect is only +0.02 NDCG within `high`
  and flat in `low`; not enough leverage.

## Files

- `bucket_summary.csv` — per-bucket counts, rates, model metrics
- `booked_hotel_profile_by_bucket.csv` — winner-hotel feature means/medians
- `segment_summary.csv` — full cross-tabulation (3 segmentations × 3 buckets)
