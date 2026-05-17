# Failure-pattern analysis — V6 booked vs top-wrong

**Source:** temporal_val. 27,972 booked queries. V6 9-member rank-average ensemble predicted top-1.
- **Successes** (V6 top-1 = booked): 6,717 (24.0%)
- **Failures** (V6 top-1 ≠ booked): 21,255 (76.0%)

**Booked-hotel rank profile within failures:**

| V6 booked rank bin | n | % |
|---|---:|---:|
| 2–5 | 10,734 | **50.5%** |
| 6–10 | 5,236 | 24.6% |
| 11–20 | 3,892 | 18.3% |
| 20+ | 1,393 | 6.6% |

→ **Half of all failures are "near misses" (rank 2–5).** The model is in the right neighborhood; it's the final ordering that's off. Only 6.6% are deep failures.

**V4 vs V6 head-to-head (all 27,972 booked):**

| metric | V4_ANCHOR | V6 ensemble |
|---|---:|---:|
| mean booked position | 6.275 | 6.182 |
| V6 helps (booked pos lower) | — | 23.4% (6,551) |
| V6 hurts (booked pos higher) | — | 19.9% (5,568) |
| Net V6 top-5 advantage | — | +145 queries |

---

## Headline directional pattern (across ALL failures)

The booked hotel vs the model's top-wrong, averaged across 21,255 failures (sorted by `|Δ| / |booked|`):

| feature | booked mean | top_wrong mean | delta | reading |
|---|---:|---:|---:|---|
| `price_vs_median` | −3.68 | **−10.06** | **+6.38** | booked is **+$6 more expensive** vs query median |
| `prop_dest_book_rate` | 0.037 | **0.076** | −0.039 | booked is **half as popular** in this destination |
| `comp_rate_advantage` | 0.11 | 0.21 | −0.10 | booked has **less competitor advantage** |
| `prop_book_rate` | 0.035 | 0.059 | −0.024 | booked is **less booked historically** |
| `prop_click_rate` | 0.053 | 0.083 | −0.030 | booked is **less clicked historically** |
| `promotion_flag` | 0.26 | **0.38** | −0.13 | booked is **less often promoted** |
| `location2_rank_norm` | 0.38 | 0.23 | +0.16 | booked has **worse loc2 rank within query** |
| `prop_location_score2` | 0.16 | 0.22 | −0.06 | booked has **lower raw loc2 score** |
| `price_vs_mean` | −16.7 | −23.1 | +6.38 | booked is **+$6 above query mean** |
| `value_score_rank_norm` | 0.44 | 0.28 | +0.16 | booked has **worse value-rank within query** |
| `price_vs_prop_mean` | −57.7 | −73.7 | **+16.0** | booked is **+$16 above its own historical price** |
| `prop_avg_position` | 14.9 | 11.5 | +3.42 | booked sits **deeper in historical display** |

**Synthesis: the model OVERWEIGHTS popularity + promotion + comp advantage + value/cheapness.**
**The booked hotel is consistently MORE expensive, LESS popular, LESS promoted than the model's pick.**
Users are paying a premium for something idiosyncratic that doesn't show up in the aggregate features.

---

## Pattern 1 — Model overvalues historical popularity

**Segment:** all failures (sign is identical across every dest/cand/los/dom/family/window bucket).

**Observed difference:** booked hotel has 30–50% lower `prop_dest_book_rate`,
`prop_click_rate`, `prop_book_rate`, and shallower historical position
(`prop_avg_position` is 3–4 positions HIGHER on average).

**Evidence (across all 21,255 failures):**

| feature | booked | top_wrong | delta |
|---|---:|---:|---:|
| `prop_dest_book_rate` | 0.037 | 0.076 | −0.039 |
| `prop_click_rate` | 0.053 | 0.083 | −0.030 |
| `prop_book_rate` | 0.035 | 0.059 | −0.024 |
| `prop_avg_position` | 14.9 | 11.5 | +3.42 |

**Possible feature:** **`prop_underdog_indicator`** — a flag/score for hotels
in the bottom quartile of `prop_book_rate` BUT with rising recent engagement
(e.g., short-window click rate > long-window click rate). Or:
**`prop_popularity_residual`** = the model's V6 score with `prop_*_rate`
features ablated, minus the actual rank → captures rows where TE is misleading.

**Expected direction:** features should encode "this hotel is bookable
*despite* low TE" — so they should have POSITIVE sign for booked underdogs.

**Risk:** MEDIUM. Adding a "popularity counter-signal" risks introducing
drift if recent vs historical engagement differs between train and test.
Verify adversarial AUC < 0.6 before adopting.

---

## Pattern 2 — Booked hotels are systematically MORE expensive than the model's pick

**Segment:** all failures, especially short booking window + low candidate count.

**Observed difference:** `price_vs_prop_mean` is +$16 higher for the booked
hotel than for the model's top-wrong. The model picks the cheaper-than-prop-history option;
users book the priced-above-history option.

**Evidence (per segment, delta = booked − top_wrong on `price_vs_prop_mean`):**

| segment | n_failures | Δ price_vs_prop_mean | reading |
|---|---:|---:|---|
| short booking window (≤7d) | 8,266 | **+$21.07** | strongest |
| mid booking window (7–30d) | 6,791 | **+$24.25** | strongest |
| long booking window (>30d) | 6,198 | +$0.26 | NEGLIGIBLE |
| low candidate count | 6,294 | **+$24.82** | strongest |
| high candidate count | 9,722 | +$17.64 | strong |
| mid candidate count | 5,239 | +$2.44 | weak |
| low dest_click_rate | 7,650 | +$11.71 | strong |
| high dest_click_rate | 6,675 | +$17.94 | strong |
| LOS ≤ 2 | 15,735 | +$16.99 | strong |
| LOS ≥ 3 | 5,520 | +$13.25 | moderate |
| domestic | 13,900 | +$17.24 | strong |
| international | 7,355 | +$13.72 | moderate |

**Possible feature:** **`price_premium_vs_prop_hist`** = clip(`price_vs_prop_mean`, 0, +inf)
× `is_short_window` (or × `1 / log1p(srch_booking_window)`). Encodes "user is
willing to pay above this prop's normal price, in a short-window booking".

Alternative: **`price_premium_segment_indicator`** — a single ternary signal
indicating whether the row is in a "premium-acceptable" segment (short window
OR low cand_count) AND priced above prop history.

**Expected direction:** positive coefficient — the model currently rejects
above-history pricing, this feature should partly restore it.

**Risk:** LOW–MEDIUM. The pattern is segment-consistent and follows from
booking psychology (urgency → less price-sensitive). Drift risk low because
`price_vs_prop_mean` itself was used in V5 and adversarial AUC was bounded.

---

## Pattern 3 — Long booking window inverts the price pattern

**Segment:** booking window > 30 days.

**Observed difference:** In long-window searches, the `price_vs_prop_mean`
delta collapses to +$0.26 (essentially flat), but `price_vs_median` jumps to
**+$13.08** — booked is more expensive vs the query median, but matches its
own prop history. This means in long-window bookings users are NOT seeking
"discounted" prop instances; they're seeking prop instances that are normally
priced but happen to be the higher-end ones within the query.

**Evidence:**

| segment | Δ price_vs_prop_mean | Δ price_vs_median |
|---|---:|---:|
| short booking window | +$21.07 | +$4.52 |
| mid booking window | +$24.25 | +$2.53 |
| long booking window | +$0.26 | **+$13.08** |

**Possible feature:** **`is_long_window_x_price_in_top_quartile`** —
indicator for long-window queries where this hotel's price is in the top
25% within-query. Captures "people booking far in advance for nicer hotels".

**Expected direction:** positive — model should upweight top-quartile-priced
hotels specifically in long-window queries.

**Risk:** LOW. Long-window queries are 29% of failures, well-represented.

---

## Pattern 4 — Long stays book RARER hotels

**Segment:** LOS ≥ 3 days, family searches, international.

**Observed difference:** the booked hotel has 25 fewer historical impressions
(`prop_count`) than the top-wrong in long stays — a 25-point gap is substantial
given p50(prop_count) = 12.

**Evidence:**

| segment | Δ prop_count | Δ prop_dest_book_rate |
|---|---:|---:|
| LOS ≤ 2 | −18.9 | −0.040 |
| **LOS ≥ 3** | **−25.3** | −0.034 |
| no family | −19.4 | −0.039 |
| **family** | **−23.5** | −0.037 |
| international | −21.0 | −0.036 |
| long booking window | −20.3 | −0.037 |

**Possible feature:** **`prop_rare_x_long_trip`** — `1 / log1p(prop_count) × srch_length_of_stay`. Encodes the interaction of "rare property × long stay".

Also: **`prop_dest_safe` (DS) should be helping here** — it has a 3-way fallback that protects rare pairs. Worth checking DS importance specifically in this segment.

**Expected direction:** positive on long stays + family + international. The
"safer" smoothing of DS already partially captures this, but the interaction
with stay length should add signal.

**Risk:** LOW. Cold-start handling is well-understood; adding a stay-length
interaction is a small extension.

---

## Pattern 5 — Model overvalues promotion flag

**Segment:** all failures, strongest in low-engagement + high-candidate.

**Observed difference:** booked hotels are 11–14 percentage points LESS often
promoted than top-wrong. The model treats `promotion_flag = 1` as a strong
booking signal; in practice, users book non-promoted hotels MORE often in
failures.

**Evidence:**

| segment | Δ promotion_flag | n_failures |
|---|---:|---:|
| low_dest | −0.136 | 7,650 |
| high_cand | −0.145 | 9,722 |
| short_window | −0.127 | 8,266 |
| high_dest | −0.111 | 6,675 |
| international | −0.119 | 7,355 |

**Possible feature:** **`promotion_x_dest_engagement`** —
`promotion_flag × dest_click_rate`. The hypothesis: promotion is a *negative*
signal in low-engagement destinations (people see through "fake discount"
promotion in obscure destinations), but useful in popular destinations.

**Expected direction:** the existing `promotion_flag` should keep its
effect; this interaction should let the model DOWN-weight promotion in
low-engagement segments.

**Risk:** MEDIUM. Promotion has known sign-flip behaviour; adversarial check
recommended.

---

## Pattern 6 — Model overvalues raw location_score2 but undervalues review_rank in long stays

**Segment:** all failures, with twist on review in international/long-stay.

**Observed difference:** booked has LOWER `prop_location_score2` (−0.06)
AND worse `location2_rank_norm` (+0.16) — the model picks the high-loc2 hotel
within the query, but users sometimes choose a worse-loc2 hotel. Inverted
for review_rank: booked has slightly WORSE `review_rank_norm` (+0.03-0.05).

**Evidence:**

| segment | Δ prop_location_score2 | Δ location2_rank_norm | Δ review_rank_norm |
|---|---:|---:|---:|
| short_window | −0.068 | +0.166 | +0.045 |
| international | −0.054 | +0.140 | **+0.046** |
| high_dest | −0.073 | +0.148 | **+0.050** |
| LOS ≥ 3 | −0.057 | +0.148 | +0.028 |
| family | −0.062 | +0.167 | +0.027 |

**Possible feature:** **`is_long_stay_x_review_premium`** — for LOS ≥ 3,
upweight `review_rank_norm` (e.g. `(LOS ≥ 3) × (1 - review_rank_norm)`).
For international: same recipe with `(1 - is_domestic)`.

**Expected direction:** positive — review matters more for international and
long-stay users; model under-weights it currently.

**Risk:** LOW. Reviews are stable across train/test (no drift), and the
direction is intuitively well-grounded.

---

## Pattern 7 — Domestic users book BRANDED hotels more than the model expects

**Segment:** domestic vs international.

**Observed difference:** `prop_brand_bool` delta flips sign by domestic flag —
booked is **+0.048 more branded** in domestic, **−0.034 less branded** in international.

**Evidence:**

| segment | Δ prop_brand_bool | sign |
|---|---:|---|
| domestic | **+0.048** | booked is MORE branded |
| international | **−0.034** | booked is LESS branded |
| family | +0.025 | mixed |
| no_family | +0.018 | mixed |

**Possible feature:** **`brand_x_domestic`** — `prop_brand_bool × is_domestic`. The current model treats brand uniformly; this lets it adjust by trip type.

**Expected direction:** positive on `brand_x_domestic`, neutral on `brand × (1-is_domestic)`.

**Risk:** LOW. Brand × geography is intuitive and segment-stable.

---

## Pattern 8 — High candidate_count is the hardest segment

**Segment:** `query_hotel_count` in the top tertile (≥ 31).

**Observed difference:** highest fail rate AND largest aggregate delta on
most features. The "discriminative work" the model has to do scales badly
with candidate count.

**Evidence:**

| segment | n_failures | Δ promotion_flag | Δ comp_rate_advantage | Δ prop_count |
|---|---:|---:|---:|---:|
| low_cand | 6,294 | −0.101 | −0.047 | −10.7 |
| mid_cand | 5,239 | −0.119 | −0.091 | −19.8 |
| **high_cand** | **9,722** | **−0.145** | **−0.130** | **−27.4** |

**Possible feature:** **`query_difficulty_index`** =
`log1p(query_hotel_count) × (1 - dest_click_rate)`. Or non-linear: bucketed
candidate-count indicators (cand_count ≥ 35, ≥ 50). Allows the model to
condition its split structure on query difficulty.

**Expected direction:** the model should *change behaviour* in high-difficulty
queries (weight different features more heavily). A query-level difficulty
signal lets it do that.

**Risk:** LOW–MEDIUM. `query_hotel_count` is already a feature; combining
with `dest_click_rate` may interact weirdly with existing CP/DS, but worth
testing.

---

## Pattern 9 — V6 wins on "above-history-priced rare hotels"

**Segment:** the 797 queries where V6 ranks booked top-5 but V4 doesn't.

**Observed difference:** V6 specifically helps when the booked hotel:
- is **cheaper** vs prop history (delta = −$215 vs V6 losses)
- has **lower price_rank** (cheaper within query, Δ −0.10)
- has slightly **HIGHER review_rank_norm** (Δ +0.05)

Conversely, V6 LOSES (652 queries where V4 was right) when the booked is
PRICED ABOVE its own history (+$165 mean).

**Evidence:**

| feature | V6 wins (n=797) | V6 losses (n=652) | diff |
|---|---:|---:|---:|
| `price_vs_prop_mean` | **−51.2** | **+164.5** | **−215.7** |
| `price_rank_norm` | 0.374 | 0.474 | −0.100 |
| `review_rank_norm` | 0.392 | 0.345 | +0.047 |
| `prop_count` | 184 | 201 | −17 |

**Reading:** the V6 additions (CP, DS, label_gain variants) help when the
booked hotel is a "good value" — cheap vs prop history, low price rank, decent
reviews. They hurt when the user pays a premium. **This is consistent with
Pattern 2 (the model still overvalues cheapness; V6 just shifted the threshold
in the right direction but didn't solve the underlying issue).**

**Possible feature:** This is more of a diagnostic than a feature suggestion —
it confirms that **Pattern 2's "price-premium-segment" feature should
specifically address the V6 LOSS profile** (high price_vs_prop_mean rows).

**Expected direction:** Pattern 2's feature should convert the 652 V6 losses
into wins, without disturbing the 797 V6 wins.

**Risk:** MEDIUM. The price-premium signal is opposite-sign to most existing
features. Adversarial AUC + drift check critical before adoption.

---

## Pattern 10 — V6 helps moderately in every segment but not dramatically anywhere

**Segment:** all (uniform).

**Observed difference:** V6 has a small positive net advantage in every
segment but no segment where it shines. The net top-5 lift is +0.005 to +0.023
per segment — modest.

**Evidence:**

| segment | n_booked | net V6 top-5 advantage | V6 helps − V6 hurts |
|---|---:|---:|---:|
| mid_cand | 5,239 | **+2.29%** | +0.023 |
| mid_dest | 6,930 | **+2.06%** | +0.021 |
| high_cand | 9,722 | +1.83% | +0.018 |
| low_dest | 7,650 | +1.66% | +0.017 |
| los_le2 | 15,735 | +1.79% | +0.018 |
| domestic | 13,900 | +1.57% | +0.016 |
| international | 7,355 | +1.55% | +0.016 |
| **low_cand** | 6,294 | **+0.54%** | +0.005 |
| **los_3plus** | 5,520 | +0.91% | +0.009 |
| high_dest | 6,675 | +0.93% | +0.009 |

**Reading:** V6 helps most in mid-difficulty queries (mid_cand, mid_dest) and
the bulk of common bookings (los_le2). It barely helps in:
- **low candidate count** (already easy, hard to improve)
- **high_dest** (popular destinations where the model is already good)
- **long stays** (where V6's TE-based improvements help less)

**Possible feature:** The **next round of features should target the segments
where V6 helps LEAST**:
- Long stays (los_3plus) → Pattern 4 (rare-prop signal) + Pattern 6 (review rank)
- High-dest popular queries (already good baseline; harder to improve)
- Low_cand simple queries (already easy)

**Expected direction:** future feature gains should compound where V6 already
helps + plug the gap where V6 doesn't.

**Risk:** N/A (diagnostic).

---

## Summary table — features to test next

In priority order by expected leverage × confidence:

| # | feature | targets | risk |
|---|---|---|---|
| 1 | `price_premium_vs_prop_hist_x_short_window` | Pattern 2 + 9 (model's biggest blind spot) | LOW–MED |
| 2 | `is_long_window_x_top_quartile_price` | Pattern 3 (sign-flip pattern in long windows) | LOW |
| 3 | `prop_rare_x_long_trip` (interaction with LOS) | Pattern 4 (DS over-extension) | LOW |
| 4 | `is_long_stay_x_review_premium` | Pattern 6 (review under-weighted in long stays) | LOW |
| 5 | `brand_x_domestic` (multiplicative interaction) | Pattern 7 (domestic-only brand boost) | LOW |
| 6 | `prop_popularity_residual` or `prop_underdog_indicator` | Pattern 1 (the BIG bias, but high risk) | MED |
| 7 | `query_difficulty_index` = log(cand_count) × (1−dest_click_rate) | Pattern 8 (hard queries) | LOW–MED |
| 8 | `promotion_x_dest_engagement` | Pattern 5 (promotion sign-flip) | MED |

**Do not start with #6 even though it targets the biggest delta** — popularity
residual is the highest-risk feature for drift. Start with #1, #2, #3 (low
risk, segment-consistent patterns). If those add ≥ +0.002 each (or stack),
revisit #6.

## Files

- `pair_stats.csv` (9.1 MB) — per-failure booked + top-wrong row data (use for
  drill-downs)
- `overall_directional.csv` — features ranked by `|Δ| / |booked|`
- `segment_seg_*.csv` — per-segment booked-vs-top-wrong deltas (6 dimensions)
- `v4_vs_v6_disagreement.csv` (3.3 MB) — per-query V4 vs V6 booked position
- `v4_vs_v6_by_segment.csv` — segmented V4 vs V6 lift
- `booked_rank_profile.csv` — distribution of booked position within failures
