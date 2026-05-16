# V5 gap diagnostic — Phase 1 (`--quick`)

_Generated 2026-05-16 from diagnose_v5_gap.py --quick. Run dir: `diagnostics/v5_gap_20260516_101321/`_

| Reference | Local NDCG@5 | Kaggle public | local→Kaggle gap |
|---|---|---|---|
| V4 ensemble | 0.42512 | 0.42021 | **−0.00491** |
| **V5 ensemble (7 members)** | **0.42633** | **0.41943** | **−0.00690** |
| V4.2 single (stress-test only) | 0.42258 | 0.41639 | −0.00619 |
| V4 best single (anchor) | 0.42191 | — | — |

V5 improved local **+0.00121** vs V4 but lost **−0.00078** vs V4 on Kaggle. The gap widened, so this is a train↔test mismatch / val-overfit problem, not a model-capacity problem.

---

## Executive summary

**Smoking gun: adversarial AUC = 1.0000.** Train and test feature matrices are trivially separable. 85% of the adversarial classifier's importance falls on the `hotel_agg_TE_single` group — features like `country_book_rate` and `site_book_rate`. These are k-fold OOF means on train but full-train means on test → the *distributions themselves are shifted*, not just noisy.

**Cold-start is NOT the cause.** Only 0.28% of test prop_ids are unseen; cold-start rate is actually *lower* in the queries where V5 and V4 disagree.

**V5 and V4 rankings agree 99%** (Spearman 0.987, top-5 overlap 4.70/5). The disagreement is concentrated in queries with large candidate pools and international travel — exactly the segments where TE features carry the most weight.

**The most plausible causal chain:**
1. TE features (`country_book_rate`, `site_book_rate`, cross-key book_rates) have shifted distributions between train and test.
2. V5 members fit val NDCG slightly tighter than V4 (val 0.42633 > 0.42512).
3. That tighter val fit amplified attention to shifted features → degraded test ranking.
4. Ensembling 7 members trained on the same TE source did not diversify the drift; it amplified the same overfit.

---

## 1. Adversarial validation — `adversarial_auc.json`, `adversarial_feature_importance.csv`, `feature_risk_by_group.csv`

| | Value |
|---|---|
| AUC | **1.0000** |
| Interpretation | **SERIOUS** (beyond the >0.65 threshold) |
| Best iteration | 72 (converged fast) |
| Subsample | 500K train + 500K test |

### Top 20 drift features

| # | Feature | Adv gain | Group |
|---|---|---|---|
| 1 | `country_book_rate` | 5,384,699 | hotel_agg_TE_single |
| 2 | `site_book_rate` | 4,009,040 | hotel_agg_TE_single |
| 3 | `site_country_book_rate` | 841,565 | hotel_agg_TE_cross |
| 4 | `cpair_book_rate` | 407,427 | hotel_agg_TE_cross |
| 5 | `prop_country_id` | 191,711 | raw |
| 6 | `site_id` | 79,873 | raw |
| 7 | `dest_count` | 29,808 | hotel_agg_count |
| 8 | `visitor_location_country_id` | 24,497 | raw |
| 9 | `dest_mean_price` | 21,545 | hotel_agg_dest_stat |
| 10 | `dest_mean_star` | 19,376 | hotel_agg_dest_stat |
| 11 | `dest_click_rate` | 11,170 | hotel_agg_TE_single |
| 12 | `is_domestic` | 8,401 | interaction |
| 13 | `dest_mean_review` | 8,368 | hotel_agg_dest_stat |
| 14 | `prop_count` | 6,455 | hotel_agg_count |
| 15 | `dest_book_rate` | 5,342 | hotel_agg_TE_single |
| 16 | `distance_x_international` | 4,958 | interaction |
| 17 | `srch_destination_id` | 3,370 | raw |
| 18 | `orig_destination_distance` | 3,314 | raw |
| 19 | `log_distance` | 3,004 | raw |
| 20 | `prop_location_score1` | 2,711 | raw |

### Group-level adversarial vs ranker share (top 5 by Δ = adv − ranker)

| Group | n | Adv share | Ranker share | Δ |
|---|---|---|---|---|
| **hotel_agg_TE_single** | 7 | **85.0%** | 8.2% | **+76.8 pp** |
| hotel_agg_TE_cross | 5 | 11.3% | 7.0% | +4.3 pp |
| missing_flag | 5 | 0.0% | 0.1% | −0.1 pp |
| hotel_agg_TE_book_given_click | 1 | 0.0% | 0.7% | −0.7 pp |
| temporal | 4 | 0.0% | 0.7% | −0.7 pp |

Listwise (44 features, **38.8% ranker** share, **0.03% adv** share) and price (8 features, **6.8% ranker**, **0.03% adv**) are **drift-clean** — keep them, they carry signal without drift.

---

## 2. Position-bias — `position_correlation_features.csv`, `position_leaky_suspects.csv`

### Top 10 features by |Spearman ρ| with position (on train)

| Feature | ρ | Read |
|---|---|---|
| `prop_avg_position` | **+0.411** | Already flagged TEST_DROP. Confirmed. |
| `location2_rank` | +0.374 | Listwise — fine (rank inside srch) |
| `value_score_rank` | +0.311 | Listwise — fine |
| `query_hotel_count` | +0.306 | Group size — listwise — fine |
| `location2_rank_norm` | +0.259 | Listwise — fine |
| `location_total_rank` | +0.241 | Listwise — fine |
| `starrating_rank` | +0.240 | Listwise — fine |
| `prop_dest_book_rate` | −0.227 | Natural: better hotels → lower position |
| `prop_location_score2` | −0.223 | Raw quality, not leakage |
| `prop_click_rate` | −0.220 | TE — natural correlation |

Listwise features are computed *within* srch_id (pure rank), so correlation with position is structural, not leakage. They are safe.

### Position-leaky suspects (in top-30 pos × top-50 adv × top-50 ranker)

7 features triple-flagged:
1. **`prop_avg_position`** (ρ=+0.41, ranker rank #7) — drop / test
2. `prop_location_score2` (ρ=−0.22, ranker rank #1) — high signal, low position risk; keep
3. `prop_click_rate` (ρ=−0.22) — TE
4. `query_hotel_count` (ρ=+0.31) — listwise, safe
5. `prop_site_book_rate` (ρ=−0.18) — cross-key TE
6. `dest_click_rate` (ρ=−0.15, **adv rank #11**) — TE drift candidate
7. `loc2_vs_mean` (ρ=−0.17) — derivative quality

---

## 3. Cold-start enumeration — `cold_start_summary.csv`, `cold_start_by_disagreement_segment.csv`

| Metric | Value |
|---|---|
| Unseen prop_id rate | **0.28%** |
| Unseen srch_destination_id rate | 2.98% |
| Unseen (prop, dest) pair rate | 5.35% |
| Unseen (prop, site) pair rate | 5.77% |
| Test prop_count_in_train (p25 / p50 / p75) | 48 / 110 / 248 |

| Segment | n | Mean cold-start rate |
|---|---|---|
| Normal queries | 179,498 | 0.50% |
| High-disagreement (top 10%) | 20,051 | **0.13%** (LOWER, not higher) |

Cold-start is **not** the driver of V5/V4 disagreement.

---

## 4. V5 vs V4 ensemble disagreement — `v4_v5_disagreement_segments.csv`, `v4_v5_disagreement_by_query.csv`

| | Value |
|---|---|
| Reference | V4 ensemble (Kaggle 0.42021) |
| Mean top-5 overlap | **4.703 / 5** |
| Mean Spearman | **0.987** |
| High-disagreement threshold (top 10%) | mean |Δrank| ≥ 1.19 |

### Segments where V5 and V4 disagree most

| Attribute | Bucket | High-disagreement rate |
|---|---|---|
| **candidate_count** | 33–38 | **17.2%** |
| candidate_count | 30–32 | 17.1% |
| candidate_count | 19–29 | 6.1% |
| candidate_count | ≤18 | **0.2%** |
| **is_domestic** | mixed/foreign | **14.0%** |
| is_domestic | domestic | 7.8% |
| **length_of_stay** | 3+ nights | **16.3%** |
| length_of_stay | ≤2 nights | 7.9% |
| booking_window | >49 days | 11.5% |
| booking_window | ≤4 days | 9.2% |
| is_family / adults / children / rooms | all buckets | no signal |

Disagreement concentrates in **large-candidate, international, long-stay** queries — exactly where TE features have the most leverage.

### Intra-V5 disagreement — `v5_internal_disagreement.csv`

| Member | mean |Δrank| vs ensemble | max |Δrank| |
|---|---|---|
| **B10** (no_ipw, lg=0,2,20) | **1.283** | 20.86 |
| E3 (leaves=512, reg_λ=2) | 1.150 | 17.57 |
| D4 (top_120 features) | 1.043 | 18.00 |
| E4 (min_child=100) | 1.038 | 16.57 |
| B3 (ipw_clip3) | 1.009 | 17.71 |
| F13 (positive_q_only) | 1.009 | 16.00 |
| A1 (lg=0,2,12) | **0.941** | 16.57 |

B10 (the IPW-off member) is the most divergent. A1 is the closest-to-consensus.

---

## 5. Action table — prioritized

| # | Action | Evidence | Risk | Next experiment |
|---|---|---|---|---|
| 1 | **Investigate single-key TE distribution shift** | Adv AUC 1.0, top-4 drift features all TE | HIGH | Plot train vs test histograms of `country_book_rate`, `site_book_rate`. Quantify mean/std/KS by hotel/site. |
| 2 | **Ablation: drop top-4 drift TEs and retrain V5** | Top 4 features = 90%+ of adv gain | HIGH | Retrain the same 7 V5 members with `country_book_rate, site_book_rate, site_country_book_rate, cpair_book_rate` dropped. ~30 min. |
| 3 | **Heavy smoothing on single-key TEs** | High drift, train k-fold vs test full-train mismatch | HIGH | Bump prior_weight 10→40, 80, 200 in `src/features.py:hotel_aggregates`. |
| 4 | **TE-rank-within-srch_id** | Position-invariant, drift-resilient | MED | Replace raw TE value with its rank inside the query group. |
| 5 | **Drop prop_avg_position** | ρ=+0.41 with position, adv top-30 | MED | Phase 4 plan — promote ahead of feature pruning sweeps. |
| 6 | **top_120 feature pruning** | D4 was load-bearing in LOO | MED | Phase 4 plan. |
| 7 | Cross-key TE fallback hierarchy | site_country/cpair drift moderate | LOW | Phase 5 plan. |
| 8 | Cold-start indicators / fallbacks | <6% test rows affected | SKIP | Not warranted. |
| 9 | Hard-negative features | No signal in this diagnostic | DEFER | Phase 6. |

### Recommended next experiment (single, focused)

**TE-ablation V5 retrain**: same 7 members, drop the 4 top-drift TE features, equal-weight rank-average, Kaggle submit. If improvement ≥ +0.001 vs V5's 0.41943, TE drift confirmed and we narrow the next moves to TE remediation.

### `--temporal`: hold

V5 and V4 ensembles agree 99% — temporal split would confirm a secondary phenomenon while we already have a 1.0-AUC smoking gun on the primary cause. Run `--temporal` only if the TE ablation does **not** close the gap.

---

## Files produced

```
diagnostics/v5_gap_20260516_101321/
├── README.md                                ← this file
├── summary.json                             ← machine-readable
├── adversarial_auc.json
├── adversarial_feature_importance.csv       (top 50 drift features)
├── feature_risk_table.csv                   (per-feature ranker × adv)
├── feature_risk_by_group.csv                (per-group adv vs ranker share)
├── v4_v5_disagreement_by_query.csv          (per-srch_id)
├── v4_v5_disagreement_segments.csv          (segmented)
├── v5_internal_disagreement.csv             (per-V5 member vs ensemble)
├── position_correlation_features.csv        (top 30)
├── position_leaky_suspects.csv              (7 triple-flagged)
├── cold_start_summary.csv
└── cold_start_by_disagreement_segment.csv
```

_Phase 2 (`--temporal`) not yet run. Awaiting decision after TE-ablation experiment._
