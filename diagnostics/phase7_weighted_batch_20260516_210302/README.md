# Phase 7 weighted-ensemble batch — 20260516_210302

_Generated 2026-05-16T21:09:56.050645+00:00 • elapsed 6.9 min_

## Baselines

- V4_ANCHOR_TEMPORAL = 0.40401
- V6_LOO9_TEMPORAL   = 0.40896
- submission threshold = 0.4095  (+0.00054 over V6)

## 1. Did any Phase 7 member help when weighted lightly?

**Answer:** NO — best weighted ensemble = 0.40894, Δ vs V6 LOO-9 = -0.00002.

Top 10 Phase A weighted ensembles:
```
                                                                  test_id   ndcg5  delta_vs_v6_loo9  v6_weight  total_added_weight
                                         v6+query_difficulty_index@w=0.05 0.40894          -0.00002    0.95000             0.05000
                            v6+is_long_window_x_top_quartile_price@w=0.05 0.40889          -0.00007    0.95000             0.05000
                                               v6+brand_x_domestic@w=0.05 0.40889          -0.00007    0.95000             0.05000
v6+query_difficulty_index@0.050+is_long_window_x_top_quartile_price@0.050 0.40881          -0.00015    0.90000             0.10000
                            v6+is_long_window_x_top_quartile_price@w=0.15 0.40880          -0.00016    0.85000             0.15000
                                          v6+prop_rare_x_long_trip@w=0.05 0.40876          -0.00020    0.95000             0.05000
v6+query_difficulty_index@0.100+is_long_window_x_top_quartile_price@0.050 0.40876          -0.00020    0.85000             0.15000
                            v6+is_long_window_x_top_quartile_price@w=0.10 0.40872          -0.00024    0.90000             0.10000
                                         v6+query_difficulty_index@w=0.15 0.40872          -0.00024    0.85000             0.15000
                                         v6+query_difficulty_index@w=0.10 0.40867          -0.00029    0.90000             0.10000
```

## 2. Best ensemble and weights

- **Test ID:** `v6+combo_brand_dom_x_query_diff@w=0.05`
- **NDCG@5:** 0.40898
- **Members added:** `combo_brand_dom_x_query_diff`
- **Weights:** `0.0500` (V6 weight = 0.9500)
- **Δ vs V6 LOO-9 (0.40896):** +0.00002
- **Δ vs V4_ANCHOR (0.40401):** +0.00497

## 3. Leave-one-out on best Phase A ensemble

_(LOO not run — best was a single-member ensemble.)_

## 4. Phase B — combo models
```
                                combo_id status   ndcg5  delta_vs_v4_anchor  delta_vs_v6_loo9  best_iter  n_features
            combo_brand_dom_x_query_diff     ok 0.40429             0.00028          -0.00467        464         145
combo_long_window_price_x_rare_long_trip     ok 0.40417             0.00016          -0.00479        407         145
```

Top 5 Phase B weighted ensembles:
```
                                           test_id   ndcg5  delta_vs_v6_loo9  v6_weight
            v6+combo_brand_dom_x_query_diff@w=0.05 0.40898           0.00002    0.95000
v6+combo_long_window_price_x_rare_long_trip@w=0.05 0.40886          -0.00010    0.95000
v6+combo_long_window_price_x_rare_long_trip@w=0.10 0.40866          -0.00030    0.90000
v6+combo_long_window_price_x_rare_long_trip@w=0.15 0.40864          -0.00032    0.85000
            v6+combo_brand_dom_x_query_diff@w=0.10 0.40848          -0.00048    0.90000
```

## 5. Submission status: **below_threshold_no_submission**

## 6. Recommendation

Phase 7 weighted ensemble beat V6 by only +0.00002 — within noise. **Recommended:** STOP Phase 7. Move to structural changes (loss-side position handling, heterogeneous learners, adversarial reweighting). See docs/next_steps.md.
