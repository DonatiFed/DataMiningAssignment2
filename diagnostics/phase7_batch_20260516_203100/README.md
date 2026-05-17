# Phase 7 batch — 20260516_203100

_Generated 2026-05-16T20:47:22.367610+00:00 • elapsed 16.4 min_

- V4_ANCHOR_TEMPORAL = 0.40401
- V6_LOO9_TEMPORAL   = 0.40896
- submission threshold = 0.4095  (+0.00054 over V6 LOO-9)

## Variant results

```
                               variant_id status   ndcg5  delta_vs_v4_anchor  delta_vs_v6_loo9 decision  best_iter  n_features  high_drift  max_drift_ratio  elapsed_min
price_premium_vs_prop_hist_x_short_window     ok 0.40295            -0.00106          -0.00601   REJECT        315         145        True          0.24017      2.70000
      is_long_window_x_top_quartile_price     ok 0.40455             0.00054          -0.00441     HOLD        392         144       False          0.02557      2.98000
                    prop_rare_x_long_trip     ok 0.40446             0.00045          -0.00450     HOLD        519         144       False              NaN      3.53000
                         brand_x_domestic     ok 0.40492             0.00091          -0.00404     HOLD        508         144       False          0.01951      3.48000
                   query_difficulty_index     ok 0.40475             0.00074          -0.00421     HOLD        522         144       False          0.03820      3.54000
```

## Ensemble results
```
                                        ensemble  n_members                                                                             members   ndcg5  recall1  recall5  mean_booked_rank  delta_vs_v6_loo9  delta_vs_v4_anchor
                                v6_loo9_baseline          1                                                                             v6_loo9 0.40896  0.24013  0.62387           6.18182           0.00000             0.00495
                                v6_loo9_plus_all          4 v6_loo9+is_long_window_x_top_quartile_price+brand_x_domestic+query_difficulty_index 0.40714  0.23963  0.62201           6.20674          -0.00182             0.00313
                               v6_loo9_plus_top2          3                        v6_loo9+brand_x_domestic+is_long_window_x_top_quartile_price 0.40696  0.23835  0.62209           6.20810          -0.00200             0.00295
                   v6_loo9_plus_brand_x_domestic          2                                                            v6_loo9+brand_x_domestic 0.40695  0.23774  0.62216           6.20778          -0.00201             0.00294
v6_loo9_plus_is_long_window_x_top_quartile_price          2                                         v6_loo9+is_long_window_x_top_quartile_price 0.40662  0.23931  0.62126           6.21672          -0.00234             0.00261
             v6_loo9_plus_query_difficulty_index          2                                                      v6_loo9+query_difficulty_index 0.40660  0.23774  0.62069           6.21300          -0.00236             0.00259
```

**Best ensemble:** `v6_loo9_baseline`  NDCG@5 = 0.40896

## Submission: **below_threshold**
