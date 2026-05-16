# EDA summary — 5 outputs for the report

Source: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/data/training_set_VU_DM.csv` (4,958,347 rows)

## 1+2. Click & booking rate by position
File: `position_rates.csv`. Selected positions:

```
 position  count  click_rate  book_rate
        1 199415      0.1925     0.1410
        5   9312      0.0226     0.0116
       10 173451      0.0435     0.0255
       15 167552      0.0294     0.0162
       20 153861      0.0217     0.0109
       25 140267      0.0168     0.0083
       30 123017      0.0138     0.0062
       35  87619      0.0115     0.0049
       40     66      0.0152     0.0152
```

## 3. Rates by month / week
Files: `rates_by_month.csv`, `rates_by_week.csv`.

### Monthly

```
  month  count  click_rate  book_rate
2012-11 496217      0.0448     0.0282
2012-12 479739      0.0442     0.0279
2013-01 607578      0.0438     0.0264
2013-02 593893      0.0443     0.0277
2013-03 700876      0.0448     0.0277
2013-04 642877      0.0450     0.0279
2013-05 699776      0.0457     0.0287
2013-06 737391      0.0451     0.0286
```

## 4. TE feature importance (V5 ensemble, 7 members)

```
        feature  n_models  gain_median  gain_pct_median  gain_pct_mean  split_median  split_pct_median  gain_rank_median  gain_rank_min  gain_rank_max
prop_click_rate         7    818870.72             3.88           3.97       3158.00              1.56              3.00              3              5
  prop_rel_rate         7    372734.40             1.81           1.78       2345.00              1.19             11.00             10             13
 prop_book_rate         7    165830.68             0.77           0.79       2312.00              1.16             39.00             28             41
```

Per-member detail: `te_feature_importance_per_member.csv`

## 5. prop_id count distribution

```
          stat      value
         count  129113.00
          mean      38.40
           std      81.89
           min       1.00
           10%       2.00
           25%       4.00
           50%      12.00
           75%      40.00
           90%      97.00
           95%     156.00
           99%     363.00
           max    2357.00
unique_prop_id  129113.00
    rows_total 4958347.00
         p99.9     827.89
```
