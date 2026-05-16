# Temporal rescore — overnight boosters

Source: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/models/overnight_20260516_003443` (85 boosters)
Predicted on: `/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/diagnostics/eval_variants/base_features_temporal_val.parquet` (978,308 rows / 39,959 searches)
Successfully predicted: 85/85

## ⚠️ LEAKAGE WARNING — read first

The overnight boosters were trained on a **random 90% split** (seed=456, val_frac=0.1) of the full training set — NOT on temporal_train. Roughly half of our temporal_val rows (dated after 2013-05-21) were inside their training data. Because of that:

- The temporal NDCG@5 values reported below are **inflated by memorisation**,   NOT directly comparable to V4_ANCHOR_TEMPORAL = 0.40401   (which was trained on temporal_train only).
- These predictions **cannot be used as ensemble members for temporal val**   — they would import the leakage into the ensemble.
- The *relative ranking* of configs may still be informative for picking   Kaggle-promising candidates, since Kaggle's test set has no overlap with   training. For ensemble-on-Kaggle, the model boosters themselves can be   reused; only their *temporal val* scores are unreliable.

Treat the numbers below as a config-ranking diagnostic, not a metric.

## Headline (leakage-affected, see warning)

- Best temporal NDCG@5: **0.53956** (B6), Δ vs V4_ANCHOR(0.40401) = **+0.13555**
- Random-split score for same config: 0.42197 (temporal − random = +0.11759)

- Random ↔ temporal correlation across 85 configs: Pearson=0.424, Spearman=0.533

## Top 20 by temporal NDCG@5

```
config_name label_gain  seed  num_leaves  n_features  random_ndcg5  temporal_ndcg5  delta_temporal_vs_anchor
         B6     0,2,15   456         400         143       0.42197         0.53956                   0.13555
         E3     0,2,15   456         512         143       0.42184         0.53274                   0.12873
        D13     0,2,15   456         400          80       0.39999         0.52499                   0.12098
         I6     0,2,15   456         400         143       0.42163         0.51524                   0.11123
         C9     0,3,15  2024         400         143       0.42145         0.51408                   0.11007
         E7     0,2,15   456         400         143       0.42251         0.51164                   0.10763
         A2     0,2,18   456         400         143       0.42115         0.50766                   0.10365
         F8     0,2,15   456         400         143       0.42166         0.50690                   0.10289
         E4     0,2,15   456         400         143       0.42291         0.50584                   0.10183
         C6     0,3,15    42         400         143       0.42106         0.50545                   0.10144
        B12     0,2,20   456         400         143       0.42056         0.50440                   0.10039
        D11     0,2,15   456         400         140       0.42145         0.50437                   0.10036
        A13     0,1,30   456         400         143       0.42169         0.50427                   0.10026
         D4     0,2,15   456         400         120       0.42228         0.50405                   0.10004
         H2     0,2,15   456         400         143       0.42124         0.50301                   0.09900
        F13     0,3,15   456         400         143       0.42253         0.50219                   0.09818
         C8     0,3,15   789         400         143       0.42110         0.50077                   0.09676
        D10     0,2,15   456         400         138       0.42056         0.50065                   0.09664
         A7     0,3,18   456         400         143       0.42087         0.49938                   0.09537
         D1     0,2,15   456         400         142       0.41928         0.49833                   0.09432
```

## Bottom 5 by temporal NDCG@5

```
config_name label_gain  seed  num_leaves  n_features  random_ndcg5  temporal_ndcg5  delta_temporal_vs_anchor
         H1              456         400         143       0.41766         0.45262                   0.04861
         I5     0,3,30   456         400         143       0.39602         0.44279                   0.03878
         F7     0,2,15   456         400         143       0.39550         0.44152                   0.03751
         I1     0,2,15   456          63         143       0.42124         0.44140                   0.03739
         E2     0,2,15   456         128         143       0.42083         0.44136                   0.03735
```

## Files

- `temporal_rescore.csv` — full 85-row table, all metrics
- `temporal_pred_<config>.npy` — top 30 predictions, for downstream ensemble experiments
