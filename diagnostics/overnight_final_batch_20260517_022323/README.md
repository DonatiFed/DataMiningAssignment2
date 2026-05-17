# Overnight final batch — 20260517_022323

_Generated 2026-05-17T04:39:34.249955+00:00 • elapsed 136.2 min_

## Baselines
- V4_ANCHOR temporal: 0.40401
- V6 LOO-9 temporal: 0.40896
- structural best (V6 + rank_xendcg_regularized@0.10): 0.40933
- submission threshold: 0.4095

## Single-model results (top 20 by NDCG)
```
             model_id       framework  phase              status    ndcg5  delta_vs_v4_anchor  delta_vs_v6_loo9   best_iter  elapsed_min
    reg_bal15_seed123            lgbm      5                  ok +0.40736            +0.00335          -0.00160  +635.00000     +2.99000
       ds_reg_seed456            lgbm      5                  ok +0.40707            +0.00306          -0.00189  +586.00000     +2.86000
       ds_reg_seed123            lgbm      5                  ok +0.40651            +0.00250          -0.00245  +653.00000     +3.07000
       cp_reg_seed456            lgbm      5                  ok +0.40647            +0.00246          -0.00249  +520.00000     +2.60000
       cp_reg_seed123            lgbm      5                  ok +0.40622            +0.00221          -0.00274  +486.00000     +2.47000
    reg_bal15_seed456            lgbm      5                  ok +0.40621            +0.00220          -0.00275  +443.00000     +2.29000
        cp_reg_seed42            lgbm      5                  ok +0.40597            +0.00196          -0.00299  +417.00000     +2.23000
     cb_rank_C_deeper        catboost      3                  ok +0.40578            +0.00177          -0.00318 +1989.00000    +22.95000
     reg_bal15_seed42            lgbm      5                  ok +0.40529            +0.00128          -0.00367  +406.00000     +2.16000
        ds_reg_seed42            lgbm      5                  ok +0.40509            +0.00108          -0.00387  +431.00000     +2.31000
            cb_rank_A        catboost      3                  ok +0.40451            +0.00050          -0.00445 +1541.00000    +17.97000
  xendcg_conservative            lgbm      1                  ok +0.40414            +0.00013          -0.00482  +637.00000     +2.47000
   xendcg_reg_seed123            lgbm      1                  ok +0.40355            -0.00046          -0.00541  +582.00000     +2.76000
    xendcg_reg_seed42            lgbm      1                  ok +0.40307            -0.00094          -0.00589  +635.00000     +2.92000
   xendcg_reg_seed789            lgbm      1                  ok +0.40281            -0.00120          -0.00615  +528.00000     +2.57000
   xendcg_reg_seed456            lgbm      1                  ok +0.40226            -0.00175          -0.00670  +654.00000     +2.99000
  xendcg_reg_seed2024            lgbm      1                  ok +0.40226            -0.00175          -0.00670  +602.00000     +2.98000
cb_rank_B_regularized        catboost      3                  ok +0.39926            -0.00475          -0.00970 +1974.00000     +6.56000
       cb_booking_clf catboost_binary      4                  ok +0.38907            -0.01494          -0.01989  +485.00000     +1.63000
           xgb_rank_A             xgb      2 failed:XGBoostError      NaN                 NaN               NaN         NaN          NaN
```

### Failed models (5)
```
              model_id  framework              status                                                                                                                                                                                                                                                                                                          error
            xgb_rank_A        xgb failed:XGBoostError [02:40:09] /__w/xgboost/xgboost/src/data/data.cc:1196: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`\nStack trace:\n  [bt] (0) /home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/.venv/lib/python3.12/site-packages/xgboost/lib/l
xgb_rank_B_regularized        xgb failed:XGBoostError [02:40:13] /__w/xgboost/xgboost/src/data/data.cc:1196: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`\nStack trace:\n  [bt] (0) /home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/.venv/lib/python3.12/site-packages/xgboost/lib/l
    xgb_rank_C_shallow        xgb failed:XGBoostError [02:40:17] /__w/xgboost/xgboost/src/data/data.cc:1196: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`\nStack trace:\n  [bt] (0) /home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/.venv/lib/python3.12/site-packages/xgboost/lib/l
       xgb_booking_clf xgb_binary failed:XGBoostError [03:27:49] /__w/xgboost/xgboost/src/data/data.cc:1196: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`\nStack trace:\n  [bt] (0) /home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/.venv/lib/python3.12/site-packages/xgboost/lib/l
         xgb_click_clf xgb_binary failed:XGBoostError [03:27:53] /__w/xgboost/xgboost/src/data/data.cc:1196: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`\nStack trace:\n  [bt] (0) /home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/.venv/lib/python3.12/site-packages/xgboost/lib/l
```

## Ensemble top-15
```
                                                                                        test_id  n_members    ndcg5  delta_vs_v6_loo9  v6_weight
  v6+pool_cb_rank_C_deeper+struct_rank_xendcg_regularized+cb_rank_A+xendcg_reg_seed42@0.075each          5 +0.40979          +0.00083   +0.70000
            v6+pool_cb_rank_C_deeper+struct_rank_xendcg_regularized+xendcg_reg_seed42@0.075each          4 +0.40972          +0.00076   +0.77500
              v6+pool_cb_rank_C_deeper+xendcg_conservative+cb_rank_A+xendcg_reg_seed42@0.05each          5 +0.40971          +0.00075   +0.80000
                              v6+pool_xendcg_conservative+cb_rank_A+xendcg_reg_seed42@0.075each          4 +0.40968          +0.00072   +0.77500
                   v6+pool_struct_rank_xendcg_regularized+cb_rank_A+xendcg_reg_seed42@0.075each          4 +0.40968          +0.00072   +0.77500
                                 v6+pool_cb_rank_C_deeper+cb_rank_A+xendcg_reg_seed42@0.075each          4 +0.40967          +0.00071   +0.77500
                                                v6+pool_xendcg_conservative+cb_rank_A@0.075each          3 +0.40965          +0.00069   +0.85000
                       v6+pool_cb_rank_C_deeper+xendcg_conservative+xendcg_reg_seed42@0.075each          4 +0.40965          +0.00069   +0.77500
                    v6+pool_struct_rank_xendcg_regularized+cb_rank_A+xendcg_reg_seed42@0.05each          4 +0.40965          +0.00069   +0.85000
             v6+pool_cb_rank_C_deeper+struct_rank_xendcg_regularized+xendcg_reg_seed42@0.05each          4 +0.40964          +0.00068   +0.85000
                        v6+pool_cb_rank_C_deeper+xendcg_conservative+xendcg_reg_seed42@0.05each          4 +0.40962          +0.00066   +0.85000
             v6+pool_cb_rank_C_deeper+xendcg_conservative+cb_rank_A+xendcg_reg_seed42@0.075each          5 +0.40959          +0.00063   +0.70000
                               v6+pool_xendcg_conservative+cb_rank_A+xendcg_reg_seed42@0.05each          4 +0.40959          +0.00063   +0.85000
v6+pool_xendcg_conservative+struct_rank_xendcg_regularized+cb_rank_A+xendcg_reg_seed42@0.05each          5 +0.40958          +0.00062   +0.80000
v6+pool_cb_rank_C_deeper+xendcg_conservative+struct_rank_xendcg_regularized+cb_rank_A@0.075each          5 +0.40956          +0.00060   +0.70000
```

## 6. Best ensemble and weights

- **Test ID:** `v6+pool_cb_rank_C_deeper+struct_rank_xendcg_regularized+cb_rank_A+xendcg_reg_seed42@0.075each`
- **NDCG@5:** 0.40979
- **Members:** `cb_rank_C_deeper+struct_rank_xendcg_regularized+cb_rank_A+xendcg_reg_seed42`
- **Weights:** `0.0750,0.0750,0.0750,0.0750` (V6 = 0.7000)
- **Δ vs V6 LOO-9:** +0.00083
- **Δ vs structural best (0.40933):** +0.00046

## Q&A

### 1. Did rank_xendcg seeds improve the previous +0.00037?
Best xendcg seed: 0.40414, mean: 0.40301

### 2. Did XGBoost add useful diversity?
5 XGB models trained, best NDCG: nan

### 3. Did CatBoost add useful diversity?
4 CatBoost models trained, best NDCG: 0.40578

### 4. Did binary classifiers help or hurt?
3 classifiers trained, best NDCG: 0.38907

### 5. Did CP/DS/regularized seeds help?
14 seed-expansion models, best NDCG: 0.40736

### 7. Submission candidates ready

- **best_overall:** status=skipped_struct_dep  NDCG@5=nan  path=``
- **best_diverse:** status=skipped_struct_dep  NDCG@5=nan  path=``
- **best_conservative:** status=ok  NDCG@5=0.40943  path=`/home/ubuntu/DataMiningTechniques/Assignment2/DataMiningAssignment2/submissions/submission_overnight_best_conservative_20260517_022323.csv`

### 8. Recommended upload order tomorrow morning

1. `submission_overnight_best_conservative_20260517_022323.csv` (temporal 0.40943)

### 9. What to do next if none beats V4

Best temporal ensemble reached 0.40979 — worth uploading. If Kaggle disappoints, the next levers are adversarial sample reweighting and hard-negative mining as features.