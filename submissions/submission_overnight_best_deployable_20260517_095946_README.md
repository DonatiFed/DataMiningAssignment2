# Submission OVERNIGHT BEST DEPLOYABLE — 20260517_095946

## Why this submission

The overnight final batch (`overnight_final_batch_20260517_022323`) found its #1 ensemble at temporal NDCG@5 = 0.40979, but that ensemble depended on a model from the previous structural batch that the auto-submission step couldn't retrain. This submission targets the **best ensemble whose members are ALL in this batch** (temporal NDCG@5 = 0.40971).

## Members

- **V6 LOO-9** (weight 0.8000) — the 9-member ensemble that produced Kaggle 0.42004
- **cb_rank_C_deeper** (weight 0.05) — newly trained in overnight batch
- **xendcg_conservative** (weight 0.05) — newly trained in overnight batch
- **cb_rank_A** (weight 0.05) — newly trained in overnight batch
- **xendcg_reg_seed42** (weight 0.05) — newly trained in overnight batch

## Temporal benchmarks

- V4_ANCHOR temporal:        0.40401
- V6 LOO-9 temporal:         0.40896  → Kaggle 0.42004
- This ensemble (temporal):  0.40971  → projected Kaggle ~0.4209–0.4211

## Validation
- header: `srch_id,prop_id` ✓
- rows: 4,959,183 (matches sample)
- unique searches: 199,549
- duplicates: 0
- NaN: 0

## Risk notes
- This ensemble adds **CatBoost diversity** (cb_rank_C_deeper, cb_rank_A) to V6 — a genuinely different model class than V6's all-LightGBM members. The +0.00075 local gain over V6 is the strongest the session has produced. Projected Kaggle delta is ~+0.0006 — small but positive vs V4.
- The local→Kaggle correlation has been weak this session (V6 local 0.40896 → Kaggle 0.42004 = +0.011 gap). If this ensemble follows the same ratio, projected Kaggle is 0.42085 — about +0.0006 above V4 0.42021.
