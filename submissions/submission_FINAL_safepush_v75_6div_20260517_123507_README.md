# SAFE-PUSH — final submission (20260517_123507)

## File
`submission_FINAL_safepush_v75_6div_20260517_123507.csv`

## Validation
- header: srch_id,prop_id ✓
- rows: 4,959,183 (matches sample)
- 0 NaN, 0 duplicates

## Method

NO retraining — pure rank-average of saved test predictions.

### SAFE-PUSH (sub 2 — closer to proven winner)
V6 LOO-9 @ 0.75 + 6 diversifiers @ 0.0417 each.
Diversifiers: cb_rank_C_deeper, cb_rank_A, xendcg_conservative,
xendcg_reg_seed42/123/456.
Slight extension of overnight winner (V6@0.80 + 4 div = Kaggle 0.42012).
Adds 2 more diversifiers and slightly reduces V6 weight for variance reduction.

### MEGA-BAG (sub 1 — wild card)
All 25 trained models, equal rank-average. V6 effective weight = 9/25 = 36%.
Pure diversity bet — no special V6 backbone preference.

## Expected Kaggle
- SAFE-PUSH: 0.4205–0.4220 (most likely Kaggle-improver vs 0.42012)
- MEGA-BAG: 0.418–0.422 (high variance hedge)
- Stretch best case: 0.422–0.425 if either surprises
