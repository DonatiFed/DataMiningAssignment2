# Temporal validation — STEP 1 results

_Run dir: `diagnostics/temporal_validation_20260516_113831/`. Wall-clock 23.7 min._

## Baselines

| Reference | NDCG@5 |
|---|---|
| V4 anchor (bal15) random val (V4 stage3) | 0.42191 |
| B3 random val (overnight) | 0.42396 |
| V4 ensemble Kaggle public | 0.42021 |
| V5 ensemble Kaggle public | 0.41943 |
| V5 TE-ablation Kaggle public | 0.41929 |

## Splits (leakage assertions all passed)

- **Temporal cutoff** (min(date_time) per srch_id, earliest 80%): **2013-05-21 16:55:42**
- Total: 4,958,347 rows / 199,795 searches
- **temporal_train**: 159,836 searches / 3,980,039 rows · 2012-11-01 → 2013-05-21
- **temporal_val**:   39,959 searches / 978,308 rows · 2013-05-21 → 2013-06-30
- **inner_train** (90% of temporal_train, seed=42): 143,853 searches / 3,583,644 rows · 2012-11-01 → 2013-05-21
- **inner_random_val** (10%): 15,983 searches / 396,395 rows · 2012-11-01 → 2013-05-21

All validation features built with `agg_source = <train subset>, is_train=False` — validation rows never feed any TE / count / aggregate. All srch_id sets disjoint; row counts sum to total.

## Results

| model | val_setup | NDCG@5 | Recall@5 | MBR | best_iter |
|---|---|---|---|---|---|
| V4_ANCHOR | random_control | **0.41587** | 0.6362 | 5.976 | 332 |
| B3        | random_control | **0.41638** | 0.6347 | 5.984 | 440 |
| V4_ANCHOR | temporal | **0.40401** | 0.6187 | 6.275 | 509 |
| B3        | temporal | **0.40398** | 0.6173 | 6.268 | 402 |

## Key deltas

| Comparison | Δ NDCG@5 |
|---|---|
| Random ↘ Temporal (V4_ANCHOR) | **−0.01186** |
| Random ↘ Temporal (B3) | **−0.01240** |
| Random-control: B3 vs V4_ANCHOR | **+0.00051** (B3 wins) |
| **Temporal: B3 vs V4_ANCHOR** | **−0.00003** (tied; V4 barely ahead) |
| Original overnight B3 − V4 anchor (random 90/10) | +0.00205 |

---

## Interpretation

### 1. Temporal shift is the dominant issue

Both models drop **~0.012 NDCG@5** going from random validation to temporal validation, despite using effectively the same training data size (random-control trains on 3.58M rows, temporal trains on 3.98M rows from the same time window). **The test set being future data costs ~0.012 in NDCG@5 directly** — far larger than any model-selection margin we've been chasing.

**Random val has been systematically overstating performance.**

### 2. B3 advantage over V4_ANCHOR is mostly random-val overfit

Tracking B3's lead over V4_ANCHOR across three validation regimes:

| Setup | B3 − V4_ANCHOR |
|---|---|
| Overnight random val (90/10 over full data) | +0.00205 |
| Random control (90/10 inside earliest 80%) | +0.00051 |
| **Temporal val (latest 20%)** | **−0.00003** |

The "B3 wins" signal **collapses to zero on temporal validation**. The +0.00205 from overnight was random-val noise / overfit. Picking B3 as the V5 ensemble center wasn't well-founded once you look at it under temporal eval.

### 3. The temporal shift is *general*, not B3-specific

Both V4_ANCHOR and B3 lose almost exactly the same amount on temporal (−0.0119 and −0.0124). The drift isn't picking on one weighting scheme over another — it hits every model the same way. This rules out "B3's clip-3 weighting is uniquely bad on future data".

### 4. Decision rules (applied)

| Rule | Applies? | Evidence |
|---|---|---|
| B3 wins random but loses temporal → B3 is random-val overfit | **PARTIAL** | Lead shrinks from +0.0005 → −0.00003 (loses by a hair) |
| Both models drop similarly on temporal → general temporal shift | **YES** | −0.0119 vs −0.0124 (within 5%) |
| Temporal ranking differs from random ranking → future selection must use temporal | **YES** | Random: B3 > V4; Temporal: V4 barely ahead |
| Random ≈ temporal → temporal split isn't the issue | **NO** | They differ by ~0.012, ~24× the model-selection margin |

---

## Implications for next steps

1. **All future model selection must include temporal validation.** Random 90/10 is misleading by ~0.012 in NDCG@5. The original overnight ensemble search (which used random val) over-promoted models that happened to fit the random val window better than the future.

2. **TE_SAFE_SINGLE (STEP 2) must be judged on temporal val first.** If TE_SAFE beats B3 on random control but not temporal, do NOT submit. The B3-vs-V4_ANCHOR story above is the cautionary case.

3. **V5's Kaggle underperformance is now largely explained.** V5 was assembled by maximizing random-val NDCG@5 (greedy + manual). That metric is biased upward ~0.012 vs future-test performance. V5 found the local maximum on a misleading metric, and Kaggle (genuine future data) punished that overfit.

4. **TE drift is real but secondary.** Adversarial AUC = 1.0 confirmed feature-distribution shift, but the TE-ablation showed dropping those features actually hurt slightly (Kaggle −0.00014). The bigger problem is *temporal generalization across the board*, not specific features. Aggressive smoothing + TE-rank-within-srch (STEP 2) may still help, but they're not the primary lever.

5. **B3 as ensemble center is NOT validated.** Until we have a temporal-validation-passing single model, V4 anchor remains the more conservative choice. **Do not** ensemble more B3-variants without temporal-val confirmation that each variant beats V4_ANCHOR on temporal.

---

## Caveats / honest limitations

- Training data is **smaller** in temporal setup (3.98M rows) than the original V4 random-val training (~4.5M). Some of the drop could be due to less data, not pure temporal shift. Best_iter also varies (509 vs 332 for V4_ANCHOR).
- Temporal val period is **short** (2013-05-21 → 2013-06-30, ~6 weeks). The Kaggle test set may be from a different time window. We can't perfectly calibrate "temporal val" ↔ "Kaggle public".
- Absolute numbers (0.404 temporal vs 0.42 Kaggle) aren't directly comparable since Kaggle's V4 ensemble was retrained on 100% of data on a different test set. **Only deltas between models on the same val set are reliable.**

---

## Recommendation

**Proceed to STEP 2 (`run_te_safe_single.py`)** with temporal-validation-first decision rules:
- Run TE_SAFE on both random_control and temporal.
- Only consider submission if TE_SAFE **beats B3 on temporal** (not just random).
- If TE_SAFE drops ~0.0124 on temporal vs random (like B3 did), that's just the general shift; only the *TE_SAFE-temporal vs B3-temporal* delta matters.

**Do not run anything else** (no ensembles, no more ablations, no submissions) until STEP 2 is in and reviewed.

---

## Files

- `temporal_results.csv` — 4-row machine-readable results.
- `split_meta.json` — split sizes, dates, leakage assertions.
- `model_V4_ANCHOR_random_control.txt`, `model_V4_ANCHOR_temporal.txt`, `model_B3_random_control.txt`, `model_B3_temporal.txt` — saved boosters.
- `../../logs/temporal_validation_20260516_113831.log` — training log.
