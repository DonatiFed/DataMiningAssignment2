# Structural diversity batch — 20260516_212040

_Single-model batch + weighted-ensemble grid + LOO on best multi-member._

## Baselines

- V4_ANCHOR_TEMPORAL = 0.40401
- V6_LOO9_TEMPORAL   = 0.40896
- submission threshold = 0.4095
- near-miss range = [0.4092, 0.4095)

## Single-model results (sorted)
```
                model_id status    ndcg5  delta_vs_v4_anchor  delta_vs_v6_loo9    decision  best_iter  reused_from_v6  elapsed_min
          DS_regularized     ok +0.40632            +0.00231          -0.00264 STRONG_KEEP        659           False     +3.12000
       regularized_bal15     ok +0.40600            +0.00199          -0.00296        KEEP        465           False     +2.38000
          CP_regularized     ok +0.40588            +0.00187          -0.00308        KEEP        564           False     +2.75000
               lg_0_2_15     ok +0.40547            +0.00146          -0.00349        KEEP        441           False     +3.24000
low_lr_regularized_bal15     ok +0.40534            +0.00133          -0.00362        KEEP        543           False     +3.70000
               lg_0_3_15     ok +0.40430            +0.00029          -0.00466        HOLD        319           False     +2.62000
               lg_0_1_20     ok +0.40420            +0.00019          -0.00476        HOLD        476           False     +3.36000
 rank_xendcg_regularized     ok +0.40402            +0.00001          -0.00494        HOLD        604           False     +3.09000
               lg_0_1_10     ok +0.40380            -0.00021          -0.00516      REJECT        359           False     +2.84000
         ipw_clip3_bal15     ok +0.40368            -0.00033          -0.00528      REJECT        352           False     +2.74000
            no_ipw_bal15     ok +0.40339            -0.00062          -0.00557      REJECT        295            True     +0.01000
   random_upweight_bal15     ok +0.40154            -0.00247          -0.00742      REJECT        259            True     +0.01000
  booking_clf_calibrated     ok +0.38682            -0.01719          -0.02214      REJECT         70            True     +0.01000
```

## 2. Which model family helped most?

- **label_gain (A)**: best NDCG@5 = 0.40547, mean = 0.40444
- **weighting (B)**: best NDCG@5 = 0.40368, mean = 0.40287
- **regularization (C)**: best NDCG@5 = 0.40600, mean = 0.40567
- **objective (D)**: best NDCG@5 = 0.40402, mean = 0.39542
- **regularized with extra feature**: best NDCG@5 = 0.40632, mean = 0.40610

## Ensemble top-12
```
                                                                test_id  n_members    ndcg5  delta_vs_v6_loo9  v6_weight
                                      v6+rank_xendcg_regularized@w=0.10          2 +0.40933          +0.00037   +0.90000
 v6+rank_xendcg_regularized@0.1+CP_regularized@0.05+DS_regularized@0.05          4 +0.40923          +0.00027   +0.80000
 v6+rank_xendcg_regularized@0.05+CP_regularized@0.05+DS_regularized@0.1          4 +0.40918          +0.00022   +0.80000
                                      v6+rank_xendcg_regularized@w=0.05          2 +0.40918          +0.00022   +0.95000
  v6+rank_xendcg_regularized@0.1+CP_regularized@0.05+DS_regularized@0.1          4 +0.40913          +0.00017   +0.75000
 v6+rank_xendcg_regularized@0.05+CP_regularized@0.1+DS_regularized@0.05          4 +0.40912          +0.00016   +0.80000
                                               v6+CP_regularized@w=0.10          2 +0.40912          +0.00016   +0.90000
v6+rank_xendcg_regularized@0.05+CP_regularized@0.05+DS_regularized@0.05          4 +0.40912          +0.00016   +0.85000
  v6+rank_xendcg_regularized@0.1+CP_regularized@0.1+DS_regularized@0.05          4 +0.40911          +0.00015   +0.75000
                                      v6+rank_xendcg_regularized@w=0.15          2 +0.40907          +0.00011   +0.85000
                                               v6+DS_regularized@w=0.10          2 +0.40906          +0.00010   +0.90000
  v6+rank_xendcg_regularized@0.05+CP_regularized@0.1+DS_regularized@0.1          4 +0.40905          +0.00009   +0.75000
```

## 4. Best ensemble and weights

- **Test ID:** `v6+rank_xendcg_regularized@w=0.10`
- **NDCG@5:** 0.40933
- **Members added:** `rank_xendcg_regularized`
- **Weights:** `0.1000` (V6 weight = 0.9000)
- **Δ vs V6 LOO-9 (0.40896):** +0.00037
- **Δ vs V4_ANCHOR (0.40401):** +0.00532

## LOO on best MULTI-MEMBER ensemble

(Best overall is single-member; running LOO on the best 4-member combo `v6+rank_xendcg@0.1+CP@0.05+DS@0.05` = 0.40923)

```
                dropped  n_remaining    ndcg5  delta_vs_best_multi
rank_xendcg_regularized            2 +0.40885             -0.00038
         DS_regularized            2 +0.40904             -0.00019
         CP_regularized            2 +0.40920             -0.00003
```

## 1. Did structural changes beat V6 LOO-9?

**YES** — best ensemble beat V6 by **+0.00037** (0.40933 vs 0.40896). This is the first positive signal of the session.

## 3. Did any new model help as low-weight ensemble member?

**YES — 17 tests beat V6 baseline.** Best low-weight test: `v6+rank_xendcg_regularized@w=0.10` (NDCG@5 = 0.40933, Δ_v6 = +0.00037).

## Submission status: **near_miss_no_submission**

Best = 0.40933. Submission threshold = 0.4095. Near-miss range = [0.4092, 0.4095). Per user rule: 'If best is 0.40920–0.40950, report as near-miss but do not create submission unless explicitly approved.' → **NO submission CSV built.**

## 5. Recommendation

**Continue this structural direction.** This is the first batch in the session where new models actually IMPROVE V6 LOO-9 in ensemble (best Δ = +0.00037). Three independent observations support more work here:

1. **`rank_xendcg_regularized` is the standout new member.** Even though its solo NDCG (0.40402) is barely above V4_ANCHOR, it adds independent signal: `v6 + rank_xendcg @ 0.10` reaches **0.40933** (+0.00037 vs V6). The different objective (xendcg vs lambdarank) is the diversity V6 was missing.

2. **`CP_regularized` and `DS_regularized` also help** when added at small weights (both +0.00010 to +0.00016). They are stronger than their non-regularized siblings already in V6 — the V6 batch did NOT regularize CP/DS.

3. **Best 4-member ensemble** (V6 + rank_xendcg@0.10 + CP_reg@0.05 + DS_reg@0.05) = **0.40923**, just below the +0.00037 single. LOO above shows which member is the pivot. The gap from threshold (0.40950) is only 0.00017.

**Next experiments worth running:**
- Try `rank_xendcg_regularized` with more weight in the existing V6 ensemble (swap one of the weaker V6 members for it).
- Train an EXPANDED V6: 9 V6 LOO members + `rank_xendcg_regularized` as the 10th, each as a full member at 1/10 weight. May break the dilution problem entirely.
- Train `rank_xendcg_regularized` with different seeds (3-5) and rank-average across seeds — should boost the +0.00037 single-member gain further.
- If +0.0006 over V6 LOO-9 is reached (i.e. ≥ 0.40956), build the Kaggle submission from the new ensemble (retrain `rank_xendcg_regularized` on full train, splice into V6 test predictions).
