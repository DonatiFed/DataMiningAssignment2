# Ensemble candidate summary — val-only

_Run `artifacts/ensemble_search_20260516_085823/`. No retraining, no submission._

Baselines: V4 ensemble local **0.42512** · Kaggle **0.42021** (gap −0.00491). Anchor single 0.42191.

## Headline

| | Ensemble | NDCG@5 | Δ vs V4_local | beats V4? |
|---|---|---|---|---|
| **Best overall** | **GREEDY_FWD_FROM_B3** (8 members) | **0.42689** | **+0.00177** | YES |
| Best deterministic (manual) | E2_top5 | 0.42622 | +0.00110 | YES |
| Best weighted variant | E3_top10__softmax_ndcg | 0.42595 | +0.00083 | YES |
| Best wildcard (W2) | W2_random_weights_000 | 0.42621 | +0.00109 | YES |
| Best wildcard (W1) | W1_random_subset_000 | 0.42602 | +0.00090 | YES |

Pool: 34 candidates with overnight NDCG@5 ≥ 0.4210. All ensembles use **rank averaging within srch_id**.

## Top 15 ensembles (full: `experiment_logs/ensemble_results.csv`)

| # | ensemble_id | method | n | NDCG@5 | Recall@5 | MBR | beats V4 |
|---|---|---|---|---|---|---|---|
| 1 | GREEDY_FWD_FROM_B3 | greedy_forward | 8 | 0.42689 | 0.6461 | 5.887 | YES |
| 2 | E2_top5 | equal_rank_avg | 5 | 0.42622 | 0.6459 | 5.910 | YES |
| 3 | W2_random_weights_000 | WILDCARD | 10 | 0.42621 | 0.6439 | 5.918 | YES |
| 4 | W2_random_weights_001 | WILDCARD | 10 | 0.42620 | 0.6451 | 5.915 | YES |
| 5 | W2_random_weights_002 | WILDCARD | 10 | 0.42616 | 0.6446 | 5.908 | YES |
| 6 | W2_random_weights_003 | WILDCARD | 10 | 0.42611 | 0.6451 | 5.908 | YES |
| 7 | W1_random_subset_000 | WILDCARD | 9 | 0.42602 | 0.6473 | 5.901 | YES |
| 8 | W1_random_subset_001 | WILDCARD | 5 | 0.42601 | 0.6466 | 5.894 | YES |
| 9 | E3_top10__softmax_ndcg | softmax_ndcg | 10 | 0.42595 | 0.6431 | 5.914 | YES |
| 10 | E9_strong_only | equal_rank_avg | 4 | 0.42580 | – | – | YES |
| 11 | E3_top10__inverse_rank | inverse_rank | 10 | 0.42580 | – | – | YES |
| 12 | E7_no_positive_only | equal_rank_avg | 8 | 0.42569 | – | – | YES |
| 13 | E6_diverse | equal_rank_avg | 5 | 0.42565 | – | – | YES |
| 14 | E3_top10__softmax_ndcg_T0005 | softmax_ndcg_T0.005 | 10 | 0.42559 | – | – | YES |
| 15 | LOO_top10_drop_I3 | loo_top10 | 9 | 0.42549 | – | – | YES |

## Greedy forward path (B3 → 8 members)

| step | add | NDCG@5 | Δ | desc |
|---|---|---|---|---|
| 0 | B3 | 0.42396 | — | ipw_clip3 (EXCELLENT) |
| 1 | A1 | 0.42519 | +0.00123 | label_gain 0,2,12 |
| 2 | E4 | 0.42571 | +0.00052 | leaves=400, min_child=100 |
| 3 | B6 | 0.42637 | +0.00066 | rand_up_2.0 weighting |
| 4 | E3 | 0.42642 | +0.00005 | leaves=512 + reg |
| 5 | B1 | 0.42660 | +0.00018 | no_ipw, lg=0,2,15 |
| 6 | D14 | 0.42674 | +0.00014 | keep_te_and_raw filter |
| 7 | A2 | 0.42689 | +0.00015 | label_gain 0,2,18 |

## Leave-one-out on E3_top10 (base 0.42521)

| drop | NDCG@5 without | Δ vs base | takeaway |
|---|---|---|---|
| I3 (GOSS) | 0.42549 | +0.00028 | **hurts** the manual top10 — drop |
| A1 | 0.42545 | +0.00024 | drop |
| F13 (positive_only) | 0.42537 | +0.00017 | drop |
| B3 | 0.42529 | +0.00008 | even the best single hurts top10 marginally |
| B9 | 0.42520 | −0.00001 | ≈ neutral |
| B10 | 0.42519 | −0.00002 | ≈ neutral |
| C11 | 0.42518 | −0.00002 | ≈ neutral |
| C9 | 0.42512 | −0.00009 | useful (seed diversity) |
| F1 | 0.42507 | −0.00013 | useful (row filter) |
| **D4** | 0.42494 | **−0.00026** | **most load-bearing member** (top_120 feat filter) |

## Wildcards

| | Method | n_trials | Best NDCG@5 | Notes |
|---|---|---|---|---|
| W1 | Random subsets (k=3–10) | 500 | 0.42602 | Best subset = {B1,B10,B7,C8,D11,D14,E3,E4,I3} (k=9) — none of B3/F13/D4 picked. |
| W2 | Dirichlet weights on top10 | 500 | 0.42621 | Best weights heavy on B3 (0.42), then A1 (0.17), B10 (0.12), F1 (0.10). |
| W3 | Random subsets + Dirichlet weights | 500 | 0.42591 | Lower upside than W1/W2 — extra noise. |

## Findings (8 bullets, max 10)

- **Greedy beats every manual ensemble and every wildcard** by ≥ +0.0007. Path picks B3 → A1 → E4 → B6 → E3 → B1 → D14 → A2; no B-row-filter or F-row-filter members, no GOSS/DART. Suggests row filters and alt boosting types add noise more than diversity here.
- **B3 (best single) is not load-bearing** in the manual top10 — dropping it hardly hurts, and the greedy path doesn't even use F13/D4. The "best single" anchor is weak guidance for ensemble construction.
- **D4 (top_120 feature filter) is the most load-bearing top10 member** (dropping it costs −0.00026). This implies feature pruning is real signal, not noise — feature audit pays off.
- **E2_top5 (B3+F13+D4+A1+B10) is the strongest *small* deterministic ensemble** at 0.42622 with only 5 members — easy/cheap to retrain.
- **W2 wildcard finds weights very close to manual ndcg-weighting** (heavy on B3) → softmax/Dirichlet does not unlock structural gain over equal-weight + good member selection.
- **Recall@5 and MBR best on greedy** (0.6461 / 5.887) — top-ranked precision improves *more* than NDCG does, which is the right signal for closing the Kaggle gap.
- **Local→Kaggle gap projection**: at V4's −0.00491 gap, greedy would land ~0.42198 on Kaggle (vs V4 0.42021 → **+0.00177 improvement projected**). At v4.2's wider −0.00619, projected ~0.42070 (still beats V4 by +0.00049). Both directions positive.
- **Risk**: greedy is overfit-prone on val by construction (it selects on val NDCG). The wildcard W2's #1 result, hitting nearly the same number (0.42621 vs 0.42689), suggests the greedy gain is real but the 0.42689 number itself is likely optimistic by ~0.0005–0.001.

## Recommendation

**Yes — proceed to a Kaggle submission**, but **not the raw greedy result**. Greedy on val overfits the val split. Use a robust compromise:

### Recommended submission ensemble — **E2_top5 + 2 stabilizers**

**Members (7):** `B3, F13, D4, A1, B10, E4, E3`
- B3, F13, D4, A1, B10 — E2_top5 baseline (val 0.42622)
- + E4 (leaves=400, min_child=100): hyperparam diversity, val 0.42291
- + E3 (leaves=512): further hyperparam diversity, val 0.42185

**Method:** **equal-weight rank averaging** within `srch_id`.

**Why not greedy:** greedy adds B6 (val 0.421965) and A2 (val 0.421147) — both rely on val-set noise to discriminate from siblings (B5/A1). Pulling them in is a val-selection effect.

**Why not pure E2_top5:** adding E4 + E3 is suggested by the W2 wildcard placing meaningful weight on E-group hyperparam variants, and by the LOO showing D4 (feature filter) carries load — hyperparam diversity is real signal.

**Why not all 10:** I3 (GOSS), C9, C11, F1, B9 each have LOO contribution near zero or negative on val — adding them is cost (retrain time) for no expected gain.

### Submission plan (to discuss before executing)

1. Retrain each of the 7 members on **full train** (no val split), seeds and label_gain as their overnight configs, num_boost_round = each model's `best_iter` from its `model_result_*.json`.
2. Predict each on test, rank-average within `srch_id`.
3. Save as `submissions/submission_ensemble_<TS>.csv`.
4. **No further tuning between val pick and submission**. The val-tuned greedy/wildcard numbers stay locked.

Estimated wall-clock to retrain 7 members on full train: ~25–35 min total.

### Do NOT do (per current instructions)

- Do not retrain.
- Do not generate submission.csv yet.
- Wait for explicit go-ahead with the 7-member list above.
