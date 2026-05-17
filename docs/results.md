# Results timeline

Detailed per-experiment NDCG@5 scores. See `CHANGELOG.md` for the
version-by-version narrative and `final_kaggle_results.md` for the
authoritative scoreboard.

## Kaggle public leaderboard (final)

| version | submission file | Kaggle public NDCG@5 | Δ vs V4 |
|---|---|---:|---:|
| **V4 ensemble** ★ | `submissions/submission_v4_20260515_151132.csv` | **0.42021** | — |
| V9 overnight | `submissions/submission_overnight_best_deployable_20260517_095946.csv` | 0.42012 | −0.00009 |
| V6 LOO-9 | `submissions/submission_v6_loo9_20260516_184304.csv` | 0.42004 | −0.00017 |
| V11 MEGA-BAG | `submissions/submission_FINAL_megabag_25equal_*.csv` | 0.42003 | −0.00018 |
| V11 SAFE-PUSH | `submissions/submission_FINAL_safepush_v75_6div_*.csv` | 0.41995 | −0.00026 |
| V5 ensemble | `submissions/submission_v5_ensemble_20260516_094741.csv` | 0.41943 | −0.00078 |
| V10 adversarial reweight | `submissions/submission_adv_reweight_20260517_111219.csv` | 0.41903 | −0.00118 |
| V4.2 (Phase 2 best) | `submissions/submission_phase2_best_20260515_225726.csv` | 0.41639 | −0.00382 |

★ = highest-scoring submission, used as the team's primary upload.
Submitted as **team_80** for the
[DMT 2026 — 2nd Assignment](https://www.kaggle.com/competitions/dmt-2026-2nd-assignment).
Top of the public leaderboard at submission time: approximately 0.46.
Gap from V4: +0.04.

## V6 — single-member NDCG@5 on temporal val

| member | type | label_gain | weight | extra_feature | NDCG@5 | Δ anchor | best_iter |
|---|---|---|---|---|---:|---:|---:|
| **CP** | lambdarank | 0,1,15 | ipw | OOF pos-adj click | **0.40533** | **+0.00132** | 486 |
| DS | lambdarank | 0,1,15 | ipw | smoothed prop×dest | 0.40495 | +0.00094 | 439 |
| lambdarank_base | lambdarank | 0,1,31 | ipw | — | 0.40486 | +0.00085 | 656 |
| lambdarank_click3 | lambdarank | 0,3,31 | ipw | — | 0.40430 | +0.00029 | 522 |
| lambdarank_bal15 | lambdarank | 0,1,15 | ipw | — | 0.40401 | 0 | 509 |
| lambdarank_book50 | lambdarank | 0,1,50 | ipw | — | 0.40348 | −0.00053 | 400 |
| lambdarank_noipw | lambdarank | 0,1,15 | none | — | 0.40339 | −0.00062 | 295 |
| rank_xendcg | xendcg | 0,1,15 | ipw | — | 0.40216 | −0.00185 | 663 |
| lambdarank_randup | lambdarank | 0,2,25 | randup | — | 0.40154 | −0.00247 | 259 |
| booking_clf | binary | — | none | — | 0.38682 | −0.01719 | 70 |

Anchor: `V4_ANCHOR_TEMPORAL = 0.40401`.

## V6 — rank-average ensembles

Built by rank-averaging within `srch_id` (equal weights). Each per-row
score is converted to within-`srch_id` rank, ranks are averaged across
members, then sorted ascending.

| ensemble | n members | NDCG@5 | Δ anchor | Δ quick (0.40679) |
|---|---:|---:|---:|---:|
| **9-member LOO-best** (drop booking_clf) | 9 | **0.40896** | **+0.00495** | **+0.00217** |
| v4_only + CP + DS | 10 | 0.40841 | +0.00440 | +0.00162 |
| v4_only + CP | 9 | 0.40793 | +0.00392 | +0.00114 |
| v4_only + DS | 9 | 0.40768 | +0.00368 | +0.00090 |
| v4_only | 8 | 0.40753 | +0.00352 | +0.00074 |
| above_median (5) | 5 | 0.40691 | +0.00290 | +0.00012 |
| CP + DS | 2 | 0.40604 | +0.00203 | −0.00075 |

"Quick" benchmark = the 3-model `V4_ANCHOR + CP + DS` rank-average from
the pre-gym session (NDCG@5 = 0.40679).

## V6 — leave-one-out on the 10-member ensemble

Dropping each member, scoring the remaining 9-member rank-average:

| dropped member | n remaining | NDCG@5 | Δ vs 10-member |
|---|---:|---:|---:|
| booking_clf | 9 | **0.40896** | **+0.00056** ← only positive |
| lambdarank_book50 | 9 | 0.40831 | −0.00010 |
| lambdarank_base | 9 | 0.40809 | −0.00031 |
| lambdarank_click3 | 9 | 0.40799 | −0.00041 |
| lambdarank_bal15 | 9 | 0.40799 | −0.00042 |
| DS | 9 | 0.40793 | −0.00048 |
| lambdarank_noipw | 9 | 0.40780 | −0.00061 |
| CP | 9 | 0.40769 | −0.00072 |
| lambdarank_randup | 9 | 0.40748 | −0.00093 |
| rank_xendcg | 9 | 0.40740 | −0.00101 |

`booking_clf` was the only member actively hurting the ensemble. Removing
it lifted the ensemble to its LOO-best (which is what V6 submitted).

## Single-variant evaluations on temporal val (12 runs)

From `pipelines/evaluate_variant.py`. Each is a single-feature addition
or modification on top of the V4_ANCHOR base.

| variant | NDCG@5 | Δ vs anchor | decision |
|---|---:|---:|---|
| `prop_click_rate_pos_adj_s40_oof` (CP) | 0.40533 | +0.00132 | KEEP |
| `prop_price_zscore_clipped_x_dest_click` | 0.40499 | +0.00098 | HOLD |
| `prop_dest_book_rate_safe` (DS) | 0.40495 | +0.00094 | HOLD |
| `te_rank` (12 pct-rank columns) | 0.40466 | +0.00065 | HOLD |
| `prop_click_posadj_plus_prop_dest_safe` (combo) | 0.40466 | +0.00065 | HOLD ←anti-additive |
| `price_vs_mean_x_dest_click` | 0.40457 | +0.00056 | HOLD |
| `prop_rel_rate_pos_adj_s40_oof` | 0.40416 | +0.00015 | HOLD |
| `prop_book_rate_pos_adj_s40_oof` | 0.40387 | −0.00014 | REJECT |
| `price_vs_prop_mean_x_dest_click` | 0.40312 | −0.00089 | REJECT ← tail-driven overfit |
| `prop_click_posadj_plus_price_dest` (combo) | 0.40314 | −0.00087 | REJECT ←anti-additive |
| `prop_click_posadj_plus_te_rank` (combo) | 0.40241 | −0.00160 | REJECT ←anti-additive |
| `drop_prop_avg_position` | 0.40166 | −0.00235 | REJECT |
| `prop_click_rate_pos_adj_s40` (leaky, no OOF) | 0.37378 | −0.03023 | REJECT |
| `replace_prop_click_with_posadj` | 0.40385 | −0.00016 | REJECT |

Decision thresholds: REJECT (Δ < 0), HOLD (0 ≤ Δ < +0.001), KEEP (+0.001 ≤ Δ < +0.002), STRONG_KEEP (Δ ≥ +0.002).

## Drift across temporal train/val for V6 candidate features

`|Δμ|/σ` = absolute difference in mean across train vs val, normalised by
train std. Lower is better; > 0.1 starts to be a concern for Kaggle.

| feature | `|Δμ|/σ` | comment |
|---|---:|---|
| `prop_dest_book_rate_safe` | **0.004** | cleanest drift of any positive variant |
| `prop_click_rate_pos_adj_s40_oof` | 0.018 | clean |
| `prop_book_rate_pos_adj_s40_oof` | 0.016 | clean (but feature itself was REJECT) |
| `prop_rel_rate_pos_adj_s40_oof` | 0.017 | clean |
| `te_rank` (12 cols) | ≤ 0.031 | all clean |
| `price_vs_mean_x_dest_click` | ~0 | trivially zero (mean-centered parents) |
| `prop_price_zscore_clipped_x_dest_click` | 0.208 | high — risky on Kaggle |
| `price_vs_prop_mean_x_dest_click` | tail-heavy | distribution shifts heavily train→val |

## Random↔temporal correlation (overnight rescore)

The 85 overnight V4-style boosters were rescored on temporal val
(`scripts/temporal_rescore_overnight.py`). Because they were trained on a
random 90% split (seed=456), ~half of the temporal val rows were in their
training set — so the absolute temporal NDCG numbers are inflated by
memorisation. The *relative* config ranking is still informative:

- 85 configs, random val NDCG range: 0.396 – 0.424
- 85 configs, leaked temporal NDCG range: 0.441 – 0.540
- Pearson(random, temporal): **0.42**
- Spearman(random, temporal): **0.53**

Best on random ≠ best on temporal. Config selection by random-val rank
alone is unreliable for predicting Kaggle behaviour.
