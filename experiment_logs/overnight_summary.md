# Overnight run summary — `overnight_20260516_003443`

_Generated 2026-05-16 from saved artifacts (no retraining)._

Anchor: V4 best single 0.42191 · V4 ensemble local 0.42512 · Kaggle 0.42021.

## Counts

| Bucket | N |
|---|---|
| Completed | 85 |
| Skipped (`skip_reason`) | 7 — F11, G1, G2, G3, G4, H6, H7 |
| Interrupted (DART/I-tail) | 7 — I7, I8, I9, I10, I11, I12, I13 |
| Total configs | 99 |

Verification: `model_result_*.json` files = 85, real-overnight rows in `model_results.csv` = 85 (1:1).

## Categorization (NDCG@5 thresholds: ≥0.423 EXCELLENT, ≥0.422 STRONG, ≥0.421 CANDIDATE, ≥0.418 NEUTRAL, <0.418 WEAK, ≥0.424 POSSIBLE_OVERFIT)

| EXCELLENT | STRONG | CANDIDATE | NEUTRAL | WEAK | POSSIBLE_OVERFIT |
|---|---|---|---|---|---|
| 1 | 6 | 27 | 38 | 13 | 0 |

## Best per group

| Group | Best | NDCG@5 | Kept | Theme |
|---|---|---|---|---|
| A (label-gain, 13) | A1 | 0.42205 | STRONG | `0,2,12` |
| B (weighting, 12) | **B3** | **0.42396** | **EXCELLENT** | `ipw_clip3` × `lg=0,2,15` |
| C (seeds, 12) | C9 | 0.42145 | CANDIDATE | seed=2024 |
| D (feature filters, 14) | D4 | 0.42228 | STRONG | top-120 features |
| E (hyperparams, 10) | E4 | 0.42291 | STRONG | leaves=400, min_child=100 |
| F (row filters, 13) | F13 | 0.42253 | STRONG | positive queries only |
| H (alt objectives, 5) | H2 | 0.42124 | CANDIDATE | DART |
| I (creative, 6) | I3 | 0.42181 | CANDIDATE | GOSS |

## Top 30 by NDCG@5 (full table: `overnight_top30_ndcg5.csv`)

| # | config | NDCG@5 | Recall@5 | MBR | kept | description |
|---|---|---|---|---|---|---|
| 1 | B3 | 0.42396 | 0.6400 | 5.945 | EXCELLENT | lg=0,2,15, w=ipw_clip3 |
| 2 | E4 | 0.42291 | 0.6421 | 5.947 | STRONG | leaves=400, min_child=100 |
| 3 | F13 | 0.42253 | 0.6423 | 5.969 | STRONG | positive queries only, lg=0,3,15 |
| 4 | E7 | 0.42251 | 0.6411 | 5.939 | STRONG | lr=0.02 |
| 5 | D4 | 0.42229 | 0.6391 | 5.958 | STRONG | top_120 features |
| 6 | A1 | 0.42205 | 0.6377 | 5.977 | STRONG | label_gain=0,2,12 |
| 7 | E1 | 0.42203 | 0.6409 | 5.960 | STRONG | leaves=255, min_child=100 |
| 8 | A6 | 0.42198 | 0.6409 | 5.965 | CANDIDATE | label_gain=0,3,15 |
| 9 | B10 | 0.42198 | 0.6394 | 6.005 | CANDIDATE | lg=0,2,20, w=no_ipw |
| 10 | B6 | 0.42197 | 0.6389 | 5.956 | CANDIDATE | lg=0,2,15, w=rand_up_2.0 |
| 11 | E3 | 0.42185 | 0.6413 | 5.961 | CANDIDATE | leaves=512, reg_λ tuned |
| 12 | I3 | 0.42181 | 0.6403 | 5.952 | CANDIDATE | GOSS boosting |
| 13 | A13 | 0.42169 | 0.6398 | 5.963 | CANDIDATE | label_gain=0,1,30 |
| 14 | B9 | 0.42168 | 0.6400 | 5.976 | CANDIDATE | lg=0,3,15, w=rand_up_1.5 |
| 15 | F8 | 0.42166 | 0.6402 | 5.980 | CANDIDATE | random rows weight 3x |
| 16 | I6 | 0.42162 | 0.6402 | 5.945 | CANDIDATE | DART, drop_rate=0.05 |
| 17 | C9 | 0.42145 | 0.6385 | 5.970 | CANDIDATE | seed=2024 |
| 18 | D11 | 0.42145 | 0.6408 | 5.973 | CANDIDATE | drop booking buckets |
| 19 | F2 | 0.42140 | 0.6411 | 5.961 | CANDIDATE | downweight all-zero queries 0.1x |
| 20 | E6 | 0.42140 | 0.6411 | 5.961 | CANDIDATE | subsample=0.6 |
| 21 | I1 | 0.42124 | 0.6376 | 5.951 | CANDIDATE | heavy regularization |
| 22 | H2 | 0.42124 | 0.6379 | 5.966 | CANDIDATE | DART (alt boosting) |
| 23 | C11 | 0.42121 | 0.6400 | 5.950 | CANDIDATE | seed=123, lg=0,2,20 |
| 24 | B1 | 0.42120 | 0.6370 | 5.993 | CANDIDATE | lg=0,2,15, w=no_ipw |
| 25 | A2 | 0.42115 | 0.6383 | 5.961 | CANDIDATE | label_gain=0,2,18 |
| 26 | B7 | 0.42111 | 0.6379 | 6.002 | CANDIDATE | lg=0,3,15, w=no_ipw |
| 27 | C8 | 0.42110 | 0.6403 | 5.955 | CANDIDATE | seed=789, lg=0,3,15 |
| 28 | E9 | 0.42109 | 0.6386 | 5.972 | CANDIDATE | reg_lambda=3.0 |
| 29 | A5 | 0.42109 | 0.6404 | 5.974 | CANDIDATE | label_gain=0,3,12 |
| 30 | A11 | 0.42109 | 0.6404 | 5.974 | CANDIDATE | label_gain=0,5,20 |

## Top 20 by Recall@5 (full table: `overnight_top20_recall5.csv`)

| # | config | Recall@5 | NDCG@5 | description |
|---|---|---|---|---|
| 1 | F13 | 0.6423 | 0.42253 | positive queries only |
| 2 | E4 | 0.6421 | 0.42291 | leaves=400, min_child=100 |
| 3 | E3 | 0.6413 | 0.42185 | leaves=512 + tuned reg |
| 4 | E6 | 0.6411 | 0.42140 | subsample=0.6 |
| 5 | E7 | 0.6411 | 0.42251 | lr=0.02 |
| 6 | F2 | 0.6411 | 0.42140 | downweight all-zero queries |
| 7 | E1 | 0.6409 | 0.42203 | leaves=255, min_child=100 |
| 8 | A6 | 0.6409 | 0.42198 | label_gain=0,3,15 |
| 9 | C3 | 0.6408 | 0.42094 | seed=789, lg=0,2,15 |
| 10 | A10 | 0.6408 | 0.42044 | label_gain=0,4,20 |
| 11 | D11 | 0.6408 | 0.42145 | drop booking buckets |
| 12 | E10 | 0.6404 | 0.42102 | reg_alpha=0.5 |
| 13 | A11 | 0.6404 | 0.42109 | label_gain=0,5,20 |
| 14 | A12 | 0.6404 | 0.42071 | label_gain=0,1,25 |
| 15 | A5 | 0.6404 | 0.42109 | label_gain=0,3,12 |
| 16 | C8 | 0.6403 | 0.42110 | seed=789, lg=0,3,15 |
| 17 | I3 | 0.6403 | 0.42181 | GOSS |
| 18 | A4 | 0.6402 | 0.42073 | label_gain=0,2,25 |
| 19 | I6 | 0.6402 | 0.42162 | DART drop=0.05 |
| 20 | F8 | 0.6402 | 0.42166 | random rows ×3 |

## Recommended ensemble pool (10 members, full: `candidate_models_for_ensemble.csv`)

Greedy NDCG-first with diversity on (label_gain, weighting, boosting, feat_filter, row_filter, seed). **Manually add E7 + E4 + I6** — they bring hyperparameter and DART diversity that the signature didn't capture.

| # | config | NDCG@5 | axis | description |
|---|---|---|---|---|
| 1 | B3 | 0.42396 | weighting | ipw_clip3 |
| 2 | F13 | 0.42253 | row filter | positive queries only |
| 3 | D4 | 0.42229 | feature filter | top_120 |
| 4 | A1 | 0.42205 | label_gain | 0,2,12 |
| 5 | E4 | 0.42291 | hyperparams ★ | leaves=400, min_child=100 |
| 6 | E7 | 0.42251 | learning rate ★ | lr=0.02 |
| 7 | B10 | 0.42198 | weighting | no_ipw + lg=0,2,20 |
| 8 | I3 | 0.42181 | boosting | GOSS |
| 9 | I6 | 0.42162 | boosting ★ | DART drop=0.05 |
| 10 | B9 | 0.42168 | weighting | rand_up_1.5 + lg=0,3,15 |
| 11 | C9 | 0.42145 | seed | seed=2024 |
| 12 | F1 | 0.42104 | row filter | drop all-zero queries |

★ = added manually for diversity beyond auto-greedy.

## Notes (10 max)

- **B3 is the only EXCELLENT** (0.42396 vs anchor 0.42191; +0.00205). Single-model val deltas remain *directional* per the gap rule — do not submit alone.
- **Group E (hyperparams) is the strongest group on aggregate**: 3/10 STRONG, 7/10 ≥ CANDIDATE, best mean NDCG@5. Worth more sweeps next round.
- **Group B (weighting) has the highest variance** (best 0.42396, mean 0.41650). `ipw_clip3` is a clear winner; `ipw_clip5` and `rand_up_*` mixed.
- **Group C (seeds) is flat** — all 12 seeds within 0.0007 of each other; confirms the V4-style pattern is deterministic and seed shuffling alone won't help.
- **Group H is mostly weak** (best 0.42124). Binary classifiers (H3-H5) all landed ~0.413; rank_xendcg (H1) ~0.418. DART (H2) is the only useful H member.
- **Group I (creative) is mixed**: GOSS (I3) and DART (I6) are good; row-filter combos (I4/I5) are weak. I5 at 0.39602 is the worst result of the run.
- **No POSSIBLE_OVERFIT** (>0.4240) hit — caps the upside. The risky lookout is the top-3 (B3, E4, F13) where val gain may not transfer to Kaggle given the v4.2 lesson (−0.00619 gap).
- **Risky local-overfit signals** (high NDCG@5 but Recall@5 or MBR worse than pool mean): F13, D4, A1, B10. Treat their val gains conservatively.
- **mean_booked_rank tightest** (best ranking precision): E7, B3, I6, E4, C11. These are the most "confident at the top" — likely good ensemble members regardless of NDCG ordering.
- **Interrupted I7-I13**: heavy regularization, DART variants, max_bin=64, focus_on_signal, aggressive early stop, fast shrinkage, no subsampling. None ran. Re-launching the script will pick them up via resume logic.

## Files written

- `experiment_logs/overnight_summary.md` (this file)
- `experiment_logs/candidate_models_for_ensemble.csv` (34 rows: all CANDIDATE+ models)
- `experiment_logs/overnight_top30_ndcg5.csv`
- `experiment_logs/overnight_top20_recall5.csv`
- `experiment_logs/overnight_top30_mbr.csv`
