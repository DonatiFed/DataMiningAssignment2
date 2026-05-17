# Final Kaggle scoreboard

Authoritative list of every submission uploaded to Kaggle for the
[DMT 2026 — 2nd Assignment](https://www.kaggle.com/competitions/dmt-2026-2nd-assignment)
competition (submitted as **team_80**), with the resulting public NDCG@5
and the metadata needed to reproduce each one.

## Headline

**Best Kaggle public NDCG@5: 0.42021 (V4 ensemble).**

V4 remained the highest-scoring submission across all attempted versions.
The two highest-scoring uploads — V4 (0.42021) and V9 overnight best
deployable (0.42012) — were used as the team's primary submissions for
private leaderboard scoring.

## Submission-by-submission

| upload order | submission file | local NDCG@5 | Kaggle public | Δ vs V4 | composition |
|---|---|---:|---:|---:|---|
| 1 | `submission_v4_20260515_151132.csv` ★ | 0.42512 (random val) | **0.42021** | — | V4 ensemble (multi `label_gain` LambdaRank, IPW, seed 456) |
| 2 | `submission_phase2_best_20260515_225726.csv` | 0.42258 (random val) | 0.41639 | −0.00382 | Phase 2 single `lg=0,2,15` (suspected V4 anchor invariant violation) |
| 3 | `submission_v5_ensemble_20260516_094741.csv` | 0.42633 (random val) | 0.41943 | −0.00078 | V5 with cross-key TEs (adversarial AUC=1.0 confirmed drift) |
| 4 | `submission_v5_te_ablation_20260516_103823.csv` | — | (skipped on Kaggle) | — | V5.2 drift-TE ablation |
| 5 | `submission_v6_loo9_20260516_184304.csv` | 0.40896 (temporal val) | 0.42004 | −0.00017 | V6 LOO-9 (9 LambdaRank members, temporal split) |
| 6 | `submission_overnight_best_conservative_20260517_022323.csv` | 0.40943 (temporal val) | (not uploaded) | — | V9 conservative XEN+CP+DS seed averages |
| 7 | `submission_overnight_best_deployable_20260517_095946.csv` | 0.40971 (temporal val) | 0.42012 | −0.00009 | V9 overnight (V6 @ 0.80 + cb_C, cb_A, xen_cons, xen_seed42 each @ 0.05) |
| 8 | `submission_adv_reweight_20260517_111219.csv` | 0.40997 (temporal val) | 0.41903 | −0.00118 | V10 adversarial (adv-V6 @ 0.50 + 4 diversifiers @ 0.125 each) |
| 9 | `submission_FINAL_safepush_v75_6div_20260517_123507.csv` | ≈ 0.4099 (estimated) | 0.41995 | −0.00026 | V11a SAFE-PUSH (V6 @ 0.75 + 6 diversifiers @ 0.0417 each) |
| 10 | `submission_FINAL_megabag_25equal_20260517_123507.csv` | ≈ 0.4090 (estimated) | 0.42003 | −0.00018 | V11b MEGA-BAG (23 trained models equal rank-average) |

★ = highest-scoring submission, used as the team's primary upload.

## Ranked by Kaggle public score

| rank | Kaggle | submission |
|---:|---:|---|
| 🥇 1 | **0.42021** | V4 ensemble |
| 🥈 2 | 0.42012 | V9 overnight best deployable |
| 🥉 3 | 0.42004 | V6 LOO-9 |
| 4 | 0.42003 | V11 MEGA-BAG |
| 5 | 0.41995 | V11 SAFE-PUSH |
| 6 | 0.41943 | V5 ensemble |
| 7 | 0.41903 | V10 adversarial reweight |
| 8 | 0.41639 | V4.2 / Phase 2 best |

## Reproducibility quick-reference

To reproduce **V4 ensemble** (the winning submission):
```bash
uv run python pipelines/legacy/v4_ensemble.py
```
Output: `submissions/submission_v4_<TS>.csv`. Anchor invariant: random
val NDCG@5 = 0.42191.

To reproduce **V6 LOO-9** (the temporal-clean baseline):
```bash
uv run python pipelines/v6.py
uv run python pipelines/v6_submit.py
```
Output: `submissions/submission_v6_loo9_<TS>.csv`.

To reproduce **V9 overnight best deployable**:
```bash
uv run python pipelines/overnight_final_batch.py
# automatically builds submission_overnight_best_*.csv
# if it crashes mid-submission, recover with:
uv run python scripts/overnight_submit_best.py
```

To reproduce **V11 final 2 submissions** (no training, just ensemble math
over saved test predictions):
```bash
uv run python scripts/build_two_final_submissions.py
# outputs both SAFE-PUSH and MEGA-BAG CSVs
```

## Comparison to the top of the public leaderboard

The top of the public leaderboard at submission time was approximately
0.46. Gap from V4: **+0.04 NDCG@5**.

This gap is roughly five times the total variance between our attempted
submissions (0.41639 to 0.42021, a spread of 0.0038). Closing it would
require structurally different approaches not attempted here — see
`next_steps.md` for the forward queue.
