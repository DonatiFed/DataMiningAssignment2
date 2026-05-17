# V6 LOO-9 submission — 20260516_184304

Generated: 2026-05-16T19:12:06.940270+00:00

**Submission CSV:** `submission_v6_loo9_20260516_184304.csv`
**Header:** `srch_id,prop_id` (matches `data/submission_sample.csv`; original write produced `SearchId,PropertyId` and was corrected post-hoc — data rows untouched).
**Rows:** 4,959,183 · **Unique searches:** 199,549 · **Avg props/search:** 24.9 · **Duplicates / NaNs:** 0

## Members

- `lambdarank_base` (type=lambdarank, lg=0,1,31, weight=ipw, extra=None)
- `lambdarank_click3` (type=lambdarank, lg=0,3,31, weight=ipw, extra=None)
- `lambdarank_bal15` (type=lambdarank, lg=0,1,15, weight=ipw, extra=None)
- `lambdarank_book50` (type=lambdarank, lg=0,1,50, weight=ipw, extra=None)
- `lambdarank_noipw` (type=lambdarank, lg=0,1,15, weight=none, extra=None)
- `rank_xendcg` (type=xendcg, lg=0,1,15, weight=ipw, extra=None)
- `lambdarank_randup` (type=lambdarank, lg=0,2,25, weight=randup, extra=None)
- `CP` (type=lambdarank, lg=0,1,15, weight=ipw, extra=CP)
- `DS` (type=lambdarank, lg=0,1,15, weight=ipw, extra=DS)

### Dropped (per LOO)
- `booking_clf` — drops local NDCG when included; LOO showed removing it yields +0.00056

## Temporal NDCG @ 5 — local benchmarks

- V4_ANCHOR baseline:                0.40401
- V4+CP+DS quick-3-model ensemble:   0.40679
- 10-member v4+CP+DS V6 ensemble:    0.40841
- **LOO-9 ensemble (this submission, local):  0.40896**

- Δ vs V4_ANCHOR:    +0.00495
- Δ vs quick (V4+CP+DS):  +0.00217
  (threshold for submission was +0.0003 above quick = 0.40709; exceeded by 0.00187)

## Kaggle public result (post-submission)

- **Kaggle public NDCG@5: 0.42004** (vs V4 public 0.42021 = −0.00017).
- Local +0.00217 did NOT translate. Drift gap is the prime suspect (see
  `CHANGELOG.md` and `docs/next_steps.md` for the analysis).

## Justification

1. All members trained or reused from temporal-clean training only (no train/test leakage).
2. Member diversity spans label_gain variants (lg=0,1,15 / 0,1,31 / 0,3,31 / 0,1,50 / 0,2,25), weighting schemes (IPW / none / random-upweight), and a `rank_xendcg` objective, plus the two validated V6 candidates (CP = OOF position-adjusted click TE; DS = smoothed (prop, dest) booking rate with 3-way fallback).
3. LOO indicated `booking_clf` was actively hurting the ensemble (its solo NDCG@5 was 0.387, ~0.017 below anchor — a binary P(booking) score does not rank well listwise). Dropping it yields the LOO-best 9-member ensemble at local NDCG@5 = 0.40896.
4. Rank-average within `srch_id` is the simplest, drift-robust combination — no learned weights, no tunable hyperparameters.

## Risk notes

- Local→Kaggle correlation in this project is imperfect (V5 ensemble beat V4 locally but lost on Kaggle due to drift in raw TE features). However, every member here was trained on temporal_train only with drift-aware features. CP and DS both showed train/val drift `|Δμ|/σ ≤ 0.018` and `0.004` respectively.
- The `rank_xendcg` and `lambdarank_randup` members had below-anchor solo NDCG (−0.00185 and −0.00247). LOO showed they still help the ensemble through diversity. If the Kaggle delta is much smaller than the local +0.00217, the weakest members are the suspects.
- CP/DS feature build on test used the FULL train (4.96M rows) as agg_source — a clean reproduction of the train-time recipe.
