# Model Development

The model evolved over four versions, each fixing a specific flaw exposed by the previous one. The headline result is the **V4 ensemble: Kaggle NDCG@5 `0.42021`**. The path there is as instructive as the number.

## Overview

- **V1&rarr;V4 progression** &mdash; each version is a diagnosis, not just a tweak: leak detected (V1), leak fixed (V2&ndash;V3), variance reduced (V4).
- **V4 is an 8-model LightGBM ensemble** &mdash; diverse members blended by within-query rank averaging.
- **Core lesson** &mdash; a single model cannot beat the ensemble on Kaggle even when it scores higher locally; final submissions are always ensembles.

## Model progression

| Version | Local NDCG@5 | Kaggle NDCG@5 | Key change |
|---|---|---|---|
| V1 | 0.468 | 0.38208 | First pipeline &mdash; **leaked** same-row target aggregates |
| V2 | &mdash; | 0.39149 | Leak fixed (OOF target encoding) |
| V3 | 0.412&ndash;0.417 | 0.41392 | First honest model &mdash; OOF + IPW + full feature set |
| **V4** | **0.42512** | **0.42021** | 8-model rank-averaged ensemble |

### V1 &mdash; local 0.468 / Kaggle 0.38208

First end-to-end pipeline: LightGBM LambdaRank with basic hotel/search features plus target aggregates (click rate, booking rate per `prop_id`) computed on the **full training set, including each row's own label**.

The **0.086 local&ndash;Kaggle gap** was the diagnostic. A gap that large is the fingerprint of label leakage: same-row target aggregates encoded the current row's own outcome into its feature vector, making the label trivially recoverable at training time and useless at test time. **Decision:** move all target-derived aggregates to k-fold OOF &mdash; see [feature-engineering.md](feature-engineering.md).

### V2 &mdash; Kaggle 0.39149

Addressed the leak. The Kaggle score rose from 0.382 to 0.391, confirming the V1 gap was entirely leakage and not model quality. No local score tracked at this stage.

### V3 &mdash; local 0.412&ndash;0.417 / Kaggle 0.41392

The first honest model. Three changes from V2:

1. **Full OOF target encoding** &mdash; all aggregates computed fold-by-fold; val and test receive features from the training source only.
2. **Inverse Propensity Weighting** &mdash; non-random rows reweighted to correct display-position bias.
3. **Full feature set** &mdash; the complete 143-feature catalog.

The local&ndash;Kaggle gap dropped to ~0, confirming the [validation strategy](validation.md) is reliable. V3 becomes **the reference baseline**: any new configuration must beat 0.412 locally before being considered. Its base LightGBM configuration is documented in [hyperparameters.md](hyperparameters.md).

### V4 &mdash; Kaggle 0.42021

The current production reference: an 8-model ensemble (`pipelines/v4_ensemble.py`), run as a 4-stage gated pipeline. Validation NDCG@5 0.42512, Kaggle 0.42021 &mdash; **+0.006 over V3**.

---

## Why ensemble

The [seed-robustness check](validation.md) shows single-model NDCG@5 varies **±0.0037** across split seeds. A single V3 model on a different 10% holdout could land anywhere from 0.411 to 0.422. Ensembling does two things:

- **Reduces variance** &mdash; averaging over members narrows the spread around the true mean.
- **Improves the mean** &mdash; *when members make different errors*, the blend is better than any single member.

The second point only holds if members are genuinely diverse. Eight copies of the same configuration produce correlated predictions, and ensembling correlated models gives diminishing returns.

## The 8-member diversity design

Diversity was built deliberately along five axes:

| Axis | Members | Rationale |
|---|---|---|
| Label gain (`{0,1,5}` mapping) | `lambdarank_base`, `lambdarank_click3`, `lambdarank_bal15`, `lambdarank_book50` | Tests how aggressively to weight bookings vs clicks. `click3` recovers gradient from the 30.7% no-booking queries; `book50` over-bets on bookings. |
| Loss function | `rank_xendcg` | A different NDCG surrogate &mdash; errors not perfectly correlated with LambdaRank. |
| IPW on/off | `lambdarank_noipw` | An implicit A/B on whether IPW helps at the model level. |
| Data weighting | `lambdarank_randup` | Random-exposure rows ×2 &mdash; trades non-random signal for cleaner labels. |
| Objective type | `booking_clf` | Pointwise binary classifier on `booking_bool` &mdash; fundamentally different from listwise; adds grade-5-vs-rest signal. |

The full member roster:

| Member | seed | label_gain | Special |
|---|---|---|---|
| `lambdarank_base` | 42 | 0,1,31 | default IPW |
| `lambdarank_click3` | 123 | 0,3,31 | upweight clicks |
| `lambdarank_bal15` | 456 | 0,1,15 | balanced gain &mdash; **the anchor** |
| `lambdarank_book50` | 789 | 0,1,50 | `num_leaves=300`, `lr=0.05` |
| `lambdarank_noipw` | 2024 | 0,1,31 | no IPW (weight = 1) |
| `rank_xendcg` | 314 | n/a | `rank_xendcg` objective, `num_leaves=350` |
| `lambdarank_randup` | 555 | 0,2,25 | `random_bool=1` rows ×2 weight |
| `booking_clf` | 666 | n/a | binary `booking_bool`, AUC |

`lambdarank_bal15` is the **anchor**: a `seed=456, label_gain="0,1,15"` model that must reproduce val NDCG@5 0.42191 ([validation.md](validation.md)).

## Blending

Member scores are **not comparable across members** &mdash; different objectives produce different score scales, and direct averaging would be dominated by the widest-range member. Instead:

1. For each member, scores are converted to **within-query percentile ranks**.
2. Ranks are averaged, weighted by each member's validation NDCG@5.
3. A simple unweighted average is also computed; whichever scores higher on validation is used for the test submission.

## Staged gating

The V4 pipeline runs four checkpoints to prevent submitting a regression:

1. **10% sample sanity check** &mdash; abort if NDCG@5 < 0.28.
2. **Full data, single model** &mdash; gate on NDCG@5 &ge; 0.412 (V3 baseline).
3. **Full ensemble** &mdash; keep only above-median members.
4. **Full retrain** at best iteration on the full training set &rarr; generate the test submission.

## The v4.2 single-model stress test

Phase 2 retrained the best single configuration (`lg_0_2_15`) on full train and submitted it as v4.2. Result: **local 0.42258, Kaggle 0.41639** &mdash; a &minus;0.00619 gap, *wider* than the V4 ensemble's &minus;0.00491.

Two compounding effects:

1. **Validation overfit** &mdash; 0.42258 was a noisy local maximum, only 0.0007 above the anchor (well inside the ±0.0037 noise floor).
2. **Loss of ensemble diversity** &mdash; V4's eight members hedge against test distribution shift; a single model cannot.

This confirmed the rule: single-model val numbers are directional only, and **final submissions stay ensembles**. The label-gain sweep that produced `lg_0_2_15` is detailed in [hyperparameters.md](hyperparameters.md).
