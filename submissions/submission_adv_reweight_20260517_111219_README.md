# Adversarial reweight submission — 20260517_111219

## Strategy

Reweight train rows by importance ratio `P(test|x) / (1 - P(test|x))` to correct train→test distribution drift. V5 had adversarial AUC=1.0 — perfect distinguishability. This submission retrains V6's IPW-using members with combined weights = IPW × adv.

## Adversarial classifier diagnostics (full-train)
- Holdout AUC: 1.0000
- best_iter: 73
- P(test) percentiles on train: p10=0.0127, p50=0.0134, p90=0.0193

## Composition
- V6 (adv-retrained + 2 unchanged): weight 0.5000
  - 9 V6 members
- 4 diversifiers @ 0.1250 each

## Local result
- Best ensemble: adv_v6@0.50+4_div@0.1250 NDCG@5=0.40997
- Δ vs V6 LOO-9 (0.40896): +0.00101

## Risk notes
- Local→Kaggle gap before adv: +0.011 (V6 0.40896 → 0.42004)
- Adversarial reweighting SPECIFICALLY targets this gap
- Best case: Kaggle gain matches local gain (+0.001-0.003)
- Risk case: local drops and Kaggle drops too
