# Adversarial reweight batch — 20260517_111219

_Generated 2026-05-17T12:12:13.320792 • elapsed 59.9 min_

## Hypothesis
V5 had adversarial AUC=1.0 — perfect train/test distinguishability on features. Local gains don't translate to Kaggle because models overfit train distribution. Reweight train rows by importance ratio so loss focuses on test-like rows.


## Adversarial classifier (eval phase)

- Holdout AUC: **1.0000**
  → **HIGH drift confirmed.** Reweighting should help.
- best_iter: 92
- P(test) on train: p50=0.0057, p90=0.0090

## V6 member retrains with adv*IPW weights
```
             model_id status    ndcg5  delta_vs_v6_loo9  best_iter  elapsed_min
  lambdarank_base_adv     ok +0.40345          -0.00551        603     +3.76000
lambdarank_click3_adv     ok +0.40431          -0.00465        406     +2.88000
 lambdarank_bal15_adv     ok +0.40466          -0.00430        382     +2.77000
lambdarank_book50_adv     ok +0.40447          -0.00449        408     +2.87000
   rank_xendcg_v6_adv     ok +0.40488          -0.00408        591     +3.12000
               CP_adv     ok +0.40574          -0.00322        511     +3.39000
               DS_adv     ok +0.40469          -0.00427        614     +3.80000
```

## Phase A ensembles
```
                 test_id   method  n_members  v6_weight    ndcg5  delta_vs_v6_loo9
            adv_v6_alone rank_avg          9   +1.00000 +0.40818          -0.00078
adv_v6@0.50+4_div@0.1250 rank_avg          5   +0.50000 +0.40997          +0.00101
adv_v6@0.55+4_div@0.1125 rank_avg          5   +0.55000 +0.40992          +0.00096
adv_v6@0.60+4_div@0.1000 rank_avg          5   +0.60000 +0.40980          +0.00084
adv_v6@0.65+4_div@0.0875 rank_avg          5   +0.65000 +0.40962          +0.00066
adv_v6@0.70+4_div@0.0750 rank_avg          5   +0.70000 +0.40961          +0.00065
adv_v6@0.75+4_div@0.0625 rank_avg          5   +0.75000 +0.40968          +0.00072
adv_v6@0.80+4_div@0.0500 rank_avg          5   +0.80000 +0.40947          +0.00051
adv_v6@0.85+4_div@0.0375 rank_avg          5   +0.85000 +0.40901          +0.00005
```

## Best Phase A: adv_v6@0.50+4_div@0.1250

- NDCG@5 = 0.40997
- Δ vs V6 LOO-9 (0.40896): +0.00101
- Δ vs overnight Kaggle (0.42012): N/A (this is temporal val, not Kaggle)

## Submission status

**Built:** `submission_adv_reweight_20260517_111219.csv`
- V6 weight: 0.5000
- V6 members: 9
- Diversifiers: 4
- Adv AUC (full train): 1.0000
