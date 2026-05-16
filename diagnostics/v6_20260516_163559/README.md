# V6 gym batch — 20260516_163559

Generated: 2026-05-16T17:10:02.655870+00:00

- V4_ANCHOR temporal NDCG@5 baseline: **0.40401**
- Quick ensemble V4+CP+DS benchmark: **0.40679**
- Submission threshold (≥+0.0003 vs quick): **0.40709**

## Member results

```
        member_id       type label_gain weight extra_feature  n_features  best_iter   ndcg5  recall5  mean_booked_rank  delta_vs_v4_anchor  elapsed_min
               CP lambdarank     0,1,15    ipw            CP         144         -1 0.40533  0.61937           6.24135             0.00132      0.13000
               DS lambdarank     0,1,15    ipw            DS         144         -1 0.40495  0.61851           6.27624             0.00094      0.11000
  lambdarank_base lambdarank     0,1,31    ipw                       143        656 0.40486  0.61794           6.26212             0.00085      4.16000
lambdarank_click3 lambdarank     0,3,31    ipw                       143        522 0.40430  0.61708           6.28103             0.00029      3.52000
 lambdarank_bal15 lambdarank     0,1,15    ipw                       143         -1 0.40401  0.61869           6.27503            -0.00000      0.14000
lambdarank_book50 lambdarank     0,1,50    ipw                       143        400 0.40348  0.61655           6.29068            -0.00053      2.97000
 lambdarank_noipw lambdarank     0,1,15   none                       143        295 0.40339  0.61790           6.29265            -0.00062      2.01000
      rank_xendcg     xendcg     0,1,15    ipw                       143        663 0.40216  0.61601           6.31006            -0.00185      3.43000
lambdarank_randup lambdarank     0,2,25 randup                       143        259 0.40154  0.61272           6.32146            -0.00247      1.84000
      booking_clf     binary        NaN   none                       143         70 0.38682  0.59888           6.58151            -0.01719      1.10000
```

## Ensemble results (rank-average)

```
            config  n_members   ndcg5  recall5  mean_booked_rank  delta_vs_v4_anchor  delta_vs_quick                                                                                                                               members
v4_plus_CP_plus_DS         10 0.40841  0.62437           6.19112             0.00440         0.00162 lambdarank_base,lambdarank_click3,lambdarank_bal15,lambdarank_book50,lambdarank_noipw,rank_xendcg,lambdarank_randup,booking_clf,CP,DS
        v4_plus_CP          9 0.40793  0.62355           6.19005             0.00392         0.00114    lambdarank_base,lambdarank_click3,lambdarank_bal15,lambdarank_book50,lambdarank_noipw,rank_xendcg,lambdarank_randup,booking_clf,CP
        v4_plus_DS          9 0.40769  0.62284           6.19612             0.00368         0.00090    lambdarank_base,lambdarank_click3,lambdarank_bal15,lambdarank_book50,lambdarank_noipw,rank_xendcg,lambdarank_randup,booking_clf,DS
           v4_only          8 0.40753  0.62234           6.19659             0.00352         0.00074       lambdarank_base,lambdarank_click3,lambdarank_bal15,lambdarank_book50,lambdarank_noipw,rank_xendcg,lambdarank_randup,booking_clf
      above_median          5 0.40691  0.62201           6.21539             0.00290         0.00012                                                                              lambdarank_base,lambdarank_click3,lambdarank_bal15,CP,DS
        CP_plus_DS          2 0.40604  0.62019           6.23102             0.00203        -0.00075                                                                                                                                 CP,DS
```

## Best ensemble: **v4_plus_CP_plus_DS** (NDCG@5=0.40841)

- Δ vs V4_ANCHOR  (0.40401): **+0.00440**
- Δ vs V4+CP+DS quick ensemble (0.40679): **+0.00162**
- Members: `lambdarank_base,lambdarank_click3,lambdarank_bal15,lambdarank_book50,lambdarank_noipw,rank_xendcg,lambdarank_randup,booking_clf,CP,DS`

## Leave-one-out (on best ensemble)

```
          dropped  n_remaining   ndcg5  delta_vs_best  recall5     mbr
      rank_xendcg            9 0.40740       -0.00101  0.62255 6.21200
lambdarank_randup            9 0.40748       -0.00093  0.62248 6.19455
               CP            9 0.40769       -0.00072  0.62284 6.19612
 lambdarank_noipw            9 0.40780       -0.00061  0.62312 6.19237
               DS            9 0.40793       -0.00048  0.62355 6.19005
 lambdarank_bal15            9 0.40799       -0.00042  0.62334 6.18980
lambdarank_click3            9 0.40799       -0.00041  0.62366 6.19126
  lambdarank_base            9 0.40809       -0.00031  0.62359 6.19580
lambdarank_book50            9 0.40831       -0.00010  0.62341 6.18805
      booking_clf            9 0.40896        0.00056  0.62387 6.18182
```

## Submission

- status: **failed:ValueError**
- submission attempted but failed; see logs.
