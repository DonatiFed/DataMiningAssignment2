# Ensemble normalization search — 20260517_103124

_Generated 2026-05-17T10:35:04.328821 • elapsed 3.7 min_

**Tests run:** 345  ·  **Members in pool:** 14  ·  **Missing predictions:** 0


## 1. Best normalization method

```
method
blend_rank_global_z   0.41005
rank_avg              0.41002
blend_rank_query_z    0.40996
global_z              0.40995
query_z               0.40985
```

**Winner method:** `blend_rank_global_z` (best NDCG@5 = 0.41005)

## 2. Best ensemble score

**NDCG@5 = 0.41005**

- Δ vs V6 LOO-9 (0.40896): **+0.00109**
- Δ vs overnight best (0.40979): **+0.00026**

## 3. Best weights

- **Test ID:** `B_n5_v6=0.60`
- **Method:** `blend_rank_global_z`
- **V6 weight:** 0.6000
- **Members added:** `cb_rank_C_deeper+cb_rank_A+struct_rank_xendcg_regularized+xendcg_reg_seed42+xendcg_conservative`
- **Weights:** `0.0800,0.0800,0.0800,0.0800,0.0800`
- **Total added weight:** 0.4000

## 4–5. Deltas

- Δ vs V6 LOO-9: **+0.00109**
- Δ vs overnight best (0.40979): **+0.00026**

## 6. Did aggressive low-V6 weights help?

Best aggressive (V6 ≤ 0.60): `B_n5_v6=0.60` NDCG@5=0.41005, method=blend_rank_global_z, V6_w=0.600

→ Aggressive helped (best overall is aggressive).

## 7. Did any cross 0.41000?

**YES** — 2 ensembles ≥ 0.41000. Top 5:
```
              test_id              method    ndcg5  delta_vs_v6_loo9
         B_n5_v6=0.60 blend_rank_global_z +0.41005          +0.00109
B_XEN_skew2.0_v6=0.70            rank_avg +0.41002          +0.00106
```

## Top 20 ensembles
```
               test_id              method  n_added  v6_weight    ndcg5  delta_vs_v6_loo9  delta_vs_overnight_best
          B_n5_v6=0.60 blend_rank_global_z        5   +0.60000 +0.41005          +0.00109                 +0.00026
 B_XEN_skew2.0_v6=0.70            rank_avg        4   +0.70000 +0.41002          +0.00106                 +0.00023
  B_CB_skew2.0_v6=0.50  blend_rank_query_z        4   +0.50000 +0.40996          +0.00100                 +0.00017
          A_n5_v6=0.75            global_z        5   +0.75000 +0.40995          +0.00099                 +0.00016
 B_XEN_skew1.5_v6=0.70            rank_avg        4   +0.70000 +0.40995          +0.00099                 +0.00016
C_xen_skew1.5_v6=0.700            rank_avg        4   +0.70000 +0.40995          +0.00099                 +0.00016
 C_cb_skew1.5_v6=0.600 blend_rank_global_z        4   +0.60000 +0.40994          +0.00098                 +0.00015
  B_CB_skew1.5_v6=0.60 blend_rank_global_z        4   +0.60000 +0.40994          +0.00098                 +0.00015
          A_n4_v6=0.75            global_z        4   +0.75000 +0.40993          +0.00097                 +0.00014
      C_equal_v6=0.750            global_z        4   +0.75000 +0.40993          +0.00097                 +0.00014
          A_n4_v6=0.65 blend_rank_global_z        4   +0.65000 +0.40991          +0.00095                 +0.00012
      C_equal_v6=0.650 blend_rank_global_z        4   +0.65000 +0.40991          +0.00095                 +0.00012
C_xen_skew1.5_v6=0.725            rank_avg        4   +0.72500 +0.40990          +0.00094                 +0.00011
 C_cb_skew1.5_v6=0.725            global_z        4   +0.72500 +0.40990          +0.00094                 +0.00011
          B_n4_v6=0.50 blend_rank_global_z        4   +0.50000 +0.40989          +0.00093                 +0.00010
  B_CB_skew1.5_v6=0.50 blend_rank_global_z        4   +0.50000 +0.40989          +0.00093                 +0.00010
          B_n4_v6=0.60            rank_avg        4   +0.60000 +0.40987          +0.00091                 +0.00008
      C_equal_v6=0.600            rank_avg        4   +0.60000 +0.40987          +0.00091                 +0.00008
  B_CB_skew2.0_v6=0.60 blend_rank_global_z        4   +0.60000 +0.40986          +0.00090                 +0.00007
          A_n5_v6=0.65 blend_rank_global_z        5   +0.65000 +0.40986          +0.00090                 +0.00007
```

## 8. Recommended next action

Improvement of +0.00026 is within noise. **Recommend KEEPING the existing overnight best submission**; do not retrain.
