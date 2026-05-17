"""Ensemble normalization search — paper-inspired methods on saved predictions.

Read-only. Does NOT train, does NOT submit, does NOT modify any running script.
Tests 5 normalization methods × multiple weight grids on the V6 LOO-9 baseline
+ Phase 7/structural/overnight batch member predictions to see whether a
better ensemble exists without retraining.

Methods:
  1. rank_avg              within-srch rank, lower=better → negate
  2. global_z              (score - global_mean) / global_std
  3. query_z               within-srch (score - mean) / std
  4. blend_rank_query_z    0.5·rank + 0.5·query_z (each standardized first)
  5. blend_rank_global_z   0.5·rank + 0.5·global_z (each standardized first)

Failure isolation: each combo in try/except; bad ones log to errors/.
"""
from __future__ import annotations
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.temporal_validation import eval_metrics  # noqa: E402

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "diagnostics" / f"ensemble_normalization_{TIMESTAMP}"
ERRORS = OUT / "errors"

CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"

# V6 LOO-9 base
V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V6_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]

# Search paths for non-V6 member predictions (priority order — first hit wins)
PRED_SEARCH_DIRS = [
    ROOT / "diagnostics" / "overnight_final_batch_20260517_022323" / "predictions",
    ROOT / "diagnostics" / "structural_batch_20260516_212040" / "predictions",
    ROOT / "diagnostics" / "phase7_batch_20260516_203100" / "predictions",
    ROOT / "diagnostics" / "phase7_weighted_batch_20260516_210302" / "predictions",
    ROOT / "diagnostics" / "v6_20260516_163559",  # fallback (V6 internal)
]

# Priority candidate pool (from user spec)
PRIORITY_NON_V6 = [
    "cb_rank_C_deeper",
    "cb_rank_A",
    "struct_rank_xendcg_regularized",  # may resolve to structural batch
    "rank_xendcg_regularized",          # structural batch original name
    "xendcg_reg_seed42",
    "xendcg_conservative",
    "xendcg_reg_seed123",
    "xendcg_reg_seed456",
    "xendcg_reg_seed789",
    "xendcg_reg_seed2024",
    "cb_rank_B_regularized",
    "reg_bal15_seed42",
    "cp_reg_seed42",
    "ds_reg_seed42",
]

V4_ANCHOR_TEMPORAL = 0.40401
V6_LOO9_TEMPORAL = 0.40896
OVERNIGHT_BEST = 0.40979


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_write_df(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_write_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ============================================================================
# Locate prediction files
# ============================================================================
def resolve_pred(member_id: str) -> Path | None:
    """Find the val_pred file for a given member id across all search dirs.

    For `struct_rank_xendcg_regularized` also accepts the structural batch's
    original name `rank_xendcg_regularized`.
    """
    aliases = [member_id]
    if member_id == "struct_rank_xendcg_regularized":
        aliases.append("rank_xendcg_regularized")
    for a in aliases:
        for d in PRED_SEARCH_DIRS:
            p = d / f"val_pred_{a}.npy"
            if p.exists():
                return p
    return None


# ============================================================================
# Normalization functions
# ============================================================================
def within_srch_rank(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """1=best, fractional ties. Lower = better."""
    return pd.Series(scores).groupby(pd.Series(srch_id), sort=False).rank(
        method="average", ascending=False
    ).values.astype(np.float32)


def query_z(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    gb = s.groupby(pd.Series(srch_id), sort=False)
    mu = gb.transform("mean")
    sd = gb.transform("std").replace(0, np.nan).fillna(1.0)
    return ((s - mu) / sd).values.astype(np.float32)


def global_z(scores: np.ndarray) -> np.ndarray:
    s = scores.astype(np.float64)
    mu = float(np.nanmean(s))
    sd = float(np.nanstd(s)) + 1e-12
    return ((s - mu) / sd).astype(np.float32)


def standardize_inplace(arr: np.ndarray) -> np.ndarray:
    """Standardize a 1D array to unit std (zero mean already centered usually)."""
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr)) + 1e-12
    return ((arr - mu) / sd).astype(np.float32)


def to_higher_better(scores: np.ndarray, method: str, srch_id: np.ndarray) -> np.ndarray:
    """Produce a 'higher = better' representation per row."""
    if method == "rank_avg":
        r = within_srch_rank(srch_id, scores)
        return (-r).astype(np.float32)
    if method == "global_z":
        return global_z(scores)
    if method == "query_z":
        return query_z(srch_id, scores)
    if method == "blend_rank_query_z":
        a = to_higher_better(scores, "rank_avg", srch_id)
        b = to_higher_better(scores, "query_z", srch_id)
        return (0.5 * standardize_inplace(a) + 0.5 * standardize_inplace(b)).astype(np.float32)
    if method == "blend_rank_global_z":
        a = to_higher_better(scores, "rank_avg", srch_id)
        b = to_higher_better(scores, "global_z", srch_id)
        return (0.5 * standardize_inplace(a) + 0.5 * standardize_inplace(b)).astype(np.float32)
    raise ValueError(f"unknown method: {method}")


METHODS = ["rank_avg", "global_z", "query_z", "blend_rank_query_z", "blend_rank_global_z"]


# ============================================================================
# Main
# ============================================================================
def main():
    for d in (OUT, ERRORS):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"ENSEMBLE NORMALIZATION SEARCH — {TIMESTAMP}")
    log(f"out: {OUT}")

    # ---- Load val ----
    log("Loading temporal_val parquet…")
    val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    srch = val["srch_id"].values
    log(f"  rows={len(val):,}  searches={val['srch_id'].nunique():,}")

    # ---- Load V6 LOO-9 base (precompute as rank average) ----
    log("Loading V6 LOO-9 ranks…")
    v6_ranks = []
    missing_v6 = []
    for m in V6_MEMBERS:
        p = V6_DIR / f"val_pred_{m}.npy"
        if not p.exists():
            missing_v6.append(m)
            continue
        s = np.load(p).astype(np.float32)
        v6_ranks.append(within_srch_rank(srch, s))
    if missing_v6:
        log(f"  ! missing V6 preds: {missing_v6}")
    assert v6_ranks, "no V6 predictions loaded — cannot proceed"
    v6_rank_score = -np.mean(v6_ranks, axis=0)  # higher = better
    log(f"  V6 LOO-9 base: {len(v6_ranks)} members combined into 'V6 score'")

    # ---- Resolve non-V6 member predictions ----
    log("\nResolving non-V6 prediction files…")
    member_paths = {}
    missing_rows = []
    for mid in PRIORITY_NON_V6:
        p = resolve_pred(mid)
        if p is None:
            log(f"  MISS  {mid}")
            missing_rows.append({"member_id": mid, "status": "not_found"})
        else:
            member_paths[mid] = p
            log(f"  OK    {mid}  ←  {p.parent.parent.name}/{p.parent.name}/{p.name}")
    if missing_rows:
        safe_write_df(pd.DataFrame(missing_rows), OUT / "missing_predictions.csv")

    # Load raw scores for found members
    log(f"\nLoading {len(member_paths)} member predictions…")
    member_scores = {}
    for mid, p in member_paths.items():
        try:
            s = np.load(p).astype(np.float32)
            if s.shape != (len(val),):
                log(f"  ! {mid} shape {s.shape} != ({len(val)},) — SKIP")
                missing_rows.append({"member_id": mid, "status": f"shape_mismatch_{s.shape}"})
                continue
            n_bad = int(((~np.isfinite(s))).sum())
            if n_bad:
                log(f"  ! {mid} has {n_bad} non-finite values — replacing with 0")
                s = np.where(np.isfinite(s), s, 0.0).astype(np.float32)
            member_scores[mid] = s
        except Exception as e:
            log(f"  ! {mid} load failed: {e}")
            missing_rows.append({"member_id": mid, "status": f"load_error:{type(e).__name__}"})
    if missing_rows:
        safe_write_df(pd.DataFrame(missing_rows), OUT / "missing_predictions.csv")
    log(f"  available members: {list(member_scores.keys())}")

    # Precompute per-method higher-better arrays per member (cache)
    log("Precomputing normalized arrays per (member, method)…")
    cache: dict[tuple[str, str], np.ndarray] = {}
    for mid, s in member_scores.items():
        for method in METHODS:
            try:
                cache[(mid, method)] = to_higher_better(s, method, srch)
            except Exception as e:
                log(f"  ! {mid}/{method} failed: {e}")
    log(f"  cached {len(cache)} (member, method) arrays")

    # Same for V6 (already a rank-avg score; can still normalize)
    v6_cache: dict[str, np.ndarray] = {}
    for method in METHODS:
        try:
            if method == "rank_avg":
                v6_cache[method] = v6_rank_score  # already 'higher=better'
            else:
                v6_cache[method] = to_higher_better(v6_rank_score, method, srch)
        except Exception as e:
            log(f"  ! V6/{method} failed: {e}")

    # ---- Scoring helper ----
    def score_ensemble(test_id, method, v6_w, member_ids, member_weights):
        try:
            v6_arr = v6_cache.get(method)
            assert v6_arr is not None, f"V6/{method} not available"
            avail = [(m, w) for m, w in zip(member_ids, member_weights) if (m, method) in cache]
            if len(avail) < len(member_ids):
                missing = [m for m in member_ids if (m, method) not in cache]
                raise KeyError(f"missing (m,method) in cache: {missing}")
            agg = v6_w * v6_arr
            for m, w in avail:
                agg = agg + w * cache[(m, method)]
            metrics = eval_metrics(val, agg)
            return {
                "test_id": test_id, "method": method, "v6_weight": v6_w,
                "n_added": len(avail), "members_added": "+".join(member_ids),
                "weights_added": ",".join(f"{w:.4f}" for w in member_weights),
                "total_added_weight": sum(member_weights),
                "ndcg5": float(metrics["ndcg5"]),
                "recall1": float(metrics["recall1"]),
                "recall5": float(metrics["recall5"]),
                "mean_booked_rank": float(metrics["mean_booked_rank"]),
                "delta_vs_v6_loo9": float(metrics["ndcg5"]) - V6_LOO9_TEMPORAL,
                "delta_vs_overnight_best": float(metrics["ndcg5"]) - OVERNIGHT_BEST,
            }
        except Exception as e:
            log(f"  ✗ {test_id}/{method}: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                             ERRORS / f"ERR_{test_id[:80].replace('+','_').replace('/','_')}_{method}.txt")
            return None

    # ---- Define pools ----
    # Top non-V6 by importance (per user spec priority)
    top_5 = [m for m in ["cb_rank_C_deeper", "cb_rank_A",
                          "struct_rank_xendcg_regularized", "xendcg_reg_seed42",
                          "xendcg_conservative"]
             if m in member_scores]
    log(f"\nTop pool (size up to 5): {top_5}")
    top_4 = top_5[:4]

    results = []

    # ==== GRID A — conservative ====
    log("\n=== GRID A — conservative ===")
    A_v6_weights = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for n_added in (2, 3, 4, 5):
        pool = top_5[:n_added]
        if len(pool) < n_added:
            continue
        for v6_w in A_v6_weights:
            total_added = 1.0 - v6_w
            w_each = total_added / n_added
            for method in METHODS:
                r = score_ensemble(
                    f"A_n{n_added}_v6={v6_w:.2f}", method, v6_w,
                    pool, [w_each] * n_added)
                if r:
                    r["grid"] = "A_conservative"
                    results.append(r)
        safe_write_df(pd.DataFrame(results), OUT / "normalization_results.csv")
    log(f"  Grid A done. Cumulative tests: {len(results)}")

    # ==== GRID B — aggressive ====
    log("\n=== GRID B — aggressive (low V6 weight) ===")
    B_v6_weights = [0.30, 0.40, 0.50, 0.60]
    for n_added in (3, 4, 5):
        pool = top_5[:n_added]
        if len(pool) < n_added:
            continue
        for v6_w in B_v6_weights:
            total_added = 1.0 - v6_w
            w_each = total_added / n_added
            for method in METHODS:
                r = score_ensemble(
                    f"B_n{n_added}_v6={v6_w:.2f}", method, v6_w,
                    pool, [w_each] * n_added)
                if r:
                    r["grid"] = "B_aggressive"
                    results.append(r)
    # B custom skews — give CatBoost or rank_xendcg more weight
    cb_members = [m for m in top_5 if m.startswith("cb_")]
    xen_members = [m for m in top_5 if "xendcg" in m or "xen" in m]
    if len(top_4) == 4:
        for v6_w in (0.50, 0.60, 0.70):
            added_total = 1.0 - v6_w
            # CB heavy: 2x weight on cb members, 1x on others
            for cb_skew in (1.5, 2.0):
                w_cb = added_total * cb_skew / (cb_skew * len(cb_members) + len([m for m in top_4 if m not in cb_members]))
                w_other = added_total / (cb_skew * len(cb_members) + len([m for m in top_4 if m not in cb_members]))
                weights = [w_cb if m in cb_members else w_other for m in top_4]
                if abs(sum(weights) - added_total) > 0.001:
                    continue
                for method in METHODS:
                    r = score_ensemble(
                        f"B_CB_skew{cb_skew}_v6={v6_w:.2f}", method, v6_w,
                        top_4, weights)
                    if r:
                        r["grid"] = "B_aggressive_cb_skew"
                        results.append(r)
            # xen heavy
            for xen_skew in (1.5, 2.0):
                xen_in_pool = [m for m in top_4 if m in xen_members]
                w_xen = added_total * xen_skew / (xen_skew * len(xen_in_pool) + len([m for m in top_4 if m not in xen_in_pool]))
                w_other = added_total / (xen_skew * len(xen_in_pool) + len([m for m in top_4 if m not in xen_in_pool]))
                weights = [w_xen if m in xen_in_pool else w_other for m in top_4]
                if abs(sum(weights) - added_total) > 0.001:
                    continue
                for method in METHODS:
                    r = score_ensemble(
                        f"B_XEN_skew{xen_skew}_v6={v6_w:.2f}", method, v6_w,
                        top_4, weights)
                    if r:
                        r["grid"] = "B_aggressive_xen_skew"
                        results.append(r)
    safe_write_df(pd.DataFrame(results), OUT / "normalization_results.csv")
    log(f"  Grid B done. Cumulative tests: {len(results)}")

    # ==== GRID C — fine around current best ====
    log("\n=== GRID C — fine around current best ===")
    C_v6_weights = [0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80]
    if len(top_4) >= 4:
        for v6_w in C_v6_weights:
            added_total = 1.0 - v6_w
            # Equal
            w_each = added_total / 4
            for method in METHODS:
                r = score_ensemble(
                    f"C_equal_v6={v6_w:.3f}", method, v6_w,
                    top_4, [w_each] * 4)
                if r:
                    r["grid"] = "C_fine_equal"
                    results.append(r)
            # CB heavier (skew = 1.5)
            cb_w = added_total * 1.5 / (1.5 * len(cb_members) + (4 - len(cb_members)))
            o_w = added_total / (1.5 * len(cb_members) + (4 - len(cb_members)))
            weights = [cb_w if m in cb_members else o_w for m in top_4]
            if abs(sum(weights) - added_total) < 0.001:
                for method in METHODS:
                    r = score_ensemble(
                        f"C_cb_skew1.5_v6={v6_w:.3f}", method, v6_w,
                        top_4, weights)
                    if r:
                        r["grid"] = "C_fine_cb_skew"
                        results.append(r)
            # xen heavier
            xen_in_pool = [m for m in top_4 if m in xen_members]
            if xen_in_pool:
                xen_w = added_total * 1.5 / (1.5 * len(xen_in_pool) + (4 - len(xen_in_pool)))
                o_w = added_total / (1.5 * len(xen_in_pool) + (4 - len(xen_in_pool)))
                weights = [xen_w if m in xen_in_pool else o_w for m in top_4]
                if abs(sum(weights) - added_total) < 0.001:
                    for method in METHODS:
                        r = score_ensemble(
                            f"C_xen_skew1.5_v6={v6_w:.3f}", method, v6_w,
                            top_4, weights)
                        if r:
                            r["grid"] = "C_fine_xen_skew"
                            results.append(r)
    safe_write_df(pd.DataFrame(results), OUT / "normalization_results.csv")
    log(f"  Grid C done. Cumulative tests: {len(results)}")

    # ---- Aggregate ----
    res_df = pd.DataFrame(results).sort_values("ndcg5", ascending=False).reset_index(drop=True)
    safe_write_df(res_df, OUT / "normalization_results.csv")
    best = res_df.head(20).copy()
    safe_write_df(best, OUT / "best_ensembles.csv")

    log(f"\n=== Top 10 ensembles ===")
    cols = ["test_id", "method", "n_added", "v6_weight", "ndcg5", "delta_vs_v6_loo9", "delta_vs_overnight_best"]
    log(res_df.head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    # ---- README ----
    top = res_df.iloc[0]
    by_method = res_df.groupby("method")["ndcg5"].max().sort_values(ascending=False)
    best_aggressive = res_df[res_df["v6_weight"] <= 0.60]
    crosses_0_41 = res_df[res_df["ndcg5"] >= 0.41000]

    L = [f"# Ensemble normalization search — {TIMESTAMP}\n"]
    L.append(f"_Generated {datetime.now().isoformat()} • "
             f"elapsed {(time.time() - t_start)/60:.1f} min_\n")
    L.append(f"**Tests run:** {len(res_df)}  ·  **Members in pool:** {len(member_scores)}  ·  "
             f"**Missing predictions:** {len(missing_rows)}\n\n")

    L.append("## 1. Best normalization method\n")
    L.append("```")
    L.append(by_method.to_string(float_format=lambda x: f"{x:.5f}"))
    L.append("```\n")
    L.append(f"**Winner method:** `{by_method.index[0]}` "
             f"(best NDCG@5 = {by_method.iloc[0]:.5f})\n")

    L.append("## 2. Best ensemble score\n")
    L.append(f"**NDCG@5 = {top['ndcg5']:.5f}**\n")
    L.append(f"- Δ vs V6 LOO-9 (0.40896): **{top['delta_vs_v6_loo9']:+.5f}**")
    L.append(f"- Δ vs overnight best (0.40979): **{top['delta_vs_overnight_best']:+.5f}**\n")

    L.append("## 3. Best weights\n")
    L.append(f"- **Test ID:** `{top['test_id']}`")
    L.append(f"- **Method:** `{top['method']}`")
    L.append(f"- **V6 weight:** {top['v6_weight']:.4f}")
    L.append(f"- **Members added:** `{top['members_added']}`")
    L.append(f"- **Weights:** `{top['weights_added']}`")
    L.append(f"- **Total added weight:** {top['total_added_weight']:.4f}\n")

    L.append("## 4–5. Deltas\n")
    L.append(f"- Δ vs V6 LOO-9: **{top['delta_vs_v6_loo9']:+.5f}**")
    L.append(f"- Δ vs overnight best (0.40979): **{top['delta_vs_overnight_best']:+.5f}**\n")

    L.append("## 6. Did aggressive low-V6 weights help?\n")
    if len(best_aggressive):
        log_b = best_aggressive.iloc[0]
        L.append(f"Best aggressive (V6 ≤ 0.60): `{log_b['test_id']}` "
                 f"NDCG@5={log_b['ndcg5']:.5f}, method={log_b['method']}, "
                 f"V6_w={log_b['v6_weight']:.3f}\n")
        if log_b["ndcg5"] >= top["ndcg5"]:
            L.append("→ Aggressive helped (best overall is aggressive).\n")
        else:
            L.append(f"→ No, conservative (high V6 weight) wins by "
                     f"{top['ndcg5'] - log_b['ndcg5']:+.5f}.\n")
    else:
        L.append("No aggressive tests met conditions.\n")

    L.append("## 7. Did any cross 0.41000?\n")
    if len(crosses_0_41):
        L.append(f"**YES** — {len(crosses_0_41)} ensembles ≥ 0.41000. Top 5:\n```")
        L.append(crosses_0_41.head(5)[["test_id", "method", "ndcg5",
                                           "delta_vs_v6_loo9"]].to_string(
            index=False, float_format=lambda x: f"{x:+.5f}"))
        L.append("```\n")
    else:
        L.append(f"**No.** Best = {top['ndcg5']:.5f} (max ceiling found).\n")

    L.append("## Top 20 ensembles\n```")
    L.append(res_df.head(20)[cols].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    L.append("```\n")

    # Recommendation
    L.append("## 8. Recommended next action\n")
    if top["ndcg5"] >= 0.41050:
        L.append("**BUILD A NEW SUBMISSION** from this ensemble. Worth retraining the "
                 "necessary members on full train; the gain over current overnight best "
                 f"({top['delta_vs_overnight_best']:+.5f}) is meaningful.\n")
    elif top["ndcg5"] >= OVERNIGHT_BEST + 0.0005:
        L.append("**Consider building a new submission** — improvement of "
                 f"{top['delta_vs_overnight_best']:+.5f} over overnight best is real but small. "
                 "Weigh the retrain cost vs the expected Kaggle delta.\n")
    elif top["ndcg5"] > OVERNIGHT_BEST:
        L.append(f"Improvement of {top['delta_vs_overnight_best']:+.5f} is within noise. "
                 "**Recommend KEEPING the existing overnight best submission**; do not retrain.\n")
    else:
        L.append("**No improvement** over current overnight best (0.40979). Normalization did not "
                 "unlock new gain. **Recommend STOP** further normalization work. The next lever is "
                 "either fixing XGBoost (planned in `overnight_xgb_rescue.py`) or moving to NN "
                 "listwise + adversarial reweighting (described in `docs/next_steps.md`).\n")

    safe_write_text("\n".join(L), OUT / "README.md")
    log(f"\n=== DONE in {(time.time() - t_start)/60:.1f} min ===")
    log(f"outputs: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ERRORS.mkdir(parents=True, exist_ok=True)
        safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                         ERRORS / "FATAL.txt")
        log(f"FATAL: {e}")
        raise
