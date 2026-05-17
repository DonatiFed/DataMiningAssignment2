"""Phase 7 batch — failure-pattern-driven feature variants on temporal val.

Tests up to 5 surgical feature variants identified from the V6 failure-pattern
analysis. Each variant gets ONE model (V4_ANCHOR config) trained on
temporal_train and scored on temporal_val. Then a rank-average ensemble
phase combines the V6 LOO-9 baseline with the KEEP/HOLD-eligible variants.

Hard requirements:
- Fault-tolerant: per-variant try/except, FATAL.txt on global crash.
- Resumable: skip any artifact that already exists.
- Atomic writes: tmp + rename for every CSV and JSON.
- No leaked overnight temporal predictions.
- No automatic submission. Conditional submission CSV only if best ≥ 0.40950.

Outputs (diagnostics/phase7_batch_<ts>/):
  README.md
  feature_variant_results.csv
  ensemble_results.csv
  leave_one_out.csv
  selected_feature_ideas.md
  errors/                    (per-feature ERROR_<id>.txt or FATAL.txt)
  predictions/               (val_pred_<id>.npy, test_pred_<id>.npy)
  models/                    (model_<id>.txt)
  feature_importances/       (importance_<id>.csv, drift_<id>.csv)
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_train, load_test, make_target, get_feature_columns  # noqa: E402
from src.features import build_features, compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402
from pipelines.temporal_validation import (  # noqa: E402
    compute_ipw_weights, make_group_counts, eval_metrics, BASE_PARAMS,
)

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
V4_ANCHOR_TEMPORAL = 0.40401
V6_LOO9_TEMPORAL = 0.40896
SUBMISSION_THRESHOLD = 0.40950

CACHE_TRAIN = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_train.parquet"
CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"

V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V6_MODELS_DIR = ROOT / "models" / "v6_20260516_163559"
V6_LOO9_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]

OUT = ROOT / "diagnostics" / f"phase7_batch_{TIMESTAMP}"
ERRORS_DIR = OUT / "errors"
PREDS_DIR = OUT / "predictions"
MODELS_DIR = OUT / "models"
IMP_DIR = OUT / "feature_importances"


# ============================================================================
# Logging + atomic IO
# ============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_write_df(df: pd.DataFrame, path: Path) -> None:
    """Atomic CSV write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def safe_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ============================================================================
# Feature builders (each returns: new_cols dict, drift dict)
# ============================================================================
def _drift_dict(name: str, tr: np.ndarray, va: np.ndarray) -> dict:
    tr = np.asarray(tr, dtype=np.float64)
    va = np.asarray(va, dtype=np.float64)
    tmean = float(np.nanmean(tr))
    vmean = float(np.nanmean(va))
    tstd = float(np.nanstd(tr))
    vstd = float(np.nanstd(va))
    tnull = float(np.isnan(tr).mean())
    vnull = float(np.isnan(va).mean())
    drift_ratio = abs(vmean - tmean) / (tstd + 1e-12)
    null_drift = abs(vnull - tnull)
    return {
        "feature": name,
        "train_mean": tmean,
        "val_mean": vmean,
        "train_std": tstd,
        "val_std": vstd,
        "train_null_rate": tnull,
        "val_null_rate": vnull,
        "abs_mean_delta_over_train_std": drift_ratio,
        "high_drift": bool(drift_ratio > 0.20),
        "high_null_drift": bool(null_drift > 0.10),
        "train_min": float(np.nanmin(tr)) if len(tr) else float("nan"),
        "train_max": float(np.nanmax(tr)) if len(tr) else float("nan"),
        "val_min": float(np.nanmin(va)) if len(va) else float("nan"),
        "val_max": float(np.nanmax(va)) if len(va) else float("nan"),
    }


def feat_premium_vs_prop_hist_x_short_window(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, list[dict]]:
    """premium_vs_prop_hist clipped to [0, P99] × 1/log1p(booking_window+1)
    plus a binary is_price_premium_x_short_window indicator."""
    p99 = float(np.nanquantile(train["price_vs_prop_mean"].clip(lower=0), 0.99))
    log(f"    premium clip P99 from temporal_train = {p99:.2f}")

    def _build(df: pd.DataFrame) -> dict:
        prem = df["price_vs_prop_mean"].astype(np.float64).clip(lower=0, upper=p99)
        short_w = 1.0 / np.log1p(df["srch_booking_window"].astype(np.float64) + 1.0)
        cont = (prem * short_w).astype(np.float32)
        binx = ((df["price_vs_prop_mean"] > 0) & (df["srch_booking_window"] <= 30)).astype(np.int8)
        return {
            "price_premium_vs_prop_hist_x_short_window": cont,
            "is_price_premium_x_short_window": binx,
        }

    tr = _build(train)
    va = _build(val)
    drifts = []
    for k in tr.keys():
        train[k] = tr[k]
        val[k] = va[k]
        drifts.append(_drift_dict(k, tr[k].values, va[k].values))
    return {"params": {"p99_clip": p99}, "new_cols": list(tr.keys())}, drifts


def feat_long_window_x_top_quartile_price(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, list[dict]]:
    def _build(df: pd.DataFrame) -> dict:
        is_long = (df["srch_booking_window"] > 30).astype(np.int8)
        is_topq = (df["price_rank_norm"] >= 0.75).astype(np.int8)
        return {"is_long_window_x_top_quartile_price": (is_long * is_topq).astype(np.int8)}

    tr = _build(train)
    va = _build(val)
    drifts = []
    for k in tr.keys():
        train[k] = tr[k]
        val[k] = va[k]
        drifts.append(_drift_dict(k, tr[k].values, va[k].values))
    return {"new_cols": list(tr.keys())}, drifts


def feat_prop_rare_x_long_trip(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, list[dict]]:
    def _build(df: pd.DataFrame) -> dict:
        rarity = 1.0 / np.log1p(df["prop_count"].astype(np.float64))
        long_strength = np.log1p(df["srch_length_of_stay"].astype(np.float64))
        return {"prop_rare_x_long_trip": (rarity * long_strength).astype(np.float32)}

    tr = _build(train)
    va = _build(val)
    drifts = []
    for k in tr.keys():
        train[k] = tr[k]
        val[k] = va[k]
        drifts.append(_drift_dict(k, tr[k].values, va[k].values))
    return {"new_cols": list(tr.keys())}, drifts


def feat_brand_x_domestic(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, list[dict]]:
    def _build(df: pd.DataFrame) -> dict:
        return {"brand_x_domestic": (df["prop_brand_bool"] * df["is_domestic"]).astype(np.int8)}

    tr = _build(train)
    va = _build(val)
    drifts = []
    for k in tr.keys():
        train[k] = tr[k]
        val[k] = va[k]
        drifts.append(_drift_dict(k, tr[k].values, va[k].values))
    return {"new_cols": list(tr.keys())}, drifts


def feat_query_difficulty_index(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, list[dict]]:
    def _build(df: pd.DataFrame) -> dict:
        ch = np.log1p(df["query_hotel_count"].astype(np.float64))
        de = 1.0 - df["dest_click_rate"].astype(np.float64)
        return {"query_difficulty_index": (ch * de).astype(np.float32)}

    tr = _build(train)
    va = _build(val)
    drifts = []
    for k in tr.keys():
        train[k] = tr[k]
        val[k] = va[k]
        drifts.append(_drift_dict(k, tr[k].values, va[k].values))
    return {"new_cols": list(tr.keys())}, drifts


# ============================================================================
# Variant specs
# ============================================================================
CORE_VARIANTS = [
    {"id": "price_premium_vs_prop_hist_x_short_window", "builder": feat_premium_vs_prop_hist_x_short_window},
    {"id": "is_long_window_x_top_quartile_price",       "builder": feat_long_window_x_top_quartile_price},
    {"id": "prop_rare_x_long_trip",                      "builder": feat_prop_rare_x_long_trip},
]
OPTIONAL_VARIANTS = [
    {"id": "brand_x_domestic",                           "builder": feat_brand_x_domestic},
    {"id": "query_difficulty_index",                     "builder": feat_query_difficulty_index},
]


# ============================================================================
# Training (V4_ANCHOR config) + eval
# ============================================================================
def label_remap(s: pd.Series) -> np.ndarray:
    return s.map({0: 0, 1: 1, 5: 2}).astype(np.int32).values


def train_v4anchor(train_feat: pd.DataFrame, val_feat: pd.DataFrame,
                    extra_cols: list[str], propensity) -> tuple[lgb.Booster, int, list[str]]:
    feat_cols = [c for c in get_feature_columns(train_feat) if c in val_feat.columns]
    leaked = set(feat_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden leaked: {leaked}"
    weights = compute_ipw_weights(train_feat, propensity, clip_hi=10.0, clip_lo=0.1)

    train_label = label_remap(train_feat["relevance"])
    val_label = label_remap(val_feat["relevance"])
    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    params = BASE_PARAMS.copy()
    params["label_gain"] = "0,1,15"

    ds_tr = lgb.Dataset(train_feat[feat_cols], label=train_label,
                        group=train_groups, weight=weights)
    ds_va = lgb.Dataset(val_feat[feat_cols], label=val_label,
                        group=val_groups, reference=ds_tr)
    model = lgb.train(
        params, ds_tr, num_boost_round=2000,
        valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
    )
    return model, int(model.best_iteration), feat_cols


def classify(delta: float) -> str:
    if delta < 0:           return "REJECT"
    if delta < 0.001:       return "HOLD"
    if delta < 0.002:       return "KEEP"
    return "STRONG_KEEP"


# ============================================================================
# Ensemble helpers (rank-average within srch_id)
# ============================================================================
def grouped_rank(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(pd.Series(srch_id), sort=False).rank(
        method="average", ascending=False
    ).values.astype(np.float32)


def metrics_from_avg_rank(val_feat: pd.DataFrame, avg_rank: np.ndarray) -> dict:
    inv = -avg_rank  # eval_metrics sorts descending, so invert
    return eval_metrics(val_feat, inv)


# ============================================================================
# Plan printing
# ============================================================================
def print_plan(do_optional: bool) -> None:
    n_var = len(CORE_VARIANTS) + (len(OPTIONAL_VARIANTS) if do_optional else 0)
    print("=" * 78)
    print(f"PHASE 7 BATCH — plan")
    print("=" * 78)
    print(f"Timestamp:          {TIMESTAMP}")
    print(f"Output dir:         {OUT}")
    print()
    print(f"Inputs read:")
    print(f"  features cache:   {CACHE_TRAIN}")
    print(f"                    {CACHE_VAL}")
    print(f"  V6 LOO-9 preds:   {V6_DIR}/val_pred_<member>.npy   (×{len(V6_LOO9_MEMBERS)})")
    print(f"                    test_pred_<member>.npy           (×{len(V6_LOO9_MEMBERS)})")
    print()
    print(f"Variants to test ({n_var}):")
    for v in CORE_VARIANTS:
        print(f"  CORE      - {v['id']}")
    if do_optional:
        for v in OPTIONAL_VARIANTS:
            print(f"  OPTIONAL  - {v['id']}")
    else:
        print(f"  (optional variants skipped: pass --include-optional to enable)")
    print()
    print(f"Per-variant work (resumable):")
    print(f"  1. build feature on train+val")
    print(f"  2. drift check  (HIGH_DRIFT if |Δμ|/σ > 0.20)")
    print(f"  3. train V4_ANCHOR config (lambdarank lg=0,1,15, IPW, max=2000, es=80)")
    print(f"  4. predict on val, eval NDCG@5/R@1/R@5/MBR")
    print(f"  5. save model, predictions, importance, drift, append to results.csv")
    print()
    print(f"Ensemble phase (after all variants done):")
    print(f"  - V6 LOO-9 baseline rank-avg")
    print(f"  - V6_LOO9 + each eligible feature member")
    print(f"  - V6_LOO9 + all eligible feature members")
    print(f"  - V6_LOO9 + top-2 eligible feature members")
    print(f"  - LOO on the best ensemble")
    print(f"  Eligibility: KEEP, or HOLD with Δ ≥ +0.0005 and not HIGH_DRIFT")
    print()
    print(f"Baselines:")
    print(f"  V4_ANCHOR_TEMPORAL = {V4_ANCHOR_TEMPORAL}")
    print(f"  V6_LOO9_TEMPORAL   = {V6_LOO9_TEMPORAL}")
    print(f"  SUBMISSION threshold = {SUBMISSION_THRESHOLD}  "
          f"(+{SUBMISSION_THRESHOLD - V6_LOO9_TEMPORAL:.5f} over V6 LOO-9)")
    print()
    print(f"Outputs written:")
    print(f"  {OUT}/README.md")
    print(f"  {OUT}/feature_variant_results.csv     (atomic; appended per variant)")
    print(f"  {OUT}/ensemble_results.csv")
    print(f"  {OUT}/leave_one_out.csv")
    print(f"  {OUT}/selected_feature_ideas.md")
    print(f"  {OUT}/predictions/val_pred_<id>.npy")
    print(f"  {OUT}/models/model_<id>.txt")
    print(f"  {OUT}/feature_importances/importance_<id>.csv")
    print(f"  {OUT}/feature_importances/drift_<id>.csv")
    print(f"  {OUT}/errors/ERROR_<id>.txt  (only if variant fails)")
    print(f"  {OUT}/errors/FATAL.txt       (only if whole script crashes)")
    print()
    print(f"Estimated runtime:")
    print(f"  3 core variants × ~5 min each = ~15 min")
    if do_optional:
        print(f"  + 2 optional × ~5 min = ~10 min")
    print(f"  + ensemble: ~3 min")
    print(f"  + submission (only if best ≥ {SUBMISSION_THRESHOLD}): ~40–60 min")
    print(f"  Total: 20-30 min if no submission, 60-90 min if submission triggered")
    print()
    print("=" * 78)


# ============================================================================
# Main per-variant runner
# ============================================================================
def run_variant(spec: dict, base_train: pd.DataFrame, base_val: pd.DataFrame,
                propensity, variant_results: list, results_path: Path) -> None:
    variant_id = spec["id"]
    model_path = MODELS_DIR / f"model_{variant_id}.txt"
    val_pred_path = PREDS_DIR / f"val_pred_{variant_id}.npy"
    imp_path = IMP_DIR / f"importance_{variant_id}.csv"
    drift_path = IMP_DIR / f"drift_{variant_id}.csv"

    log(f"\n--- variant: {variant_id} ---")

    row = {
        "variant_id": variant_id,
        "status": "pending",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        t0 = time.time()
        # Copy base parquets so we don't mutate them across variants.
        train_feat = base_train.copy()
        val_feat = base_val.copy()
        log(f"  base feature_count: train={train_feat.shape[1]}, val={val_feat.shape[1]}")

        # Build feature(s)
        builder = spec["builder"]
        info, drift_list = builder(train_feat, val_feat)
        log(f"  added {len(info['new_cols'])} feature(s): {info['new_cols']}")
        drift_df = pd.DataFrame(drift_list)
        safe_write_df(drift_df, drift_path)
        high_drift = bool(drift_df["high_drift"].any())
        high_null_drift = bool(drift_df["high_null_drift"].any())
        log(f"  drift: max |Δμ|/σ = {drift_df['abs_mean_delta_over_train_std'].max():.4f}  "
            f"HIGH_DRIFT={high_drift}  HIGH_NULL_DRIFT={high_null_drift}")

        # Resume: if model + pred already exist, skip training/predict
        if model_path.exists() and val_pred_path.exists():
            log(f"  RESUME: model + val_pred already exist; reload + reevaluate")
            booster = lgb.Booster(model_file=str(model_path))
            scores = np.load(val_pred_path).astype(np.float32)
            best_iter = booster.current_iteration()
            feat_cols = booster.feature_name()
        else:
            booster, best_iter, feat_cols = train_v4anchor(
                train_feat, val_feat, info["new_cols"], propensity
            )
            booster.save_model(str(model_path))
            scores = booster.predict(val_feat[feat_cols]).astype(np.float32)
            np.save(val_pred_path, scores)

        m = eval_metrics(val_feat, scores)
        delta_anchor = float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL
        decision = classify(delta_anchor)

        # Save importance
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        safe_write_df(imp_df, imp_path)

        # Rank of new features
        rank_lookup = {f: i + 1 for i, f in enumerate(imp_df["feature"].tolist())}
        new_feat_ranks = {c: rank_lookup.get(c, -1) for c in info["new_cols"]}

        row.update({
            "status": "ok",
            "n_features": len(feat_cols),
            "best_iter": int(best_iter),
            "ndcg5": float(m["ndcg5"]),
            "recall1": float(m["recall1"]),
            "recall5": float(m["recall5"]),
            "mean_booked_rank": float(m["mean_booked_rank"]),
            "delta_vs_v4_anchor": delta_anchor,
            "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
            "decision": decision,
            "high_drift": high_drift,
            "high_null_drift": high_null_drift,
            "max_drift_ratio": float(drift_df["abs_mean_delta_over_train_std"].max()),
            "new_feature_ranks": json.dumps(new_feat_ranks),
            "elapsed_min": round((time.time() - t0) / 60, 2),
            "model_path": str(model_path),
            "val_pred_path": str(val_pred_path),
        })
        log(f"  ✓ NDCG@5={m['ndcg5']:.5f}  Δ_anchor={delta_anchor:+.5f}  "
            f"Δ_v6={row['delta_vs_v6_loo9']:+.5f}  best_iter={best_iter}  → {decision}")
        log(f"  new feature ranks: {new_feat_ranks}")

        del booster, train_feat, val_feat, scores
        gc.collect()
    except Exception as e:
        log(f"  ✗ FAILED: {e}")
        err_text = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        safe_write_text(err_text, ERRORS_DIR / f"ERROR_{variant_id}.txt")
        row.update({
            "status": f"failed:{type(e).__name__}",
            "error_message": str(e)[:300],
        })

    variant_results.append(row)
    # Atomic append to results CSV
    try:
        safe_write_df(pd.DataFrame(variant_results), results_path)
    except Exception as e:
        log(f"  ! could not save results csv: {e}")


# ============================================================================
# Ensemble phase
# ============================================================================
def load_v6_loo9_val_rank(srch_id_arr: np.ndarray) -> np.ndarray:
    """Rank-average rank of the 9 V6 LOO members on val."""
    ranks = []
    for m in V6_LOO9_MEMBERS:
        p = V6_DIR / f"val_pred_{m}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        s = np.load(p).astype(np.float32)
        ranks.append(grouped_rank(srch_id_arr, s))
    return np.mean(ranks, axis=0)


def ensemble_phase(val_feat: pd.DataFrame, variant_results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    log("\n" + "=" * 60)
    log("ENSEMBLE PHASE")
    log("=" * 60)

    srch = val_feat["srch_id"].values
    log("Building V6 LOO-9 baseline rank-average…")
    v6_rank = load_v6_loo9_val_rank(srch)
    v6_metrics = metrics_from_avg_rank(val_feat, v6_rank)
    log(f"  V6_LOO9 reproduced: NDCG@5={v6_metrics['ndcg5']:.5f}  "
        f"(target {V6_LOO9_TEMPORAL})")

    # Eligibility: status==ok AND (KEEP/STRONG_KEEP) OR (HOLD with delta >= +0.0005 and not high_drift)
    eligible = []
    for r in variant_results:
        if r.get("status") != "ok":
            continue
        d = r.get("decision")
        delta = r.get("delta_vs_v4_anchor", 0)
        hd = r.get("high_drift", False)
        is_eligible = d in ("KEEP", "STRONG_KEEP") or (d == "HOLD" and delta >= 0.0005 and not hd)
        log(f"  {r['variant_id']:50s} Δ={delta:+.5f} decision={d:11s} "
            f"high_drift={hd}  eligible={is_eligible}")
        if is_eligible:
            eligible.append(r["variant_id"])

    # Load eligible variant rank arrays
    variant_ranks: dict[str, np.ndarray] = {}
    for vid in eligible:
        p = PREDS_DIR / f"val_pred_{vid}.npy"
        s = np.load(p).astype(np.float32)
        variant_ranks[vid] = grouped_rank(srch, s)

    # Build ensembles
    rows = []

    def score(name: str, members: list[str], extra_ranks: list[np.ndarray]):
        all_r = [v6_rank] + extra_ranks
        avg = np.mean(all_r, axis=0)
        m = metrics_from_avg_rank(val_feat, avg)
        rows.append({
            "ensemble": name,
            "n_members": len(members),
            "members": "+".join(members),
            "ndcg5": float(m["ndcg5"]),
            "recall1": float(m["recall1"]),
            "recall5": float(m["recall5"]),
            "mean_booked_rank": float(m["mean_booked_rank"]),
            "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
            "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
        })
        log(f"  {name:60s} n={len(members)+1:2d}  NDCG@5={m['ndcg5']:.5f}  "
            f"Δ_v6={float(m['ndcg5']) - V6_LOO9_TEMPORAL:+.5f}")
        return rows[-1]

    # V6 baseline alone
    score("v6_loo9_baseline", ["v6_loo9"], [])
    # V6 + each
    per_member: list[tuple[str, float]] = []
    for vid in eligible:
        res = score(f"v6_loo9_plus_{vid}", ["v6_loo9", vid], [variant_ranks[vid]])
        per_member.append((vid, res["ndcg5"]))
    # V6 + all
    if eligible:
        score("v6_loo9_plus_all", ["v6_loo9"] + eligible, [variant_ranks[v] for v in eligible])
    # V6 + top-2
    if len(eligible) >= 2:
        top2 = sorted(per_member, key=lambda x: -x[1])[:2]
        ids = [t[0] for t in top2]
        score("v6_loo9_plus_top2", ["v6_loo9"] + ids, [variant_ranks[v] for v in ids])

    ens_df = pd.DataFrame(rows).sort_values("ndcg5", ascending=False).reset_index(drop=True)
    safe_write_df(ens_df, OUT / "ensemble_results.csv")

    # LOO on best
    best = ens_df.iloc[0]
    best_members = best["members"].split("+")
    best_ndcg = float(best["ndcg5"])
    log(f"\nBest ensemble: {best['ensemble']}  NDCG@5={best_ndcg:.5f}  "
        f"({len(best_members)} members)")
    loo_rows = []
    if len(best_members) > 1:
        for drop in best_members:
            remaining = [m for m in best_members if m != drop]
            extra = []
            for m in remaining:
                if m == "v6_loo9":
                    continue
                extra.append(variant_ranks[m])
            if "v6_loo9" in remaining:
                avg = np.mean([v6_rank] + extra, axis=0)
            else:
                avg = np.mean(extra, axis=0) if extra else None
            if avg is None:
                continue
            mret = metrics_from_avg_rank(val_feat, avg)
            loo_rows.append({
                "dropped": drop,
                "n_remaining": len(remaining),
                "ndcg5": float(mret["ndcg5"]),
                "delta_vs_best": float(mret["ndcg5"]) - best_ndcg,
            })
            log(f"  drop {drop:50s} ndcg={mret['ndcg5']:.5f}  Δ={float(mret['ndcg5']) - best_ndcg:+.5f}")
    loo_df = pd.DataFrame(loo_rows).sort_values("delta_vs_best", ascending=True) if loo_rows else pd.DataFrame()
    if len(loo_df):
        safe_write_df(loo_df, OUT / "leave_one_out.csv")
    return ens_df, loo_df, eligible, best_ndcg


# ============================================================================
# Conditional submission
# ============================================================================
def build_submission(best_ensemble_row: dict, eligible: list[str],
                     variant_results: list[dict]) -> str:
    """Retrain selected new feature models on full train; build rank-average submission."""
    log("\n" + "=" * 60)
    log("SUBMISSION PHASE (best ensemble exceeded threshold)")
    log("=" * 60)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = sub_dir / f"submission_phase7_{TIMESTAMP}.csv"
    sub_meta = OUT / "submission_meta.json"

    members = best_ensemble_row["members"].split("+")
    new_feat_members = [m for m in members if m != "v6_loo9"]
    log(f"  members for submission: v6_loo9 + {new_feat_members}")

    # Load V6 LOO-9 test rank-average using existing test_pred .npy files
    log("  loading V6 LOO-9 test predictions…")
    v6_test_preds = []
    for m in V6_LOO9_MEMBERS:
        p = V6_DIR / f"test_pred_{m}.npy"
        if not p.exists():
            raise FileNotFoundError(f"V6 LOO-9 test prediction missing: {p}")
        v6_test_preds.append(np.load(p).astype(np.float32))
    # Need test alignment (srch_id, prop_id ordering)
    log("  loading test set (for srch_id alignment)…")
    test_raw = load_test()
    test_raw = test_raw.reset_index(drop=True)
    test_srch = test_raw["srch_id"].values
    v6_test_ranks = np.mean(
        [grouped_rank(test_srch, p) for p in v6_test_preds], axis=0
    )
    log(f"  V6 LOO-9 test rank-average computed for {len(test_raw):,} rows")

    # For new feature members: need to retrain on FULL train, predict on test
    # First build features
    log("  building full-train + test features (this takes ~5-8 min)…")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    t = time.time()
    train_full = build_features(train_raw, agg_source=train_raw, is_train=True)
    log(f"    train_full features ready in {(time.time()-t)/60:.1f} min "
        f"(cols={train_full.shape[1]})")
    t = time.time()
    test_full = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"    test_full features ready in {(time.time()-t)/60:.1f} min "
        f"(cols={test_full.shape[1]})")
    del train_raw, test_raw
    gc.collect()

    # IPW propensity for full train
    propensity = compute_position_propensity(train_full)

    new_test_ranks = []
    for vid in new_feat_members:
        log(f"  retraining {vid} on full train…")
        r = next((rr for rr in variant_results if rr["variant_id"] == vid), None)
        best_iter = int(r.get("best_iter", 500))
        spec = next((s for s in CORE_VARIANTS + OPTIONAL_VARIANTS if s["id"] == vid), None)
        # Build feature on full train + test
        tr2 = train_full.copy()
        te2 = test_full.copy()
        # The builders need `prop_count` etc.; they all use base feature cols which exist post-build_features.
        try:
            spec["builder"](tr2, te2)
        except Exception as e:
            log(f"    ✗ feature build failed for {vid}: {e}")
            log(f"    skipping this member; will rank-average without it")
            continue
        feat_cols = [c for c in get_feature_columns(tr2) if c in te2.columns]
        weights = compute_ipw_weights(tr2, propensity, clip_hi=10.0, clip_lo=0.1)
        label = label_remap(tr2["relevance"])
        groups = make_group_counts(tr2)
        params = BASE_PARAMS.copy()
        params["label_gain"] = "0,1,15"
        ds_tr = lgb.Dataset(tr2[feat_cols], label=label, group=groups, weight=weights)
        booster = lgb.train(params, ds_tr, num_boost_round=best_iter,
                            callbacks=[lgb.log_evaluation(0)])
        booster.save_model(str(MODELS_DIR / f"model_{vid}_FULL.txt"))
        test_scores = booster.predict(te2[feat_cols]).astype(np.float32)
        np.save(PREDS_DIR / f"test_pred_{vid}.npy", test_scores)
        new_test_ranks.append(grouped_rank(test_srch, test_scores))
        log(f"    ✓ {vid} retrained ({best_iter} rounds), saved test_pred + FULL model")
        del booster, tr2, te2
        gc.collect()

    # Rank-average V6 + new feature ranks
    all_ranks = [v6_test_ranks] + new_test_ranks
    avg_rank = np.mean(all_ranks, axis=0)
    sub_df = pd.DataFrame({
        "srch_id": test_srch,
        "prop_id": test_full["prop_id"].values,
        "_rk": avg_rank,
    }).sort_values(["srch_id", "_rk"])[["srch_id", "prop_id"]]
    sub_df.to_csv(sub_csv, index=False)
    log(f"\n✓ submission written: {sub_csv}")
    log(f"  rows={len(sub_df):,}  searches={sub_df['srch_id'].nunique():,}")

    safe_write_json({
        "submission_csv": str(sub_csv),
        "members": members,
        "best_ensemble_ndcg5_temporal": float(best_ensemble_row["ndcg5"]),
        "timestamp": TIMESTAMP,
    }, sub_meta)
    return str(sub_csv)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true",
                        help="Also run the 2 optional variants (brand_x_domestic, query_difficulty_index)")
    parser.add_argument("--plan", action="store_true",
                        help="Print plan and exit without running")
    args = parser.parse_args()

    print_plan(args.include_optional)
    if args.plan:
        return

    # Create dirs lazily inside main()
    for d in (OUT, ERRORS_DIR, PREDS_DIR, MODELS_DIR, IMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log(f"PHASE 7 BATCH start — {TIMESTAMP}")
    log(f"output dir: {OUT}")

    # Sanity checks on inputs
    assert CACHE_TRAIN.exists() and CACHE_VAL.exists(), "feature cache missing"
    for m in V6_LOO9_MEMBERS:
        p = V6_DIR / f"val_pred_{m}.npy"
        assert p.exists(), f"missing V6 LOO-9 val pred: {p}"

    log("Loading cached temporal_train + temporal_val features…")
    base_train = pd.read_parquet(CACHE_TRAIN).sort_values("srch_id").reset_index(drop=True)
    base_val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    log(f"  train rows={len(base_train):,}  val rows={len(base_val):,}  "
        f"base cols={base_train.shape[1]}")

    log("Computing position propensity (for IPW)…")
    propensity = compute_position_propensity(base_train)

    variants_to_run = list(CORE_VARIANTS)
    if args.include_optional:
        variants_to_run += list(OPTIONAL_VARIANTS)

    variant_results: list[dict] = []
    results_path = OUT / "feature_variant_results.csv"

    for spec in variants_to_run:
        run_variant(spec, base_train, base_val, propensity, variant_results, results_path)
        elapsed = (time.time() - t_start) / 60
        log(f"  cumulative elapsed: {elapsed:.1f} min")

    n_ok = sum(1 for r in variant_results if r.get("status") == "ok")
    log(f"\nVariant phase done. ok={n_ok}/{len(variant_results)}  "
        f"({(time.time() - t_start)/60:.1f} min elapsed)")

    # Ensemble phase
    if n_ok == 0:
        log("\nNo successful variants — skipping ensemble + submission.")
        safe_write_text("No variant produced a valid model. See errors/.",
                        OUT / "selected_feature_ideas.md")
        write_readme(variant_results, None, [], None, "no_variants_ok", t_start)
        return

    ens_df, loo_df, eligible, best_ndcg = ensemble_phase(base_val, variant_results)

    # Submission decision
    sub_status = "below_threshold"
    sub_csv = None
    best_row = ens_df.iloc[0].to_dict()
    log(f"\nbest ensemble NDCG@5 = {best_ndcg:.5f}  "
        f"(submission threshold {SUBMISSION_THRESHOLD})")
    if best_ndcg >= SUBMISSION_THRESHOLD:
        try:
            sub_csv = build_submission(best_row, eligible, variant_results)
            sub_status = "ok"
        except Exception as e:
            log(f"submission FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                            ERRORS_DIR / "SUBMISSION_ERROR.txt")
            sub_status = f"failed:{type(e).__name__}"
    else:
        log("  best ensemble below threshold — no submission produced.")

    # Final summaries
    write_selected_ideas(variant_results, ens_df, best_row, sub_status)
    write_readme(variant_results, ens_df, eligible, best_row, sub_status, t_start, sub_csv)

    log(f"\n=== ALL DONE in {(time.time() - t_start)/60:.1f} min ===")
    log(f"outputs: {OUT}")


def write_selected_ideas(variant_results, ens_df, best_row, sub_status):
    lines = ["# Phase 7 — Selected feature ideas\n"]
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()}_\n")
    lines.append("## Variant outcomes\n")
    df = pd.DataFrame(variant_results)
    if "ndcg5" in df.columns:
        df = df.sort_values("ndcg5", ascending=False)
    cols = ["variant_id", "status", "ndcg5", "delta_vs_v4_anchor", "decision",
            "best_iter", "high_drift", "max_drift_ratio"]
    cols = [c for c in cols if c in df.columns]
    lines.append("```")
    lines.append(df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("```\n")
    if ens_df is not None:
        lines.append("## Ensemble outcomes\n```")
        lines.append(ens_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        lines.append("```\n")
        lines.append(f"**Best ensemble:** `{best_row['ensemble']}` — NDCG@5 = {best_row['ndcg5']:.5f}")
        lines.append(f"- Δ vs V4_ANCHOR (0.40401): {best_row['delta_vs_v4_anchor']:+.5f}")
        lines.append(f"- Δ vs V6 LOO-9 (0.40896): {best_row['delta_vs_v6_loo9']:+.5f}\n")
    lines.append(f"## Submission status: **{sub_status}**\n")
    safe_write_text("\n".join(lines), OUT / "selected_feature_ideas.md")


def write_readme(variant_results, ens_df, eligible, best_row, sub_status, t_start, sub_csv=None):
    lines = [f"# Phase 7 batch — {TIMESTAMP}\n"]
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} • "
                 f"elapsed {(time.time() - t_start)/60:.1f} min_\n")
    lines.append(f"- V4_ANCHOR_TEMPORAL = {V4_ANCHOR_TEMPORAL}")
    lines.append(f"- V6_LOO9_TEMPORAL   = {V6_LOO9_TEMPORAL}")
    lines.append(f"- submission threshold = {SUBMISSION_THRESHOLD}  "
                 f"(+{SUBMISSION_THRESHOLD - V6_LOO9_TEMPORAL:.5f} over V6 LOO-9)\n")
    df = pd.DataFrame(variant_results)
    lines.append("## Variant results\n")
    if not df.empty:
        cols = ["variant_id", "status", "ndcg5", "delta_vs_v4_anchor",
                "delta_vs_v6_loo9", "decision", "best_iter", "n_features",
                "high_drift", "max_drift_ratio", "elapsed_min"]
        cols = [c for c in cols if c in df.columns]
        lines.append("```")
        lines.append(df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        lines.append("```\n")
    if ens_df is not None:
        lines.append("## Ensemble results\n```")
        lines.append(ens_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        lines.append("```\n")
        if best_row is not None:
            lines.append(f"**Best ensemble:** `{best_row['ensemble']}`  "
                         f"NDCG@5 = {best_row['ndcg5']:.5f}")
    lines.append(f"\n## Submission: **{sub_status}**\n")
    if sub_csv:
        lines.append(f"- CSV: `{sub_csv}`")
    safe_write_text("\n".join(lines), OUT / "README.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            (OUT / "errors").mkdir(parents=True, exist_ok=True)
            safe_write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                OUT / "errors" / "FATAL.txt",
            )
        except Exception:
            pass
        log(f"FATAL: {e}")
        raise
