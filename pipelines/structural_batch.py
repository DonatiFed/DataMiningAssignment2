"""Structural diversity batch — V6-style ensemble with model/loss/weighting variants.

After Phase 7 weighted batch confirmed no Phase 7 micro-feature helps V6 LOO-9,
this batch tests STRUCTURAL diversity: label_gain shifts, weighting schemes,
regularization, and objective changes. Each is a separate LightGBM model
trained on temporal_train; the ensemble phase rank-averages them with V6 LOO-9
using small weights (same approach as phase7_weighted_batch.py).

Hard requirements:
- Temporal val only.
- Resumable: skip any model whose .txt + val_pred already exist.
- Per-model try/except; FATAL.txt on outer crash.
- Atomic CSV/JSON writes.
- No automatic submission.

Reuse from V6 batch (same configs already trained on temporal_train):
- no_ipw_bal15           → V6 lambdarank_noipw  (lg=0,1,15, no weights)
- random_upweight_bal15  → V6 lambdarank_randup (lg=0,2,25, random×2)
- booking_clf_calibrated → V6 booking_clf       (binary on booking_bool)

Outputs in diagnostics/structural_batch_<ts>/:
  member_results.csv
  ensemble_results.csv
  leave_one_out.csv
  README.md
  predictions/   models/   feature_importances/   errors/
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import get_feature_columns  # noqa: E402
from src.features import compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402
from pipelines.temporal_validation import (  # noqa: E402
    compute_ipw_weights, make_group_counts, eval_metrics, BASE_PARAMS,
)
from pipelines.evaluate_variant import _pos_adj_oof_te, _prop_dest_book_rate_safe  # noqa: E402

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
V4_ANCHOR_TEMPORAL = 0.40401
V6_LOO9_TEMPORAL = 0.40896
SUBMISSION_THRESHOLD = 0.40950
NEAR_MISS_LO = 0.40920
POOL_NDCG_MIN = 0.4035

CACHE_TRAIN = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_train.parquet"
CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"
V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V6_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]

OUT = ROOT / "diagnostics" / f"structural_batch_{TIMESTAMP}"
ERRORS_DIR = OUT / "errors"
PREDS_DIR = OUT / "predictions"
MODELS_DIR = OUT / "models"
IMP_DIR = OUT / "feature_importances"

# Models trained in V6 batch we can reuse directly.
REUSE_MAP = {
    "no_ipw_bal15":           {"v6_id": "lambdarank_noipw"},
    "random_upweight_bal15":  {"v6_id": "lambdarank_randup"},
    "booking_clf_calibrated": {"v6_id": "booking_clf"},
}


# ============================================================================
# Model specifications
# ============================================================================
# `params` override individual BASE_PARAMS keys.
# `weight`: ipw / none / ipw_clip3 / randup.
# `extra_feature`: CP / DS / None (only for OPTIONAL_MODELS).
MANDATORY_MODELS = [
    # GROUP A — label_gain variants
    {"id": "lg_0_1_10",                 "type": "lambdarank", "label_gain": "0,1,10",  "weight": "ipw",        "params": {}},
    {"id": "lg_0_1_20",                 "type": "lambdarank", "label_gain": "0,1,20",  "weight": "ipw",        "params": {}},
    {"id": "lg_0_2_15",                 "type": "lambdarank", "label_gain": "0,2,15",  "weight": "ipw",        "params": {}},
    {"id": "lg_0_3_15",                 "type": "lambdarank", "label_gain": "0,3,15",  "weight": "ipw",        "params": {}},
    # GROUP B — weighting variants
    {"id": "no_ipw_bal15",              "type": "lambdarank", "label_gain": "0,1,15",  "weight": "none",       "params": {}},
    {"id": "ipw_clip3_bal15",           "type": "lambdarank", "label_gain": "0,1,15",  "weight": "ipw_clip3",  "params": {}},
    {"id": "random_upweight_bal15",     "type": "lambdarank", "label_gain": "0,2,25",  "weight": "randup",     "params": {}},
    # GROUP C — regularization
    {"id": "regularized_bal15",         "type": "lambdarank", "label_gain": "0,1,15",  "weight": "ipw",        "params": {
        "num_leaves": 250, "min_child_samples": 100, "reg_lambda": 3.0,
        "feature_fraction": 0.55, "bagging_fraction": 0.7, "bagging_freq": 1,
    }},
    {"id": "low_lr_regularized_bal15",  "type": "lambdarank", "label_gain": "0,1,15",  "weight": "ipw",        "params": {
        "learning_rate": 0.02, "num_leaves": 300, "min_child_samples": 80,
        "reg_lambda": 2.0,
    }, "num_boost_round": 4000},
    # GROUP D — objective diversity
    {"id": "rank_xendcg_regularized",   "type": "xendcg",     "label_gain": "0,1,15",  "weight": "ipw",        "params": {
        "num_leaves": 300, "min_child_samples": 80, "reg_lambda": 2.0,
    }},
    {"id": "booking_clf_calibrated",    "type": "binary",     "label_gain": None,      "weight": "none",       "params": {}},
]

OPTIONAL_MODELS = [
    {"id": "CP_regularized", "type": "lambdarank", "label_gain": "0,1,15", "weight": "ipw",
     "extra_feature": "CP", "params": {
        "num_leaves": 250, "min_child_samples": 100, "reg_lambda": 3.0,
        "feature_fraction": 0.55, "bagging_fraction": 0.7, "bagging_freq": 1,
     }},
    {"id": "DS_regularized", "type": "lambdarank", "label_gain": "0,1,15", "weight": "ipw",
     "extra_feature": "DS", "params": {
        "num_leaves": 250, "min_child_samples": 100, "reg_lambda": 3.0,
        "feature_fraction": 0.55, "bagging_fraction": 0.7, "bagging_freq": 1,
     }},
]


# ============================================================================
# Logging + atomic IO
# ============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ============================================================================
# Helpers
# ============================================================================
def label_remap(s: pd.Series) -> np.ndarray:
    return s.map({0: 0, 1: 1, 5: 2}).astype(np.int32).values


def compute_weights(spec, train_feat, propensity):
    w = spec["weight"]
    if w == "none":
        return None
    if w == "ipw":
        return compute_ipw_weights(train_feat, propensity, clip_hi=10.0, clip_lo=0.1)
    if w == "ipw_clip3":
        return compute_ipw_weights(train_feat, propensity, clip_hi=3.0, clip_lo=0.1)
    if w == "randup":
        rb = train_feat["random_bool"].values
        return np.where(rb == 1, 2.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown weight scheme: {w}")


def add_extra_feature(extra: str | None, train_feat: pd.DataFrame, val_feat: pd.DataFrame) -> None:
    if extra == "CP":
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="click_bool",
                        col_new="prop_click_rate_pos_adj_s40_oof",
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
    elif extra == "DS":
        _prop_dest_book_rate_safe(train_feat, val_feat,
                                   col_new="prop_dest_book_rate_safe",
                                   alpha=40.0, n_folds=5, seed=42)


def grouped_rank(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(pd.Series(srch_id), sort=False).rank(
        method="average", ascending=False
    ).values.astype(np.float32)


def metrics_from_avg_rank(val_feat: pd.DataFrame, avg_rank: np.ndarray) -> dict:
    return eval_metrics(val_feat, -avg_rank)


def classify(delta: float) -> str:
    if delta < 0:           return "REJECT"
    if delta < 0.001:       return "HOLD"
    if delta < 0.002:       return "KEEP"
    return "STRONG_KEEP"


# ============================================================================
# Per-model training
# ============================================================================
def train_one(spec, train_feat, val_feat, propensity):
    extra = spec.get("extra_feature")
    if extra:
        add_extra_feature(extra, train_feat, val_feat)

    feat_cols = [c for c in get_feature_columns(train_feat) if c in val_feat.columns]
    # remove other extras the model shouldn't see
    extras_all = {"prop_click_rate_pos_adj_s40_oof", "prop_dest_book_rate_safe"}
    if extra == "CP":
        feat_cols = [c for c in feat_cols if c != "prop_dest_book_rate_safe"]
    elif extra == "DS":
        feat_cols = [c for c in feat_cols if c != "prop_click_rate_pos_adj_s40_oof"]
    else:
        feat_cols = [c for c in feat_cols if c not in extras_all]
    leaked = set(feat_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden leaked: {leaked}"

    weights = compute_weights(spec, train_feat, propensity)
    log(f"    n_features={len(feat_cols)}, weight={spec['weight']}, extra={extra}")
    if weights is not None:
        log(f"    weights: min={weights.min():.3f}  max={weights.max():.3f}  mean={weights.mean():.3f}")

    params = BASE_PARAMS.copy()
    if spec["type"] == "lambdarank":
        params["objective"] = "lambdarank"
        params["metric"] = "ndcg"
        params["label_gain"] = spec["label_gain"]
    elif spec["type"] == "xendcg":
        params["objective"] = "rank_xendcg"
        params["metric"] = "ndcg"
        params["label_gain"] = spec["label_gain"]
    elif spec["type"] == "binary":
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
        params.pop("eval_at", None)
    else:
        raise ValueError(f"unknown type: {spec['type']}")
    # Apply per-spec param overrides
    for k, v in spec.get("params", {}).items():
        params[k] = v
    log(f"    LGB params overrides: {spec.get('params', {})}")

    num_boost_round = spec.get("num_boost_round", 2000)

    if spec["type"] in ("lambdarank", "xendcg"):
        train_label = label_remap(train_feat["relevance"])
        val_label = label_remap(val_feat["relevance"])
        train_groups = make_group_counts(train_feat)
        val_groups = make_group_counts(val_feat)
        ds_tr = lgb.Dataset(train_feat[feat_cols], label=train_label,
                            group=train_groups, weight=weights)
        ds_va = lgb.Dataset(val_feat[feat_cols], label=val_label,
                            group=val_groups, reference=ds_tr)
    else:
        train_label = train_feat["booking_bool"].values.astype(np.int32)
        val_label = val_feat["booking_bool"].values.astype(np.int32)
        ds_tr = lgb.Dataset(train_feat[feat_cols], label=train_label, weight=weights)
        ds_va = lgb.Dataset(val_feat[feat_cols], label=val_label, reference=ds_tr)

    model = lgb.train(
        params, ds_tr, num_boost_round=num_boost_round,
        valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
    )
    return model, int(model.best_iteration), feat_cols


# ============================================================================
# Per-model runner
# ============================================================================
def run_one_model(spec, base_train, base_val, propensity, rows, results_path):
    mid = spec["id"]
    model_path = MODELS_DIR / f"model_{mid}.txt"
    val_pred_path = PREDS_DIR / f"val_pred_{mid}.npy"
    imp_path = IMP_DIR / f"importance_{mid}.csv"

    log(f"\n--- model: {mid} ---")
    row = {"model_id": mid, "type": spec["type"], "label_gain": spec.get("label_gain"),
           "weight": spec["weight"], "extra_feature": spec.get("extra_feature"),
           "params_override": json.dumps(spec.get("params", {})),
           "status": "pending"}

    try:
        t0 = time.time()

        # Reuse from V6 batch if applicable
        reused = False
        if mid in REUSE_MAP:
            v6_id = REUSE_MAP[mid]["v6_id"]
            v6_model = ROOT / "models" / "v6_20260516_163559" / f"model_{v6_id}.txt"
            v6_pred = V6_DIR / f"val_pred_{v6_id}.npy"
            if v6_model.exists() and v6_pred.exists():
                log(f"  REUSE: copying from V6 batch ({v6_id})")
                # Copy model + prediction into this batch dir for self-contained reproducibility
                booster = lgb.Booster(model_file=str(v6_model))
                booster.save_model(str(model_path))
                scores = np.load(v6_pred).astype(np.float32)
                np.save(val_pred_path, scores)
                best_iter = booster.current_iteration()
                feat_cols = booster.feature_name()
                reused = True

        # Local resume
        if not reused and model_path.exists() and val_pred_path.exists():
            log(f"  RESUME: local model+pred exist")
            booster = lgb.Booster(model_file=str(model_path))
            scores = np.load(val_pred_path).astype(np.float32)
            best_iter = booster.current_iteration()
            feat_cols = booster.feature_name()

        if not (reused or (model_path.exists() and val_pred_path.exists())):
            train_feat = base_train.copy()
            val_feat = base_val.copy()
            booster, best_iter, feat_cols = train_one(spec, train_feat, val_feat, propensity)
            booster.save_model(str(model_path))
            scores = booster.predict(val_feat[feat_cols]).astype(np.float32)
            np.save(val_pred_path, scores)
            del train_feat, val_feat
            gc.collect()

        # Eval
        m = eval_metrics(base_val, scores)
        delta_anchor = float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL
        decision = classify(delta_anchor)

        # Importance
        try:
            imp_df = pd.DataFrame({
                "feature": feat_cols,
                "gain": booster.feature_importance(importance_type="gain"),
                "split": booster.feature_importance(importance_type="split"),
            }).sort_values("gain", ascending=False)
            safe_write_df(imp_df, imp_path)
        except Exception as e:
            log(f"  ! importance save failed: {e}")

        row.update({
            "status": "ok",
            "reused_from_v6": reused,
            "n_features": len(feat_cols),
            "best_iter": int(best_iter),
            "ndcg5": float(m["ndcg5"]),
            "recall1": float(m["recall1"]),
            "recall5": float(m["recall5"]),
            "mean_booked_rank": float(m["mean_booked_rank"]),
            "delta_vs_v4_anchor": delta_anchor,
            "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
            "decision": decision,
            "elapsed_min": round((time.time() - t0) / 60, 2),
        })
        log(f"  ✓ NDCG@5={m['ndcg5']:.5f}  Δ_anchor={delta_anchor:+.5f}  "
            f"Δ_v6={row['delta_vs_v6_loo9']:+.5f}  best_iter={best_iter}  → {decision}  "
            f"{'(reused)' if reused else ''}")

        del booster
        gc.collect()
    except Exception as e:
        log(f"  ✗ FAILED: {e}")
        safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                         ERRORS_DIR / f"ERROR_{mid}.txt")
        row.update({"status": f"failed:{type(e).__name__}", "error": str(e)[:300]})

    rows.append(row)
    try:
        safe_write_df(pd.DataFrame(rows), results_path)
    except Exception as e:
        log(f"  ! results.csv save failed: {e}")


# ============================================================================
# Ensemble phase
# ============================================================================
def load_v6_rank(srch: np.ndarray) -> np.ndarray:
    ranks = []
    for m in V6_MEMBERS:
        p = V6_DIR / f"val_pred_{m}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        ranks.append(grouped_rank(srch, np.load(p).astype(np.float32)))
    return np.mean(ranks, axis=0)


def weighted_rank(v6_rank, member_ranks, weights):
    base_w = 1.0 - sum(weights)
    assert 0 < base_w <= 1.0, f"V6 weight out of range: {base_w}"
    out = base_w * v6_rank
    for w, r in zip(weights, member_ranks):
        out = out + w * r
    return out


def ensemble_phase(val_feat, member_results, t_start):
    log("\n" + "=" * 70)
    log("ENSEMBLE PHASE")
    log("=" * 70)

    srch = val_feat["srch_id"].values
    v6_rank = load_v6_rank(srch)
    v6_metrics = metrics_from_avg_rank(val_feat, v6_rank)
    log(f"V6 LOO-9 baseline reproduced: NDCG@5={v6_metrics['ndcg5']:.5f}")

    # Pool: status ok and NDCG >= POOL_NDCG_MIN
    ok = [r for r in member_results if r.get("status") == "ok"]
    pool = [r for r in ok if r.get("ndcg5", 0) >= POOL_NDCG_MIN]
    excluded = [r["model_id"] for r in ok if r not in pool]
    log(f"Pool: {len(pool)}/{len(ok)} models pass NDCG≥{POOL_NDCG_MIN} "
        f"(excluded: {excluded})")

    # Load each pool member's val pred → within-srch rank
    member_ranks = {}
    for r in pool:
        p = PREDS_DIR / f"val_pred_{r['model_id']}.npy"
        s = np.load(p).astype(np.float32)
        member_ranks[r["model_id"]] = grouped_rank(srch, s)

    rows = []

    def score(name, member_ids, weights):
        try:
            mr = [member_ranks[m] for m in member_ids]
            avg = weighted_rank(v6_rank, mr, weights)
            m = metrics_from_avg_rank(val_feat, avg)
            row = {
                "test_id": name, "n_members": 1 + len(member_ids),
                "members_added": "+".join(member_ids),
                "weights_added": ",".join(f"{w:.4f}" for w in weights),
                "v6_weight": 1.0 - sum(weights),
                "ndcg5": float(m["ndcg5"]),
                "recall1": float(m["recall1"]),
                "recall5": float(m["recall5"]),
                "mean_booked_rank": float(m["mean_booked_rank"]),
                "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
                "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
            }
            rows.append(row)
            log(f"  {name:70s} NDCG@5={m['ndcg5']:.5f}  "
                f"Δ_v6={float(m['ndcg5']) - V6_LOO9_TEMPORAL:+.5f}")
            return row
        except Exception as e:
            log(f"  ✗ {name} FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / f"ENS_ERROR_{name[:80].replace('/', '_')}.txt")
            return None

    # 1. baseline
    score("v6_loo9_baseline", [], [])
    safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # 2. new_models_only above median (rank-average of new models, NO V6)
    if pool:
        ndcgs = sorted([r["ndcg5"] for r in pool])
        median = ndcgs[len(ndcgs) // 2]
        above_median = [r["model_id"] for r in pool if r["ndcg5"] > median]
        log(f"  median pool NDCG = {median:.5f}; above-median set: {above_median}")
        if above_median:
            try:
                am_avg = np.mean([member_ranks[m] for m in above_median], axis=0)
                mr = metrics_from_avg_rank(val_feat, am_avg)
                rows.append({
                    "test_id": "new_models_only_above_median",
                    "n_members": len(above_median),
                    "members_added": "+".join(above_median),
                    "weights_added": "equal",
                    "v6_weight": 0.0,
                    "ndcg5": float(mr["ndcg5"]),
                    "recall1": float(mr["recall1"]),
                    "recall5": float(mr["recall5"]),
                    "mean_booked_rank": float(mr["mean_booked_rank"]),
                    "delta_vs_v6_loo9": float(mr["ndcg5"]) - V6_LOO9_TEMPORAL,
                    "delta_vs_v4_anchor": float(mr["ndcg5"]) - V4_ANCHOR_TEMPORAL,
                })
                log(f"  {'new_models_only_above_median':70s} NDCG@5={mr['ndcg5']:.5f}  "
                    f"Δ_v6={float(mr['ndcg5']) - V6_LOO9_TEMPORAL:+.5f}")
            except Exception as e:
                log(f"  ! new_models_only_above_median failed: {e}")
    safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # 3. V6 + each new model with weights 0.05/0.10/0.15/0.20
    log(f"\nSingle-member × 4 weights × {len(pool)} = {4 * len(pool)} tests")
    best_single = {}
    for r in pool:
        mid = r["model_id"]
        for w in (0.05, 0.10, 0.15, 0.20):
            res = score(f"v6+{mid}@w={w:.2f}", [mid], [w])
            if res and (mid not in best_single or res["ndcg5"] > best_single[mid]):
                best_single[mid] = res["ndcg5"]
    safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # 4. V6 + top-3 new models with grid {0.05, 0.10}, total ≤ 0.25
    top3 = sorted(best_single.items(), key=lambda x: -x[1])[:3]
    top3_ids = [m for m, _ in top3]
    log(f"\nTop-3 by best single-weight NDCG: {top3_ids}")
    if len(top3_ids) >= 2:
        ws = (0.05, 0.10)
        for w1, w2, w3 in product(ws, repeat=3):
            if w1 + w2 + w3 > 0.25:
                continue
            score(f"v6+{top3_ids[0]}@{w1}+{top3_ids[1]}@{w2}+{top3_ids[2]}@{w3}",
                  top3_ids, [w1, w2, w3])
    safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # 5. V6 + all above-median with equal total weight 0.20
    if pool and len(above_median) >= 1:
        n = len(above_median)
        per_w = 0.20 / n
        score(f"v6+all_above_median_equal_{per_w:.4f}", above_median, [per_w] * n)
    safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    ens_df = pd.DataFrame(rows).sort_values("ndcg5", ascending=False).reset_index(drop=True)
    safe_write_df(ens_df, OUT / "ensemble_results.csv")

    # 6. LOO on best
    best = ens_df.iloc[0]
    log(f"\nBest ensemble: {best['test_id']}  NDCG@5={best['ndcg5']:.5f}")
    loo_df = pd.DataFrame()
    if best["n_members"] > 1 and "+" in best.get("members_added", ""):
        members = best["members_added"].split("+")
        weights_str = best["weights_added"]
        if weights_str == "equal":
            n = len(members)
            weights = [(1.0 - best["v6_weight"]) / max(n, 1)] * n
        else:
            weights = [float(w) for w in weights_str.split(",")]
        loo_rows = []
        for i, drop in enumerate(members):
            rem_m = [m for j, m in enumerate(members) if j != i]
            rem_w = [w for j, w in enumerate(weights) if j != i]
            try:
                if best["v6_weight"] > 0:
                    mr = [member_ranks[m] for m in rem_m]
                    avg = weighted_rank(v6_rank, mr, rem_w) if rem_m else v6_rank
                else:
                    avg = np.mean([member_ranks[m] for m in rem_m], axis=0) if rem_m else v6_rank
                mret = metrics_from_avg_rank(val_feat, avg)
                loo_rows.append({
                    "dropped": drop,
                    "n_remaining": len(rem_m),
                    "ndcg5": float(mret["ndcg5"]),
                    "delta_vs_best": float(mret["ndcg5"]) - float(best["ndcg5"]),
                })
                log(f"  drop {drop:50s} NDCG@5={mret['ndcg5']:.5f}  "
                    f"Δ={float(mret['ndcg5']) - float(best['ndcg5']):+.5f}")
            except Exception as e:
                log(f"  ✗ LOO drop {drop} failed: {e}")
        loo_df = pd.DataFrame(loo_rows).sort_values("delta_vs_best")
        if len(loo_df):
            safe_write_df(loo_df, OUT / "leave_one_out.csv")

    return ens_df, loo_df, best


# ============================================================================
# README
# ============================================================================
def write_readme(member_df, ens_df, loo_df, best_row, sub_status, recommendation, t_start):
    L = [f"# Structural diversity batch — {TIMESTAMP}\n"]
    L.append(f"_Generated {datetime.now(timezone.utc).isoformat()} • "
             f"elapsed {(time.time() - t_start)/60:.1f} min_\n")
    L.append(f"## Baselines\n")
    L.append(f"- V4_ANCHOR_TEMPORAL = {V4_ANCHOR_TEMPORAL}")
    L.append(f"- V6_LOO9_TEMPORAL   = {V6_LOO9_TEMPORAL}")
    L.append(f"- submission threshold = {SUBMISSION_THRESHOLD}\n")

    L.append("## Single-model results\n")
    if "ndcg5" in member_df.columns:
        sorted_df = member_df.sort_values("ndcg5", ascending=False)
        cols = ["model_id", "status", "reused_from_v6", "ndcg5",
                "delta_vs_v4_anchor", "delta_vs_v6_loo9", "decision",
                "best_iter", "n_features", "elapsed_min"]
        cols = [c for c in cols if c in sorted_df.columns]
        L.append("```")
        L.append(sorted_df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        L.append("```\n")

    # Group winner analysis
    L.append("## 2. Which model family helped most?\n")
    groups = {
        "label_gain (A)": ["lg_0_1_10", "lg_0_1_20", "lg_0_2_15", "lg_0_3_15"],
        "weighting (B)":  ["no_ipw_bal15", "ipw_clip3_bal15", "random_upweight_bal15"],
        "regularization (C)": ["regularized_bal15", "low_lr_regularized_bal15"],
        "objective (D)":  ["rank_xendcg_regularized", "booking_clf_calibrated"],
    }
    for g, ids in groups.items():
        ndcgs = [r["ndcg5"] for r in member_df.to_dict("records")
                 if r.get("model_id") in ids and r.get("status") == "ok"]
        if ndcgs:
            L.append(f"- **{g}**: best NDCG@5 = {max(ndcgs):.5f}, "
                     f"mean = {sum(ndcgs)/len(ndcgs):.5f}")
    L.append("")

    L.append("## Ensemble top-10\n")
    if ens_df is not None and len(ens_df):
        top = ens_df.head(10)
        cols = ["test_id", "n_members", "ndcg5", "delta_vs_v6_loo9", "v6_weight"]
        cols = [c for c in cols if c in top.columns]
        L.append("```")
        L.append(top[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        L.append("```\n")

    L.append("## 4. Best temporal ensemble and weights\n")
    if best_row is not None:
        L.append(f"- **Test ID:** `{best_row.get('test_id')}`")
        L.append(f"- **NDCG@5:** {best_row.get('ndcg5'):.5f}")
        L.append(f"- **Members added:** `{best_row.get('members_added', '')}`")
        L.append(f"- **Weights:** `{best_row.get('weights_added', '')}` "
                 f"(V6 weight = {best_row.get('v6_weight', 1.0):.4f})")
        L.append(f"- **Δ vs V6 LOO-9:** {best_row.get('delta_vs_v6_loo9'):+.5f}")
        L.append(f"- **Δ vs V4_ANCHOR:** {best_row.get('delta_vs_v4_anchor'):+.5f}\n")

    if loo_df is not None and len(loo_df):
        L.append("## LOO on best ensemble\n```")
        L.append(loo_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        L.append("```\n")

    L.append("## 1. Did structural changes beat V6 LOO-9?\n")
    if best_row is not None:
        delta = float(best_row.get("delta_vs_v6_loo9", 0))
        if delta > 0:
            L.append(f"**YES** — best ensemble beat V6 by {delta:+.5f}.\n")
        else:
            L.append(f"**NO** — best ensemble was {delta:+.5f} relative to V6.\n")

    L.append("## 3. Did any single new model help as low-weight member?\n")
    if ens_df is not None and len(ens_df):
        helpers = ens_df[(ens_df["delta_vs_v6_loo9"] > 0) & (ens_df["v6_weight"] > 0) &
                          (ens_df["v6_weight"] < 1.0)]
        n = len(helpers)
        if n:
            top = helpers.iloc[0]
            L.append(f"{n} test(s) beat V6 baseline. "
                     f"Best low-weight test: `{top['test_id']}` "
                     f"(NDCG@5 = {top['ndcg5']:.5f}, Δ_v6 = {top['delta_vs_v6_loo9']:+.5f})\n")
        else:
            L.append("0 tests beat V6 baseline.\n")

    L.append(f"## Submission status: **{sub_status}**\n")
    L.append(f"## 5. Recommendation\n\n{recommendation}\n")
    safe_write_text("\n".join(L), OUT / "README.md")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true",
                        help="Run optional CP/DS_regularized models if time permits")
    args = parser.parse_args()

    for d in (OUT, ERRORS_DIR, PREDS_DIR, MODELS_DIR, IMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"STRUCTURAL BATCH — {TIMESTAMP}")
    log(f"out: {OUT}")

    assert CACHE_TRAIN.exists() and CACHE_VAL.exists()
    for m in V6_MEMBERS:
        assert (V6_DIR / f"val_pred_{m}.npy").exists(), f"missing V6 pred {m}"

    log("Loading cached features…")
    base_train = pd.read_parquet(CACHE_TRAIN).sort_values("srch_id").reset_index(drop=True)
    base_val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    log(f"  train={len(base_train):,}  val={len(base_val):,}")

    log("Computing IPW propensity…")
    propensity = compute_position_propensity(base_train)

    models_to_run = list(MANDATORY_MODELS)
    if args.include_optional:
        models_to_run += list(OPTIONAL_MODELS)

    rows = []
    results_path = OUT / "member_results.csv"
    for spec in models_to_run:
        run_one_model(spec, base_train, base_val, propensity, rows, results_path)
        elapsed = (time.time() - t_start) / 60
        log(f"  cumulative elapsed: {elapsed:.1f} min")

    member_df = pd.DataFrame(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    log(f"\nModel phase done: {n_ok}/{len(rows)} ok  "
        f"({(time.time() - t_start)/60:.1f} min)")
    if n_ok == 0:
        log("No models succeeded; aborting ensemble.")
        write_readme(member_df, None, None, None, "no_models",
                      "All models failed; investigate errors/.", t_start)
        return

    ens_df, loo_df, best = ensemble_phase(base_val, rows, t_start)

    # Submission decision
    best_ndcg = float(best["ndcg5"])
    if best_ndcg >= SUBMISSION_THRESHOLD:
        sub_status = "ABOVE_THRESHOLD_submission_recommended_but_not_built"
    elif best_ndcg >= NEAR_MISS_LO:
        sub_status = "near_miss_no_submission"
    else:
        sub_status = "below_threshold_no_submission"
    log(f"\nBest = {best_ndcg:.5f}  → {sub_status}")

    # Recommendation
    delta_v6 = best_ndcg - V6_LOO9_TEMPORAL
    n_above = sum(1 for r in rows if r.get("status") == "ok" and
                   r.get("ndcg5", 0) > V6_LOO9_TEMPORAL - 0.001)
    if delta_v6 >= 0.0005:
        recommendation = (
            f"Structural diversity improved over V6 by {delta_v6:+.5f}. "
            f"**Continue tuning** the winning families. Build a Kaggle submission "
            f"from the best ensemble (requires full-train retrain of new members).\n\n"
            f"_(Submission CSV not generated automatically — see Submission status.)_"
        )
    elif delta_v6 > 0:
        recommendation = (
            f"Best ensemble was {delta_v6:+.5f} over V6 — within noise. "
            f"**Recommended:** STOP further LGBM tuning. The remaining levers are "
            f"hard-negative mining, heterogeneous learners (XGBoost rank, "
            f"CatBoost listwise), or adversarial sample reweighting. "
            f"See `docs/next_steps.md`."
        )
    else:
        recommendation = (
            f"Structural changes did NOT beat V6 LOO-9. "
            f"V6 LOO-9 = {V6_LOO9_TEMPORAL} remains the best local ensemble.\n\n"
            f"**Strong recommendation:** STOP tuning LightGBM lambdarank within this "
            f"feature set. The remaining realistic moves are (in order of expected "
            f"leverage):\n"
            f"1. Heterogeneous base learners (XGBoost rank, CatBoost listwise)\n"
            f"2. Adversarial sample reweighting (V5 had adversarial AUC=1.0)\n"
            f"3. Hard-negative mining as features\n"
            f"V4 0.42021 Kaggle remains the production reference."
        )

    write_readme(member_df, ens_df, loo_df, best, sub_status, recommendation, t_start)
    log(f"\n=== ALL DONE in {(time.time() - t_start)/60:.1f} min ===")
    log(f"outputs: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / "FATAL.txt")
        except Exception:
            pass
        log(f"FATAL: {e}")
        raise
