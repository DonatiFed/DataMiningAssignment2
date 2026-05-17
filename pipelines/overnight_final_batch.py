"""Overnight final pre-submission batch.

7 phases:
  1. rank_xendcg diversity (5 seeds + 1 conservative variant)
  2. XGBoost rankers (3 variants)
  3. CatBoost rankers (3 variants)
  4. Binary classifiers (XGB booking, XGB click, CatBoost booking) — optional
  5. LGBM regularized seed expansion (regularized_bal15, CP_regularized,
     DS_regularized × 3 seeds each = 9 models)
  6. Controlled ensemble search (rank-average weighted with V6 LOO-9)
  7. Up to 3 submission candidates (best overall / diverse / conservative)

Bullet-proof isolation:
- Each model trains in its own try/except → errors/ERROR_<id>.txt on failure
- Each ensemble test in try/except → errors/ENS_ERROR_<id>.txt
- Each submission attempt in try/except → errors/SUB_ERROR_<id>.txt
- Outer try/except → errors/FATAL.txt
- Atomic CSV/JSON writes after every save
- Resumable: skip any model whose .txt/.json + val_pred.npy exist
- Predictions validated (no NaN/Inf, correct shape) before save
- Submissions validated (header, row count, srch_id set) before write
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_train, load_test, make_target, get_feature_columns  # noqa: E402
from src.features import build_features, compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402
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
STRUCTURAL_BEST = 0.40933  # v6 + rank_xendcg_regularized@0.10
SUBMISSION_THRESHOLD = 0.40950
NEAR_MISS_LO = 0.40920

CACHE_TRAIN = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_train.parquet"
CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"

V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V6_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]

STRUCTURAL_DIR = ROOT / "diagnostics" / "structural_batch_20260516_212040"
STRUCTURAL_PREDS = STRUCTURAL_DIR / "predictions"

OUT = ROOT / "diagnostics" / f"overnight_final_batch_{TIMESTAMP}"
ERRORS_DIR = OUT / "errors"
PREDS_DIR = OUT / "predictions"
MODELS_DIR = OUT / "models"
IMP_DIR = OUT / "feature_importances"
SUB_DIR = ROOT / "submissions"

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


def safe_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def validate_scores(scores: np.ndarray, expected_n: int, name: str) -> None:
    assert scores.shape == (expected_n,), f"{name} shape {scores.shape} != ({expected_n},)"
    n_nan = int(np.isnan(scores).sum())
    n_inf = int(np.isinf(scores).sum())
    assert n_nan == 0, f"{name} has {n_nan} NaN"
    assert n_inf == 0, f"{name} has {n_inf} Inf"


# ============================================================================
# Helpers
# ============================================================================
def label_remap(s: pd.Series) -> np.ndarray:
    return s.map({0: 0, 1: 1, 5: 2}).astype(np.int32).values


def feature_cols_for(extra: str | None, train_feat: pd.DataFrame, val_feat: pd.DataFrame) -> list[str]:
    cols = [c for c in get_feature_columns(train_feat) if c in val_feat.columns]
    extras_all = {"prop_click_rate_pos_adj_s40_oof", "prop_dest_book_rate_safe"}
    if extra == "CP":
        cols = [c for c in cols if c != "prop_dest_book_rate_safe"]
    elif extra == "DS":
        cols = [c for c in cols if c != "prop_click_rate_pos_adj_s40_oof"]
    else:
        cols = [c for c in cols if c not in extras_all]
    assert not (set(cols) & FORBIDDEN_FEATURES), "forbidden leaked"
    return cols


def grouped_rank(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(pd.Series(srch_id), sort=False).rank(
        method="average", ascending=False
    ).values.astype(np.float32)


def metrics_from_avg_rank(val_feat: pd.DataFrame, avg_rank: np.ndarray) -> dict:
    return eval_metrics(val_feat, -avg_rank)


# ============================================================================
# Per-framework training
# ============================================================================
def add_extra(extra: str | None, train_feat, val_feat):
    if extra == "CP":
        _pos_adj_oof_te(train_feat, val_feat, target_col="click_bool",
                        col_new="prop_click_rate_pos_adj_s40_oof",
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
    elif extra == "DS":
        _prop_dest_book_rate_safe(train_feat, val_feat,
                                   col_new="prop_dest_book_rate_safe",
                                   alpha=40.0, n_folds=5, seed=42)


def train_lgbm(spec, base_train, base_val, propensity):
    import lightgbm as lgb
    train = base_train.copy()
    val = base_val.copy()
    add_extra(spec.get("extra_feature"), train, val)
    feat_cols = feature_cols_for(spec.get("extra_feature"), train, val)
    if spec["weight"] == "ipw":
        weights = compute_ipw_weights(train, propensity, clip_hi=10.0, clip_lo=0.1)
    elif spec["weight"] == "none":
        weights = None
    else:
        raise ValueError(f"unknown weight: {spec['weight']}")

    params = BASE_PARAMS.copy()
    params["objective"] = spec["type"] if spec["type"] in ("lambdarank", "rank_xendcg") else "lambdarank"
    params["metric"] = "ndcg"
    params["label_gain"] = spec["label_gain"]
    params["seed"] = spec.get("seed", 456)
    for k, v in spec.get("params", {}).items():
        params[k] = v

    train_label = label_remap(train["relevance"])
    val_label = label_remap(val["relevance"])
    train_groups = make_group_counts(train)
    val_groups = make_group_counts(val)
    ds_tr = lgb.Dataset(train[feat_cols], label=train_label, group=train_groups, weight=weights)
    ds_va = lgb.Dataset(val[feat_cols], label=val_label, group=val_groups, reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=spec.get("num_boost_round", 2000),
                       valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
                       callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    scores = model.predict(val[feat_cols]).astype(np.float32)
    best_iter = int(model.best_iteration)
    return ("lgbm", model, best_iter, feat_cols, scores)


def train_xgb_rank(spec, base_train, base_val):
    import xgboost as xgb
    train = base_train.copy()
    val = base_val.copy()
    feat_cols = feature_cols_for(None, train, val)
    label = label_remap(train["relevance"])
    label_val = label_remap(val["relevance"])
    group_tr = make_group_counts(train)
    group_va = make_group_counts(val)

    dtrain = xgb.DMatrix(train[feat_cols].to_numpy(dtype=np.float32),
                          label=label, feature_names=feat_cols)
    dtrain.set_group(group_tr)
    dval = xgb.DMatrix(val[feat_cols].to_numpy(dtype=np.float32),
                        label=label_val, feature_names=feat_cols)
    dval.set_group(group_va)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "tree_method": "hist",
        "seed": spec.get("seed", 42),
        "verbosity": 1,
    }
    for k, v in spec.get("params", {}).items():
        params[k] = v

    evals = [(dval, "val")]
    model = xgb.train(params, dtrain, num_boost_round=spec.get("num_boost_round", 2000),
                       evals=evals, early_stopping_rounds=80, verbose_eval=0)
    scores = model.predict(dval, iteration_range=(0, model.best_iteration + 1)).astype(np.float32)
    return ("xgb", model, int(model.best_iteration), feat_cols, scores)


def train_xgb_binary(spec, base_train, base_val):
    import xgboost as xgb
    train = base_train.copy()
    val = base_val.copy()
    feat_cols = feature_cols_for(None, train, val)
    target_col = spec.get("target", "booking_bool")
    label = train[target_col].values.astype(np.int32)
    label_val = val[target_col].values.astype(np.int32)

    dtrain = xgb.DMatrix(train[feat_cols].to_numpy(dtype=np.float32),
                          label=label, feature_names=feat_cols)
    dval = xgb.DMatrix(val[feat_cols].to_numpy(dtype=np.float32),
                        label=label_val, feature_names=feat_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": spec.get("seed", 42),
        "verbosity": 1,
    }
    for k, v in spec.get("params", {}).items():
        params[k] = v
    model = xgb.train(params, dtrain, num_boost_round=spec.get("num_boost_round", 2000),
                       evals=[(dval, "val")], early_stopping_rounds=80, verbose_eval=0)
    scores = model.predict(dval, iteration_range=(0, model.best_iteration + 1)).astype(np.float32)
    return ("xgb_binary", model, int(model.best_iteration), feat_cols, scores)


def train_catboost_rank(spec, base_train, base_val):
    from catboost import CatBoost, Pool
    train = base_train.copy()
    val = base_val.copy()
    feat_cols = feature_cols_for(None, train, val)
    label = label_remap(train["relevance"])
    label_val = label_remap(val["relevance"])

    train_pool = Pool(train[feat_cols].to_numpy(dtype=np.float32), label=label,
                       group_id=train["srch_id"].values, feature_names=feat_cols)
    val_pool = Pool(val[feat_cols].to_numpy(dtype=np.float32), label=label_val,
                     group_id=val["srch_id"].values, feature_names=feat_cols)

    params = {
        "loss_function": spec.get("loss_function", "YetiRank"),
        "iterations": spec.get("iterations", 2000),
        "learning_rate": spec.get("learning_rate", 0.03),
        "depth": spec.get("depth", 6),
        "l2_leaf_reg": spec.get("l2_leaf_reg", 5),
        "random_seed": spec.get("seed", 42),
        "early_stopping_rounds": 80,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if spec.get("eval_metric"):
        params["eval_metric"] = spec["eval_metric"]
    for k, v in spec.get("params", {}).items():
        params[k] = v

    model = CatBoost(params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    scores = model.predict(val_pool).astype(np.float32)
    return ("catboost", model, int(model.tree_count_), feat_cols, scores)


def train_catboost_binary(spec, base_train, base_val):
    from catboost import CatBoostClassifier, Pool
    train = base_train.copy()
    val = base_val.copy()
    feat_cols = feature_cols_for(None, train, val)
    target_col = spec.get("target", "booking_bool")
    label = train[target_col].values.astype(np.int32)
    label_val = val[target_col].values.astype(np.int32)

    train_pool = Pool(train[feat_cols].to_numpy(dtype=np.float32), label=label, feature_names=feat_cols)
    val_pool = Pool(val[feat_cols].to_numpy(dtype=np.float32), label=label_val, feature_names=feat_cols)

    params = {
        "iterations": spec.get("iterations", 2000),
        "learning_rate": spec.get("learning_rate", 0.03),
        "depth": spec.get("depth", 5),
        "l2_leaf_reg": spec.get("l2_leaf_reg", 5),
        "random_seed": spec.get("seed", 42),
        "early_stopping_rounds": 80,
        "verbose": 0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "allow_writing_files": False,
    }
    for k, v in spec.get("params", {}).items():
        params[k] = v
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    scores = model.predict_proba(val_pool)[:, 1].astype(np.float32)
    return ("catboost_binary", model, int(model.tree_count_), feat_cols, scores)


# ============================================================================
# Save model (per framework)
# ============================================================================
def save_model_any(framework: str, model, path: Path) -> None:
    if framework == "lgbm":
        model.save_model(str(path))
    elif framework in ("xgb", "xgb_binary"):
        # JSON format
        model.save_model(str(path.with_suffix(".json")))
    elif framework in ("catboost", "catboost_binary"):
        model.save_model(str(path.with_suffix(".cbm")))
    else:
        raise ValueError(f"unknown framework: {framework}")


def model_files_exist(framework: str, path: Path) -> bool:
    if framework == "lgbm":
        return path.exists()
    if framework in ("xgb", "xgb_binary"):
        return path.with_suffix(".json").exists()
    if framework in ("catboost", "catboost_binary"):
        return path.with_suffix(".cbm").exists()
    return False


def load_model_any(framework: str, path: Path):
    if framework == "lgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    if framework in ("xgb", "xgb_binary"):
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(str(path.with_suffix(".json")))
        return m
    if framework == "catboost":
        from catboost import CatBoost
        m = CatBoost()
        m.load_model(str(path.with_suffix(".cbm")))
        return m
    if framework == "catboost_binary":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(str(path.with_suffix(".cbm")))
        return m
    raise ValueError(f"unknown framework: {framework}")


# ============================================================================
# Save feature importance (per framework)
# ============================================================================
def save_importance(framework: str, model, feat_cols: list[str], path: Path) -> None:
    try:
        if framework == "lgbm":
            df = pd.DataFrame({
                "feature": model.feature_name(),
                "gain": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
            }).sort_values("gain", ascending=False)
        elif framework in ("xgb", "xgb_binary"):
            gain_map = model.get_score(importance_type="gain")
            weight_map = model.get_score(importance_type="weight")
            df = pd.DataFrame({
                "feature": feat_cols,
                "gain": [gain_map.get(f, 0.0) for f in feat_cols],
                "weight": [weight_map.get(f, 0) for f in feat_cols],
            }).sort_values("gain", ascending=False)
        elif framework in ("catboost", "catboost_binary"):
            fi = model.get_feature_importance()
            df = pd.DataFrame({"feature": feat_cols, "importance": fi}).sort_values(
                "importance", ascending=False)
        else:
            return
        safe_write_df(df, path)
    except Exception as e:
        log(f"  ! importance save failed: {e}")


# ============================================================================
# Per-model runner (resumable, fault-isolated)
# ============================================================================
def run_one_model(spec: dict, base_train, base_val, propensity, rows, results_path):
    mid = spec["id"]
    framework = spec["framework"]
    model_path = MODELS_DIR / f"model_{mid}.txt"  # extension set by save_model_any
    val_pred_path = PREDS_DIR / f"val_pred_{mid}.npy"
    imp_path = IMP_DIR / f"importance_{mid}.csv"

    log(f"\n--- model: {mid}  [{framework}] ---")
    row = {"model_id": mid, "framework": framework, "type": spec.get("type"),
           "phase": spec.get("phase"), "seed": spec.get("seed"),
           "extra_feature": spec.get("extra_feature"), "status": "pending"}

    try:
        t0 = time.time()

        # Resume check
        if model_files_exist(framework, model_path) and val_pred_path.exists():
            log(f"  RESUME: model + val_pred exist")
            model = load_model_any(framework, model_path)
            scores = np.load(val_pred_path).astype(np.float32)
            best_iter = -1  # not easily recoverable for all frameworks
            feat_cols = (model.feature_name() if framework == "lgbm"
                          else feature_cols_for(spec.get("extra_feature"), base_train, base_val))
        else:
            if framework == "lgbm":
                framework_ret, model, best_iter, feat_cols, scores = train_lgbm(
                    spec, base_train, base_val, propensity
                )
            elif framework == "xgb":
                framework_ret, model, best_iter, feat_cols, scores = train_xgb_rank(spec, base_train, base_val)
            elif framework == "xgb_binary":
                framework_ret, model, best_iter, feat_cols, scores = train_xgb_binary(spec, base_train, base_val)
            elif framework == "catboost":
                framework_ret, model, best_iter, feat_cols, scores = train_catboost_rank(spec, base_train, base_val)
            elif framework == "catboost_binary":
                framework_ret, model, best_iter, feat_cols, scores = train_catboost_binary(spec, base_train, base_val)
            else:
                raise ValueError(f"unknown framework: {framework}")

            validate_scores(scores, len(base_val), mid)
            np.save(val_pred_path, scores)
            save_model_any(framework, model, model_path)

        m = eval_metrics(base_val, scores)
        delta_anchor = float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL
        save_importance(framework, model, feat_cols, imp_path)

        row.update({
            "status": "ok",
            "n_features": len(feat_cols),
            "best_iter": int(best_iter) if best_iter is not None else -1,
            "ndcg5": float(m["ndcg5"]),
            "recall1": float(m["recall1"]),
            "recall5": float(m["recall5"]),
            "mean_booked_rank": float(m["mean_booked_rank"]),
            "delta_vs_v4_anchor": delta_anchor,
            "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
            "elapsed_min": round((time.time() - t0) / 60, 2),
        })
        log(f"  ✓ NDCG@5={m['ndcg5']:.5f}  Δ_anchor={delta_anchor:+.5f}  "
            f"Δ_v6={row['delta_vs_v6_loo9']:+.5f}  best_iter={best_iter}  "
            f"in {(time.time()-t0)/60:.1f} min")
        del model
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
# Model specifications
# ============================================================================
def make_specs() -> list[dict]:
    specs = []
    # PHASE 1: rank_xendcg seeds + conservative
    base_xen = {
        "framework": "lgbm", "type": "rank_xendcg", "label_gain": "0,1,15",
        "weight": "ipw", "params": {
            "num_leaves": 300, "min_child_samples": 80, "reg_lambda": 2.0,
            "learning_rate": 0.03, "feature_fraction": 0.6,
            "bagging_fraction": 0.7, "bagging_freq": 1,
        },
    }
    for seed in (42, 123, 456, 789, 2024):
        specs.append({**base_xen, "id": f"xendcg_reg_seed{seed}", "seed": seed, "phase": 1})
    specs.append({
        "framework": "lgbm", "type": "rank_xendcg", "label_gain": "0,1,15",
        "weight": "ipw", "id": "xendcg_conservative", "phase": 1,
        "params": {
            "num_leaves": 200, "min_child_samples": 120, "reg_lambda": 5.0,
            "learning_rate": 0.03, "feature_fraction": 0.55,
            "bagging_fraction": 0.7, "bagging_freq": 1,
        }
    })

    # PHASE 2: XGBoost rankers
    specs += [
        {"framework": "xgb", "id": "xgb_rank_A", "phase": 2, "params": {
            "max_depth": 6, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
            "min_child_weight": 50, "lambda": 2.0,
        }},
        {"framework": "xgb", "id": "xgb_rank_B_regularized", "phase": 2, "params": {
            "max_depth": 5, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
            "min_child_weight": 100, "lambda": 5.0,
        }},
        {"framework": "xgb", "id": "xgb_rank_C_shallow", "phase": 2, "params": {
            "max_depth": 4, "eta": 0.04, "subsample": 0.8, "colsample_bytree": 0.7,
            "min_child_weight": 150, "lambda": 8.0,
        }},
    ]

    # PHASE 3: CatBoost rankers
    specs += [
        {"framework": "catboost", "id": "cb_rank_A", "phase": 3,
         "loss_function": "YetiRank", "eval_metric": "NDCG:top=5",
         "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2000},
        {"framework": "catboost", "id": "cb_rank_B_regularized", "phase": 3,
         "loss_function": "QueryRMSE", "eval_metric": "NDCG:top=5",
         "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 10, "iterations": 2000},
        {"framework": "catboost", "id": "cb_rank_C_deeper", "phase": 3,
         "loss_function": "YetiRank", "eval_metric": "NDCG:top=5",
         "depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 8, "iterations": 2000},
    ]

    # PHASE 4: Binary classifiers
    specs += [
        {"framework": "xgb_binary", "id": "xgb_booking_clf", "phase": 4,
         "target": "booking_bool", "params": {
            "max_depth": 5, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
            "min_child_weight": 100, "lambda": 5.0,
        }},
        {"framework": "xgb_binary", "id": "xgb_click_clf", "phase": 4,
         "target": "click_bool", "params": {
            "max_depth": 5, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
            "min_child_weight": 100, "lambda": 5.0,
        }},
        {"framework": "catboost_binary", "id": "cb_booking_clf", "phase": 4,
         "target": "booking_bool", "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5},
    ]

    # PHASE 5: LGBM regularized seed expansion
    reg_base = {
        "framework": "lgbm", "type": "lambdarank", "label_gain": "0,1,15",
        "weight": "ipw", "params": {
            "num_leaves": 250, "min_child_samples": 100, "reg_lambda": 3.0,
            "feature_fraction": 0.55, "bagging_fraction": 0.7, "bagging_freq": 1,
            "learning_rate": 0.03,
        }
    }
    for seed in (42, 123, 456):
        specs.append({**reg_base, "id": f"reg_bal15_seed{seed}", "seed": seed,
                      "extra_feature": None, "phase": 5})
        specs.append({**reg_base, "id": f"cp_reg_seed{seed}", "seed": seed,
                      "extra_feature": "CP", "phase": 5})
        specs.append({**reg_base, "id": f"ds_reg_seed{seed}", "seed": seed,
                      "extra_feature": "DS", "phase": 5})
    return specs


# ============================================================================
# Phase 6 ensemble search
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
    assert -0.001 < base_w <= 1.001, f"V6 weight out of range: {base_w}"
    out = base_w * v6_rank
    for w, r in zip(weights, member_ranks):
        out = out + w * r
    return out


def score_test(test_id: str, member_ids: list[str], weights: list[float],
                val_feat, v6_rank, member_ranks_map: dict, rows: list, phase_name: str):
    try:
        if member_ids:
            mr = [member_ranks_map[m] for m in member_ids]
            avg = weighted_rank(v6_rank, mr, weights)
        else:
            avg = v6_rank
        m = metrics_from_avg_rank(val_feat, avg)
        row = {
            "test_id": test_id,
            "phase": phase_name,
            "n_members": 1 + len(member_ids),
            "members_added": "+".join(member_ids),
            "weights_added": ",".join(f"{w:.4f}" for w in weights),
            "v6_weight": 1.0 - sum(weights),
            "total_added_weight": sum(weights),
            "ndcg5": float(m["ndcg5"]),
            "recall1": float(m["recall1"]),
            "recall5": float(m["recall5"]),
            "mean_booked_rank": float(m["mean_booked_rank"]),
            "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
            "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
        }
        rows.append(row)
        return row
    except Exception as e:
        log(f"  ✗ {test_id} FAILED: {e}")
        safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                         ERRORS_DIR / f"ENS_ERROR_{test_id[:80].replace('/', '_').replace('+', '_')}.txt")
        return None


def phase6_ensemble(model_rows, val_feat):
    log("\n" + "=" * 70)
    log("PHASE 6 — controlled ensemble search")
    log("=" * 70)
    srch = val_feat["srch_id"].values
    v6_rank = load_v6_rank(srch)
    v6_m = metrics_from_avg_rank(val_feat, v6_rank)
    log(f"V6 LOO-9 reproduced: NDCG@5={v6_m['ndcg5']:.5f}")

    # Also load existing structural batch members for extra diversity
    structural_extras = {}
    if STRUCTURAL_PREDS.exists():
        for name in ("rank_xendcg_regularized", "CP_regularized", "DS_regularized"):
            p = STRUCTURAL_PREDS / f"val_pred_{name}.npy"
            if p.exists():
                s = np.load(p).astype(np.float32)
                structural_extras[f"struct_{name}"] = grouped_rank(srch, s)
        log(f"  loaded {len(structural_extras)} structural batch extras")

    # Load this batch's predictions
    member_ranks_map = dict(structural_extras)  # start with extras
    for r in model_rows:
        if r.get("status") != "ok":
            continue
        p = PREDS_DIR / f"val_pred_{r['model_id']}.npy"
        if not p.exists():
            continue
        s = np.load(p).astype(np.float32)
        member_ranks_map[r["model_id"]] = grouped_rank(srch, s)
    log(f"  total member rank arrays in pool: {len(member_ranks_map)}")

    rows = []
    score_test("v6_loo9_baseline", [], [], val_feat, v6_rank, member_ranks_map, rows, "baseline")

    # ---- 1. V6 + each new model individually ----
    log("\n1. V6 + each new model individually")
    weights_1 = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
    best_single: dict[str, float] = {}
    for mid in member_ranks_map.keys():
        for w in weights_1:
            res = score_test(f"v6+{mid}@w={w:.3f}", [mid], [w],
                              val_feat, v6_rank, member_ranks_map, rows, "1_single")
            if res:
                cur = best_single.get(mid, -1)
                if res["ndcg5"] > cur:
                    best_single[mid] = res["ndcg5"]
    safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # Top single-member candidates
    ranked = sorted(best_single.items(), key=lambda x: -x[1])
    log("Top single-member ensembles (best NDCG by member):")
    for mid, n in ranked[:10]:
        log(f"  {mid:50s} best NDCG = {n:.5f}  Δ_v6 = {n - V6_LOO9_TEMPORAL:+.5f}")

    # ---- 2. V6 + rank_xendcg seed ensemble ----
    log("\n2. V6 + rank_xendcg seed ensemble")
    xen_seeds = [k for k in member_ranks_map if k.startswith("xendcg_reg_seed") or k == "xendcg_conservative"]
    if xen_seeds:
        xen_avg = np.mean([member_ranks_map[k] for k in xen_seeds], axis=0)
        member_ranks_map["XEN_SEED_AVG"] = xen_avg
        for w in (0.05, 0.075, 0.10, 0.15, 0.20, 0.25):
            score_test(f"v6+XEN_SEED_AVG[{len(xen_seeds)}]@w={w:.3f}", ["XEN_SEED_AVG"], [w],
                        val_feat, v6_rank, member_ranks_map, rows, "2_xendcg_seed_avg")
        safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # ---- 3. V6 + best XGB ----
    log("\n3. V6 + best XGBoost model")
    xgb_pool = [mid for mid in best_single if mid.startswith("xgb_rank_")]
    if xgb_pool:
        best_xgb = sorted(xgb_pool, key=lambda m: -best_single[m])[0]
        log(f"  best XGB rank = {best_xgb}")
        for w in (0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
            score_test(f"v6+{best_xgb}@w={w:.3f}_FOCUS", [best_xgb], [w],
                        val_feat, v6_rank, member_ranks_map, rows, "3_best_xgb")
        safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # ---- 4. V6 + best CatBoost ----
    log("\n4. V6 + best CatBoost model")
    cb_pool = [mid for mid in best_single if mid.startswith("cb_rank_")]
    if cb_pool:
        best_cb = sorted(cb_pool, key=lambda m: -best_single[m])[0]
        log(f"  best CatBoost = {best_cb}")
        for w in (0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
            score_test(f"v6+{best_cb}@w={w:.3f}_FOCUS", [best_cb], [w],
                        val_feat, v6_rank, member_ranks_map, rows, "4_best_catboost")
        safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # ---- 5. V6 + XEN_SEED_AVG + best XGB + best CatBoost ----
    log("\n5. V6 + XEN_SEED_AVG + best_xgb + best_catboost (small grid, total ≤ 0.30)")
    if "XEN_SEED_AVG" in member_ranks_map and xgb_pool and cb_pool:
        for w1, w2, w3 in product((0.05, 0.10), repeat=3):
            if w1 + w2 + w3 > 0.30:
                continue
            score_test(f"v6+XEN_SEED_AVG@{w1}+{best_xgb}@{w2}+{best_cb}@{w3}",
                        ["XEN_SEED_AVG", best_xgb, best_cb], [w1, w2, w3],
                        val_feat, v6_rank, member_ranks_map, rows, "5_triple_diversity")
        safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # ---- 6. V6 + XEN_SEED_AVG + CP_reg_seed_avg + DS_reg_seed_avg ----
    log("\n6. V6 + XEN + CP_reg_seed_avg + DS_reg_seed_avg")
    cp_seeds = [k for k in member_ranks_map if k.startswith("cp_reg_seed")]
    ds_seeds = [k for k in member_ranks_map if k.startswith("ds_reg_seed")]
    if cp_seeds:
        member_ranks_map["CP_REG_SEED_AVG"] = np.mean([member_ranks_map[k] for k in cp_seeds], axis=0)
    if ds_seeds:
        member_ranks_map["DS_REG_SEED_AVG"] = np.mean([member_ranks_map[k] for k in ds_seeds], axis=0)
    have = [k for k in ("XEN_SEED_AVG", "CP_REG_SEED_AVG", "DS_REG_SEED_AVG") if k in member_ranks_map]
    if len(have) >= 2:
        ws = (0.05, 0.075, 0.10)
        for wcombo in product(ws, repeat=len(have)):
            if sum(wcombo) > 0.30:
                continue
            score_test(f"v6+{'+'.join(f'{n}@{w}' for n, w in zip(have, wcombo))}",
                        have, list(wcombo), val_feat, v6_rank, member_ranks_map, rows, "6_seed_avg_triple")
        safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    # ---- 7. Candidate pool small combinations ----
    log("\n7. Candidate pool small combinations (max 5 components, total weight ≤ 0.35)")
    candidates = set()
    # criterion A: single NDCG ≥ 0.4040
    for r in model_rows:
        if r.get("status") == "ok" and r.get("ndcg5", 0) >= 0.4040:
            candidates.add(r["model_id"])
    # criterion B: improves V6 by ≥ +0.00010 at some small weight
    for mid, n in best_single.items():
        if n - V6_LOO9_TEMPORAL >= 0.00010:
            candidates.add(mid)
    candidates = sorted(candidates)
    log(f"  candidate pool ({len(candidates)}): {candidates}")
    # Bounded search: pick top by best_single, run small grid on those
    cand_ranked = sorted([c for c in candidates if c in best_single],
                          key=lambda m: -best_single[m])[:5]
    log(f"  evaluating combinations of top {len(cand_ranked)}: {cand_ranked}")
    if len(cand_ranked) >= 2:
        for n_sel in (2, 3, 4):
            if n_sel > len(cand_ranked):
                continue
            for w_each in (0.05, 0.075):
                if w_each * n_sel > 0.35:
                    continue
                from itertools import combinations
                for combo in combinations(cand_ranked, n_sel):
                    score_test(
                        f"v6+pool_{'+'.join(combo)}@{w_each}each",
                        list(combo), [w_each] * n_sel,
                        val_feat, v6_rank, member_ranks_map, rows, f"7_pool_n{n_sel}"
                    )
        safe_write_df(pd.DataFrame(rows), OUT / "ensemble_results.csv")

    ens_df = pd.DataFrame(rows).sort_values("ndcg5", ascending=False).reset_index(drop=True)
    safe_write_df(ens_df, OUT / "ensemble_results.csv")

    # LOO on top-3 ensembles
    log("\n--- LOO on top-3 ensembles ---")
    loo_rows = []
    for i, top_row in ens_df.head(3).iterrows():
        if top_row["n_members"] <= 1:
            continue
        members = top_row["members_added"].split("+")
        weights = [float(w) for w in top_row["weights_added"].split(",")]
        for j, drop in enumerate(members):
            rem_m = [m for k, m in enumerate(members) if k != j]
            rem_w = [w for k, w in enumerate(weights) if k != j]
            try:
                if rem_m:
                    mr = [member_ranks_map[m] for m in rem_m]
                    avg = weighted_rank(v6_rank, mr, rem_w)
                else:
                    avg = v6_rank
                mret = metrics_from_avg_rank(val_feat, avg)
                loo_rows.append({
                    "best_ensemble_rank": i + 1,
                    "best_ensemble_id": top_row["test_id"],
                    "dropped": drop,
                    "ndcg5": float(mret["ndcg5"]),
                    "delta_vs_best": float(mret["ndcg5"]) - float(top_row["ndcg5"]),
                })
            except Exception as e:
                log(f"  ✗ LOO drop {drop}: {e}")
    if loo_rows:
        loo_df = pd.DataFrame(loo_rows)
        safe_write_df(loo_df, OUT / "leave_one_out.csv")

    return ens_df, member_ranks_map, v6_rank, best_single


# ============================================================================
# PHASE 7 — submission candidates
# ============================================================================
def validate_submission(sub_df: pd.DataFrame, sample_df: pd.DataFrame, name: str) -> str:
    if list(sub_df.columns) != ["srch_id", "prop_id"]:
        return f"BAD HEADER: {list(sub_df.columns)}"
    if len(sub_df) != len(sample_df):
        return f"ROW COUNT: sub={len(sub_df):,}  sample={len(sample_df):,}"
    if sub_df["srch_id"].isna().any() or sub_df["prop_id"].isna().any():
        return "NaN in srch_id or prop_id"
    if set(sub_df["srch_id"].unique()) != set(sample_df["srch_id"].unique()):
        return "srch_id set mismatch with sample"
    if sub_df.duplicated().any():
        return f"{int(sub_df.duplicated().sum())} duplicate rows"
    return "OK"


def build_submission(name: str, best_row: dict, model_rows: list[dict],
                      v6_test_ranks: np.ndarray, test_full: pd.DataFrame,
                      test_srch: np.ndarray, propensity_full, train_full=None) -> dict:
    """Retrain selected new members on full train, predict on test, write submission.

    Returns dict with status + path.
    """
    sub_csv = SUB_DIR / f"submission_overnight_{name}_{TIMESTAMP}.csv"
    log(f"\n--- building submission '{name}' → {sub_csv.name} ---")
    log(f"  members: {best_row['members_added']}")
    members = best_row["members_added"].split("+")
    weights_str = best_row["weights_added"]
    weights = [float(w) for w in weights_str.split(",")] if weights_str else []
    new_members = [m for m in members if not m.startswith("V6") and m != "v6_loo9"]
    # Aggregate "AVG" members back to their components (skip for now — they were derived from this batch's seeds)
    # For simplicity, we EXCLUDE aggregate AVG names from per-model retrain; we'll rebuild them by retraining the constituent seeds.
    expand_map = {}
    for m in new_members:
        if m == "XEN_SEED_AVG":
            expand_map[m] = [r["model_id"] for r in model_rows
                              if r.get("status") == "ok" and (r["model_id"].startswith("xendcg_reg_seed")
                                                                or r["model_id"] == "xendcg_conservative")]
        elif m == "CP_REG_SEED_AVG":
            expand_map[m] = [r["model_id"] for r in model_rows
                              if r.get("status") == "ok" and r["model_id"].startswith("cp_reg_seed")]
        elif m == "DS_REG_SEED_AVG":
            expand_map[m] = [r["model_id"] for r in model_rows
                              if r.get("status") == "ok" and r["model_id"].startswith("ds_reg_seed")]
        else:
            expand_map[m] = [m]
    # Skip if any required model is structural-batch member (those need separate retrain on full train)
    needs_retrain = []
    for m in members:
        if m == "v6_loo9":
            continue
        for sub in expand_map.get(m, [m]):
            if sub.startswith("struct_"):
                log(f"  ! skipping submission '{name}' — depends on structural batch member {sub}")
                return {"status": "skipped_struct_dep", "name": name}
            if sub not in [r["model_id"] for r in model_rows if r.get("status") == "ok"]:
                log(f"  ! submission '{name}' — required member {sub} not in this batch (skip)")
                return {"status": f"skipped_missing_{sub}", "name": name}
            needs_retrain.append(sub)
    needs_retrain = sorted(set(needs_retrain))
    log(f"  members to retrain on full train: {needs_retrain}")

    # Retrain each needed member on full train
    test_ranks_per_member = {}
    for sub_mid in needs_retrain:
        try:
            full_pred_path = PREDS_DIR / f"test_pred_{sub_mid}.npy"
            if full_pred_path.exists():
                log(f"    reuse test_pred_{sub_mid}.npy")
                test_scores = np.load(full_pred_path).astype(np.float32)
            else:
                row = next(r for r in model_rows if r["model_id"] == sub_mid)
                spec = SPEC_BY_ID[sub_mid]
                best_iter = row.get("best_iter", 0) or 500
                t0 = time.time()
                framework = spec["framework"]
                if framework == "lgbm":
                    import lightgbm as lgb
                    tr = train_full.copy()
                    te = test_full.copy()
                    add_extra(spec.get("extra_feature"), tr, te)
                    feat_cols = feature_cols_for(spec.get("extra_feature"), tr, te)
                    if spec["weight"] == "ipw":
                        w = compute_ipw_weights(tr, propensity_full, clip_hi=10.0, clip_lo=0.1)
                    else:
                        w = None
                    params = BASE_PARAMS.copy()
                    params["objective"] = spec["type"]
                    params["metric"] = "ndcg"
                    params["label_gain"] = spec["label_gain"]
                    params["seed"] = spec.get("seed", 456)
                    for k, v in spec.get("params", {}).items():
                        params[k] = v
                    label = label_remap(tr["relevance"])
                    groups = make_group_counts(tr)
                    ds_tr = lgb.Dataset(tr[feat_cols], label=label, group=groups, weight=w)
                    model = lgb.train(params, ds_tr, num_boost_round=best_iter,
                                       callbacks=[lgb.log_evaluation(0)])
                    test_scores = model.predict(te[feat_cols]).astype(np.float32)
                elif framework == "xgb":
                    import xgboost as xgb
                    tr = train_full.copy()
                    te = test_full.copy()
                    feat_cols = feature_cols_for(None, tr, te)
                    label = label_remap(tr["relevance"])
                    group_tr = make_group_counts(tr)
                    dtrain = xgb.DMatrix(tr[feat_cols].to_numpy(dtype=np.float32),
                                          label=label, feature_names=feat_cols)
                    dtrain.set_group(group_tr)
                    dtest = xgb.DMatrix(te[feat_cols].to_numpy(dtype=np.float32),
                                         feature_names=feat_cols)
                    params = {"objective": "rank:ndcg", "tree_method": "hist",
                              "seed": spec.get("seed", 42), "verbosity": 1}
                    for k, v in spec.get("params", {}).items():
                        params[k] = v
                    model = xgb.train(params, dtrain, num_boost_round=best_iter, verbose_eval=0)
                    test_scores = model.predict(dtest).astype(np.float32)
                elif framework == "catboost":
                    from catboost import CatBoost, Pool
                    tr = train_full.copy()
                    te = test_full.copy()
                    feat_cols = feature_cols_for(None, tr, te)
                    label = label_remap(tr["relevance"])
                    train_pool = Pool(tr[feat_cols].to_numpy(dtype=np.float32), label=label,
                                       group_id=tr["srch_id"].values, feature_names=feat_cols)
                    test_pool = Pool(te[feat_cols].to_numpy(dtype=np.float32),
                                      group_id=te["srch_id"].values, feature_names=feat_cols)
                    params = {
                        "loss_function": spec.get("loss_function", "YetiRank"),
                        "iterations": best_iter,
                        "learning_rate": spec.get("learning_rate", 0.03),
                        "depth": spec.get("depth", 6),
                        "l2_leaf_reg": spec.get("l2_leaf_reg", 5),
                        "random_seed": spec.get("seed", 42),
                        "verbose": 0,
                        "allow_writing_files": False,
                    }
                    model = CatBoost(params)
                    model.fit(train_pool)
                    test_scores = model.predict(test_pool).astype(np.float32)
                else:
                    raise ValueError(f"submission retrain not supported for {framework}")
                np.save(full_pred_path, test_scores)
                log(f"    retrained {sub_mid} in {(time.time()-t0)/60:.1f} min "
                    f"({best_iter} rounds)")
                del model
                gc.collect()
            test_ranks_per_member[sub_mid] = grouped_rank(test_srch, test_scores)
        except Exception as e:
            log(f"    ✗ retrain {sub_mid} FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / f"SUB_RETRAIN_ERROR_{sub_mid}.txt")
            return {"status": f"failed_retrain:{sub_mid}", "name": name}

    # Build aggregate ranks for AVG members
    member_test_ranks = {}
    for m in members:
        if m == "v6_loo9":
            continue
        subs = expand_map[m]
        avail = [s for s in subs if s in test_ranks_per_member]
        if not avail:
            return {"status": f"no_subs_for_{m}", "name": name}
        member_test_ranks[m] = np.mean([test_ranks_per_member[s] for s in avail], axis=0)

    # Compose weighted rank on test
    member_only = [m for m in members if m != "v6_loo9"]
    base_w = 1.0 - sum(weights)
    avg = base_w * v6_test_ranks
    for m, w in zip(member_only, weights):
        avg = avg + w * member_test_ranks[m]

    # Sort and write
    sub_df = pd.DataFrame({
        "srch_id": test_srch,
        "prop_id": test_full["prop_id"].values,
        "_rk": avg,
    }).sort_values(["srch_id", "_rk"])[["srch_id", "prop_id"]]
    # Validate
    sample = pd.read_csv(ROOT / "data" / "submission_sample.csv")
    issue = validate_submission(sub_df, sample, name)
    if issue != "OK":
        log(f"  ✗ validation FAILED: {issue}")
        safe_write_text(f"validation issue: {issue}\nsub head:\n{sub_df.head()}",
                         ERRORS_DIR / f"SUB_VALIDATE_ERROR_{name}.txt")
        return {"status": f"validation_failed:{issue}", "name": name}

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_csv, index=False)

    readme_path = SUB_DIR / f"submission_overnight_{name}_{TIMESTAMP}_README.md"
    safe_write_text(
        f"# Submission {name} — {TIMESTAMP}\n\n"
        f"**Members:** {best_row['members_added']}\n"
        f"**Weights:** {best_row['weights_added']}\n"
        f"**V6 weight:** {best_row['v6_weight']:.4f}\n"
        f"**Temporal NDCG@5:** {best_row['ndcg5']:.5f}\n"
        f"**Δ vs V6 LOO-9 (0.40896):** {best_row['delta_vs_v6_loo9']:+.5f}\n"
        f"**Δ vs V4_ANCHOR (0.40401):** {best_row['delta_vs_v4_anchor']:+.5f}\n\n"
        f"## Validation\n"
        f"- header: `srch_id,prop_id` ✓\n"
        f"- rows: {len(sub_df):,} (matches sample)\n"
        f"- unique searches: {sub_df['srch_id'].nunique():,}\n"
        f"- duplicates: 0\n"
        f"- NaN: 0\n",
        readme_path
    )
    log(f"  ✓ submission written: {sub_csv.name}  ({len(sub_df):,} rows, {sub_df['srch_id'].nunique():,} searches)")
    return {"status": "ok", "name": name, "path": str(sub_csv), "ndcg5": best_row["ndcg5"]}


def phase7_submissions(ens_df, model_rows, val_feat) -> list[dict]:
    log("\n" + "=" * 70)
    log("PHASE 7 — submission candidates")
    log("=" * 70)
    cands = []

    # Identify candidates
    best_overall = ens_df.iloc[0]
    log(f"best overall: {best_overall['test_id']}  NDCG={best_overall['ndcg5']:.5f}")
    if best_overall["ndcg5"] < STRUCTURAL_BEST:
        log(f"  best overall ({best_overall['ndcg5']:.5f}) < structural best ({STRUCTURAL_BEST}) — no submissions built")
        return cands

    # Filter ensembles that include XGB or CatBoost member
    has_xgb_cb = ens_df["members_added"].fillna("").str.contains("xgb_|cb_|XGB|CB")
    diverse_pool = ens_df[has_xgb_cb & (ens_df["ndcg5"] >= STRUCTURAL_BEST)]
    best_diverse = diverse_pool.iloc[0] if len(diverse_pool) else None

    # Conservative: only XEN + reg LGBM members, no classifiers, no XGB/CB
    is_classifier = ens_df["members_added"].fillna("").str.contains("clf")
    has_xgb_cb_any = ens_df["members_added"].fillna("").str.contains("xgb_|cb_")
    conservative_pool = ens_df[~is_classifier & ~has_xgb_cb_any &
                                 (ens_df["ndcg5"] >= STRUCTURAL_BEST)]
    best_conservative = conservative_pool.iloc[0] if len(conservative_pool) else None

    # Load V6 test ranks (for all submissions)
    log("Loading V6 LOO-9 test predictions…")
    v6_test_preds = []
    for m in V6_MEMBERS:
        p = V6_DIR / f"test_pred_{m}.npy"
        if not p.exists():
            log(f"  ✗ MISSING {p}")
            return cands
        v6_test_preds.append(np.load(p).astype(np.float32))

    log("Loading test set + building features (this takes ~10 min)…")
    test_raw = load_test().reset_index(drop=True)
    train_raw = load_train()
    train_raw = make_target(train_raw).sort_values("srch_id").reset_index(drop=True)

    t = time.time()
    train_full = build_features(train_raw, agg_source=train_raw, is_train=True)
    log(f"  train_full features ready in {(time.time()-t)/60:.1f} min ({train_full.shape[1]} cols)")
    t = time.time()
    test_full = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"  test_full features ready in {(time.time()-t)/60:.1f} min ({test_full.shape[1]} cols)")
    del train_raw, test_raw
    gc.collect()
    test_srch = test_full["srch_id"].values
    v6_test_ranks = np.mean([grouped_rank(test_srch, p) for p in v6_test_preds], axis=0)
    propensity_full = compute_position_propensity(train_full)
    log(f"  v6 test rank-average ready")

    # 1. best overall
    if best_overall["ndcg5"] >= SUBMISSION_THRESHOLD:
        log(f"\nbuilding BEST_OVERALL submission")
        try:
            r = build_submission("best_overall", best_overall.to_dict(), model_rows,
                                  v6_test_ranks, test_full, test_srch, propensity_full, train_full)
            cands.append(r)
        except Exception as e:
            log(f"  ✗ best_overall FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / "SUB_ERROR_best_overall.txt")
            cands.append({"status": f"failed:{type(e).__name__}", "name": "best_overall"})
    else:
        log(f"\nbest_overall ({best_overall['ndcg5']:.5f}) < submission threshold ({SUBMISSION_THRESHOLD}) — skipping")

    # 2. best diverse
    if best_diverse is not None:
        diff = best_overall["ndcg5"] - best_diverse["ndcg5"]
        if best_diverse["ndcg5"] >= STRUCTURAL_BEST and diff <= 0.00025:
            log(f"\nbuilding BEST_DIVERSE submission ({best_diverse['ndcg5']:.5f}, gap {diff:.5f})")
            try:
                r = build_submission("best_diverse", best_diverse.to_dict(), model_rows,
                                      v6_test_ranks, test_full, test_srch, propensity_full, train_full)
                cands.append(r)
            except Exception as e:
                log(f"  ✗ best_diverse FAILED: {e}")
                safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                                 ERRORS_DIR / "SUB_ERROR_best_diverse.txt")

    # 3. best conservative
    if best_conservative is not None and best_conservative["ndcg5"] >= STRUCTURAL_BEST:
        log(f"\nbuilding BEST_CONSERVATIVE submission ({best_conservative['ndcg5']:.5f})")
        try:
            r = build_submission("best_conservative", best_conservative.to_dict(), model_rows,
                                  v6_test_ranks, test_full, test_srch, propensity_full, train_full)
            cands.append(r)
        except Exception as e:
            log(f"  ✗ best_conservative FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / "SUB_ERROR_best_conservative.txt")

    return cands


# ============================================================================
# README writer
# ============================================================================
def write_readme(model_df, ens_df, sub_cands, t_start):
    L = [f"# Overnight final batch — {TIMESTAMP}\n"]
    L.append(f"_Generated {datetime.now(timezone.utc).isoformat()} • "
             f"elapsed {(time.time() - t_start)/60:.1f} min_\n")
    L.append(f"## Baselines\n- V4_ANCHOR temporal: {V4_ANCHOR_TEMPORAL}\n"
             f"- V6 LOO-9 temporal: {V6_LOO9_TEMPORAL}\n"
             f"- structural best (V6 + rank_xendcg_regularized@0.10): {STRUCTURAL_BEST}\n"
             f"- submission threshold: {SUBMISSION_THRESHOLD}\n")

    L.append("## Single-model results (top 20 by NDCG)\n```")
    cols = ["model_id", "framework", "phase", "status", "ndcg5",
            "delta_vs_v4_anchor", "delta_vs_v6_loo9", "best_iter", "elapsed_min"]
    cols = [c for c in cols if c in model_df.columns]
    sdf = model_df.sort_values("ndcg5", ascending=False, na_position="last").head(20)
    L.append(sdf[cols].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    L.append("```\n")

    failed = model_df[model_df["status"].astype(str).str.startswith("failed", na=False)]
    if len(failed):
        L.append(f"### Failed models ({len(failed)})\n```")
        L.append(failed[["model_id", "framework", "status", "error"]].to_string(
            index=False) if "error" in failed.columns else failed[["model_id", "status"]].to_string(index=False))
        L.append("```\n")

    L.append("## Ensemble top-15\n```")
    L.append(ens_df.head(15)[["test_id", "n_members", "ndcg5",
                                "delta_vs_v6_loo9", "v6_weight"]].to_string(
        index=False, float_format=lambda x: f"{x:+.5f}"))
    L.append("```\n")

    best = ens_df.iloc[0]
    L.append("## 6. Best ensemble and weights\n")
    L.append(f"- **Test ID:** `{best['test_id']}`")
    L.append(f"- **NDCG@5:** {best['ndcg5']:.5f}")
    L.append(f"- **Members:** `{best['members_added']}`")
    L.append(f"- **Weights:** `{best['weights_added']}` (V6 = {best['v6_weight']:.4f})")
    L.append(f"- **Δ vs V6 LOO-9:** {best['delta_vs_v6_loo9']:+.5f}")
    L.append(f"- **Δ vs structural best (0.40933):** {best['ndcg5'] - STRUCTURAL_BEST:+.5f}\n")

    # Group answers
    L.append("## Q&A\n")
    xen_ndcg = model_df[model_df["model_id"].astype(str).str.startswith("xendcg_reg_seed")
                          | (model_df["model_id"] == "xendcg_conservative")]
    if len(xen_ndcg):
        L.append(f"### 1. Did rank_xendcg seeds improve the previous +0.00037?\n"
                 f"Best xendcg seed: {xen_ndcg['ndcg5'].max():.5f}, mean: {xen_ndcg['ndcg5'].mean():.5f}\n")

    xgb_rows = model_df[model_df["framework"].astype(str).str.startswith("xgb")]
    if len(xgb_rows):
        L.append(f"### 2. Did XGBoost add useful diversity?\n"
                 f"{len(xgb_rows)} XGB models trained, best NDCG: {xgb_rows['ndcg5'].max():.5f}\n")

    cb_rows = model_df[model_df["framework"].astype(str).str.startswith("catboost")]
    if len(cb_rows):
        L.append(f"### 3. Did CatBoost add useful diversity?\n"
                 f"{len(cb_rows)} CatBoost models trained, best NDCG: {cb_rows['ndcg5'].max():.5f}\n")

    clf_rows = model_df[model_df["model_id"].astype(str).str.contains("clf")]
    if len(clf_rows):
        L.append(f"### 4. Did binary classifiers help or hurt?\n"
                 f"{len(clf_rows)} classifiers trained, best NDCG: {clf_rows['ndcg5'].max():.5f}\n")

    reg_seed_rows = model_df[model_df["model_id"].astype(str).str.contains("_seed")]
    if len(reg_seed_rows):
        L.append(f"### 5. Did CP/DS/regularized seeds help?\n"
                 f"{len(reg_seed_rows)} seed-expansion models, best NDCG: {reg_seed_rows['ndcg5'].max():.5f}\n")

    L.append("### 7. Submission candidates ready\n")
    if sub_cands:
        for c in sub_cands:
            L.append(f"- **{c.get('name', '?')}:** status={c.get('status', '?')}  "
                     f"NDCG@5={c.get('ndcg5', float('nan')):.5f}  path=`{c.get('path', '')}`")
    else:
        L.append("- _No submissions were built (best ensemble did not clear required thresholds, or retrain failures)._\n")

    L.append("\n### 8. Recommended upload order tomorrow morning\n")
    if sub_cands:
        ok = [c for c in sub_cands if c.get("status") == "ok"]
        for i, c in enumerate(sorted(ok, key=lambda x: -x.get("ndcg5", 0)), 1):
            L.append(f"{i}. `{Path(c['path']).name}` (temporal {c['ndcg5']:.5f})")
    else:
        L.append("- No submissions to upload.\n")

    # Q9
    L.append("\n### 9. What to do next if none beats V4\n")
    if best["ndcg5"] >= SUBMISSION_THRESHOLD:
        L.append(f"Best temporal ensemble reached {best['ndcg5']:.5f} — worth uploading. "
                 "If Kaggle disappoints, the next levers are adversarial sample reweighting "
                 "and hard-negative mining as features.")
    else:
        L.append(f"Best temporal stuck at {best['ndcg5']:.5f}. Strong signal that we have hit "
                 "the ceiling of this feature set + LambdaRank/xendcg/XGB/CatBoost diversity. "
                 "The remaining lever is adversarial sample reweighting (V5 had adv_AUC=1.0). "
                 "If that fails, V4 0.42021 is the realistic ceiling.")

    safe_write_text("\n".join(L), OUT / "README.md")


# ============================================================================
# Main
# ============================================================================
SPEC_BY_ID = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-submission", action="store_true",
                        help="Skip Phase 7 submission building")
    args = parser.parse_args()

    for d in (OUT, ERRORS_DIR, PREDS_DIR, MODELS_DIR, IMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"OVERNIGHT FINAL BATCH — {TIMESTAMP}")
    log(f"out: {OUT}")

    # Sanity checks
    assert CACHE_TRAIN.exists() and CACHE_VAL.exists()
    for m in V6_MEMBERS:
        assert (V6_DIR / f"val_pred_{m}.npy").exists(), f"missing V6 pred {m}"

    log("Loading cached features…")
    base_train = pd.read_parquet(CACHE_TRAIN).sort_values("srch_id").reset_index(drop=True)
    base_val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    log(f"  train={len(base_train):,}  val={len(base_val):,}")

    log("Computing IPW propensity (for LGBM IPW)…")
    propensity = compute_position_propensity(base_train)

    specs = make_specs()
    for s in specs:
        SPEC_BY_ID[s["id"]] = s
    log(f"Total model specs: {len(specs)}")
    log(f"  by phase: " + ", ".join(f"P{p}={sum(1 for s in specs if s.get('phase')==p)}"
                                       for p in (1, 2, 3, 4, 5)))

    # ---- PHASES 1–5: train all models ----
    rows = []
    results_path = OUT / "model_results.csv"
    for spec in specs:
        run_one_model(spec, base_train, base_val, propensity, rows, results_path)
        log(f"  cumulative elapsed: {(time.time() - t_start)/60:.1f} min")
    safe_write_df(pd.DataFrame(rows), results_path)
    log(f"\nAll model training done. {sum(1 for r in rows if r.get('status')=='ok')}/{len(rows)} ok  "
        f"({(time.time() - t_start)/60:.1f} min)")

    # Free cached features memory before ensemble + submission phases
    del base_train
    gc.collect()

    # ---- PHASE 6: ensemble search ----
    try:
        ens_df, member_ranks_map, v6_rank, best_single = phase6_ensemble(rows, base_val)
    except Exception as e:
        log(f"PHASE 6 FAILED: {e}")
        safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                         ERRORS_DIR / "PHASE_6_ERROR.txt")
        write_readme(pd.DataFrame(rows), pd.DataFrame(), [], t_start)
        return

    # ---- PHASE 7: submissions ----
    sub_cands = []
    if not args.skip_submission:
        try:
            sub_cands = phase7_submissions(ens_df, rows, base_val)
        except Exception as e:
            log(f"PHASE 7 FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / "PHASE_7_ERROR.txt")
    if sub_cands:
        safe_write_df(pd.DataFrame(sub_cands), OUT / "submission_candidates.csv")

    write_readme(pd.DataFrame(rows), ens_df, sub_cands, t_start)
    log(f"\n=== ALL DONE in {(time.time() - t_start)/60:.1f} min ===")
    log(f"outputs: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            safe_write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                ERRORS_DIR / "FATAL.txt"
            )
        except Exception:
            pass
        log(f"FATAL: {e}")
        raise
