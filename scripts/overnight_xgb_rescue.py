"""XGB rescue — retrain the 5 failed XGB models with NaN/Inf cleanup.

The overnight final batch failed all 5 XGB models with:
  "Input data contains `inf` or a value too large, while `missing` is not set to `inf`"

Fix: replace ±inf with NaN before DMatrix construction. XGB then treats them
as missing (same as LGBM/CatBoost behavior).

Resumable: skip any model whose val_pred .npy already exists.

After all 5 train: re-run ensemble search adding the XGB models to the existing
pool. Save ensemble_results_xgb.csv. If a new best is found, also write
submission_overnight_best_with_xgb_<TS>.csv (requires full-train XGB retrains).
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

from src.data_loader import load_train, load_test, make_target  # noqa: E402
from src.features import build_features, compute_position_propensity  # noqa: E402
from pipelines.overnight_final_batch import (  # noqa: E402
    grouped_rank, label_remap, feature_cols_for, metrics_from_avg_rank,
    weighted_rank, V6_DIR, V6_MEMBERS, CACHE_TRAIN, CACHE_VAL,
    V4_ANCHOR_TEMPORAL, V6_LOO9_TEMPORAL,
    safe_write_df, safe_write_text,
)
from pipelines.temporal_validation import make_group_counts, eval_metrics  # noqa: E402

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "diagnostics" / "overnight_final_batch_20260517_022323"
PREDS_DIR = RUN_DIR / "predictions"
MODELS_DIR = RUN_DIR / "models"
ERRORS_DIR = RUN_DIR / "errors"
SUB_DIR = ROOT / "submissions"

XGB_SPECS = [
    {"id": "xgb_rank_A", "target": "rank", "params": {
        "max_depth": 6, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
        "min_child_weight": 50, "lambda": 2.0,
    }},
    {"id": "xgb_rank_B_regularized", "target": "rank", "params": {
        "max_depth": 5, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
        "min_child_weight": 100, "lambda": 5.0,
    }},
    {"id": "xgb_rank_C_shallow", "target": "rank", "params": {
        "max_depth": 4, "eta": 0.04, "subsample": 0.8, "colsample_bytree": 0.7,
        "min_child_weight": 150, "lambda": 8.0,
    }},
    {"id": "xgb_booking_clf", "target": "booking_bool", "params": {
        "max_depth": 5, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
        "min_child_weight": 100, "lambda": 5.0,
    }},
    {"id": "xgb_click_clf", "target": "click_bool", "params": {
        "max_depth": 5, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
        "min_child_weight": 100, "lambda": 5.0,
    }},
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_inf(arr: np.ndarray) -> np.ndarray:
    """Replace ±inf with NaN so XGB treats them as missing."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = ~np.isfinite(arr)
    if mask.any():
        log(f"    cleaned {int(mask.sum()):,} non-finite values → NaN")
        arr = arr.copy()
        arr[mask] = np.nan
    return arr


def train_xgb(spec, train, val):
    import xgboost as xgb
    feat_cols = feature_cols_for(None, train, val)
    X_tr = clean_inf(train[feat_cols].to_numpy(dtype=np.float32))
    X_va = clean_inf(val[feat_cols].to_numpy(dtype=np.float32))

    if spec["target"] == "rank":
        label = label_remap(train["relevance"])
        label_v = label_remap(val["relevance"])
        group_tr = make_group_counts(train)
        group_va = make_group_counts(val)
        dtrain = xgb.DMatrix(X_tr, label=label, feature_names=feat_cols, missing=np.nan)
        dtrain.set_group(group_tr)
        dval = xgb.DMatrix(X_va, label=label_v, feature_names=feat_cols, missing=np.nan)
        dval.set_group(group_va)
        params = {"objective": "rank:ndcg", "eval_metric": "ndcg@5",
                   "tree_method": "hist", "seed": 42, "verbosity": 1}
    else:
        label = train[spec["target"]].values.astype(np.int32)
        label_v = val[spec["target"]].values.astype(np.int32)
        dtrain = xgb.DMatrix(X_tr, label=label, feature_names=feat_cols, missing=np.nan)
        dval = xgb.DMatrix(X_va, label=label_v, feature_names=feat_cols, missing=np.nan)
        params = {"objective": "binary:logistic", "eval_metric": "logloss",
                   "tree_method": "hist", "seed": 42, "verbosity": 1}

    for k, v in spec["params"].items():
        params[k] = v
    model = xgb.train(params, dtrain, num_boost_round=2000,
                       evals=[(dval, "val")], early_stopping_rounds=80, verbose_eval=0)
    scores = model.predict(dval, iteration_range=(0, model.best_iteration + 1)).astype(np.float32)
    return model, int(model.best_iteration), feat_cols, scores


def main():
    log(f"XGB RESCUE — {TIMESTAMP}")
    log(f"  source batch: {RUN_DIR}")
    log(f"  reusing predictions/models/errors dirs")

    log("Loading cached features…")
    base_train = pd.read_parquet(CACHE_TRAIN).sort_values("srch_id").reset_index(drop=True)
    base_val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    log(f"  train={len(base_train):,}  val={len(base_val):,}")

    new_rows = []
    for spec in XGB_SPECS:
        mid = spec["id"]
        val_pred_path = PREDS_DIR / f"val_pred_{mid}.npy"
        model_path = MODELS_DIR / f"model_{mid}.json"
        log(f"\n--- {mid} ---")
        try:
            t0 = time.time()
            if val_pred_path.exists() and model_path.exists():
                log("  RESUME: predictions + model exist")
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(model_path))
                scores = np.load(val_pred_path).astype(np.float32)
                best_iter = -1
            else:
                model, best_iter, feat_cols, scores = train_xgb(spec, base_train, base_val)
                # Validate
                assert scores.shape == (len(base_val),), f"bad shape {scores.shape}"
                assert np.isfinite(scores).all(), f"non-finite scores"
                np.save(val_pred_path, scores)
                model.save_model(str(model_path))
            m = eval_metrics(base_val, scores)
            new_rows.append({
                "model_id": mid, "framework": "xgb", "status": "ok",
                "best_iter": int(best_iter),
                "ndcg5": float(m["ndcg5"]),
                "recall1": float(m["recall1"]),
                "recall5": float(m["recall5"]),
                "mean_booked_rank": float(m["mean_booked_rank"]),
                "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
                "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
                "elapsed_min": round((time.time() - t0) / 60, 2),
            })
            log(f"  ✓ NDCG@5={m['ndcg5']:.5f}  Δ_v6={float(m['ndcg5']) - V6_LOO9_TEMPORAL:+.5f}  "
                f"best_iter={best_iter}  in {(time.time()-t0)/60:.1f} min")
            del model
            gc.collect()
        except Exception as e:
            log(f"  ✗ FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / f"ERROR_xgb_rescue_{mid}.txt")
            new_rows.append({"model_id": mid, "framework": "xgb",
                              "status": f"failed:{type(e).__name__}", "error": str(e)[:300]})

    safe_write_df(pd.DataFrame(new_rows), RUN_DIR / "xgb_rescue_results.csv")
    n_ok = sum(1 for r in new_rows if r.get("status") == "ok")
    log(f"\nXGB rescue done: {n_ok}/{len(new_rows)} ok")
    if n_ok == 0:
        log("No XGB models succeeded; skipping ensemble update")
        return

    # ---- Re-run ensemble search adding XGB ----
    log("\n=== Ensemble re-run with XGB included ===")
    srch = base_val["srch_id"].values
    v6_rank = np.mean([grouped_rank(srch, np.load(V6_DIR / f"val_pred_{m}.npy").astype(np.float32))
                        for m in V6_MEMBERS], axis=0)

    # Load winning batch members for top-3 deployable plus all XGB
    candidates = ["cb_rank_C_deeper", "xendcg_conservative", "cb_rank_A",
                   "xendcg_reg_seed42", "xendcg_reg_seed456", "xendcg_reg_seed123",
                   "reg_bal15_seed123", "ds_reg_seed456"]
    candidates += [r["model_id"] for r in new_rows if r.get("status") == "ok"]
    member_ranks = {}
    for mid in candidates:
        p = PREDS_DIR / f"val_pred_{mid}.npy"
        if p.exists():
            member_ranks[mid] = grouped_rank(srch, np.load(p).astype(np.float32))
    log(f"  pool size: {len(member_ranks)}")

    # XGB seed average
    xgb_ids = [r["model_id"] for r in new_rows if r.get("status") == "ok"
                and r["model_id"].startswith("xgb_rank")]
    if len(xgb_ids) >= 2:
        member_ranks["XGB_RANK_AVG"] = np.mean([member_ranks[m] for m in xgb_ids], axis=0)
        log(f"  XGB_RANK_AVG composed from {xgb_ids}")

    rows = []
    def score(name, members, weights):
        try:
            mr = [member_ranks[m] for m in members]
            avg = weighted_rank(v6_rank, mr, weights)
            m = metrics_from_avg_rank(base_val, avg)
            row = {"test_id": name, "n_members": 1 + len(members),
                   "members_added": "+".join(members),
                   "weights_added": ",".join(f"{w:.4f}" for w in weights),
                   "v6_weight": 1.0 - sum(weights),
                   "ndcg5": float(m["ndcg5"]),
                   "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
                   "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
                   }
            rows.append(row)
            log(f"  {name:80s} NDCG={m['ndcg5']:.5f}  Δ_v6={float(m['ndcg5'])-V6_LOO9_TEMPORAL:+.5f}")
            return row
        except Exception as e:
            log(f"  ✗ {name}: {e}")
            return None

    # Baseline
    score("v6_loo9_baseline", [], [])
    # Each XGB alone
    for mid in xgb_ids:
        for w in (0.03, 0.05, 0.075, 0.10, 0.15):
            score(f"v6+{mid}@w={w:.3f}", [mid], [w])
    if "XGB_RANK_AVG" in member_ranks:
        for w in (0.05, 0.075, 0.10, 0.15, 0.20):
            score(f"v6+XGB_RANK_AVG@w={w:.3f}", ["XGB_RANK_AVG"], [w])
    # XGB binary classifiers alone
    for mid in [r["model_id"] for r in new_rows if r.get("status") == "ok"
                 and r["model_id"].endswith("_clf")]:
        for w in (0.03, 0.05, 0.075):
            score(f"v6+{mid}@w={w:.3f}", [mid], [w])

    # Add XGB to existing deployable best (cb_C_deeper + xen_cons + cb_A + xen_42)
    base_combo = ["cb_rank_C_deeper", "xendcg_conservative", "cb_rank_A", "xendcg_reg_seed42"]
    have_all = all(m in member_ranks for m in base_combo)
    if have_all and "XGB_RANK_AVG" in member_ranks:
        for w_base in (0.05, 0.05, 0.075):
            for w_xgb in (0.05, 0.075, 0.10):
                if 4 * w_base + w_xgb > 0.35:
                    continue
                weights = [w_base] * 4 + [w_xgb]
                score(f"v6+best_deployable@{w_base}each+XGB_AVG@{w_xgb}",
                       base_combo + ["XGB_RANK_AVG"], weights)
        # Try also with best single XGB
        if xgb_ids:
            best_single_xgb = max(xgb_ids, key=lambda mid: float([r for r in new_rows
                                                                    if r["model_id"] == mid][0]["ndcg5"]))
            for w_base in (0.05, 0.075):
                for w_xgb in (0.03, 0.05, 0.075):
                    if 4 * w_base + w_xgb > 0.35:
                        continue
                    weights = [w_base] * 4 + [w_xgb]
                    score(f"v6+best_deployable@{w_base}each+{best_single_xgb}@{w_xgb}",
                           base_combo + [best_single_xgb], weights)

    ens_df = pd.DataFrame(rows).sort_values("ndcg5", ascending=False).reset_index(drop=True)
    safe_write_df(ens_df, RUN_DIR / "ensemble_results_xgb_rescue.csv")
    log(f"\n=== Saved {len(ens_df)} ensemble tests ===")
    log("Top 10:")
    for _, r in ens_df.head(10).iterrows():
        log(f"  {r['test_id']:80s} NDCG={r['ndcg5']:.5f}  Δ_v6={r['delta_vs_v6_loo9']:+.5f}")

    log(f"\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                         ERRORS_DIR / "XGB_RESCUE_FATAL.txt")
        log(f"FATAL: {e}")
        raise
