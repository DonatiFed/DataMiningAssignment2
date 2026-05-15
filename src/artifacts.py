"""Artifact saving utilities for experiment tracking.

Layout per run_id:
  models/<run_id>/model_<config>.txt           - LightGBM booster
  artifacts/<run_id>/val_pred_<config>.npy     - validation predictions (float32)
  artifacts/<run_id>/val_meta.parquet          - srch_id, prop_id, relevance (shared across models)
  artifacts/<run_id>/importance_<config>.csv   - feature, gain, split
  artifacts/<run_id>/model_result_<config>.json - per-model metrics + params
  artifacts/<run_id>/model_results.csv         - one row per single model in this run
  artifacts/<run_id>/feature_cols.json         - feature column list
  artifacts/<run_id>/run_config.json           - run-level config (seed, split, etc.)
  artifacts/<run_id>/git_commit.txt            - HEAD commit at run time
  artifacts/<run_id>/val_pred_<ensemble>.npy   - ensemble validation predictions
  artifacts/<run_id>/ensemble_result_<name>.json - ensemble metadata + metrics
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_dirs(run_id):
    model_dir = ROOT_DIR / "models" / run_id
    art_dir = ROOT_DIR / "artifacts" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, art_dir


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def save_run_config(run_id, config_dict):
    _, art_dir = run_dirs(run_id)
    with open(art_dir / "run_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


def save_git_commit(run_id):
    _, art_dir = run_dirs(run_id)
    with open(art_dir / "git_commit.txt", "w") as f:
        f.write(get_git_commit() + "\n")


def save_feature_cols(run_id, feature_cols):
    _, art_dir = run_dirs(run_id)
    with open(art_dir / "feature_cols.json", "w") as f:
        json.dump(list(feature_cols), f, indent=2)


def save_val_meta(run_id, val_df):
    """Save validation metadata once per run."""
    _, art_dir = run_dirs(run_id)
    out = art_dir / "val_meta.parquet"
    if out.exists():
        return out
    cols = [c for c in ("srch_id", "prop_id", "relevance") if c in val_df.columns]
    val_df[cols].to_parquet(out, index=False)
    return out


def save_model_artifacts(run_id, config_name, model, val_pred, feature_cols, metrics, params):
    """Save per-model artifacts and append a row to model_results.csv."""
    model_dir, art_dir = run_dirs(run_id)

    model.save_model(str(model_dir / f"model_{config_name}.txt"))
    np.save(art_dir / f"val_pred_{config_name}.npy", np.asarray(val_pred, dtype=np.float32))

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df.to_csv(art_dir / f"importance_{config_name}.csv", index=False)

    best_iter = getattr(model, "best_iteration", None)
    result = {
        "config_name": config_name,
        "best_iter": int(best_iter) if best_iter else None,
        "n_features": len(feature_cols),
        "params": {k: str(v) for k, v in params.items()},
        **{k: _jsonable(v) for k, v in metrics.items()},
    }
    with open(art_dir / f"model_result_{config_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    csv_path = art_dir / "model_results.csv"
    row = {
        "config_name": config_name,
        "objective": params.get("objective"),
        "boosting": params.get("boosting_type"),
        "label_gain": params.get("label_gain", ""),
        "seed": params.get("seed"),
        "num_leaves": params.get("num_leaves"),
        "learning_rate": params.get("learning_rate"),
        "best_iter": result["best_iter"],
        "n_features": len(feature_cols),
        "ndcg5": metrics.get("ndcg5"),
        "recall1": metrics.get("recall1"),
        "recall5": metrics.get("recall5"),
        "mean_booked_rank": metrics.get("mean_booked_rank"),
    }
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["config_name"] != config_name]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(csv_path, index=False)
    return result


def save_ensemble_artifacts(run_id, ensemble_name, member_names, weights, val_pred, metrics, agg_method):
    """Save ensemble-level artifacts."""
    _, art_dir = run_dirs(run_id)
    np.save(art_dir / f"val_pred_{ensemble_name}.npy", np.asarray(val_pred, dtype=np.float32))
    info = {
        "ensemble_name": ensemble_name,
        "member_names": list(member_names),
        "weights": {k: float(v) for k, v in (weights or {}).items()},
        "agg_method": agg_method,
        **{k: _jsonable(v) for k, v in metrics.items()},
    }
    with open(art_dir / f"ensemble_result_{ensemble_name}.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return info
