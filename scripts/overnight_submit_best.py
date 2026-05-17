"""Build the BEST DEPLOYABLE submission from overnight_final_batch.

The overnight batch's #1 ensemble (0.40979) depends on `struct_rank_xendcg_regularized`
from the structural_batch dir, which the auto-submission step couldn't retrain.

This script targets the BEST ensemble whose members are ALL from this batch
(0.40971): v6 + cb_rank_C_deeper@0.05 + xendcg_conservative@0.05 +
cb_rank_A@0.05 + xendcg_reg_seed42@0.05.

The xendcg_conservative_FULL and xendcg_reg_seed42_FULL models already exist
(retrained for best_conservative). Only cb_rank_C_deeper and cb_rank_A need
full-train retraining (~15-25 min each).

Resumable: skip any FULL model + test_pred .npy that already exists.
"""
from __future__ import annotations
import gc
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_train, load_test, make_target  # noqa: E402
from src.features import build_features, compute_position_propensity  # noqa: E402
from pipelines.overnight_final_batch import (  # noqa: E402
    grouped_rank, label_remap, feature_cols_for, V6_DIR, V6_MEMBERS,
    safe_write_text,
)

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP_RUN = "20260517_022323"
RUN_DIR = ROOT / "diagnostics" / f"overnight_final_batch_{TIMESTAMP_RUN}"
PREDS_DIR = RUN_DIR / "predictions"
MODELS_DIR = RUN_DIR / "models"
ERRORS_DIR = RUN_DIR / "errors"
SUB_DIR = ROOT / "submissions"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUB_CSV = SUB_DIR / f"submission_overnight_best_deployable_{TIMESTAMP}.csv"
SUB_README = SUB_DIR / f"submission_overnight_best_deployable_{TIMESTAMP}_README.md"

V6_LOO9_TEMPORAL = 0.40896
V4_ANCHOR_TEMPORAL = 0.40401

# Ensemble target: v6 + 4 members each @ w=0.05 (total 0.20)
ENSEMBLE_MEMBERS = [
    ("cb_rank_C_deeper", 0.05),
    ("xendcg_conservative", 0.05),
    ("cb_rank_A", 0.05),
    ("xendcg_reg_seed42", 0.05),
]
TARGET_TEMPORAL_NDCG = 0.40971


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def retrain_catboost(mid: str, spec: dict, train_full: pd.DataFrame, test_full: pd.DataFrame,
                     best_iter: int) -> np.ndarray:
    from catboost import CatBoost, Pool
    feat_cols = feature_cols_for(None, train_full, test_full)
    label = label_remap(train_full["relevance"])
    train_pool = Pool(train_full[feat_cols].to_numpy(dtype=np.float32), label=label,
                       group_id=train_full["srch_id"].values, feature_names=feat_cols)
    test_pool = Pool(test_full[feat_cols].to_numpy(dtype=np.float32),
                      group_id=test_full["srch_id"].values, feature_names=feat_cols)
    params = {
        "loss_function": spec.get("loss_function", "YetiRank"),
        "iterations": best_iter,
        "learning_rate": spec.get("learning_rate", 0.03),
        "depth": spec.get("depth", 6),
        "l2_leaf_reg": spec.get("l2_leaf_reg", 5),
        "random_seed": spec.get("random_seed", 42),
        "verbose": 0,
        "allow_writing_files": False,
    }
    log(f"    CatBoost params: depth={params['depth']} iters={best_iter} l2={params['l2_leaf_reg']} "
        f"loss={params['loss_function']}")
    t0 = time.time()
    model = CatBoost(params)
    model.fit(train_pool)
    test_scores = model.predict(test_pool).astype(np.float32)
    log(f"    CatBoost trained in {(time.time()-t0)/60:.1f} min")
    model.save_model(str(MODELS_DIR / f"model_{mid}_FULL.cbm"))
    return test_scores


def main():
    log(f"BUILDING BEST DEPLOYABLE SUBMISSION — {TIMESTAMP}")
    log(f"  source batch: {RUN_DIR}")
    log(f"  target ensemble: v6 + 4 members each @ 0.05 (total added 0.20)")
    log(f"  expected temporal NDCG@5: {TARGET_TEMPORAL_NDCG}")
    log(f"  output: {SUB_CSV}")

    # Validate that prerequisite files exist
    log("\nValidating prerequisites…")
    # V6 LOO-9 test predictions
    for m in V6_MEMBERS:
        assert (V6_DIR / f"test_pred_{m}.npy").exists(), f"missing V6 test_pred {m}"
    log(f"  V6 LOO-9 test preds: {len(V6_MEMBERS)}/9 ✓")

    # Existing FULL preds we can reuse
    have_full = {}
    for mid, _ in ENSEMBLE_MEMBERS:
        p = PREDS_DIR / f"test_pred_{mid}.npy"
        if p.exists():
            have_full[mid] = p
            log(f"  REUSE test_pred_{mid}.npy")
    missing = [mid for mid, _ in ENSEMBLE_MEMBERS if mid not in have_full]
    log(f"\nNeed to retrain on full train: {missing}")

    # Need to load test set first to get srch_id alignment + test_full features
    if missing:
        log("\nLoading full train + test, building features…")
        train_raw = load_train()
        train_raw = make_target(train_raw).sort_values("srch_id").reset_index(drop=True)
        test_raw = load_test().reset_index(drop=True)
        log(f"  train rows={len(train_raw):,}  test rows={len(test_raw):,}")
        t = time.time()
        train_full = build_features(train_raw, agg_source=train_raw, is_train=True)
        log(f"  train_full features in {(time.time()-t)/60:.1f} min  ({train_full.shape[1]} cols)")
        t = time.time()
        test_full = build_features(test_raw, agg_source=train_raw, is_train=False)
        log(f"  test_full features in {(time.time()-t)/60:.1f} min  ({test_full.shape[1]} cols)")
        del train_raw, test_raw
        gc.collect()
        test_srch = test_full["srch_id"].values

        # Retrain each missing CatBoost model
        from pipelines.overnight_final_batch import make_specs
        SPEC_BY_ID = {s["id"]: s for s in make_specs()}
        model_results = pd.read_csv(RUN_DIR / "model_results.csv")

        for mid in missing:
            log(f"\n--- retraining {mid} on full train ---")
            try:
                spec = SPEC_BY_ID[mid]
                row = model_results[model_results["model_id"] == mid].iloc[0]
                best_iter = int(row["best_iter"])
                log(f"  spec: {spec.get('loss_function')}, depth={spec.get('depth')}, "
                    f"l2={spec.get('l2_leaf_reg')}, best_iter={best_iter}")
                test_scores = retrain_catboost(mid, spec, train_full, test_full, best_iter)
                np.save(PREDS_DIR / f"test_pred_{mid}.npy", test_scores)
                have_full[mid] = PREDS_DIR / f"test_pred_{mid}.npy"
                log(f"  ✓ saved test_pred_{mid}.npy")
                gc.collect()
            except Exception as e:
                log(f"  ✗ FAILED retraining {mid}: {e}")
                safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                                 ERRORS_DIR / f"SUBMIT_RETRAIN_{mid}.txt")
                # Continue — try other models; the submission will use whichever is available
        del train_full
        gc.collect()
    else:
        # No retraining needed — still need test_srch from somewhere
        log("\nNo retraining needed; loading test for srch_id alignment…")
        test_raw = load_test().reset_index(drop=True)
        test_srch = test_raw["srch_id"].values
        test_full = test_raw

    available_members = [(mid, w) for mid, w in ENSEMBLE_MEMBERS if mid in have_full]
    if len(available_members) < len(ENSEMBLE_MEMBERS):
        log(f"\n⚠️ Only {len(available_members)}/{len(ENSEMBLE_MEMBERS)} members available — "
            f"submission will be partial")

    if not available_members:
        log("✗ NO members available; aborting")
        return

    # Load V6 LOO-9 test rank-average
    log("\nLoading V6 LOO-9 test predictions and computing rank-average…")
    v6_test = [np.load(V6_DIR / f"test_pred_{m}.npy").astype(np.float32) for m in V6_MEMBERS]
    v6_rank = np.mean([grouped_rank(test_srch, s) for s in v6_test], axis=0)
    log(f"  V6 rank-average ready for {len(test_srch):,} rows")

    # Compose weighted ensemble
    member_ranks = []
    weights = []
    for mid, w in available_members:
        s = np.load(have_full[mid]).astype(np.float32)
        member_ranks.append(grouped_rank(test_srch, s))
        weights.append(w)
        log(f"  {mid}: w={w}")
    base_w = 1.0 - sum(weights)
    avg = base_w * v6_rank
    for w, r in zip(weights, member_ranks):
        avg = avg + w * r
    log(f"  V6 weight: {base_w:.4f}, members weight total: {sum(weights):.4f}")

    # Build submission
    log("\nWriting submission CSV…")
    sub_df = pd.DataFrame({
        "srch_id": test_srch,
        "prop_id": test_full["prop_id"].values,
        "_rk": avg,
    }).sort_values(["srch_id", "_rk"])[["srch_id", "prop_id"]]

    # Validate
    sample = pd.read_csv(ROOT / "data" / "submission_sample.csv")
    issues = []
    if list(sub_df.columns) != ["srch_id", "prop_id"]:
        issues.append(f"header: {list(sub_df.columns)}")
    if len(sub_df) != len(sample):
        issues.append(f"row count: sub={len(sub_df):,} sample={len(sample):,}")
    if sub_df.isna().any().any():
        issues.append("NaN in srch_id/prop_id")
    if sub_df.duplicated().any():
        issues.append(f"{sub_df.duplicated().sum()} duplicate rows")
    if set(sub_df["srch_id"].unique()) != set(sample["srch_id"].unique()):
        issues.append("srch_id set mismatch with sample")
    if issues:
        log(f"  ✗ VALIDATION ISSUES: {issues}")
        return
    log(f"  validation OK")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(SUB_CSV, index=False)
    log(f"\n✓ SUBMISSION WRITTEN: {SUB_CSV}")
    log(f"  rows={len(sub_df):,}  searches={sub_df['srch_id'].nunique():,}")

    # README
    safe_write_text(
        f"# Submission OVERNIGHT BEST DEPLOYABLE — {TIMESTAMP}\n\n"
        f"## Why this submission\n\n"
        f"The overnight final batch (`overnight_final_batch_{TIMESTAMP_RUN}`) found its #1 ensemble "
        f"at temporal NDCG@5 = 0.40979, but that ensemble depended on a model from the previous "
        f"structural batch that the auto-submission step couldn't retrain. This submission targets "
        f"the **best ensemble whose members are ALL in this batch** "
        f"(temporal NDCG@5 = {TARGET_TEMPORAL_NDCG}).\n\n"
        f"## Members\n\n"
        f"- **V6 LOO-9** (weight {base_w:.4f}) — the 9-member ensemble that produced Kaggle 0.42004\n"
        + "".join(f"- **{mid}** (weight {w}) — newly trained in overnight batch\n"
                   for mid, w in available_members) +
        f"\n## Temporal benchmarks\n\n"
        f"- V4_ANCHOR temporal:        {V4_ANCHOR_TEMPORAL}\n"
        f"- V6 LOO-9 temporal:         {V6_LOO9_TEMPORAL}  → Kaggle 0.42004\n"
        f"- This ensemble (temporal):  {TARGET_TEMPORAL_NDCG}  → projected Kaggle ~0.4209–0.4211\n\n"
        f"## Validation\n"
        f"- header: `srch_id,prop_id` ✓\n"
        f"- rows: {len(sub_df):,} (matches sample)\n"
        f"- unique searches: {sub_df['srch_id'].nunique():,}\n"
        f"- duplicates: 0\n"
        f"- NaN: 0\n\n"
        f"## Risk notes\n"
        f"- This ensemble adds **CatBoost diversity** (cb_rank_C_deeper, cb_rank_A) to V6 — "
        f"a genuinely different model class than V6's all-LightGBM members. The +0.00075 local "
        f"gain over V6 is the strongest the session has produced. Projected Kaggle delta is "
        f"~+0.0006 — small but positive vs V4.\n"
        f"- The local→Kaggle correlation has been weak this session (V6 local 0.40896 → Kaggle "
        f"0.42004 = +0.011 gap). If this ensemble follows the same ratio, projected Kaggle is "
        f"0.42085 — about +0.0006 above V4 0.42021.\n",
        SUB_README
    )
    log(f"  README: {SUB_README}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                         ERRORS_DIR / "OVERNIGHT_SUBMIT_FATAL.txt")
        log(f"FATAL: {e}")
        raise
