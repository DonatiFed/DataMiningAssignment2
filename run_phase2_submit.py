"""
Phase 2 best-label_gain submission.

Single-model V4-config except for the winning label_gain (auto-picked by
val NDCG@5 from artifacts/phase2_labelgain/model_results.csv).

Steps:
  1. Read Phase 2 results, pick row with max ndcg5.
  2. Retrain on FULL train data with that label_gain, using the val-run's best_iter
     (no validation set, no early stopping).
  3. Featurize test (agg_source = full train_raw), predict, generate submission.
  4. Save all artifacts under artifacts/phase2_best_submit/ and copy to
     /home/ubuntu/experiment_artifacts/phase2_best_submit/.

No ensemble, no feature changes, no IPW changes.
"""
import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import ROOT_DIR, SUBMISSIONS_DIR
from src.data_loader import load_train, load_test, make_target, get_feature_columns
from src.features import (
    build_features, compute_position_propensity, compute_sample_weights,
    FORBIDDEN_FEATURES,
)
from src.artifacts import (
    run_dirs, save_run_config, save_git_commit, save_feature_cols,
)


RUN_ID = "phase2_best_submit"
EXTERNAL_COPY_ROOT = Path("/home/ubuntu/experiment_artifacts")

ANCHOR_SEED = 456

BASE_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 400,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_split_gain": 0.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": ANCHOR_SEED,
}


def log(msg):
    print(msg, flush=True)


def make_group_counts(df):
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df)
    return g


def pick_best_config(phase2_results_csv):
    df = pd.read_csv(phase2_results_csv)
    if len(df) == 0:
        raise SystemExit(f"No rows in {phase2_results_csv}")
    df = df.sort_values("ndcg5", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    log("\nPhase 2 results (sorted by val NDCG@5):")
    for _, r in df.iterrows():
        marker = "★" if r["config_name"] == best["config_name"] else " "
        log(f"  {marker} {r['config_name']:14s}  label_gain={r['label_gain']:8s}  "
            f"ndcg5={r['ndcg5']:.5f}  recall5={r['recall5']:.4f}  best_iter={int(r['best_iter'])}")
    log(f"\nWinner: {best['config_name']}  (label_gain={best['label_gain']}, best_iter={int(best['best_iter'])})")
    return best


def main():
    t0 = time.time()
    log(f"=== Phase 2 Best-Label-Gain Submission — run_id={RUN_ID} ===")

    phase2_csv = ROOT_DIR / "artifacts" / "phase2_labelgain" / "model_results.csv"
    best = pick_best_config(phase2_csv)
    best_label_gain = str(best["label_gain"])
    best_iter = int(best["best_iter"])
    val_ndcg5 = float(best["ndcg5"])
    val_recall5 = float(best["recall5"])

    # --- Set up artifact dirs ---
    model_dir, art_dir = run_dirs(RUN_ID)

    run_cfg = {
        "phase": "P2_best_submit",
        "run_id": RUN_ID,
        "date": datetime.now(timezone.utc).isoformat(),
        "selected_config": {
            "name": str(best["config_name"]),
            "label_gain": best_label_gain,
            "val_ndcg5": val_ndcg5,
            "val_recall5": val_recall5,
            "best_iter_from_val": best_iter,
        },
        "seed": ANCHOR_SEED,
        "base_params": BASE_PARAMS,
        "anchor_ref": "V4_BEST_SINGLE (lambdarank_bal15, val NDCG@5=0.42191)",
        "notes": "V4 single-model config retrained on full train. Label_gain auto-picked from Phase 2.",
    }
    save_run_config(RUN_ID, run_cfg)
    save_git_commit(RUN_ID)

    log("\nLoading FULL training data...")
    train_raw = load_train()
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    log("Computing propensity + IPW (full train)...")
    propensity = compute_position_propensity(train_raw)
    weights = compute_sample_weights(train_raw, propensity)
    log(f"  IPW range: [{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")

    log("Building features on full train (agg_source=train_raw)...")
    t1 = time.time()
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    feature_cols = get_feature_columns(full_feat)
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden columns leaked: {leaked}"
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")
    save_feature_cols(RUN_ID, feature_cols)

    full_groups = make_group_counts(full_feat)

    remap = {0: 0, 1: 1, 5: 2}
    full_label = full_feat["relevance"].map(remap).astype(np.int32)

    # --- Train on full data with best_iter rounds, no early stopping ---
    params = BASE_PARAMS.copy()
    params["label_gain"] = best_label_gain
    log(f"\nTraining single model on FULL data:")
    log(f"  label_gain={best_label_gain}, seed={ANCHOR_SEED}, num_boost_round={best_iter}")

    ds_full = lgb.Dataset(
        full_feat[feature_cols], label=full_label,
        group=full_groups, weight=weights,
    )

    t_train = time.time()
    model = lgb.train(
        params, ds_full, num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(100)],
    )
    train_elapsed = time.time() - t_train
    log(f"  Trained {best_iter} rounds in {train_elapsed/60:.1f} min")

    model.save_model(str(model_dir / "model_full.txt"))

    # Importance
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df.to_csv(art_dir / "importance.csv", index=False)

    del ds_full
    gc.collect()

    # --- Test ---
    log("\nLoading test set...")
    test_raw = load_test()
    log(f"  {len(test_raw):,} rows, {test_raw['srch_id'].nunique():,} searches")

    log("Featurizing test (agg_source=full train)...")
    t2 = time.time()
    test_feat = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"  built in {time.time()-t2:.0f}s")

    # Align columns
    for c in feature_cols:
        if c not in test_feat.columns:
            test_feat[c] = np.nan

    log("Predicting test...")
    t3 = time.time()
    test_feat["pred_score"] = model.predict(test_feat[feature_cols])
    log(f"  predicted in {time.time()-t3:.0f}s")

    # --- Submission ---
    log("Generating submission CSV...")
    ranked = (
        test_feat[["srch_id", "prop_id", "pred_score"]]
        .sort_values(["srch_id", "pred_score"], ascending=[True, False])
    )
    submission = ranked[["srch_id", "prop_id"]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_name = f"submission_phase2_best_{timestamp}.csv"
    sub_path = SUBMISSIONS_DIR / sub_name
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)

    # Also save a copy inside the artifact dir
    sub_copy = art_dir / "submission.csv"
    submission.to_csv(sub_copy, index=False)
    log(f"  Submission rows: {len(submission):,}  unique searches: {submission['srch_id'].nunique():,}")
    log(f"  Saved: {sub_path}")
    log(f"  Saved: {sub_copy}")

    # --- model_result.json ---
    result = {
        "config_name": str(best["config_name"]),
        "label_gain": best_label_gain,
        "seed": ANCHOR_SEED,
        "val_ndcg5": val_ndcg5,
        "val_recall5": val_recall5,
        "val_best_iter": best_iter,
        "trained_rounds": best_iter,
        "n_features": len(feature_cols),
        "submission_path": str(sub_path),
        "submission_rows": int(len(submission)),
        "submission_unique_searches": int(submission["srch_id"].nunique()),
        "elapsed_min": (time.time() - t0) / 60,
    }
    with open(art_dir / "model_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # --- Copy artifacts to external dir ---
    external_dir = EXTERNAL_COPY_ROOT / RUN_ID
    log(f"\nCopying artifacts to {external_dir} ...")
    if external_dir.exists():
        shutil.rmtree(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    # Copy artifact dir
    for f in art_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, external_dir / f.name)
    # Copy model dir contents too
    (external_dir / "model").mkdir(exist_ok=True)
    for f in model_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, external_dir / "model" / f.name)
    log(f"  Copied {sum(1 for _ in external_dir.rglob('*') if _.is_file())} files")

    # --- Final summary ---
    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    log(f"  Best label_gain          : {best_label_gain}")
    log(f"  Validation NDCG@5         : {val_ndcg5:.5f}")
    log(f"  Validation Recall@5       : {val_recall5:.4f}")
    log(f"  Best iteration (from val) : {best_iter}")
    log(f"  Trained on full data      : {best_iter} rounds")
    log(f"  Submission path           : {sub_path}")
    log(f"  Submission copy           : {sub_copy}")
    log(f"  External artifacts copy   : {external_dir}")
    log(f"  Elapsed                   : {(time.time()-t0)/60:.1f} min")
    log("=" * 60)


if __name__ == "__main__":
    main()
