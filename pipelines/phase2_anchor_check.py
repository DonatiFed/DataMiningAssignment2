"""
Anchor reproduction test: lg_0_1_15 with V4-style Dataset pattern.

Goal: Reproduce V4 lambdarank_bal15 local NDCG@5 = 0.42191 using identical
config (seed=456, label_gain="0,1,15") and V4's Dataset pattern:
  - Fresh lgb.Dataset(...) inside the model section
  - No explicit .construct() call
  - No free_raw_data flag

If this run hits ~0.42191, the anchor mismatch in run_phase2.py is confirmed
to be caused by the explicit pre-construct + Dataset reuse pattern.

Saves to artifacts/phase2_anchor_check/.
"""
import gc
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb

from src.data_loader import load_train, make_target, get_feature_columns, split_val
from src.features import (
    build_features, compute_position_propensity, compute_sample_weights,
    FORBIDDEN_FEATURES,
)
from src.evaluate import evaluate_ndcg
from src.artifacts import (
    run_dirs, save_run_config, save_git_commit, save_feature_cols, save_val_meta,
    save_model_artifacts,
)


RUN_ID = "phase2_anchor_check"
CONFIG = {"name": "anchor_lg_0_1_15", "label_gain": "0,1,15"}
SEED = 456

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
    "seed": SEED,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 80


def log(msg):
    print(msg, flush=True)


def make_group_counts(df):
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df)
    return g


def recall_at_k(df, score_col, k=5):
    results = []
    for _, grp in df.groupby("srch_id"):
        booked = grp[grp["relevance"] == 5]
        if len(booked) == 0:
            continue
        top_k_ids = grp.nlargest(k, score_col)["prop_id"].values
        results.append(int(booked["prop_id"].isin(top_k_ids).any()))
    return float(np.mean(results)) if results else 0.0


def mean_booked_rank(df, score_col):
    ranks = []
    for _, grp in df.groupby("srch_id"):
        booked = grp[grp["relevance"] == 5]
        if len(booked) == 0:
            continue
        ranked = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        ranked["rank_pos"] = np.arange(1, len(ranked) + 1)
        ranks.extend(ranked[ranked["prop_id"].isin(booked["prop_id"])]["rank_pos"].tolist())
    return float(np.mean(ranks)) if ranks else float("nan")


def evaluate_full(df, score_col="pred_score"):
    return {
        "ndcg5": float(evaluate_ndcg(df, score_col=score_col, k=5)),
        "recall1": recall_at_k(df, score_col, k=1),
        "recall5": recall_at_k(df, score_col, k=5),
        "mean_booked_rank": mean_booked_rank(df, score_col),
    }


def main():
    t0 = time.time()
    log(f"\n=== Phase 2 anchor check — run_id={RUN_ID} ===")
    log("Goal: reproduce V4 bal15 (0.42191) using V4-style Dataset pattern.\n")

    save_run_config(RUN_ID, {
        "phase": "P2_anchor_check",
        "purpose": "Verify the anchor-mismatch hypothesis (Dataset pre-construct used default data_random_seed=1).",
        "expected_ndcg5": 0.42191,
        "config": CONFIG,
        "seed": SEED,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": EARLY_STOPPING,
        "date": datetime.now(timezone.utc).isoformat(),
        "base_params": BASE_PARAMS,
    })
    save_git_commit(RUN_ID)

    log("Loading training data...")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    log("Computing propensity (IPW)...")
    propensity = compute_position_propensity(train_raw)

    log("Splitting train/val by srch_id (frac=0.1, seed=42)...")
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    log(f"  Train: {len(train_split):,} | Val: {len(val_split):,}")

    weights = compute_sample_weights(train_split, propensity)

    log("Building features...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden columns: {leaked}"
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")

    save_feature_cols(RUN_ID, feature_cols)
    save_val_meta(RUN_ID, val_feat)

    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    del train_raw, train_split, val_split
    gc.collect()

    # --- V4-STYLE DATASET PATTERN ---
    # Fresh Datasets, no params=, no explicit construct.
    # lgb.train will construct lazily using the training params (seed=456).
    log(f"\nTraining anchor config (V4-style Dataset pattern):")
    log(f"  label_gain={CONFIG['label_gain']}, seed={SEED}")
    log(f"  num_boost_round={NUM_BOOST_ROUND}, early_stopping={EARLY_STOPPING}\n")

    params = BASE_PARAMS.copy()
    params["label_gain"] = CONFIG["label_gain"]

    ds_train = lgb.Dataset(
        train_feat[feature_cols], label=train_label,
        group=train_groups, weight=weights,
    )
    ds_val = lgb.Dataset(
        val_feat[feature_cols], label=val_label,
        group=val_groups, reference=ds_train,
    )

    t_train = time.time()
    model = lgb.train(
        params, ds_train, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(100)],
    )
    train_elapsed = time.time() - t_train

    pred = model.predict(val_feat[feature_cols])
    val_feat["pred_score"] = pred
    metrics = evaluate_full(val_feat)

    log(f"\n>>> RESULT <<<")
    log(f"  NDCG@5 = {metrics['ndcg5']:.5f}  (V4 anchor: 0.42191, current Phase 2: 0.41951)")
    log(f"  R@1={metrics['recall1']:.4f}  R@5={metrics['recall5']:.4f}  MBR={metrics['mean_booked_rank']:.2f}")
    log(f"  best_iter={model.best_iteration}  trained in {train_elapsed/60:.1f} min")

    delta_v4 = metrics["ndcg5"] - 0.42191
    delta_p2 = metrics["ndcg5"] - 0.41951
    log(f"\n  Δ vs V4 anchor (0.42191): {delta_v4:+.5f}")
    log(f"  Δ vs Phase 2 lg_0_1_15 (0.41951): {delta_p2:+.5f}")

    if abs(delta_v4) < 0.0005:
        log("  → HYPOTHESIS CONFIRMED: V4-style Dataset pattern reproduces the anchor.")
        log("    Phase 2 mismatch caused by explicit .construct() pre-binning with default seed.")
    elif delta_v4 < -0.001:
        log("  → HYPOTHESIS REJECTED: V4-style pattern also underperforms.")
        log("    Investigate other suspects (e.g., V4's recorded 0.42191 may have been on a different feature set / different code state).")
    else:
        log("  → AMBIGUOUS: small drift but not full reproduction. Investigate further.")

    save_model_artifacts(RUN_ID, CONFIG["name"], model, pred, feature_cols, metrics, params)

    log(f"\nDone in {(time.time()-t0)/60:.1f} min")
    log(f"Artifacts: artifacts/{RUN_ID}/")


if __name__ == "__main__":
    main()
