"""
Phase 2: Label-gain sweep around V4 bal15 anchor.

Single-model only. No test predictions, no full-data retrain, no submission, no ensemble.
Reuses the V4 stage-3 training recipe (same train/val split, features, IPW weights,
seed, num_rounds, early stopping). Only label_gain varies.

V4 anchor: lambdarank_bal15, label_gain="0,1,15", local NDCG@5 = 0.42191.
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


RUN_ID = "phase2_labelgain"

# Anchor: V4 lambdarank_bal15 used seed=456, label_gain="0,1,15".
# Only label_gain varies in this sweep.
ANCHOR_SEED = 456

CONFIGS = [
    {"name": "lg_0_1_10", "label_gain": "0,1,10"},
    {"name": "lg_0_1_12", "label_gain": "0,1,12"},
    {"name": "lg_0_1_15", "label_gain": "0,1,15"},  # V4 anchor reproduction
    {"name": "lg_0_1_18", "label_gain": "0,1,18"},
    {"name": "lg_0_1_20", "label_gain": "0,1,20"},
    {"name": "lg_0_2_15", "label_gain": "0,2,15"},
]

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

NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 80


def log(msg):
    print(msg, flush=True)


def make_group_counts(df):
    groups = df.groupby("srch_id", sort=False).size().values
    assert groups.sum() == len(df)
    return groups


def assert_sorted(df):
    assert (df["srch_id"].diff().dropna() >= 0).all(), "DataFrame must be sorted by srch_id"


def assert_no_forbidden(feature_cols):
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden columns in features: {leaked}"


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
    log(f"\n=== Phase 2: Label-Gain Sweep — run_id={RUN_ID} ===")
    log(f"Anchor: V4 lambdarank_bal15 (label_gain=0,1,15, local NDCG@5=0.42191)\n")

    model_dir, art_dir = run_dirs(RUN_ID)

    run_cfg = {
        "phase": "P2_label_gain",
        "run_id": RUN_ID,
        "anchor": {
            "name": "V4_lambdarank_bal15",
            "label_gain": "0,1,15",
            "local_ndcg5": 0.42191,
        },
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": ANCHOR_SEED,
        "val_frac": 0.1,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": EARLY_STOPPING,
        "base_params": BASE_PARAMS,
        "configs": CONFIGS,
    }
    save_run_config(RUN_ID, run_cfg)
    save_git_commit(RUN_ID)

    log("Loading training data...")
    train_raw = load_train()
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    log("Computing position propensity (IPW)...")
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
    assert_no_forbidden(feature_cols)
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")

    save_feature_cols(RUN_ID, feature_cols)
    save_val_meta(RUN_ID, val_feat)

    assert_sorted(train_feat)
    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    # Labels remapped {0,1,5} → {0,1,2} for LightGBM label_gain (contiguous indexing).
    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    del train_raw, train_split, val_split
    gc.collect()

    # V4-style Dataset pattern: fresh lgb.Dataset per config inside the loop,
    # no explicit .construct() (let lgb.train construct lazily so the training
    # seed=456 reaches the binning). We accept the extra construction cost in
    # exchange for reproducibility and parity with V4 stage 3.
    log(f"\nRunning {len(CONFIGS)} label-gain configs (num_rounds={NUM_BOOST_ROUND}, "
        f"early_stopping={EARLY_STOPPING})\n")

    for i, cfg in enumerate(CONFIGS, 1):
        cname = cfg["name"]
        result_path = art_dir / f"model_result_{cname}.json"
        if result_path.exists():
            log(f"[{i}/{len(CONFIGS)}] {cname}: SKIP (already trained)")
            continue

        log(f"[{i}/{len(CONFIGS)}] {cname} — label_gain={cfg['label_gain']}")

        params = BASE_PARAMS.copy()
        params["label_gain"] = cfg["label_gain"]

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
        log(f"  NDCG@5={metrics['ndcg5']:.5f}  R@1={metrics['recall1']:.4f}  "
            f"R@5={metrics['recall5']:.4f}  MBR={metrics['mean_booked_rank']:.2f}  "
            f"best_iter={model.best_iteration}  ({train_elapsed/60:.1f} min)")

        save_model_artifacts(RUN_ID, cname, model, pred, feature_cols, metrics, params)

        del ds_train, ds_val, model
        gc.collect()

    log(f"\n=== Phase 2 done in {(time.time()-t0)/60:.1f} min ===")
    log(f"Results: artifacts/{RUN_ID}/model_results.csv")
    log(f"Compare against V4 anchor (lambdarank_bal15, NDCG@5=0.42191)")


if __name__ == "__main__":
    main()
