"""
Phase 3: weighting / IPW sweep around Phase 2's best label_gain candidates.

PREPARED BUT NOT RUN — invoke manually with:
    uv run --no-sync python run_phase3_weighting.py

Scope (14 single-model configs = 7 weighting variants × 2 label_gains):
- weighting variants : ipw_default, no_ipw, ipw_positive, ipw_clip3, ipw_clip5,
                       rand_up_1.5, rand_up_2.0
- label_gains        : 0,2,15  (Phase 2 winner) and  0,1,15  (V4 bal15 anchor)

Hard constraints (from session 2026-05-15 brief):
- V4 full feature set (143 features built by src/features.py:build_features).
- V4-style Dataset pattern: fresh lgb.Dataset(...) inside each iteration,
  no explicit .construct(), no free_raw_data flag — required for reproducibility
  (see experiment_logs/v4_phase2_summary.md §4 for the binning bug).
- Single models only. No submissions, no full-data retrain, no ensemble.
- All configs share seed=456, BASE_PARAMS otherwise identical to V4 stage 3.
- val_frac=0.1, random_state=42 (default split_val).

Output (mirrors Phase 2 layout):
    models/phase3_weighting/model_<config>.txt
    artifacts/phase3_weighting/
        run_config.json
        git_commit.txt
        feature_cols.json
        val_meta.parquet
        val_pred_<config>.npy
        importance_<config>.csv
        model_result_<config>.json
        model_results.csv

After completion (run separately):
    uv run --no-sync python scripts/aggregate_results.py \
        --run-id phase3_weighting --phase P3 \
        --change-summary "Weighting sweep (7 IPW variants × 2 label_gains)"
"""
import gc
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb

from src.data_loader import load_train, make_target, get_feature_columns, split_val
from src.features import (
    build_features, compute_position_propensity,
    FORBIDDEN_FEATURES,
)
from src.evaluate import evaluate_ndcg
from src.artifacts import (
    run_dirs, save_run_config, save_git_commit, save_feature_cols, save_val_meta,
    save_model_artifacts,
)


RUN_ID = "phase3_weighting"
ANCHOR_SEED = 456

LABEL_GAINS = ["0,2,15", "0,1,15"]

# Weighting variants. See compute_weights() below for semantics.
WEIGHTING_VARIANTS = [
    {"name": "ipw_default",   "mode": "ipw", "clip_low": 0.1, "clip_high": 10.0,
     "positive_only": False, "random_upweight": 1.0,
     "doc": "V4 default. IPW on non-random rows, clipped [0.1, 10.0]."},
    {"name": "no_ipw",        "mode": "none",
     "doc": "All weights = 1.0. Baseline for whether IPW helps at all."},
    {"name": "ipw_positive",  "mode": "ipw", "clip_low": 0.1, "clip_high": 10.0,
     "positive_only": True,  "random_upweight": 1.0,
     "doc": "IPW applied only on click_bool=1 rows (positives). Unclicked = 1.0."},
    {"name": "ipw_clip3",     "mode": "ipw", "clip_low": 0.1, "clip_high": 3.0,
     "positive_only": False, "random_upweight": 1.0,
     "doc": "Tighter clip to suppress huge weights on rare positions (default capped at 10)."},
    {"name": "ipw_clip5",     "mode": "ipw", "clip_low": 0.1, "clip_high": 5.0,
     "positive_only": False, "random_upweight": 1.0,
     "doc": "Moderate clip between default (10) and ipw_clip3."},
    {"name": "rand_up_1.5",   "mode": "ipw", "clip_low": 0.1, "clip_high": 10.0,
     "positive_only": False, "random_upweight": 1.5,
     "doc": "IPW default + random_bool=1 rows multiplied by 1.5."},
    {"name": "rand_up_2.0",   "mode": "ipw", "clip_low": 0.1, "clip_high": 10.0,
     "positive_only": False, "random_upweight": 2.0,
     "doc": "IPW default + random_bool=1 rows multiplied by 2.0 (matches V4 lambdarank_randup)."},
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
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df)
    return g


def compute_weights(train_df, propensity, variant):
    """Compute per-row training weights according to the variant spec.

    Returns a float32 numpy array, length = len(train_df).
    """
    mode = variant["mode"]
    if mode == "none":
        return np.ones(len(train_df), dtype=np.float32)

    # mode == "ipw"
    max_prop = float(propensity.max())
    pos_w = train_df["position"].map(
        lambda p: 1.0 if propensity.get(p, 0) <= 0 else max_prop / propensity[p]
    ).astype(np.float32).values

    is_nonrandom = (train_df["random_bool"] == 0).values
    w = np.where(is_nonrandom, pos_w, 1.0).astype(np.float32)

    if variant.get("positive_only"):
        is_click = (train_df["click_bool"] == 1).values
        w = np.where(is_click, w, 1.0).astype(np.float32)

    w = np.clip(w, variant["clip_low"], variant["clip_high"]).astype(np.float32)

    upw = variant.get("random_upweight", 1.0)
    if upw != 1.0:
        rand_mask = (train_df["random_bool"] == 1).values
        w[rand_mask] *= upw

    return w


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
    model_dir, art_dir = run_dirs(RUN_ID)

    log(f"\n=== Phase 3: Weighting Sweep — run_id={RUN_ID} ===")
    log(f"  Variants: {[v['name'] for v in WEIGHTING_VARIANTS]}")
    log(f"  Label gains: {LABEL_GAINS}")
    log(f"  Total configs: {len(WEIGHTING_VARIANTS) * len(LABEL_GAINS)}")
    log(f"  seed={ANCHOR_SEED}, num_rounds={NUM_BOOST_ROUND}, early_stop={EARLY_STOPPING}\n")

    # Run config
    save_run_config(RUN_ID, {
        "phase": "P3_weighting",
        "run_id": RUN_ID,
        "date": datetime.now(timezone.utc).isoformat(),
        "anchor_ref": "V4_BEST_SINGLE (label_gain=0,1,15, val NDCG@5=0.42191)",
        "phase2_winner_ref": "P2_lg_0_2_15 (label_gain=0,2,15, val NDCG@5=0.42258, Kaggle=0.41639)",
        "seed": ANCHOR_SEED,
        "val_frac": 0.1,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": EARLY_STOPPING,
        "base_params": BASE_PARAMS,
        "label_gains": LABEL_GAINS,
        "weighting_variants": WEIGHTING_VARIANTS,
        "notes": (
            "Single-model only. No retrain on full, no submission, no ensemble. "
            "V4-style Dataset pattern (fresh lgb.Dataset per config; see v4_phase2_summary.md §4)."
        ),
    })
    save_git_commit(RUN_ID)

    log("Loading training data...")
    train_raw = load_train()
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    log("Computing propensity (random_bool=1 subset, full train)...")
    propensity = compute_position_propensity(train_raw)
    log(f"  propensity (top 5 positions):\n{propensity.head().to_string()}")

    log("Splitting train/val by srch_id (frac=0.1, seed=42)...")
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    log(f"  Train: {len(train_split):,} | Val: {len(val_split):,}")

    log("Building features...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden columns leaked: {leaked}"
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")

    save_feature_cols(RUN_ID, feature_cols)
    save_val_meta(RUN_ID, val_feat)

    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    # Keep train_split around — we need its position/random_bool/click_bool for compute_weights.
    # train_split is sorted same way as train_feat (both sorted by srch_id and reset_index).
    assert len(train_split) == len(train_feat), "train_split / train_feat length mismatch"

    log(f"\nLooping over {len(WEIGHTING_VARIANTS) * len(LABEL_GAINS)} configs...\n")

    for v in WEIGHTING_VARIANTS:
        weights = compute_weights(train_split, propensity, v)
        log(f"  weights[{v['name']}]: min={weights.min():.3f}, max={weights.max():.3f}, "
            f"mean={weights.mean():.3f}, std={weights.std():.3f}")

        for lg in LABEL_GAINS:
            cname = f"w_{v['name']}_lg_{lg.replace(',', '_')}"
            result_path = art_dir / f"model_result_{cname}.json"
            if result_path.exists():
                log(f"    {cname}: SKIP (already trained)")
                continue

            log(f"    Training {cname} ...")
            params = BASE_PARAMS.copy()
            params["label_gain"] = lg

            ds_train = lgb.Dataset(
                train_feat[feature_cols], label=train_label,
                group=train_groups, weight=weights,
            )
            ds_val = lgb.Dataset(
                val_feat[feature_cols], label=val_label,
                group=val_groups, reference=ds_train,
            )

            t_tr = time.time()
            model = lgb.train(
                params, ds_train, num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(200)],
            )
            tr_elapsed = time.time() - t_tr

            pred = model.predict(val_feat[feature_cols])
            val_feat["pred_score"] = pred
            metrics = evaluate_full(val_feat)
            log(f"      NDCG@5={metrics['ndcg5']:.5f}  R@5={metrics['recall5']:.4f}  "
                f"best_iter={model.best_iteration}  ({tr_elapsed/60:.1f} min)")

            save_model_artifacts(RUN_ID, cname, model, pred, feature_cols, metrics, params)

            del ds_train, ds_val, model
            gc.collect()

    log(f"\n=== Phase 3 done in {(time.time()-t0)/60:.1f} min ===")
    log(f"Results: artifacts/{RUN_ID}/model_results.csv")
    log("Next: aggregate to experiment_logs/ via scripts/aggregate_results.py "
        "(see header docstring for the exact command).")


if __name__ == "__main__":
    main()
