"""
V4 pipeline: Staged experiments with proper logging and resumability.

Stage 1: Sanity — 10% sample, 1 model, 500 rounds, early_stopping=30
Stage 2: Full data — 1 model, gate: only if sanity NDCG@5 > 0.28
Stage 3: Diverse ensemble — only if Stage 2 NDCG@5 > V3 baseline (0.385)
Stage 4: Retrain + submission

Each stage saves to models/v4/. Resumable: skips completed stages.
"""
import gc
import json
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from pathlib import Path

from src.config import NON_FEATURE_COLS
from src.data_loader import load_train, load_test, make_target, get_feature_columns, split_val
from src.features import (
    build_features, compute_position_propensity, compute_sample_weights,
    FORBIDDEN_FEATURES,
)
from src.evaluate import evaluate_ndcg
from src.submission import generate_submission
from src.artifacts import (
    save_run_config, save_git_commit, save_feature_cols, save_val_meta,
    save_model_artifacts, save_ensemble_artifacts,
)

ARTIFACT_DIR = Path("models/v4")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

V4_RUN_ID = "v4_stage3"
V3_BASELINE_NDCG = 0.412


def log(msg):
    print(msg, flush=True)


def save_checkpoint(name, data):
    path = ARTIFACT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def checkpoint_exists(name):
    return (ARTIFACT_DIR / f"{name}.json").exists()


def load_checkpoint(name):
    with open(ARTIFACT_DIR / f"{name}.json") as f:
        return json.load(f)


def make_group_counts(df):
    """Group counts; asserts df is sorted by srch_id."""
    groups = df.groupby("srch_id", sort=False).size().values
    assert groups.sum() == len(df)
    return groups


def assert_sorted(df):
    assert (df["srch_id"].diff().dropna() >= 0).all(), "DataFrame must be sorted by srch_id"


def assert_no_forbidden(feature_cols):
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden columns in features: {leaked}"


def remap_labels_for_gain(relevance, label_gain):
    """Remap relevance {0,1,5} -> {0,1,2} and adjust label_gain accordingly.
    LightGBM label_gain expects labels 0..max_label to be contiguous.
    """
    mapping = {0: 0, 1: 1, 5: 2}
    remapped = relevance.map(mapping).astype(np.int32)
    assert remapped.notna().all(), "Unexpected relevance values"
    return remapped, label_gain


def recall_at_k(df, score_col, k=5):
    results = []
    for _, grp in df.groupby("srch_id"):
        booked = grp[grp["relevance"] == 5]
        if len(booked) == 0:
            continue
        top_k_ids = grp.nlargest(k, score_col)["prop_id"].values
        hit = booked["prop_id"].isin(top_k_ids).any()
        results.append(int(hit))
    return np.mean(results) if results else 0.0


def mean_booked_rank(df, score_col):
    ranks = []
    for _, grp in df.groupby("srch_id"):
        booked = grp[grp["relevance"] == 5]
        if len(booked) == 0:
            continue
        grp_ranked = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        grp_ranked["rank_pos"] = np.arange(1, len(grp_ranked) + 1)
        booked_ranks = grp_ranked[grp_ranked["prop_id"].isin(booked["prop_id"])]["rank_pos"]
        ranks.extend(booked_ranks.tolist())
    return np.mean(ranks) if ranks else float("nan")


def evaluate_full(df, score_col="pred_score"):
    ndcg5 = evaluate_ndcg(df, score_col=score_col, k=5)
    r1 = recall_at_k(df, score_col, k=1)
    r5 = recall_at_k(df, score_col, k=5)
    mbr = mean_booked_rank(df, score_col)
    return {"ndcg5": ndcg5, "recall1": r1, "recall5": r5, "mean_booked_rank": mbr}


def print_metrics(metrics, prefix=""):
    log(f"{prefix}NDCG@5={metrics['ndcg5']:.5f}  "
        f"R@1={metrics['recall1']:.4f}  R@5={metrics['recall5']:.4f}  "
        f"MBR={metrics['mean_booked_rank']:.2f}")


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
    "seed": 42,
}


# ========== STAGE 1: SANITY (10% sample) ==========
def stage1_sanity():
    if checkpoint_exists("stage1_result"):
        result = load_checkpoint("stage1_result")
        log(f"[Stage 1] SKIP (cached) — NDCG@5={result['ndcg5']:.5f}")
        return result

    log("\n" + "="*60)
    log("[Stage 1] SANITY — 10% sample, 1 model, 500 rounds")
    log("="*60)
    t0 = time.time()

    train_raw = load_train(sample_frac=0.1, random_state=42)
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    propensity = compute_position_propensity(train_raw)
    train_split, val_split = split_val(train_raw, val_frac=0.15)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    log(f"  Train: {len(train_split):,} | Val: {len(val_split):,}")

    weights = compute_sample_weights(train_split, propensity)

    log("  Building features...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    assert_no_forbidden(feature_cols)
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")

    assert_sorted(train_feat)
    train_set = lgb.Dataset(
        train_feat[feature_cols], label=train_feat["relevance"],
        group=make_group_counts(train_feat), weight=weights,
    )
    val_set = lgb.Dataset(
        val_feat[feature_cols], label=val_feat["relevance"],
        group=make_group_counts(val_feat), reference=train_set,
    )

    log("  Training (500 rounds, early_stopping=30)...")
    model = lgb.train(
        BASE_PARAMS, train_set, num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    pred = model.predict(val_feat[feature_cols])
    val_feat["pred_score"] = pred
    metrics = evaluate_full(val_feat)
    print_metrics(metrics, prefix="  ")

    importance = sorted(
        zip(feature_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True,
    )
    log("  Top 15 features:")
    for feat, gain in importance[:15]:
        log(f"    {feat:45s} {gain:,.0f}")

    model.save_model(str(ARTIFACT_DIR / "sanity_model.txt"))
    result = {**metrics, "best_iter": model.best_iteration,
              "n_features": len(feature_cols), "elapsed_min": (time.time()-t0)/60}
    save_checkpoint("stage1_result", result)
    save_checkpoint("stage1_features", feature_cols)

    del train_raw, train_split, val_split, train_feat, val_feat, train_set, val_set, model
    gc.collect()
    return result


# ========== STAGE 2: FULL DATA, 1 MODEL ==========
def stage2_full_single():
    if checkpoint_exists("stage2_result"):
        result = load_checkpoint("stage2_result")
        log(f"[Stage 2] SKIP (cached) — NDCG@5={result['ndcg5']:.5f}")
        return result

    log("\n" + "="*60)
    log("[Stage 2] FULL DATA — 1 model")
    log("="*60)
    t0 = time.time()

    train_raw = load_train()
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    propensity = compute_position_propensity(train_raw)
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    log(f"  Train: {len(train_split):,} | Val: {len(val_split):,}")

    weights = compute_sample_weights(train_split, propensity)

    log("  Building features...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    assert_no_forbidden(feature_cols)
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")

    assert_sorted(train_feat)
    train_set = lgb.Dataset(
        train_feat[feature_cols], label=train_feat["relevance"],
        group=make_group_counts(train_feat), weight=weights,
    )
    val_set = lgb.Dataset(
        val_feat[feature_cols], label=val_feat["relevance"],
        group=make_group_counts(val_feat), reference=train_set,
    )

    log("  Training (3000 rounds, early_stopping=100)...")
    model = lgb.train(
        BASE_PARAMS, train_set, num_boost_round=3000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    pred = model.predict(val_feat[feature_cols])
    val_feat["pred_score"] = pred
    metrics = evaluate_full(val_feat)
    print_metrics(metrics, prefix="  ")

    importance = sorted(
        zip(feature_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True,
    )
    log("  Top 25 features:")
    for feat, gain in importance[:25]:
        log(f"    {feat:45s} {gain:,.0f}")

    model.save_model(str(ARTIFACT_DIR / "full_single_model.txt"))
    result = {**metrics, "best_iter": model.best_iteration,
              "n_features": len(feature_cols), "elapsed_min": (time.time()-t0)/60}
    save_checkpoint("stage2_result", result)
    save_checkpoint("stage2_features", feature_cols)

    del train_raw, train_split, val_split, train_feat, val_feat, train_set, val_set, model
    gc.collect()
    return result


# ========== STAGE 3: DIVERSE ENSEMBLE ==========
# Labels are remapped: {0,1,5} -> {0,1,2}. label_gain has 3 entries.
ENSEMBLE_CONFIGS = [
    {"name": "lambdarank_base", "seed": 42, "label_gain": "0,1,31"},
    {"name": "lambdarank_click3", "seed": 123, "label_gain": "0,3,31"},
    {"name": "lambdarank_bal15", "seed": 456, "label_gain": "0,1,15"},
    {"name": "lambdarank_book50", "seed": 789,
     "label_gain": "0,1,50", "num_leaves": 300, "learning_rate": 0.05},
    {"name": "lambdarank_noipw", "seed": 2024, "no_ipw": True, "label_gain": "0,1,31"},
    {"name": "rank_xendcg", "seed": 314, "objective": "rank_xendcg", "num_leaves": 350},
    {"name": "lambdarank_randup", "seed": 555, "label_gain": "0,2,25", "random_upweight": 2.0},
    {"name": "booking_clf", "seed": 666, "objective": "binary", "metric": "auc",
     "target_override": "booking_bool"},
]


def stage3_ensemble():
    if checkpoint_exists("stage3_result"):
        result = load_checkpoint("stage3_result")
        log(f"[Stage 3] SKIP (cached) — Ensemble NDCG@5={result['ensemble_ndcg5']:.5f}")
        return result

    log("\n" + "="*60)
    log("[Stage 3] DIVERSE ENSEMBLE")
    log("="*60)
    t0 = time.time()

    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(train_raw):,} rows")

    propensity = compute_position_propensity(train_raw)
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)

    default_weights = compute_sample_weights(train_split, propensity)

    log("  Building features...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    assert_no_forbidden(feature_cols)
    log(f"  {len(feature_cols)} features | {time.time()-t1:.0f}s")

    run_cfg = {
        "phase": "P0_v4_ensemble",
        "run_id": V4_RUN_ID,
        "date": datetime.now(timezone.utc).isoformat(),
        "val_frac": 0.1,
        "num_boost_round": 2000,
        "early_stopping": 80,
        "base_params": BASE_PARAMS,
        "ensemble_configs": ENSEMBLE_CONFIGS,
    }
    save_run_config(V4_RUN_ID, run_cfg)
    save_git_commit(V4_RUN_ID)
    save_feature_cols(V4_RUN_ID, feature_cols)
    save_val_meta(V4_RUN_ID, val_feat)

    assert_sorted(train_feat)
    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    # Precompute remapped labels (for lambdarank with label_gain)
    remap = {0: 0, 1: 1, 5: 2}
    train_relevance_remapped = train_feat["relevance"].map(remap).astype(np.int32)
    val_relevance_remapped = val_feat["relevance"].map(remap).astype(np.int32)

    models_info = []
    all_val_preds = {}

    for i, config in enumerate(ENSEMBLE_CONFIGS):
        cname = config["name"]
        ckpt_name = f"stage3_model_{cname}"

        if checkpoint_exists(ckpt_name):
            prev = load_checkpoint(ckpt_name)
            log(f"\n  [{i+1}/{len(ENSEMBLE_CONFIGS)}] {cname}: SKIP (NDCG@5={prev['ndcg5']:.5f})")
            models_info.append(prev)
            pred_path = ARTIFACT_DIR / f"val_preds_{cname}.npy"
            if pred_path.exists():
                all_val_preds[cname] = np.load(pred_path)
            continue

        log(f"\n  [{i+1}/{len(ENSEMBLE_CONFIGS)}] Training: {cname}")

        params = BASE_PARAMS.copy()
        config_clean = {k: v for k, v in config.items()
                        if k not in ("name", "no_ipw", "random_upweight", "target_override")}
        params.update(config_clean)

        # Weights
        if config.get("no_ipw"):
            w = np.ones(len(train_feat), dtype=np.float32)
        elif config.get("random_upweight"):
            w = default_weights.copy()
            rand_mask = train_split["random_bool"].values == 1
            w[rand_mask] *= config["random_upweight"]
        else:
            w = default_weights

        is_ranking = params.get("objective", "lambdarank") in ("lambdarank", "rank_xendcg")

        if is_ranking:
            # Use remapped labels for lambdarank with label_gain
            use_label = train_relevance_remapped if "label_gain" in params else train_feat["relevance"]
            val_label = val_relevance_remapped if "label_gain" in params else val_feat["relevance"]
            ds_train = lgb.Dataset(train_feat[feature_cols], label=use_label,
                                   group=train_groups, weight=w)
            ds_val = lgb.Dataset(val_feat[feature_cols], label=val_label,
                                 group=val_groups, reference=ds_train)
        else:
            target_col = config.get("target_override", "relevance")
            train_label = train_feat[target_col].values if target_col in train_feat.columns else train_split[target_col].values
            val_label = val_feat[target_col].values if target_col in val_feat.columns else val_split[target_col].values
            params.pop("eval_at", None)
            ds_train = lgb.Dataset(train_feat[feature_cols], label=train_label, weight=w)
            ds_val = lgb.Dataset(val_feat[feature_cols], label=val_label, reference=ds_train)

        model = lgb.train(
            params, ds_train, num_boost_round=2000,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
        )

        pred = model.predict(val_feat[feature_cols])
        all_val_preds[cname] = pred
        np.save(ARTIFACT_DIR / f"val_preds_{cname}.npy", pred)
        model.save_model(str(ARTIFACT_DIR / f"model_{cname}.txt"))

        val_feat["pred_score"] = pred
        metrics = evaluate_full(val_feat)
        print_metrics(metrics, prefix=f"    ")

        save_model_artifacts(V4_RUN_ID, cname, model, pred, feature_cols, metrics, params)

        info = {**metrics, "name": cname, "best_iter": model.best_iteration,
                "params": {k: str(v) for k, v in params.items()}}
        models_info.append(info)
        save_checkpoint(ckpt_name, info)

        del ds_train, ds_val, model
        gc.collect()

    # --- Select top diverse models for ensemble ---
    log("\n  Model summary:")
    for m in sorted(models_info, key=lambda x: x["ndcg5"], reverse=True):
        log(f"    {m['name']:30s}  NDCG@5={m['ndcg5']:.5f}  R@5={m['recall5']:.4f}")

    # Keep models above median NDCG@5
    ndcg_values = [m["ndcg5"] for m in models_info]
    median_ndcg = sorted(ndcg_values)[len(ndcg_values) // 2]
    ensemble_models = {m["name"]: all_val_preds[m["name"]]
                       for m in models_info
                       if m["ndcg5"] >= median_ndcg and m["name"] in all_val_preds}
    log(f"\n  Ensemble: {len(ensemble_models)} models (above median={median_ndcg:.5f})")

    # --- Rank-based ensemble (default) ---
    n_val = len(val_feat)
    rank_scores = np.zeros(n_val)
    model_weights = {}
    total_w = 0
    for cname, preds in ensemble_models.items():
        w = next(m["ndcg5"] for m in models_info if m["name"] == cname)
        model_weights[cname] = w
        temp = val_feat[["srch_id"]].copy()
        temp["raw"] = preds
        temp["pct_rank"] = temp.groupby("srch_id")["raw"].rank(pct=True)
        rank_scores += temp["pct_rank"].values * w
        total_w += w
    rank_scores /= total_w

    val_feat["pred_score"] = rank_scores
    rank_metrics = evaluate_full(val_feat)
    print_metrics(rank_metrics, prefix="  >>> RANK ENSEMBLE: ")

    save_ensemble_artifacts(
        V4_RUN_ID, "v4_rank", list(ensemble_models.keys()),
        model_weights, rank_scores, rank_metrics, agg_method="rank_avg_ndcg_weighted",
    )

    # Simple avg for comparison
    simple_avg = np.zeros(n_val)
    for preds in ensemble_models.values():
        simple_avg += preds
    simple_avg /= len(ensemble_models)
    val_feat["pred_score"] = simple_avg
    simple_metrics = evaluate_full(val_feat)
    print_metrics(simple_metrics, prefix="  >>> SIMPLE AVG:    ")

    save_ensemble_artifacts(
        V4_RUN_ID, "v4_simple", list(ensemble_models.keys()),
        None, simple_avg, simple_metrics, agg_method="simple_mean",
    )

    best_method = "rank" if rank_metrics["ndcg5"] >= simple_metrics["ndcg5"] else "simple"
    best_metrics = rank_metrics if best_method == "rank" else simple_metrics
    log(f"  Best: {best_method}")

    result = {
        "ensemble_ndcg5": best_metrics["ndcg5"],
        "ensemble_method": best_method,
        "rank_metrics": rank_metrics,
        "simple_metrics": simple_metrics,
        "model_weights": model_weights,
        "models": [{k: v for k, v in m.items() if k != "params"} for m in models_info],
        "elapsed_min": (time.time()-t0)/60,
    }
    save_checkpoint("stage3_result", result)
    save_checkpoint("stage3_features", feature_cols)

    del train_raw, train_split, val_split, train_feat, val_feat
    gc.collect()
    return result


# ========== STAGE 4: RETRAIN + SUBMISSION ==========
def stage4_submission():
    if checkpoint_exists("stage4_result"):
        result = load_checkpoint("stage4_result")
        log(f"[Stage 4] SKIP (cached) — {result['submission_path']}")
        return result

    s3 = load_checkpoint("stage3_result")
    feature_cols = load_checkpoint("stage3_features")
    ensemble_method = s3["ensemble_method"]
    model_weights = s3["model_weights"]
    selected_names = set(model_weights.keys())

    log("\n" + "="*60)
    log(f"[Stage 4] RETRAIN + SUBMISSION ({len(selected_names)} models)")
    log("="*60)
    t0 = time.time()

    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    propensity = compute_position_propensity(train_raw)
    default_weights = compute_sample_weights(train_raw, propensity)

    log("  Building features on full train...")
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    full_feature_cols = [c for c in feature_cols if c in full_feat.columns]
    assert_sorted(full_feat)
    full_groups = make_group_counts(full_feat)

    remap = {0: 0, 1: 1, 5: 2}
    full_relevance_remapped = full_feat["relevance"].map(remap).astype(np.int32)

    final_models = {}
    for config in ENSEMBLE_CONFIGS:
        cname = config["name"]
        if cname not in selected_names:
            continue

        ckpt = load_checkpoint(f"stage3_model_{cname}")
        best_iter = ckpt["best_iter"]

        params = BASE_PARAMS.copy()
        config_clean = {k: v for k, v in config.items()
                        if k not in ("name", "no_ipw", "random_upweight", "target_override")}
        params.update(config_clean)

        if config.get("no_ipw"):
            w = np.ones(len(full_feat), dtype=np.float32)
        elif config.get("random_upweight"):
            w = default_weights.copy()
            w[train_raw["random_bool"].values == 1] *= config["random_upweight"]
        else:
            w = default_weights

        is_ranking = params.get("objective", "lambdarank") in ("lambdarank", "rank_xendcg")

        if is_ranking:
            use_label = full_relevance_remapped if "label_gain" in params else full_feat["relevance"]
            ds = lgb.Dataset(full_feat[full_feature_cols], label=use_label,
                             group=full_groups, weight=w)
        else:
            target_col = config.get("target_override", "relevance")
            label = full_feat[target_col].values if target_col in full_feat.columns else train_raw[target_col].values
            params.pop("eval_at", None)
            ds = lgb.Dataset(full_feat[full_feature_cols], label=label, weight=w)

        log(f"  Retraining {cname} ({best_iter} rounds)...")
        model = lgb.train(params, ds, num_boost_round=best_iter)
        final_models[cname] = model
        del ds
        gc.collect()

    del full_feat
    gc.collect()

    # Test predictions
    log("  Loading and featurizing test...")
    test_raw = load_test()
    test_feat = build_features(test_raw, agg_source=train_raw, is_train=False)
    del test_raw, train_raw
    gc.collect()

    for c in full_feature_cols:
        if c not in test_feat.columns:
            test_feat[c] = np.nan

    if ensemble_method == "rank":
        rank_scores = np.zeros(len(test_feat))
        total_w = 0
        for cname, model in final_models.items():
            preds = model.predict(test_feat[full_feature_cols])
            w = model_weights[cname]
            temp = test_feat[["srch_id"]].copy()
            temp["raw"] = preds
            temp["pct_rank"] = temp.groupby("srch_id")["raw"].rank(pct=True)
            rank_scores += temp["pct_rank"].values * w
            total_w += w
        test_feat["pred_score"] = rank_scores / total_w
    else:
        test_preds = np.zeros(len(test_feat))
        for model in final_models.values():
            test_preds += model.predict(test_feat[full_feature_cols])
        test_feat["pred_score"] = test_preds / len(final_models)

    log("  Generating submission...")
    sub_path = generate_submission(test_feat, score_col="pred_score", tag="v4")

    result = {"submission_path": str(sub_path), "elapsed_min": (time.time()-t0)/60}
    save_checkpoint("stage4_result", result)
    return result


# ========== MAIN ==========
def main():
    t0 = time.time()

    s1 = stage1_sanity()
    if s1["ndcg5"] < 0.28:
        log(f"\n[ABORT] Sanity NDCG@5={s1['ndcg5']:.5f} < 0.28. Fix features before proceeding.")
        return

    s2 = stage2_full_single()
    log(f"\n  Sanity={s1['ndcg5']:.5f} → Full={s2['ndcg5']:.5f} (V3 baseline={V3_BASELINE_NDCG})")

    if s2["ndcg5"] < V3_BASELINE_NDCG:
        log(f"\n[GATE] Full NDCG@5={s2['ndcg5']:.5f} < V3={V3_BASELINE_NDCG}. "
            f"Stopping. Fix features/params before ensemble.")
        return

    s3 = stage3_ensemble()
    s4 = stage4_submission()

    log(f"\n{'='*60}")
    log(f"DONE in {(time.time()-t0)/60:.1f} min")
    log(f"  Sanity:   {s1['ndcg5']:.5f}")
    log(f"  Full:     {s2['ndcg5']:.5f}")
    log(f"  Ensemble: {s3['ensemble_ndcg5']:.5f}")
    log(f"  Sub:      {s4['submission_path']}")
    log("="*60)


if __name__ == "__main__":
    main()
