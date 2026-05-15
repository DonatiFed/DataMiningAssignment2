"""
V5 pipeline: V4 features + label_gain tuning + DART member + more diverse ensemble.
"""
import gc
import time
import numpy as np
import lightgbm as lgb

from src.config import RANDOM_SEED
from src.data_loader import load_train, load_test, make_target, get_feature_columns, split_val
from src.features import build_features, compute_position_propensity, compute_sample_weights
from src.evaluate import evaluate_ndcg
from src.submission import generate_submission


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
}

# Label gains for relevance levels 0, 1, 5
# Default would be 2^0-1=0, 2^1-1=1, 2^5-1=31
# We tune this: booking (5) vs click (1) ratio matters
ENSEMBLE_CONFIGS = [
    # Standard models with default gain (0, 1, 31)
    {"seed": 42, "label_gain": [0, 1, 31]},
    {"seed": 123, "label_gain": [0, 1, 31]},
    # Boosted click importance (clicks matter more for NDCG since they're common)
    {"seed": 456, "label_gain": [0, 3, 31]},
    # Lower booking premium (more balanced)
    {"seed": 789, "num_leaves": 300, "learning_rate": 0.05, "subsample": 0.8,
     "label_gain": [0, 1, 15]},
    # Aggressive booking focus
    {"seed": 2024, "num_leaves": 500, "colsample_bytree": 0.5, "subsample": 0.65,
     "label_gain": [0, 1, 50]},
    # Different regularisation
    {"seed": 314, "num_leaves": 350, "reg_alpha": 0.5, "reg_lambda": 2.0,
     "label_gain": [0, 2, 25]},
    # DART boosting (different regularisation mechanism)
    {"seed": 555, "boosting_type": "dart", "num_leaves": 300, "learning_rate": 0.05,
     "drop_rate": 0.1, "skip_drop": 0.5, "label_gain": [0, 1, 31],
     "max_rounds": 800},  # DART is slower; fewer rounds
]


def make_group_counts(df):
    return df.groupby("srch_id").size().values


def main():
    t0 = time.time()

    print("Loading training data...")
    train_raw = load_train()
    print(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")

    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    print("Computing position propensity (IPW)...")
    propensity = compute_position_propensity(train_raw)

    print("Splitting train/val by srch_id...")
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    print(f"  Train: {len(train_split):,} | Val: {len(val_split):,}")

    weights = compute_sample_weights(train_split, propensity)
    print(f"  IPW range: [{weights.min():.2f}, {weights.max():.2f}]")

    print("Engineering features (V5: enhanced + cross-entity + interactions)...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    print(f"  {len(feature_cols)} features | {time.time() - t1:.0f}s")

    train_set = lgb.Dataset(
        train_feat[feature_cols], label=train_feat["relevance"],
        group=make_group_counts(train_feat), weight=weights
    )
    val_set = lgb.Dataset(
        val_feat[feature_cols], label=val_feat["relevance"],
        group=make_group_counts(val_feat), reference=train_set
    )

    # --- Ensemble training ---
    print(f"\nTraining {len(ENSEMBLE_CONFIGS)}-model ensemble...")
    models = []
    val_preds = np.zeros(len(val_feat))

    for i, config in enumerate(ENSEMBLE_CONFIGS):
        params = BASE_PARAMS.copy()
        config = config.copy()
        max_rounds = config.pop("max_rounds", 3000)
        params.update(config)
        seed = params.pop("seed")
        params["seed"] = seed

        label_gain = params.get("label_gain")
        label_gain_str = f", gain={label_gain}" if label_gain else ""
        boosting = params.get("boosting_type", "gbdt")
        print(f"\n  Model {i+1}/{len(ENSEMBLE_CONFIGS)} (seed={seed}, {boosting}, "
              f"leaves={params.get('num_leaves',400)}{label_gain_str})...")

        # DART doesn't support early stopping well
        if boosting == "dart":
            model = lgb.train(
                params, train_set, num_boost_round=max_rounds,
                valid_sets=[val_set],
                callbacks=[lgb.log_evaluation(200)],
            )
            best_iter = max_rounds
        else:
            model = lgb.train(
                params, train_set, num_boost_round=max_rounds,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
            )
            best_iter = model.best_iteration

        pred = model.predict(val_feat[feature_cols])
        val_preds += pred
        models.append((model, best_iter, params))

        single_ndcg = evaluate_ndcg(
            val_feat.assign(pred_score=pred), score_col="pred_score", k=5
        )
        print(f"    best_iter={best_iter}, NDCG@5={single_ndcg:.5f}")

    val_preds /= len(ENSEMBLE_CONFIGS)
    val_feat_eval = val_feat.copy()
    val_feat_eval["pred_score"] = val_preds
    ensemble_ndcg = evaluate_ndcg(val_feat_eval, score_col="pred_score", k=5)
    print(f"\n  >>> Ensemble Validation NDCG@5: {ensemble_ndcg:.5f} <<<")

    importance = sorted(
        zip(feature_cols, models[0][0].feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop 30 features (model 1):")
    for feat, gain in importance[:30]:
        print(f"  {feat:45s} {gain:,.0f}")

    # --- Retrain on full data ---
    print(f"\nRetraining {len(ENSEMBLE_CONFIGS)} models on full data...")
    del train_split, val_split, train_feat, val_feat, val_feat_eval, train_set, val_set
    gc.collect()

    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    full_weights = compute_sample_weights(train_raw, propensity)
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    full_feature_cols = [c for c in feature_cols if c in full_feat.columns]

    full_set = lgb.Dataset(
        full_feat[full_feature_cols], label=full_feat["relevance"],
        group=make_group_counts(full_feat), weight=full_weights
    )

    final_models = []
    for i, (_, best_iter, params) in enumerate(models):
        print(f"  Retraining model {i+1} ({best_iter} rounds, {params.get('boosting_type','gbdt')})...")
        final_model = lgb.train(params, full_set, num_boost_round=best_iter)
        final_models.append(final_model)

    final_models[0].save_model("models/v5_gbdt.txt")

    # --- Test ---
    agg_source = train_raw
    del full_feat
    gc.collect()

    print("Loading and featurizing test...")
    test_raw = load_test()
    test_feat = build_features(test_raw, agg_source=agg_source, is_train=False)
    del test_raw, agg_source
    gc.collect()

    for c in full_feature_cols:
        if c not in test_feat.columns:
            test_feat[c] = np.nan

    test_preds = np.zeros(len(test_feat))
    for model in final_models:
        test_preds += model.predict(test_feat[full_feature_cols])
    test_preds /= len(final_models)
    test_feat["pred_score"] = test_preds

    print("Generating submission...")
    sub_path = generate_submission(test_feat, score_col="pred_score", tag="v5")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min | Ensemble NDCG@5: {ensemble_ndcg:.5f}")
    return ensemble_ndcg, sub_path


if __name__ == "__main__":
    main()
