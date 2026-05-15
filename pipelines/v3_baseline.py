"""
V3 pipeline: K-fold target encoding + IPW + tuned GBDT.
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


PARAMS = {
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
    "seed": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}


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

    print("Engineering features...")
    t1 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c in val_feat.columns]
    print(f"  {len(feature_cols)} features | {time.time() - t1:.0f}s")

    # --- Train ---
    print("\nTraining LambdaRank (GBDT, lr=0.03, leaves=400)...")
    train_set = lgb.Dataset(
        train_feat[feature_cols], label=train_feat["relevance"],
        group=make_group_counts(train_feat), weight=weights
    )
    val_set = lgb.Dataset(
        val_feat[feature_cols], label=val_feat["relevance"],
        group=make_group_counts(val_feat), reference=train_set
    )
    model = lgb.train(
        PARAMS, train_set, num_boost_round=3000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    val_feat = val_feat.copy()
    val_feat["pred_score"] = model.predict(val_feat[feature_cols])
    ndcg = evaluate_ndcg(val_feat, score_col="pred_score", k=5)
    print(f"\n  >>> Validation NDCG@5: {ndcg:.5f} <<<")

    importance = sorted(
        zip(feature_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop 20 features:")
    for feat, gain in importance[:20]:
        print(f"  {feat:45s} {gain:,.0f}")

    # --- Retrain on full data ---
    print(f"\nRetraining on full data ({best_iter} rounds)...")
    del train_split, val_split, train_feat, val_feat
    gc.collect()

    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    full_weights = compute_sample_weights(train_raw, propensity)
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    full_feature_cols = [c for c in feature_cols if c in full_feat.columns]

    full_set = lgb.Dataset(
        full_feat[full_feature_cols], label=full_feat["relevance"],
        group=make_group_counts(full_feat), weight=full_weights
    )
    final_model = lgb.train(PARAMS, full_set, num_boost_round=best_iter)
    final_model.save_model("models/v3_gbdt.txt")

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

    test_feat["pred_score"] = final_model.predict(test_feat[full_feature_cols])

    print("Generating submission...")
    sub_path = generate_submission(test_feat, score_col="pred_score", tag="v3")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min | Val NDCG@5: {ndcg:.5f}")
    return ndcg, sub_path


if __name__ == "__main__":
    main()
