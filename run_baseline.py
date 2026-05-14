"""
V2 pipeline: LightGBM LambdaRank with EDA-driven feature engineering.
Key improvements over v1:
  - Bayesian-smoothed aggregates to reduce leakage
  - Proper train/val split BEFORE computing aggregates (no leakage)
  - Missing value indicator flags
  - Within-query z-scores and normalized ranks
  - Quality interaction features (value_score, is_domestic, etc.)
  - Better hyperparameters (more trees, lower LR)
"""
import gc
import time
import numpy as np
import lightgbm as lgb

from src.config import RANDOM_SEED
from src.data_loader import load_train, load_test, make_target, get_feature_columns, split_val
from src.features import build_features
from src.evaluate import evaluate_ndcg
from src.submission import generate_submission


def make_group_counts(df):
    return df.groupby("srch_id").size().values


def train_lambdarank(train_df, val_df, feature_cols):
    X_train = train_df[feature_cols]
    y_train = train_df["relevance"]
    groups_train = make_group_counts(train_df)

    X_val = val_df[feature_cols]
    y_val = val_df["relevance"]
    groups_val = make_group_counts(val_df)

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_set = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_set)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
        "seed": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80),
            lgb.log_evaluation(period=100),
        ],
    )

    return model


def main():
    t0 = time.time()

    # --- Load ---
    print("Loading training data...")
    train_raw = load_train()
    print(f"  Loaded: {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")

    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    # --- Split BEFORE feature engineering to prevent aggregate leakage ---
    print("Splitting train/val by srch_id (BEFORE aggregates)...")
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    print(f"  Train: {len(train_split):,} rows ({train_split['srch_id'].nunique():,} searches)")
    print(f"  Val:   {len(val_split):,} rows ({val_split['srch_id'].nunique():,} searches)")

    # --- Feature engineering (aggregates computed from train_split only) ---
    print("Engineering features...")
    t1 = time.time()
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)

    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)

    feature_cols = get_feature_columns(train_feat)
    missing_in_val = [c for c in feature_cols if c not in val_feat.columns]
    if missing_in_val:
        print(f"  Warning: {len(missing_in_val)} cols missing in val, filling NaN")
        for c in missing_in_val:
            val_feat[c] = np.nan
    feature_cols = [c for c in feature_cols if c in val_feat.columns]

    print(f"  Features: {len(feature_cols)} | Time: {time.time() - t1:.1f}s")

    # --- Train ---
    print("\nTraining LambdaRank...")
    model = train_lambdarank(train_feat, val_feat, feature_cols)
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # --- Evaluate ---
    print("\nEvaluating on validation set...")
    val_feat = val_feat.copy()
    val_feat["pred_score"] = model.predict(val_feat[feature_cols])
    ndcg = evaluate_ndcg(val_feat, score_col="pred_score", k=5)
    print(f"  Validation NDCG@5: {ndcg:.5f}")

    # --- Feature importance ---
    importance = sorted(
        zip(feature_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 25 features by gain:")
    for feat, gain in importance[:25]:
        print(f"  {feat:45s} {gain:,.0f}")

    # --- Retrain on ALL training data for submission ---
    print("\n--- Retraining on full training data for submission ---")
    del train_split, val_split, train_feat, val_feat
    gc.collect()

    t2 = time.time()
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    full_feature_cols = [c for c in feature_cols if c in full_feat.columns]
    print(f"  Feature engineering: {time.time() - t2:.1f}s")

    X_full = full_feat[full_feature_cols]
    y_full = full_feat["relevance"]
    groups_full = make_group_counts(full_feat)

    full_set = lgb.Dataset(X_full, label=y_full, group=groups_full)
    params_final = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
        "seed": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    final_model = lgb.train(params_final, full_set, num_boost_round=best_iter)

    model_path = "models/v2_lambdarank.txt"
    final_model.save_model(model_path)
    print(f"  Model saved: {model_path}")

    # --- Predict on test ---
    # Keep train_raw for aggregates, free everything else
    agg_source = train_raw
    del full_feat, X_full, y_full
    gc.collect()

    print("\nLoading test data...")
    test_raw = load_test()
    print(f"  Test: {len(test_raw):,} rows, {test_raw['srch_id'].nunique():,} searches")

    print("Engineering test features...")
    t3 = time.time()
    test_feat = build_features(test_raw, agg_source=agg_source, is_train=False)
    del test_raw, agg_source
    gc.collect()
    print(f"  Time: {time.time() - t3:.1f}s")

    missing_in_test = [c for c in full_feature_cols if c not in test_feat.columns]
    if missing_in_test:
        print(f"  Warning: {len(missing_in_test)} features missing in test: {missing_in_test}")
        for c in missing_in_test:
            test_feat[c] = np.nan

    test_feat["pred_score"] = final_model.predict(test_feat[full_feature_cols])

    # --- Generate submission ---
    print("\nGenerating submission...")
    sub_path = generate_submission(test_feat, score_col="pred_score", tag="v2_eda")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"Validation NDCG@5: {ndcg:.5f}")

    return ndcg, sub_path


if __name__ == "__main__":
    main()
