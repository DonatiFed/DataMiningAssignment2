"""
Full baseline pipeline: LightGBM LambdaRank with complete feature engineering.
"""
import time
import numpy as np
import lightgbm as lgb

from src.config import RANDOM_SEED
from src.data_loader import load_train, load_test, make_target, get_feature_columns, split_val
from src.features import build_features
from src.evaluate import evaluate_ndcg
from src.submission import generate_submission

SAMPLE_FRAC = None  # use all data


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
        "num_leaves": 127,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def main():
    t0 = time.time()

    # --- Load ---
    print("Loading training data...")
    train_raw = load_train(sample_frac=SAMPLE_FRAC)
    print(f"  Loaded: {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")

    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)

    # --- Feature engineering ---
    print("Engineering features (full pipeline)...")
    t1 = time.time()
    train_feat = build_features(train_raw, agg_source=train_raw)
    feature_cols = get_feature_columns(train_feat)
    print(f"  Features: {len(feature_cols)} | Time: {time.time() - t1:.1f}s")

    # --- Split ---
    print("Splitting train/val by srch_id...")
    train_df, val_df = split_val(train_feat, val_frac=0.1)
    print(f"  Train: {len(train_df):,} rows ({train_df['srch_id'].nunique():,} searches)")
    print(f"  Val:   {len(val_df):,} rows ({val_df['srch_id'].nunique():,} searches)")

    # --- Train ---
    print("\nTraining LambdaRank...")
    model = train_lambdarank(train_df, val_df, feature_cols)
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # --- Evaluate ---
    print("\nEvaluating on validation set...")
    val_df = val_df.copy()
    val_df["pred_score"] = model.predict(val_df[feature_cols])
    ndcg = evaluate_ndcg(val_df, score_col="pred_score", k=5)
    print(f"  Validation NDCG@5: {ndcg:.5f}")

    # --- Feature importance ---
    importance = sorted(
        zip(feature_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 20 features by gain:")
    for feat, gain in importance[:20]:
        print(f"  {feat:40s} {gain:,.0f}")

    # --- Retrain on full data ---
    print("\nRetraining on full data for submission...")
    full_train = train_feat.sort_values("srch_id").reset_index(drop=True)
    X_full = full_train[feature_cols]
    y_full = full_train["relevance"]
    groups_full = make_group_counts(full_train)

    full_set = lgb.Dataset(X_full, label=y_full, group=groups_full)
    params_final = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    final_model = lgb.train(params_final, full_set, num_boost_round=best_iter)

    model_path = "models/baseline_lambdarank.txt"
    final_model.save_model(model_path)
    print(f"  Model saved: {model_path}")

    # --- Predict on test ---
    print("\nLoading test data...")
    test_raw = load_test()
    print(f"  Test: {len(test_raw):,} rows, {test_raw['srch_id'].nunique():,} searches")

    print("Engineering test features...")
    t2 = time.time()
    test_feat = build_features(test_raw, agg_source=train_raw)
    print(f"  Time: {time.time() - t2:.1f}s")

    # Align columns: only use features that exist in both train and test
    missing_in_test = [c for c in feature_cols if c not in test_feat.columns]
    if missing_in_test:
        print(f"  Warning: {len(missing_in_test)} features missing in test, filling NaN: {missing_in_test}")
        for c in missing_in_test:
            test_feat[c] = np.nan

    test_feat["pred_score"] = final_model.predict(test_feat[feature_cols])

    # --- Generate submission ---
    print("\nGenerating submission...")
    sub_path = generate_submission(test_feat, score_col="pred_score", tag="v1_full")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"Validation NDCG@5: {ndcg:.5f}")

    return ndcg, sub_path


if __name__ == "__main__":
    main()
