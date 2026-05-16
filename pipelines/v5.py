"""
V5 ensemble submission — 7-member rank-averaged Kaggle submission.

Members (from overnight_20260516_003443):
  B3, F13, D4, A1, B10, E4, E3

Method:
  - Retrain each member on FULL train using its exact overnight config.
  - num_boost_round = that member's val best_iter.
  - No early stopping, no validation set.
  - Predict test for each member.
  - Within srch_id: convert each model's scores to ranks (1=top).
  - Average ranks across members with equal weights → final ordering.
  - Save submission CSV.

Outputs:
  artifacts/v5_ensemble_submit/
    ├── ensemble_config.json
    ├── feature_cols_<member>.json
    ├── importance_<member>.csv
    ├── test_pred_<member>.npy
    ├── ensemble_test_rank_avg.npy
    ├── submission.csv
    └── run_summary.md
  models/v5_ensemble_submit/model_<member>.txt
  submissions/submission_v5_ensemble_<TS>.csv
  /home/ubuntu/experiment_artifacts/v5_ensemble_submit/  (copy)

Constraints (per user instructions):
  - 7 members exactly. No extras. No greedy/wildcard members.
  - Equal-weight rank averaging. No weight tuning.
  - No feature changes beyond per-member feature_filter recipes.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS_DIR  # noqa: E402
from src.data_loader import load_train, load_test, make_target, get_feature_columns  # noqa: E402
from src.features import (  # noqa: E402
    build_features, compute_position_propensity, FORBIDDEN_FEATURES,
)
from src.artifacts import run_dirs, save_run_config, save_git_commit  # noqa: E402

# ============================================================================
# Constants
# ============================================================================
RUN_ID = "v5_ensemble_submit"
EXTERNAL_COPY_ROOT = Path("/home/ubuntu/experiment_artifacts")
OVERNIGHT_RUN = ROOT / "artifacts" / "overnight_20260516_003443"
FEATURE_AUDIT_CSV = ROOT / "experiment_logs" / "feature_audit.csv"

V4_ENSEMBLE_LOCAL = 0.42512
V4_ENSEMBLE_KAGGLE = 0.42021
ANCHOR_LOCAL = 0.42191

# Base params (V4 backbone — match BASE_PARAMS in run_overnight_experiments.py)
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
    "seed": 456,
}

# Weighting profiles (copied from run_overnight_experiments.py)
WEIGHTING_PROFILES = {
    "ipw_default":  {"base": "ipw", "clip": (0.1, 10.0)},
    "no_ipw":       {"base": "none"},
    "ipw_clip3":    {"base": "ipw", "clip": (0.1, 3.0)},
}

# Member spec — exactly the 7 approved members.
# Each entry mirrors the overnight config for that id.
MEMBERS = [
    {  # B3
        "id": "B3", "label_gain": "0,2,15", "weighting": "ipw_clip3",
        "feature_filter": None, "row_filter": None, "param_overrides": {},
        "best_iter": 326,  "val_ndcg5": 0.42396,
        "description": "lg=0,2,15, w=ipw_clip3 (overnight EXCELLENT)",
    },
    {  # F13
        "id": "F13", "label_gain": "0,3,15", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": "positive_q_only", "param_overrides": {},
        "best_iter": 500, "val_ndcg5": 0.42253,
        "description": "positive queries only, lg=0,3,15",
    },
    {  # D4
        "id": "D4", "label_gain": "0,2,15", "weighting": "ipw_default",
        "feature_filter": "top_120", "row_filter": None, "param_overrides": {},
        "best_iter": 523, "val_ndcg5": 0.42228,
        "description": "feature_filter=top_120, lg=0,2,15",
    },
    {  # A1
        "id": "A1", "label_gain": "0,2,12", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": None, "param_overrides": {},
        "best_iter": 385, "val_ndcg5": 0.42205,
        "description": "label_gain=0,2,12",
    },
    {  # B10
        "id": "B10", "label_gain": "0,2,20", "weighting": "no_ipw",
        "feature_filter": None, "row_filter": None, "param_overrides": {},
        "best_iter": 358, "val_ndcg5": 0.42198,
        "description": "lg=0,2,20, w=no_ipw",
    },
    {  # E4
        "id": "E4", "label_gain": "0,2,15", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": None,
        "param_overrides": {"num_leaves": 400, "min_child_samples": 100},
        "best_iter": 518, "val_ndcg5": 0.42291,
        "description": "num_leaves=400, min_child=100",
    },
    {  # E3
        "id": "E3", "label_gain": "0,2,15", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": None,
        "param_overrides": {"num_leaves": 512, "min_child_samples": 100, "reg_lambda": 2.0},
        "best_iter": 551, "val_ndcg5": 0.42185,
        "description": "num_leaves=512, min_child=100, reg_lambda=2.0",
    },
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Feature filter (copied from run_overnight_experiments.py)
# ============================================================================
def apply_feature_filter(feature_cols: list[str], filter_name: str | None,
                         audit_df: pd.DataFrame | None) -> list[str]:
    if filter_name is None:
        return list(feature_cols)
    if filter_name.startswith("top_"):
        n = int(filter_name.split("_")[1])
        if audit_df is None:
            raise RuntimeError("feature_audit.csv required for top_N filter")
        keep = set(audit_df.head(n)["feature"].tolist())
        return [c for c in feature_cols if c in keep]
    raise ValueError(f"Unsupported feature_filter in v5_ensemble_submit: {filter_name}")


# ============================================================================
# Row filter
# ============================================================================
def apply_row_filter(train_split: pd.DataFrame, filter_name: str | None) -> np.ndarray:
    n = len(train_split)
    if filter_name is None:
        return np.ones(n, dtype=bool)
    if filter_name == "positive_q_only":
        eng = train_split.groupby("srch_id")["click_bool"].transform("sum")
        return (eng.values > 0)
    raise ValueError(f"Unsupported row_filter in v5_ensemble_submit: {filter_name}")


# ============================================================================
# Weights
# ============================================================================
def compute_weights(train_split: pd.DataFrame, propensity: pd.Series,
                    mode: str) -> np.ndarray:
    if mode not in WEIGHTING_PROFILES:
        raise ValueError(f"Unknown weighting mode: {mode}")
    profile = WEIGHTING_PROFILES[mode]
    n = len(train_split)
    if profile["base"] == "none":
        return np.ones(n, dtype=np.float32)
    max_prop = float(propensity.max())
    pos_w = train_split["position"].map(
        lambda p: 1.0 if propensity.get(p, 0) <= 0 else max_prop / propensity[p]
    ).astype(np.float32).values
    is_nonrandom = (train_split["random_bool"].values == 0)
    w = np.where(is_nonrandom, pos_w, 1.0).astype(np.float32)
    clip_lo, clip_hi = profile.get("clip", (0.1, 10.0))
    w = np.clip(w, clip_lo, clip_hi).astype(np.float32)
    return w


def make_group_counts(df: pd.DataFrame) -> np.ndarray:
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df), "group counts mismatch"
    return g


# ============================================================================
# Rank averaging
# ============================================================================
def ranks_within_srch(scores: np.ndarray, srch_id: np.ndarray) -> np.ndarray:
    """Within each srch_id, rank descending: best=1, worst=group size. Stable ties."""
    # Block-wise group structure (srch_id is contiguous after sorting)
    changes = np.concatenate([[0], np.where(np.diff(srch_id) != 0)[0] + 1, [len(srch_id)]])
    starts = changes[:-1]
    sizes = np.diff(changes)
    ranks = np.empty_like(scores, dtype=np.float32)
    for s, n in zip(starts, sizes):
        sl = slice(s, s + n)
        order = np.argsort(-scores[sl], kind="stable")
        r = np.empty(n, dtype=np.float32)
        r[order] = np.arange(1, n + 1, dtype=np.float32)
        ranks[sl] = r
    return ranks


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    log(f"=== V5 ensemble submission — run_id={RUN_ID} ===")
    log(f"Members: {[m['id'] for m in MEMBERS]}")

    # ---- Output dirs ----
    model_dir, art_dir = run_dirs(RUN_ID)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    save_git_commit(RUN_ID)

    ensemble_cfg = {
        "phase": "V5_ENSEMBLE_SUBMIT",
        "run_id": RUN_ID,
        "date": datetime.now(timezone.utc).isoformat(),
        "method": "equal_weight_rank_average_within_srch_id",
        "members": MEMBERS,
        "n_members": len(MEMBERS),
        "anchor_ref": "V4_ENSEMBLE (local=0.42512, Kaggle=0.42021)",
        "expected_val_basis": (
            "Val NDCG@5 for E2_top5 was 0.42622; for E2_top5+E4+E3 we did NOT "
            "have a frozen number — the LOO/W2 results suggested 7 robust members."
        ),
        "notes": (
            "Ensemble selected from val-only search (artifacts/ensemble_search_20260516_085823). "
            "Not greedy. No weight tuning. No feature changes beyond per-member feature_filter."
        ),
    }
    save_run_config(RUN_ID, ensemble_cfg)
    json.dump(ensemble_cfg, open(art_dir / "ensemble_config.json", "w"), indent=2, default=str)

    # ---- Load audit (top_120 needs it) ----
    audit_df = pd.read_csv(FEATURE_AUDIT_CSV)
    audit_df = audit_df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    log(f"feature_audit.csv loaded: {len(audit_df)} features")

    # ---- Load + featurize full train ONCE ----
    log("Loading full train...")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")

    log("Computing propensity (full train)...")
    propensity = compute_position_propensity(train_raw)
    log(f"  propensity computed (max={propensity.max():.4f})")

    log("Building features on full train (agg_source=train_raw, is_train=True)...")
    t1 = time.time()
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    feature_cols_all = get_feature_columns(full_feat)
    leaked = set(feature_cols_all) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden columns leaked: {leaked}"
    log(f"  features built: {len(feature_cols_all)} cols | {time.time()-t1:.0f}s")

    # ---- Build full label (remapped 0/1/2 = 0/click/booked → gain 0/1/5) ----
    remap = {0: 0, 1: 1, 5: 2}
    full_label_all = full_feat["relevance"].map(remap).astype(np.int32)

    # We need train_raw aligned with full_feat for weighting / row filter
    # build_features keeps rows ordered by srch_id; train_raw is also sorted.
    # The output preserves row order and srch_id grouping.
    assert (full_feat["srch_id"].values == train_raw["srch_id"].values).all(), \
        "row order mismatch between train_raw and full_feat"

    # ============================================================
    # Per-member training on full train
    # ============================================================
    train_models = {}
    importance_dfs = {}

    for member in MEMBERS:
        mid = member["id"]
        log(f"\n--- Training member {mid} ({member['description']}) ---")

        # Feature subset
        feat_cols = apply_feature_filter(feature_cols_all, member["feature_filter"], audit_df)
        feat_cols = [c for c in feat_cols if c in full_feat.columns]
        log(f"  features: {len(feat_cols)}")
        json.dump(list(feat_cols), open(art_dir / f"feature_cols_{mid}.json", "w"), indent=2)

        # Row mask
        row_mask = apply_row_filter(train_raw, member["row_filter"])
        if not row_mask.all():
            full_feat_m = full_feat.loc[row_mask].reset_index(drop=True)
            train_raw_m = train_raw.loc[row_mask].reset_index(drop=True)
            label_m = full_label_all.loc[row_mask].reset_index(drop=True).values
            log(f"  row filter '{member['row_filter']}': "
                f"{len(full_feat_m):,}/{len(full_feat):,} rows ({100*len(full_feat_m)/len(full_feat):.1f}%)")
        else:
            full_feat_m = full_feat
            train_raw_m = train_raw
            label_m = full_label_all.values

        # Weights
        weights = compute_weights(train_raw_m, propensity, member["weighting"])
        log(f"  weights: range=[{weights.min():.3f}, {weights.max():.3f}] mean={weights.mean():.3f}")

        # Group counts
        groups = make_group_counts(full_feat_m)

        # Params
        params = BASE_PARAMS.copy()
        if member["label_gain"]:
            params["label_gain"] = member["label_gain"]
        params.update(member["param_overrides"])

        nbr = int(member["best_iter"])
        log(f"  params: lg={params.get('label_gain')}, num_leaves={params['num_leaves']}, "
            f"min_child={params['min_child_samples']}, reg_λ={params['reg_lambda']}, lr={params['learning_rate']}")
        log(f"  num_boost_round = {nbr} (= val best_iter)")

        # Fresh Dataset (V4-style — no .construct(), no free_raw_data)
        ds = lgb.Dataset(
            full_feat_m[feat_cols], label=label_m, group=groups, weight=weights,
        )

        t_train = time.time()
        model = lgb.train(
            params, ds, num_boost_round=nbr,
            callbacks=[lgb.log_evaluation(max(nbr // 5, 50))],
        )
        log(f"  trained {nbr} rounds in {(time.time()-t_train)/60:.1f} min")

        # Save model
        model.save_model(str(model_dir / f"model_{mid}.txt"))
        train_models[mid] = model

        # Importance
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        imp_df.to_csv(art_dir / f"importance_{mid}.csv", index=False)
        importance_dfs[mid] = imp_df

        # Cleanup
        del ds, full_feat_m, train_raw_m, label_m, weights, groups
        gc.collect()

    # ---- Free train features before loading test ----
    log("\nFreeing full_feat / train_raw before test featurization...")
    del full_feat, full_label_all
    gc.collect()

    # ============================================================
    # Test featurization (once)
    # ============================================================
    log("\nLoading test set...")
    test_raw = load_test()
    test_raw = test_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(test_raw):,} rows, {test_raw['srch_id'].nunique():,} searches")

    log("Building test features (agg_source=full train)...")
    t2 = time.time()
    test_feat = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"  features built: {len(test_feat):,} rows | {time.time()-t2:.0f}s")

    # train_raw not needed past this point
    del train_raw
    gc.collect()

    # ============================================================
    # Per-member test prediction + rank averaging
    # ============================================================
    test_srch_ids = test_feat["srch_id"].values
    test_prop_ids = test_feat["prop_id"].values
    n_test = len(test_feat)

    # Sum of ranks (for equal-weight averaging we keep the sum and divide at the end —
    # for sorting purposes the constant divisor doesn't matter).
    rank_sum = np.zeros(n_test, dtype=np.float64)

    for member in MEMBERS:
        mid = member["id"]
        log(f"\n--- Predicting test with {mid} ---")
        feat_cols = json.load(open(art_dir / f"feature_cols_{mid}.json"))
        # Add any missing cols as NaN (mirrors phase2_submit.py behavior)
        for c in feat_cols:
            if c not in test_feat.columns:
                test_feat[c] = np.nan
        model = train_models[mid]
        t_pred = time.time()
        preds = model.predict(test_feat[feat_cols]).astype(np.float32)
        log(f"  predicted in {time.time()-t_pred:.0f}s  pred range=[{preds.min():.3f}, {preds.max():.3f}]")
        np.save(art_dir / f"test_pred_{mid}.npy", preds)

        # Ranks within srch_id (best=1)
        t_rank = time.time()
        ranks = ranks_within_srch(preds, test_srch_ids)
        log(f"  ranks computed in {time.time()-t_rank:.0f}s")
        rank_sum += ranks

    # Equal-weight average
    rank_avg = rank_sum / len(MEMBERS)
    np.save(art_dir / "ensemble_test_rank_avg.npy", rank_avg.astype(np.float32))

    # ============================================================
    # Submission CSV
    # ============================================================
    log("\nGenerating submission CSV (sort by rank ascending within srch_id)...")
    sub_df = pd.DataFrame({
        "srch_id": test_srch_ids,
        "prop_id": test_prop_ids,
        "rank_avg": rank_avg,
    })
    sub_df = sub_df.sort_values(["srch_id", "rank_avg"], ascending=[True, True]).reset_index(drop=True)
    submission = sub_df[["srch_id", "prop_id"]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_name = f"submission_v5_ensemble_{timestamp}.csv"
    sub_path = SUBMISSIONS_DIR / sub_name
    submission.to_csv(sub_path, index=False)
    submission.to_csv(art_dir / "submission.csv", index=False)
    log(f"  submission rows: {len(submission):,}  unique searches: {submission['srch_id'].nunique():,}")
    log(f"  saved: {sub_path}")

    # ============================================================
    # Run summary markdown
    # ============================================================
    elapsed_min = (time.time() - t0) / 60
    summary_md = f"""# V5 Ensemble Submission — {RUN_ID}

_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

## Members (7) — equal-weight rank averaging within srch_id

| id  | val NDCG@5 | best_iter | label_gain | weighting   | feature_filter | row_filter      | params override |
|-----|------------|-----------|------------|-------------|----------------|-----------------|------------------|
""" + "\n".join([
        f"| {m['id']:3s} | {m['val_ndcg5']:.5f}    | {m['best_iter']:9d} | "
        f"{m['label_gain']:10s} | {m['weighting']:11s} | "
        f"{str(m['feature_filter'] or '-'):14s} | {str(m['row_filter'] or '-'):15s} | "
        f"{str(m['param_overrides']) if m['param_overrides'] else '-'} |"
        for m in MEMBERS
    ]) + f"""

## Baselines

| Reference | NDCG@5 |
|-----------|--------|
| Anchor single (V4 bal15) | {ANCHOR_LOCAL:.5f} |
| V4 ensemble local         | {V4_ENSEMBLE_LOCAL:.5f} |
| V4 ensemble Kaggle public | {V4_ENSEMBLE_KAGGLE:.5f} |
| Ensemble val NDCG@5 (E2_top5 + E4 + E3, from val-only search) | ~0.42622+ (val) |

## Outputs

- Submission CSV: `{sub_path}`
- Submission copy: `{art_dir / 'submission.csv'}`
- Submission rows: {len(submission):,}
- Unique srch_ids: {submission['srch_id'].nunique():,}
- Models: `models/{RUN_ID}/model_<id>.txt` (one per member)
- Per-member test predictions: `{art_dir / 'test_pred_<id>.npy'}`
- Final rank average: `{art_dir / 'ensemble_test_rank_avg.npy'}`
- Run config: `{art_dir / 'ensemble_config.json'}`

## Wall-clock

- Total: {elapsed_min:.1f} min

## Method

For each member m: train LightGBM on FULL train with its exact overnight config
(label_gain, weighting, feature_filter, row_filter, param_overrides, seed=456),
for `best_iter` rounds (no early stopping, no validation). Predict each test row
to get continuous score `p_m`. Within each `srch_id`, rank rows by `p_m` descending
to get integer ranks `r_m` (1=best). The ensemble score for each row is
`mean(r_1, …, r_7)`. Sort rows ascending by ensemble score within srch_id;
emit `(srch_id, prop_id)`.
"""
    (art_dir / "run_summary.md").write_text(summary_md)
    log(f"  saved: {art_dir / 'run_summary.md'}")

    # ============================================================
    # External copy
    # ============================================================
    external_dir = EXTERNAL_COPY_ROOT / RUN_ID
    log(f"\nCopying artifacts to {external_dir} ...")
    if external_dir.exists():
        shutil.rmtree(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    for f in art_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, external_dir / f.name)
    (external_dir / "model").mkdir(exist_ok=True)
    for f in model_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, external_dir / "model" / f.name)
    n_copied = sum(1 for _ in external_dir.rglob('*') if _.is_file())
    log(f"  copied {n_copied} files")

    # ============================================================
    # Final
    # ============================================================
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"  Members              : {', '.join(m['id'] for m in MEMBERS)}")
    log(f"  Method               : equal-weight rank averaging within srch_id")
    log(f"  Submission path      : {sub_path}")
    log(f"  Submission copy      : {art_dir / 'submission.csv'}")
    log(f"  External archive     : {external_dir}")
    log(f"  Elapsed              : {elapsed_min:.1f} min")
    log("=" * 70)
    log(f"\n*** SUBMISSION FILE: {sub_path} ***")


if __name__ == "__main__":
    main()
