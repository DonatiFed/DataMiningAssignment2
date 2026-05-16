"""
V5 TE-ablation — test the diagnostic finding that V5 underperformed on Kaggle
because the top-drift target-encoding features shifted between train and test.

Setup is identical to pipelines/v5_ensemble_submit.py with ONE difference:
the 4 top-drift TE features are removed from train and test feature matrices
before any member trains.

Members (unchanged):  B3, F13, D4, A1, B10, E4, E3
Method (unchanged):   equal-weight rank averaging within srch_id
Seeds / label gains / weighting / hyperparameters / best_iter: unchanged
Same V4-style fresh lgb.Dataset pattern (no .construct(), no free_raw_data)

Dropped features (top adversarial-gain TEs, AUC=1.0 diagnostic):
  - country_book_rate
  - site_book_rate
  - site_country_book_rate
  - cpair_book_rate

Baselines to beat:
  V4 ensemble Kaggle    = 0.42021
  V5 ensemble Kaggle    = 0.41943  (must beat to confirm hypothesis)
  V5 ensemble local val = 0.42633

Outputs:
  artifacts/v5_te_ablation_<TS>/
    ├── ablation_config.json
    ├── feature_cols_<member>.json   (per-member feature lists post-drop)
    ├── importance_<member>.csv
    ├── test_pred_<member>.npy
    ├── ensemble_test_rank_avg.npy
    ├── submission.csv
    └── run_summary.md
  models/v5_te_ablation_<TS>/model_<member>.txt
  submissions/submission_v5_te_ablation_<TS>.csv
  /home/ubuntu/experiment_artifacts/v5_te_ablation_<TS>/  (mirror)
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
from src.features import build_features, compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402
from src.artifacts import run_dirs, save_run_config, save_git_commit  # noqa: E402

# ============================================================================
# Constants
# ============================================================================
EXTERNAL_COPY_ROOT = Path("/home/ubuntu/experiment_artifacts")
FEATURE_AUDIT_CSV = ROOT / "experiment_logs" / "feature_audit.csv"

# Baselines from prior work
V4_ENSEMBLE_LOCAL = 0.42512
V4_ENSEMBLE_KAGGLE = 0.42021
V5_ENSEMBLE_LOCAL = 0.42633
V5_ENSEMBLE_KAGGLE = 0.41943

# Top-drift TE features from adversarial diagnostic (AUC=1.0, accounted for >90% of drift gain)
DROPPED_FEATURES = [
    "country_book_rate",
    "site_book_rate",
    "site_country_book_rate",
    "cpair_book_rate",
]

# Base params (V4 backbone) — identical to v5_ensemble_submit
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

# Weighting profiles — identical to v5_ensemble_submit
WEIGHTING_PROFILES = {
    "ipw_default":  {"base": "ipw", "clip": (0.1, 10.0)},
    "no_ipw":       {"base": "none"},
    "ipw_clip3":    {"base": "ipw", "clip": (0.1, 3.0)},
}

# Members — identical to v5_ensemble_submit (same configs, same best_iter)
MEMBERS = [
    {
        "id": "B3", "label_gain": "0,2,15", "weighting": "ipw_clip3",
        "feature_filter": None, "row_filter": None, "param_overrides": {},
        "best_iter": 326, "val_ndcg5": 0.42396,
        "description": "lg=0,2,15, w=ipw_clip3",
    },
    {
        "id": "F13", "label_gain": "0,3,15", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": "positive_q_only", "param_overrides": {},
        "best_iter": 500, "val_ndcg5": 0.42253,
        "description": "positive queries only, lg=0,3,15",
    },
    {
        "id": "D4", "label_gain": "0,2,15", "weighting": "ipw_default",
        "feature_filter": "top_120", "row_filter": None, "param_overrides": {},
        "best_iter": 523, "val_ndcg5": 0.42228,
        "description": "feature_filter=top_120, lg=0,2,15",
    },
    {
        "id": "A1", "label_gain": "0,2,12", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": None, "param_overrides": {},
        "best_iter": 385, "val_ndcg5": 0.42205,
        "description": "label_gain=0,2,12",
    },
    {
        "id": "B10", "label_gain": "0,2,20", "weighting": "no_ipw",
        "feature_filter": None, "row_filter": None, "param_overrides": {},
        "best_iter": 358, "val_ndcg5": 0.42198,
        "description": "lg=0,2,20, w=no_ipw",
    },
    {
        "id": "E4", "label_gain": "0,2,15", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": None,
        "param_overrides": {"num_leaves": 400, "min_child_samples": 100},
        "best_iter": 518, "val_ndcg5": 0.42291,
        "description": "num_leaves=400, min_child=100",
    },
    {
        "id": "E3", "label_gain": "0,2,15", "weighting": "ipw_default",
        "feature_filter": None, "row_filter": None,
        "param_overrides": {"num_leaves": 512, "min_child_samples": 100, "reg_lambda": 2.0},
        "best_iter": 551, "val_ndcg5": 0.42185,
        "description": "num_leaves=512, min_child=100, reg_lambda=2.0",
    },
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def apply_feature_filter(feature_cols, filter_name, audit_df):
    if filter_name is None:
        return list(feature_cols)
    if filter_name.startswith("top_"):
        n = int(filter_name.split("_")[1])
        keep = set(audit_df.head(n)["feature"].tolist())
        return [c for c in feature_cols if c in keep]
    raise ValueError(f"Unsupported feature_filter: {filter_name}")


def apply_row_filter(train_split, filter_name):
    n = len(train_split)
    if filter_name is None:
        return np.ones(n, dtype=bool)
    if filter_name == "positive_q_only":
        eng = train_split.groupby("srch_id")["click_bool"].transform("sum")
        return (eng.values > 0)
    raise ValueError(f"Unsupported row_filter: {filter_name}")


def compute_weights(train_split, propensity, mode):
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
    nonrand = (train_split["random_bool"].values == 0)
    w = np.where(nonrand, pos_w, 1.0).astype(np.float32)
    clip_lo, clip_hi = profile.get("clip", (0.1, 10.0))
    return np.clip(w, clip_lo, clip_hi).astype(np.float32)


def make_group_counts(df):
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df), "group counts mismatch"
    return g


def ranks_within_srch(scores, srch_id):
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


def drop_features(feature_cols):
    """Remove DROPPED_FEATURES from a feature list. Returns (new_list, n_dropped)."""
    drop_set = set(DROPPED_FEATURES)
    new_cols = [c for c in feature_cols if c not in drop_set]
    return new_cols, len(feature_cols) - len(new_cols)


# ============================================================================
def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"v5_te_ablation_{ts}"
    log(f"=== V5 TE-ablation submission — run_id={run_id} ===")
    log(f"Members: {[m['id'] for m in MEMBERS]}")
    log(f"Dropped features ({len(DROPPED_FEATURES)}): {DROPPED_FEATURES}")

    model_dir, art_dir = run_dirs(run_id)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_git_commit(run_id)

    ablation_cfg = {
        "phase": "V5_TE_ABLATION",
        "run_id": run_id,
        "parent": "v5_ensemble_submit (Kaggle 0.41943)",
        "date": datetime.now(timezone.utc).isoformat(),
        "method": "equal_weight_rank_average_within_srch_id",
        "members": MEMBERS,
        "n_members": len(MEMBERS),
        "dropped_features": DROPPED_FEATURES,
        "drop_rationale": (
            "Adversarial validation (diagnose_v5_gap.py --quick) found AUC=1.0 "
            "between train and test feature matrices, with the 4 features above "
            "accounting for >90% of the adversarial gain. These are single-key/cross-key "
            "TE features where train uses k-fold OOF means and test uses full-train "
            "as source, producing shifted value distributions."
        ),
        "baselines": {
            "v4_ensemble_kaggle": V4_ENSEMBLE_KAGGLE,
            "v5_ensemble_kaggle": V5_ENSEMBLE_KAGGLE,
            "v5_ensemble_local": V5_ENSEMBLE_LOCAL,
        },
    }
    save_run_config(run_id, ablation_cfg)
    json.dump(ablation_cfg, open(art_dir / "ablation_config.json", "w"), indent=2, default=str)

    # ---- Load audit (top_120 needs it) ----
    audit_df = pd.read_csv(FEATURE_AUDIT_CSV)
    audit_df = audit_df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    log(f"feature_audit.csv loaded: {len(audit_df)} features")

    # ---- Load + featurize full train ----
    log("Loading full train…")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(train_raw):,} rows, {train_raw['srch_id'].nunique():,} searches")

    log("Computing propensity (full train)…")
    propensity = compute_position_propensity(train_raw)
    log(f"  propensity max={propensity.max():.4f}")

    log("Building features on full train (agg_source=train_raw, is_train=True)…")
    t1 = time.time()
    full_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    feature_cols_all = get_feature_columns(full_feat)
    leaked = set(feature_cols_all) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden cols leaked: {leaked}"
    log(f"  features built: {len(feature_cols_all)} cols | {time.time()-t1:.0f}s")

    # ---- Drop the 4 ablation features (train side) ----
    missing = [c for c in DROPPED_FEATURES if c not in full_feat.columns]
    if missing:
        raise RuntimeError(f"Expected drop features not present in features.py output: {missing}")
    feature_cols_all_post, n_dropped = drop_features(feature_cols_all)
    log(f"  DROPPED {n_dropped} features → {len(feature_cols_all_post)} remain (from {len(feature_cols_all)})")
    assert n_dropped == len(DROPPED_FEATURES), f"expected to drop {len(DROPPED_FEATURES)}, dropped {n_dropped}"

    remap = {0: 0, 1: 1, 5: 2}
    full_label_all = full_feat["relevance"].map(remap).astype(np.int32)
    assert (full_feat["srch_id"].values == train_raw["srch_id"].values).all(), "row order mismatch"

    # ---- Train per member ----
    train_models = {}
    for member in MEMBERS:
        mid = member["id"]
        log(f"\n--- Training member {mid} ({member['description']}) ---")

        feat_cols = apply_feature_filter(feature_cols_all_post, member["feature_filter"], audit_df)
        feat_cols = [c for c in feat_cols if c in full_feat.columns]
        log(f"  features: {len(feat_cols)} (post-ablation)")
        json.dump(list(feat_cols), open(art_dir / f"feature_cols_{mid}.json", "w"), indent=2)

        row_mask = apply_row_filter(train_raw, member["row_filter"])
        if not row_mask.all():
            full_feat_m = full_feat.loc[row_mask].reset_index(drop=True)
            train_raw_m = train_raw.loc[row_mask].reset_index(drop=True)
            label_m = full_label_all.loc[row_mask].reset_index(drop=True).values
            log(f"  row filter '{member['row_filter']}': {len(full_feat_m):,}/{len(full_feat):,} rows")
        else:
            full_feat_m = full_feat
            train_raw_m = train_raw
            label_m = full_label_all.values

        weights = compute_weights(train_raw_m, propensity, member["weighting"])
        log(f"  weights: [{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")

        groups = make_group_counts(full_feat_m)

        params = BASE_PARAMS.copy()
        if member["label_gain"]:
            params["label_gain"] = member["label_gain"]
        params.update(member["param_overrides"])

        nbr = int(member["best_iter"])
        log(f"  num_boost_round = {nbr}")

        ds = lgb.Dataset(
            full_feat_m[feat_cols], label=label_m, group=groups, weight=weights,
        )

        t_train = time.time()
        model = lgb.train(
            params, ds, num_boost_round=nbr,
            callbacks=[lgb.log_evaluation(max(nbr // 5, 50))],
        )
        log(f"  trained {nbr} rounds in {(time.time()-t_train)/60:.1f} min")
        model.save_model(str(model_dir / f"model_{mid}.txt"))
        train_models[mid] = model

        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        imp_df.to_csv(art_dir / f"importance_{mid}.csv", index=False)

        del ds, full_feat_m, train_raw_m, label_m, weights, groups
        gc.collect()

    log("\nFreeing train features before test featurization…")
    del full_feat, full_label_all
    gc.collect()

    # ---- Test featurization + predictions ----
    log("\nLoading test set…")
    test_raw = load_test()
    test_raw = test_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(test_raw):,} rows, {test_raw['srch_id'].nunique():,} searches")

    log("Building test features (agg_source=full train)…")
    t2 = time.time()
    test_feat = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"  features built: {len(test_feat):,} rows | {time.time()-t2:.0f}s")

    missing_test = [c for c in DROPPED_FEATURES if c not in test_feat.columns]
    if missing_test:
        log(f"  ! ablation features not in test_feat (already absent?): {missing_test}")
    else:
        log(f"  ablation features confirmed present in test_feat — they will be excluded by per-member feature_cols.")

    del train_raw
    gc.collect()

    test_srch_ids = test_feat["srch_id"].values
    test_prop_ids = test_feat["prop_id"].values
    n_test = len(test_feat)
    rank_sum = np.zeros(n_test, dtype=np.float64)

    for member in MEMBERS:
        mid = member["id"]
        log(f"\n--- Predicting test with {mid} ---")
        feat_cols = json.load(open(art_dir / f"feature_cols_{mid}.json"))
        for c in feat_cols:
            if c not in test_feat.columns:
                test_feat[c] = np.nan
        model = train_models[mid]
        t_pred = time.time()
        preds = model.predict(test_feat[feat_cols]).astype(np.float32)
        log(f"  predicted in {time.time()-t_pred:.0f}s  range=[{preds.min():.3f}, {preds.max():.3f}]")
        np.save(art_dir / f"test_pred_{mid}.npy", preds)

        ranks = ranks_within_srch(preds, test_srch_ids)
        rank_sum += ranks

    rank_avg = rank_sum / len(MEMBERS)
    np.save(art_dir / "ensemble_test_rank_avg.npy", rank_avg.astype(np.float32))

    # ---- Submission ----
    log("\nGenerating submission CSV…")
    sub_df = pd.DataFrame({
        "srch_id": test_srch_ids,
        "prop_id": test_prop_ids,
        "rank_avg": rank_avg,
    }).sort_values(["srch_id", "rank_avg"], ascending=[True, True]).reset_index(drop=True)
    submission = sub_df[["srch_id", "prop_id"]]

    sub_name = f"submission_v5_te_ablation_{ts}.csv"
    sub_path = SUBMISSIONS_DIR / sub_name
    submission.to_csv(sub_path, index=False)
    submission.to_csv(art_dir / "submission.csv", index=False)
    log(f"  submission rows: {len(submission):,}  unique searches: {submission['srch_id'].nunique():,}")
    log(f"  saved: {sub_path}")

    # ---- Run summary ----
    elapsed_min = (time.time() - t0) / 60
    summary_md = f"""# V5 TE-ablation — {run_id}

_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

## Hypothesis
V5 lost on Kaggle (0.41943 vs V4 0.42021, Δ=−0.00078) because high-drift single-key
target-encoding features (`country_book_rate`, `site_book_rate`, `site_country_book_rate`,
`cpair_book_rate`) shifted distributions between train and test (adversarial AUC=1.0).

## Setup

| | Value |
|---|---|
| Members | {', '.join(m['id'] for m in MEMBERS)} |
| Method | equal-weight rank averaging within srch_id |
| Dropped features | `{', '.join(DROPPED_FEATURES)}` |
| Feature count (before) | {len(feature_cols_all)} |
| Feature count (after, full set) | {len(feature_cols_all_post)} |
| D4 feature count (top_120 ∩ post-drop) | {len(json.load(open(art_dir / 'feature_cols_D4.json')))} |
| seeds / label_gains / weighting / params / best_iter | identical to v5_ensemble_submit |

## Baselines

| Reference | Kaggle public | Local val |
|---|---|---|
| V4 ensemble | {V4_ENSEMBLE_KAGGLE:.5f} | {V4_ENSEMBLE_LOCAL:.5f} |
| V5 ensemble | {V5_ENSEMBLE_KAGGLE:.5f} | {V5_ENSEMBLE_LOCAL:.5f} |
| **V5 TE-ablation (this run)** | _pending Kaggle upload_ | _not computed (no val split)_ |

## Outputs

- Submission CSV: `{sub_path}`
- Submission copy: `{art_dir / 'submission.csv'}`
- Submission rows: {len(submission):,} · unique srch_ids: {submission['srch_id'].nunique():,}
- Models: `models/{run_id}/model_<id>.txt`
- Per-member test predictions: `{art_dir / 'test_pred_<id>.npy'}`
- Final rank average: `{art_dir / 'ensemble_test_rank_avg.npy'}`
- Ablation config: `{art_dir / 'ablation_config.json'}`

## Interpretation rubric

| New Kaggle | Δ vs V5 (0.41943) | Δ vs V4 (0.42021) | Conclusion |
|---|---|---|---|
| ≥ 0.4210 | ≥ +0.0016 | ≥ +0.0008 | **TE drift confirmed**. Beats V4. Promote ablation to v6 base; sweep TE smoothing next. |
| 0.4200–0.4210 | +0.0006 to +0.0016 | −0.0002 to +0.0008 | TE drift partially confirmed. Closes most of the gap. Worth more TE work. |
| 0.4194–0.4200 | 0 to +0.0006 | small loss | Marginal — TE drift is one of multiple factors. Run `--temporal` next. |
| < 0.4194 | < 0 | unchanged | TE drift NOT the dominant factor. Don't blame TEs; revisit `--temporal` + position-bias actions. |

## Wall-clock

Total: {elapsed_min:.1f} min
"""
    (art_dir / "run_summary.md").write_text(summary_md)
    log(f"  saved: {art_dir / 'run_summary.md'}")

    # ---- External archive copy ----
    external_dir = EXTERNAL_COPY_ROOT / run_id
    log(f"\nCopying artifacts to {external_dir} …")
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

    # ---- Final ----
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"  Members             : {', '.join(m['id'] for m in MEMBERS)}")
    log(f"  Dropped features    : {DROPPED_FEATURES}")
    log(f"  Method              : equal-weight rank averaging within srch_id")
    log(f"  Submission path     : {sub_path}")
    log(f"  Submission copy     : {art_dir / 'submission.csv'}")
    log(f"  External archive    : {external_dir}")
    log(f"  Elapsed             : {elapsed_min:.1f} min")
    log("=" * 70)
    log(f"\n*** SUBMISSION FILE: {sub_path} ***")


if __name__ == "__main__":
    main()
