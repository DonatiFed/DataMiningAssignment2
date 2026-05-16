"""V6 — temporal-clean ensemble training and evaluation.

Trains 10 diverse members on temporal_train (with reuse of already-trained
boosters where available), evaluates each on temporal_val, builds rank-average
ensembles, runs leave-one-out on the best ensemble, and conditionally produces
a Kaggle submission if the best ensemble beats V4+CP+DS = 0.40679 by ≥ +0.0003.

Safety:
- Per-member try/except — one training failure does not kill the batch.
- Incremental writes: member_results.csv is rewritten after each completed member.
- Resumable: if a model file already exists in MODELS_OUT, training is skipped
  and the saved model is loaded instead.
- Phase isolation: temporal-validation results are persisted before any
  submission step is attempted.

NO new features beyond CP and DS. CP and DS are SEPARATE ensemble members
(never combined inside a single model). Hard-negative and random-val selection
are explicitly OFF.
"""
from __future__ import annotations
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_train, make_target, get_feature_columns  # noqa: E402
from src.features import (  # noqa: E402
    build_features,
    compute_position_propensity,
    FORBIDDEN_FEATURES,
)
from pipelines.temporal_validation import (  # noqa: E402
    temporal_split,
    compute_ipw_weights,
    make_group_counts,
    eval_metrics,
    BASE_PARAMS,
)
from pipelines.evaluate_variant import (  # noqa: E402
    _pos_adj_oof_te,
    _prop_dest_book_rate_safe,
)

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
V4_ANCHOR_TEMPORAL = 0.40401
QUICK_ENSEMBLE_BENCHMARK = 0.40679
SUBMISSION_THRESHOLD = QUICK_ENSEMBLE_BENCHMARK + 0.0003  # 0.40709

OUT = ROOT / "diagnostics" / f"v6_{TIMESTAMP}"
MODELS_OUT = ROOT / "models" / f"v6_{TIMESTAMP}"
# Dir creation moved into main() so importing this module does not create
# empty stub dirs as a side effect.

CACHE_TRAIN = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_train.parquet"
CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"

# Reusable models from prior sessions (already trained on temporal_train).
REUSE_MODELS = {
    "lambdarank_bal15": ROOT / "diagnostics" / "temporal_validation_20260516_113831" / "model_V4_ANCHOR_temporal.txt",
    "CP": ROOT / "diagnostics" / "eval_variants" / "model_prop_click_rate_pos_adj_s40_oof_temporal.txt",
    "DS": ROOT / "diagnostics" / "eval_variants" / "model_prop_dest_book_rate_safe_temporal.txt",
}

# ============================================================================
# Member specifications
# ============================================================================
# Each spec is self-describing; the trainer dispatches on `type`.
# weight: "ipw" (V4 default), "none" (uniform), or "randup" (random_bool×2 else 1).
# extra_feature: "CP" / "DS" / None — added to feature set if not None.
MEMBERS = [
    {"id": "lambdarank_base",   "type": "lambdarank", "label_gain": "0,1,31", "weight": "ipw",    "extra_feature": None},
    {"id": "lambdarank_click3", "type": "lambdarank", "label_gain": "0,3,31", "weight": "ipw",    "extra_feature": None},
    {"id": "lambdarank_bal15",  "type": "lambdarank", "label_gain": "0,1,15", "weight": "ipw",    "extra_feature": None},  # = V4_ANCHOR
    {"id": "lambdarank_book50", "type": "lambdarank", "label_gain": "0,1,50", "weight": "ipw",    "extra_feature": None},
    {"id": "lambdarank_noipw",  "type": "lambdarank", "label_gain": "0,1,15", "weight": "none",   "extra_feature": None},
    {"id": "rank_xendcg",       "type": "xendcg",     "label_gain": "0,1,15", "weight": "ipw",    "extra_feature": None},
    {"id": "lambdarank_randup", "type": "lambdarank", "label_gain": "0,2,25", "weight": "randup", "extra_feature": None},
    {"id": "booking_clf",       "type": "binary",     "label_gain": None,     "weight": "none",   "extra_feature": None},
    {"id": "CP",                "type": "lambdarank", "label_gain": "0,1,15", "weight": "ipw",    "extra_feature": "CP"},
    {"id": "DS",                "type": "lambdarank", "label_gain": "0,1,15", "weight": "ipw",    "extra_feature": "DS"},
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic CSV write: write to tmp, rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# ============================================================================
# Phase 0: load features and build extra (CP, DS) columns
# ============================================================================
def load_features():
    log("=" * 70)
    log("PHASE 0: loading cached temporal_train / temporal_val features")
    log(f"  train: {CACHE_TRAIN}")
    log(f"  val:   {CACHE_VAL}")
    assert CACHE_TRAIN.exists() and CACHE_VAL.exists(), "feature cache missing"
    train_feat = pd.read_parquet(CACHE_TRAIN)
    val_feat = pd.read_parquet(CACHE_VAL)
    train_feat = train_feat.sort_values("srch_id").reset_index(drop=True)
    val_feat = val_feat.sort_values("srch_id").reset_index(drop=True)
    log(f"  train rows={len(train_feat):,}  val rows={len(val_feat):,}  "
        f"base cols={train_feat.shape[1]}")
    return train_feat, val_feat


def add_extra_features(train_feat: pd.DataFrame, val_feat: pd.DataFrame) -> None:
    """Add CP and DS feature columns to train_feat and val_feat (in-place)."""
    log("Building 'prop_click_rate_pos_adj_s40_oof' (CP) on train + val…")
    _pos_adj_oof_te(
        train_feat, val_feat,
        target_col="click_bool",
        col_new="prop_click_rate_pos_adj_s40_oof",
        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42,
    )
    log("Building 'prop_dest_book_rate_safe' (DS) on train + val…")
    _prop_dest_book_rate_safe(
        train_feat, val_feat,
        col_new="prop_dest_book_rate_safe",
        alpha=40.0, n_folds=5, seed=42,
    )
    log(f"  feature counts now: train={train_feat.shape[1]}  val={val_feat.shape[1]}")


# ============================================================================
# Phase 1: train (or load) each member and predict on val
# ============================================================================
def make_label_remap(s: pd.Series) -> np.ndarray:
    remap = {0: 0, 1: 1, 5: 2}
    return s.map(remap).astype(np.int32).values


def compute_weights(spec, temporal_train_df, propensity):
    if spec["weight"] == "none":
        return None
    if spec["weight"] == "ipw":
        w = compute_ipw_weights(temporal_train_df, propensity, clip_hi=10.0, clip_lo=0.1)
        return w
    if spec["weight"] == "randup":
        rb = temporal_train_df["random_bool"].values
        return np.where(rb == 1, 2.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown weight scheme: {spec['weight']}")


def feature_columns_for(spec, train_feat) -> list[str]:
    cols = get_feature_columns(train_feat)
    leaked = set(cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden cols leaked: {leaked}"
    if spec["extra_feature"] == "CP":
        # CP member uses base + the OOF column
        cols = [c for c in cols if c == "prop_click_rate_pos_adj_s40_oof" or c != "prop_dest_book_rate_safe"]
    elif spec["extra_feature"] == "DS":
        cols = [c for c in cols if c == "prop_dest_book_rate_safe" or c != "prop_click_rate_pos_adj_s40_oof"]
    else:
        cols = [c for c in cols if c not in ("prop_click_rate_pos_adj_s40_oof", "prop_dest_book_rate_safe")]
    return cols


def train_one(spec, train_feat, val_feat, temporal_train_df, propensity):
    """Train one member. Returns (booster, best_iter, feat_cols)."""
    feat_cols = feature_columns_for(spec, train_feat)
    weights = compute_weights(spec, temporal_train_df, propensity)
    log(f"  features used: {len(feat_cols)}  (extra={spec.get('extra_feature')})")
    if weights is not None:
        log(f"  sample weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

    params = BASE_PARAMS.copy()
    if spec["type"] == "lambdarank":
        params["objective"] = "lambdarank"
        params["label_gain"] = spec["label_gain"]
        params["metric"] = "ndcg"
        train_label = make_label_remap(train_feat["relevance"])
        val_label = make_label_remap(val_feat["relevance"])
        train_groups = make_group_counts(train_feat)
        val_groups = make_group_counts(val_feat)
        ds_tr = lgb.Dataset(train_feat[feat_cols], label=train_label,
                            group=train_groups, weight=weights)
        ds_va = lgb.Dataset(val_feat[feat_cols], label=val_label,
                            group=val_groups, reference=ds_tr)
        model = lgb.train(
            params, ds_tr, num_boost_round=2000,
            valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
        )

    elif spec["type"] == "xendcg":
        params["objective"] = "rank_xendcg"
        params["label_gain"] = spec["label_gain"]
        params["metric"] = "ndcg"
        train_label = make_label_remap(train_feat["relevance"])
        val_label = make_label_remap(val_feat["relevance"])
        train_groups = make_group_counts(train_feat)
        val_groups = make_group_counts(val_feat)
        ds_tr = lgb.Dataset(train_feat[feat_cols], label=train_label,
                            group=train_groups, weight=weights)
        ds_va = lgb.Dataset(val_feat[feat_cols], label=val_label,
                            group=val_groups, reference=ds_tr)
        model = lgb.train(
            params, ds_tr, num_boost_round=2000,
            valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
        )

    elif spec["type"] == "binary":
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
        params.pop("eval_at", None)
        train_label = train_feat["booking_bool"].values.astype(np.int32)
        val_label = val_feat["booking_bool"].values.astype(np.int32)
        ds_tr = lgb.Dataset(train_feat[feat_cols], label=train_label, weight=weights)
        ds_va = lgb.Dataset(val_feat[feat_cols], label=val_label, reference=ds_tr)
        model = lgb.train(
            params, ds_tr, num_boost_round=2000,
            valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
        )
    else:
        raise ValueError(f"unknown type: {spec['type']}")

    return model, int(model.best_iteration), feat_cols


def predict_member(spec, model, val_feat):
    feat_cols = model.feature_name()
    missing = [c for c in feat_cols if c not in val_feat.columns]
    assert not missing, f"val missing features: {missing[:5]}"
    scores = model.predict(val_feat[feat_cols]).astype(np.float32)
    return scores


def evaluate_member(scores, val_feat):
    return eval_metrics(val_feat, scores)


def save_importance(model, member_id):
    if not hasattr(model, "feature_importance"):
        return
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    safe_write_csv(imp, OUT / f"importance_{member_id}.csv")


# ============================================================================
# Phase 2: ensembles
# ============================================================================
def _grouped_rank(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Average rank within each srch_id, descending by score (1 = best)."""
    s = pd.Series(scores)
    g = s.groupby(pd.Series(srch_id), sort=False)
    return g.rank(method="average", ascending=False).values.astype(np.float32)


def metrics_from_avg_rank(val_feat, avg_rank: np.ndarray) -> dict:
    """Score using inverse rank as a higher-is-better signal."""
    inv = -avg_rank  # so eval_metrics can sort by descending
    return eval_metrics(val_feat, inv)


def build_ensembles(member_ranks: dict[str, np.ndarray], val_feat, available):
    """Build the 6 named ensembles + LOO afterwards."""
    v4_style_ids = [
        "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
        "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
        "lambdarank_randup", "booking_clf",
    ]
    v4_available = [i for i in v4_style_ids if i in available]
    cfgs = {
        "v4_only":           v4_available,
        "v4_plus_CP":        v4_available + (["CP"] if "CP" in available else []),
        "v4_plus_DS":        v4_available + (["DS"] if "DS" in available else []),
        "v4_plus_CP_plus_DS": v4_available + [m for m in ("CP", "DS") if m in available],
        "CP_plus_DS":        [m for m in ("CP", "DS") if m in available],
    }
    return cfgs


def score_ensemble(cfg_members: list[str], member_ranks: dict, val_feat) -> dict:
    if not cfg_members:
        return {"ndcg5": float("nan"), "n_members": 0}
    avg = np.mean([member_ranks[m] for m in cfg_members], axis=0)
    m = metrics_from_avg_rank(val_feat, avg)
    return {**m, "n_members": len(cfg_members), "members": ",".join(cfg_members)}


# ============================================================================
# Submission helpers (used only if temporal ensemble beats threshold)
# ============================================================================
def build_full_train_features():
    """Load full train + test and build features with full-train as agg_source."""
    log("Loading full train + test for submission feature build…")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  train rows={len(train_raw):,}")
    from src.data_loader import load_test
    test_raw = load_test()
    log(f"  test  rows={len(test_raw):,}")

    log("Building full-train features (agg_source = full train)…")
    train_full = build_features(train_raw, agg_source=train_raw, is_train=True)
    log("Building test features (agg_source = full train, is_train=False)…")
    test_full = build_features(test_raw, agg_source=train_raw, is_train=False)
    return train_full, test_full


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    log(f"V6 gym batch — {TIMESTAMP}")
    log(f"out:        {OUT}")
    log(f"models_out: {MODELS_OUT}")
    log(f"benchmark to beat: V4+CP+DS = {QUICK_ENSEMBLE_BENCHMARK:.5f}; "
        f"submission iff best ≥ {SUBMISSION_THRESHOLD:.5f}")

    # --- Persist config -----------------------------------------------------
    cfg_json = OUT / "run_config.json"
    cfg_json.write_text(json.dumps({
        "timestamp": TIMESTAMP,
        "members": MEMBERS,
        "v4_anchor_temporal": V4_ANCHOR_TEMPORAL,
        "quick_ensemble_benchmark": QUICK_ENSEMBLE_BENCHMARK,
        "submission_threshold": SUBMISSION_THRESHOLD,
        "reuse_models": {k: str(v) for k, v in REUSE_MODELS.items()},
    }, indent=2))

    # --- Phase 0: features --------------------------------------------------
    train_feat, val_feat = load_features()
    add_extra_features(train_feat, val_feat)

    # IPW propensity computed once
    log("Computing position propensity for IPW (from train)…")
    propensity = compute_position_propensity(train_feat)
    temporal_train_df = train_feat  # alias; build_features kept all rows

    # --- Phase 1: train each member ----------------------------------------
    log("=" * 70)
    log("PHASE 1: training/loading 10 members on temporal_train")

    member_rows = []
    member_preds: dict[str, np.ndarray] = {}
    member_results_path = OUT / "member_results.csv"

    for i, spec in enumerate(MEMBERS, 1):
        member_id = spec["id"]
        model_path = MODELS_OUT / f"model_{member_id}.txt"
        log(f"\n--- [{i}/{len(MEMBERS)}] member: {member_id} ---")
        log(f"  type={spec['type']}  lg={spec['label_gain']}  weight={spec['weight']}  "
            f"extra={spec.get('extra_feature')}")

        row = {"member_id": member_id, "type": spec["type"],
               "label_gain": spec["label_gain"], "weight": spec["weight"],
               "extra_feature": spec.get("extra_feature") or "",
               "status": "pending"}

        try:
            t_member = time.time()

            # ---- Load if already exists (resume) ----
            if model_path.exists():
                log(f"  resume: found {model_path.name}, loading instead of training")
                model = lgb.Booster(model_file=str(model_path))
                row["best_iter"] = int(getattr(model, "best_iteration", -1) or model.current_iteration())
            elif member_id in REUSE_MODELS and REUSE_MODELS[member_id].exists():
                log(f"  reuse: loading prior temporal-clean model from "
                    f"{REUSE_MODELS[member_id]}")
                model = lgb.Booster(model_file=str(REUSE_MODELS[member_id]))
                # Persist copy in this batch's models dir for self-contained reproducibility.
                model.save_model(str(model_path))
                row["best_iter"] = int(getattr(model, "best_iteration", -1) or model.current_iteration())
            else:
                model, best_iter, feat_cols = train_one(
                    spec, train_feat, val_feat, temporal_train_df, propensity
                )
                model.save_model(str(model_path))
                row["best_iter"] = best_iter
                log(f"  trained best_iter={best_iter} in {(time.time()-t_member)/60:.1f} min")

            # ---- Predict on val ----
            scores = predict_member(spec, model, val_feat)
            np.save(OUT / f"val_pred_{member_id}.npy", scores)
            m = evaluate_member(scores, val_feat)
            row.update({
                "n_features": len(model.feature_name()),
                "ndcg5": float(m["ndcg5"]),
                "recall1": float(m["recall1"]),
                "recall5": float(m["recall5"]),
                "mean_booked_rank": float(m["mean_booked_rank"]),
                "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
                "status": "ok",
                "model_path": str(model_path),
                "elapsed_min": round((time.time() - t_member) / 60, 2),
            })
            member_preds[member_id] = scores
            log(f"  ✓ NDCG@5={m['ndcg5']:.5f}  Δ={float(m['ndcg5']) - V4_ANCHOR_TEMPORAL:+.5f}  "
                f"R@5={m['recall5']:.4f}  MBR={m['mean_booked_rank']:.3f}")

            # ---- Save importance ----
            try:
                save_importance(model, member_id)
            except Exception as e:
                log(f"  ! could not save importance: {e}")

            # free model to save RAM
            del model
            gc.collect()

        except Exception as e:
            log(f"  ✗ FAILED: {e}")
            log(traceback.format_exc())
            row["status"] = f"failed:{type(e).__name__}"
            row["error"] = str(e)[:300]

        member_rows.append(row)
        # incremental save
        try:
            safe_write_csv(pd.DataFrame(member_rows), member_results_path)
        except Exception as e:
            log(f"  ! could not save member_results.csv: {e}")

    log(f"\nPhase 1 done in {(time.time()-t0)/60:.1f} min total")

    available = sorted(member_preds.keys())
    log(f"successful members: {available}")
    if not available:
        log("✗ NO MEMBERS SUCCEEDED — aborting before ensembles.")
        return

    # --- Phase 2: ensembles ------------------------------------------------
    log("=" * 70)
    log("PHASE 2: rank-average ensembles")

    member_ranks: dict[str, np.ndarray] = {}
    srch_id_arr = val_feat["srch_id"].values
    for m_id, scores in member_preds.items():
        member_ranks[m_id] = _grouped_rank(srch_id_arr, scores)

    cfgs = build_ensembles(member_ranks, val_feat, available)
    ens_rows = []
    for cfg_name, members in cfgs.items():
        res = score_ensemble(members, member_ranks, val_feat)
        res.update({
            "config": cfg_name,
            "delta_vs_v4_anchor": res.get("ndcg5", float("nan")) - V4_ANCHOR_TEMPORAL,
            "delta_vs_quick": res.get("ndcg5", float("nan")) - QUICK_ENSEMBLE_BENCHMARK,
        })
        ens_rows.append(res)
        log(f"  {cfg_name:24s} n={res['n_members']:2d}  "
            f"NDCG@5={res.get('ndcg5', float('nan')):.5f}  "
            f"Δ_anchor={res.get('delta_vs_v4_anchor', float('nan')):+.5f}  "
            f"Δ_quick={res.get('delta_vs_quick', float('nan')):+.5f}")

    # above-median selection (V4-style): take members with NDCG above median
    members_df = pd.DataFrame(member_rows)
    ok_df = members_df[members_df["status"] == "ok"].copy()
    median_score = ok_df["ndcg5"].median()
    above_median = ok_df[ok_df["ndcg5"] > median_score]["member_id"].tolist()
    am_res = score_ensemble(above_median, member_ranks, val_feat)
    am_res.update({
        "config": "above_median",
        "members": ",".join(above_median),
        "delta_vs_v4_anchor": am_res.get("ndcg5", float("nan")) - V4_ANCHOR_TEMPORAL,
        "delta_vs_quick": am_res.get("ndcg5", float("nan")) - QUICK_ENSEMBLE_BENCHMARK,
    })
    ens_rows.append(am_res)
    log(f"  {'above_median':24s} n={am_res['n_members']:2d}  "
        f"NDCG@5={am_res.get('ndcg5', float('nan')):.5f}  "
        f"Δ_anchor={am_res.get('delta_vs_v4_anchor', float('nan')):+.5f}  "
        f"Δ_quick={am_res.get('delta_vs_quick', float('nan')):+.5f}  "
        f"(median={median_score:.5f}, above={len(above_median)})")

    ensemble_df = pd.DataFrame(ens_rows).sort_values("ndcg5", ascending=False).reset_index(drop=True)
    safe_write_csv(ensemble_df, OUT / "ensemble_results.csv")
    log("Saved ensemble_results.csv")

    # --- Leave-one-out on the best ensemble --------------------------------
    log("\n--- Leave-one-out on the best ensemble ---")
    best_row = ensemble_df.iloc[0]
    best_members = best_row["members"].split(",")
    best_score = float(best_row["ndcg5"])
    log(f"Best ensemble: {best_row['config']} with {len(best_members)} members, "
        f"NDCG@5={best_score:.5f}")

    loo_rows = []
    for drop in best_members:
        remaining = [m for m in best_members if m != drop]
        res = score_ensemble(remaining, member_ranks, val_feat)
        loo_rows.append({
            "dropped": drop,
            "n_remaining": len(remaining),
            "ndcg5": res.get("ndcg5", float("nan")),
            "delta_vs_best": res.get("ndcg5", float("nan")) - best_score,
            "recall5": res.get("recall5", float("nan")),
            "mbr": res.get("mean_booked_rank", float("nan")),
        })
        log(f"  drop {drop:20s} → NDCG@5={res.get('ndcg5', float('nan')):.5f}  "
            f"Δ={res.get('ndcg5', float('nan')) - best_score:+.5f}")
    loo_df = pd.DataFrame(loo_rows).sort_values("delta_vs_best", ascending=True)
    safe_write_csv(loo_df, OUT / "leave_one_out.csv")

    # --- Phase 3 (conditional): submission ---------------------------------
    sub_status = "not_attempted"
    sub_score = float("nan")
    if best_score >= SUBMISSION_THRESHOLD:
        log("\n" + "=" * 70)
        log(f"PHASE 3: best ensemble ({best_score:.5f}) ≥ submission threshold "
            f"({SUBMISSION_THRESHOLD:.5f}). Building submission.")
        try:
            sub_score = build_submission(best_members, member_rows)
            sub_status = "ok"
        except Exception as e:
            log(f"  ✗ submission failed: {e}")
            log(traceback.format_exc())
            sub_status = f"failed:{type(e).__name__}"
    else:
        log(f"\nBest ensemble {best_score:.5f} < threshold "
            f"{SUBMISSION_THRESHOLD:.5f}. No submission produced.")
        sub_status = "below_threshold"

    # --- README ------------------------------------------------------------
    write_readme(member_rows, ensemble_df, loo_df, best_row, sub_status, sub_score)
    log(f"\n=== ALL DONE in {(time.time()-t0)/60:.1f} min ===")
    log(f"Outputs: {OUT}")


def build_submission(best_members: list[str], member_rows: list[dict]) -> float:
    """Retrain selected members on full train; predict on test; rank-average."""
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = sub_dir / f"submission_v6_{TIMESTAMP}.csv"
    sub_meta = OUT / "submission_meta.json"

    train_full, test_full = build_full_train_features()
    # Add CP / DS features on full train and test if any selected member uses them.
    needs_cp = "CP" in best_members
    needs_ds = "DS" in best_members
    if needs_cp:
        log("Building CP feature on full train + test…")
        _pos_adj_oof_te(
            train_full, test_full,
            target_col="click_bool",
            col_new="prop_click_rate_pos_adj_s40_oof",
            alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42,
        )
    if needs_ds:
        log("Building DS feature on full train + test…")
        _prop_dest_book_rate_safe(
            train_full, test_full,
            col_new="prop_dest_book_rate_safe",
            alpha=40.0, n_folds=5, seed=42,
        )

    propensity = compute_position_propensity(train_full)
    test_ranks: list[np.ndarray] = []
    for m_row in [r for r in member_rows if r["member_id"] in best_members]:
        spec = next(s for s in MEMBERS if s["id"] == m_row["member_id"])
        best_iter = int(m_row.get("best_iter", 500))
        log(f"  retraining {spec['id']} on full train ({best_iter} rounds)…")
        feat_cols = feature_columns_for(spec, train_full)
        weights = compute_weights(spec, train_full, propensity)

        params = BASE_PARAMS.copy()
        if spec["type"] == "lambdarank":
            params["objective"] = "lambdarank"
            params["label_gain"] = spec["label_gain"]
            params["metric"] = "ndcg"
            label = make_label_remap(train_full["relevance"])
            groups = make_group_counts(train_full)
            ds_tr = lgb.Dataset(train_full[feat_cols], label=label, group=groups, weight=weights)
            model = lgb.train(params, ds_tr, num_boost_round=best_iter,
                              callbacks=[lgb.log_evaluation(0)])
        elif spec["type"] == "xendcg":
            params["objective"] = "rank_xendcg"
            params["label_gain"] = spec["label_gain"]
            params["metric"] = "ndcg"
            label = make_label_remap(train_full["relevance"])
            groups = make_group_counts(train_full)
            ds_tr = lgb.Dataset(train_full[feat_cols], label=label, group=groups, weight=weights)
            model = lgb.train(params, ds_tr, num_boost_round=best_iter,
                              callbacks=[lgb.log_evaluation(0)])
        elif spec["type"] == "binary":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            params.pop("eval_at", None)
            label = train_full["booking_bool"].values.astype(np.int32)
            ds_tr = lgb.Dataset(train_full[feat_cols], label=label, weight=weights)
            model = lgb.train(params, ds_tr, num_boost_round=best_iter,
                              callbacks=[lgb.log_evaluation(0)])
        else:
            raise ValueError(f"unknown type: {spec['type']}")

        model.save_model(str(MODELS_OUT / f"model_{spec['id']}_FULL.txt"))
        test_scores = model.predict(test_full[feat_cols]).astype(np.float32)
        np.save(OUT / f"test_pred_{spec['id']}.npy", test_scores)
        rk = _grouped_rank(test_full["srch_id"].values, test_scores)
        test_ranks.append(rk)
        del model
        gc.collect()

    avg_rank = np.mean(test_ranks, axis=0)
    test_full["_rk"] = avg_rank
    # Kaggle Expedia: header MUST be `srch_id,prop_id` (lowercase, matching submission_sample.csv).
    sub = (test_full.sort_values(["srch_id", "_rk"])
           [["srch_id", "prop_id"]])
    sub.to_csv(sub_csv, index=False)
    log(f"  submission written: {sub_csv}")
    log(f"  rows={len(sub):,}  unique searches={sub['SearchId'].nunique():,}")

    sub_meta.write_text(json.dumps({
        "submission_path": str(sub_csv),
        "members": best_members,
        "timestamp": TIMESTAMP,
    }, indent=2))
    return float("nan")  # actual NDCG on test is unknown


def write_readme(member_rows, ensemble_df, loo_df, best_row, sub_status, sub_score):
    readme = OUT / "README.md"
    members_df = pd.DataFrame(member_rows)
    ok_df = members_df[members_df["status"] == "ok"].sort_values("ndcg5", ascending=False) if len(members_df) else members_df
    with readme.open("w") as f:
        f.write(f"# V6 gym batch — {TIMESTAMP}\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(f"- V4_ANCHOR temporal NDCG@5 baseline: **{V4_ANCHOR_TEMPORAL}**\n")
        f.write(f"- Quick ensemble V4+CP+DS benchmark: **{QUICK_ENSEMBLE_BENCHMARK}**\n")
        f.write(f"- Submission threshold (≥+0.0003 vs quick): "
                f"**{SUBMISSION_THRESHOLD:.5f}**\n\n")

        f.write("## Member results\n\n```\n")
        if len(ok_df):
            cols = ["member_id", "type", "label_gain", "weight", "extra_feature",
                    "n_features", "best_iter", "ndcg5", "recall5",
                    "mean_booked_rank", "delta_vs_v4_anchor", "elapsed_min"]
            cols = [c for c in cols if c in ok_df.columns]
            f.write(ok_df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        bad = members_df[members_df["status"] != "ok"]
        if len(bad):
            f.write("\n\n# Failed members:\n")
            f.write(bad[["member_id", "status", "error"]].to_string(index=False))
        f.write("\n```\n\n")

        f.write("## Ensemble results (rank-average)\n\n```\n")
        f.write(ensemble_df[[
            "config", "n_members", "ndcg5", "recall5", "mean_booked_rank",
            "delta_vs_v4_anchor", "delta_vs_quick", "members",
        ]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        f.write("\n```\n\n")

        f.write(f"## Best ensemble: **{best_row['config']}** "
                f"(NDCG@5={best_row['ndcg5']:.5f})\n\n")
        f.write(f"- Δ vs V4_ANCHOR  ({V4_ANCHOR_TEMPORAL}): "
                f"**{best_row['delta_vs_v4_anchor']:+.5f}**\n")
        f.write(f"- Δ vs V4+CP+DS quick ensemble ({QUICK_ENSEMBLE_BENCHMARK}): "
                f"**{best_row['delta_vs_quick']:+.5f}**\n")
        f.write(f"- Members: `{best_row['members']}`\n\n")

        f.write("## Leave-one-out (on best ensemble)\n\n```\n")
        f.write(loo_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        f.write("\n```\n\n")

        f.write("## Submission\n\n")
        f.write(f"- status: **{sub_status}**\n")
        if sub_status == "ok":
            f.write(f"- output: `submissions/submission_v6_{TIMESTAMP}.csv`\n")
            f.write("- Members retrained on FULL train with their temporal best_iter.\n")
            f.write("- Test features rebuilt with full train as `agg_source`.\n")
            f.write("- Risk notes: temporal NDCG is the local proxy — Kaggle test "
                    "drift unknown. CP feature has clean drift on temporal val; "
                    "DS feature has the cleanest drift in the session. Other members "
                    "(rank_xendcg, booking_clf, randup) inherit V4-style feature drift.\n")
        elif sub_status == "below_threshold":
            f.write("- best temporal ensemble did not exceed quick benchmark + 0.0003; "
                    "no submission created.\n")
        else:
            f.write("- submission attempted but failed; see logs.\n")

    log(f"Wrote {readme}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # last-ditch — log to a file even if everything else fails
        err = OUT / "FATAL.txt"
        err.write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        log(f"FATAL: {e}")
        raise
