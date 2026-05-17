"""Adversarial-reweighting batch — last shot to close the train→test gap.

Hypothesis: V5 had adversarial AUC=1.0 on raw features (perfect train/test
distinguishability). Our local gains don't translate to Kaggle because the
model overfits the train distribution. Reweight train rows by importance
ratio P(test|x) / (1 - P(test|x)) so the loss focuses on rows that LOOK
LIKE TEST.

Phase A (always): train adv classifier on temporal_train vs test, retrain
7 V6 members on temporal_train with adv*IPW weights, evaluate on
temporal_val.

Phase B (if temporal >= 0.40500): retrain on FULL train, build the final
Kaggle submission.

Bullet-proof: per-model try/except, atomic writes, FATAL.txt.
"""
from __future__ import annotations
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_train, load_test, make_target, get_feature_columns  # noqa: E402
from src.features import build_features, compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402
from pipelines.temporal_validation import (  # noqa: E402
    compute_ipw_weights, make_group_counts, eval_metrics, BASE_PARAMS,
)
from pipelines.evaluate_variant import _pos_adj_oof_te, _prop_dest_book_rate_safe  # noqa: E402

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
V4_ANCHOR_TEMPORAL = 0.40401
V6_LOO9_TEMPORAL = 0.40896
OVERNIGHT_BEST_KAGGLE = 0.42012  # the just-tested submission

CACHE_TRAIN = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_train.parquet"
CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"

V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V6_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]

OVERNIGHT_DIR = ROOT / "diagnostics" / "overnight_final_batch_20260517_022323"

OUT = ROOT / "diagnostics" / f"adv_reweight_batch_{TIMESTAMP}"
ERRORS_DIR = OUT / "errors"
PREDS_DIR = OUT / "predictions"
MODELS_DIR = OUT / "models"

# V6 members we retrain with adv weights (those that use sample weights):
# We exclude lambdarank_noipw (originally no weights), lambdarank_randup
# (special random_bool×2 scheme), booking_clf (binary).
ADV_RETRAIN_MEMBERS = [
    {"id": "lambdarank_base",   "label_gain": "0,1,31", "best_iter": 656},
    {"id": "lambdarank_click3", "label_gain": "0,3,31", "best_iter": 522},
    {"id": "lambdarank_bal15",  "label_gain": "0,1,15", "best_iter": 509},
    {"id": "lambdarank_book50", "label_gain": "0,1,50", "best_iter": 400},
    {"id": "rank_xendcg_v6",    "label_gain": "0,1,15", "best_iter": 800, "objective": "rank_xendcg"},
    {"id": "CP",                "label_gain": "0,1,15", "best_iter": 486, "extra": "CP"},
    {"id": "DS",                "label_gain": "0,1,15", "best_iter": 439, "extra": "DS"},
]

# These stay as their original V6 LOO predictions for ensembling
NON_ADV_V6_MEMBERS = ["lambdarank_noipw", "lambdarank_randup"]

# Extra diversifiers (existing test predictions from overnight batch)
DIVERSIFIERS = ["cb_rank_C_deeper", "cb_rank_A", "xendcg_conservative", "xendcg_reg_seed42"]

# Threshold: build submission if temporal NDCG of adv-corrected ensemble >= this
SUBMISSION_PROCEED_THRESHOLD = 0.40500


# ============================================================================
# Logging + atomic IO
# ============================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_write_df(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_write_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def safe_write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def label_remap(s):
    return s.map({0: 0, 1: 1, 5: 2}).astype(np.int32).values


def grouped_rank(srch_id, scores):
    return pd.Series(scores).groupby(pd.Series(srch_id), sort=False).rank(
        method="average", ascending=False
    ).values.astype(np.float32)


# ============================================================================
# Adversarial classifier
# ============================================================================
def train_adv_classifier(train_feat: pd.DataFrame, test_feat: pd.DataFrame,
                          feat_cols: list[str], name: str,
                          save_importance_to: Path | None = None) -> tuple[np.ndarray, dict]:
    """Train LGBM binary classifier: train (0) vs test (1).
    Returns P(test|x) for the train rows + diagnostics dict.
    """
    log(f"  Training adversarial classifier '{name}'…")
    # Stack
    X_tr = train_feat[feat_cols].astype(np.float32).to_numpy()
    X_te = test_feat[feat_cols].astype(np.float32).to_numpy()
    y_tr = np.zeros(len(X_tr), dtype=np.int32)
    y_te = np.ones(len(X_te), dtype=np.int32)
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])

    # 90/10 holdout (random across stack) for AUC measure
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    n_hold = int(0.1 * len(X))
    hold_idx = perm[:n_hold]
    fit_idx = perm[n_hold:]
    log(f"    train+test stack: {len(X):,}  (fit {len(fit_idx):,}, holdout {n_hold:,})")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    ds_fit = lgb.Dataset(X[fit_idx], label=y[fit_idx], feature_name=feat_cols)
    ds_hold = lgb.Dataset(X[hold_idx], label=y[hold_idx], reference=ds_fit)
    model = lgb.train(params, ds_fit, num_boost_round=1000,
                       valid_sets=[ds_hold], valid_names=["hold"],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    log(f"    best_iter={model.best_iteration}, holdout AUC={model.best_score['hold']['auc']:.4f}")

    # Feature importance — what makes train look different from test?
    if save_importance_to:
        imp = pd.DataFrame({
            "feature": feat_cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        safe_write_df(imp, save_importance_to)
        log(f"    top-10 adv-discriminative features:")
        for _, r in imp.head(10).iterrows():
            log(f"      {r['feature']:35s} gain={r['gain']:>12,.0f}")

    # Predict on ALL train rows (not just fit portion)
    p_test = model.predict(X_tr).astype(np.float64)
    diagnostics = {
        "holdout_auc": float(model.best_score["hold"]["auc"]),
        "best_iter": int(model.best_iteration),
        "p_test_mean": float(p_test.mean()),
        "p_test_std": float(p_test.std()),
        "p_test_p10": float(np.percentile(p_test, 10)),
        "p_test_p50": float(np.percentile(p_test, 50)),
        "p_test_p90": float(np.percentile(p_test, 90)),
        "p_test_p99": float(np.percentile(p_test, 99)),
    }
    return p_test, diagnostics


def importance_weights(p_test: np.ndarray, clip_hi: float = 10.0, clip_lo: float = 0.1) -> np.ndarray:
    """w = P(test|x) / (1 - P(test|x)), clipped and renormalized to mean=1."""
    # Avoid 0/inf
    p = np.clip(p_test, 1e-4, 1 - 1e-4)
    w = p / (1.0 - p)
    w = np.clip(w, clip_lo, clip_hi)
    # Renormalize to mean 1
    w = w / w.mean()
    return w.astype(np.float32)


# ============================================================================
# Per-V6-member retrain with adv weights
# ============================================================================
def retrain_v6_member(spec: dict, train_feat: pd.DataFrame, val_feat: pd.DataFrame,
                      base_weights: np.ndarray, adv_weights: np.ndarray,
                      num_boost_round_max: int = 2000) -> tuple[lgb.Booster, int, np.ndarray, list[str]]:
    """Retrain one V6 member with combined weights = base × adv.
    Uses early stopping on val.
    """
    # Add CP/DS feature if needed
    train = train_feat
    val = val_feat
    extra = spec.get("extra")
    # NOTE: Caller should have already added the extra feature so we don't rebuild it.

    # Feature cols
    cols = [c for c in get_feature_columns(train) if c in val.columns]
    if extra == "CP":
        cols = [c for c in cols if c != "prop_dest_book_rate_safe"]
    elif extra == "DS":
        cols = [c for c in cols if c != "prop_click_rate_pos_adj_s40_oof"]
    else:
        cols = [c for c in cols if c not in ("prop_click_rate_pos_adj_s40_oof",
                                                "prop_dest_book_rate_safe")]
    assert not (set(cols) & FORBIDDEN_FEATURES)

    weights = base_weights * adv_weights
    # Renormalize so mean = 1 (keeps loss scale stable)
    weights = (weights / weights.mean()).astype(np.float32)

    params = BASE_PARAMS.copy()
    params["objective"] = spec.get("objective", "lambdarank")
    params["metric"] = "ndcg"
    params["label_gain"] = spec["label_gain"]

    train_label = label_remap(train["relevance"])
    val_label = label_remap(val["relevance"])
    train_groups = make_group_counts(train)
    val_groups = make_group_counts(val)
    ds_tr = lgb.Dataset(train[cols], label=train_label, group=train_groups, weight=weights)
    ds_va = lgb.Dataset(val[cols], label=val_label, group=val_groups, reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=num_boost_round_max,
                       valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
                       callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    scores = model.predict(val[cols]).astype(np.float32)
    return model, int(model.best_iteration), scores, cols


# ============================================================================
# Main
# ============================================================================
def main():
    for d in (OUT, ERRORS_DIR, PREDS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"ADVERSARIAL REWEIGHT BATCH — {TIMESTAMP}")
    log(f"out: {OUT}")
    log(f"V4 Kaggle: 0.42021 | V6 Kaggle: 0.42004 | overnight Kaggle: {OVERNIGHT_BEST_KAGGLE}")
    log(f"submission proceed threshold (temporal): {SUBMISSION_PROCEED_THRESHOLD}")

    # Load cached temporal features
    log("\n--- Load temporal_train + temporal_val features ---")
    base_train = pd.read_parquet(CACHE_TRAIN).sort_values("srch_id").reset_index(drop=True)
    base_val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    log(f"  train: {len(base_train):,}  val: {len(base_val):,}  cols: {base_train.shape[1]}")

    # Build test features with temporal_train as agg_source (for adv classifier on Phase A)
    log("\n--- Build test features (agg_source=temporal_train) ---")
    test_raw = load_test().reset_index(drop=True)
    t = time.time()
    test_feat_for_adv = build_features(test_raw, agg_source=base_train, is_train=False)
    log(f"  test_feat (eval): {len(test_feat_for_adv):,} × {test_feat_for_adv.shape[1]} in "
        f"{(time.time()-t)/60:.1f} min")

    # ---- Train adv classifier for EVAL phase ----
    log("\n--- Adversarial classifier (eval phase) ---")
    # Use only the SHARED base feature columns (143 V4 features)
    base_cols = [c for c in get_feature_columns(base_train) if c in test_feat_for_adv.columns]
    # Exclude any CP/DS extras that exist in train but not test
    base_cols = [c for c in base_cols if c not in ("prop_click_rate_pos_adj_s40_oof",
                                                       "prop_dest_book_rate_safe")]
    log(f"  using {len(base_cols)} shared feature cols for adv classifier")
    try:
        p_test_train, adv_diag = train_adv_classifier(
            base_train, test_feat_for_adv, base_cols, "adv_eval",
            save_importance_to=OUT / "adv_feature_importance_eval.csv",
        )
        safe_write_json(adv_diag, OUT / "adv_classifier_eval_diagnostics.json")
        np.save(OUT / "p_test_train.npy", p_test_train)
        # Choose weight scheme based on adv AUC strength:
        # - AUC > 0.95: drift is extreme → sqrt (less aggressive, avoid OOM importance ratios)
        # - AUC 0.7-0.95: moderate drift → standard clip [0.1, 10]
        # - AUC < 0.7: low drift → softer clip [0.2, 5]
        auc = adv_diag["holdout_auc"]
        if auc > 0.95:
            log(f"  adv AUC {auc:.4f} > 0.95 — using SQRT(importance) for stability")
            raw = importance_weights(p_test_train, clip_hi=25.0, clip_lo=0.04)
            adv_w = np.sqrt(raw).clip(0.2, 5.0).astype(np.float32)
            adv_w = (adv_w / adv_w.mean()).astype(np.float32)
            adv_scheme = "sqrt_clip5"
        elif auc > 0.7:
            log(f"  adv AUC {auc:.4f} — using STANDARD importance clip [0.1, 10]")
            adv_w = importance_weights(p_test_train, clip_hi=10.0, clip_lo=0.1)
            adv_scheme = "standard_clip10"
        else:
            log(f"  adv AUC {auc:.4f} < 0.7 — soft clip [0.2, 5]")
            adv_w = importance_weights(p_test_train, clip_hi=5.0, clip_lo=0.2)
            adv_scheme = "soft_clip5"
        log(f"  CHOSEN adv weight scheme: {adv_scheme}")
        log(f"  adv weights: mean={adv_w.mean():.3f}  median={np.median(adv_w):.3f}  "
            f"min={adv_w.min():.4f}  max={adv_w.max():.4f}  std={adv_w.std():.3f}")
    except Exception as e:
        log(f"  ✗ adv classifier FAILED: {e}")
        safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                         ERRORS_DIR / "ADV_CLF_EVAL_ERROR.txt")
        return

    # Free test features (will reuse in Phase B with different agg_source if triggered)
    del test_feat_for_adv
    gc.collect()

    # ---- IPW (V6's original sample weights) ----
    log("\n--- Compute IPW (V6 original weights) ---")
    propensity = compute_position_propensity(base_train)
    ipw = compute_ipw_weights(base_train, propensity, clip_hi=10.0, clip_lo=0.1)
    log(f"  IPW: mean={ipw.mean():.3f}  min={ipw.min():.3f}  max={ipw.max():.3f}")

    # ---- Build CP and DS features on temporal_train + temporal_val (once) ----
    log("\n--- Build CP + DS extras on temporal_train + temporal_val ---")
    # Need a fresh copy with CP and DS columns added (will be used for CP/DS retrains)
    train_with_extras = base_train.copy()
    val_with_extras = base_val.copy()
    try:
        _pos_adj_oof_te(train_with_extras, val_with_extras, target_col="click_bool",
                        col_new="prop_click_rate_pos_adj_s40_oof",
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        _prop_dest_book_rate_safe(train_with_extras, val_with_extras,
                                   col_new="prop_dest_book_rate_safe",
                                   alpha=40.0, n_folds=5, seed=42)
        log(f"  train+val now have CP+DS columns ({train_with_extras.shape[1]} cols)")
    except Exception as e:
        log(f"  ✗ CP+DS build FAILED: {e}")
        safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                         ERRORS_DIR / "CP_DS_BUILD_ERROR.txt")
        return

    # ---- Retrain 7 V6 members with adv weights ----
    log("\n--- Retrain V6 members with adv*IPW weights ---")
    member_rows = []
    adv_member_preds = {}  # mid -> val predictions
    for spec in ADV_RETRAIN_MEMBERS:
        mid = spec["id"]
        log(f"\n  retraining {mid} (lg={spec['label_gain']}, extra={spec.get('extra')})…")
        try:
            t0 = time.time()
            train_use = train_with_extras
            val_use = val_with_extras
            model, best_iter, scores, cols = retrain_v6_member(
                spec, train_use, val_use, base_weights=ipw, adv_weights=adv_w
            )
            m = eval_metrics(val_use, scores)
            # Save
            model.save_model(str(MODELS_DIR / f"model_{mid}_adv.txt"))
            np.save(PREDS_DIR / f"val_pred_{mid}_adv.npy", scores)
            adv_member_preds[mid] = scores
            row = {
                "model_id": f"{mid}_adv", "status": "ok",
                "best_iter": best_iter, "ndcg5": float(m["ndcg5"]),
                "recall1": float(m["recall1"]), "recall5": float(m["recall5"]),
                "mean_booked_rank": float(m["mean_booked_rank"]),
                "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
                "elapsed_min": round((time.time()-t0)/60, 2),
            }
            member_rows.append(row)
            # Update the spec's best_iter to the adv-found value (used by Phase B)
            spec["adv_best_iter"] = best_iter
            log(f"    ✓ NDCG@5={m['ndcg5']:.5f}  Δ_v6={row['delta_vs_v6_loo9']:+.5f}  "
                f"best_iter={best_iter}  in {(time.time()-t0)/60:.1f} min")
            del model
            gc.collect()
        except Exception as e:
            log(f"    ✗ FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                             ERRORS_DIR / f"ERROR_{mid}_adv.txt")
            member_rows.append({"model_id": f"{mid}_adv", "status": f"failed:{type(e).__name__}",
                                  "error": str(e)[:300]})

        safe_write_df(pd.DataFrame(member_rows), OUT / "member_results.csv")

    n_ok = sum(1 for r in member_rows if r.get("status") == "ok")
    log(f"\nRetrain done: {n_ok}/{len(ADV_RETRAIN_MEMBERS)} ok")

    # ---- Phase A ensemble: V6 adv-corrected (7 adv + 2 unchanged) + diversifiers ----
    log("\n--- Phase A ensemble eval (temporal) ---")
    srch = val_with_extras["srch_id"].values
    ens_rows = []

    # Build adv-retrained V6 LOO-9 rank-average
    adv_v6_ranks = []
    for mid in adv_member_preds:
        adv_v6_ranks.append(grouped_rank(srch, adv_member_preds[mid]))
    # Add the 2 unchanged V6 members (noipw, randup) from V6 batch
    for mid in NON_ADV_V6_MEMBERS:
        p = V6_DIR / f"val_pred_{mid}.npy"
        if p.exists():
            s = np.load(p).astype(np.float32)
            adv_v6_ranks.append(grouped_rank(srch, s))
        else:
            log(f"  ! missing {mid} pred, skipping")
    if not adv_v6_ranks:
        log("  ✗ no V6 adv ranks available; aborting")
        return
    adv_v6_rank = np.mean(adv_v6_ranks, axis=0)
    adv_v6_metrics = eval_metrics(base_val, -adv_v6_rank)
    log(f"  adv-V6 LOO-{len(adv_v6_ranks)} alone: NDCG@5={adv_v6_metrics['ndcg5']:.5f}  "
        f"Δ_v6={float(adv_v6_metrics['ndcg5']) - V6_LOO9_TEMPORAL:+.5f}")
    ens_rows.append({
        "test_id": "adv_v6_alone", "method": "rank_avg",
        "n_members": len(adv_v6_ranks), "v6_weight": 1.0,
        "ndcg5": float(adv_v6_metrics["ndcg5"]),
        "delta_vs_v6_loo9": float(adv_v6_metrics["ndcg5"]) - V6_LOO9_TEMPORAL,
    })

    # Also reference: original V6 LOO-9 baseline
    orig_v6_ranks = [grouped_rank(srch, np.load(V6_DIR / f"val_pred_{m}.npy").astype(np.float32))
                       for m in V6_MEMBERS]
    orig_v6_rank = np.mean(orig_v6_ranks, axis=0)
    orig_v6_metrics = eval_metrics(base_val, -orig_v6_rank)
    log(f"  orig V6 LOO-9 (reference): NDCG@5={orig_v6_metrics['ndcg5']:.5f}")

    # Add diversifiers (existing val preds from overnight batch)
    diversifier_ranks = {}
    for did in DIVERSIFIERS:
        p = OVERNIGHT_DIR / "predictions" / f"val_pred_{did}.npy"
        if p.exists():
            s = np.load(p).astype(np.float32)
            diversifier_ranks[did] = grouped_rank(srch, s)
            log(f"    loaded diversifier {did}")
        else:
            log(f"    ! diversifier {did} missing, skip")

    # Ensemble: try multiple V6 weight schemes with 4 diversifiers
    if len(diversifier_ranks) == 4:
        # Test grid: V6 weight in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85}
        for w_v6 in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            w_each = (1.0 - w_v6) / 4
            combo_avg = w_v6 * adv_v6_rank + sum(
                w_each * diversifier_ranks[d] for d in DIVERSIFIERS
            )
            combo_m = eval_metrics(base_val, -combo_avg)
            log(f"  adv_v6@{w_v6:.2f} + 4 diversifiers@{w_each:.4f}ea: "
                f"NDCG@5={combo_m['ndcg5']:.5f}  "
                f"Δ_v6={float(combo_m['ndcg5']) - V6_LOO9_TEMPORAL:+.5f}")
            ens_rows.append({
                "test_id": f"adv_v6@{w_v6:.2f}+4_div@{w_each:.4f}", "method": "rank_avg",
                "n_members": 5, "v6_weight": w_v6,
                "ndcg5": float(combo_m["ndcg5"]),
                "delta_vs_v6_loo9": float(combo_m["ndcg5"]) - V6_LOO9_TEMPORAL,
            })

    safe_write_df(pd.DataFrame(ens_rows), OUT / "ensemble_results.csv")
    best_ens = max(ens_rows, key=lambda r: r["ndcg5"])
    log(f"\n=== Phase A best ensemble: {best_ens['test_id']} = {best_ens['ndcg5']:.5f} ===")
    log(f"    Δ vs V6 LOO-9 = {best_ens['delta_vs_v6_loo9']:+.5f}")

    # ---- Phase B decision ----
    if best_ens["ndcg5"] < SUBMISSION_PROCEED_THRESHOLD:
        log(f"\nPhase A best ({best_ens['ndcg5']:.5f}) < threshold "
            f"({SUBMISSION_PROCEED_THRESHOLD}) — skipping submission build.")
        write_readme(member_rows, ens_rows, adv_diag, best_ens, None, t_start)
        return

    log(f"\nPhase A best ({best_ens['ndcg5']:.5f}) >= threshold "
        f"({SUBMISSION_PROCEED_THRESHOLD}) — building Phase B submission")

    # Free temporal data we won't reuse
    del train_with_extras, val_with_extras, adv_member_preds, base_train, base_val
    gc.collect()

    # ---- PHASE B: Full-train retrain + submission ----
    sub_info = None
    try:
        sub_info = build_submission(best_ens, ipw_member_specs=ADV_RETRAIN_MEMBERS)
    except Exception as e:
        log(f"\n✗ Phase B FAILED: {e}")
        safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                         ERRORS_DIR / "PHASE_B_ERROR.txt")

    write_readme(member_rows, ens_rows, adv_diag, best_ens, sub_info, t_start)
    log(f"\n=== ALL DONE in {(time.time()-t_start)/60:.1f} min ===")


def build_submission(best_ens, ipw_member_specs):
    log("\n" + "=" * 70)
    log("PHASE B — full-train retrain + submission")
    log("=" * 70)

    # Load FULL train
    log("Loading full train…")
    train_raw = load_train()
    train_raw = make_target(train_raw).sort_values("srch_id").reset_index(drop=True)
    log(f"  full train: {len(train_raw):,} rows")

    log("Loading test…")
    test_raw = load_test().reset_index(drop=True)
    log(f"  test: {len(test_raw):,}")

    log("Building full_train features (agg_source=full)…")
    t = time.time()
    train_full = build_features(train_raw, agg_source=train_raw, is_train=True)
    log(f"  train_full: {train_full.shape[1]} cols in {(time.time()-t)/60:.1f} min")

    log("Building test features (agg_source=full)…")
    t = time.time()
    test_full = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"  test_full: {test_full.shape[1]} cols in {(time.time()-t)/60:.1f} min")

    test_srch = test_full["srch_id"].values
    del train_raw, test_raw
    gc.collect()

    # Adv classifier on full train vs test
    log("\nTraining FULL-train adversarial classifier…")
    base_cols = [c for c in get_feature_columns(train_full) if c in test_full.columns]
    base_cols = [c for c in base_cols if c not in ("prop_click_rate_pos_adj_s40_oof",
                                                       "prop_dest_book_rate_safe")]
    try:
        p_test_full, adv_diag_b = train_adv_classifier(train_full, test_full, base_cols, "adv_full")
        safe_write_json(adv_diag_b, OUT / "adv_classifier_full_diagnostics.json")
        adv_w_full = importance_weights(p_test_full, clip_hi=10.0, clip_lo=0.1)
        log(f"  FULL adv weights: mean={adv_w_full.mean():.3f}  "
            f"min={adv_w_full.min():.4f}  max={adv_w_full.max():.4f}")
    except Exception as e:
        log(f"  ✗ adv classifier FULL FAILED: {e}")
        raise

    propensity_full = compute_position_propensity(train_full)
    ipw_full = compute_ipw_weights(train_full, propensity_full, clip_hi=10.0, clip_lo=0.1)

    # Build CP + DS on full train + test
    log("\nBuilding CP + DS extras on full train + test…")
    train_with_extras_full = train_full.copy()
    test_with_extras_full = test_full.copy()
    _pos_adj_oof_te(train_with_extras_full, test_with_extras_full, target_col="click_bool",
                    col_new="prop_click_rate_pos_adj_s40_oof",
                    alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
    _prop_dest_book_rate_safe(train_with_extras_full, test_with_extras_full,
                               col_new="prop_dest_book_rate_safe",
                               alpha=40.0, n_folds=5, seed=42)
    log(f"  done ({train_with_extras_full.shape[1]} cols)")

    # Retrain each member on full train + predict on test
    log("\nRetraining each V6 member on FULL train…")
    member_test_preds = {}
    for spec in ipw_member_specs:
        mid = spec["id"]
        # Use Phase A's adv-found best_iter if available, else fall back to V6's
        bi = spec.get("adv_best_iter", spec["best_iter"])
        log(f"\n  retraining {mid} on full train (best_iter={bi} "
            f"[{'adv-tuned' if 'adv_best_iter' in spec else 'V6-original'}])…")
        try:
            t0 = time.time()
            train_use = train_with_extras_full
            test_use = test_with_extras_full
            extra = spec.get("extra")
            cols = [c for c in get_feature_columns(train_use) if c in test_use.columns]
            if extra == "CP":
                cols = [c for c in cols if c != "prop_dest_book_rate_safe"]
            elif extra == "DS":
                cols = [c for c in cols if c != "prop_click_rate_pos_adj_s40_oof"]
            else:
                cols = [c for c in cols if c not in ("prop_click_rate_pos_adj_s40_oof",
                                                          "prop_dest_book_rate_safe")]
            assert not (set(cols) & FORBIDDEN_FEATURES)
            weights = (ipw_full * adv_w_full).astype(np.float64)
            weights = (weights / weights.mean()).astype(np.float32)

            params = BASE_PARAMS.copy()
            params["objective"] = spec.get("objective", "lambdarank")
            params["metric"] = "ndcg"
            params["label_gain"] = spec["label_gain"]
            label = label_remap(train_use["relevance"])
            groups = make_group_counts(train_use)
            ds_tr = lgb.Dataset(train_use[cols], label=label, group=groups, weight=weights)
            model = lgb.train(params, ds_tr, num_boost_round=bi,
                               callbacks=[lgb.log_evaluation(0)])
            test_scores = model.predict(test_use[cols]).astype(np.float32)
            np.save(PREDS_DIR / f"test_pred_{mid}_adv.npy", test_scores)
            model.save_model(str(MODELS_DIR / f"model_{mid}_adv_FULL.txt"))
            member_test_preds[mid] = test_scores
            log(f"    ✓ done in {(time.time()-t0)/60:.1f} min")
            del model
            gc.collect()
        except Exception as e:
            log(f"    ✗ FAILED retrain {mid}: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                             ERRORS_DIR / f"FULL_RETRAIN_ERROR_{mid}.txt")
            # Try to use original V6 test pred as fallback
            p = V6_DIR / f"test_pred_{mid}.npy"
            if p.exists():
                log(f"    fallback: using original V6 test pred for {mid}")
                member_test_preds[mid] = np.load(p).astype(np.float32)

    # Add non-adv V6 members from V6 batch
    for mid in NON_ADV_V6_MEMBERS:
        p = V6_DIR / f"test_pred_{mid}.npy"
        if p.exists():
            member_test_preds[mid] = np.load(p).astype(np.float32)
            log(f"  loaded {mid} test pred (V6 original)")

    # Diversifiers
    diversifier_test_ranks = {}
    for did in DIVERSIFIERS:
        p = OVERNIGHT_DIR / "predictions" / f"test_pred_{did}.npy"
        if p.exists():
            diversifier_test_ranks[did] = grouped_rank(test_srch, np.load(p).astype(np.float32))
            log(f"  loaded diversifier test pred: {did}")
        else:
            log(f"  ! diversifier {did} test pred MISSING")

    # Build ensemble
    v6_ranks = [grouped_rank(test_srch, member_test_preds[mid]) for mid in member_test_preds]
    v6_rank = np.mean(v6_ranks, axis=0)
    log(f"\nV6 adv test rank-average over {len(v6_ranks)} members ready")

    # Use best_ens['v6_weight'] and 4 diversifiers
    w_v6 = float(best_ens["v6_weight"])
    n_div = len(diversifier_test_ranks)
    w_each = (1.0 - w_v6) / n_div if n_div else 0
    avg = w_v6 * v6_rank
    for did, r in diversifier_test_ranks.items():
        avg = avg + w_each * r

    sub_df = pd.DataFrame({
        "srch_id": test_srch,
        "prop_id": test_full["prop_id"].values,
        "_rk": avg,
    }).sort_values(["srch_id", "_rk"])[["srch_id", "prop_id"]]

    # Validate
    sample = pd.read_csv(ROOT / "data" / "submission_sample.csv")
    assert list(sub_df.columns) == ["srch_id", "prop_id"]
    assert len(sub_df) == len(sample)
    assert set(sub_df["srch_id"]) == set(sample["srch_id"])
    assert not sub_df.isna().any().any()
    assert not sub_df.duplicated().any()

    sub_csv = ROOT / "submissions" / f"submission_adv_reweight_{TIMESTAMP}.csv"
    sub_csv.parent.mkdir(exist_ok=True)
    sub_df.to_csv(sub_csv, index=False)
    log(f"\n✓ SUBMISSION WRITTEN: {sub_csv}")
    log(f"  rows={len(sub_df):,}  searches={sub_df['srch_id'].nunique():,}")

    # README
    safe_write_text(
        f"# Adversarial reweight submission — {TIMESTAMP}\n\n"
        f"## Strategy\n\n"
        f"Reweight train rows by importance ratio `P(test|x) / (1 - P(test|x))` to correct "
        f"train→test distribution drift. V5 had adversarial AUC=1.0 — perfect distinguishability. "
        f"This submission retrains V6's IPW-using members with combined weights = IPW × adv.\n\n"
        f"## Adversarial classifier diagnostics (full-train)\n"
        f"- Holdout AUC: {adv_diag_b['holdout_auc']:.4f}\n"
        f"- best_iter: {adv_diag_b['best_iter']}\n"
        f"- P(test) percentiles on train: p10={adv_diag_b['p_test_p10']:.4f}, "
        f"p50={adv_diag_b['p_test_p50']:.4f}, p90={adv_diag_b['p_test_p90']:.4f}\n\n"
        f"## Composition\n"
        f"- V6 (adv-retrained + 2 unchanged): weight {w_v6:.4f}\n"
        f"  - {len(member_test_preds)} V6 members\n"
        f"- {n_div} diversifiers @ {w_each:.4f} each\n\n"
        f"## Local result\n"
        f"- Best ensemble: {best_ens['test_id']} NDCG@5={best_ens['ndcg5']:.5f}\n"
        f"- Δ vs V6 LOO-9 (0.40896): {best_ens['delta_vs_v6_loo9']:+.5f}\n\n"
        f"## Risk notes\n"
        f"- Local→Kaggle gap before adv: +0.011 (V6 0.40896 → 0.42004)\n"
        f"- Adversarial reweighting SPECIFICALLY targets this gap\n"
        f"- Best case: Kaggle gain matches local gain (+0.001-0.003)\n"
        f"- Risk case: local drops and Kaggle drops too\n",
        ROOT / "submissions" / f"submission_adv_reweight_{TIMESTAMP}_README.md"
    )

    return {"path": str(sub_csv), "v6_weight": w_v6, "n_members": len(member_test_preds),
            "n_diversifiers": n_div, "adv_diag": adv_diag_b}


def write_readme(member_rows, ens_rows, adv_diag, best_ens, sub_info, t_start):
    L = [f"# Adversarial reweight batch — {TIMESTAMP}\n"]
    L.append(f"_Generated {datetime.now().isoformat()} • "
             f"elapsed {(time.time() - t_start)/60:.1f} min_\n")
    L.append("## Hypothesis\n"
             "V5 had adversarial AUC=1.0 — perfect train/test distinguishability on features. "
             "Local gains don't translate to Kaggle because models overfit train distribution. "
             "Reweight train rows by importance ratio so loss focuses on test-like rows.\n\n")

    L.append("## Adversarial classifier (eval phase)\n")
    L.append(f"- Holdout AUC: **{adv_diag['holdout_auc']:.4f}**")
    if adv_diag["holdout_auc"] > 0.9:
        L.append("  → **HIGH drift confirmed.** Reweighting should help.")
    elif adv_diag["holdout_auc"] > 0.7:
        L.append("  → Moderate drift. Reweighting may help modestly.")
    else:
        L.append("  → LOW drift. Reweighting probably won't help.")
    L.append(f"- best_iter: {adv_diag['best_iter']}")
    L.append(f"- P(test) on train: p50={adv_diag['p_test_p50']:.4f}, "
             f"p90={adv_diag['p_test_p90']:.4f}\n")

    L.append("## V6 member retrains with adv*IPW weights\n```")
    mdf = pd.DataFrame(member_rows)
    if "ndcg5" in mdf.columns:
        cols = ["model_id", "status", "ndcg5", "delta_vs_v6_loo9", "best_iter", "elapsed_min"]
        cols = [c for c in cols if c in mdf.columns]
        L.append(mdf[cols].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    L.append("```\n")

    L.append("## Phase A ensembles\n```")
    edf = pd.DataFrame(ens_rows)
    L.append(edf.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    L.append("```\n")

    L.append(f"## Best Phase A: {best_ens['test_id']}\n")
    L.append(f"- NDCG@5 = {best_ens['ndcg5']:.5f}")
    L.append(f"- Δ vs V6 LOO-9 (0.40896): {best_ens['delta_vs_v6_loo9']:+.5f}")
    L.append(f"- Δ vs overnight Kaggle (0.42012): N/A (this is temporal val, not Kaggle)\n")

    L.append("## Submission status\n")
    if sub_info:
        L.append(f"**Built:** `{Path(sub_info['path']).name}`")
        L.append(f"- V6 weight: {sub_info['v6_weight']:.4f}")
        L.append(f"- V6 members: {sub_info['n_members']}")
        L.append(f"- Diversifiers: {sub_info['n_diversifiers']}")
        L.append(f"- Adv AUC (full train): {sub_info['adv_diag']['holdout_auc']:.4f}\n")
    else:
        L.append("Not built (Phase A below threshold or Phase B failed).\n")

    safe_write_text("\n".join(L), OUT / "README.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        safe_write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                         ERRORS_DIR / "FATAL.txt")
        log(f"FATAL: {e}")
        raise
