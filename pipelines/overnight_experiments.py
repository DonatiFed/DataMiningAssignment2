#!/usr/bin/env python3
"""
run_overnight_experiments.py — autonomous overnight candidate-model generator.

Goal:
    Generate many single-model candidates overnight for tomorrow's ensemble and
    feature analysis. Robust, resumable, logged, interpretable.

Hard rules (enforced in code):
    - Single models only.
    - No full train+validation retraining.
    - No Kaggle submissions.
    - No ensemble generation tonight.
    - V4 full feature pipeline unless a config explicitly filters features.
    - Fixed V4 validation split (val_frac=0.1, random_state=42).
    - V4-style Dataset construction: fresh lgb.Dataset per config, no
      explicit .construct(), let lgb.train construct lazily so seed reaches binning.
    - Save artifacts after every model.
    - Skip if model_result_<config_id>.json already exists in the run dir.
    - Catch exceptions per config, log them, continue.
    - No invasive feature-code changes; if a requested experiment needs them,
      skip it with a documented reason.

Modes:
    --dry-run     : print configs + paths, no data load, no training.
    --smoke-test  : 1% srch_id-level sample, 2 tiny configs + intentional
                    error-handling test, ~50 boosting rounds each.
    (no flag)     : full overnight run.

This script is meant to be launched in a tmux session and left running. The
final summary lands at experiment_logs/overnight_summary.md and the candidate
list at experiment_logs/candidate_models_for_ensemble.csv.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# sys.path bootstrap so `from src.X import …` resolves no matter how the script
# is invoked (e.g. `python run_overnight_experiments.py` from project root).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import ROOT_DIR
from src.data_loader import (
    load_train, make_target, get_feature_columns, split_val,
)
from src.features import (
    build_features, compute_position_propensity, FORBIDDEN_FEATURES,
)
from src.evaluate import evaluate_ndcg
from src.artifacts import (
    run_dirs, save_run_config, save_git_commit, save_feature_cols, save_val_meta,
    save_model_artifacts,
)


# ============================================================================
# Constants
# ============================================================================

EXTERNAL_COPY_ROOT = Path("/home/ubuntu/experiment_artifacts")
EXPERIMENT_LOGS = ROOT_DIR / "experiment_logs"
FEATURE_AUDIT_CSV = EXPERIMENT_LOGS / "feature_audit.csv"

ANCHOR_SEED = 456
ANCHOR_NDCG = 0.42191                # V4 best single (lambdarank_bal15) local
V4_ENSEMBLE_LOCAL = 0.42512
V4_KAGGLE = 0.42021
PHASE2_BEST_NDCG = 0.42258           # Phase 2 winner lg_0_2_15 local

NUM_BOOST_ROUND_DEFAULT = 2000
EARLY_STOPPING_DEFAULT = 80
LOG_EVAL_INTERVAL_DEFAULT = 200

NUM_BOOST_ROUND_SMOKE = 50
EARLY_STOPPING_SMOKE = 10
SMOKE_SAMPLE_FRAC = 0.01

CANDIDATE_THRESHOLD = 0.4210
STRONG_CANDIDATE_THRESHOLD = 0.4220
EXCELLENT_CANDIDATE_THRESHOLD = 0.4230
POSSIBLE_OVERFIT_THRESHOLD = 0.4240
WEAK_THRESHOLD = 0.4180

V5_EXPERIMENTAL_INTERACTIONS = {
    "is_last_minute", "is_short_window", "is_long_window",
    "is_family", "total_guests", "price_per_guest",
    "is_discounted", "is_overpriced",
}

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

# Each weighting profile is interpreted by compute_weights(). `base="ipw"` means
# start from V4 default IPW, then apply the listed modifiers; `base="none"` means
# all-ones (no IPW at all).
WEIGHTING_PROFILES = {
    "ipw_default":     {"base": "ipw", "clip": (0.1, 10.0)},
    "no_ipw":          {"base": "none"},
    "ipw_positive":    {"base": "ipw", "clip": (0.1, 10.0), "positive_only": True},
    "ipw_clip3":       {"base": "ipw", "clip": (0.1, 3.0)},
    "ipw_clip5":       {"base": "ipw", "clip": (0.1, 5.0)},
    "rand_up_1.5":     {"base": "ipw", "clip": (0.1, 10.0), "random_upweight": 1.5},
    "rand_up_2.0":     {"base": "ipw", "clip": (0.1, 10.0), "random_upweight": 2.0},
    "rand_up_3.0":     {"base": "ipw", "clip": (0.1, 10.0), "random_upweight": 3.0},
    "booking_q_2x":    {"base": "ipw", "clip": (0.1, 10.0), "booking_query_upweight": 2.0},
    "booked_2x":       {"base": "ipw", "clip": (0.1, 10.0), "booked_row_upweight": 2.0},
    "clicked_0.5x":    {"base": "ipw", "clip": (0.1, 10.0), "clicked_row_upweight": 0.5},
    "down_all_zero_q": {"base": "ipw", "clip": (0.1, 10.0), "all_zero_query_weight": 0.1},
}


# ============================================================================
# Logging
# ============================================================================

def log(msg: str = "") -> None:
    """Timestamped, flushed print so tail -f / tee always show fresh output."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================================
# Config catalog (declarative)
# ============================================================================

def _base_config(**overrides) -> dict:
    """Return a fully-defaulted config dict, with `overrides` applied."""
    cfg = dict(
        id="",
        group="",
        label_gain="0,2,15",
        seed=ANCHOR_SEED,
        weighting="ipw_default",
        feature_filter=None,
        row_filter=None,
        param_overrides={},
        objective="lambdarank",
        boosting="gbdt",
        target_kind="remapped",   # "remapped" | "booking" | "click" | "rel_gt_zero"
        num_boost_round=NUM_BOOST_ROUND_DEFAULT,
        early_stopping=EARLY_STOPPING_DEFAULT,
        description="",
        skip_reason=None,
    )
    cfg.update(overrides)
    return cfg


def build_all_configs() -> list[dict]:
    """Static, deterministic config list. 86 configs total."""
    configs: list[dict] = []

    # ---- Group A — extended label gains -----------------------------------
    a_specs = [
        ("A1",  "0,2,12"), ("A2",  "0,2,18"), ("A3",  "0,2,20"), ("A4",  "0,2,25"),
        ("A5",  "0,3,12"), ("A6",  "0,3,15"), ("A7",  "0,3,18"), ("A8",  "0,3,20"),
        ("A9",  "0,4,15"), ("A10", "0,4,20"), ("A11", "0,5,20"),
        ("A12", "0,1,25"), ("A13", "0,1,30"),
    ]
    for cid, lg in a_specs:
        configs.append(_base_config(
            id=cid, group="A", label_gain=lg,
            description=f"label_gain={lg}",
        ))

    # ---- Group B — weighting variants --------------------------------------
    b_specs = [
        ("B1",  "0,2,15", "no_ipw"),
        ("B2",  "0,2,15", "ipw_positive"),
        ("B3",  "0,2,15", "ipw_clip3"),
        ("B4",  "0,2,15", "ipw_clip5"),
        ("B5",  "0,2,15", "rand_up_1.5"),
        ("B6",  "0,2,15", "rand_up_2.0"),
        ("B7",  "0,3,15", "no_ipw"),
        ("B8",  "0,3,15", "ipw_positive"),
        ("B9",  "0,3,15", "rand_up_1.5"),
        ("B10", "0,2,20", "no_ipw"),
        ("B11", "0,2,20", "ipw_positive"),
        ("B12", "0,2,20", "rand_up_1.5"),
    ]
    for cid, lg, w in b_specs:
        configs.append(_base_config(
            id=cid, group="B", label_gain=lg, weighting=w,
            description=f"lg={lg}, w={w}",
        ))

    # ---- Group C — seed diversity ------------------------------------------
    c_specs = [
        ("C1",  "0,2,15", 42),    ("C2",  "0,2,15", 123),
        ("C3",  "0,2,15", 789),   ("C4",  "0,2,15", 2024),
        ("C5",  "0,2,15", 9999),
        ("C6",  "0,3,15", 42),    ("C7",  "0,3,15", 123),
        ("C8",  "0,3,15", 789),   ("C9",  "0,3,15", 2024),
        ("C10", "0,2,20", 42),    ("C11", "0,2,20", 123),
        ("C12", "0,2,20", 789),
    ]
    for cid, lg, seed in c_specs:
        configs.append(_base_config(
            id=cid, group="C", label_gain=lg, seed=seed,
            description=f"lg={lg}, seed={seed}",
        ))

    # ---- Group D — feature pruning -----------------------------------------
    d_specs = [
        ("D1",  "0,2,15", "drop_prop_avg_position"),
        ("D2",  "0,2,15", "drop_position_derived"),
        ("D3",  "0,3,15", "drop_prop_avg_position"),
        ("D4",  "0,2,15", "top_120"),
        ("D5",  "0,2,15", "top_100"),
        ("D6",  "0,2,15", "top_80"),
        ("D7",  "0,2,15", "drop_bottom_30pct"),
        ("D8",  "0,2,15", "drop_bottom_50pct"),
        ("D9",  "0,2,15", "drop_missing_flags"),
        ("D10", "0,2,15", "drop_is_best_flags"),
        ("D11", "0,2,15", "drop_booking_buckets"),
        ("D12", "0,2,15", "drop_cross_tes"),
        ("D13", "0,2,15", "keep_raw_listwise_price"),
        ("D14", "0,2,15", "keep_te_and_raw"),
    ]
    for cid, lg, ff in d_specs:
        configs.append(_base_config(
            id=cid, group="D", label_gain=lg, feature_filter=ff,
            description=f"feature_filter={ff}, lg={lg}",
        ))

    # ---- Group E — model complexity / regularization -----------------------
    e_specs = [
        ("E1",  {"num_leaves": 255, "min_child_samples": 100},                 2000, 80),
        ("E2",  {"num_leaves": 128, "min_child_samples": 200},                 2000, 80),
        ("E3",  {"num_leaves": 512, "min_child_samples": 100, "reg_lambda": 2.0}, 2000, 80),
        ("E4",  {"num_leaves": 400, "min_child_samples": 100},                 2000, 80),
        ("E5",  {"colsample_bytree": 0.5},                                     2000, 80),
        ("E6",  {"subsample": 0.6},                                            2000, 80),
        ("E7",  {"learning_rate": 0.02},                                       3000, 100),
        ("E8",  {"learning_rate": 0.05},                                       1500, 60),
        ("E9",  {"reg_lambda": 3.0},                                           2000, 80),
        ("E10", {"reg_alpha": 0.5},                                            2000, 80),
    ]
    for cid, po, nbr, es in e_specs:
        configs.append(_base_config(
            id=cid, group="E", label_gain="0,2,15",
            param_overrides=po, num_boost_round=nbr, early_stopping=es,
            description=f"params={po}",
        ))

    # ---- Group F — controlled creative variants ----------------------------
    f_specs = [
        ("F1",  "0,2,15", None,                     "drop_all_zero_q",   "ipw_default",    "drop all-zero queries"),
        ("F2",  "0,2,15", None,                     None,                "down_all_zero_q","downweight all-zero queries 0.1x"),
        ("F3",  "0,2,15", None,                     None,                "booking_q_2x",   "booking-queries weight 2x"),
        ("F4",  "0,2,15", None,                     None,                "booked_2x",      "booked rows weight 2x"),
        ("F5",  "0,2,15", None,                     None,                "clicked_0.5x",   "clicked rows weight 0.5x"),
        ("F6",  "0,2,15", None,                     "random_only_0",     "ipw_default",    "train random_bool=0 only"),
        ("F7",  "0,2,15", None,                     "random_only_1",     "no_ipw",         "train random_bool=1 only"),
        ("F8",  "0,2,15", None,                     None,                "rand_up_3.0",    "random rows weight 3x"),
        ("F9",  "0,2,15", "drop_competitor",        None,                "ipw_default",    "drop all competitor features"),
        ("F10", "0,2,15", "drop_visitor_history",   None,                "ipw_default",    "drop visitor-history features"),
        ("F11", "0,2,15", None,                     None,                "ipw_default",    "price clip p99.5 — SKIPPED"),
        ("F12", "0,2,15", None,                     "positive_q_only",   "no_ipw",         "positive queries only + no IPW"),
        ("F13", "0,3,15", None,                     "positive_q_only",   "ipw_default",    "positive queries only, lg=0/3/15"),
        ("F14", "0,2,15", "drop_prop_avg_position", "positive_q_only",   "ipw_default",    "positive queries + drop prop_avg_position"),
    ]
    for cid, lg, ff, rf, w, desc in f_specs:
        cfg = _base_config(
            id=cid, group="F", label_gain=lg,
            feature_filter=ff, row_filter=rf, weighting=w,
            description=desc,
        )
        if cid == "F11":
            cfg["skip_reason"] = (
                "Price clip at p99.5 changes raw price_usd, which propagates to "
                "all price-derived features (price_per_night, price_per_star, "
                "listwise price ranks, etc.). Requires a fresh build_features() "
                "call dedicated to this config; deferred to a stand-alone script "
                "to keep the overnight runner non-invasive."
            )
        configs.append(cfg)

    # ---- Group G — feature additions (ALL SKIPPED — invasive) --------------
    g_specs = [
        ("G1", "0,2,15", "add: TE rank within srch_id for single-key TEs",
         "Requires adding new features inside src/features.py:listwise_features (post-hotel_aggregates pass). Invasive — plan in Phase 5."),
        ("G2", "0,2,15", "add: train+test catalog stats for non-target aggregates",
         "Requires modifying src/features.py:hotel_aggregates to accept a separate non-target source. Invasive — plan in Phase 5."),
        ("G3", "0,2,15", "cross-key TE smoothing prior_weight=40",
         "prior_weight is hardcoded inside src/features.py:hotel_aggregates cross_specs. Requires either monkey-patching the module or editing the file. Invasive — plan in Phase 5."),
        ("G4", "0,2,15", "cross-key TE smoothing prior_weight=80",
         "Same as G3 — hardcoded prior_weight. Invasive — plan in Phase 5."),
    ]
    for cid, lg, desc, reason in g_specs:
        configs.append(_base_config(
            id=cid, group="G", label_gain=lg,
            description=desc, skip_reason=reason,
        ))

    # ---- Group H — different model families --------------------------------
    # H1: rank_xendcg (no label_gain — objective has its own gain function)
    configs.append(_base_config(
        id="H1", group="H", label_gain=None, objective="rank_xendcg",
        target_kind="remapped",
        description="rank_xendcg objective (diversity candidate)",
    ))
    # H2: DART boosting
    configs.append(_base_config(
        id="H2", group="H", label_gain="0,2,15", boosting="dart",
        param_overrides={"drop_rate": 0.1, "max_drop": 50},
        num_boost_round=1500, early_stopping=80,
        description="DART boosting (diversity candidate)",
    ))
    # H3: binary booking classifier
    configs.append(_base_config(
        id="H3", group="H", label_gain=None,
        objective="binary", target_kind="booking",
        param_overrides={"metric": "auc", "is_unbalance": True},
        description="binary booking classifier (eval NDCG@5 by sorted prob)",
    ))
    # H4: binary click classifier
    configs.append(_base_config(
        id="H4", group="H", label_gain=None,
        objective="binary", target_kind="click",
        param_overrides={"metric": "auc", "is_unbalance": True},
        description="binary click classifier (eval NDCG@5 by sorted prob)",
    ))
    # H5: blended target (relevance > 0) classifier
    configs.append(_base_config(
        id="H5", group="H", label_gain=None,
        objective="binary", target_kind="rel_gt_zero",
        param_overrides={"metric": "auc", "is_unbalance": True},
        description="binary (relevance>0) classifier (eval NDCG@5 by sorted prob)",
    ))
    # H6 / H7: SKIPPED per overnight rules
    configs.append(_base_config(
        id="H6", group="H", label_gain="0,2,15",
        description="XGBoost rank:ndcg (diversity candidate)",
        skip_reason="Non-trivial group-handling implementation in XGBoost. Skipping per overnight rules; revisit when we have a dedicated XGBoost script.",
    ))
    configs.append(_base_config(
        id="H7", group="H", label_gain="0,2,15",
        description="CatBoost ranker (diversity candidate)",
        skip_reason="Non-trivial implementation in CatBoost. Skipping per overnight rules.",
    ))

    # ---- Group I — creative experiments (added by Claude) ------------------
    # Each carries a clear hypothesis. Targeted at: closing the local→Kaggle
    # gap, ensemble diversity, and validating regularization assumptions.
    #
    # I1: Heavy regularization combo. Hypothesis — the local→Kaggle gap is
    #     overfit-driven; aggressive complexity caps should narrow it even if
    #     val NDCG drops a bit.
    configs.append(_base_config(
        id="I1", group="I", label_gain="0,2,15",
        param_overrides={"num_leaves": 63, "min_child_samples": 300,
                         "reg_lambda": 5.0, "reg_alpha": 1.0},
        description="Heavy regularization (num_leaves=63, min_child=300, reg_l1=1, reg_l2=5)",
    ))
    # I2: Drop the riskiest TEs + heavy reg. Strongest 'close-the-gap' bet.
    configs.append(_base_config(
        id="I2", group="I", label_gain="0,2,15",
        feature_filter="drop_cross_tes",
        param_overrides={"num_leaves": 128, "min_child_samples": 200,
                         "reg_lambda": 3.0},
        description="Generalization-first compound: drop cross-key TEs + heavy reg",
    ))
    # I3: GOSS boosting. Different sampling profile → useful ensemble diversity.
    configs.append(_base_config(
        id="I3", group="I", label_gain="0,2,15",
        boosting="goss",
        description="GOSS boosting (gradient one-side sampling, diversity)",
    ))
    # I4: Triple debias — drop prop_avg_position AND cross TEs AND no IPW.
    configs.append(_base_config(
        id="I4", group="I", label_gain="0,2,15",
        feature_filter="triple_debias", weighting="no_ipw",
        description="Triple debias: drop prop_avg_position+cross_TEs, no IPW",
    ))
    # I5: Train ONLY on random_bool=1 (unbiased positions) with very strong
    #     booking gain. Smaller training set but no position bias to learn.
    configs.append(_base_config(
        id="I5", group="I", label_gain="0,3,30",
        row_filter="random_only_1", weighting="no_ipw",
        description="random_bool=1 only + strong booking gain (0/3/30) + no IPW",
    ))
    # I6: DART, lighter drop_rate. Diversity candidate.
    configs.append(_base_config(
        id="I6", group="I", label_gain="0,2,15",
        boosting="dart",
        param_overrides={"drop_rate": 0.05, "max_drop": 30},
        num_boost_round=1500, early_stopping=80,
        description="DART boosting, light drop_rate=0.05",
    ))
    # I7: DART, heavier drop_rate. Diversity candidate.
    configs.append(_base_config(
        id="I7", group="I", label_gain="0,2,15",
        boosting="dart",
        param_overrides={"drop_rate": 0.2, "max_drop": 80},
        num_boost_round=1500, early_stopping=80,
        description="DART boosting, heavy drop_rate=0.2",
    ))
    # I8: max_bin=64. Coarser histograms → less overfit on continuous features.
    configs.append(_base_config(
        id="I8", group="I", label_gain="0,2,15",
        param_overrides={"max_bin": 64},
        description="Coarser histograms (max_bin=64) to reduce continuous-feature overfit",
    ))
    # I9: Low LR + extended budget + diverse seed (ensemble member candidate).
    configs.append(_base_config(
        id="I9", group="I", label_gain="0,2,15", seed=9999,
        param_overrides={"learning_rate": 0.02},
        num_boost_round=3500, early_stopping=120,
        description="Low LR (0.02) + extended budget + seed=9999 (ensemble-diversity slot)",
    ))
    # I10: focus_on_signal — drop bottom 30% features + prop_avg_position +
    #      moderate reg. Tests the "less noise, less overfit" hypothesis.
    configs.append(_base_config(
        id="I10", group="I", label_gain="0,2,15",
        feature_filter="focus_on_signal",
        param_overrides={"reg_lambda": 2.0},
        description="Focus on signal: drop bottom 30% + prop_avg_position + reg_l2=2",
    ))
    # I11: Aggressive early stopping (es=30) — forces the model to commit
    #      earlier and reduces overfit to val noise.
    configs.append(_base_config(
        id="I11", group="I", label_gain="0,2,15",
        early_stopping=30,
        description="Aggressive early stopping (patience=30)",
    ))
    # I12: Fast-shrinkage regularization — high LR, few rounds.
    configs.append(_base_config(
        id="I12", group="I", label_gain="0,2,15",
        param_overrides={"learning_rate": 0.1},
        num_boost_round=600, early_stopping=40,
        description="Fast shrinkage (lr=0.1, 600 rounds)",
    ))
    # I13: No subsampling. Tests whether subsample=0.7 / colsample=0.6 were
    #      helping or hurting.
    configs.append(_base_config(
        id="I13", group="I", label_gain="0,2,15",
        param_overrides={"subsample": 1.0, "colsample_bytree": 1.0},
        description="No subsampling (subsample=1.0, colsample=1.0)",
    ))

    return configs


# ============================================================================
# Filters + weights
# ============================================================================

def apply_feature_filter(feature_cols: list[str], filter_name: str | None,
                         audit_df: pd.DataFrame | None) -> list[str]:
    """Return subset of `feature_cols` according to `filter_name`."""
    if filter_name is None:
        return list(feature_cols)
    cols = list(feature_cols)

    if filter_name in ("drop_prop_avg_position", "drop_position_derived"):
        return [c for c in cols if c != "prop_avg_position"]

    if filter_name.startswith("top_"):
        n = int(filter_name.split("_")[1])
        if audit_df is None:
            raise RuntimeError("feature_audit.csv required for top_N filter")
        keep = set(audit_df.head(n)["feature"].tolist())
        return [c for c in cols if c in keep]

    if filter_name.startswith("drop_bottom_"):
        # drop_bottom_30pct → drop bottom 30% by gain
        pct = int(filter_name.split("_")[2].replace("pct", "")) / 100.0
        if audit_df is None:
            raise RuntimeError("feature_audit.csv required for drop_bottom_*")
        n_drop = int(len(audit_df) * pct)
        drop = set(audit_df.tail(n_drop)["feature"].tolist())
        return [c for c in cols if c not in drop]

    if filter_name == "drop_missing_flags":
        drop = {"has_visitor_history", "has_historical_price", "has_distance",
                "has_query_affinity"}
        return [c for c in cols if c not in drop]

    if filter_name == "drop_is_best_flags":
        drop = {"is_best_star", "is_best_review", "is_best_location1",
                "is_cheapest", "is_most_expensive"}
        return [c for c in cols if c not in drop]

    if filter_name == "drop_booking_buckets":
        drop = {"is_last_minute", "is_short_window", "is_long_window"}
        return [c for c in cols if c not in drop]

    if filter_name == "drop_cross_tes":
        drop = {"prop_dest_book_rate", "prop_site_book_rate",
                "site_dest_book_rate", "site_country_book_rate",
                "cpair_book_rate"}
        return [c for c in cols if c not in drop]

    if filter_name == "keep_raw_listwise_price":
        if audit_df is None:
            raise RuntimeError("feature_audit.csv required for keep_raw_listwise_price")
        keep_groups = {"raw", "listwise_within_query", "price", "quality"}
        keep = set(audit_df[audit_df["group"].isin(keep_groups)]["feature"].tolist())
        return [c for c in cols if c in keep]

    if filter_name == "keep_te_and_raw":
        # Drop only the V5 experimental interaction features.
        return [c for c in cols if c not in V5_EXPERIMENTAL_INTERACTIONS]

    if filter_name == "drop_competitor":
        return [c for c in cols if not c.startswith("comp")]

    if filter_name == "drop_visitor_history":
        drop = {"star_diff", "abs_star_diff",
                "price_diff_from_visitor_hist", "price_ratio_to_visitor_hist",
                "visitor_hist_starrating", "visitor_hist_adr_usd",
                "has_visitor_history"}
        return [c for c in cols if c not in drop]

    # ---- compound filters used by Group I (creative configs) -------------
    if filter_name == "triple_debias":
        # drop prop_avg_position AND all 5 cross-key TEs
        drop = {"prop_avg_position",
                "prop_dest_book_rate", "prop_site_book_rate",
                "site_dest_book_rate", "site_country_book_rate",
                "cpair_book_rate"}
        return [c for c in cols if c not in drop]

    if filter_name == "focus_on_signal":
        # drop bottom 30% by gain AND prop_avg_position (compound)
        if audit_df is None:
            raise RuntimeError("feature_audit.csv required for focus_on_signal")
        n_drop = int(len(audit_df) * 0.30)
        drop = set(audit_df.tail(n_drop)["feature"].tolist()) | {"prop_avg_position"}
        return [c for c in cols if c not in drop]

    raise ValueError(f"Unknown feature_filter: {filter_name}")


def apply_row_filter(train_split: pd.DataFrame, filter_name: str | None) -> np.ndarray:
    """Return boolean mask over train rows."""
    n = len(train_split)
    if filter_name is None:
        return np.ones(n, dtype=bool)

    if filter_name in ("drop_all_zero_q", "positive_q_only"):
        # In this dataset booking_bool=1 implies click_bool=1, so 'positive query'
        # and 'non-all-zero query' are the same set: srch_id with sum(click) > 0.
        eng = train_split.groupby("srch_id")["click_bool"].transform("sum")
        return (eng.values > 0)

    if filter_name == "random_only_0":
        return (train_split["random_bool"].values == 0)
    if filter_name == "random_only_1":
        return (train_split["random_bool"].values == 1)

    raise ValueError(f"Unknown row_filter: {filter_name}")


def compute_weights(train_split: pd.DataFrame, propensity: pd.Series,
                    mode: str) -> np.ndarray:
    """Per-row training weights according to WEIGHTING_PROFILES[mode]."""
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

    if profile.get("positive_only"):
        is_click = (train_split["click_bool"].values == 1)
        w = np.where(is_click, w, 1.0).astype(np.float32)

    clip_lo, clip_hi = profile.get("clip", (0.1, 10.0))
    w = np.clip(w, clip_lo, clip_hi).astype(np.float32)

    if (upw := profile.get("random_upweight", 1.0)) != 1.0:
        rand_mask = (train_split["random_bool"].values == 1)
        w[rand_mask] *= upw

    if (bqup := profile.get("booking_query_upweight", 1.0)) != 1.0:
        booking_q = train_split.groupby("srch_id")["booking_bool"].transform("max").values
        w = np.where(booking_q == 1, w * bqup, w).astype(np.float32)

    if (brup := profile.get("booked_row_upweight", 1.0)) != 1.0:
        booked = (train_split["booking_bool"].values == 1)
        w[booked] *= brup

    if (cwt := profile.get("clicked_row_upweight", 1.0)) != 1.0:
        # Click-only rows (click=1 but not booked)
        clicked_only = ((train_split["click_bool"].values == 1) &
                        (train_split["booking_bool"].values == 0))
        w[clicked_only] *= cwt

    if (azq := profile.get("all_zero_query_weight", 1.0)) != 1.0:
        eng = train_split.groupby("srch_id")["click_bool"].transform("sum").values
        in_zero_q = (eng == 0)
        w = np.where(in_zero_q, w * azq, w).astype(np.float32)

    return w


# ============================================================================
# Evaluation
# ============================================================================

def make_group_counts(df: pd.DataFrame) -> np.ndarray:
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df), "group counts mismatch"
    return g


def recall_at_k(df: pd.DataFrame, score_col: str, k: int = 5) -> float:
    results = []
    for _, grp in df.groupby("srch_id"):
        booked = grp[grp["relevance"] == 5]
        if len(booked) == 0:
            continue
        top_k_ids = grp.nlargest(k, score_col)["prop_id"].values
        results.append(int(booked["prop_id"].isin(top_k_ids).any()))
    return float(np.mean(results)) if results else 0.0


def mean_booked_rank(df: pd.DataFrame, score_col: str) -> float:
    ranks = []
    for _, grp in df.groupby("srch_id"):
        booked = grp[grp["relevance"] == 5]
        if len(booked) == 0:
            continue
        ranked = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        ranked["rank_pos"] = np.arange(1, len(ranked) + 1)
        ranks.extend(ranked[ranked["prop_id"].isin(booked["prop_id"])]["rank_pos"].tolist())
    return float(np.mean(ranks)) if ranks else float("nan")


def evaluate_ranking(df: pd.DataFrame, score_col: str = "pred_score") -> dict:
    return {
        "ndcg5": float(evaluate_ndcg(df, score_col=score_col, k=5)),
        "recall1": recall_at_k(df, score_col, k=1),
        "recall5": recall_at_k(df, score_col, k=5),
        "mean_booked_rank": mean_booked_rank(df, score_col),
    }


# ============================================================================
# Categorize / flag
# ============================================================================

def categorize(ndcg5: float, best_iter: int | None) -> str:
    flags = []
    if ndcg5 >= EXCELLENT_CANDIDATE_THRESHOLD:
        flags.append("EXCELLENT")
    elif ndcg5 >= STRONG_CANDIDATE_THRESHOLD:
        flags.append("STRONG")
    elif ndcg5 >= CANDIDATE_THRESHOLD:
        flags.append("CANDIDATE")
    elif ndcg5 < WEAK_THRESHOLD:
        flags.append("WEAK")
    else:
        flags.append("NEUTRAL")

    if ndcg5 >= POSSIBLE_OVERFIT_THRESHOLD:
        flags.append("POSSIBLE_OVERFIT")
    if best_iter is not None and best_iter < 100:
        flags.append("EARLY_STOP_SUSPICIOUS")
    if best_iter is not None and best_iter > 1800:
        flags.append("HIT_ROUND_BUDGET")
    return "|".join(flags)


# ============================================================================
# Master tracker append
# ============================================================================

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic CSV write: write to .tmp then os.replace. Prevents corruption
    if the process is killed mid-write (master trackers are precious — they
    already contain rows from previous sessions and are committed in git)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic on POSIX


def _append_row(csv_path: Path, row: dict, key_col: str) -> None:
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if str(row[key_col]) in existing[key_col].astype(str).values:
            return  # idempotent
        combined = pd.concat([existing, df_row], ignore_index=True)
    else:
        combined = df_row
    _atomic_write_csv(combined, csv_path)


def append_to_master_trackers(run_id: str, config: dict, result: dict,
                              status: str) -> None:
    """Append one row each to experiment_logs/{model_results,experiment_tracker}.csv.
    Phase + exp_id prefix are derived from run_id so smoke runs don't pollute
    the master tracker with OVERNIGHT_-prefixed rows."""
    EXPERIMENT_LOGS.mkdir(parents=True, exist_ok=True)
    mr_path = EXPERIMENT_LOGS / "model_results.csv"
    et_path = EXPERIMENT_LOGS / "experiment_tracker.csv"

    if run_id.startswith("smoke_"):
        phase = "SMOKE"
        exp_prefix = "SMOKE_"
    else:
        phase = "OVERNIGHT"
        exp_prefix = "OVERNIGHT_"

    model_id = f"{run_id}_{config['id']}"
    exp_id = f"{exp_prefix}{config['id']}"
    date = datetime.now(timezone.utc).isoformat()

    notes_short = (
        f"[{status}] {config['description']} | "
        f"weighting={config['weighting']} feat_filter={config['feature_filter']} "
        f"row_filter={config['row_filter']} obj={config['objective']} boost={config['boosting']}"
    )

    mr_row = {
        "model_id": model_id, "exp_id": exp_id, "date": date,
        "model_type": "lightgbm", "objective": config["objective"],
        "boosting": config["boosting"],
        "label_gain": config["label_gain"] or "",
        "ipw_mode": config["weighting"],
        "n_features": result["n_features"], "seed": config["seed"],
        "num_boost_round": result["num_boost_round"],
        "best_iter": result["best_iter"],
        "local_ndcg5": result["ndcg5"], "recall_at_5": result["recall5"],
        "mean_booked_rank": result["mean_booked_rank"],
        "artifact_path": f"artifacts/{run_id}/model_result_{config['id']}.json",
        "notes": notes_short,
    }

    et_row = {
        "exp_id": exp_id, "phase": phase,
        "name": f"{run_id}/{config['id']}",
        "date": date,
        "change_summary": config["description"],
        "baseline_ref": "V4_ENSEMBLE",
        "local_ndcg5": result["ndcg5"], "kaggle_ndcg5": "",
        "delta_local_vs_v4": result["ndcg5"] - V4_ENSEMBLE_LOCAL,
        "delta_kaggle_vs_v4": "",
        "kept": status,
        "notes": notes_short,
    }

    _append_row(mr_path, mr_row, "model_id")
    _append_row(et_path, et_row, "exp_id")


def append_error_row(run_id: str, config: dict, result: dict) -> None:
    art_dir = ROOT_DIR / "artifacts" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)
    csv_path = art_dir / "errors.csv"
    row = {
        "config_id": config["id"],
        "group": config["group"],
        "description": config["description"],
        "error": result["error"],
        "traceback_tail": result["traceback"][-1000:].replace("\n", " | "),
    }
    _append_row(csv_path, row, "config_id")


# ============================================================================
# Single-config runner
# ============================================================================

def run_one_config(config: dict,
                   train_feat: pd.DataFrame, val_feat: pd.DataFrame,
                   train_split: pd.DataFrame, val_split: pd.DataFrame,
                   propensity: pd.Series, audit_df: pd.DataFrame | None,
                   feature_cols_all: list[str],
                   run_id: str) -> dict:
    """Train + evaluate + persist one config. Always returns a dict.
    On error, returns {'error': str, 'traceback': str, ...}."""
    cid = config["id"]
    t0 = time.time()
    try:
        # ---- feature subset ----
        feature_cols = apply_feature_filter(feature_cols_all, config["feature_filter"], audit_df)
        feature_cols = [c for c in feature_cols if c in train_feat.columns and c in val_feat.columns]
        if not feature_cols:
            raise ValueError("feature filter produced empty column set")

        # ---- row filter ----
        row_mask = apply_row_filter(train_split, config["row_filter"])
        if not row_mask.all():
            train_feat_local = train_feat.loc[row_mask].reset_index(drop=True)
            train_split_local = train_split.loc[row_mask].reset_index(drop=True)
        else:
            train_feat_local = train_feat
            train_split_local = train_split

        # ---- weights ----
        weights = compute_weights(train_split_local, propensity, config["weighting"])

        # ---- labels ----
        tk = config["target_kind"]
        if tk == "remapped":
            remap = {0: 0, 1: 1, 5: 2}
            train_label = train_feat_local["relevance"].map(remap).astype(np.int32)
            val_label = val_feat["relevance"].map(remap).astype(np.int32)
        elif tk == "booking":
            train_label = train_split_local["booking_bool"].astype(np.int32).values
            val_label = val_split["booking_bool"].astype(np.int32).values
        elif tk == "click":
            train_label = train_split_local["click_bool"].astype(np.int32).values
            val_label = val_split["click_bool"].astype(np.int32).values
        elif tk == "rel_gt_zero":
            train_label = (train_feat_local["relevance"] > 0).astype(np.int32).values
            val_label = (val_feat["relevance"] > 0).astype(np.int32).values
        else:
            raise ValueError(f"Unknown target_kind: {tk}")

        # ---- params ----
        params = BASE_PARAMS.copy()
        params["seed"] = config["seed"]
        params["objective"] = config["objective"]
        params["boosting_type"] = config["boosting"]
        if config["label_gain"]:
            params["label_gain"] = config["label_gain"]
        params.update(config["param_overrides"])

        is_ranking = config["objective"] in {"lambdarank", "rank_xendcg"}
        if not is_ranking:
            params.pop("eval_at", None)
            params.pop("label_gain", None)
            # binary metric is set via param_overrides (auc)

        # ---- groups (ranking only) ----
        train_groups = make_group_counts(train_feat_local) if is_ranking else None
        val_groups = make_group_counts(val_feat) if is_ranking else None

        # ---- fresh Dataset, V4-style (no .construct(), no free_raw_data) ----
        if is_ranking:
            ds_train = lgb.Dataset(
                train_feat_local[feature_cols], label=train_label,
                group=train_groups, weight=weights,
            )
            ds_val = lgb.Dataset(
                val_feat[feature_cols], label=val_label,
                group=val_groups, reference=ds_train,
            )
        else:
            ds_train = lgb.Dataset(
                train_feat_local[feature_cols], label=train_label,
                weight=weights,
            )
            ds_val = lgb.Dataset(
                val_feat[feature_cols], label=val_label, reference=ds_train,
            )

        nbr = config["num_boost_round"]
        es = config["early_stopping"]
        log_interval = max(int(nbr / 10), 50)

        callbacks = [lgb.log_evaluation(log_interval)]
        if es:
            callbacks = [lgb.early_stopping(es)] + callbacks

        t_tr = time.time()
        model = lgb.train(params, ds_train, num_boost_round=nbr,
                          valid_sets=[ds_val], callbacks=callbacks)
        train_elapsed = time.time() - t_tr

        # ---- predict + evaluate (always NDCG@5 against original relevance) ----
        pred = model.predict(val_feat[feature_cols])
        val_eval = val_feat[["srch_id", "prop_id", "relevance"]].copy()
        val_eval["pred_score"] = pred
        metrics = evaluate_ranking(val_eval)

        # ---- persist artifacts ----
        save_model_artifacts(run_id, cid, model, pred, feature_cols, metrics, params)

        best_iter = int(getattr(model, "best_iteration", 0) or nbr)
        result = {
            "config_id": cid, "group": config["group"],
            "label_gain": config["label_gain"], "seed": config["seed"],
            "weighting": config["weighting"],
            "feature_filter": config["feature_filter"],
            "row_filter": config["row_filter"],
            "objective": config["objective"], "boosting": config["boosting"],
            "target_kind": tk,
            "n_features": len(feature_cols),
            "num_boost_round": nbr, "best_iter": best_iter,
            "ndcg5": metrics["ndcg5"], "recall1": metrics["recall1"],
            "recall5": metrics["recall5"],
            "mean_booked_rank": metrics["mean_booked_rank"],
            "train_time_sec": train_elapsed,
            "elapsed_sec": time.time() - t0,
            "description": config["description"],
        }

        # cleanup
        del ds_train, ds_val, model
        gc.collect()
        return result

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "config_id": cid,
            "group": config.get("group", "?"),
            "description": config.get("description", ""),
            "error": str(e),
            "traceback": tb,
            "elapsed_sec": time.time() - t0,
        }
    finally:
        gc.collect()


# ============================================================================
# Summary writer
# ============================================================================

def write_overnight_summary(run_id: str, configs: list[dict], results: list[dict],
                            errors: list[dict], skipped: list[dict],
                            setup_seconds: float, total_seconds: float) -> None:
    EXPERIMENT_LOGS.mkdir(parents=True, exist_ok=True)
    summary_path = EXPERIMENT_LOGS / "overnight_summary.md"
    candidates_path = EXPERIMENT_LOGS / "candidate_models_for_ensemble.csv"

    if results:
        df = pd.DataFrame(results)
        df["status"] = [categorize(r["ndcg5"], r.get("best_iter")) for _, r in df.iterrows()]
    else:
        df = pd.DataFrame(columns=[
            "config_id", "group", "label_gain", "weighting", "seed",
            "feature_filter", "row_filter", "objective", "n_features",
            "best_iter", "ndcg5", "recall5", "mean_booked_rank", "status",
            "description",
        ])

    # --- candidate CSV ---
    cdf = df.copy()
    if not cdf.empty:
        cdf = cdf.sort_values("ndcg5", ascending=False).reset_index(drop=True)
        cdf["artifact_path"] = [
            f"artifacts/{run_id}/model_result_{r['config_id']}.json" for _, r in cdf.iterrows()
        ]
    cdf.to_csv(candidates_path, index=False)

    # --- markdown ---
    lines: list[str] = []
    lines.append(f"# Overnight run summary — `{run_id}`")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append(f"**Baselines.** V4 ensemble Kaggle = `{V4_KAGGLE}`, local = `{V4_ENSEMBLE_LOCAL}`. "
                 f"V4 anchor (single, lg=0/1/15) local = `{ANCHOR_NDCG}`. "
                 f"Phase 2 winner (lg=0/2/15) local = `{PHASE2_BEST_NDCG}`, Kaggle = `0.41639`.")
    lines.append("")
    lines.append(f"**Wall-clock.** setup = {setup_seconds/60:.1f} min, total = {total_seconds/60:.1f} min.")
    lines.append("")
    lines.append(f"**Configs.** {len(configs)} total, {len(results)} succeeded, "
                 f"{len(errors)} errored, {len(skipped)} skipped.")
    lines.append("")

    # group breakdown
    by_group: dict[str, dict] = {}
    for c in configs:
        d = by_group.setdefault(c["group"], dict(total=0, succeeded=0, errored=0, skipped=0))
        d["total"] += 1
        if c["skip_reason"]:
            d["skipped"] += 1
    for r in results:
        by_group.setdefault(r["group"], dict(total=0, succeeded=0, errored=0, skipped=0))["succeeded"] += 1
    for e in errors:
        for c in configs:
            if c["id"] == e["config_id"]:
                by_group[c["group"]]["errored"] += 1
                break

    lines.append("## Group breakdown")
    lines.append("")
    lines.append("| Group | Total | Succeeded | Errored | Skipped |")
    lines.append("|---|---|---|---|---|")
    for g in sorted(by_group):
        d = by_group[g]
        lines.append(f"| {g} | {d['total']} | {d['succeeded']} | {d['errored']} | {d['skipped']} |")
    lines.append("")

    # status counts
    n_excellent = (df["status"].str.contains("EXCELLENT")).sum() if not df.empty else 0
    n_strong = (df["status"].str.contains("STRONG") & ~df["status"].str.contains("EXCELLENT")).sum() if not df.empty else 0
    n_candidate = (df["status"].str.contains("CANDIDATE") & ~df["status"].str.contains("STRONG|EXCELLENT")).sum() if not df.empty else 0
    n_overfit = (df["status"].str.contains("POSSIBLE_OVERFIT")).sum() if not df.empty else 0
    n_weak = (df["status"].str.contains("WEAK")).sum() if not df.empty else 0
    n_early = (df["status"].str.contains("EARLY_STOP_SUSPICIOUS")).sum() if not df.empty else 0
    n_budget = (df["status"].str.contains("HIT_ROUND_BUDGET")).sum() if not df.empty else 0

    lines.append("## Candidate categorization")
    lines.append("")
    lines.append(f"- EXCELLENT (NDCG@5 ≥ {EXCELLENT_CANDIDATE_THRESHOLD}): **{n_excellent}**")
    lines.append(f"- STRONG    (NDCG@5 ≥ {STRONG_CANDIDATE_THRESHOLD}): **{n_strong}**")
    lines.append(f"- CANDIDATE (NDCG@5 ≥ {CANDIDATE_THRESHOLD}): **{n_candidate}**")
    lines.append(f"- POSSIBLE_OVERFIT (NDCG@5 ≥ {POSSIBLE_OVERFIT_THRESHOLD}): {n_overfit}")
    lines.append(f"- WEAK (NDCG@5 < {WEAK_THRESHOLD}): {n_weak}")
    lines.append(f"- EARLY_STOP_SUSPICIOUS (best_iter < 100): {n_early}")
    lines.append(f"- HIT_ROUND_BUDGET (best_iter > 1800): {n_budget}")
    lines.append("")

    # top 20 by NDCG@5
    if not df.empty:
        top = df.sort_values("ndcg5", ascending=False).head(20)
        lines.append("## Top 20 by validation NDCG@5")
        lines.append("")
        lines.append("| Rank | Config | Group | label_gain | weighting | feat_filter | row_filter | seed | n_feat | best_iter | NDCG@5 | Recall@5 | MBR | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(top.iterrows()):
            lines.append(
                f"| {i+1} | {r['config_id']} | {r['group']} | {r['label_gain']} | "
                f"{r['weighting']} | {r.get('feature_filter') or '—'} | "
                f"{r.get('row_filter') or '—'} | {r['seed']} | {r['n_features']} | "
                f"{r['best_iter']} | {r['ndcg5']:.5f} | {r['recall5']:.4f} | "
                f"{r['mean_booked_rank']:.2f} | {r['status']} |"
            )
        lines.append("")

        # top 10 by recall5
        top_r = df.sort_values("recall5", ascending=False).head(10)
        lines.append("## Top 10 by Recall@5")
        lines.append("")
        lines.append("| Config | Group | label_gain | weighting | Recall@5 | NDCG@5 | Status |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in top_r.iterrows():
            lines.append(
                f"| {r['config_id']} | {r['group']} | {r['label_gain']} | "
                f"{r['weighting']} | {r['recall5']:.4f} | {r['ndcg5']:.5f} | {r['status']} |"
            )
        lines.append("")

        # top 10 by lowest MBR
        top_m = df.sort_values("mean_booked_rank", ascending=True).head(10)
        lines.append("## Top 10 by lowest Mean Booked Rank")
        lines.append("")
        lines.append("| Config | Group | label_gain | weighting | MBR | NDCG@5 | Status |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in top_m.iterrows():
            lines.append(
                f"| {r['config_id']} | {r['group']} | {r['label_gain']} | "
                f"{r['weighting']} | {r['mean_booked_rank']:.2f} | {r['ndcg5']:.5f} | {r['status']} |"
            )
        lines.append("")

        # diversity candidates: top-quartile Recall@5 or low MBR even if NDCG < CANDIDATE
        q75_recall = df["recall5"].quantile(0.75)
        q25_mbr = df["mean_booked_rank"].quantile(0.25)
        diversity = df[
            ((df["recall5"] >= q75_recall) | (df["mean_booked_rank"] <= q25_mbr))
            & (df["ndcg5"] < CANDIDATE_THRESHOLD)
        ].sort_values("recall5", ascending=False).head(10)
        if len(diversity) > 0:
            lines.append("## Diversity picks (high Recall@5 or low MBR but NDCG<candidate)")
            lines.append("")
            lines.append("| Config | Group | NDCG@5 | Recall@5 | MBR | Reason |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in diversity.iterrows():
                reason = []
                if r["recall5"] >= q75_recall:
                    reason.append(f"Recall@5≥Q75 ({q75_recall:.4f})")
                if r["mean_booked_rank"] <= q25_mbr:
                    reason.append(f"MBR≤Q25 ({q25_mbr:.2f})")
                lines.append(
                    f"| {r['config_id']} | {r['group']} | {r['ndcg5']:.5f} | "
                    f"{r['recall5']:.4f} | {r['mean_booked_rank']:.2f} | {'; '.join(reason)} |"
                )
            lines.append("")

    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- **{e['config_id']}** ({e.get('group', '?')}): "
                         f"`{e['error']}` — {e.get('description', '')}")
        lines.append("")

    if skipped:
        lines.append("## Skipped configs")
        lines.append("")
        for s in skipped:
            lines.append(f"- **{s['id']}** ({s['group']}): {s['description']}")
            lines.append(f"  - Reason: {s['skip_reason']}")
        lines.append("")

    lines.append("## Artifact paths")
    lines.append("")
    lines.append(f"- Per-run artifacts: `artifacts/{run_id}/`")
    lines.append(f"- External copy:    `/home/ubuntu/experiment_artifacts/{run_id}/`")
    lines.append(f"- Models:           `models/{run_id}/`")
    lines.append(f"- Master trackers:  `experiment_logs/model_results.csv`, "
                 f"`experiment_logs/experiment_tracker.csv`")
    lines.append(f"- This summary:     `experiment_logs/overnight_summary.md`")
    lines.append(f"- Candidate list:   `experiment_logs/candidate_models_for_ensemble.csv`")
    lines.append("")

    lines.append("## How to inspect tomorrow")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1) Quick overview")
    lines.append("less experiment_logs/overnight_summary.md")
    lines.append("")
    lines.append("# 2) Top 20 by NDCG@5 (sortable CSV)")
    lines.append(("uv run --no-sync python -c \"import pandas as pd; "
                  "df=pd.read_csv('experiment_logs/candidate_models_for_ensemble.csv'); "
                  "print(df.sort_values('ndcg5', ascending=False).head(20)"
                  "[['config_id','group','label_gain','weighting','feature_filter','row_filter',"
                  "'seed','n_features','best_iter','ndcg5','recall5','mean_booked_rank','status']]"
                  ".to_string(index=False))\""))
    lines.append("")
    lines.append("# 3) Just CANDIDATE/STRONG/EXCELLENT rows")
    lines.append(("uv run --no-sync python -c \"import pandas as pd; "
                  "df=pd.read_csv('experiment_logs/candidate_models_for_ensemble.csv'); "
                  "df=df[df['status'].str.contains('CANDIDATE|STRONG|EXCELLENT', na=False)]; "
                  "print(df.sort_values('ndcg5', ascending=False)"
                  "[['config_id','group','label_gain','weighting','ndcg5','status']].to_string(index=False))\""))
    lines.append("")
    lines.append("# 4) Errors (if any)")
    lines.append(f"test -f artifacts/{run_id}/errors.csv && cat artifacts/{run_id}/errors.csv")
    lines.append("```")

    summary_path.write_text("\n".join(lines))
    log(f"Wrote {summary_path}")
    log(f"Wrote {candidates_path}")


# ============================================================================
# Mode: dry-run
# ============================================================================

def mode_dry_run() -> int:
    configs = build_all_configs()
    runnable = [c for c in configs if not c["skip_reason"]]
    skipped = [c for c in configs if c["skip_reason"]]

    print("=" * 100)
    print("DRY RUN — overnight experiment plan")
    print("=" * 100)
    print(f"Total configs   : {len(configs)}")
    print(f"  Runnable      : {len(runnable)}")
    print(f"  Skipped       : {len(skipped)}")
    print()

    # group breakdown
    groups = sorted({c["group"] for c in configs})
    print("By group:")
    for g in groups:
        gc_list = [c for c in configs if c["group"] == g]
        gskip = [c for c in gc_list if c["skip_reason"]]
        print(f"  {g}: {len(gc_list)} total ({len(gskip)} skipped)")
    print()

    print("Run-id template      : overnight_<YYYYMMDD_HHMMSS>")
    print("Artifact root        : artifacts/<run-id>/")
    print("External archive     : /home/ubuntu/experiment_artifacts/<run-id>/")
    print("Models               : models/<run-id>/")
    print("Master trackers      : experiment_logs/model_results.csv, experiment_tracker.csv")
    print("Summary              : experiment_logs/overnight_summary.md")
    print("Candidate list       : experiment_logs/candidate_models_for_ensemble.csv")
    print()

    print("Config catalog:")
    hdr = (f"{'id':<5} {'grp':<4} {'lg':<8} {'seed':<5} {'weighting':<16} "
           f"{'feat_filter':<26} {'row_filter':<20} {'obj':<13} {'boost':<6} "
           f"{'rounds':<7} note")
    print(hdr)
    print("-" * len(hdr))
    for c in configs:
        lg = c["label_gain"] or "—"
        ff = c["feature_filter"] or "—"
        rf = c["row_filter"] or "—"
        note = c["description"][:50]
        if c["skip_reason"]:
            note = f"SKIP — {note[:44]}"
        print(f"{c['id']:<5} {c['group']:<4} {lg:<8} {c['seed']:<5} "
              f"{c['weighting']:<16} {ff:<26} {rf:<20} "
              f"{c['objective']:<13} {c['boosting']:<6} {c['num_boost_round']:<7} {note}")
    print()

    if skipped:
        print("Skipped configs (with reason):")
        for c in skipped:
            print(f"  {c['id']} ({c['group']}): {c['description']}")
            for line in (c["skip_reason"] or "").split("\n"):
                print(f"     • {line}")
        print()

    runnable_minutes = len(runnable) * 3.5
    print(f"Estimated wall-clock : ~{runnable_minutes:.0f} min ({runnable_minutes/60:.1f} h) for "
          f"{len(runnable)} configs at ~3.5 min each, plus ~3 min one-time feature build.")
    print()
    print("Output paths confirmed. To execute: run without --dry-run.")
    return 0


# ============================================================================
# Mode: smoke-test
# ============================================================================

def mode_smoke_test() -> int:
    run_id = f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _model_dir, art_dir = run_dirs(run_id)

    log("=" * 80)
    log(f"SMOKE TEST — run_id={run_id}")
    log("=" * 80)

    if not FEATURE_AUDIT_CSV.exists():
        log(f"WARN: {FEATURE_AUDIT_CSV} not found; feature filters relying on the audit will skip in smoke.")
        audit_df = None
    else:
        audit_df = pd.read_csv(FEATURE_AUDIT_CSV).sort_values(
            "importance_gain", ascending=False
        ).reset_index(drop=True)

    log(f"Loading {SMOKE_SAMPLE_FRAC*100:.0f}% sample…")
    train_raw = load_train(sample_frac=SMOKE_SAMPLE_FRAC, random_state=42)
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  rows={len(train_raw):,}, srch_ids={train_raw['srch_id'].nunique():,}")

    log("Propensity + split…")
    propensity = compute_position_propensity(train_raw)
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    log(f"  train={len(train_split):,}, val={len(val_split):,}")

    log("Build features…")
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols_all = get_feature_columns(train_feat)
    feature_cols_all = [c for c in feature_cols_all if c in val_feat.columns]
    log(f"  features built: {len(feature_cols_all)}")

    save_feature_cols(run_id, feature_cols_all)
    save_val_meta(run_id, val_feat)
    save_run_config(run_id, {"phase": "SMOKE_TEST", "run_id": run_id,
                             "sample_frac": SMOKE_SAMPLE_FRAC,
                             "num_boost_round": NUM_BOOST_ROUND_SMOKE})
    save_git_commit(run_id)

    del train_raw
    gc.collect()

    smoke_configs = [
        _base_config(
            id="SMOKE_BASE", group="SMOKE",
            label_gain="0,2,15", weighting="ipw_default",
            num_boost_round=NUM_BOOST_ROUND_SMOKE,
            early_stopping=EARLY_STOPPING_SMOKE,
            description="Smoke: lg=0/2/15, ipw_default, full features",
        ),
        _base_config(
            id="SMOKE_NOIPW", group="SMOKE",
            label_gain="0,2,15", weighting="no_ipw",
            feature_filter="drop_prop_avg_position",
            num_boost_round=NUM_BOOST_ROUND_SMOKE,
            early_stopping=EARLY_STOPPING_SMOKE,
            description="Smoke: lg=0/2/15, no_ipw, drop prop_avg_position",
        ),
    ]

    log("")
    log("Running 2 smoke configs…")
    for cfg in smoke_configs:
        log(f"  [{cfg['id']}] {cfg['description']}")
        result = run_one_config(cfg, train_feat, val_feat, train_split, val_split,
                                propensity, audit_df, feature_cols_all, run_id)
        if "error" in result:
            log(f"  FAIL: {cfg['id']} errored: {result['error']}")
            log(f"  Traceback:\n{result['traceback']}")
            return 1
        log(f"  PASS: {cfg['id']} NDCG@5={result['ndcg5']:.5f} "
            f"R@5={result['recall5']:.4f} best_iter={result['best_iter']} "
            f"({result['train_time_sec']:.0f}s)")
        status = categorize(result["ndcg5"], result["best_iter"])
        append_to_master_trackers(run_id, cfg, result, status)

    # Intentional-failure check (catches a bad weighting mode at the right place)
    log("")
    log("Intentional-failure check (weighting='INVALID_MODE')…")
    bad_cfg = _base_config(
        id="SMOKE_FAIL", group="SMOKE",
        label_gain="0,2,15", weighting="INVALID_MODE",
        num_boost_round=NUM_BOOST_ROUND_SMOKE,
        early_stopping=EARLY_STOPPING_SMOKE,
        description="Smoke: should fail in compute_weights",
    )
    result = run_one_config(bad_cfg, train_feat, val_feat, train_split, val_split,
                            propensity, audit_df, feature_cols_all, run_id)
    if "error" not in result:
        log("  FAIL: bad config did NOT error — error handling broken")
        return 1
    log(f"  PASS: caught error: {result['error']}")
    append_error_row(run_id, bad_cfg, result)

    # Verify trackers got the smoke rows
    mr = pd.read_csv(EXPERIMENT_LOGS / "model_results.csv") if (EXPERIMENT_LOGS / "model_results.csv").exists() else pd.DataFrame()
    if mr.empty or not mr["model_id"].astype(str).str.startswith(run_id).any():
        log("  FAIL: CSV trackers did not receive smoke rows")
        return 1
    log("  PASS: CSV trackers received smoke rows")

    # Verify per-config artifacts exist
    for cfg in smoke_configs:
        result_json = art_dir / f"model_result_{cfg['id']}.json"
        if not result_json.exists():
            log(f"  FAIL: artifact missing {result_json}")
            return 1
    log("  PASS: per-config artifacts present (model_result_*.json)")

    log("")
    log("=== SMOKE TEST PASSED ===")
    log(f"Smoke artifacts: artifacts/{run_id}/")
    return 0


# ============================================================================
# Mode: full overnight run
# ============================================================================

def mode_full_run() -> int:
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"overnight_{timestamp}"
    model_dir, art_dir = run_dirs(run_id)

    log("=" * 80)
    log(f"FULL OVERNIGHT RUN — run_id={run_id}")
    log("=" * 80)

    # ---- audit ----
    if not FEATURE_AUDIT_CSV.exists():
        log(f"FATAL: {FEATURE_AUDIT_CSV} missing. Required for top_N / drop_bottom feature filters.")
        return 2
    audit_df = pd.read_csv(FEATURE_AUDIT_CSV).sort_values(
        "importance_gain", ascending=False
    ).reset_index(drop=True)

    # ---- configs ----
    configs = build_all_configs()
    n_runnable = sum(1 for c in configs if not c["skip_reason"])
    log(f"Configs: {len(configs)} total, {n_runnable} runnable, "
        f"{len(configs)-n_runnable} skipped.")

    # ---- run-level metadata ----
    save_run_config(run_id, {
        "phase": "OVERNIGHT_CANDIDATE_GENERATION",
        "run_id": run_id,
        "date": datetime.now(timezone.utc).isoformat(),
        "v4_ensemble_kaggle": V4_KAGGLE,
        "v4_ensemble_local": V4_ENSEMBLE_LOCAL,
        "anchor_local": ANCHOR_NDCG,
        "anchor_seed": ANCHOR_SEED,
        "base_params": BASE_PARAMS,
        "weighting_profiles": WEIGHTING_PROFILES,
        "candidate_thresholds": {
            "candidate": CANDIDATE_THRESHOLD,
            "strong": STRONG_CANDIDATE_THRESHOLD,
            "excellent": EXCELLENT_CANDIDATE_THRESHOLD,
            "possible_overfit": POSSIBLE_OVERFIT_THRESHOLD,
            "weak": WEAK_THRESHOLD,
        },
        "n_configs_total": len(configs),
        "n_configs_runnable": n_runnable,
        "rules": [
            "Single models only",
            "V4 full feature set unless feature_filter is set",
            "V4-style Dataset pattern (fresh per config; no .construct(); no free_raw_data)",
            "Fixed V4 split (val_frac=0.1, random_state=42)",
            "Resumable: skip if model_result_<id>.json already exists",
            "Catch exceptions per config; log and continue",
            "No submissions; no full-data retrain; no ensembles",
        ],
    })
    save_git_commit(run_id)

    # ---- shared data + features (build once) ----
    log("[setup] Load train / propensity / split / features (one-time)…")
    t_setup = time.time()
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  rows={len(train_raw):,}")

    propensity = compute_position_propensity(train_raw)
    train_split, val_split = split_val(train_raw, val_frac=0.1)
    train_split = train_split.sort_values("srch_id").reset_index(drop=True)
    val_split = val_split.sort_values("srch_id").reset_index(drop=True)
    log(f"  train={len(train_split):,}  val={len(val_split):,}")

    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols_all = get_feature_columns(train_feat)
    feature_cols_all = [c for c in feature_cols_all if c in val_feat.columns]
    leaked = set(feature_cols_all) & FORBIDDEN_FEATURES
    assert not leaked, f"Forbidden cols leaked: {leaked}"
    log(f"  features built: {len(feature_cols_all)}")

    save_feature_cols(run_id, feature_cols_all)
    save_val_meta(run_id, val_feat)

    del train_raw
    gc.collect()
    setup_seconds = time.time() - t_setup
    log(f"[setup] Done in {setup_seconds/60:.1f} min.")

    # ---- main loop ----
    results: list[dict] = []
    errors: list[dict] = []
    skipped: list[dict] = []

    for i, cfg in enumerate(configs, 1):
        cid = cfg["id"]

        if cfg["skip_reason"]:
            log(f"[{i}/{len(configs)}] {cid} ({cfg['group']}): SKIP — {cfg['skip_reason'][:90]}")
            skipped.append(cfg)
            continue

        result_path = art_dir / f"model_result_{cid}.json"
        if result_path.exists():
            log(f"[{i}/{len(configs)}] {cid}: SKIP (already completed; resume)")
            try:
                with open(result_path) as f:
                    prev = json.load(f)
                results.append({
                    "config_id": cid, "group": cfg["group"],
                    "label_gain": cfg["label_gain"], "seed": cfg["seed"],
                    "weighting": cfg["weighting"],
                    "feature_filter": cfg["feature_filter"],
                    "row_filter": cfg["row_filter"],
                    "objective": cfg["objective"], "boosting": cfg["boosting"],
                    "target_kind": cfg["target_kind"],
                    "n_features": prev.get("n_features"),
                    "num_boost_round": cfg["num_boost_round"],
                    "best_iter": prev.get("best_iter"),
                    "ndcg5": prev.get("ndcg5"),
                    "recall1": prev.get("recall1"),
                    "recall5": prev.get("recall5"),
                    "mean_booked_rank": prev.get("mean_booked_rank"),
                    "description": cfg["description"],
                    "train_time_sec": None, "elapsed_sec": None,
                })
            except Exception:
                pass
            continue

        log(f"[{i}/{len(configs)}] {cid} ({cfg['group']}): {cfg['description']}")
        result = run_one_config(
            cfg, train_feat, val_feat, train_split, val_split,
            propensity, audit_df, feature_cols_all, run_id,
        )

        if "error" in result:
            log(f"  ERROR: {result['error']}")
            errors.append(result)
            append_error_row(run_id, cfg, result)
            continue

        status = categorize(result["ndcg5"], result["best_iter"])
        log(f"  NDCG@5={result['ndcg5']:.5f}  R@5={result['recall5']:.4f}  "
            f"MBR={result['mean_booked_rank']:.2f}  best_iter={result['best_iter']}  "
            f"({result['train_time_sec']/60:.1f} min)  status={status}")
        result["status"] = status
        results.append(result)
        append_to_master_trackers(run_id, cfg, result, status)

    # ---- final summaries (wrapped so a writer crash can't lose results) ----
    total_seconds = time.time() - t_start
    try:
        write_overnight_summary(run_id, configs, results, errors, skipped,
                                setup_seconds, total_seconds)
    except Exception as e:
        log(f"WARN: write_overnight_summary failed — {e}")
        # last-ditch: dump raw results to a fallback CSV so nothing is lost
        try:
            fb = ROOT_DIR / "artifacts" / run_id / "results_fallback.json"
            fb.write_text(json.dumps({
                "results": results, "errors": [
                    {k: v for k, v in e.items() if k != "traceback"} for e in errors
                ], "skipped": [c["id"] for c in skipped],
            }, default=str, indent=2))
            log(f"  fallback dump → {fb}")
        except Exception as inner:
            log(f"  fallback dump also failed: {inner}")

    # ---- copy to external archive ----
    external_dir = EXTERNAL_COPY_ROOT / run_id
    try:
        if external_dir.exists():
            shutil.rmtree(external_dir)
        external_dir.mkdir(parents=True, exist_ok=True)
        for item in art_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, external_dir / item.name)
        # also copy models dir
        ext_models = external_dir / "models"
        ext_models.mkdir(exist_ok=True)
        for item in model_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, ext_models / item.name)
        # and the master trackers / summary
        for src in [
            EXPERIMENT_LOGS / "overnight_summary.md",
            EXPERIMENT_LOGS / "candidate_models_for_ensemble.csv",
        ]:
            if src.exists():
                shutil.copy2(src, external_dir / src.name)
        log(f"External copy: {external_dir}")
    except Exception as e:
        log(f"WARN: failed to copy to external archive — {e}")

    # ---- final banner ----
    log("")
    log("=" * 80)
    log(f"OVERNIGHT DONE  —  run_id = {run_id}")
    log(f"  succeeded: {len(results)}  errored: {len(errors)}  skipped: {len(skipped)}")
    log(f"  total wall-clock: {total_seconds/60:.1f} min")
    log(f"  artifacts:        artifacts/{run_id}/")
    log(f"  models:           models/{run_id}/")
    log(f"  external copy:    {external_dir}")
    log(f"  master trackers:  experiment_logs/model_results.csv, experiment_tracker.csv")
    log(f"  summary:          experiment_logs/overnight_summary.md")
    log(f"  candidates:       experiment_logs/candidate_models_for_ensemble.csv")
    log("=" * 80)
    return 0


# ============================================================================
# Entry point
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + paths; no data load, no training.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run 2 tiny configs on a 1%% srch_id sample + error-handling check.")
    args = ap.parse_args()

    if args.dry_run:
        return mode_dry_run()
    if args.smoke_test:
        return mode_smoke_test()
    return mode_full_run()


if __name__ == "__main__":
    sys.exit(main())
