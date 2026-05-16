"""
V5 gap diagnostic.

V5 robust ensemble: local NDCG@5=0.42633, Kaggle=0.41943 → gap=-0.00690.
V4 ensemble:         local NDCG@5=0.42512, Kaggle=0.42021 → gap=-0.00491.
V5 improved local but worsened Kaggle: train→test mismatch / val overfit.

Phase 1 — quick (default):
  1. Adversarial validation (train vs test) on full feature set.
  2. V5 vs V4 ensemble ranking disagreement + intra-V5 member disagreement.
     (V4 = Kaggle 0.42021 reference. Submission CSV must be placed at
      submissions/submission_v4_ensemble.csv — see "Required files" below.)
  3. Position-bias correlation on train.
  4. Cold-start enumeration (test items unseen in train).

Phase 2 — temporal (only with --temporal or --all):
  Retrain V4 anchor (lg=0,1,15) + B3 (lg=0,2,15, ipw_clip3) on temporal split.

Phase 3 — final report:
  diagnostics/v5_gap_<TS>/README.md.

CLI:
  uv run python diagnose_v5_gap.py --quick
  uv run python diagnose_v5_gap.py --temporal
  uv run python diagnose_v5_gap.py --all
  uv run python diagnose_v5_gap.py --plan         # print plan and exit, no work

Constraints (per spec):
  No submission generation. No ensemble tuning. No retraining beyond
  the two temporal-split models in Phase 2.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Constants from prior work
V4_LOCAL = 0.42512
V4_KAGGLE = 0.42021
V5_LOCAL = 0.42633
V5_KAGGLE = 0.41943
V4_2_KAGGLE = 0.41639
ANCHOR_LOCAL = 0.42191

OVERNIGHT_RUN = ROOT / "artifacts" / "overnight_20260516_003443"
V5_SUBMIT_DIR = ROOT / "artifacts" / "v5_ensemble_submit"
V5_SUBMISSION_CSV = ROOT / "submissions" / "submission_v5_ensemble_20260516_094741.csv"
# V4 ensemble (the actual Kaggle 0.42021 reference).
V4_SUBMISSION_CSV = ROOT / "submissions" / "submission_v4_20260515_151132.csv"
# V4.2 single (0.41639) — kept as secondary reference only (it was a stress-test, not
# the V4 production model). Skipped if not present.
V4_2_SUBMISSION_CSV = ROOT / "submissions" / "submission_phase2_best_20260515_225726.csv"
FEATURE_AUDIT_CSV = ROOT / "experiment_logs" / "feature_audit.csv"

V5_MEMBERS = ["B3", "F13", "D4", "A1", "B10", "E4", "E3"]

# Feature groups for adversarial drift analysis
FEATURE_GROUPS = [
    "raw", "listwise_within_query",
    "hotel_agg_TE_single", "hotel_agg_TE_cross",
    "hotel_agg_position", "hotel_agg_count",
    "hotel_agg_price_stat", "hotel_agg_dest_stat",
    "price", "competitor", "visitor", "interaction",
    "temporal", "missing_flag",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def short_log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================
# Plan (printed for --plan, and at start of any run)
# ============================================================================
PLAN_HEADER = f"""\
diagnose_v5_gap.py — execution plan
====================================

Required files:
  data/training_set_VU_DM.csv                         (4.96M rows)
  data/test_set_VU_DM.csv                             (4.96M rows)
  experiment_logs/feature_audit.csv                   (143 features w/ ranker importance)
  experiment_logs/model_results.csv                   (V4/V5 ranker val metrics)
  submissions/submission_v5_ensemble_20260516_094741.csv      (V5 Kaggle 0.41943) [present]
  submissions/submission_v4_20260515_151132.csv               (V4 Kaggle 0.42021) [present]
  artifacts/v5_ensemble_submit/test_pred_<id>.npy             (7 V5 member raw test scores)

Optional / fallback:
  submissions/submission_phase2_best_20260515_225726.csv     (V4.2 single Kaggle 0.41639) [present]
  → If V4 ensemble CSV is missing, diagnostic falls back to V4.2 but flags the limitation.

NOT available (will state clearly in output):
  • V4 ensemble raw test scores (only the submission CSV ranking is usable).
  • Cached train/test feature matrices → features REBUILT via src.features.build_features.

Writes (under diagnostics/v5_gap_<TS>/):
  Phase 1 (--quick, default):
    adversarial_auc.json
    adversarial_feature_importance.csv          (top 50 drift features)
    feature_risk_table.csv                      (ranker × adversarial × risk_label × action)
    feature_risk_by_group.csv                   (per-group adversarial vs ranker share)
    v4_v5_disagreement_by_query.csv             (per-srch_id: top5 overlap, rank diff, ranks)
    v4_v5_disagreement_segments.csv             (segmented by site, dest, country, window, …)
    v5_internal_disagreement.csv                (intra-V5 member disagreement)
    v5_vs_v42_disagreement_by_query.csv         (secondary reference if V4.2 present)
    position_correlation_features.csv           (top 30 features correlated with position)
    position_leaky_suspects.csv                 (high ranker × adversarial × position-corr)
    cold_start_summary.csv                      (% unseen rates: prop, dest, (prop,dest), (prop,site))
    cold_start_by_disagreement_segment.csv      (cold-start rate in high-disagreement vs normal)
    README.md                                   (executive summary + action table)
  Phase 2 (--temporal or --all):
    temporal_split_results.csv                  (V4 anchor + B3 NDCG@5 on temporal val)
    model_V4_anchor_temporal.txt                (booster)
    model_B3_temporal.txt                       (booster)

Runtime estimates (Linux/AWS, 122 GB RAM, single-process):
  --quick    ≈ 10–15 min   (feature build ≈ 3–4 min train + 3–4 min test;
                            adversarial LGBM ≈ 3 min; everything else < 2 min)
  --temporal ≈ 10–15 min   (feature build ≈ 3–4 min on full train; 2× LGBM ≈ 6–8 min)
  --all      ≈ 25–30 min   (features built once and reused across phases)

Commands (run from project root):
  uv run python diagnose_v5_gap.py --quick      # Phase 1 only (recommended first)
  uv run python diagnose_v5_gap.py --temporal   # Phase 2 only
  uv run python diagnose_v5_gap.py --all        # both phases + README
  uv run python diagnose_v5_gap.py --plan       # print this plan and exit

Hard constraints honored by this script:
  - No new submissions written.
  - No ensemble tuning.
  - Retraining ONLY for the two temporal-split models in Phase 2, and only with --temporal/--all.
  - V4-style lgb.Dataset pattern (fresh per config; no .construct()).
"""


# ============================================================================
# Lazy loads (only invoked when actually running, not on --plan)
# ============================================================================
def _load_raw():
    from src.data_loader import load_train, load_test, make_target
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    test_raw = load_test()
    test_raw = test_raw.sort_values("srch_id").reset_index(drop=True)
    return train_raw, test_raw


def _build_features_train_test(train_raw, test_raw):
    """Build features for both train and test using same agg_source (train_raw).
    Returns (train_feat, test_feat, feature_cols)."""
    from src.features import build_features, FORBIDDEN_FEATURES
    from src.data_loader import get_feature_columns

    log("Building features on train (agg_source=train_raw, is_train=True)…")
    t0 = time.time()
    train_feat = build_features(train_raw, agg_source=train_raw, is_train=True)
    log(f"  train_feat: {len(train_feat):,} rows × {train_feat.shape[1]} cols ({time.time()-t0:.0f}s)")

    log("Building features on test (agg_source=train_raw, is_train=False)…")
    t1 = time.time()
    test_feat = build_features(test_raw, agg_source=train_raw, is_train=False)
    log(f"  test_feat:  {len(test_feat):,} rows × {test_feat.shape[1]} cols ({time.time()-t1:.0f}s)")

    feature_cols = get_feature_columns(train_feat)
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden cols leaked: {leaked}"
    return train_feat, test_feat, feature_cols


# ============================================================================
# Diagnostic 1: Adversarial validation
# ============================================================================
def diag_adversarial(train_feat, test_feat, feature_cols, audit_df, out_dir: Path) -> dict:
    """Train binary LightGBM: train_rows=0, test_rows=1. Report AUC + top drift features."""
    import lightgbm as lgb

    log("\n=== Diagnostic 1: Adversarial validation ===")
    log("Building adversarial dataset (subsample 500K each side for speed)…")

    rng = np.random.default_rng(42)
    n_sub = 500_000
    train_idx = rng.choice(len(train_feat), size=min(n_sub, len(train_feat)), replace=False)
    test_idx = rng.choice(len(test_feat), size=min(n_sub, len(test_feat)), replace=False)

    X_train_sub = train_feat[feature_cols].iloc[train_idx].copy()
    X_test_sub = test_feat[feature_cols].iloc[test_idx].copy()
    X_train_sub["__src__"] = 0
    X_test_sub["__src__"] = 1
    X_all = pd.concat([X_train_sub, X_test_sub], ignore_index=True)
    y_all = X_all.pop("__src__").astype(np.int8).values
    X_all = X_all[feature_cols]

    # 80/20 internal split for AUC reporting (random)
    perm = rng.permutation(len(X_all))
    split = int(len(X_all) * 0.8)
    tr_idx, va_idx = perm[:split], perm[split:]

    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 200,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    ds_tr = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx])
    ds_va = lgb.Dataset(X_all.iloc[va_idx], label=y_all[va_idx], reference=ds_tr)

    log("Training adversarial LightGBM (max 500 rounds, ES=30)…")
    t0 = time.time()
    model = lgb.train(
        params, ds_tr, num_boost_round=500, valid_sets=[ds_tr, ds_va],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    auc = float(model.best_score["val"]["auc"])
    log(f"adversarial AUC = {auc:.4f}   ({time.time()-t0:.0f}s)")

    interp = (
        "LOW" if auc < 0.55 else ("MODERATE" if auc < 0.65 else "SERIOUS")
    )

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "adv_gain": model.feature_importance(importance_type="gain"),
        "adv_split": model.feature_importance(importance_type="split"),
    }).sort_values("adv_gain", ascending=False).reset_index(drop=True)
    imp_df.head(50).to_csv(out_dir / "adversarial_feature_importance.csv", index=False)

    # Group-level summary (using feature_audit)
    grp = imp_df.merge(audit_df[["feature", "group", "importance_gain"]],
                       on="feature", how="left").rename(
        columns={"importance_gain": "ranker_gain", "group": "feature_group"}
    )
    grp["adv_share"] = grp["adv_gain"] / grp["adv_gain"].sum()
    grp["ranker_share"] = grp["ranker_gain"] / grp["ranker_gain"].sum()
    grp.to_csv(out_dir / "feature_risk_table.csv", index=False)

    group_summary = grp.groupby("feature_group").agg(
        n=("feature", "count"),
        sum_adv_gain=("adv_gain", "sum"),
        sum_ranker_gain=("ranker_gain", "sum"),
    )
    group_summary["adv_share"] = group_summary["sum_adv_gain"] / group_summary["sum_adv_gain"].sum()
    group_summary["ranker_share"] = group_summary["sum_ranker_gain"] / group_summary["sum_ranker_gain"].sum()
    group_summary["adv_minus_ranker"] = group_summary["adv_share"] - group_summary["ranker_share"]
    group_summary = group_summary.sort_values("adv_minus_ranker", ascending=False)
    group_summary.to_csv(out_dir / "feature_risk_by_group.csv")

    auc_payload = {
        "adversarial_auc": auc,
        "interpretation": interp,
        "n_train_subsample": int(len(train_idx)),
        "n_test_subsample": int(len(test_idx)),
        "best_iteration": int(model.best_iteration),
        "top10_drift_features": imp_df.head(10)["feature"].tolist(),
    }
    json.dump(auc_payload, open(out_dir / "adversarial_auc.json", "w"), indent=2)

    short_log("  Top 10 drift features (by adversarial gain):")
    for _, r in imp_df.head(10).iterrows():
        short_log(f"    {r['feature']:35s} adv_gain={r['adv_gain']:>10.0f}")
    short_log(f"  Top 5 feature groups most overweighted in drift (adv − ranker share):")
    for grp_name, row in group_summary.head(5).iterrows():
        short_log(f"    {str(grp_name):28s} adv={row['adv_share']*100:5.1f}%  ranker={row['ranker_share']*100:5.1f}%  Δ={row['adv_minus_ranker']*100:+5.1f}pp")

    return auc_payload


# ============================================================================
# Diagnostic 2: V4.2 vs V5 + intra-V5 disagreement
# ============================================================================
def diag_disagreement(train_raw, test_raw, out_dir: Path) -> dict:
    """Compare V5 ensemble ranking vs V4 ensemble ranking; segment by query attrs.
    Also: intra-V5 disagreement (each member vs ensemble).
    Fallback to V4.2 only if V4 submission is missing (and flag the limitation)."""
    log("\n=== Diagnostic 2: V5 vs V4 ensemble + intra-V5 disagreement ===")

    log(f"  Loading V5 submission: {V5_SUBMISSION_CSV.name}")
    v5 = pd.read_csv(V5_SUBMISSION_CSV)

    used_ref = None
    if V4_SUBMISSION_CSV.exists():
        log(f"  Loading V4 ensemble submission: {V4_SUBMISSION_CSV.name}  ← primary reference (Kaggle 0.42021)")
        ref = pd.read_csv(V4_SUBMISSION_CSV)
        used_ref = "V4_ensemble"
    elif V4_2_SUBMISSION_CSV.exists():
        log(f"  ! V4 ensemble CSV not found — falling back to V4.2 single ({V4_2_SUBMISSION_CSV.name}).")
        log(f"  ! WARNING: V4.2 (Kaggle 0.41639) underperformed V4 (Kaggle 0.42021),")
        log(f"  ! so V5 vs V4.2 compares two failed models, NOT V5 vs the V4 reference.")
        log(f"  ! To get a meaningful comparison, drop V4 ensemble submission at:")
        log(f"  !   {V4_SUBMISSION_CSV}")
        ref = pd.read_csv(V4_2_SUBMISSION_CSV)
        used_ref = "V4.2_single_FALLBACK"
    else:
        raise FileNotFoundError(
            f"Neither V4 nor V4.2 submission found.\n"
            f"  Expected V4 (primary)   : {V4_SUBMISSION_CSV}\n"
            f"  Or V4.2 (fallback only) : {V4_2_SUBMISSION_CSV}"
        )

    v5["v5_rank"] = v5.groupby("srch_id").cumcount() + 1
    ref["ref_rank"] = ref.groupby("srch_id").cumcount() + 1

    merged = v5.merge(ref, on=["srch_id", "prop_id"], how="inner")
    log(f"  reference={used_ref}  joined rows: {len(merged):,} "
        f"(V5={len(v5):,}, ref={len(ref):,})")

    # Per-srch_id disagreement
    def per_srch(g):
        top5_v5 = set(g.nsmallest(5, "v5_rank")["prop_id"])
        top5_ref = set(g.nsmallest(5, "ref_rank")["prop_id"])
        overlap5 = len(top5_v5 & top5_ref)
        try:
            from scipy.stats import spearmanr
            sp = spearmanr(g["v5_rank"], g["ref_rank"]).statistic
        except Exception:
            sp = float("nan")
        return pd.Series({
            "n_candidates": len(g),
            "top5_overlap": overlap5,
            "spearman": sp,
            "mean_abs_rank_diff": (g["v5_rank"] - g["ref_rank"]).abs().mean(),
        })

    log("  Computing per-srch disagreement (this may take ~30s)…")
    t0 = time.time()
    by_q = merged.groupby("srch_id", sort=False).apply(per_srch).reset_index()
    log(f"  per-srch done ({time.time()-t0:.0f}s, n={len(by_q):,})")
    by_q.to_csv(out_dir / "v4_v5_disagreement_by_query.csv", index=False)

    # Segment by query attributes
    test_attrs = test_raw[[
        "srch_id", "site_id", "srch_destination_id",
        "visitor_location_country_id", "prop_country_id",
        "srch_booking_window", "srch_length_of_stay",
        "srch_adults_count", "srch_children_count", "srch_room_count",
        "price_usd",
    ]].copy()
    test_attrs["is_family"] = (test_attrs["srch_children_count"] > 0).astype(int)
    test_attrs["is_domestic"] = (
        test_attrs["visitor_location_country_id"] == test_attrs["prop_country_id"]
    ).astype(int)

    q_attrs = test_attrs.groupby("srch_id").agg(
        site_id=("site_id", "first"),
        srch_destination_id=("srch_destination_id", "first"),
        visitor_country=("visitor_location_country_id", "first"),
        prop_country_first=("prop_country_id", "first"),
        booking_window=("srch_booking_window", "first"),
        length_of_stay=("srch_length_of_stay", "first"),
        adults=("srch_adults_count", "first"),
        children=("srch_children_count", "first"),
        rooms=("srch_room_count", "first"),
        is_family=("is_family", "first"),
        is_domestic_frac=("is_domestic", "mean"),
        candidate_count=("price_usd", "count"),
        price_min=("price_usd", "min"),
        price_max=("price_usd", "max"),
    ).reset_index()
    q_attrs["price_spread"] = q_attrs["price_max"] - q_attrs["price_min"]

    by_q = by_q.merge(q_attrs, on="srch_id", how="left")

    # High-disagreement = top 10% by mean_abs_rank_diff
    thresh = by_q["mean_abs_rank_diff"].quantile(0.90)
    by_q["is_high_disagreement"] = (by_q["mean_abs_rank_diff"] >= thresh).astype(int)

    seg_rows = []
    for col in ["site_id", "is_family", "is_domestic_frac", "booking_window",
                "length_of_stay", "adults", "children", "rooms", "candidate_count"]:
        if col in ("is_domestic_frac",):
            by_q["_bucket"] = pd.cut(by_q[col], bins=[-0.01, 0.5, 1.01],
                                     labels=["mixed/foreign", "domestic"])
        elif col in ("booking_window", "length_of_stay", "candidate_count", "adults", "children", "rooms"):
            by_q["_bucket"] = pd.qcut(by_q[col], q=4, duplicates="drop")
        else:
            by_q["_bucket"] = by_q[col].astype(str)
        seg = by_q.groupby("_bucket", observed=False).agg(
            n=("srch_id", "count"),
            high_disagreement_rate=("is_high_disagreement", "mean"),
            mean_top5_overlap=("top5_overlap", "mean"),
            mean_abs_rank_diff=("mean_abs_rank_diff", "mean"),
        )
        seg.insert(0, "segment_attr", col)
        seg_rows.append(seg.reset_index().rename(columns={"_bucket": "bucket"}))

    seg_all = pd.concat(seg_rows, ignore_index=True)
    seg_all.to_csv(out_dir / "v4_v5_disagreement_segments.csv", index=False)
    by_q.drop(columns="_bucket", errors="ignore").to_csv(
        out_dir / "v4_v5_disagreement_by_query.csv", index=False
    )

    short_log(f"  V5 vs {used_ref} mean top5 overlap : {by_q['top5_overlap'].mean():.3f} / 5")
    short_log(f"  V5 vs {used_ref} mean Spearman    : {by_q['spearman'].mean():.4f}")
    short_log(f"  High-disagreement threshold (top 10%): mean_abs_rank_diff ≥ {thresh:.2f}")

    # --- Intra-V5 disagreement: per-member rank vs ensemble rank ---
    log("  Computing intra-V5 member disagreement…")
    ens_rank = np.load(V5_SUBMIT_DIR / "ensemble_test_rank_avg.npy")
    test_srch = test_raw["srch_id"].values

    # Sort test by srch_id to align with stored test_pred (built in pipeline this way)
    # Per the pipeline, test_feat was sorted by srch_id and predictions follow that order.
    rows = []
    from scripts.ensemble_search import compute_ranks_within_group, _group_index
    starts, sizes = _group_index(test_srch)
    for mid in V5_MEMBERS:
        preds = np.load(V5_SUBMIT_DIR / f"test_pred_{mid}.npy")
        member_rank = compute_ranks_within_group(preds, starts, sizes)
        diff = member_rank - ens_rank
        rows.append({
            "member": mid,
            "mean_abs_rank_diff_vs_ensemble": float(np.abs(diff).mean()),
            "max_abs_rank_diff": float(np.abs(diff).max()),
            "n_top1_disagreements": int((np.abs(diff) > 0).sum()),
        })
    intra = pd.DataFrame(rows).sort_values("mean_abs_rank_diff_vs_ensemble", ascending=False)
    intra.to_csv(out_dir / "v5_internal_disagreement.csv", index=False)
    short_log("  Intra-V5 (member rank vs ensemble rank, abs diff):")
    for _, r in intra.iterrows():
        short_log(f"    {r['member']:4s}: mean |Δ|={r['mean_abs_rank_diff_vs_ensemble']:.3f}  max={r['max_abs_rank_diff']:.1f}")

    return {
        "reference_used": used_ref,
        "v5_vs_ref_mean_top5_overlap": float(by_q["top5_overlap"].mean()),
        "v5_vs_ref_mean_spearman": float(by_q["spearman"].mean()),
        "high_disagreement_threshold": float(thresh),
        "intra_v5_max_mean_abs_diff": float(intra["mean_abs_rank_diff_vs_ensemble"].max()),
        "intra_v5_max_member": str(intra.iloc[0]["member"]),
    }


# ============================================================================
# Diagnostic 3: Position-bias
# ============================================================================
def diag_position_corr(train_feat, feature_cols, audit_df, adv_imp: pd.DataFrame,
                       out_dir: Path) -> dict:
    log("\n=== Diagnostic 3: Position-bias diagnostic ===")
    if "position" not in train_feat.columns:
        log("  ! 'position' column not in train_feat — skipping (test has no position).")
        return {}

    log("  Computing per-feature Spearman ρ with position (train only)…")
    t0 = time.time()
    rows = []
    pos = train_feat["position"].values
    for f in feature_cols:
        x = train_feat[f].values
        # quick spearman via rankdata (numpy + scipy.stats.rankdata)
        try:
            from scipy.stats import spearmanr
            mask = np.isfinite(x)
            if mask.sum() < 1000:
                continue
            r = spearmanr(x[mask], pos[mask]).statistic
        except Exception:
            r = float("nan")
        rows.append({"feature": f, "spearman_with_position": r})
    df = pd.DataFrame(rows)
    df["abs_corr"] = df["spearman_with_position"].abs()
    df = df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    df.head(30).to_csv(out_dir / "position_correlation_features.csv", index=False)
    log(f"  position correlations computed ({time.time()-t0:.0f}s)")

    # Position-leaky suspects: high ranker × high adversarial × high position-corr
    adv = adv_imp.set_index("feature")["adv_gain"]
    ranker = audit_df.set_index("feature")["importance_gain"]
    sus = df.copy()
    sus["adv_gain"] = sus["feature"].map(adv)
    sus["ranker_gain"] = sus["feature"].map(ranker)

    # Score: rank in each, take top features that appear in top-30 of all three
    def top_rank(s, asc=False):
        return s.rank(ascending=asc, method="min")
    sus["pos_rank"] = top_rank(sus["abs_corr"])
    sus["adv_rank"] = top_rank(sus["adv_gain"].fillna(0))
    sus["ranker_rank"] = top_rank(sus["ranker_gain"].fillna(0))
    sus["sum_rank"] = sus[["pos_rank", "adv_rank", "ranker_rank"]].sum(axis=1)
    sus = sus.sort_values("sum_rank").reset_index(drop=True)
    suspects = sus[(sus["pos_rank"] <= 30) & (sus["adv_rank"] <= 50) & (sus["ranker_rank"] <= 50)]
    suspects[["feature", "spearman_with_position", "abs_corr", "adv_gain", "ranker_gain",
              "pos_rank", "adv_rank", "ranker_rank"]].to_csv(
        out_dir / "position_leaky_suspects.csv", index=False
    )

    short_log("  Top 10 position-correlated features:")
    for _, r in df.head(10).iterrows():
        short_log(f"    {r['feature']:35s} ρ={r['spearman_with_position']:+.3f}")
    short_log(f"  Position-leaky suspects (in top30 pos AND top50 adv AND top50 ranker): {len(suspects)}")
    for _, r in suspects.head(10).iterrows():
        short_log(f"    {r['feature']:35s} ρ={r['spearman_with_position']:+.3f}  adv={r['adv_gain']:.0f}  ranker={r['ranker_gain']:.0f}")

    return {
        "top_pos_corr_features": df.head(10)["feature"].tolist(),
        "n_position_leaky_suspects": int(len(suspects)),
        "position_leaky_suspects": suspects.head(10)["feature"].tolist(),
    }


# ============================================================================
# Diagnostic 4: Cold-start enumeration
# ============================================================================
def diag_cold_start(train_raw, test_raw, by_q_path: Path, out_dir: Path) -> dict:
    log("\n=== Diagnostic 4: Cold-start enumeration ===")
    train_props = set(train_raw["prop_id"].unique())
    train_dests = set(train_raw["srch_destination_id"].unique())
    train_prop_dest = set(zip(train_raw["prop_id"], train_raw["srch_destination_id"]))
    train_prop_site = set(zip(train_raw["prop_id"], train_raw["site_id"]))
    train_prop_counts = train_raw.groupby("prop_id").size().rename("prop_count_in_train")

    test_pd = list(zip(test_raw["prop_id"], test_raw["srch_destination_id"]))
    test_ps = list(zip(test_raw["prop_id"], test_raw["site_id"]))

    unseen_prop = 1 - test_raw["prop_id"].isin(train_props).mean()
    unseen_dest = 1 - test_raw["srch_destination_id"].isin(train_dests).mean()
    unseen_prop_dest = 1 - np.mean([t in train_prop_dest for t in test_pd])
    unseen_prop_site = 1 - np.mean([t in train_prop_site for t in test_ps])

    test_with_count = test_raw[["srch_id", "prop_id"]].merge(
        train_prop_counts, left_on="prop_id", right_index=True, how="left"
    )
    test_with_count["prop_count_in_train"] = test_with_count["prop_count_in_train"].fillna(0)
    dist = test_with_count["prop_count_in_train"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )

    cs = pd.DataFrame({
        "metric": [
            "unseen_prop_id_rate", "unseen_srch_destination_id_rate",
            "unseen_prop_x_dest_rate", "unseen_prop_x_site_rate",
            "test_prop_count_min", "test_prop_count_p10", "test_prop_count_p25",
            "test_prop_count_p50", "test_prop_count_p75", "test_prop_count_p90",
            "test_prop_count_max", "test_prop_count_mean",
        ],
        "value": [
            unseen_prop, unseen_dest, unseen_prop_dest, unseen_prop_site,
            dist["min"], dist["10%"], dist["25%"], dist["50%"], dist["75%"], dist["90%"],
            dist["max"], dist["mean"],
        ],
    })
    cs.to_csv(out_dir / "cold_start_summary.csv", index=False)

    short_log(f"  % test prop_id never seen in train       : {unseen_prop*100:.2f}%")
    short_log(f"  % test srch_destination_id never seen     : {unseen_dest*100:.2f}%")
    short_log(f"  % test (prop, dest) pair never seen       : {unseen_prop_dest*100:.2f}%")
    short_log(f"  % test (prop, site) pair never seen       : {unseen_prop_site*100:.2f}%")
    short_log(f"  test prop_count_in_train distribution     : "
              f"p25={dist['25%']:.0f}  p50={dist['50%']:.0f}  p75={dist['75%']:.0f}")

    # Cold-start rate among high-disagreement queries vs the rest
    if by_q_path.exists():
        by_q = pd.read_csv(by_q_path)
        test_with_count["cold_start_prop"] = ~test_with_count["prop_id"].isin(train_props)
        cs_per_q = test_with_count.groupby("srch_id")["cold_start_prop"].mean().rename(
            "cold_start_rate"
        ).reset_index()
        by_q = by_q.merge(cs_per_q, on="srch_id", how="left")
        seg = by_q.groupby("is_high_disagreement").agg(
            n=("srch_id", "count"),
            mean_cold_start_rate=("cold_start_rate", "mean"),
            median_cold_start_rate=("cold_start_rate", "median"),
        )
        seg.to_csv(out_dir / "cold_start_by_disagreement_segment.csv")
        short_log("  Cold-start rate by disagreement segment:")
        for idx, row in seg.iterrows():
            label = "high_disagreement" if idx == 1 else "normal"
            short_log(f"    {label:18s} n={int(row['n']):>5d}  mean_cs={row['mean_cold_start_rate']*100:5.2f}%")

    return {
        "unseen_prop_rate": float(unseen_prop),
        "unseen_dest_rate": float(unseen_dest),
        "unseen_prop_x_dest_rate": float(unseen_prop_dest),
        "unseen_prop_x_site_rate": float(unseen_prop_site),
        "test_prop_count_median": float(dist["50%"]),
    }


# ============================================================================
# Phase 2: Temporal split retraining
# ============================================================================
def diag_temporal_split(train_raw, out_dir: Path) -> dict:
    """Sort srch_ids by date_time; earliest 80% → train, latest 20% → val.
    Train V4 anchor (lg=0,1,15) + B3 (lg=0,2,15, ipw_clip3). Compare to random-val NDCG@5."""
    import lightgbm as lgb
    from src.features import build_features, compute_position_propensity
    from src.data_loader import get_feature_columns

    log("\n=== Phase 2: Temporal split retraining ===")
    if "date_time" not in train_raw.columns:
        log("  ! train_raw has no 'date_time' — falling back to srch_id ordering.")
        srch_order = train_raw[["srch_id"]].drop_duplicates().sort_values("srch_id")
    else:
        srch_order = (
            train_raw[["srch_id", "date_time"]]
            .groupby("srch_id", sort=False)["date_time"].min()
            .reset_index()
            .sort_values("date_time")
        )

    n_srch = len(srch_order)
    n_train = int(n_srch * 0.8)
    train_srch_ids = set(srch_order.iloc[:n_train]["srch_id"])
    val_srch_ids = set(srch_order.iloc[n_train:]["srch_id"])
    log(f"  searches: {n_srch:,} → train={len(train_srch_ids):,} (earliest 80%) | val={len(val_srch_ids):,} (latest 20%)")

    train_split = train_raw[train_raw["srch_id"].isin(train_srch_ids)].reset_index(drop=True)
    val_split = train_raw[train_raw["srch_id"].isin(val_srch_ids)].reset_index(drop=True)
    log(f"  rows: train={len(train_split):,} | val={len(val_split):,}")

    # Build features (agg_source = train_split, V4-style)
    log("  Building features for temporal split (agg_source=train_split)…")
    t0 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    log(f"  features: train={len(train_feat):,} val={len(val_feat):,} cols={train_feat.shape[1]} ({time.time()-t0:.0f}s)")

    feature_cols = get_feature_columns(train_feat)
    propensity = compute_position_propensity(train_split)
    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    def gc_(df):
        return df.groupby("srch_id", sort=False).size().values

    train_groups = gc_(train_feat)
    val_groups = gc_(val_feat)

    # --- compute weights helpers (copied from v5_ensemble_submit) ---
    def weights_default(df):
        max_p = float(propensity.max())
        pos_w = df["position"].map(
            lambda p: 1.0 if propensity.get(p, 0) <= 0 else max_p / propensity[p]
        ).astype(np.float32).values
        nonrand = (df["random_bool"].values == 0)
        w = np.where(nonrand, pos_w, 1.0).astype(np.float32)
        return np.clip(w, 0.1, 10.0).astype(np.float32)

    def weights_clip3(df):
        max_p = float(propensity.max())
        pos_w = df["position"].map(
            lambda p: 1.0 if propensity.get(p, 0) <= 0 else max_p / propensity[p]
        ).astype(np.float32).values
        nonrand = (df["random_bool"].values == 0)
        w = np.where(nonrand, pos_w, 1.0).astype(np.float32)
        return np.clip(w, 0.1, 3.0).astype(np.float32)

    BASE = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [5],
        "boosting_type": "gbdt", "learning_rate": 0.03, "num_leaves": 400,
        "max_depth": -1, "min_child_samples": 50,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "min_split_gain": 0.0,
        "verbose": -1, "n_jobs": -1, "seed": 456,
    }

    results = []
    for name, lg, wfn in [
        ("V4_anchor_lg_0_1_15", "0,1,15", weights_default),
        ("B3_lg_0_2_15_clip3",   "0,2,15", weights_clip3),
    ]:
        log(f"\n  Training {name} on temporal split (early_stop=80, max=2000 rounds)…")
        w = wfn(train_split)
        params = BASE.copy()
        params["label_gain"] = lg
        ds_tr = lgb.Dataset(train_feat[feature_cols], label=train_label,
                            group=train_groups, weight=w)
        ds_va = lgb.Dataset(val_feat[feature_cols], label=val_label,
                            group=val_groups, reference=ds_tr)
        t = time.time()
        model = lgb.train(
            params, ds_tr, num_boost_round=2000,
            valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
        )
        ndcg5 = float(model.best_score["val"]["ndcg@5"])
        bi = int(model.best_iteration)
        log(f"  {name}: NDCG@5 (temporal val) = {ndcg5:.5f}  best_iter={bi}  ({(time.time()-t)/60:.1f} min)")
        results.append({"model": name, "label_gain": lg,
                        "temporal_val_ndcg5": ndcg5, "best_iter": bi})
        model.save_model(str(out_dir / f"model_{name}_temporal.txt"))
        del ds_tr, ds_va, model
        gc.collect()

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "temporal_split_results.csv", index=False)
    short_log("\n  Temporal split results:")
    for r in results:
        short_log(f"    {r['model']:25s}  NDCG@5={r['temporal_val_ndcg5']:.5f}  best_iter={r['best_iter']}")
    short_log(f"\n  Compare to random-val: V4 anchor 0.42191 | B3 0.42396")
    return {"temporal": results}


# ============================================================================
# Report writer
# ============================================================================
def write_readme(out_dir: Path, summary: dict) -> None:
    md = ["# V5 gap diagnostic — " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""]
    md.append(f"V4 ensemble local={V4_LOCAL:.5f}, Kaggle={V4_KAGGLE:.5f}, gap=−0.00491")
    md.append(f"V5 ensemble local={V5_LOCAL:.5f}, Kaggle={V5_KAGGLE:.5f}, gap=−0.00690")
    md.append("")
    md.append("## 1. Executive summary")
    md.append("_See README at end of full run; this header is written by `--all`._")
    md.append("")
    md.append("## 2. Adversarial validation")
    if "adversarial" in summary:
        a = summary["adversarial"]
        md.append(f"- AUC = **{a['adversarial_auc']:.4f}** (interpretation: **{a['interpretation']}** drift)")
        md.append(f"- Top 10 drift features: {', '.join(a['top10_drift_features'])}")
    md.append("- See `adversarial_feature_importance.csv` and `feature_risk_by_group.csv`.")
    md.append("")
    md.append("## 3. Position-bias")
    if "position" in summary:
        p = summary["position"]
        md.append(f"- Top 10 features by |Spearman ρ| with position: {', '.join(p['top_pos_corr_features'])}")
        md.append(f"- Position-leaky suspects (top30 pos × top50 adv × top50 ranker): "
                  f"{p['n_position_leaky_suspects']} → {', '.join(p['position_leaky_suspects'])}")
    md.append("- See `position_correlation_features.csv` and `position_leaky_suspects.csv`.")
    md.append("")
    md.append("## 4. Cold-start")
    if "cold_start" in summary:
        c = summary["cold_start"]
        md.append(f"- Unseen prop_id rate: **{c['unseen_prop_rate']*100:.2f}%**")
        md.append(f"- Unseen srch_destination_id rate: **{c['unseen_dest_rate']*100:.2f}%**")
        md.append(f"- Unseen (prop,dest) pair rate: **{c['unseen_prop_x_dest_rate']*100:.2f}%**")
        md.append(f"- Unseen (prop,site) pair rate: **{c['unseen_prop_x_site_rate']*100:.2f}%**")
        md.append(f"- Test prop_count_in_train median: {c['test_prop_count_median']:.0f}")
    md.append("")
    md.append("## 5. V5 vs reference + intra-V5 disagreement")
    if "disagreement" in summary:
        d = summary["disagreement"]
        md.append(f"- Reference used: **{d['reference_used']}**")
        md.append(f"- V5 vs ref mean top-5 overlap: **{d['v5_vs_ref_mean_top5_overlap']:.3f} / 5**")
        md.append(f"- V5 vs ref mean Spearman: **{d['v5_vs_ref_mean_spearman']:.4f}**")
        md.append(f"- Intra-V5 most divergent member from ensemble: "
                  f"**{d['intra_v5_max_member']}** (mean |Δrank|={d['intra_v5_max_mean_abs_diff']:.3f})")
        if d["reference_used"].endswith("FALLBACK"):
            md.append(f"- ⚠️ V4.2 used as fallback because V4 ensemble CSV was missing. "
                      f"Drop it at `submissions/submission_v4_ensemble.csv` and re-run for the primary signal.")
    md.append("- See `v4_v5_disagreement_segments.csv`, `v5_internal_disagreement.csv`.")
    md.append("")
    md.append("## 6. Temporal split")
    if "temporal" in summary:
        md.append("| model | temporal NDCG@5 | best_iter | random val ref |")
        md.append("|---|---|---|---|")
        for r in summary["temporal"]:
            ref = ANCHOR_LOCAL if "anchor" in r["model"] else 0.42396
            md.append(f"| {r['model']} | {r['temporal_val_ndcg5']:.5f} | {r['best_iter']} | {ref:.5f} |")
    else:
        md.append("_Not run (pass `--temporal` or `--all`)._")
    md.append("")
    md.append("## 7. Action table (filled by analyst from findings)")
    md.append("| feature_or_group | evidence | risk | proposed action | next experiment |")
    md.append("|---|---|---|---|---|")
    md.append("| _to fill_ | _adv AUC, top drift, position-corr_ | _HIGH/MED/LOW_ | _drop / smooth / TE-rank / cold-start indicator_ | _phase-N script_ |")
    md.append("")
    md.append("**Candidate actions** (from spec, prioritize based on Phase 1 evidence):")
    md.append("- drop/test `prop_avg_position`")
    md.append("- top120/top100 feature pruning")
    md.append("- cross-key TE smoothing 40/80")
    md.append("- cross-key TE fallback hierarchy")
    md.append("- TE rank within srch_id")
    md.append("- train+test non-target catalog stats")
    md.append("- cold-start indicators / fallback features")
    md.append("- hard-negative features")
    (out_dir / "README.md").write_text("\n".join(md))


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Phase 1 diagnostics only (default)")
    parser.add_argument("--temporal", action="store_true", help="Phase 2 only: temporal-split retrain")
    parser.add_argument("--all", action="store_true", help="Phase 1 + Phase 2 + README")
    parser.add_argument("--plan", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    short_log(PLAN_HEADER)
    if args.plan:
        return

    # If nothing specified, default to --quick
    if not (args.quick or args.temporal or args.all):
        args.quick = True

    do_phase1 = args.quick or args.all
    do_phase2 = args.temporal or args.all

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "diagnostics" / f"v5_gap_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    short_log(f"\n>>> Output dir: {out_dir}")
    short_log(f">>> Phase 1 (--quick): {'YES' if do_phase1 else 'no'}")
    short_log(f">>> Phase 2 (--temporal): {'YES' if do_phase2 else 'no'}\n")

    summary: dict = {"timestamp": ts, "out_dir": str(out_dir)}

    train_raw, test_raw = _load_raw()
    log(f"Train: {len(train_raw):,} rows / {train_raw['srch_id'].nunique():,} searches")
    log(f"Test : {len(test_raw):,} rows / {test_raw['srch_id'].nunique():,} searches")

    if do_phase1:
        train_feat, test_feat, feature_cols = _build_features_train_test(train_raw, test_raw)
        audit_df = pd.read_csv(FEATURE_AUDIT_CSV)

        # 1. Adversarial
        adv_payload = diag_adversarial(train_feat, test_feat, feature_cols, audit_df, out_dir)
        summary["adversarial"] = adv_payload

        # 2. Disagreement (also writes by_q file)
        dis_payload = diag_disagreement(train_raw, test_raw, out_dir)
        summary["disagreement"] = dis_payload

        # 3. Position-bias
        adv_imp = pd.read_csv(out_dir / "adversarial_feature_importance.csv")
        pos_payload = diag_position_corr(train_feat, feature_cols, audit_df, adv_imp, out_dir)
        summary["position"] = pos_payload

        # 4. Cold-start
        cs_payload = diag_cold_start(train_raw, test_raw,
                                     out_dir / "v4_v5_disagreement_by_query.csv", out_dir)
        summary["cold_start"] = cs_payload

        # Free memory before Phase 2
        del train_feat, test_feat, feature_cols, adv_imp
        gc.collect()

    if do_phase2:
        temporal_payload = diag_temporal_split(train_raw, out_dir)
        summary["temporal"] = temporal_payload["temporal"]

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, default=str)
    write_readme(out_dir, summary)
    short_log(f"\n>>> Wrote README: {out_dir / 'README.md'}")
    short_log(f">>> Wrote summary.json: {out_dir / 'summary.json'}")
    short_log(f">>> Done.")


if __name__ == "__main__":
    main()
