"""
STEP 2 — TE_SAFE_SINGLE.

Purpose:
  Build ONE focused TE-safe single model on top of B3. Evaluate on both
  random-control and temporal validation. No submission, no ensemble.

DO NOT RUN until temporal validation (pipelines/temporal_validation.py) results are
reviewed. Decision rule there determines whether B3 is the right base.

Setup:
  Base config: B3
    label_gain = "0,2,15"
    weighting  = IPW clipped at 3
    base_params = V4 backbone

Feature changes:
  (A) Add pct-rank-within-srch_id for 12 TE features:
        prop_click_rate, prop_book_rate, prop_rel_rate,
        dest_click_rate, dest_book_rate,
        country_book_rate, site_book_rate,
        prop_dest_book_rate, prop_site_book_rate,
        site_dest_book_rate, site_country_book_rate, cpair_book_rate
      New columns: <feature>_pct_rank. Original kept.

  (B) Heavier smoothing for the 5 cross-key TEs (prior_weight=80).
      Implementation: build features normally, then OVERWRITE the 5 cross-key
      TE columns by calling kfold_target_encode (train) / target_encode_from_source
      (val) with prior_weight=80. This avoids editing src/features.py.

  (C) Fallback hierarchy:
      *** SKIPPED — fallback hierarchy not implemented here. ***
      Reason: requires per-row rarity tracking in train, plus single-key/global
      cascade logic, which is invasive to add cleanly. Implement after we see
      whether (A) + (B) alone help.

Evaluation:
  Uses the SAME temporal_split + random-control logic as pipelines/temporal_validation.py
  so numbers are directly comparable to V4_ANCHOR / B3 from STEP 1.

Outputs:
  diagnostics/te_safe_<TS>/
    te_safe_results.csv          (TE_SAFE × {random_control, temporal})
    README.md                    (decision tree applied + recommendation slot)
    model_TE_SAFE_<setup>.txt
    split_meta.json              (same split discipline as STEP 1)
  artifacts/te_safe_<TS>/         (importance + feature_cols + run_summary)

Hard constraints:
  No submission generated. No ensemble. No other model variants. Same V4-style
  fresh lgb.Dataset (no .construct(), no free_raw_data).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_train, make_target, get_feature_columns  # noqa: E402
from src.features import (  # noqa: E402
    build_features, compute_position_propensity, FORBIDDEN_FEATURES,
    kfold_target_encode, target_encode_from_source, _make_cross_key,
)

# Reuse split + train helpers from pipelines/temporal_validation.py for *exact* parity
from pipelines.temporal_validation import (  # noqa: E402
    temporal_split, random_inner_split,
    compute_ipw_weights, make_group_counts, eval_metrics,
    BASE_PARAMS, RANDOM_CONTROL_SEED, RANDOM_VAL_FRAC,
)

# ============================================================================
# Constants
# ============================================================================
V4_ENSEMBLE_KAGGLE = 0.42021
V4_ENSEMBLE_LOCAL = 0.42512
V4_ANCHOR_RANDOM_VAL = 0.42191
B3_RANDOM_VAL = 0.42396
ANCHOR_SEED = 456

# B3 best_iter from overnight (used as soft reference; we still use early stopping here
# because we have a real val set in each setup).
B3_BEST_ITER_RANDOM_VAL = 326

# 12 TE features that get a within-srch pct-rank companion column
TE_FEATURES_FOR_RANK = [
    "prop_click_rate", "prop_book_rate", "prop_rel_rate",
    "dest_click_rate", "dest_book_rate",
    "country_book_rate", "site_book_rate",
    "prop_dest_book_rate", "prop_site_book_rate",
    "site_dest_book_rate", "site_country_book_rate", "cpair_book_rate",
]

# 5 cross-key TEs to re-encode with heavier smoothing (prior_weight=80)
HEAVY_SMOOTHING_CROSS_TES = [
    # (col_a, col_b, target_col, prefix)
    ("prop_id", "srch_destination_id", "booking_bool", "prop_dest_book"),
    ("site_id", "srch_destination_id", "booking_bool", "site_dest_book"),
    ("prop_id", "site_id",            "booking_bool", "prop_site_book"),
    ("visitor_location_country_id", "prop_country_id", "booking_bool", "cpair_book"),
    ("site_id", "prop_country_id",    "booking_bool", "site_country_book"),
]
HEAVY_PRIOR_WEIGHT = 80


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Plan header (printed at start)
# ============================================================================
PLAN = """\
run_te_safe_single.py — execution plan
=======================================

Reads:
  data/training_set_VU_DM.csv                  (4.96M rows; 'date_time' required)
  pipelines/temporal_validation.py                   (split + helper functions imported)

Writes (under diagnostics/te_safe_<TS>/ and artifacts/te_safe_<TS>/):
  diagnostics/te_safe_<TS>/
    te_safe_results.csv          (2 rows: TE_SAFE × {random_control, temporal})
    README.md                    (with decision rules applied)
    model_TE_SAFE_<setup>.txt    (2 boosters)
    split_meta.json              (parity check with STEP 1)
  artifacts/te_safe_<TS>/
    feature_cols_TE_SAFE.json
    importance_TE_SAFE_<setup>.csv
    run_summary.md

Estimated runtime:
  Feature builds (2 setups × 2 subsets) : ~14 min
  Heavy-smoothing re-encode             : ~30 s × 4 = ~2 min
  pct-rank-within-srch                  : ~30 s × 4 = ~2 min
  LGBM trains                           : 2 × ~3–5 min = ~8 min
  Total                                 : ~26–30 min

Command (DO NOT RUN until STEP 1 reviewed):
  uv run python -m py_compile run_te_safe_single.py
  uv run python run_te_safe_single.py

Hard constraints honored:
  - No submission. No ensemble. Only one model trained per setup.
  - Same fresh-Dataset / no-construct pattern as STEP 1.
  - Same temporal + random-control splits as STEP 1 (imported, not re-implemented).
  - Fallback hierarchy: SKIPPED with explicit note (not implemented).
"""


# ============================================================================
# Feature engineering helpers
# ============================================================================
def add_pct_rank_within_srch(df: pd.DataFrame, te_features: list[str]) -> list[str]:
    """For each TE feature present, add <name>_pct_rank. Returns list of new col names."""
    new_cols = []
    for f in te_features:
        if f not in df.columns:
            continue
        new = f"{f}_pct_rank"
        df[new] = df.groupby("srch_id")[f].rank(pct=True, method="average").astype(np.float32)
        new_cols.append(new)
    return new_cols


def reencode_cross_tes_heavy(train_df: pd.DataFrame, target_df: pd.DataFrame,
                              is_train: bool, prior_weight: int) -> int:
    """OVERWRITE the 5 cross-key TE columns with a heavier-smoothed re-encode.
    Returns count of columns overwritten."""
    n = 0
    for col_a, col_b, target_col, prefix in HEAVY_SMOOTHING_CROSS_TES:
        col_name = f"{prefix}_rate"
        if col_name not in target_df.columns:
            continue
        cross_key = f"_xkey_he_{prefix}"
        # Ensure both train and target have the cross_key built consistently
        train_tmp = train_df.assign(**{cross_key: _make_cross_key(train_df, col_a, col_b)})
        target_df[cross_key] = _make_cross_key(target_df, col_a, col_b)
        if is_train:
            target_df[col_name] = kfold_target_encode(
                target_df, cross_key, target_col, prefix, prior_weight=prior_weight
            )
        else:
            target_df[col_name] = target_encode_from_source(
                train_tmp, target_df, cross_key, target_col, prefix, prior_weight=prior_weight
            )
        target_df.drop(columns=[cross_key], inplace=True)
        del train_tmp
        n += 1
    return n


# ============================================================================
# Train one setup
# ============================================================================
def train_te_safe(setup_name, train_split, val_split, propensity,
                  out_dir, art_dir, log_prefix=""):
    log(f"{log_prefix}building features for setup={setup_name}…")
    t0 = time.time()

    # Standard features
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    base_feature_cols = get_feature_columns(train_feat)
    log(f"{log_prefix}  base features built: train={len(train_feat):,} val={len(val_feat):,} "
        f"cols={len(base_feature_cols)} ({(time.time()-t0)/60:.1f} min)")

    # (B) Re-encode 5 cross-key TEs with heavier smoothing
    n_train = reencode_cross_tes_heavy(train_split, train_feat, is_train=True,
                                       prior_weight=HEAVY_PRIOR_WEIGHT)
    n_val = reencode_cross_tes_heavy(train_split, val_feat, is_train=False,
                                     prior_weight=HEAVY_PRIOR_WEIGHT)
    log(f"{log_prefix}  re-encoded {n_train} cross-key TEs (prior_weight={HEAVY_PRIOR_WEIGHT}) "
        f"on train, {n_val} on val")

    # (A) Add pct-rank-within-srch columns
    new_train = add_pct_rank_within_srch(train_feat, TE_FEATURES_FOR_RANK)
    new_val = add_pct_rank_within_srch(val_feat, TE_FEATURES_FOR_RANK)
    assert set(new_train) == set(new_val), "pct-rank columns differ between train and val"
    log(f"{log_prefix}  added {len(new_train)} pct-rank columns")

    feature_cols = base_feature_cols + new_train
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden cols leaked: {leaked}"
    json.dump(list(feature_cols), open(art_dir / "feature_cols_TE_SAFE.json", "w"), indent=2)
    log(f"{log_prefix}  final feature count: {len(feature_cols)}")
    assert list(train_feat[feature_cols].columns) == list(val_feat[feature_cols].columns), \
        "train/val column order disagrees"

    # Weights (B3 weighting = ipw_clip3)
    weights = compute_ipw_weights(train_split, propensity, clip_hi=3.0, clip_lo=0.1)
    log(f"{log_prefix}  weights: [{weights.min():.3f}, {weights.max():.3f}] mean={weights.mean():.3f}")

    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    params = BASE_PARAMS.copy()
    params["label_gain"] = "0,2,15"

    log(f"{log_prefix}  training (early_stop=80, max=2000 rounds)…")
    t1 = time.time()
    ds_tr = lgb.Dataset(train_feat[feature_cols], label=train_label,
                        group=make_group_counts(train_feat), weight=weights)
    ds_va = lgb.Dataset(val_feat[feature_cols], label=val_label,
                        group=make_group_counts(val_feat), reference=ds_tr)
    model = lgb.train(
        params, ds_tr, num_boost_round=2000,
        valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )
    best_iter = int(model.best_iteration)
    log(f"{log_prefix}  trained best_iter={best_iter} in {(time.time()-t1)/60:.1f} min")

    val_pred = model.predict(val_feat[feature_cols]).astype(np.float32)
    metrics = eval_metrics(val_feat, val_pred)
    log(f"{log_prefix}  NDCG@5={metrics['ndcg5']:.5f}  Recall@5={metrics['recall5']:.4f}  "
        f"MBR={metrics['mean_booked_rank']:.3f}")

    model_path = out_dir / f"model_TE_SAFE_{setup_name}.txt"
    model.save_model(str(model_path))

    # Importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(art_dir / f"importance_TE_SAFE_{setup_name}.csv", index=False)

    result = {
        "model_id": "TE_SAFE",
        "train_setup": setup_name,
        "val_setup": setup_name,
        "label_gain": "0,2,15",
        "weighting": "ipw_clip3",
        "best_iteration": best_iter,
        "n_features": len(feature_cols),
        "n_pct_rank_added": len(new_train),
        "n_cross_tes_reencoded": n_train,
        "heavy_prior_weight": HEAVY_PRIOR_WEIGHT,
        "fallback_hierarchy": "SKIPPED (not implemented in this script)",
        "model_path": str(model_path),
        **metrics,
    }

    del ds_tr, ds_va, model, val_pred, train_feat, val_feat, weights, train_label, val_label
    gc.collect()
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print(PLAN, flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "diagnostics" / f"te_safe_{ts}"
    art_dir = ROOT / "artifacts" / f"te_safe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    log(f"out_dir: {out_dir}")
    log(f"art_dir: {art_dir}")

    log("Loading full train…")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(train_raw):,} rows / {train_raw['srch_id'].nunique():,} searches")

    # Splits — re-uses the same logic as STEP 1 for parity
    log("\nComputing temporal split…")
    temporal_train_ids, temporal_val_ids, cutoff = temporal_split(train_raw)
    inner_train_ids, inner_random_val_ids = random_inner_split(
        train_raw, temporal_train_ids, RANDOM_VAL_FRAC, RANDOM_CONTROL_SEED
    )
    log(f"  temporal_cutoff={cutoff}")
    log(f"  temporal_train searches={len(temporal_train_ids):,}, val={len(temporal_val_ids):,}")
    log(f"  inner_train searches={len(inner_train_ids):,}, inner_random_val={len(inner_random_val_ids):,}")

    assert temporal_train_ids.isdisjoint(temporal_val_ids), "temporal sets overlap!"
    assert inner_train_ids.isdisjoint(inner_random_val_ids), "inner sets overlap!"

    temporal_train_df = train_raw[train_raw["srch_id"].isin(temporal_train_ids)].reset_index(drop=True)
    temporal_val_df = train_raw[train_raw["srch_id"].isin(temporal_val_ids)].reset_index(drop=True)
    inner_train_df = train_raw[train_raw["srch_id"].isin(inner_train_ids)].reset_index(drop=True)
    inner_random_val_df = train_raw[train_raw["srch_id"].isin(inner_random_val_ids)].reset_index(drop=True)

    split_meta = {
        "timestamp": ts,
        "temporal_cutoff_date": str(cutoff),
        "temporal_train_searches": len(temporal_train_ids),
        "temporal_val_searches": len(temporal_val_ids),
        "inner_train_searches": len(inner_train_ids),
        "inner_random_val_searches": len(inner_random_val_ids),
        "temporal_train_rows": int(len(temporal_train_df)),
        "temporal_val_rows": int(len(temporal_val_df)),
        "inner_train_rows": int(len(inner_train_df)),
        "inner_random_val_rows": int(len(inner_random_val_df)),
        "leakage_assertions_passed": True,
        "te_features_for_rank": TE_FEATURES_FOR_RANK,
        "heavy_smoothing_cross_tes": [c[3] for c in HEAVY_SMOOTHING_CROSS_TES],
        "heavy_prior_weight": HEAVY_PRIOR_WEIGHT,
        "fallback_hierarchy": "SKIPPED",
    }
    json.dump(split_meta, open(out_dir / "split_meta.json", "w"), indent=2, default=str)

    del train_raw
    gc.collect()

    results = []

    # ----- Setup 1: random-control -----
    log("\n" + "=" * 70)
    log("SETUP 1: random-control (train=inner_train, val=inner_random_val)")
    log("=" * 70)
    propensity_inner = compute_position_propensity(inner_train_df)
    r1 = train_te_safe(
        "random_control", inner_train_df, inner_random_val_df,
        propensity_inner, out_dir, art_dir,
        log_prefix="  [TE_SAFE | random_control] ",
    )
    results.append(r1)
    del inner_train_df, inner_random_val_df, propensity_inner
    gc.collect()

    # ----- Setup 2: temporal -----
    log("\n" + "=" * 70)
    log("SETUP 2: temporal (train=temporal_train, val=temporal_val)")
    log("=" * 70)
    propensity_temporal = compute_position_propensity(temporal_train_df)
    r2 = train_te_safe(
        "temporal", temporal_train_df, temporal_val_df,
        propensity_temporal, out_dir, art_dir,
        log_prefix="  [TE_SAFE | temporal] ",
    )
    results.append(r2)
    del temporal_train_df, temporal_val_df, propensity_temporal
    gc.collect()

    # ---- Save ----
    df = pd.DataFrame(results, columns=[
        "model_id", "train_setup", "val_setup", "label_gain", "weighting",
        "ndcg5", "recall1", "recall5", "mean_booked_rank",
        "best_iteration", "n_features", "n_pct_rank_added", "n_cross_tes_reencoded",
        "heavy_prior_weight", "fallback_hierarchy",
        "n_val_queries", "n_val_booked", "model_path",
    ])
    df.to_csv(out_dir / "te_safe_results.csv", index=False)
    log(f"\nresults CSV: {out_dir / 'te_safe_results.csv'}")

    # ---- README ----
    md = [
        f"# TE_SAFE_SINGLE — {ts}",
        "",
        f"Base config: B3 (label_gain=0,2,15, IPW clip3).",
        f"V4 ensemble Kaggle = {V4_ENSEMBLE_KAGGLE:.5f} · V4 ensemble local = {V4_ENSEMBLE_LOCAL:.5f}",
        f"Reference random-val NDCG@5: V4 anchor = {V4_ANCHOR_RANDOM_VAL:.5f} · B3 = {B3_RANDOM_VAL:.5f}",
        "",
        "## Feature changes vs B3",
        "",
        f"- **(A)** Added pct-rank-within-srch for {len(TE_FEATURES_FOR_RANK)} TE features "
        f"({', '.join(TE_FEATURES_FOR_RANK)}).",
        f"- **(B)** Re-encoded 5 cross-key TEs (`prop_dest_book_rate`, `site_dest_book_rate`, "
        f"`prop_site_book_rate`, `cpair_book_rate`, `site_country_book_rate`) "
        f"with prior_weight={HEAVY_PRIOR_WEIGHT} (was 10/15/15/20/20).",
        "- **(C)** Fallback hierarchy: **SKIPPED** — not implemented in this script. "
        "Reason: implementing single-key/global fallback per-row inside features.py is "
        "invasive enough to warrant a separate review.",
        "",
        "## Results",
        "",
        "| model | val_setup | NDCG@5 | Recall@5 | MBR | best_iter | n_feat |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md.append(f"| TE_SAFE | {r['val_setup']} | {r['ndcg5']:.5f} | "
                  f"{r['recall5']:.4f} | {r['mean_booked_rank']:.3f} | "
                  f"{int(r['best_iteration'])} | {int(r['n_features'])} |")
    md += [
        "",
        "## Decision rule (carry from spec)",
        "",
        "- Prefer **temporal** over random validation.",
        "- TE_SAFE beats B3 on **temporal**, even if random is flat → good submission candidate.",
        "- TE_SAFE beats B3 on **random only** → DO NOT submit.",
        "- TE_SAFE similar on both → keep as candidate, do not rush submit.",
        "- TE_SAFE worse on **temporal** → pivot to feature pruning / position-bias.",
        "",
        "## Files",
        "",
        "- `te_safe_results.csv` — 2 rows (random_control, temporal).",
        "- `split_meta.json` — leakage assertions + smoothing/rank config.",
        "- `model_TE_SAFE_<setup>.txt` — 2 LightGBM boosters.",
        "- `../../artifacts/te_safe_<TS>/` — feature_cols + importance + run_summary.",
    ]
    (out_dir / "README.md").write_text("\n".join(md))
    log(f"README:    {out_dir / 'README.md'}")

    (art_dir / "run_summary.md").write_text((out_dir / "README.md").read_text())

    log("\n" + "=" * 70)
    log("FINAL — TE_SAFE results")
    log("=" * 70)
    for _, r in df.iterrows():
        log(f"  TE_SAFE | {r['val_setup']:14s} | NDCG@5={r['ndcg5']:.5f}  "
            f"Recall@5={r['recall5']:.4f}  MBR={r['mean_booked_rank']:.3f}  "
            f"best_iter={int(r['best_iteration'])}")
    log(f"\nElapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
