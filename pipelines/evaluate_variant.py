"""
evaluate_variant.py — minimal temporal-validation harness for single feature variants.

One variant at a time. V4_ANCHOR config (label_gain=0,1,15, IPW default).
Temporal validation only. No random-control. No submission. No ensembles.

The only baseline reference: V4_ANCHOR temporal NDCG@5 = 0.40401.

Variants:
  --variant drop_prop_avg_position
      remove `prop_avg_position` if present.
  --variant te_rank
      add `<feature>_rank_norm` (rank within srch_id, normalized to [0,1]) for the
      12 TE features listed in TE_FEATURES. Missing features are skipped and logged.
  --variant drop_prop_avg_position_plus_te_rank
      both of the above combined.
  --variant prop_click_rate_pos_adj_s40_oof
      same recipe as `prop_click_rate_pos_adj_s40` but with 5-fold OOF on the
      temporal_train side (folds split by srch_id, seed=42). For each fold the
      per-prop_id aggregate is computed from the other 4 folds only and assigned
      to the held-out fold. The val side uses the per-prop_id aggregate computed
      on all of temporal_train. The original `prop_click_rate` is kept.
  --variant prop_rel_rate_pos_adj_s40_oof
      same OOF mechanic but with target=relevance. Original `prop_rel_rate` kept.
  --variant prop_book_rate_pos_adj_s40_oof
      same OOF mechanic but with target=booking_bool. Original `prop_book_rate` kept.
  --variant prop_click_posadj_plus_te_rank
      combines `prop_click_rate_pos_adj_s40_oof` AND the 12 `te_rank` pct-rank
      features in one model. Originals kept. Other pos-adj OOF siblings NOT added.
  --variant price_vs_mean_x_dest_click
      adds `price_vs_mean_x_dest_click = price_vs_mean * dest_click_rate`.
      Motivated by EDA: booked-hotel `price_vs_mean` shifts ~94 USD between
      low and high `dest_click_rate` buckets.
  --variant price_vs_prop_mean_x_dest_click
      same idea but with the stronger price-deviation parent `price_vs_prop_mean`
      (rank #5 by gain in current model, vs #42 for `price_vs_mean`).
  --variant prop_price_zscore_clipped_x_dest_click
      `clip(prop_price_zscore, -5, 5) * dest_click_rate`. Clipping the parent
      removes the extreme tail (max train value ~375K, max val ~950K) so the
      product distribution stays within the bulk of the data.
  --variant prop_click_posadj_plus_price_dest
      stacks two independent winners: `prop_click_rate_pos_adj_s40_oof` (the
      OOF position-adjusted click TE) AND
      `prop_price_zscore_clipped_x_dest_click` (the bounded price × destination
      interaction). Different feature families — tests whether they add cleanly.
  --variant prop_dest_book_rate_safe
      smoothed (prop_id, srch_destination_id) booking rate with a 3-way fallback
      (per-prop, per-dest, global). 5-fold OOF on temporal_train (by srch_id),
      full-train aggregate on val. alpha=40. Keeps raw `prop_dest_book_rate`.
  --variant prop_click_posadj_plus_prop_dest_safe
      stacks `prop_click_rate_pos_adj_s40_oof` AND `prop_dest_book_rate_safe`.
      Both have low train/val drift — tests whether they compose cleanly.
  --variant replace_prop_click_with_posadj
      replaces raw `prop_click_rate` with `prop_click_rate_pos_adj_s40_oof`
      (add the OOF version, drop the raw). Tests whether replacement is
      better than addition (which has been anti-additive in earlier combos).
  --variant prop_click_rate_pos_adj_s40
      add a position-adjusted, smoothed `prop_click_rate_pos_adj_s40` alongside
      the original `prop_click_rate`. Aggregation source: temporal_train only.
      Steps:
        global_click_rate = mean(click_bool) on temporal_train
        position_click_rate[p] = mean(click_bool) per position on temporal_train
        exposure_weight = clip(global_click_rate / position_click_rate[position],
                               0.2, 3.0)
        per prop_id: weighted_click_sum = Σ exposure_weight · click_bool
                     weighted_exposure_sum = Σ exposure_weight
        prop_click_rate_pos_adj_s40 =
            (weighted_click_sum + 40·global_click_rate) /
            (weighted_exposure_sum + 40)
        Joined to train (self-included, smoothing handles leakage) and val
        (val maps from train-only stats). The original `prop_click_rate` is kept.

Splits reuse `pipelines.temporal_validation.temporal_split`, which is identical to
the one used in `diagnostics/temporal_validation_20260516_113831/`. We assert
against that directory's split_meta.json if present.

Cache (optional):
  diagnostics/eval_variants/base_features_temporal_train.parquet
  diagnostics/eval_variants/base_features_temporal_val.parquet
  diagnostics/eval_variants/base_features_meta.json
First run builds + caches. Subsequent runs read parquet directly (~30s instead of ~4 min).
If sidecar JSON disagrees with current split, cache is rebuilt with a warning.

Outputs:
  diagnostics/eval_variants/results.csv                    (one row appended per run)
  diagnostics/eval_variants/model_<variant>_temporal.txt
  diagnostics/eval_variants/feature_importance_<variant>_temporal.csv
  diagnostics/eval_variants/feature_drift_<variant>.csv
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
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_train, make_target, get_feature_columns  # noqa: E402
from src.features import build_features, compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402

# Reuse split + helpers from STEP 1 for exact parity
from pipelines.temporal_validation import (  # noqa: E402
    temporal_split, compute_ipw_weights, make_group_counts, eval_metrics,
    BASE_PARAMS,
)

# ============================================================================
# Constants
# ============================================================================
V4_ANCHOR_TEMPORAL = 0.40401          # the only baseline reference
B3_TEMPORAL = 0.40398                 # FYI only

OUT_DIR = ROOT / "diagnostics" / "eval_variants"
CACHE_TRAIN = OUT_DIR / "base_features_temporal_train.parquet"
CACHE_VAL = OUT_DIR / "base_features_temporal_val.parquet"
CACHE_META = OUT_DIR / "base_features_meta.json"
RESULTS_CSV = OUT_DIR / "results.csv"

REFERENCE_SPLIT_META = ROOT / "diagnostics" / "temporal_validation_20260516_113831" / "split_meta.json"

TE_FEATURES = [
    "prop_click_rate", "prop_book_rate", "prop_rel_rate",
    "dest_click_rate", "dest_book_rate",
    "country_book_rate", "site_book_rate",
    "prop_dest_book_rate", "prop_site_book_rate",
    "site_dest_book_rate", "site_country_book_rate", "cpair_book_rate",
]

VALID_VARIANTS = [
    "drop_prop_avg_position",
    "te_rank",
    "drop_prop_avg_position_plus_te_rank",
    "prop_click_rate_pos_adj_s40",
    "prop_click_rate_pos_adj_s40_oof",
    "prop_rel_rate_pos_adj_s40_oof",
    "prop_book_rate_pos_adj_s40_oof",
    "prop_click_posadj_plus_te_rank",
    "price_vs_mean_x_dest_click",
    "price_vs_prop_mean_x_dest_click",
    "prop_price_zscore_clipped_x_dest_click",
    "prop_click_posadj_plus_price_dest",
    "prop_dest_book_rate_safe",
    "prop_click_posadj_plus_prop_dest_safe",
    "replace_prop_click_with_posadj",
]

# Maps a generated OOF feature → the original raw TE it should be compared against.
OOF_TO_RAW = {
    "prop_click_rate_pos_adj_s40_oof": "prop_click_rate",
    "prop_rel_rate_pos_adj_s40_oof":   "prop_rel_rate",
    "prop_book_rate_pos_adj_s40_oof":  "prop_book_rate",
    "prop_dest_book_rate_safe":        "prop_dest_book_rate",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Plan
# ============================================================================
PLAN = f"""\
evaluate_variant.py — execution plan
=====================================

Reads:
  data/training_set_VU_DM.csv                                            (4.96M rows; date_time required)
  diagnostics/temporal_validation_20260516_113831/split_meta.json        (parity check; optional)
  diagnostics/eval_variants/base_features_temporal_*.parquet             (if cache valid)

Writes (under diagnostics/eval_variants/):
  base_features_temporal_train.parquet      (first run only)
  base_features_temporal_val.parquet        (first run only)
  base_features_meta.json                   (sidecar)
  results.csv                               (appended row per --variant run)
  model_<variant>_temporal.txt              (one booster)
  feature_importance_<variant>_temporal.csv
  feature_drift_<variant>.csv

Estimated runtime:
  First run  (cold cache) : ~8–10 min   (3–4 min train feats + 1–2 min val feats + ~3–4 min LGBM)
  Subsequent runs (warm)  : ~4–5 min    (parquet load 20–40s + ~3–4 min LGBM)

Commands:
  uv run python evaluate_variant.py --variant drop_prop_avg_position
  uv run python evaluate_variant.py --variant te_rank
  uv run python evaluate_variant.py --variant drop_prop_avg_position_plus_te_rank

Hard constraints honored:
  - Temporal validation ONLY (no random-control here).
  - V4_ANCHOR config (lg=0,1,15, IPW default) ONLY.
  - One variant per run; no other modeling work.
  - No submission. No ensemble. No new splits.
  - All val features built with agg_source=temporal_train, is_train=False (leakage-safe).

Decision rule (printed at end):
  REJECT       if delta < 0
  HOLD         if  0 <= delta < +0.001
  KEEP         if delta >= +0.001
  STRONG_KEEP  if delta >= +0.002
"""


# ============================================================================
# Cache management
# ============================================================================
def cache_is_valid(expected_meta: dict) -> bool:
    if not (CACHE_TRAIN.exists() and CACHE_VAL.exists() and CACHE_META.exists()):
        return False
    try:
        meta = json.load(open(CACHE_META))
    except Exception as e:
        log(f"  ! could not read cache sidecar ({e}); will rebuild.")
        return False
    keys = ("n_train_rows", "n_val_rows", "n_train_searches", "n_val_searches")
    for k in keys:
        if meta.get(k) != expected_meta.get(k):
            log(f"  ! cache mismatch on {k}: cached={meta.get(k)}  expected={expected_meta.get(k)}; will rebuild.")
            return False
    return True


def build_and_cache(temporal_train_df, temporal_val_df):
    log("Building base features (no variant applied)…")
    t0 = time.time()
    train_feat = build_features(temporal_train_df, agg_source=temporal_train_df, is_train=True)
    log(f"  train_feat: {len(train_feat):,} rows × {train_feat.shape[1]} cols ({(time.time()-t0)/60:.1f} min)")
    t1 = time.time()
    val_feat = build_features(temporal_val_df, agg_source=temporal_train_df, is_train=False)
    log(f"  val_feat:   {len(val_feat):,} rows × {val_feat.shape[1]} cols ({(time.time()-t1)/60:.1f} min)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Writing parquet cache…")
    train_feat.to_parquet(CACHE_TRAIN, index=False)
    val_feat.to_parquet(CACHE_VAL, index=False)
    meta = {
        "n_train_rows": int(len(train_feat)),
        "n_val_rows": int(len(val_feat)),
        "n_train_searches": int(train_feat["srch_id"].nunique()),
        "n_val_searches": int(val_feat["srch_id"].nunique()),
        "feature_count": int(train_feat.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(meta, open(CACHE_META, "w"), indent=2)
    log(f"  cached: {CACHE_TRAIN.name}, {CACHE_VAL.name}, {CACHE_META.name}")
    return train_feat, val_feat


def load_cache():
    log("Loading base features from parquet cache…")
    t0 = time.time()
    train_feat = pd.read_parquet(CACHE_TRAIN)
    val_feat = pd.read_parquet(CACHE_VAL)
    log(f"  loaded train={len(train_feat):,} val={len(val_feat):,} in {time.time()-t0:.1f}s")
    return train_feat, val_feat


# ============================================================================
# Variant logic
# ============================================================================
def _pos_adj_oof_te(train_feat: pd.DataFrame, val_feat: pd.DataFrame,
                    target_col: str, col_new: str, *,
                    alpha: float = 40.0, clip: tuple = (0.2, 3.0),
                    n_folds: int = 5, seed: int = 42) -> None:
    """Position-adjusted, 5-fold OOF (by srch_id) smoothed per-prop_id rate.
    Train side: held-out fold's value is computed from the OTHER 4 folds.
    Val side: aggregate is computed on the FULL temporal_train.
    Both sides smoothed with `alpha` against the global rate.
    """
    for required in ("prop_id", "position", "srch_id"):
        assert required in train_feat.columns, f"missing '{required}' in train_feat"
    # val side only uses prop_id; position/srch_id are train-only.
    assert "prop_id" in val_feat.columns, "missing 'prop_id' in val_feat"
    assert target_col in train_feat.columns, \
        f"missing target column '{target_col}' in train_feat"

    global_rate = float(train_feat[target_col].mean())
    pos_rate = train_feat.groupby("position")[target_col].mean()
    n_zero = int((pos_rate == 0).sum())
    pos_rate_safe = pos_rate.where(pos_rate > 0, global_rate)
    log(f"  global_{target_col}_rate = {global_rate:.6f}")
    log(f"  position_{target_col}_rate: {len(pos_rate)} positions "
        f"(min={pos_rate.min():.6f}, max={pos_rate.max():.6f}, "
        f"zero_positions={n_zero} → fallback to global)")

    ew_train = (
        global_rate
        / train_feat["position"].map(pos_rate_safe).astype(np.float64)
    ).clip(lower=clip[0], upper=clip[1])
    log(f"  exposure_weight (train): "
        f"min={ew_train.min():.3f}  max={ew_train.max():.3f}  mean={ew_train.mean():.3f}")

    srch_unique = train_feat["srch_id"].unique()
    rng = np.random.RandomState(seed)
    srch_fold = pd.Series(
        rng.randint(0, n_folds, size=len(srch_unique)), index=srch_unique
    )
    fold_ids = train_feat["srch_id"].map(srch_fold).values
    _check = pd.DataFrame({"srch_id": train_feat["srch_id"].values, "fold": fold_ids})
    assert _check.groupby("srch_id")["fold"].nunique().max() == 1, \
        "srch_id split leakage in fold assignment"
    del _check

    y = train_feat[target_col].values.astype(np.float64)
    w = ew_train.values
    wt = w * y
    tmp = pd.DataFrame(
        {"prop_id": train_feat["prop_id"].values, "w": w, "wt": wt},
        index=train_feat.index,
    )

    oof_result = pd.Series(np.nan, index=train_feat.index, dtype=np.float64)
    for fold in range(n_folds):
        mask = (fold_ids == fold)
        oof = tmp.loc[~mask]
        agg = oof.groupby("prop_id").agg(w_sum=("w", "sum"), wt_sum=("wt", "sum"))
        agg["enc"] = (
            (agg["wt_sum"] + alpha * global_rate) / (agg["w_sum"] + alpha)
        )
        mapped = train_feat.loc[mask, "prop_id"].map(agg["enc"])
        log(f"  fold {fold}: {int(mask.sum()):,} held-out rows, "
            f"{int(mapped.isna().sum()):,} prop_ids unseen in other folds (→ global)")
        oof_result.loc[mask] = mapped.values
    train_feat[col_new] = oof_result.fillna(global_rate).astype(np.float32)

    full = tmp.groupby("prop_id").agg(w_sum=("w", "sum"), wt_sum=("wt", "sum"))
    full["enc"] = (
        (full["wt_sum"] + alpha * global_rate) / (full["w_sum"] + alpha)
    ).astype(np.float32)
    val_feat[col_new] = (
        val_feat["prop_id"].map(full["enc"]).fillna(global_rate).astype(np.float32)
    )
    val_cold = int(val_feat["prop_id"].map(full["enc"]).isna().sum())
    log(f"  val cold-start prop_ids (filled with global): {val_cold:,}")
    log(f"  joined '{col_new}'  "
        f"mean(train)={train_feat[col_new].mean():.5f}  "
        f"mean(val)={val_feat[col_new].mean():.5f}")


def _prop_dest_book_rate_safe(train_feat: pd.DataFrame, val_feat: pd.DataFrame, *,
                              col_new: str = "prop_dest_book_rate_safe",
                              alpha: float = 40.0, n_folds: int = 5,
                              seed: int = 42) -> None:
    """Smoothed (prop_id, srch_destination_id) booking rate with a 3-way fallback.

    Train side: 5-fold OOF by srch_id. For each held-out fold, all stats
      (per-pair, per-prop, per-dest, global) are computed from the OTHER 4 folds.
    Val side: stats computed from full temporal_train and joined.

    fallback_rate = 0.5·prop_rate + 0.3·dest_rate + 0.2·global_rate
    feature       = (pair_book + alpha·fallback_rate) / (pair_count + alpha)
    """
    for required in ("prop_id", "srch_destination_id", "srch_id", "booking_bool"):
        assert required in train_feat.columns, f"missing '{required}' in train_feat"
    for required in ("prop_id", "srch_destination_id"):
        assert required in val_feat.columns, f"missing '{required}' in val_feat"

    srch_unique = train_feat["srch_id"].unique()
    rng = np.random.RandomState(seed)
    srch_fold = pd.Series(
        rng.randint(0, n_folds, size=len(srch_unique)), index=srch_unique
    )
    fold_ids = train_feat["srch_id"].map(srch_fold).values
    _check = pd.DataFrame({"srch_id": train_feat["srch_id"].values, "fold": fold_ids})
    assert _check.groupby("srch_id")["fold"].nunique().max() == 1, \
        "srch_id split leakage in fold assignment"
    del _check
    fold_counts = pd.Series(fold_ids).value_counts().sort_index()
    log(f"  fold row counts: {dict(fold_counts)}")

    src_df = pd.DataFrame({
        "prop_id": train_feat["prop_id"].values,
        "srch_destination_id": train_feat["srch_destination_id"].values,
        "booking_bool": train_feat["booking_bool"].values.astype(np.float64),
    }, index=train_feat.index)

    def _stats(src: pd.DataFrame):
        n = len(src)
        assert n > 0, "empty source"
        global_rate = float(src["booking_bool"].sum() / n)
        prop_g = src.groupby("prop_id")["booking_bool"].agg(["sum", "count"])
        dest_g = src.groupby("srch_destination_id")["booking_bool"].agg(["sum", "count"])
        pair_g = src.groupby(
            ["prop_id", "srch_destination_id"]
        )["booking_bool"].agg(["sum", "count"])
        prop_rate = (prop_g["sum"] / prop_g["count"]).astype(np.float64)
        dest_rate = (dest_g["sum"] / dest_g["count"]).astype(np.float64)
        return global_rate, prop_rate, dest_rate, pair_g

    def _apply(target_df, global_rate, prop_rate, dest_rate, pair_g):
        pr_fb = target_df["prop_id"].map(prop_rate).fillna(global_rate).astype(np.float64).values
        dr_fb = target_df["srch_destination_id"].map(dest_rate).fillna(global_rate).astype(np.float64).values
        fallback = 0.5 * pr_fb + 0.3 * dr_fb + 0.2 * global_rate
        keys = pd.MultiIndex.from_arrays(
            [target_df["prop_id"].values, target_df["srch_destination_id"].values]
        )
        pair_book = pair_g["sum"].reindex(keys).fillna(0).values.astype(np.float64)
        pair_count = pair_g["count"].reindex(keys).fillna(0).values.astype(np.float64)
        feat = (pair_book + alpha * fallback) / (pair_count + alpha)
        return feat.astype(np.float32), pair_count

    oof_result = np.empty(len(train_feat), dtype=np.float32)
    for fold in range(n_folds):
        mask = (fold_ids == fold)
        oof_src = src_df.loc[~mask]
        g, pr, dr, pg = _stats(oof_src)
        held = train_feat.loc[mask, ["prop_id", "srch_destination_id"]]
        feat_h, pc_h = _apply(held, g, pr, dr, pg)
        oof_result[mask.nonzero()[0]] = feat_h
        log(f"  fold {fold}: {int(mask.sum()):,} held-out rows, "
            f"global={g:.5f}, n_pairs={len(pg):,}, "
            f"unseen_pairs_in_held={int((pc_h == 0).sum()):,} (→ fallback only)")
    train_feat[col_new] = oof_result

    g, pr, dr, pg = _stats(src_df)
    log(f"  val source (full temporal_train): global={g:.5f}, "
        f"n_pairs={len(pg):,}, n_props={len(pr):,}, n_dests={len(dr):,}")
    feat_v, pc_v = _apply(val_feat[["prop_id", "srch_destination_id"]], g, pr, dr, pg)
    val_feat[col_new] = feat_v
    n_unseen = int((pc_v == 0).sum())
    log(f"  val: {n_unseen:,} rows with unseen (prop, dest) pairs (→ fallback only) "
        f"= {n_unseen / len(val_feat):.2%}")
    log(f"  joined '{col_new}'  "
        f"mean(train)={train_feat[col_new].mean():.5f}  "
        f"mean(val)={val_feat[col_new].mean():.5f}  "
        f"std(train)={train_feat[col_new].std():.5f}")


def apply_variant(variant: str, train_feat: pd.DataFrame, val_feat: pd.DataFrame
                  ) -> tuple[list[str], list[str], list[str]]:
    """Mutates train_feat and val_feat in-place (adds/removes columns).
    Returns (added_cols, removed_cols, skipped_cols)."""
    added, removed, skipped = [], [], []

    if variant in ("drop_prop_avg_position", "drop_prop_avg_position_plus_te_rank"):
        col = "prop_avg_position"
        if col in train_feat.columns:
            removed.append(col)
            log(f"  variant: will REMOVE '{col}' from feature_cols")
        else:
            skipped.append(col)
            log(f"  variant: '{col}' not in features (already absent)")

    if variant == "prop_click_rate_pos_adj_s40":
        col_new = "prop_click_rate_pos_adj_s40"
        alpha = 40.0
        clip_lo, clip_hi = 0.2, 3.0
        for required in ("prop_id", "click_bool", "position"):
            assert required in train_feat.columns, f"missing '{required}' in train_feat"
            assert required in val_feat.columns, f"missing '{required}' in val_feat"

        global_click_rate = float(train_feat["click_bool"].mean())
        pos_click_rate = train_feat.groupby("position")["click_bool"].mean()
        log(f"  global_click_rate = {global_click_rate:.6f}")
        log(f"  position_click_rate: {len(pos_click_rate)} positions "
            f"(min={pos_click_rate.min():.5f}, max={pos_click_rate.max():.5f})")

        ew_train = (
            global_click_rate
            / train_feat["position"].map(pos_click_rate).astype(np.float64)
        ).clip(lower=clip_lo, upper=clip_hi)
        log(f"  exposure_weight (train): "
            f"min={ew_train.min():.3f}  max={ew_train.max():.3f}  mean={ew_train.mean():.3f}")

        tmp = pd.DataFrame({
            "prop_id": train_feat["prop_id"].values,
            "w": ew_train.values,
            "wc": ew_train.values * train_feat["click_bool"].values.astype(np.float64),
        })
        agg = tmp.groupby("prop_id").agg(w_sum=("w", "sum"), wc_sum=("wc", "sum"))
        agg[col_new] = (
            (agg["wc_sum"] + alpha * global_click_rate)
            / (agg["w_sum"] + alpha)
        ).astype(np.float32)
        mapping = agg[col_new]
        log(f"  per-prop_id smoothed values: n={len(mapping):,}  "
            f"mean={mapping.mean():.5f}  std={mapping.std():.5f}")

        train_feat[col_new] = (
            train_feat["prop_id"].map(mapping).fillna(global_click_rate).astype(np.float32)
        )
        val_feat[col_new] = (
            val_feat["prop_id"].map(mapping).fillna(global_click_rate).astype(np.float32)
        )
        val_unmapped = val_feat[col_new].isna().sum()
        assert val_unmapped == 0, f"unexpected NaN after fillna on val: {val_unmapped}"
        log(f"  joined '{col_new}'  "
            f"mean(train)={train_feat[col_new].mean():.5f}  "
            f"mean(val)={val_feat[col_new].mean():.5f}  "
            f"(original prop_click_rate kept)")
        added.append(col_new)

    if variant == "price_vs_mean_x_dest_click":
        col_new = "price_vs_mean_x_dest_click"
        for required in ("price_vs_mean", "dest_click_rate"):
            assert required in train_feat.columns, f"missing '{required}' in train_feat"
            assert required in val_feat.columns, f"missing '{required}' in val_feat"
        train_feat[col_new] = (
            train_feat["price_vs_mean"].astype(np.float64)
            * train_feat["dest_click_rate"].astype(np.float64)
        ).astype(np.float32)
        val_feat[col_new] = (
            val_feat["price_vs_mean"].astype(np.float64)
            * val_feat["dest_click_rate"].astype(np.float64)
        ).astype(np.float32)
        log(f"  joined '{col_new}'  "
            f"mean(train)={train_feat[col_new].mean():.4f}  "
            f"mean(val)={val_feat[col_new].mean():.4f}  "
            f"std(train)={train_feat[col_new].std():.4f}")
        added.append(col_new)

    if variant == "price_vs_prop_mean_x_dest_click":
        col_new = "price_vs_prop_mean_x_dest_click"
        for required in ("price_vs_prop_mean", "dest_click_rate"):
            assert required in train_feat.columns, f"missing '{required}' in train_feat"
            assert required in val_feat.columns, f"missing '{required}' in val_feat"
        train_feat[col_new] = (
            train_feat["price_vs_prop_mean"].astype(np.float64)
            * train_feat["dest_click_rate"].astype(np.float64)
        ).astype(np.float32)
        val_feat[col_new] = (
            val_feat["price_vs_prop_mean"].astype(np.float64)
            * val_feat["dest_click_rate"].astype(np.float64)
        ).astype(np.float32)
        log(f"  joined '{col_new}'  "
            f"mean(train)={train_feat[col_new].mean():.4f}  "
            f"mean(val)={val_feat[col_new].mean():.4f}  "
            f"std(train)={train_feat[col_new].std():.4f}")
        added.append(col_new)

    if variant == "prop_price_zscore_clipped_x_dest_click":
        col_new = "prop_price_zscore_clipped_x_dest_click"
        clip_lo, clip_hi = -5.0, 5.0
        for required in ("prop_price_zscore", "dest_click_rate"):
            assert required in train_feat.columns, f"missing '{required}' in train_feat"
            assert required in val_feat.columns, f"missing '{required}' in val_feat"
        zs_tr = train_feat["prop_price_zscore"].astype(np.float64).clip(clip_lo, clip_hi)
        zs_va = val_feat["prop_price_zscore"].astype(np.float64).clip(clip_lo, clip_hi)
        log(f"  clip(prop_price_zscore, {clip_lo}, {clip_hi}): "
            f"train clipped_lo={int((train_feat['prop_price_zscore']<clip_lo).sum()):,} "
            f"clipped_hi={int((train_feat['prop_price_zscore']>clip_hi).sum()):,}  "
            f"val clipped_lo={int((val_feat['prop_price_zscore']<clip_lo).sum()):,} "
            f"clipped_hi={int((val_feat['prop_price_zscore']>clip_hi).sum()):,}")
        train_feat[col_new] = (zs_tr * train_feat["dest_click_rate"].astype(np.float64)).astype(np.float32)
        val_feat[col_new] = (zs_va * val_feat["dest_click_rate"].astype(np.float64)).astype(np.float32)
        log(f"  joined '{col_new}'  "
            f"mean(train)={train_feat[col_new].mean():.4f}  "
            f"mean(val)={val_feat[col_new].mean():.4f}  "
            f"std(train)={train_feat[col_new].std():.4f}  "
            f"null(train)={train_feat[col_new].isna().mean():.4%}  "
            f"null(val)={val_feat[col_new].isna().mean():.4%}")
        added.append(col_new)

    if variant == "prop_rel_rate_pos_adj_s40_oof":
        col_new = "prop_rel_rate_pos_adj_s40_oof"
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="relevance", col_new=col_new,
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        log("  (original prop_rel_rate kept)")
        added.append(col_new)

    if variant == "prop_book_rate_pos_adj_s40_oof":
        col_new = "prop_book_rate_pos_adj_s40_oof"
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="booking_bool", col_new=col_new,
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        log("  (original prop_book_rate kept)")
        added.append(col_new)

    if variant == "prop_click_rate_pos_adj_s40_oof":
        col_new = "prop_click_rate_pos_adj_s40_oof"
        alpha = 40.0
        clip_lo, clip_hi = 0.2, 3.0
        n_folds = 5
        seed = 42
        for required in ("prop_id", "click_bool", "position", "srch_id"):
            assert required in train_feat.columns, f"missing '{required}' in train_feat"
            assert required in val_feat.columns or required == "click_bool", \
                f"missing '{required}' in val_feat"

        global_click_rate = float(train_feat["click_bool"].mean())
        pos_click_rate = train_feat.groupby("position")["click_bool"].mean()
        log(f"  global_click_rate = {global_click_rate:.6f}")
        log(f"  position_click_rate: {len(pos_click_rate)} positions "
            f"(min={pos_click_rate.min():.5f}, max={pos_click_rate.max():.5f})")

        ew_train = (
            global_click_rate
            / train_feat["position"].map(pos_click_rate).astype(np.float64)
        ).clip(lower=clip_lo, upper=clip_hi)
        log(f"  exposure_weight (train): "
            f"min={ew_train.min():.3f}  max={ew_train.max():.3f}  mean={ew_train.mean():.3f}")

        # Fold assignment by srch_id (same as src/features.py:_get_srch_folds)
        srch_unique = train_feat["srch_id"].unique()
        rng = np.random.RandomState(seed)
        srch_fold = pd.Series(
            rng.randint(0, n_folds, size=len(srch_unique)), index=srch_unique
        )
        fold_ids = train_feat["srch_id"].map(srch_fold).values
        # Hard guarantee: rows from the same srch_id share a fold.
        _check = pd.DataFrame({"srch_id": train_feat["srch_id"].values, "fold": fold_ids})
        assert _check.groupby("srch_id")["fold"].nunique().max() == 1, \
            "srch_id split leakage in fold assignment"
        del _check
        fold_counts = pd.Series(fold_ids).value_counts().sort_index()
        log(f"  fold row counts: {dict(fold_counts)}")

        click_arr = train_feat["click_bool"].values.astype(np.float64)
        w_arr = ew_train.values
        wc_arr = w_arr * click_arr
        tmp = pd.DataFrame(
            {"prop_id": train_feat["prop_id"].values, "w": w_arr, "wc": wc_arr,
             "_fold": fold_ids},
            index=train_feat.index,
        )

        # 5-fold OOF on train: for each held-out fold, sum over the other 4 folds.
        oof_result = pd.Series(np.nan, index=train_feat.index, dtype=np.float64)
        oof_fill_counts = []
        for fold in range(n_folds):
            mask = (fold_ids == fold)
            oof = tmp.loc[~mask]
            agg = oof.groupby("prop_id").agg(w_sum=("w", "sum"), wc_sum=("wc", "sum"))
            agg["enc"] = (
                (agg["wc_sum"] + alpha * global_click_rate)
                / (agg["w_sum"] + alpha)
            )
            mapped = train_feat.loc[mask, "prop_id"].map(agg["enc"])
            n_nan = int(mapped.isna().sum())
            oof_fill_counts.append((fold, int(mask.sum()), n_nan))
            oof_result.loc[mask] = mapped.values
        for fold, n_rows, n_nan in oof_fill_counts:
            log(f"  fold {fold}: {n_rows:,} held-out rows, "
                f"{n_nan:,} prop_ids unseen in other folds (→ global_click_rate)")
        train_feat[col_new] = oof_result.fillna(global_click_rate).astype(np.float32)

        # Val: per-prop_id aggregate from the FULL temporal_train.
        full = tmp.groupby("prop_id").agg(w_sum=("w", "sum"), wc_sum=("wc", "sum"))
        full["enc"] = (
            (full["wc_sum"] + alpha * global_click_rate)
            / (full["w_sum"] + alpha)
        ).astype(np.float32)
        val_feat[col_new] = (
            val_feat["prop_id"].map(full["enc"]).fillna(global_click_rate).astype(np.float32)
        )
        val_cold = int(val_feat["prop_id"].map(full["enc"]).isna().sum())
        log(f"  val cold-start prop_ids (filled with global_click_rate): {val_cold:,}")
        log(f"  joined '{col_new}'  "
            f"mean(train)={train_feat[col_new].mean():.5f}  "
            f"mean(val)={val_feat[col_new].mean():.5f}  "
            f"(original prop_click_rate kept)")
        added.append(col_new)

    if variant == "prop_dest_book_rate_safe":
        _prop_dest_book_rate_safe(train_feat, val_feat,
                                  col_new="prop_dest_book_rate_safe",
                                  alpha=40.0, n_folds=5, seed=42)
        log("  (original prop_dest_book_rate kept)")
        added.append("prop_dest_book_rate_safe")

    if variant == "replace_prop_click_with_posadj":
        col_new = "prop_click_rate_pos_adj_s40_oof"
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="click_bool", col_new=col_new,
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        added.append(col_new)
        raw_col = "prop_click_rate"
        if raw_col in train_feat.columns:
            removed.append(raw_col)
            log(f"  variant: will REMOVE raw '{raw_col}' from feature_cols "
                f"(replaced by '{col_new}')")
        else:
            skipped.append(raw_col)
            log(f"  variant: raw '{raw_col}' not in features (skip)")

    if variant == "prop_click_posadj_plus_prop_dest_safe":
        # Part 1: OOF position-adjusted click TE.
        col_a = "prop_click_rate_pos_adj_s40_oof"
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="click_bool", col_new=col_a,
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        log("  (original prop_click_rate kept)")
        added.append(col_a)

        # Part 2: smoothed (prop, dest) booking rate with 3-way fallback.
        col_b = "prop_dest_book_rate_safe"
        _prop_dest_book_rate_safe(train_feat, val_feat,
                                  col_new=col_b,
                                  alpha=40.0, n_folds=5, seed=42)
        log("  (original prop_dest_book_rate kept)")
        added.append(col_b)

    if variant == "prop_click_posadj_plus_price_dest":
        # Part 1: OOF position-adjusted click TE (the validated KEEP feature).
        col_oof = "prop_click_rate_pos_adj_s40_oof"
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="click_bool", col_new=col_oof,
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        log("  (original prop_click_rate kept)")
        added.append(col_oof)

        # Part 2: bounded price × dest_click interaction.
        col_int = "prop_price_zscore_clipped_x_dest_click"
        clip_lo, clip_hi = -5.0, 5.0
        for required in ("prop_price_zscore", "dest_click_rate"):
            assert required in train_feat.columns, f"missing '{required}' in train_feat"
            assert required in val_feat.columns, f"missing '{required}' in val_feat"
        zs_tr = train_feat["prop_price_zscore"].astype(np.float64).clip(clip_lo, clip_hi)
        zs_va = val_feat["prop_price_zscore"].astype(np.float64).clip(clip_lo, clip_hi)
        log(f"  clip(prop_price_zscore, {clip_lo}, {clip_hi}): "
            f"train clipped_lo={int((train_feat['prop_price_zscore']<clip_lo).sum()):,} "
            f"clipped_hi={int((train_feat['prop_price_zscore']>clip_hi).sum()):,}  "
            f"val clipped_lo={int((val_feat['prop_price_zscore']<clip_lo).sum()):,} "
            f"clipped_hi={int((val_feat['prop_price_zscore']>clip_hi).sum()):,}")
        train_feat[col_int] = (zs_tr * train_feat["dest_click_rate"].astype(np.float64)).astype(np.float32)
        val_feat[col_int] = (zs_va * val_feat["dest_click_rate"].astype(np.float64)).astype(np.float32)
        log(f"  joined '{col_int}'  "
            f"mean(train)={train_feat[col_int].mean():.4f}  "
            f"mean(val)={val_feat[col_int].mean():.4f}  "
            f"std(train)={train_feat[col_int].std():.4f}")
        added.append(col_int)

    if variant == "prop_click_posadj_plus_te_rank":
        col_new = "prop_click_rate_pos_adj_s40_oof"
        _pos_adj_oof_te(train_feat, val_feat,
                        target_col="click_bool", col_new=col_new,
                        alpha=40.0, clip=(0.2, 3.0), n_folds=5, seed=42)
        log("  (original prop_click_rate kept)")
        added.append(col_new)

    if variant in ("te_rank", "drop_prop_avg_position_plus_te_rank",
                    "prop_click_posadj_plus_te_rank"):
        for f in TE_FEATURES:
            if f not in train_feat.columns or f not in val_feat.columns:
                skipped.append(f)
                log(f"  variant: SKIP '{f}' (not in features)")
                continue
            new = f"{f}_rank_norm"
            train_feat[new] = train_feat.groupby("srch_id")[f].rank(
                pct=True, method="average"
            ).astype(np.float32)
            val_feat[new] = val_feat.groupby("srch_id")[f].rank(
                pct=True, method="average"
            ).astype(np.float32)
            added.append(new)
        if added:
            log(f"  variant: ADDED {len(added)} pct-rank columns")
    return added, removed, skipped


def feature_drift_table(train_feat, val_feat, added, removed) -> pd.DataFrame:
    """For each added or removed feature, report mean/std/null stats on train vs val."""
    rows = []
    for f in added + removed:
        action = "added" if f in added else "removed"
        # The feature must still exist in both — for removed columns we report
        # its stats before removal from feature_cols (column is still in the
        # DataFrame at this point).
        if f not in train_feat.columns or f not in val_feat.columns:
            continue
        t = train_feat[f].astype("float64")
        v = val_feat[f].astype("float64")
        t_mean = float(t.mean())
        v_mean = float(v.mean())
        t_std = float(t.std())
        delta = abs(v_mean - t_mean) / (t_std + 1e-12)
        t_null = float(t.isna().mean())
        v_null = float(v.isna().mean())
        rows.append({
            "feature": f,
            "action": action,
            "train_mean": t_mean,
            "val_mean": v_mean,
            "train_std": t_std,
            "abs_delta_mean_over_train_std": delta,
            "train_null_rate": t_null,
            "val_null_rate": v_null,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Train + eval
# ============================================================================
def train_and_eval(variant, train_feat, val_feat, temporal_train_df, removed, added):
    """Train V4_ANCHOR config on temporal_train, evaluate on temporal_val. Returns metric dict."""
    feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in feature_cols if c not in removed and c in val_feat.columns]
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden cols leaked: {leaked}"
    assert list(train_feat[feature_cols].columns) == list(val_feat[feature_cols].columns), \
        "train/val column order differs"
    log(f"  feature_count after variant: {len(feature_cols)}")

    propensity = compute_position_propensity(temporal_train_df)
    weights = compute_ipw_weights(temporal_train_df, propensity, clip_hi=10.0, clip_lo=0.1)
    log(f"  IPW weights: [{weights.min():.3f}, {weights.max():.3f}] mean={weights.mean():.3f}")

    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    params = BASE_PARAMS.copy()
    params["label_gain"] = "0,1,15"   # V4_ANCHOR

    log("  training V4_ANCHOR (early_stop=80, max=2000 rounds)…")
    t0 = time.time()
    ds_tr = lgb.Dataset(train_feat[feature_cols], label=train_label,
                        group=train_groups, weight=weights)
    ds_va = lgb.Dataset(val_feat[feature_cols], label=val_label,
                        group=val_groups, reference=ds_tr)
    model = lgb.train(
        params, ds_tr, num_boost_round=2000,
        valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )
    best_iter = int(model.best_iteration)
    log(f"  trained best_iter={best_iter} in {(time.time()-t0)/60:.1f} min")

    val_pred = model.predict(val_feat[feature_cols]).astype(np.float32)
    metrics = eval_metrics(val_feat, val_pred)
    log(f"  NDCG@5={metrics['ndcg5']:.5f}  Recall@5={metrics['recall5']:.4f}  "
        f"MBR={metrics['mean_booked_rank']:.3f}")

    model_path = OUT_DIR / f"model_{variant}_temporal.txt"
    model.save_model(str(model_path))
    log(f"  model saved: {model_path}")

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp_path = OUT_DIR / f"feature_importance_{variant}_temporal.csv"
    imp_df.to_csv(imp_path, index=False)
    log(f"  importance saved: {imp_path}")

    # Side-by-side: every added feature, plus the raw TE counterpart if known.
    rank_lookup = {row["feature"]: (i + 1, row["gain"], row["split"])
                   for i, row in imp_df.iterrows()}
    for f_new in added:
        new_r = rank_lookup.get(f_new)
        if new_r:
            log(f"  rank/gain | NEW {f_new:<42s}: rank={new_r[0]:>3d} gain={new_r[1]:>12,.0f} split={new_r[2]:>5d}")
        raw = OOF_TO_RAW.get(f_new)
        if raw is not None:
            raw_r = rank_lookup.get(raw)
            if raw_r:
                log(f"  rank/gain | RAW {raw:<42s}: rank={raw_r[0]:>3d} gain={raw_r[1]:>12,.0f} split={raw_r[2]:>5d}")

    return {
        "best_iter": best_iter,
        "n_features": len(feature_cols),
        "model_path": str(model_path),
        **metrics,
    }


def classify(delta: float) -> str:
    if delta < 0:
        return "REJECT"
    if delta < 0.001:
        return "HOLD"
    if delta < 0.002:
        return "KEEP"
    return "STRONG_KEEP"


def append_results_row(row: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        new = pd.concat([old, df_row], ignore_index=True)
    else:
        new = df_row
    new.to_csv(RESULTS_CSV, index=False)
    log(f"  results appended → {RESULTS_CSV}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS,
                        help="Which variant to evaluate.")
    parser.add_argument("--plan", action="store_true",
                        help="Print plan and exit without doing anything.")
    args = parser.parse_args()

    print(PLAN, flush=True)
    if args.plan:
        return

    t0 = time.time()
    log(f"=== evaluate_variant.py — variant={args.variant} ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Compute the temporal split (same logic as STEP 1) ----
    log("Loading train + computing temporal split (same algorithm as STEP 1)…")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    log(f"  {len(train_raw):,} rows / {train_raw['srch_id'].nunique():,} searches")

    temporal_train_ids, temporal_val_ids, cutoff = temporal_split(train_raw)
    assert temporal_train_ids.isdisjoint(temporal_val_ids), "temporal_train ∩ temporal_val not empty!"
    log(f"  temporal cutoff: {cutoff}")
    log(f"  temporal_train searches: {len(temporal_train_ids):,}  "
        f"temporal_val searches: {len(temporal_val_ids):,}")

    # Parity check vs STEP 1 split_meta if present
    if REFERENCE_SPLIT_META.exists():
        ref = json.load(open(REFERENCE_SPLIT_META))
        if int(ref.get("temporal_train_searches", -1)) != len(temporal_train_ids):
            log(f"  ! parity MISMATCH with STEP 1 split_meta: "
                f"STEP1={ref['temporal_train_searches']:,}  this={len(temporal_train_ids):,}")
            sys.exit(1)
        if str(ref.get("temporal_cutoff_date", "")) != str(cutoff):
            log(f"  ! parity MISMATCH on cutoff date: STEP1={ref['temporal_cutoff_date']}  this={cutoff}")
            sys.exit(1)
        log(f"  parity OK with STEP 1 split_meta ({REFERENCE_SPLIT_META.parent.name})")

    temporal_train_df = train_raw[train_raw["srch_id"].isin(temporal_train_ids)].reset_index(drop=True)
    temporal_val_df = train_raw[train_raw["srch_id"].isin(temporal_val_ids)].reset_index(drop=True)
    log(f"  rows: train={len(temporal_train_df):,}  val={len(temporal_val_df):,}")

    expected_meta = {
        "n_train_rows": len(temporal_train_df),
        "n_val_rows": len(temporal_val_df),
        "n_train_searches": len(temporal_train_ids),
        "n_val_searches": len(temporal_val_ids),
    }

    # ---- Cache load or build ----
    if cache_is_valid(expected_meta):
        log("Cache valid — using parquet.")
        train_feat, val_feat = load_cache()
    else:
        log("Cache missing or stale — building features now.")
        train_feat, val_feat = build_and_cache(temporal_train_df, temporal_val_df)

    base_feat_count = len([c for c in train_feat.columns if c in get_feature_columns(train_feat)])
    log(f"  base feature count: {base_feat_count}")

    # ---- Apply variant ----
    log(f"\nApplying variant: {args.variant}")
    added, removed, skipped = apply_variant(args.variant, train_feat, val_feat)
    log(f"  added={len(added)} removed={len(removed)} skipped={len(skipped)}")

    # ---- Drift table on added + removed features ----
    drift_df = feature_drift_table(train_feat, val_feat, added, removed)
    drift_path = OUT_DIR / f"feature_drift_{args.variant}.csv"
    drift_df.to_csv(drift_path, index=False)
    log(f"  drift table saved: {drift_path}")
    if not drift_df.empty:
        log("  per-feature drift (action | feature | train_mean | val_mean | "
            "abs_Δμ/σ | t_null | v_null):")
        for _, r in drift_df.iterrows():
            log(f"    {r['action']:7s}  {r['feature']:32s}  "
                f"tμ={r['train_mean']:+.4f}  vμ={r['val_mean']:+.4f}  "
                f"|Δμ|/σ={r['abs_delta_mean_over_train_std']:.3f}  "
                f"tN={r['train_null_rate']:.4f}  vN={r['val_null_rate']:.4f}")

    # ---- Train + eval ----
    res = train_and_eval(args.variant, train_feat, val_feat, temporal_train_df, removed, added)

    # ---- Decision ----
    delta = res["ndcg5"] - V4_ANCHOR_TEMPORAL
    decision = classify(delta)
    log(f"\nDecision: NDCG@5={res['ndcg5']:.5f}  Δ vs V4_ANCHOR_temporal({V4_ANCHOR_TEMPORAL:.5f}) "
        f"= {delta:+.5f}  →  {decision}")

    # ---- Append results.csv ----
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "ndcg5": float(res["ndcg5"]),
        "recall1": float(res["recall1"]),
        "recall5": float(res["recall5"]),
        "mean_booked_rank": float(res["mean_booked_rank"]),
        "best_iter": int(res["best_iter"]),
        "n_features": int(res["n_features"]),
        "delta_vs_v4_anchor_temporal": float(delta),
        "decision": decision,
        "n_added": len(added),
        "n_removed": len(removed),
        "n_skipped": len(skipped),
        "added": ",".join(added),
        "removed": ",".join(removed),
        "skipped": ",".join(skipped),
        "n_val_queries": int(res["n_val_queries"]),
        "n_val_booked": int(res["n_val_booked"]),
        "model_path": res["model_path"],
    }
    append_results_row(row)

    log(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
    log(f"out_dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
