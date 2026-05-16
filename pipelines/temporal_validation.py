"""
STEP 1 — Leakage-safe temporal validation.

Purpose:
  Compare V4 anchor vs B3 under (a) a random-control split inside the temporal-train
  window and (b) the temporal split itself. Distinguish "B3 is bad on future data"
  from "the temporal subset itself changes training dynamics".

Leakage rules (CRITICAL):
  All target-encoded, count, and aggregate features must be computed using ONLY
  the temporal-train subset (or the inner random-train subset). Validation rows
  are treated like test: features built with `agg_source = <train subset>` and
  `is_train=False`, which routes through `target_encode_from_source` / non-kfold
  paths — no validation rows feed any aggregate.

Splits:
  - Sort unique srch_id by min(date_time).
  - temporal_train = earliest 80% of srch_id groups.
  - temporal_val   = latest   20% of srch_id groups.
  - Inside temporal_train, draw 90/10 random search-level split:
      temporal_train_inner_train = 90% of temporal_train srch_id (random)
      temporal_train_inner_random_val = 10% of temporal_train srch_id (random)
  - Search groups never split across sides.

Models (V4-style fresh lgb.Dataset, no .construct()):
  A. V4_ANCHOR: label_gain="0,1,15", IPW default (clip [0.1, 10]), base params.
  B. B3:        label_gain="0,2,15", IPW clipped at 3 (clip [0.1, 3]), base params.

Each model × {random_control, temporal} = 4 training runs total.

Outputs:
  diagnostics/temporal_validation_<TS>/
    temporal_results.csv           (one row per model × setup)
    README.md                      (executive interpretation)
    model_<id>_<setup>.txt         (4 boosters)
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
)

# ============================================================================
# Constants
# ============================================================================
V4_ENSEMBLE_KAGGLE = 0.42021
V4_ENSEMBLE_LOCAL = 0.42512
V4_ANCHOR_RANDOM_VAL = 0.42191       # V4 bal15 on the random 90/10 val
B3_RANDOM_VAL = 0.42396              # B3 overnight on the same random val
ANCHOR_SEED = 456
RANDOM_CONTROL_SEED = 42
RANDOM_VAL_FRAC = 0.10

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


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Plan header
# ============================================================================
PLAN = """\
pipelines/temporal_validation.py — execution plan
============================================

Reads:
  data/training_set_VU_DM.csv                  (4.96M rows; 'date_time' required)

Writes (under diagnostics/temporal_validation_<TS>/):
  temporal_results.csv     (4 rows: 2 models × 2 setups)
  README.md                (interpretation + decision rules)
  model_V4_ANCHOR_random_control.txt
  model_V4_ANCHOR_temporal.txt
  model_B3_random_control.txt
  model_B3_temporal.txt
  split_meta.json          (cutoff date, search/row counts, leakage assertions)

Estimated runtime:
  Feature builds: 4 × ~3–4 min = ~14 min
  LGBM trains   : 4 × ~3–5 min = ~16 min
  Total         : ~30–35 min

Command:
  uv run python -m py_compile pipelines/temporal_validation.py
  uv run python pipelines/temporal_validation.py

Hard constraints honored:
  - No submission generated.
  - No ensemble work.
  - Only the 2 specified models × 2 splits trained.
  - V4-style fresh lgb.Dataset per training; no .construct(); no free_raw_data.
"""


# ============================================================================
# Helpers
# ============================================================================
def make_group_counts(df):
    g = df.groupby("srch_id", sort=False).size().values
    assert g.sum() == len(df), "group counts mismatch"
    return g


def compute_ipw_weights(train_split, propensity, clip_hi=10.0, clip_lo=0.1):
    """V4-style IPW. Non-random rows get max_prop/propensity[position]; random rows = 1.0."""
    max_prop = float(propensity.max())
    pos_w = train_split["position"].map(
        lambda p: 1.0 if propensity.get(p, 0) <= 0 else max_prop / propensity[p]
    ).astype(np.float32).values
    is_nonrand = (train_split["random_bool"].values == 0)
    w = np.where(is_nonrand, pos_w, 1.0).astype(np.float32)
    return np.clip(w, clip_lo, clip_hi).astype(np.float32)


def temporal_split(train_raw: pd.DataFrame) -> tuple[set, set, pd.Timestamp]:
    """Split srch_id groups by min(date_time): earliest 80% train, latest 20% val."""
    if "date_time" not in train_raw.columns:
        raise RuntimeError("train_raw lacks 'date_time' — temporal split needs it.")
    train_raw = train_raw.copy()
    train_raw["date_time"] = pd.to_datetime(train_raw["date_time"])
    srch_dt = (
        train_raw.groupby("srch_id", sort=False)["date_time"].min()
        .reset_index().sort_values("date_time").reset_index(drop=True)
    )
    n_srch = len(srch_dt)
    n_train = int(n_srch * 0.8)
    cutoff = srch_dt.iloc[n_train]["date_time"]
    train_ids = set(srch_dt.iloc[:n_train]["srch_id"])
    val_ids = set(srch_dt.iloc[n_train:]["srch_id"])
    return train_ids, val_ids, cutoff


def random_inner_split(train_raw: pd.DataFrame, srch_ids: set, val_frac: float,
                       seed: int) -> tuple[set, set]:
    """Inside a given srch_id pool, draw search-level 90/10 random split (no row split)."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(srch_ids))  # deterministic order before sampling
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def eval_metrics(val_feat: pd.DataFrame, scores: np.ndarray, k: int = 5) -> dict:
    """Return NDCG@5, Recall@1, Recall@5, MeanBookedRank. Vectorized per-group."""
    log2_disc = 1.0 / np.log2(np.arange(2, k + 2))
    srch_id = val_feat["srch_id"].values
    rel = val_feat["relevance"].values.astype(np.int16)
    # group boundaries (val_feat sorted by srch_id)
    changes = np.concatenate([[0], np.where(np.diff(srch_id) != 0)[0] + 1, [len(srch_id)]])
    starts = changes[:-1]
    sizes = np.diff(changes)
    n_q = len(starts)
    ndcgs = np.empty(n_q, dtype=np.float64)
    booked_ranks = []
    rec1 = rec5 = 0
    for qi, (s, n) in enumerate(zip(starts, sizes)):
        sl = slice(s, s + n)
        rels_g = rel[sl]
        best_rels = np.sort(rels_g)[::-1][:k]
        best_dcg = float((best_rels * log2_disc[: len(best_rels)]).sum())
        if best_dcg == 0:
            ndcgs[qi] = 0.0
        else:
            order = np.argsort(-scores[sl], kind="stable")
            top_rels = rels_g[order][:k]
            ndcgs[qi] = float((top_rels * log2_disc[: len(top_rels)]).sum()) / best_dcg
        booked_mask = rels_g == 5
        if booked_mask.any():
            order = np.argsort(-scores[sl], kind="stable")
            r = int(np.where(booked_mask[order])[0][0]) + 1
            booked_ranks.append(r)
            if r == 1:
                rec1 += 1
            if r <= 5:
                rec5 += 1
    n_booked = len(booked_ranks)
    return {
        "ndcg5": float(ndcgs.mean()),
        "recall1": rec1 / n_booked if n_booked else 0.0,
        "recall5": rec5 / n_booked if n_booked else 0.0,
        "mean_booked_rank": float(np.mean(booked_ranks)) if booked_ranks else float("nan"),
        "n_val_queries": int(n_q),
        "n_val_booked": int(n_booked),
    }


# ============================================================================
# Training of one model on one setup
# ============================================================================
MODELS_SPEC = [
    {"id": "V4_ANCHOR", "label_gain": "0,1,15", "weighting": "ipw_default"},
    {"id": "B3",        "label_gain": "0,2,15", "weighting": "ipw_clip3"},
]


def train_one(model_spec, setup_name, train_split, val_split, propensity,
              out_dir, log_prefix=""):
    """Train one model on one setup. Returns metric dict."""
    log(f"{log_prefix}building features for setup={setup_name}, model={model_spec['id']}…")

    t0 = time.time()
    train_feat = build_features(train_split, agg_source=train_split, is_train=True)
    val_feat = build_features(val_split, agg_source=train_split, is_train=False)
    feature_cols = get_feature_columns(train_feat)
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden cols leaked: {leaked}"
    log(f"{log_prefix}  features: train={len(train_feat):,} val={len(val_feat):,} "
        f"cols={len(feature_cols)} ({(time.time()-t0)/60:.1f} min)")
    assert list(train_feat[feature_cols].columns) == list(val_feat[feature_cols].columns), \
        "train/val feature columns disagree"

    # Weights for the train side only
    if model_spec["weighting"] == "ipw_default":
        weights = compute_ipw_weights(train_split, propensity, clip_hi=10.0, clip_lo=0.1)
    elif model_spec["weighting"] == "ipw_clip3":
        weights = compute_ipw_weights(train_split, propensity, clip_hi=3.0, clip_lo=0.1)
    else:
        raise ValueError(f"unknown weighting: {model_spec['weighting']}")
    log(f"{log_prefix}  weights: [{weights.min():.3f}, {weights.max():.3f}] mean={weights.mean():.3f}")

    remap = {0: 0, 1: 1, 5: 2}
    train_label = train_feat["relevance"].map(remap).astype(np.int32)
    val_label = val_feat["relevance"].map(remap).astype(np.int32)

    train_groups = make_group_counts(train_feat)
    val_groups = make_group_counts(val_feat)

    params = BASE_PARAMS.copy()
    params["label_gain"] = model_spec["label_gain"]

    log(f"{log_prefix}  training (early_stop=80, max=2000 rounds)…")
    t1 = time.time()
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
    log(f"{log_prefix}  trained best_iter={best_iter} in {(time.time()-t1)/60:.1f} min")

    # Full eval (LGBM eval gives NDCG@5; we recompute + add Recall@1/5/MBR)
    val_pred = model.predict(val_feat[feature_cols]).astype(np.float32)
    metrics = eval_metrics(val_feat, val_pred)
    log(f"{log_prefix}  NDCG@5={metrics['ndcg5']:.5f}  Recall@1={metrics['recall1']:.4f}  "
        f"Recall@5={metrics['recall5']:.4f}  MBR={metrics['mean_booked_rank']:.3f}")

    model_path = out_dir / f"model_{model_spec['id']}_{setup_name}.txt"
    model.save_model(str(model_path))

    result = {
        "model_id": model_spec["id"],
        "train_setup": setup_name,
        "val_setup": setup_name,
        "label_gain": model_spec["label_gain"],
        "weighting": model_spec["weighting"],
        "best_iteration": best_iter,
        "n_features": len(feature_cols),
        "model_path": str(model_path),
        **metrics,
    }
    # free memory before next training
    del ds_tr, ds_va, model, val_pred, train_feat, val_feat, weights, train_label, val_label
    del train_groups, val_groups
    gc.collect()
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print(PLAN, flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "diagnostics" / f"temporal_validation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"out_dir: {out_dir}")

    log("Loading full train…")
    train_raw = load_train()
    train_raw = make_target(train_raw)
    train_raw = train_raw.sort_values("srch_id").reset_index(drop=True)
    n_total_rows = len(train_raw)
    n_total_srch = train_raw["srch_id"].nunique()
    log(f"  {n_total_rows:,} rows / {n_total_srch:,} searches")

    # ---- Splits ------------------------------------------------------------
    log("\nComputing temporal split (by min(date_time) per srch_id, 80/20)…")
    temporal_train_ids, temporal_val_ids, cutoff = temporal_split(train_raw)
    log(f"  temporal cutoff: {cutoff}")
    log(f"  temporal_train searches: {len(temporal_train_ids):,}")
    log(f"  temporal_val   searches: {len(temporal_val_ids):,}")

    log("\nComputing random-control split inside temporal_train (90/10, seed=42)…")
    inner_train_ids, inner_random_val_ids = random_inner_split(
        train_raw, temporal_train_ids, RANDOM_VAL_FRAC, RANDOM_CONTROL_SEED
    )
    log(f"  inner_train       searches: {len(inner_train_ids):,}")
    log(f"  inner_random_val  searches: {len(inner_random_val_ids):,}")

    # ---- Sanity checks -----------------------------------------------------
    assert temporal_train_ids.isdisjoint(temporal_val_ids), "temporal_train ∩ temporal_val not empty!"
    assert inner_train_ids.isdisjoint(inner_random_val_ids), "inner_train ∩ inner_random_val not empty!"
    assert inner_train_ids.union(inner_random_val_ids) == temporal_train_ids, \
        "inner_train ∪ inner_random_val != temporal_train"
    assert temporal_train_ids.union(temporal_val_ids) == set(train_raw["srch_id"].unique()), \
        "temporal_train ∪ temporal_val != all srch_ids"
    log("\n  sanity OK: train/val srch_id sets are disjoint and union to total.")

    # Build subset DataFrames
    log("\nSlicing train_raw into 4 subsets…")
    temporal_train_df = train_raw[train_raw["srch_id"].isin(temporal_train_ids)].reset_index(drop=True)
    temporal_val_df = train_raw[train_raw["srch_id"].isin(temporal_val_ids)].reset_index(drop=True)
    inner_train_df = train_raw[train_raw["srch_id"].isin(inner_train_ids)].reset_index(drop=True)
    inner_random_val_df = train_raw[train_raw["srch_id"].isin(inner_random_val_ids)].reset_index(drop=True)

    # Row counts
    log(f"  temporal_train rows: {len(temporal_train_df):,}")
    log(f"  temporal_val   rows: {len(temporal_val_df):,}")
    log(f"  inner_train    rows: {len(inner_train_df):,}")
    log(f"  inner_random_val rows: {len(inner_random_val_df):,}")
    assert len(temporal_train_df) + len(temporal_val_df) == n_total_rows, \
        f"temporal row total {len(temporal_train_df)+len(temporal_val_df)} != {n_total_rows}"
    assert len(inner_train_df) + len(inner_random_val_df) == len(temporal_train_df), \
        f"inner row total {len(inner_train_df)+len(inner_random_val_df)} != {len(temporal_train_df)}"

    # Date ranges
    def _date_range(df):
        d = pd.to_datetime(df["date_time"])
        return d.min(), d.max()
    tt_min, tt_max = _date_range(temporal_train_df)
    tv_min, tv_max = _date_range(temporal_val_df)
    it_min, it_max = _date_range(inner_train_df)
    iv_min, iv_max = _date_range(inner_random_val_df)
    log(f"\n  temporal_train date range: {tt_min}  →  {tt_max}")
    log(f"  temporal_val   date range: {tv_min}  →  {tv_max}")
    log(f"  inner_train    date range: {it_min}  →  {it_max}")
    log(f"  inner_random_val date range: {iv_min}  →  {iv_max}")

    split_meta = {
        "timestamp": ts,
        "total_rows": int(n_total_rows),
        "total_searches": int(n_total_srch),
        "temporal_cutoff_date": str(cutoff),
        "temporal_train_searches": len(temporal_train_ids),
        "temporal_val_searches": len(temporal_val_ids),
        "inner_train_searches": len(inner_train_ids),
        "inner_random_val_searches": len(inner_random_val_ids),
        "temporal_train_rows": int(len(temporal_train_df)),
        "temporal_val_rows": int(len(temporal_val_df)),
        "inner_train_rows": int(len(inner_train_df)),
        "inner_random_val_rows": int(len(inner_random_val_df)),
        "date_ranges": {
            "temporal_train": [str(tt_min), str(tt_max)],
            "temporal_val": [str(tv_min), str(tv_max)],
            "inner_train": [str(it_min), str(it_max)],
            "inner_random_val": [str(iv_min), str(iv_max)],
        },
        "leakage_assertions_passed": True,
    }
    json.dump(split_meta, open(out_dir / "split_meta.json", "w"), indent=2, default=str)
    log(f"  split_meta saved: {out_dir / 'split_meta.json'}")

    # Free train_raw before per-setup feature builds
    del train_raw
    gc.collect()

    # ---- 4 trainings ------------------------------------------------------
    results = []

    # ----- Setup 1: random-control -----
    log("\n" + "=" * 70)
    log("SETUP 1: random-control (train=inner_train, val=inner_random_val)")
    log(f"  agg_source = inner_train ({len(inner_train_df):,} rows, "
        f"{inner_train_df['srch_id'].nunique():,} searches)")
    log("=" * 70)

    propensity_inner = compute_position_propensity(inner_train_df)
    log(f"  propensity max={propensity_inner.max():.4f}")

    for spec in MODELS_SPEC:
        r = train_one(
            spec, setup_name="random_control",
            train_split=inner_train_df, val_split=inner_random_val_df,
            propensity=propensity_inner,
            out_dir=out_dir,
            log_prefix=f"  [{spec['id']} | random_control] ",
        )
        results.append(r)

    # Free random-control subsets before temporal setup
    del inner_train_df, inner_random_val_df, propensity_inner
    gc.collect()

    # ----- Setup 2: temporal -----
    log("\n" + "=" * 70)
    log("SETUP 2: temporal (train=temporal_train, val=temporal_val)")
    log(f"  agg_source = temporal_train ({len(temporal_train_df):,} rows, "
        f"{temporal_train_df['srch_id'].nunique():,} searches)")
    log("=" * 70)

    propensity_temporal = compute_position_propensity(temporal_train_df)
    log(f"  propensity max={propensity_temporal.max():.4f}")

    for spec in MODELS_SPEC:
        r = train_one(
            spec, setup_name="temporal",
            train_split=temporal_train_df, val_split=temporal_val_df,
            propensity=propensity_temporal,
            out_dir=out_dir,
            log_prefix=f"  [{spec['id']} | temporal] ",
        )
        results.append(r)

    del temporal_train_df, temporal_val_df, propensity_temporal
    gc.collect()

    # ---- Save ------------------------------------------------------------
    df = pd.DataFrame(results, columns=[
        "model_id", "train_setup", "val_setup", "label_gain", "weighting",
        "ndcg5", "recall1", "recall5", "mean_booked_rank",
        "best_iteration", "n_features", "n_val_queries", "n_val_booked",
        "model_path",
    ])
    df.to_csv(out_dir / "temporal_results.csv", index=False)
    log(f"\nresults CSV: {out_dir / 'temporal_results.csv'}")

    # ---- README ----------------------------------------------------------
    pivot = df.pivot_table(
        index="model_id", columns="val_setup",
        values=["ndcg5", "recall5", "mean_booked_rank", "best_iteration"],
    )
    md = [
        f"# Temporal validation — {ts}",
        "",
        f"V4 ensemble Kaggle = {V4_ENSEMBLE_KAGGLE:.5f} · V4 ensemble local = {V4_ENSEMBLE_LOCAL:.5f}",
        f"Reference random-val NDCG@5: V4 anchor (bal15) = {V4_ANCHOR_RANDOM_VAL:.5f} · B3 = {B3_RANDOM_VAL:.5f}",
        "",
        "## Splits",
        "",
        f"- Temporal cutoff (min(date_time) per srch_id): **{cutoff}**",
        f"- Total: {n_total_rows:,} rows / {n_total_srch:,} searches",
        f"- temporal_train: {len(temporal_train_ids):,} searches, {split_meta['temporal_train_rows']:,} rows  ({tt_min} → {tt_max})",
        f"- temporal_val:   {len(temporal_val_ids):,} searches, {split_meta['temporal_val_rows']:,} rows  ({tv_min} → {tv_max})",
        f"- inner_train (90% of temporal_train, seed=42): {len(inner_train_ids):,} searches  ({it_min} → {it_max})",
        f"- inner_random_val (10%): {len(inner_random_val_ids):,} searches  ({iv_min} → {iv_max})",
        "",
        "Leakage assertions: srch_id sets disjoint; row counts sum to total. "
        "All validation features built with `agg_source = <train subset>, is_train=False` — "
        "validation rows never feed any TE / count / aggregate.",
        "",
        "## Results",
        "",
        "| model | val_setup | NDCG@5 | Recall@5 | MBR | best_iter |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md.append(f"| {r['model_id']} | {r['val_setup']} | {r['ndcg5']:.5f} | "
                  f"{r['recall5']:.4f} | {r['mean_booked_rank']:.3f} | {int(r['best_iteration'])} |")
    md += [
        "",
        "## Random-val anchor references",
        "",
        f"- V4 anchor on random val (V4 stage3): **{V4_ANCHOR_RANDOM_VAL:.5f}**",
        f"- B3 on random val (overnight): **{B3_RANDOM_VAL:.5f}**",
        "",
        "## Interpretation (fill below from the results above)",
        "",
        "**Decision rules:**",
        "- If B3 wins random-control but loses temporal → **B3 is random-val overfit**.",
        "- If B3 wins both → B3/ipw_clip3 is genuinely better.",
        "- If both models drop similarly on temporal → temporal shift is general, not model-specific.",
        "- If temporal ranking differs from random ranking → **future selection must include temporal val**.",
        "- If random-control ≈ temporal → temporal split unlikely to be the main culprit.",
        "",
        "_Analyst: fill in observations + recommendation here._",
        "",
        "## Files",
        "",
        f"- `temporal_results.csv` — 4 rows (V4_ANCHOR/B3 × random_control/temporal).",
        f"- `split_meta.json` — split sizes, dates, assertions.",
        f"- `model_<id>_<setup>.txt` — 4 LightGBM boosters.",
    ]
    (out_dir / "README.md").write_text("\n".join(md))
    log(f"README:    {out_dir / 'README.md'}")

    # ---- Final ------------------------------------------------------------
    log("\n" + "=" * 70)
    log("FINAL — temporal validation table")
    log("=" * 70)
    for _, r in df.iterrows():
        log(f"  {r['model_id']:9s} | {r['val_setup']:14s} | NDCG@5={r['ndcg5']:.5f}  "
            f"Recall@5={r['recall5']:.4f}  MBR={r['mean_booked_rank']:.3f}  best_iter={int(r['best_iteration'])}")
    log(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
    log(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
