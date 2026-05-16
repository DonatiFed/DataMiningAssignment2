"""Rank-average ensemble over saved single-model boosters.

No training. Predicts each member on temporal_val, converts each prediction
to within-srch_id rank, averages the ranks, and reports NDCG@5/Recall@5/MBR
for each ensemble configuration.

Members:
  V4 = V4_ANCHOR        (143 features) — diagnostics/temporal_validation_20260516_113831/model_V4_ANCHOR_temporal.txt
  CP = click_posadj     (144 features) — diagnostics/eval_variants/model_prop_click_rate_pos_adj_s40_oof_temporal.txt
  DS = prop_dest_safe   (144 features) — diagnostics/eval_variants/model_prop_dest_book_rate_safe_temporal.txt

Ensembles (rank-average, equal weights):
  V4              (baseline, single model)
  CP              (best single model from feature session)
  DS              (cleanest-drift HOLD)
  V4+CP, V4+DS, CP+DS, V4+CP+DS

Outputs:
  diagnostics/eval_variants/ensemble_rank_avg_results.csv
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import ndcg_at_k  # noqa: E402

OUT = ROOT / "diagnostics" / "eval_variants"
CACHE_TRAIN = OUT / "base_features_temporal_train.parquet"
CACHE_VAL = OUT / "base_features_temporal_val.parquet"
MODELS = {
    "V4": ROOT / "diagnostics" / "temporal_validation_20260516_113831" / "model_V4_ANCHOR_temporal.txt",
    "CP": OUT / "model_prop_click_rate_pos_adj_s40_oof_temporal.txt",
    "DS": OUT / "model_prop_dest_book_rate_safe_temporal.txt",
}
RESULTS_CSV = OUT / "ensemble_rank_avg_results.csv"
V4_ANCHOR = 0.40401


def log(msg: str) -> None:
    print(msg, flush=True)


# ---- Load val (has the 143 base features) -----------------------------------
log("Loading temporal_val parquet…")
val = pd.read_parquet(CACHE_VAL)
log(f"  rows={len(val):,}  cols={val.shape[1]}  searches={val['srch_id'].nunique():,}")

# ---- Rebuild prop_click_rate_pos_adj_s40_oof on val (full-train aggregate) --
log("Rebuilding 'prop_click_rate_pos_adj_s40_oof' on val from temporal_train…")
train = pd.read_parquet(
    CACHE_TRAIN, columns=["prop_id", "position", "click_bool"]
)
global_click = float(train["click_bool"].mean())
pos_click = train.groupby("position")["click_bool"].mean()
pos_click_safe = pos_click.where(pos_click > 0, global_click)
ew = (global_click / train["position"].map(pos_click_safe).astype(np.float64)).clip(0.2, 3.0)
tmp = pd.DataFrame({
    "prop_id": train["prop_id"].values,
    "w": ew.values,
    "wc": ew.values * train["click_bool"].values.astype(np.float64),
})
agg = tmp.groupby("prop_id").agg(w_sum=("w", "sum"), wc_sum=("wc", "sum"))
agg["enc"] = ((agg["wc_sum"] + 40.0 * global_click) / (agg["w_sum"] + 40.0)).astype(np.float32)
val["prop_click_rate_pos_adj_s40_oof"] = (
    val["prop_id"].map(agg["enc"]).fillna(global_click).astype(np.float32)
)
del train, tmp, agg

# ---- Rebuild prop_dest_book_rate_safe on val (full-train aggregate) ---------
log("Rebuilding 'prop_dest_book_rate_safe' on val from temporal_train…")
train = pd.read_parquet(
    CACHE_TRAIN,
    columns=["prop_id", "srch_destination_id", "booking_bool"],
)
alpha = 40.0
global_book = float(train["booking_bool"].mean())
prop_g = train.groupby("prop_id")["booking_bool"].agg(["sum", "count"])
dest_g = train.groupby("srch_destination_id")["booking_bool"].agg(["sum", "count"])
pair_g = train.groupby(
    ["prop_id", "srch_destination_id"]
)["booking_bool"].agg(["sum", "count"])
prop_rate = (prop_g["sum"] / prop_g["count"]).astype(np.float64)
dest_rate = (dest_g["sum"] / dest_g["count"]).astype(np.float64)

pr_fb = val["prop_id"].map(prop_rate).fillna(global_book).astype(np.float64).values
dr_fb = val["srch_destination_id"].map(dest_rate).fillna(global_book).astype(np.float64).values
fallback = 0.5 * pr_fb + 0.3 * dr_fb + 0.2 * global_book
keys = pd.MultiIndex.from_arrays(
    [val["prop_id"].values, val["srch_destination_id"].values]
)
pair_book = pair_g["sum"].reindex(keys).fillna(0).values.astype(np.float64)
pair_count = pair_g["count"].reindex(keys).fillna(0).values.astype(np.float64)
val["prop_dest_book_rate_safe"] = (
    (pair_book + alpha * fallback) / (pair_count + alpha)
).astype(np.float32)
del train, prop_g, dest_g, pair_g, prop_rate, dest_rate, fallback, keys, pair_book, pair_count
log(f"  global_click={global_click:.5f}  global_book={global_book:.5f}")

# ---- Predict each model on val ----------------------------------------------
preds: dict[str, np.ndarray] = {}
for label, path in MODELS.items():
    log(f"Predicting {label} ({path.name})…")
    t0 = time.time()
    booster = lgb.Booster(model_file=str(path))
    feat_cols = booster.feature_name()
    missing = [c for c in feat_cols if c not in val.columns]
    assert not missing, f"{label} missing features: {missing[:5]}"
    preds[label] = booster.predict(val[feat_cols]).astype(np.float32)
    log(f"  done in {time.time()-t0:.1f}s  ({len(feat_cols)} features)")

# ---- Convert each model's pred to within-srch_id rank -----------------------
# Higher score = better, so use ascending=False with rank() so rank 1 = top.
log("Computing per-srch_id ranks for each model…")
ranks: dict[str, np.ndarray] = {}
for label, p in preds.items():
    val[f"_score_{label}"] = p
    ranks[label] = (
        val.groupby("srch_id")[f"_score_{label}"]
           .rank(method="average", ascending=False)
           .astype(np.float32)
           .values
    )
    val.drop(columns=[f"_score_{label}"], inplace=True)


# ---- NDCG/Recall/MBR helper (uses the AVG-RANK score, smaller = better) -----
def per_search_metrics(rank_score: np.ndarray, k: int = 5) -> dict:
    """rank_score is the averaged within-srch_id rank — sort ASCENDING (lower=better)."""
    df = val[["srch_id", "relevance", "booking_bool"]].copy()
    # Sort ascending by rank → first 5 rows per srch = top-5.
    df["_rk"] = rank_score
    ndcg, recall, mbr = [], [], []
    for _, gp in df.sort_values("_rk").groupby("srch_id", sort=False):
        rels = gp["relevance"].values
        ndcg.append(ndcg_at_k(rels, k))
        booked = gp["booking_bool"].values
        if booked.sum() > 0:
            recall.append(float(booked[:k].sum() > 0))
            mbr.append(int(np.argmax(booked) + 1))
    return {
        "ndcg_at_5": float(np.mean(ndcg)),
        "recall_at_5": float(np.mean(recall)),
        "mbr": float(np.mean(mbr)),
        "n_searches": int(df["srch_id"].nunique()),
        "n_booked_searches": int(len(mbr)),
    }


# ---- Score each ensemble configuration --------------------------------------
configs = [
    ("V4",       ["V4"]),
    ("CP",       ["CP"]),
    ("DS",       ["DS"]),
    ("V4+CP",    ["V4", "CP"]),
    ("V4+DS",    ["V4", "DS"]),
    ("CP+DS",    ["CP", "DS"]),
    ("V4+CP+DS", ["V4", "CP", "DS"]),
]

rows = []
log("\nScoring ensembles…")
for name, members in configs:
    avg_rank = np.mean([ranks[m] for m in members], axis=0)
    m = per_search_metrics(avg_rank)
    delta = m["ndcg_at_5"] - V4_ANCHOR
    rows.append({
        "ensemble": name,
        "members": "+".join(members),
        "n_members": len(members),
        "ndcg_at_5": m["ndcg_at_5"],
        "recall_at_5": m["recall_at_5"],
        "mbr": m["mbr"],
        "delta_vs_v4_anchor": delta,
    })
    log(f"  {name:10s}  NDCG@5={m['ndcg_at_5']:.5f}  "
        f"Δ={delta:+.5f}  Recall@5={m['recall_at_5']:.4f}  MBR={m['mbr']:.3f}")

results = pd.DataFrame(rows).sort_values("ndcg_at_5", ascending=False)
results.to_csv(RESULTS_CSV, index=False)
log(f"\nResults saved to {RESULTS_CSV}")
log("\nFinal ranked table:")
log(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
