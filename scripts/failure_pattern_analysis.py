"""Failure-pattern analysis on temporal_val.

For each "difficult" query (where the V6 ensemble's predicted top-1 hotel is
NOT the booked one), compare the booked hotel vs the model's top-wrong hotel
across price, location, quality, TE, cold-start, promotion, competitor, party,
and query-structure dimensions. Segment by dest_click_rate buckets,
candidate_count, length-of-stay, domestic flag.

Also compares V6 vs V4_ANCHOR predictions to identify where V6 helps or hurts.

Outputs (in diagnostics/failure_patterns/):
  - pair_stats.csv             per-failure booked vs top-wrong row counts
  - segment_<dim>.csv          one per segment dimension
  - v4_vs_v6_disagreement.csv  where the two models rank the booked hotel differently
  - patterns.md                synthesized structured findings
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

VAL_PQ = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"
V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V4_MODEL = ROOT / "diagnostics" / "temporal_validation_20260516_113831" / "model_V4_ANCHOR_temporal.txt"
OUT = ROOT / "diagnostics" / "failure_patterns"
OUT.mkdir(parents=True, exist_ok=True)

# 9-member LOO-best V6 ensemble (drops booking_clf)
V6_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grouped_rank(srch_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Within-srch_id rank, ascending=False (1 = best, fractional ties)."""
    return pd.Series(scores).groupby(pd.Series(srch_id), sort=False).rank(
        method="average", ascending=False
    ).values.astype(np.float32)


# ============================================================================
# 1. Load features + predictions
# ============================================================================
log("Loading temporal_val parquet…")
val = pd.read_parquet(VAL_PQ)
val = val.sort_values("srch_id").reset_index(drop=True)
log(f"  rows={len(val):,}  searches={val['srch_id'].nunique():,}")

log(f"Loading V6 ensemble val predictions ({len(V6_MEMBERS)} members)…")
v6_ranks = []
for m in V6_MEMBERS:
    p = V6_DIR / f"val_pred_{m}.npy"
    s = np.load(p).astype(np.float32)
    r = grouped_rank(val["srch_id"].values, s)
    v6_ranks.append(r)
val["v6_rank"] = np.mean(v6_ranks, axis=0)
val["v6_pred_pos"] = val.groupby("srch_id", sort=False)["v6_rank"].rank(
    method="first", ascending=True
).astype(np.int32)  # 1 = top, 2 = second, etc.

log("Loading V4_ANCHOR + predicting on val…")
v4 = lgb.Booster(model_file=str(V4_MODEL))
v4_feats = v4.feature_name()
val["v4_score"] = v4.predict(val[v4_feats]).astype(np.float32)
val["v4_pred_pos"] = val.groupby("srch_id", sort=False)["v4_score"].rank(
    method="first", ascending=False
).astype(np.int32)

# ============================================================================
# 2. Identify failures: queries with a booking where V6 top-1 ≠ booked
# ============================================================================
log("Building booked vs top-wrong frame…")
booked_mask = val["booking_bool"] == 1
# Booked rows: one per booked query
booked = val[booked_mask].copy()
booked["booked_v6_pos"] = booked["v6_pred_pos"]
booked["booked_v4_pos"] = booked["v4_pred_pos"]

# Top-wrong: per srch_id, the row with v6_pred_pos=1 IF it's not the booked one
top1 = val[val["v6_pred_pos"] == 1].copy()
top1 = top1.set_index("srch_id")

# Join
booked = booked.set_index("srch_id")
booked["top1_prop_id"] = top1["prop_id"]
booked["top1_is_booked"] = (booked["top1_prop_id"] == booked["prop_id"])

# Failures: booked_v6_pos != 1
failures = booked[~booked["top1_is_booked"]].copy()
successes = booked[booked["top1_is_booked"]].copy()
log(f"  booked queries: {len(booked):,}  successes (V6 top-1 = booked): {len(successes):,}  "
    f"failures: {len(failures):,}  fail rate: {len(failures)/len(booked):.2%}")

# For each failure, also pull the top-wrong row from val (it's the top-1 ranked row of that srch_id)
top_wrong = val[val["v6_pred_pos"] == 1].merge(
    failures[["top1_prop_id"]].reset_index(),
    left_on=["srch_id", "prop_id"], right_on=["srch_id", "top1_prop_id"], how="inner"
).drop(columns=["top1_prop_id"])
log(f"  top_wrong rows: {len(top_wrong):,}")
assert len(top_wrong) == len(failures), f"mismatch: failures={len(failures)} top_wrong={len(top_wrong)}"

# ============================================================================
# 3. Build pair frame (one row per failure: booked side + top_wrong side)
# ============================================================================
failures_r = failures.reset_index()
top_wrong = top_wrong.set_index("srch_id").reindex(failures_r["srch_id"].values).reset_index()

COMPARE_COLS = [
    "price_usd", "price_rank_norm", "price_vs_mean", "price_vs_median", "price_vs_prop_mean",
    "price_ratio_to_hist", "value_score", "value_score_rank_norm", "quality_rank_avg",
    "prop_starrating", "prop_review_score", "prop_brand_bool", "promotion_flag",
    "prop_location_score1", "prop_location_score2", "location1_rank_norm", "location2_rank_norm",
    "review_rank_norm", "star_percentile",
    "prop_count", "prop_click_rate", "prop_book_rate", "prop_dest_book_rate", "prop_avg_position",
    "orig_destination_distance", "comp_rate_advantage", "comp_no_inv_count",
]
COMPARE_COLS = [c for c in COMPARE_COLS if c in val.columns]
log(f"  comparing {len(COMPARE_COLS)} features per booked vs top_wrong row")

# ============================================================================
# 4. Segment definitions
# ============================================================================
log("Building segment columns…")
# Per-search dest_click_rate (constant within srch_id)
srch_dest = val.groupby("srch_id")["dest_click_rate"].first()
q1_d, q2_d = srch_dest.quantile([1/3, 2/3]).values
def dc_bucket(x):
    return np.where(x < q1_d, "low_dest", np.where(x < q2_d, "mid_dest", "high_dest"))

cand_count = val.groupby("srch_id")["query_hotel_count"].first()
q1_c, q2_c = cand_count.quantile([1/3, 2/3]).values
def cc_bucket(x):
    return np.where(x < q1_c, "low_cand", np.where(x < q2_c, "mid_cand", "high_cand"))

failures_r["seg_dest"] = dc_bucket(failures_r["dest_click_rate"].values)
failures_r["seg_cand"] = cc_bucket(failures_r["query_hotel_count"].values)
failures_r["seg_los"] = np.where(failures_r["srch_length_of_stay"] <= 2, "los_le2", "los_3plus")
failures_r["seg_dom"] = np.where(failures_r["is_domestic"] == 1, "domestic", "international")
failures_r["seg_family"] = np.where(failures_r["srch_children_count"] > 0, "family", "no_family")
failures_r["seg_bookwin"] = np.where(
    failures_r["srch_booking_window"] <= 7, "short_window",
    np.where(failures_r["srch_booking_window"] <= 30, "mid_window", "long_window"),
)

# Map segments to top_wrong (same per srch_id)
top_wrong = top_wrong.merge(
    failures_r[["srch_id", "seg_dest", "seg_cand", "seg_los", "seg_dom", "seg_family", "seg_bookwin"]],
    on="srch_id", how="left",
)

# ============================================================================
# 5. Per-segment booked vs top_wrong directional stats
# ============================================================================
def directional_stats(seg_name: str, seg_values: list[str]) -> pd.DataFrame:
    rows = []
    for seg_val in seg_values:
        b = failures_r[failures_r[seg_name] == seg_val]
        w = top_wrong[top_wrong[seg_name] == seg_val]
        if len(b) == 0:
            continue
        n = len(b)
        row = {"segment_dim": seg_name, "segment_value": seg_val, "n_failures": int(n)}
        for c in COMPARE_COLS:
            bm = b[c].astype(float).mean()
            wm = w[c].astype(float).mean()
            row[f"{c}_booked"] = float(bm)
            row[f"{c}_topwrong"] = float(wm)
            row[f"{c}_delta_book_minus_wrong"] = float(bm - wm)
        rows.append(row)
    return pd.DataFrame(rows)


log("Computing per-segment directional stats…")
seg_dfs = {}
for seg_name, seg_values in [
    ("seg_dest", ["low_dest", "mid_dest", "high_dest"]),
    ("seg_cand", ["low_cand", "mid_cand", "high_cand"]),
    ("seg_los", ["los_le2", "los_3plus"]),
    ("seg_dom", ["domestic", "international"]),
    ("seg_family", ["no_family", "family"]),
    ("seg_bookwin", ["short_window", "mid_window", "long_window"]),
]:
    df = directional_stats(seg_name, seg_values)
    df.to_csv(OUT / f"segment_{seg_name}.csv", index=False)
    seg_dfs[seg_name] = df
    log(f"  saved segment_{seg_name}.csv ({len(df)} rows)")

# ============================================================================
# 6. Overall (across all failures) booked vs top_wrong
# ============================================================================
overall_rows = []
n = len(failures_r)
for c in COMPARE_COLS:
    bm = failures_r[c].astype(float).mean()
    wm = top_wrong[c].astype(float).mean()
    overall_rows.append({
        "feature": c,
        "booked_mean": bm,
        "topwrong_mean": wm,
        "delta_book_minus_wrong": bm - wm,
        "abs_delta_pct_of_booked": abs(bm - wm) / (abs(bm) + 1e-9),
    })
overall_df = pd.DataFrame(overall_rows).sort_values("abs_delta_pct_of_booked", ascending=False)
overall_df.to_csv(OUT / "overall_directional.csv", index=False)
log(f"  saved overall_directional.csv ({len(overall_df)} rows)")

# ============================================================================
# 7. V4 vs V6 disagreement
# ============================================================================
log("Computing V4 vs V6 disagreement on booked queries…")
disagree = booked.reset_index()[["srch_id", "prop_id", "booked_v4_pos", "booked_v6_pos"]].copy()
disagree["v6_helps"] = (disagree["booked_v6_pos"] < disagree["booked_v4_pos"]).astype(int)
disagree["v6_hurts"] = (disagree["booked_v6_pos"] > disagree["booked_v4_pos"]).astype(int)
disagree["v6_top5_only"] = ((disagree["booked_v6_pos"] <= 5) & (disagree["booked_v4_pos"] > 5)).astype(int)
disagree["v4_top5_only"] = ((disagree["booked_v4_pos"] <= 5) & (disagree["booked_v6_pos"] > 5)).astype(int)

# Add segments
disagree = disagree.merge(
    failures_r[["srch_id", "seg_dest", "seg_cand", "seg_los", "seg_dom"]],
    on="srch_id", how="left",
)
disagree = disagree.merge(
    booked.reset_index()[["srch_id", "prop_click_rate", "prop_dest_book_rate",
                           "prop_count", "price_vs_prop_mean", "value_score_rank_norm",
                           "price_rank_norm", "review_rank_norm", "is_domestic",
                           "srch_length_of_stay"]],
    on="srch_id", how="left", suffixes=("", "_booked"),
)
disagree.to_csv(OUT / "v4_vs_v6_disagreement.csv", index=False)
log(f"  saved v4_vs_v6_disagreement.csv ({len(disagree):,} rows)")

# Per-segment v4↔v6 movement
def v4v6_segment_summary(col):
    rows = []
    for s in disagree[col].dropna().unique():
        sub = disagree[disagree[col] == s]
        rows.append({
            "segment_dim": col,
            "segment_value": s,
            "n_booked": int(len(sub)),
            "v6_top5_only_pct": float(sub["v6_top5_only"].mean()),
            "v4_top5_only_pct": float(sub["v4_top5_only"].mean()),
            "v6_helps_pct": float(sub["v6_helps"].mean()),
            "v6_hurts_pct": float(sub["v6_hurts"].mean()),
            "net_v6_advantage_pct": float(sub["v6_helps"].mean() - sub["v6_hurts"].mean()),
            "v6_mean_pos": float(sub["booked_v6_pos"].mean()),
            "v4_mean_pos": float(sub["booked_v4_pos"].mean()),
        })
    return pd.DataFrame(rows)


v4v6_segs = pd.concat([
    v4v6_segment_summary("seg_dest"),
    v4v6_segment_summary("seg_cand"),
    v4v6_segment_summary("seg_los"),
    v4v6_segment_summary("seg_dom"),
], ignore_index=True)
v4v6_segs.to_csv(OUT / "v4_vs_v6_by_segment.csv", index=False)

# ============================================================================
# 8. Rank profile of booked hotel in failures (rank 6-10? deep?)
# ============================================================================
log("Booked-rank profile in failures…")
rank_profile = failures_r.copy()
rank_profile["v6_pos_bin"] = pd.cut(
    rank_profile["booked_v6_pos"],
    bins=[0, 1, 5, 10, 20, 999],
    labels=["1", "2-5", "6-10", "11-20", "20+"],
)
profile_summary = rank_profile.groupby("v6_pos_bin", observed=True).size().reset_index(name="n_failures")
profile_summary["pct"] = profile_summary["n_failures"] / profile_summary["n_failures"].sum()
profile_summary.to_csv(OUT / "booked_rank_profile.csv", index=False)
log(profile_summary.to_string(index=False))

# ============================================================================
# 9. Pair stats (per failure, with booked + top_wrong rows side-by-side)
# ============================================================================
keep_cols = COMPARE_COLS + ["seg_dest", "seg_cand", "seg_los", "seg_dom",
                             "booked_v6_pos", "booked_v4_pos"]
pair_stats = failures_r[["srch_id", "prop_id"] + keep_cols].copy()
pair_stats = pair_stats.rename(columns={c: f"{c}_booked" for c in COMPARE_COLS})
tw = top_wrong[["srch_id", "prop_id"] + COMPARE_COLS].copy()
tw = tw.rename(columns={"prop_id": "prop_id_wrong", **{c: f"{c}_wrong" for c in COMPARE_COLS}})
pair_stats = pair_stats.merge(tw, on="srch_id", how="left")
pair_stats.to_csv(OUT / "pair_stats.csv", index=False)
log(f"  saved pair_stats.csv ({len(pair_stats):,} rows)")

log(f"\n=== outputs in {OUT} ===")
for p in sorted(OUT.iterdir()):
    log(f"  {p.name}  ({p.stat().st_size//1024} KB)")
