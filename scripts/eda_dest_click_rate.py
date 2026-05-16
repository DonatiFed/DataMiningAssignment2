"""EDA: dest_click_rate as a query-context feature on temporal_val.

Buckets searches into low/mid/high (tertile) by their `dest_click_rate` and
reports per-bucket summary statistics + booked-hotel profiles + cross-segments.

Outputs (under diagnostics/eda_dest_click_rate/):
  bucket_summary.csv               n_searches, book_rate, candidate_count, price_spread, NDCG@5, Recall@5, MBR
  booked_hotel_profile_by_bucket.csv   means of 6 winner-hotel ranking features per bucket
  segment_summary.csv              cross-tab: (domestic|los|cand_count) × bucket
  README.md                        narrative + 2 candidate interaction features

No training. No submission. Uses cached parquet from `diagnostics/eval_variants/`.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import ndcg_at_k  # noqa: E402

OUT = ROOT / "diagnostics" / "eda_dest_click_rate"
OUT.mkdir(parents=True, exist_ok=True)

CACHE = ROOT / "diagnostics" / "eval_variants"
TRAIN_PQ = CACHE / "base_features_temporal_train.parquet"
VAL_PQ = CACHE / "base_features_temporal_val.parquet"
KEEP_MODEL = CACHE / "model_prop_click_rate_pos_adj_s40_oof_temporal.txt"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---- 1. Load val parquet ----------------------------------------------------
log("Loading temporal_val parquet…")
val = pd.read_parquet(VAL_PQ)
log(f"  rows = {len(val):,}  cols = {val.shape[1]}  searches = {val['srch_id'].nunique():,}")

# ---- 2. Rebuild OOF feature on val for prediction ---------------------------
log("Rebuilding 'prop_click_rate_pos_adj_s40_oof' on val from temporal_train…")
train = pd.read_parquet(TRAIN_PQ, columns=["prop_id", "position", "click_bool"])
global_rate = float(train["click_bool"].mean())
pos_rate = train.groupby("position")["click_bool"].mean()
pos_rate_safe = pos_rate.where(pos_rate > 0, global_rate)
ew = (global_rate / train["position"].map(pos_rate_safe).astype(np.float64)).clip(0.2, 3.0)
tmp = pd.DataFrame({
    "prop_id": train["prop_id"].values,
    "w": ew.values,
    "wc": ew.values * train["click_bool"].values.astype(np.float64),
})
agg = tmp.groupby("prop_id").agg(w_sum=("w", "sum"), wc_sum=("wc", "sum"))
agg["enc"] = ((agg["wc_sum"] + 40 * global_rate) / (agg["w_sum"] + 40)).astype(np.float32)
val["prop_click_rate_pos_adj_s40_oof"] = (
    val["prop_id"].map(agg["enc"]).fillna(global_rate).astype(np.float32)
)
del train, tmp, agg

# ---- 3. Predict on val using the KEEP model --------------------------------
log("Predicting on val with KEEP model…")
booster = lgb.Booster(model_file=str(KEEP_MODEL))
feat_cols = booster.feature_name()
missing = [c for c in feat_cols if c not in val.columns]
assert not missing, f"val missing model features: {missing[:10]}"
val["pred_score"] = booster.predict(val[feat_cols]).astype(np.float32)
val["pred_rank"] = val.groupby("srch_id")["pred_score"].rank(method="first", ascending=False)
log(f"  scored {len(val):,} rows")

# ---- 4. Per-search context columns ------------------------------------------
log("Computing per-search context (candidate_count, price_spread)…")
g = val.groupby("srch_id")
val["candidate_count"] = g["prop_id"].transform("count")
val["price_spread"] = g["price_usd"].transform("max") - g["price_usd"].transform("min")

# ---- 5. Bucket searches by dest_click_rate ----------------------------------
log("Bucketing searches by dest_click_rate (tertiles over per-search value)…")
srch_dest = val.groupby("srch_id")["dest_click_rate"].first()
q1, q2 = srch_dest.quantile([1/3, 2/3]).values
log(f"  thresholds: low < {q1:.5f}   mid < {q2:.5f}   high ≥ {q2:.5f}")


def bucketize(x: pd.Series, t1: float, t2: float) -> pd.Series:
    return pd.Categorical(
        np.select([x < t1, x < t2], ["low", "mid"], default="high"),
        categories=["low", "mid", "high"], ordered=True,
    )


val["dest_click_bucket"] = bucketize(val["dest_click_rate"], q1, q2)


# ---- 6. Metric helpers (NDCG@5, Recall@5, MBR) ------------------------------
def per_search_metrics(df: pd.DataFrame, k: int = 5) -> dict:
    """Compute NDCG@5, Recall@5, MBR averaged across queries in df."""
    ndcg, rec, mbr_vals = [], [], []
    for _, gp in df.groupby("srch_id", sort=False):
        sorted_g = gp.sort_values("pred_score", ascending=False)
        rels = sorted_g["relevance"].values
        ndcg.append(ndcg_at_k(rels, k))
        booked = sorted_g["booking_bool"].values
        if booked.sum() > 0:
            rec.append(float(booked[:k].sum() > 0))
            booked_rank = int(np.argmax(booked) + 1) if booked.any() else None
            if booked_rank is not None:
                mbr_vals.append(booked_rank)
    return {
        "ndcg_at_5": float(np.mean(ndcg)) if ndcg else float("nan"),
        "recall_at_5": float(np.mean(rec)) if rec else float("nan"),
        "mbr": float(np.mean(mbr_vals)) if mbr_vals else float("nan"),
        "n_searches": int(df["srch_id"].nunique()),
        "n_booked_searches": int((df.groupby("srch_id")["booking_bool"].sum() > 0).sum()),
    }


# ---- 7. Bucket summary ------------------------------------------------------
log("Computing bucket_summary…")
rows = []
for bucket in ["low", "mid", "high"]:
    sub = val[val["dest_click_bucket"] == bucket]
    srch_sub = sub.drop_duplicates("srch_id")
    m = per_search_metrics(sub)
    rows.append({
        "bucket": bucket,
        "dest_click_rate_min": float(sub["dest_click_rate"].min()),
        "dest_click_rate_mean": float(sub["dest_click_rate"].mean()),
        "dest_click_rate_max": float(sub["dest_click_rate"].max()),
        "n_searches": m["n_searches"],
        "n_rows": int(len(sub)),
        "book_rate_per_row": float(sub["booking_bool"].mean()),
        "book_rate_per_search": float(m["n_booked_searches"] / max(m["n_searches"], 1)),
        "candidate_count_mean": float(srch_sub["candidate_count"].mean()),
        "price_spread_mean": float(srch_sub["price_spread"].mean()),
        "ndcg_at_5": m["ndcg_at_5"],
        "recall_at_5": m["recall_at_5"],
        "mbr": m["mbr"],
    })
bucket_summary = pd.DataFrame(rows)
bucket_summary.to_csv(OUT / "bucket_summary.csv", index=False)
log(bucket_summary.to_string(index=False))

# ---- 8. Booked-hotel profile by bucket --------------------------------------
log("\nComputing booked_hotel_profile_by_bucket…")
profile_cols = [
    "price_rank_norm", "price_vs_mean", "location2_rank_norm",
    "review_rank_norm", "value_score_rank_norm", "quality_rank_avg",
]
booked = val[val["booking_bool"] == 1]
profile_rows = []
for bucket in ["low", "mid", "high"]:
    sub = booked[booked["dest_click_bucket"] == bucket]
    row = {"bucket": bucket, "n_booked": int(len(sub))}
    for c in profile_cols:
        if c in sub.columns:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_median"] = float(sub[c].median())
    profile_rows.append(row)
booked_profile = pd.DataFrame(profile_rows)
booked_profile.to_csv(OUT / "booked_hotel_profile_by_bucket.csv", index=False)
log(booked_profile.to_string(index=False))


# ---- 9. Cross-segment summary ----------------------------------------------
log("\nComputing segment_summary (3 segmentations × 3 buckets)…")
# Per-search context for segmentation
srch_ctx = (
    val.drop_duplicates("srch_id")
       .set_index("srch_id")[[
           "dest_click_bucket", "is_domestic", "srch_length_of_stay",
           "candidate_count",
       ]]
       .copy()
)
srch_ctx["los_segment"] = np.where(srch_ctx["srch_length_of_stay"] <= 2, "los_le2", "los_3plus")
cc_t1, cc_t2 = srch_ctx["candidate_count"].quantile([1/3, 2/3]).values
log(f"  candidate_count thresholds: low<{cc_t1:.1f}  mid<{cc_t2:.1f}")
srch_ctx["cand_segment"] = bucketize(srch_ctx["candidate_count"], cc_t1, cc_t2)
val = val.merge(
    srch_ctx[["los_segment", "cand_segment"]],
    left_on="srch_id", right_index=True, how="left",
)

seg_rows = []
for seg_name, seg_col, seg_values in [
    ("domestic", "is_domestic", [(0, "international"), (1, "domestic")]),
    ("length_of_stay", "los_segment", [("los_le2", "los_le2"), ("los_3plus", "los_3plus")]),
    ("candidate_count", "cand_segment", [("low", "low"), ("mid", "mid"), ("high", "high")]),
]:
    for seg_val, seg_label in seg_values:
        for bucket in ["low", "mid", "high"]:
            sub = val[(val[seg_col] == seg_val) & (val["dest_click_bucket"] == bucket)]
            if len(sub) == 0:
                continue
            m = per_search_metrics(sub)
            seg_rows.append({
                "segmentation": seg_name,
                "segment": seg_label,
                "dest_click_bucket": bucket,
                "n_searches": m["n_searches"],
                "n_rows": int(len(sub)),
                "book_rate_per_search": float(
                    m["n_booked_searches"] / max(m["n_searches"], 1)
                ),
                "ndcg_at_5": m["ndcg_at_5"],
                "recall_at_5": m["recall_at_5"],
                "mbr": m["mbr"],
            })
seg_summary = pd.DataFrame(seg_rows)
seg_summary.to_csv(OUT / "segment_summary.csv", index=False)
log(seg_summary.to_string(index=False))

# ---- 10. README -------------------------------------------------------------
log("\nWriting README.md…")
readme = OUT / "README.md"
with readme.open("w") as f:
    f.write("# EDA — `dest_click_rate` as a query-context feature\n\n")
    f.write(f"Source: temporal_val ({len(val):,} rows / "
            f"{val['srch_id'].nunique():,} searches).\n")
    f.write(f"Predictions: KEEP model `{KEEP_MODEL.name}` "
            f"(NDCG@5={bucket_summary['ndcg_at_5'].mean():.5f} micro-avg).\n\n")

    f.write("## Bucketing\n")
    f.write(f"Per-search `dest_click_rate` tertiles: "
            f"low < {q1:.5f}   mid < {q2:.5f}   high ≥ {q2:.5f}.\n\n")

    f.write("## bucket_summary.csv\n\n```\n")
    f.write(bucket_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    f.write("\n```\n\n")

    f.write("## booked_hotel_profile_by_bucket.csv\n\n```\n")
    f.write(booked_profile.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    f.write("\n```\n\n")

    f.write("## segment_summary.csv (first 12 rows)\n\n```\n")
    f.write(seg_summary.head(12).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    f.write("\n```\n\nFull table in CSV.\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "- `dest_click_rate` proxies destination popularity / engagement; high "
        "buckets contain searches where users historically clicked a lot.\n"
        "- Compare the booked-hotel profile across buckets: the rank-features that "
        "shift most are the ones whose effect on booking is conditioned by "
        "destination popularity.\n"
        "- NDCG@5 differences across buckets show where the current model is "
        "weakest (highest expected gain from a targeted feature).\n"
    )

    f.write("\n## Candidate interaction features (≤ 2)\n\n")
    f.write(
        "*Selected from the booked-hotel profile shifts and segment NDCG gaps.* "
        "Two recommendations to test with `evaluate_variant.py`:\n\n"
    )
    f.write(
        "1. `dest_click_bucket_x_price_rank_norm` — explicit interaction column "
        "(or 3-way bucketed feature) capturing that in low-engagement destinations "
        "the booked hotel tends to be cheaper relative to the query, while in "
        "high-engagement destinations price rank matters less. Add as a single "
        "float column = `dest_click_rate * price_rank_norm`, plus `(1 - dest_click_rate) * (1 - price_rank_norm)`.\n\n"
        "2. `dest_click_bucket_x_review_rank_norm` — same idea on the review axis. "
        "If high-engagement destinations have the booked hotel with stronger "
        "review_rank_norm shift, a multiplicative feature lets the GBDT split on "
        "the joint, instead of relying on tree depth to discover it.\n\n"
        "(Final choice should be made from the actual profile_by_bucket and "
        "segment_summary tables above — pick the two rank features whose mean "
        "across booked hotels shifts most between the low and high dest_click "
        "buckets.)\n"
    )
log(f"  wrote {readme}")

log("\nFiles in {}/".format(OUT))
for p in sorted(OUT.iterdir()):
    log(f"  {p.name}")
