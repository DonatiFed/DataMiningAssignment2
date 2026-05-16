"""Produce 5 EDA outputs requested for the report.

Outputs in diagnostics/eda_summary/:
  1. position_rates.csv      click_rate & book_rate per position
  2. rates_by_month.csv      click/book rate per month
  3. rates_by_week.csv       click/book rate per ISO week
  4. te_feature_importance.csv  median FI (gain & split) for prop_click_rate,
                                prop_book_rate, prop_rel_rate across ensemble
  5. prop_id_count_dist.csv  per-prop_id row-count distribution (p25/p50/p75 + extras)
"""
import sys, glob, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TRAIN_FILE

OUT = ROOT / "diagnostics" / "eda_summary"
OUT.mkdir(parents=True, exist_ok=True)

USE_COLS = ["srch_id", "prop_id", "date_time", "position",
            "click_bool", "booking_bool"]

print("Loading training data (subset of columns)...")
df = pd.read_csv(TRAIN_FILE, usecols=USE_COLS, na_values="NULL")
df["date_time"] = pd.to_datetime(df["date_time"])
print(f"  rows = {len(df):,}")

# ---- 1 & 2: rates by position ----
print("\n[1+2] Click & book rate per position")
pos = (df.groupby("position")
         .agg(count=("click_bool", "size"),
              click_rate=("click_bool", "mean"),
              book_rate=("booking_bool", "mean"))
         .reset_index())
pos.to_csv(OUT / "position_rates.csv", index=False)
print(pos.head(10).to_string(index=False))
print("...")
print(pos.tail(5).to_string(index=False))

# ---- 3: rates per month and per ISO week ----
print("\n[3a] Click & book rate per month")
df["month"] = df["date_time"].dt.to_period("M").astype(str)
monthly = (df.groupby("month")
             .agg(count=("click_bool", "size"),
                  click_rate=("click_bool", "mean"),
                  book_rate=("booking_bool", "mean"))
             .reset_index())
monthly.to_csv(OUT / "rates_by_month.csv", index=False)
print(monthly.to_string(index=False))

print("\n[3b] Click & book rate per ISO week")
iso = df["date_time"].dt.isocalendar()
df["iso_year_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
weekly = (df.groupby("iso_year_week")
            .agg(count=("click_bool", "size"),
                 click_rate=("click_bool", "mean"),
                 book_rate=("booking_bool", "mean"))
            .reset_index())
weekly.to_csv(OUT / "rates_by_week.csv", index=False)
print(f"  weeks = {len(weekly)} (first 3, last 3 below)")
print(weekly.head(3).to_string(index=False))
print(weekly.tail(3).to_string(index=False))

# ---- 5: prop_id count distribution ----
print("\n[5] prop_id count distribution")
prop_counts = df.groupby("prop_id").size()
desc = prop_counts.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
dist = pd.DataFrame({"stat": desc.index, "value": desc.values})
extra = pd.DataFrame({
    "stat": ["unique_prop_id", "rows_total", "p99.9"],
    "value": [int(prop_counts.size), int(prop_counts.sum()),
              float(prop_counts.quantile(0.999))],
})
dist = pd.concat([dist, extra], ignore_index=True)
dist.to_csv(OUT / "prop_id_count_dist.csv", index=False)
print(dist.to_string(index=False))

# ---- 4: feature importance of TE features across V5 ensemble ----
print("\n[4] Feature importance for TE features (median across ensemble members)")
fi_dir = ROOT / "artifacts" / "v5_te_ablation_20260516_103823"
csvs = sorted(glob.glob(str(fi_dir / "importance_*.csv")))
print(f"  sources: {fi_dir.name} ({len(csvs)} member models)")
targets = ["prop_click_rate", "prop_book_rate", "prop_rel_rate"]
rows = []
for f in csvs:
    member = Path(f).stem.replace("importance_", "")
    d = pd.read_csv(f).set_index("feature")
    total_gain = d["gain"].sum()
    total_split = d["split"].sum()
    for t in targets:
        if t in d.index:
            g = float(d.loc[t, "gain"]); s = float(d.loc[t, "split"])
            rank_g = int((d["gain"].rank(ascending=False, method="min").loc[t]))
            rows.append(dict(member=member, feature=t, gain=g, split=s,
                             gain_pct=g/total_gain*100, split_pct=s/total_split*100,
                             gain_rank=rank_g))
fi_long = pd.DataFrame(rows)
fi_long.to_csv(OUT / "te_feature_importance_per_member.csv", index=False)

agg = (fi_long.groupby("feature")
        .agg(n_models=("gain", "size"),
             gain_median=("gain", "median"),
             gain_pct_median=("gain_pct", "median"),
             gain_pct_mean=("gain_pct", "mean"),
             split_median=("split", "median"),
             split_pct_median=("split_pct", "median"),
             gain_rank_median=("gain_rank", "median"),
             gain_rank_min=("gain_rank", "min"),
             gain_rank_max=("gain_rank", "max"))
        .reset_index()
        .sort_values("gain_pct_median", ascending=False))
agg.to_csv(OUT / "te_feature_importance.csv", index=False)
print(agg.to_string(index=False))

# ---- summary ----
def fmt(d, floatfmt=".4f"):
    return d.to_string(index=False, float_format=lambda x: format(x, floatfmt))

md = OUT / "summary.md"
with md.open("w") as f:
    f.write("# EDA summary — 5 outputs for the report\n\n")
    f.write(f"Source: `{TRAIN_FILE}` ({len(df):,} rows)\n\n")
    f.write("## 1+2. Click & booking rate by position\n")
    f.write("File: `position_rates.csv`. Selected positions:\n\n```\n")
    sel = pos[pos["position"].isin([1, 5, 10, 15, 20, 25, 30, 35, 40])]
    f.write(fmt(sel) + "\n```\n\n")
    f.write("## 3. Rates by month / week\n")
    f.write("Files: `rates_by_month.csv`, `rates_by_week.csv`.\n\n")
    f.write("### Monthly\n\n```\n")
    f.write(fmt(monthly) + "\n```\n\n")
    f.write(f"## 4. TE feature importance (V5 ensemble, {len(csvs)} members)\n\n```\n")
    f.write(fmt(agg, ".2f") + "\n```\n\n")
    f.write("Per-member detail: `te_feature_importance_per_member.csv`\n\n")
    f.write("## 5. prop_id count distribution\n\n```\n")
    f.write(fmt(dist, ".2f") + "\n```\n")

print(f"\nAll outputs written to {OUT}/")
print("  files:", *(p.name for p in sorted(OUT.iterdir())))
