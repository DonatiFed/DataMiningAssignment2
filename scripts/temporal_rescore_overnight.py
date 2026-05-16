"""Temporal rescore for the 85 overnight boosters.

No retraining. Loads each saved booster from `models/overnight_20260516_003443/`,
predicts on the cached temporal_val parquet, and writes per-model metrics
(NDCG@5, Recall@1, Recall@5, MBR) joined with random-split scores from
`artifacts/overnight_20260516_003443/model_results.csv`.

Outputs:
  diagnostics/temporal_rescore_overnight/temporal_rescore.csv
  diagnostics/temporal_rescore_overnight/temporal_pred_<config>.npy   (top-30)
  diagnostics/temporal_rescore_overnight/README.md
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.temporal_validation import eval_metrics  # noqa: E402

CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"
MODELS_DIR = ROOT / "models" / "overnight_20260516_003443"
RANDOM_RESULTS = ROOT / "artifacts" / "overnight_20260516_003443" / "model_results.csv"
OUT = ROOT / "diagnostics" / "temporal_rescore_overnight"
OUT.mkdir(parents=True, exist_ok=True)
V4_ANCHOR_TEMPORAL = 0.40401
TOP_N_TO_SAVE = 30


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- Load val once ----------------------------------------------------------
log("Loading temporal_val parquet…")
val = pd.read_parquet(CACHE_VAL)
val = val.sort_values("srch_id").reset_index(drop=True)
assert "relevance" in val.columns, "val missing 'relevance'"
log(f"  rows={len(val):,}  searches={val['srch_id'].nunique():,}  cols={val.shape[1]}")

# ---- Random-split scores for join ------------------------------------------
random_df = pd.read_csv(RANDOM_RESULTS)
random_lookup = random_df.set_index("config_name")
log(f"  random-split results: {len(random_df)} configs")

# ---- Boosters ---------------------------------------------------------------
model_paths = sorted(MODELS_DIR.glob("model_*.txt"))
log(f"  found {len(model_paths)} boosters in {MODELS_DIR}")
assert len(model_paths) > 0, f"no boosters in {MODELS_DIR}"

rows = []
preds_keep: dict[str, np.ndarray] = {}
t_start = time.time()
for i, mp in enumerate(model_paths, 1):
    config_name = mp.stem.replace("model_", "")
    t0 = time.time()
    booster = lgb.Booster(model_file=str(mp))
    feat_cols = booster.feature_name()
    missing = [c for c in feat_cols if c not in val.columns]
    if missing:
        log(f"  [{i:2d}/{len(model_paths)}] {config_name}: SKIP "
            f"({len(missing)} unknown feats, e.g. {missing[:3]})")
        continue

    scores = booster.predict(val[feat_cols]).astype(np.float32)
    m = eval_metrics(val, scores)

    def _g(col, default, cast=None):
        if config_name not in random_lookup.index:
            return default
        v = random_lookup.loc[config_name, col]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return cast(v) if cast else v

    rnd_score = _g("ndcg5", float("nan"), float)
    rnd_best_iter = _g("best_iter", -1, lambda x: int(float(x)))
    rnd_label_gain = _g("label_gain", "", str)
    rnd_num_leaves = _g("num_leaves", -1, lambda x: int(float(x)))
    rnd_lr = _g("learning_rate", float("nan"), float)
    rnd_n_feats = _g("n_features", len(feat_cols), lambda x: int(float(x)))
    rnd_seed = _g("seed", -1, lambda x: int(float(x)))

    rows.append({
        "config_name": config_name,
        "label_gain": rnd_label_gain,
        "seed": rnd_seed,
        "num_leaves": rnd_num_leaves,
        "learning_rate": rnd_lr,
        "best_iter_random": rnd_best_iter,
        "n_features": len(feat_cols),
        "n_features_reported": rnd_n_feats,
        "random_ndcg5": rnd_score,
        "temporal_ndcg5": float(m["ndcg5"]),
        "temporal_recall1": float(m["recall1"]),
        "temporal_recall5": float(m["recall5"]),
        "temporal_mbr": float(m["mean_booked_rank"]),
        "delta_temporal_vs_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
        "delta_temporal_vs_random": float(m["ndcg5"]) - rnd_score if rnd_score == rnd_score else float("nan"),
    })
    preds_keep[config_name] = scores
    log(f"  [{i:2d}/{len(model_paths)}] {config_name}: temporal NDCG@5={m['ndcg5']:.5f}  "
        f"(random {rnd_score:.5f}, Δ_t={float(m['ndcg5']) - V4_ANCHOR_TEMPORAL:+.5f})  "
        f"in {time.time()-t0:.1f}s")

log(f"\nFinished {len(rows)}/{len(model_paths)} models in {(time.time()-t_start)/60:.1f} min")

results = pd.DataFrame(rows).sort_values("temporal_ndcg5", ascending=False).reset_index(drop=True)
results.to_csv(OUT / "temporal_rescore.csv", index=False)
log(f"Results CSV: {OUT / 'temporal_rescore.csv'}")

# ---- Save top-N temporal predictions ---------------------------------------
log(f"\nSaving top-{TOP_N_TO_SAVE} temporal predictions as .npy …")
top_configs = results.head(TOP_N_TO_SAVE)["config_name"].tolist()
for cfg in top_configs:
    np.save(OUT / f"temporal_pred_{cfg}.npy", preds_keep[cfg])
log(f"  saved {len(top_configs)} .npy files")

# ---- README -----------------------------------------------------------------
def fmt(d, fmt_str=".5f"):
    return d.to_string(index=False, float_format=lambda x: format(x, fmt_str))

readme = OUT / "README.md"
with readme.open("w") as f:
    f.write("# Temporal rescore — overnight boosters\n\n")
    f.write(f"Source: `{MODELS_DIR}` ({len(model_paths)} boosters)\n")
    f.write(f"Predicted on: `{CACHE_VAL}` "
            f"({len(val):,} rows / {val['srch_id'].nunique():,} searches)\n")
    f.write(f"Successfully predicted: {len(rows)}/{len(model_paths)}\n\n")

    f.write("## ⚠️ LEAKAGE WARNING — read first\n\n")
    f.write(
        "The overnight boosters were trained on a **random 90% split** "
        "(seed=456, val_frac=0.1) of the full training set — NOT on temporal_train. "
        "Roughly half of our temporal_val rows (dated after 2013-05-21) were inside "
        "their training data. Because of that:\n\n"
        "- The temporal NDCG@5 values reported below are **inflated by memorisation**, "
        "  NOT directly comparable to V4_ANCHOR_TEMPORAL = 0.40401 "
        "  (which was trained on temporal_train only).\n"
        "- These predictions **cannot be used as ensemble members for temporal val** "
        "  — they would import the leakage into the ensemble.\n"
        "- The *relative ranking* of configs may still be informative for picking "
        "  Kaggle-promising candidates, since Kaggle's test set has no overlap with "
        "  training. For ensemble-on-Kaggle, the model boosters themselves can be "
        "  reused; only their *temporal val* scores are unreliable.\n\n"
        "Treat the numbers below as a config-ranking diagnostic, not a metric.\n\n"
    )

    f.write("## Headline (leakage-affected, see warning)\n\n")
    best = results.iloc[0]
    f.write(f"- Best temporal NDCG@5: **{best['temporal_ndcg5']:.5f}** "
            f"({best['config_name']}), Δ vs V4_ANCHOR({V4_ANCHOR_TEMPORAL}) = "
            f"**{best['delta_temporal_vs_anchor']:+.5f}**\n")
    f.write(f"- Random-split score for same config: {best['random_ndcg5']:.5f} "
            f"(temporal − random = {best['delta_temporal_vs_random']:+.5f})\n\n")

    # random vs temporal correlation
    valid = results.dropna(subset=["random_ndcg5"])
    if len(valid) > 5:
        corr_p = valid[["random_ndcg5", "temporal_ndcg5"]].corr().iloc[0, 1]
        corr_s = valid[["random_ndcg5", "temporal_ndcg5"]].corr(method="spearman").iloc[0, 1]
        f.write(f"- Random ↔ temporal correlation across {len(valid)} configs: "
                f"Pearson={corr_p:.3f}, Spearman={corr_s:.3f}\n\n")

    f.write("## Top 20 by temporal NDCG@5\n\n```\n")
    show_cols = ["config_name", "label_gain", "seed", "num_leaves",
                 "n_features", "random_ndcg5", "temporal_ndcg5",
                 "delta_temporal_vs_anchor"]
    f.write(fmt(results.head(20)[show_cols]))
    f.write("\n```\n\n")

    f.write("## Bottom 5 by temporal NDCG@5\n\n```\n")
    f.write(fmt(results.tail(5)[show_cols]))
    f.write("\n```\n\n")

    f.write(f"## Files\n\n")
    f.write(f"- `temporal_rescore.csv` — full {len(rows)}-row table, all metrics\n")
    f.write(f"- `temporal_pred_<config>.npy` — top {TOP_N_TO_SAVE} predictions, "
            "for downstream ensemble experiments\n")
log(f"README: {readme}")

# ---- Print summary ----------------------------------------------------------
log("\n=== TOP 10 by temporal NDCG@5 ===")
print(results.head(10)[
    ["config_name", "label_gain", "num_leaves", "n_features",
     "random_ndcg5", "temporal_ndcg5", "delta_temporal_vs_anchor"]
].to_string(index=False))
