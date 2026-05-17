"""Phase 7 weighted-ensemble calibration batch.

Phase A: weighted rank-average ensembles between V6 LOO-9 and the 4 eligible
Phase 7 single-feature models. Tests small weights (5–25%) for each member
instead of equal-weight (50%).

Phase B (conditional, only if Phase A < 0.40950): trains 2 multi-feature
models from Phase 7 finding combinations, then repeats the weighted-ensemble
grid.

Hard rules followed:
- Phase A uses ONLY saved predictions from diagnostics/phase7_batch_*/.
- Per-test try/except so one failure does not kill the batch.
- Atomic writes everywhere.
- FATAL.txt on outer crash, per-test ERROR_<id>.txt on inner failure.
- Conditional submission only if best ≥ 0.40950.
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import get_feature_columns  # noqa: E402
from src.features import compute_position_propensity, FORBIDDEN_FEATURES  # noqa: E402
from pipelines.temporal_validation import (  # noqa: E402
    compute_ipw_weights, make_group_counts, eval_metrics, BASE_PARAMS,
)
from pipelines.phase7_batch import (  # noqa: E402
    feat_brand_x_domestic, feat_query_difficulty_index,
    feat_long_window_x_top_quartile_price, feat_prop_rare_x_long_trip,
    grouped_rank, metrics_from_avg_rank, label_remap,
)

# ============================================================================
# Constants
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
V4_ANCHOR_TEMPORAL = 0.40401
V6_LOO9_TEMPORAL = 0.40896
SUBMISSION_THRESHOLD = 0.40950
NEAR_MISS_LO = 0.40920

CACHE_TRAIN = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_train.parquet"
CACHE_VAL = ROOT / "diagnostics" / "eval_variants" / "base_features_temporal_val.parquet"

V6_DIR = ROOT / "diagnostics" / "v6_20260516_163559"
V6_MEMBERS = [
    "lambdarank_base", "lambdarank_click3", "lambdarank_bal15",
    "lambdarank_book50", "lambdarank_noipw", "rank_xendcg",
    "lambdarank_randup", "CP", "DS",
]

PHASE7_DIR = ROOT / "diagnostics" / "phase7_batch_20260516_203100"
ELIGIBLE = [
    "brand_x_domestic",
    "query_difficulty_index",
    "is_long_window_x_top_quartile_price",
    "prop_rare_x_long_trip",
]

# Phase A weight grids
SINGLE_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]
TOP2_WEIGHTS = [0.05, 0.10, 0.15]
TOP3_WEIGHTS = [0.05, 0.075, 0.10]
TOTAL_WEIGHT_CAP = 0.25

# Phase B combos
COMBO_SPECS = [
    {
        "id": "combo_brand_dom_x_query_diff",
        "builders": [feat_brand_x_domestic, feat_query_difficulty_index],
    },
    {
        "id": "combo_long_window_price_x_rare_long_trip",
        "builders": [feat_long_window_x_top_quartile_price, feat_prop_rare_x_long_trip],
    },
]

OUT = ROOT / "diagnostics" / f"phase7_weighted_batch_{TIMESTAMP}"
ERRORS_DIR = OUT / "errors"
PREDS_DIR = OUT / "predictions"
MODELS_DIR = OUT / "models"


# ============================================================================
# Logging + atomic IO
# ============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def safe_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


# ============================================================================
# Phase A — weighted rank ensemble using saved predictions
# ============================================================================
def load_member_rank(member_id: str, srch_id: np.ndarray) -> np.ndarray:
    """Load Phase 7 single-feature val_pred and convert to within-srch_id rank."""
    p = PHASE7_DIR / "predictions" / f"val_pred_{member_id}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    s = np.load(p).astype(np.float32)
    return grouped_rank(srch_id, s)


def load_v6_baseline_rank(srch_id: np.ndarray) -> np.ndarray:
    ranks = []
    for m in V6_MEMBERS:
        p = V6_DIR / f"val_pred_{m}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        s = np.load(p).astype(np.float32)
        ranks.append(grouped_rank(srch_id, s))
    return np.mean(ranks, axis=0)


def weighted_rank(v6_rank: np.ndarray, member_ranks: list[np.ndarray],
                   weights: list[float]) -> np.ndarray:
    """final = (1 - sum(w_i)) * V6_rank + sum(w_i * member_i_rank)."""
    total_w = sum(weights)
    assert 0 < total_w <= 0.5, f"weight total out of range: {total_w}"
    base_w = 1.0 - total_w
    out = base_w * v6_rank
    for w, r in zip(weights, member_ranks):
        out = out + w * r
    return out


def score_weighted(val_feat: pd.DataFrame, v6_rank: np.ndarray,
                    members: list[str], member_ranks_map: dict[str, np.ndarray],
                    weights: list[float]) -> dict:
    mr = [member_ranks_map[m] for m in members]
    avg = weighted_rank(v6_rank, mr, weights)
    m = metrics_from_avg_rank(val_feat, avg)
    return {
        "n_members": 1 + len(members),
        "members_added": "+".join(members),
        "weights_added": ",".join(f"{w:.4f}" for w in weights),
        "v6_weight": 1.0 - sum(weights),
        "total_added_weight": sum(weights),
        "ndcg5": float(m["ndcg5"]),
        "recall1": float(m["recall1"]),
        "recall5": float(m["recall5"]),
        "mean_booked_rank": float(m["mean_booked_rank"]),
        "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
        "delta_vs_v4_anchor": float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL,
    }


def phase_a(val_feat: pd.DataFrame, v6_rank: np.ndarray,
             member_ranks_map: dict[str, np.ndarray]) -> pd.DataFrame:
    log("=" * 70)
    log("PHASE A — weighted rank ensembles (V6 LOO-9 base + small member weights)")
    log("=" * 70)

    rows = []
    # Baseline (V6 alone)
    base_metrics = metrics_from_avg_rank(val_feat, v6_rank)
    rows.append({
        "test_id": "v6_loo9_baseline",
        "phase": "baseline",
        "n_members": 1,
        "members_added": "",
        "weights_added": "",
        "v6_weight": 1.0,
        "total_added_weight": 0.0,
        "ndcg5": float(base_metrics["ndcg5"]),
        "recall1": float(base_metrics["recall1"]),
        "recall5": float(base_metrics["recall5"]),
        "mean_booked_rank": float(base_metrics["mean_booked_rank"]),
        "delta_vs_v6_loo9": 0.0,
        "delta_vs_v4_anchor": float(base_metrics["ndcg5"]) - V4_ANCHOR_TEMPORAL,
    })
    log(f"  v6_loo9_baseline NDCG@5 = {base_metrics['ndcg5']:.5f}  "
        f"(target {V6_LOO9_TEMPORAL:.5f})")

    # A.1 — single member grid
    log("\nA.1 — single member × 5 weights = 20 tests")
    best_single: dict[str, float] = {}
    for member in ELIGIBLE:
        for w in SINGLE_WEIGHTS:
            test_id = f"v6+{member}@w={w:.2f}"
            try:
                res = score_weighted(val_feat, v6_rank, [member], member_ranks_map, [w])
                res["test_id"] = test_id
                res["phase"] = "A1_single"
                rows.append(res)
                if member not in best_single or res["ndcg5"] > best_single[member]:
                    best_single[member] = res["ndcg5"]
                log(f"  {test_id:60s} NDCG@5={res['ndcg5']:.5f}  "
                    f"Δ_v6={res['delta_vs_v6_loo9']:+.5f}")
            except Exception as e:
                log(f"  ✗ {test_id} FAILED: {e}")
                safe_write_text(
                    f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                    ERRORS_DIR / f"ERROR_{test_id.replace('/', '_')}.txt"
                )
        safe_write_df(pd.DataFrame(rows), OUT / "weighted_ensemble_results.csv")

    # Identify top 2 / top 3 single members by best NDCG
    ranked = sorted(best_single.items(), key=lambda x: -x[1])
    log(f"\n  member ranking by best single-weight NDCG: "
        f"{[(m, f'{n:.5f}') for m, n in ranked]}")
    top2 = [m for m, _ in ranked[:2]]
    top3 = [m for m, _ in ranked[:3]]

    # A.2 — top-2 grid
    log(f"\nA.2 — top-2 ({top2}) grid weights in {TOP2_WEIGHTS}, total ≤ {TOTAL_WEIGHT_CAP}")
    for w1, w2 in product(TOP2_WEIGHTS, repeat=2):
        if w1 + w2 > TOTAL_WEIGHT_CAP:
            continue
        test_id = f"v6+{top2[0]}@{w1:.3f}+{top2[1]}@{w2:.3f}"
        try:
            res = score_weighted(val_feat, v6_rank, top2, member_ranks_map, [w1, w2])
            res["test_id"] = test_id
            res["phase"] = "A2_top2"
            rows.append(res)
            log(f"  {test_id:70s} NDCG@5={res['ndcg5']:.5f}  "
                f"Δ_v6={res['delta_vs_v6_loo9']:+.5f}")
        except Exception as e:
            log(f"  ✗ {test_id} FAILED: {e}")
            safe_write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                ERRORS_DIR / f"ERROR_{test_id.replace('/', '_')}.txt"
            )
    safe_write_df(pd.DataFrame(rows), OUT / "weighted_ensemble_results.csv")

    # A.3 — top-3 grid
    log(f"\nA.3 — top-3 ({top3}) grid weights in {TOP3_WEIGHTS}, total ≤ {TOTAL_WEIGHT_CAP}")
    for w1, w2, w3 in product(TOP3_WEIGHTS, repeat=3):
        if w1 + w2 + w3 > TOTAL_WEIGHT_CAP:
            continue
        test_id = f"v6+{top3[0]}@{w1:.3f}+{top3[1]}@{w2:.3f}+{top3[2]}@{w3:.3f}"
        try:
            res = score_weighted(val_feat, v6_rank, top3, member_ranks_map, [w1, w2, w3])
            res["test_id"] = test_id
            res["phase"] = "A3_top3"
            rows.append(res)
            log(f"  {test_id:90s} NDCG@5={res['ndcg5']:.5f}  "
                f"Δ_v6={res['delta_vs_v6_loo9']:+.5f}")
        except Exception as e:
            log(f"  ✗ {test_id} FAILED: {e}")
            safe_write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                ERRORS_DIR / f"ERROR_{test_id.replace('/', '_')}.txt"
            )
    safe_write_df(pd.DataFrame(rows), OUT / "weighted_ensemble_results.csv")

    return pd.DataFrame(rows)


# ============================================================================
# Phase A LOO on best
# ============================================================================
def loo_on_best(results_df: pd.DataFrame, val_feat: pd.DataFrame,
                 v6_rank: np.ndarray, member_ranks_map: dict[str, np.ndarray]) -> pd.DataFrame:
    log("\n--- LOO on best Phase A ensemble ---")
    # Exclude the pure baseline row when picking the best multi-member ensemble.
    df = results_df.copy()
    multi = df[df["n_members"] > 1].sort_values("ndcg5", ascending=False)
    if multi.empty:
        log("  no multi-member ensemble to LOO")
        return pd.DataFrame()
    best = multi.iloc[0]
    log(f"  best multi-member ensemble: {best['test_id']}  NDCG@5={best['ndcg5']:.5f}")

    members = best["members_added"].split("+")
    weights = [float(w) for w in best["weights_added"].split(",")]
    if len(members) < 2:
        log("  best is single-member; LOO not meaningful, skipping")
        return pd.DataFrame()

    loo_rows = []
    for i, drop in enumerate(members):
        remaining_members = [m for j, m in enumerate(members) if j != i]
        remaining_weights = [w for j, w in enumerate(weights) if j != i]
        if not remaining_members:
            continue
        try:
            res = score_weighted(val_feat, v6_rank, remaining_members, member_ranks_map, remaining_weights)
            loo_rows.append({
                "dropped": drop,
                "n_remaining": len(remaining_members),
                "members_remaining": "+".join(remaining_members),
                "weights_remaining": ",".join(f"{w:.4f}" for w in remaining_weights),
                "ndcg5": res["ndcg5"],
                "delta_vs_best": res["ndcg5"] - best["ndcg5"],
            })
            log(f"  drop {drop:50s} NDCG@5={res['ndcg5']:.5f}  Δ={res['ndcg5'] - best['ndcg5']:+.5f}")
        except Exception as e:
            log(f"  ✗ LOO drop {drop} FAILED: {e}")
    loo_df = pd.DataFrame(loo_rows).sort_values("delta_vs_best")
    safe_write_df(loo_df, OUT / "leave_one_out.csv")
    return loo_df


# ============================================================================
# Phase B — train 2 combo models, then weighted ensemble each
# ============================================================================
def train_combo(spec: dict, base_train: pd.DataFrame, base_val: pd.DataFrame,
                 propensity) -> tuple[lgb.Booster, int, np.ndarray, list[str], list[str]]:
    """Build all builder features, train V4_ANCHOR, return booster + best_iter + val_scores."""
    train = base_train.copy()
    val = base_val.copy()
    added_cols = []
    for builder in spec["builders"]:
        info, _drifts = builder(train, val)
        added_cols.extend(info["new_cols"])

    feat_cols = [c for c in get_feature_columns(train) if c in val.columns]
    leaked = set(feat_cols) & FORBIDDEN_FEATURES
    assert not leaked, f"forbidden leaked: {leaked}"
    weights = compute_ipw_weights(train, propensity, clip_hi=10.0, clip_lo=0.1)
    train_label = label_remap(train["relevance"])
    val_label = label_remap(val["relevance"])
    train_groups = make_group_counts(train)
    val_groups = make_group_counts(val)

    params = BASE_PARAMS.copy()
    params["label_gain"] = "0,1,15"

    ds_tr = lgb.Dataset(train[feat_cols], label=train_label, group=train_groups, weight=weights)
    ds_va = lgb.Dataset(val[feat_cols], label=val_label, group=val_groups, reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=2000,
                       valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
                       callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    best_iter = int(model.best_iteration)
    scores = model.predict(val[feat_cols]).astype(np.float32)
    return model, best_iter, scores, feat_cols, added_cols


def phase_b(val_feat: pd.DataFrame, v6_rank: np.ndarray,
             member_ranks_map: dict[str, np.ndarray],
             rows_phase_a: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("\n" + "=" * 70)
    log("PHASE B — train 2 combo models + weighted ensemble grid")
    log("=" * 70)
    base_train = pd.read_parquet(CACHE_TRAIN).sort_values("srch_id").reset_index(drop=True)
    base_val = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    propensity = compute_position_propensity(base_train)

    optional_rows = []
    weighted_rows = []

    for spec in COMBO_SPECS:
        cid = spec["id"]
        log(f"\n--- combo: {cid} ---")
        model_path = MODELS_DIR / f"model_{cid}.txt"
        pred_path = PREDS_DIR / f"val_pred_{cid}.npy"

        try:
            t0 = time.time()
            if model_path.exists() and pred_path.exists():
                log(f"  RESUME: {model_path.name} + {pred_path.name} exist; reload")
                booster = lgb.Booster(model_file=str(model_path))
                scores = np.load(pred_path).astype(np.float32)
                best_iter = booster.current_iteration()
                feat_cols = booster.feature_name()
                added_cols = [c for c in feat_cols
                              if c not in get_feature_columns(base_train)]
            else:
                booster, best_iter, scores, feat_cols, added_cols = train_combo(
                    spec, base_train, base_val, propensity
                )
                booster.save_model(str(model_path))
                np.save(pred_path, scores)
            m = eval_metrics(base_val, scores)
            delta = float(m["ndcg5"]) - V4_ANCHOR_TEMPORAL

            row = {
                "combo_id": cid,
                "status": "ok",
                "added_features": ",".join(added_cols),
                "n_features": len(feat_cols),
                "best_iter": best_iter,
                "ndcg5": float(m["ndcg5"]),
                "recall1": float(m["recall1"]),
                "recall5": float(m["recall5"]),
                "mean_booked_rank": float(m["mean_booked_rank"]),
                "delta_vs_v4_anchor": delta,
                "delta_vs_v6_loo9": float(m["ndcg5"]) - V6_LOO9_TEMPORAL,
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
            optional_rows.append(row)
            log(f"  ✓ {cid} NDCG@5={m['ndcg5']:.5f}  Δ_anchor={delta:+.5f}  best_iter={best_iter}")

            # add this combo member to member_ranks_map for the weighted ensemble grid
            combo_rank = grouped_rank(base_val["srch_id"].values, scores)
            member_ranks_map[cid] = combo_rank

            # Weighted single-member grid for this combo
            for w in SINGLE_WEIGHTS:
                test_id = f"v6+{cid}@w={w:.2f}"
                try:
                    res = score_weighted(base_val, v6_rank, [cid], member_ranks_map, [w])
                    res["test_id"] = test_id
                    res["phase"] = "B_single"
                    weighted_rows.append(res)
                    log(f"    {test_id:70s} NDCG@5={res['ndcg5']:.5f}  "
                        f"Δ_v6={res['delta_vs_v6_loo9']:+.5f}")
                except Exception as e:
                    log(f"    ✗ {test_id} FAILED: {e}")

            del booster, scores
            gc.collect()
        except Exception as e:
            log(f"  ✗ combo {cid} FAILED: {e}")
            safe_write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                ERRORS_DIR / f"ERROR_combo_{cid}.txt"
            )
            optional_rows.append({"combo_id": cid, "status": f"failed:{type(e).__name__}",
                                  "error": str(e)[:200]})

        safe_write_df(pd.DataFrame(optional_rows), OUT / "optional_model_results.csv")
        # Append weighted rows into the main weighted CSV (additive)
        full = rows_phase_a + weighted_rows
        safe_write_df(pd.DataFrame(full), OUT / "weighted_ensemble_results.csv")

    return pd.DataFrame(optional_rows), pd.DataFrame(weighted_rows)


# ============================================================================
# README writer
# ============================================================================
def write_readme(phase_a_df, loo_df, phase_b_model_df, phase_b_weighted_df,
                  best_row, sub_status, recommendation: str, t_start: float):
    lines = [f"# Phase 7 weighted-ensemble batch — {TIMESTAMP}\n"]
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} • "
                 f"elapsed {(time.time() - t_start)/60:.1f} min_\n")
    lines.append(f"## Baselines\n")
    lines.append(f"- V4_ANCHOR_TEMPORAL = {V4_ANCHOR_TEMPORAL}")
    lines.append(f"- V6_LOO9_TEMPORAL   = {V6_LOO9_TEMPORAL}")
    lines.append(f"- submission threshold = {SUBMISSION_THRESHOLD}  "
                 f"(+{SUBMISSION_THRESHOLD - V6_LOO9_TEMPORAL:.5f} over V6)\n")

    lines.append("## 1. Did any Phase 7 member help when weighted lightly?\n")
    top_a = phase_a_df[phase_a_df["n_members"] > 1].sort_values("ndcg5", ascending=False).head(10)
    if len(top_a):
        any_helped = bool((top_a["delta_vs_v6_loo9"] > 0).any())
        lines.append(f"**Answer:** {'YES' if any_helped else 'NO'} — "
                     f"best weighted ensemble = {top_a.iloc[0]['ndcg5']:.5f}, "
                     f"Δ vs V6 LOO-9 = {top_a.iloc[0]['delta_vs_v6_loo9']:+.5f}.\n")
    lines.append("Top 10 Phase A weighted ensembles:\n```")
    cols = ["test_id", "ndcg5", "delta_vs_v6_loo9", "v6_weight", "total_added_weight"]
    lines.append(top_a[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("```\n")

    lines.append("## 2. Best ensemble and weights\n")
    if best_row is not None:
        lines.append(f"- **Test ID:** `{best_row['test_id']}`")
        lines.append(f"- **NDCG@5:** {best_row['ndcg5']:.5f}")
        lines.append(f"- **Members added:** `{best_row['members_added']}`")
        lines.append(f"- **Weights:** `{best_row['weights_added']}` (V6 weight = {best_row['v6_weight']:.4f})")
        lines.append(f"- **Δ vs V6 LOO-9 (0.40896):** {best_row['delta_vs_v6_loo9']:+.5f}")
        lines.append(f"- **Δ vs V4_ANCHOR (0.40401):** {best_row['delta_vs_v4_anchor']:+.5f}\n")

    lines.append("## 3. Leave-one-out on best Phase A ensemble\n")
    if loo_df is not None and len(loo_df):
        lines.append("```")
        lines.append(loo_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        lines.append("```\n")
    else:
        lines.append("_(LOO not run — best was a single-member ensemble.)_\n")

    if phase_b_model_df is not None and len(phase_b_model_df):
        lines.append("## 4. Phase B — combo models\n```")
        keep_cols = [c for c in ["combo_id", "status", "ndcg5", "delta_vs_v4_anchor",
                                  "delta_vs_v6_loo9", "best_iter", "n_features"]
                     if c in phase_b_model_df.columns]
        lines.append(phase_b_model_df[keep_cols].to_string(
            index=False, float_format=lambda x: f"{x:.5f}"))
        lines.append("```\n")
        if phase_b_weighted_df is not None and len(phase_b_weighted_df):
            lines.append("Top 5 Phase B weighted ensembles:\n```")
            sub = phase_b_weighted_df.sort_values("ndcg5", ascending=False).head(5)
            lines.append(sub[["test_id", "ndcg5", "delta_vs_v6_loo9",
                              "v6_weight"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
            lines.append("```\n")

    lines.append(f"## 5. Submission status: **{sub_status}**\n")
    lines.append(f"## 6. Recommendation\n\n{recommendation}\n")
    safe_write_text("\n".join(lines), OUT / "README.md")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-only", action="store_true",
                        help="Skip Phase B even if Phase A fails to reach threshold")
    args = parser.parse_args()

    for d in (OUT, ERRORS_DIR, PREDS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"PHASE 7 WEIGHTED BATCH — {TIMESTAMP}")
    log(f"out: {OUT}")
    log(f"reading V6 baseline preds from: {V6_DIR}")
    log(f"reading Phase 7 member preds from: {PHASE7_DIR / 'predictions'}")
    log(f"submission threshold: {SUBMISSION_THRESHOLD}")

    # Sanity check inputs
    for m in V6_MEMBERS:
        assert (V6_DIR / f"val_pred_{m}.npy").exists(), f"missing {m} val_pred"
    for m in ELIGIBLE:
        assert (PHASE7_DIR / "predictions" / f"val_pred_{m}.npy").exists(), \
            f"missing Phase 7 {m} val_pred"

    # Load val parquet (need for srch_id alignment + relevance)
    log("Loading temporal_val parquet…")
    val_feat = pd.read_parquet(CACHE_VAL).sort_values("srch_id").reset_index(drop=True)
    srch = val_feat["srch_id"].values
    log(f"  rows={len(val_feat):,}  searches={val_feat['srch_id'].nunique():,}")

    # Build V6 baseline rank + each Phase 7 member rank
    log("Building V6 LOO-9 baseline rank-average…")
    v6_rank = load_v6_baseline_rank(srch)
    member_ranks_map = {m: load_member_rank(m, srch) for m in ELIGIBLE}
    log(f"  loaded {len(member_ranks_map)} member rank arrays")

    # Phase A
    phase_a_df = phase_a(val_feat, v6_rank, member_ranks_map)
    safe_write_df(phase_a_df, OUT / "weighted_ensemble_results.csv")

    # LOO on Phase A best
    loo_df = loo_on_best(phase_a_df, val_feat, v6_rank, member_ranks_map)

    # Best single & best weights table
    best_multi = phase_a_df[phase_a_df["n_members"] > 1].sort_values("ndcg5", ascending=False)
    if len(best_multi):
        best_a_row = best_multi.iloc[0]
        safe_write_df(best_multi.head(10), OUT / "best_weights.csv")
    else:
        best_a_row = phase_a_df.iloc[0]

    log(f"\nPHASE A best: NDCG@5 = {best_a_row['ndcg5']:.5f}  "
        f"Δ_v6 = {best_a_row['delta_vs_v6_loo9']:+.5f}")

    # Phase B (only if Phase A < threshold and not phase-a-only)
    phase_b_models_df = None
    phase_b_weighted_df = None
    if float(best_a_row["ndcg5"]) >= SUBMISSION_THRESHOLD or args.phase_a_only:
        if args.phase_a_only:
            log("Phase B skipped (--phase-a-only).")
        else:
            log(f"Phase A already ≥ threshold ({SUBMISSION_THRESHOLD}); Phase B skipped.")
    else:
        log(f"Phase A best ({best_a_row['ndcg5']:.5f}) < threshold "
            f"({SUBMISSION_THRESHOLD}); running Phase B.")
        try:
            phase_b_models_df, phase_b_weighted_df = phase_b(
                val_feat, v6_rank, member_ranks_map,
                phase_a_df.to_dict(orient="records"),
            )
        except Exception as e:
            log(f"Phase B FAILED: {e}")
            safe_write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                             ERRORS_DIR / "PHASE_B_ERROR.txt")

    # Pick global best across A + B
    full = pd.read_csv(OUT / "weighted_ensemble_results.csv")
    best_global = full.sort_values("ndcg5", ascending=False).iloc[0]
    log(f"\nGLOBAL best ensemble: {best_global['test_id']}  "
        f"NDCG@5={best_global['ndcg5']:.5f}  "
        f"Δ_v6={best_global['delta_vs_v6_loo9']:+.5f}")

    # Submission decision
    sub_status = "below_threshold"
    if best_global["ndcg5"] >= SUBMISSION_THRESHOLD:
        sub_status = "would_submit_but_not_implemented_in_phase_a_only_mode"
        # NOTE: this batch does NOT build a submission CSV (per user spec, only
        # Phase A used saved predictions; building a Kaggle CSV from a
        # weighted-rank ensemble requires retraining the V6 members on full
        # train + producing test predictions, which is out of scope for this
        # short batch). README will explicitly call this out.
    elif best_global["ndcg5"] >= NEAR_MISS_LO:
        sub_status = "near_miss_no_submission_built"
    else:
        sub_status = "below_threshold_no_submission"

    # Recommendation
    delta_v6 = float(best_global["ndcg5"]) - V6_LOO9_TEMPORAL
    if delta_v6 >= +0.0005:
        recommendation = (
            "Phase 7 + weighted ensembling produced a real gain over V6 LOO-9. "
            "**Recommended:** build a Kaggle submission from this ensemble. "
            "(Note: this batch did not produce a test-side submission CSV; that step "
            "requires building full-train predictions for the combo members.)"
        )
    elif delta_v6 > 0:
        recommendation = (
            f"Phase 7 weighted ensemble beat V6 by only {delta_v6:+.5f} — within noise. "
            "**Recommended:** STOP Phase 7. Move to structural changes (loss-side "
            "position handling, heterogeneous learners, adversarial reweighting). "
            "See docs/next_steps.md."
        )
    else:
        recommendation = (
            "Phase 7 weighted ensembling did NOT help. V6 LOO-9 remains best. "
            "**Recommended:** STOP Phase 7. Move to structural changes (loss-side "
            "position handling, heterogeneous learners, adversarial reweighting). "
            "See docs/next_steps.md."
        )

    write_readme(phase_a_df, loo_df, phase_b_models_df, phase_b_weighted_df,
                  best_global.to_dict(), sub_status, recommendation, t_start)

    log(f"\n=== ALL DONE in {(time.time() - t_start)/60:.1f} min ===")
    log(f"outputs: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            safe_write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                ERRORS_DIR / "FATAL.txt"
            )
        except Exception:
            pass
        log(f"FATAL: {e}")
        raise
