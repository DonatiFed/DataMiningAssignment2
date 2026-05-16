"""Validation-only ensemble search.

Uses saved val_pred_<id>.npy files from artifacts/overnight_20260516_003443/.
Rank-averages within srch_id. No retraining. No submissions.
"""
from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OVERNIGHT_RUN = ROOT / "artifacts" / "overnight_20260516_003443"
V4_LOCAL = 0.42512
V4_KAGGLE = 0.42021
ANCHOR = 0.42191
RNG_SEED = 42
RANDOM_SEARCH_N = 500
TOP_K_REPORT = 20


# ----- Eval kernel -----------------------------------------------------------
def _group_index(srch_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sort_perm, group_starts, group_sizes) for the val frame.
    Assumes val_meta is already grouped by srch_id (which it is, from build_features)."""
    # Verify it's contiguous by srch_id (group ids change monotonically in blocks)
    changes = np.concatenate([[0], np.where(np.diff(srch_id) != 0)[0] + 1, [len(srch_id)]])
    starts = changes[:-1]
    sizes = np.diff(changes)
    return starts, sizes


def compute_ranks_within_group(scores: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Within each group, rank descending: best score → 1, worst → group size.
    Ties broken by original position (stable). Returns float32 ranks."""
    ranks = np.empty_like(scores, dtype=np.float32)
    for s, n in zip(starts, sizes):
        sl = slice(s, s + n)
        # argsort descending; rank[i] = position+1 of element i in sorted order
        order = np.argsort(-scores[sl], kind="stable")
        r = np.empty(n, dtype=np.float32)
        r[order] = np.arange(1, n + 1, dtype=np.float32)
        ranks[sl] = r
    return ranks


def eval_from_scores(scores: np.ndarray, relevance: np.ndarray, starts: np.ndarray, sizes: np.ndarray,
                     k: int = 5) -> dict:
    """Compute NDCG@k, Recall@1, Recall@5, MeanBookedRank from per-row scores.
    Higher score = better. (For rank-averaging output, pass -avg_rank as scores.)"""
    log2_disc = 1.0 / np.log2(np.arange(2, k + 2))
    n_q = len(starts)
    ndcgs = np.empty(n_q, dtype=np.float64)
    booked_ranks = []
    rec1 = 0
    rec5 = 0
    for qi, (s, n) in enumerate(zip(starts, sizes)):
        sl = slice(s, s + n)
        rels = relevance[sl]
        # Best DCG: sort relevances descending
        best_rels = np.sort(rels)[::-1][:k]
        best_dcg = np.sum(best_rels * log2_disc[: len(best_rels)])
        if best_dcg == 0:
            ndcgs[qi] = 0.0
        else:
            order = np.argsort(-scores[sl], kind="stable")
            top_rels = rels[order][:k]
            dcg = np.sum(top_rels * log2_disc[: len(top_rels)])
            ndcgs[qi] = dcg / best_dcg
        # Booked hotel position(s) — relevance == 5
        booked_mask = rels == 5
        if booked_mask.any():
            order = np.argsort(-scores[sl], kind="stable")
            booked_pos_in_order = np.where(booked_mask[order])[0]
            r = int(booked_pos_in_order[0]) + 1  # 1-based rank of first booked
            booked_ranks.append(r)
            if r == 1:
                rec1 += 1
            if r <= 5:
                rec5 += 1
    n_booked = len(booked_ranks)
    return {
        "ndcg5": float(np.mean(ndcgs)),
        "recall1": rec1 / n_booked if n_booked else 0.0,
        "recall5": rec5 / n_booked if n_booked else 0.0,
        "mean_booked_rank": float(np.mean(booked_ranks)) if booked_ranks else float("nan"),
        "n_queries": int(n_q),
        "n_booked": int(n_booked),
    }


# ----- Load everything -------------------------------------------------------
def load_val_meta():
    m = pd.read_parquet(OVERNIGHT_RUN / "val_meta.parquet")
    starts, sizes = _group_index(m["srch_id"].to_numpy())
    rel = m["relevance"].to_numpy().astype(np.int16)
    return m, starts, sizes, rel


def load_candidate_pool(min_ndcg: float = 0.4210) -> pd.DataFrame:
    """All overnight models with NDCG@5 >= min_ndcg."""
    mr = pd.read_csv(ROOT / "experiment_logs" / "model_results.csv")
    mr = mr[mr["model_id"].str.startswith("overnight_20260516_003443", na=False)].copy()
    mr["config_id"] = mr["exp_id"].str.replace("OVERNIGHT_", "", regex=False)
    pool = mr[mr["local_ndcg5"] >= min_ndcg].sort_values("local_ndcg5", ascending=False).reset_index(drop=True)
    return pool[["config_id", "local_ndcg5", "recall_at_5", "mean_booked_rank", "best_iter",
                 "label_gain", "objective", "boosting"]]


def load_ranks_for(config_ids: list[str], starts: np.ndarray, sizes: np.ndarray) -> dict[str, np.ndarray]:
    """Cache pre-computed within-srch_id ranks per model."""
    cache = {}
    for cid in config_ids:
        path = OVERNIGHT_RUN / f"val_pred_{cid}.npy"
        scores = np.load(path).astype(np.float32)
        cache[cid] = compute_ranks_within_group(scores, starts, sizes)
    return cache


# ----- Ensemble combine ------------------------------------------------------
def ensemble_scores(rank_cache: dict[str, np.ndarray], members: list[str],
                    weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of ranks across members. Returns negated avg rank so
    higher = better (compatible with eval_from_scores which sorts descending)."""
    n = len(members)
    if weights is None:
        weights = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    stacked = np.stack([rank_cache[m] for m in members], axis=0)  # (n_members, n_rows)
    avg_rank = (stacked * weights[:, None]).sum(axis=0)
    return -avg_rank.astype(np.float32)  # negate so higher=better


def eval_ensemble(rank_cache, members, weights, rel, starts, sizes) -> dict:
    s = ensemble_scores(rank_cache, members, weights)
    return eval_from_scores(s, rel, starts, sizes)


# ----- Methods ---------------------------------------------------------------
def deterministic_ensembles() -> list[tuple[str, list[str], str]]:
    """List of (ensemble_id, members, note)."""
    return [
        ("E1_top3", ["B3", "F13", "D4"], "Top 3 by val NDCG"),
        ("E2_top5", ["B3", "F13", "D4", "A1", "B10"], "Top 5"),
        ("E3_top10", ["B3", "F13", "D4", "A1", "B10", "I3", "B9", "C9", "C11", "F1"], "Top 10 recommended pool"),
        ("E4_robust", ["B3", "D4", "B10", "I3", "F1"], "Diverse 5: weighting/feat/boost/row"),
        ("E5_recall_heavy", ["B3", "F13", "B9", "C11", "F1"], "5 best on Recall@5"),
        ("E6_diverse", ["B3", "D4", "I3", "B10", "F13"], "Maximum-axis-diversity 5"),
        ("E7_no_positive_only", ["B3", "D4", "A1", "B10", "I3", "B9", "C9", "C11"], "Top 10 minus F13/F1 (no row filters)"),
        ("E8_no_risky_position", ["D4", "F13", "B10", "I3", "F1"], "5 members, no B3"),
        ("E9_strong_only", ["B3", "F13", "D4", "A1"], "STRONG+EXCELLENT only"),
        ("E10_top2", ["B3", "F13"], "Top 2 by val NDCG"),
    ]


def weighting_methods(members: list[str], pool_df: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    """Return (method_name, weights_array, note) variants for a fixed member list."""
    sub = pool_df.set_index("config_id").loc[members]
    ndcgs = sub["local_ndcg5"].to_numpy()
    out = [
        ("equal", np.ones(len(members)), "uniform"),
        ("ndcg_weighted", ndcgs, "weights = val NDCG@5"),
        ("softmax_ndcg", np.exp((ndcgs - ndcgs.mean()) / 0.001), "softmax(NDCG, T=0.001)"),
        ("softmax_ndcg_T0005", np.exp((ndcgs - ndcgs.mean()) / 0.005), "softmax(NDCG, T=0.005)"),
        ("inverse_rank", 1.0 / np.arange(1, len(members) + 1, dtype=np.float64), "1/rank by val NDCG order"),
    ]
    return out


def greedy_forward(rank_cache, pool: list[str], starts, sizes, rel, start: str = "B3"):
    """Greedy forward selection: start from `start`, add the candidate that improves NDCG@5
    most at each step. Stop when no improvement."""
    selected = [start]
    remaining = [c for c in pool if c != start]
    cur = eval_ensemble(rank_cache, selected, None, rel, starts, sizes)
    path = [(start, cur["ndcg5"], cur)]
    while remaining:
        best_cand = None
        best_metric = cur
        for cand in remaining:
            mtr = eval_ensemble(rank_cache, selected + [cand], None, rel, starts, sizes)
            if mtr["ndcg5"] > best_metric["ndcg5"]:
                best_metric = mtr
                best_cand = cand
        if best_cand is None:
            break
        selected.append(best_cand)
        remaining.remove(best_cand)
        path.append((best_cand, best_metric["ndcg5"], best_metric))
        cur = best_metric
    return selected, path


def leave_one_out(rank_cache, members: list[str], starts, sizes, rel):
    """Drop each member from the ensemble, see which drop helps or hurts."""
    base = eval_ensemble(rank_cache, members, None, rel, starts, sizes)
    results = []
    for m in members:
        sub = [x for x in members if x != m]
        mtr = eval_ensemble(rank_cache, sub, None, rel, starts, sizes)
        results.append({"dropped": m, "ndcg5_without": mtr["ndcg5"], "delta_vs_base": mtr["ndcg5"] - base["ndcg5"],
                        "recall5_without": mtr["recall5"], "mbr_without": mtr["mean_booked_rank"]})
    return base, sorted(results, key=lambda r: -r["delta_vs_base"])


def random_subsets(rank_cache, pool: list[str], starts, sizes, rel, n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_samples):
        k = int(rng.integers(3, 11))  # [3, 10]
        members = list(rng.choice(pool, size=k, replace=False))
        mtr = eval_ensemble(rank_cache, members, None, rel, starts, sizes)
        results.append({
            "trial": i,
            "k": k,
            "members": ",".join(sorted(members)),
            **mtr,
        })
    return pd.DataFrame(results).sort_values("ndcg5", ascending=False).reset_index(drop=True)


def random_weights(rank_cache, members: list[str], starts, sizes, rel, n_samples=500, seed=42, alpha=1.0):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_samples):
        w = rng.dirichlet(np.full(len(members), alpha))
        mtr = eval_ensemble(rank_cache, members, w, rel, starts, sizes)
        results.append({
            "trial": i,
            "weights": ",".join(f"{x:.4f}" for x in w),
            **mtr,
        })
    return pd.DataFrame(results).sort_values("ndcg5", ascending=False).reset_index(drop=True)


def random_pool_and_weights(rank_cache, pool: list[str], starts, sizes, rel, n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_samples):
        k = int(rng.integers(3, 11))
        members = list(rng.choice(pool, size=k, replace=False))
        w = rng.dirichlet(np.full(k, 1.0))
        mtr = eval_ensemble(rank_cache, members, w, rel, starts, sizes)
        results.append({
            "trial": i,
            "k": k,
            "members": ",".join(sorted(members)),
            "weights": ",".join(f"{x:.4f}" for x in w),
            **mtr,
        })
    return pd.DataFrame(results).sort_values("ndcg5", ascending=False).reset_index(drop=True)


# ----- Main ------------------------------------------------------------------
def beats_v4(ndcg5: float) -> str:
    return "YES" if ndcg5 > V4_LOCAL else "NO"


def main():
    import time
    from datetime import datetime, timezone

    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "artifacts" / f"ensemble_search_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading val_meta + ranks…")
    val_meta, starts, sizes, rel = load_val_meta()
    print(f"  val rows={len(val_meta)} queries={len(starts)} booked={(rel==5).sum()}")

    pool_df = load_candidate_pool(min_ndcg=0.4210)
    pool_ids = pool_df["config_id"].tolist()
    print(f"  candidate pool (NDCG>=0.4210): {len(pool_ids)} models")

    rank_cache = load_ranks_for(pool_ids, starts, sizes)
    print(f"  rank cache built ({time.time()-t0:.1f}s)")

    all_results = []   # rows for ensemble_results.csv

    # ===== Deterministic ensembles ===========================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Deterministic ensembles (equal-weight rank avg)…")
    for eid, members, note in deterministic_ensembles():
        mtr = eval_ensemble(rank_cache, members, None, rel, starts, sizes)
        all_results.append({
            "ensemble_id": eid, "method": "equal_rank_avg", "n_members": len(members),
            "members": ",".join(members), "weights": "", "note": note,
            "beats_v4": beats_v4(mtr["ndcg5"]), **mtr,
        })
        # Persist scores for the deterministic equal-weight version
        s = ensemble_scores(rank_cache, members, None)
        np.save(out_dir / f"ensemble_val_pred_{eid}.npy", s)
        json.dump({"ensemble_id": eid, "method": "equal_rank_avg", "members": members,
                   "weights": None, "metrics": mtr, "note": note},
                  open(out_dir / f"ensemble_config_{eid}.json", "w"), indent=2)
        print(f"  {eid:22s} n={len(members):2d} ndcg5={mtr['ndcg5']:.5f} beats_v4={beats_v4(mtr['ndcg5'])}")

    # ===== Weighting methods on E3_top10 =====================================
    top10 = ["B3", "F13", "D4", "A1", "B10", "I3", "B9", "C9", "C11", "F1"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Weighting methods on E3_top10…")
    for mname, w, mnote in weighting_methods(top10, pool_df):
        mtr = eval_ensemble(rank_cache, top10, w, rel, starts, sizes)
        wnorm = w / w.sum()
        all_results.append({
            "ensemble_id": f"E3_top10__{mname}",
            "method": mname, "n_members": 10,
            "members": ",".join(top10),
            "weights": ",".join(f"{x:.4f}" for x in wnorm),
            "note": mnote,
            "beats_v4": beats_v4(mtr["ndcg5"]),
            **mtr,
        })
        print(f"  E3_top10__{mname:20s} ndcg5={mtr['ndcg5']:.5f} beats_v4={beats_v4(mtr['ndcg5'])}")

    # ===== Greedy forward from B3 ============================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Greedy forward selection from B3…")
    greedy_members, greedy_path = greedy_forward(rank_cache, pool_ids, starts, sizes, rel, start="B3")
    print(f"  greedy path ({len(greedy_members)} members):")
    for i, (m, n, _) in enumerate(greedy_path):
        print(f"    step{i:2d} add {m:5s} → ndcg5={n:.5f}")
    g_final = eval_ensemble(rank_cache, greedy_members, None, rel, starts, sizes)
    all_results.append({
        "ensemble_id": "GREEDY_FWD_FROM_B3",
        "method": "greedy_forward",
        "n_members": len(greedy_members),
        "members": ",".join(greedy_members), "weights": "",
        "note": f"Greedy forward selection from B3 over pool ({len(pool_ids)} candidates)",
        "beats_v4": beats_v4(g_final["ndcg5"]),
        **g_final,
    })
    json.dump({"members": greedy_members, "path": [(m, n) for m, n, _ in greedy_path],
               "metrics": g_final},
              open(out_dir / "ensemble_config_GREEDY_FWD_FROM_B3.json", "w"), indent=2)

    # ===== Leave-one-out from top10 ==========================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Leave-one-out on E3_top10…")
    base_loo, loo_results = leave_one_out(rank_cache, top10, starts, sizes, rel)
    print(f"  base E3_top10: ndcg5={base_loo['ndcg5']:.5f}")
    print("  drop → ndcg5_without (delta vs base):")
    for r in loo_results:
        sign = "+" if r["delta_vs_base"] >= 0 else ""
        print(f"    drop {r['dropped']:4s}: {r['ndcg5_without']:.5f}  ({sign}{r['delta_vs_base']:+.5f})")
        all_results.append({
            "ensemble_id": f"LOO_top10_drop_{r['dropped']}",
            "method": "loo_top10",
            "n_members": 9,
            "members": ",".join([m for m in top10 if m != r["dropped"]]),
            "weights": "", "note": f"Top10 minus {r['dropped']} (delta={r['delta_vs_base']:+.5f})",
            "beats_v4": beats_v4(r["ndcg5_without"]),
            "ndcg5": r["ndcg5_without"], "recall1": np.nan, "recall5": r["recall5_without"],
            "mean_booked_rank": r["mbr_without"], "n_queries": base_loo["n_queries"], "n_booked": base_loo["n_booked"],
        })
    pd.DataFrame(loo_results).to_csv(out_dir / "loo_top10.csv", index=False)

    # ===== W1 random subsets =================================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] W1 random subsets (n={RANDOM_SEARCH_N})…")
    w1 = random_subsets(rank_cache, pool_ids, starts, sizes, rel, n_samples=RANDOM_SEARCH_N, seed=RNG_SEED)
    w1_top = w1.head(TOP_K_REPORT).copy()
    w1_top.insert(0, "ensemble_id", [f"W1_random_subset_{i:03d}" for i in range(len(w1_top))])
    print(f"  W1 best ndcg5: {w1['ndcg5'].max():.5f} (k={w1.iloc[0]['k']}, members={w1.iloc[0]['members']})")
    for _, r in w1_top.iterrows():
        all_results.append({
            "ensemble_id": r["ensemble_id"], "method": "WILDCARD_random_subset",
            "n_members": int(r["k"]), "members": r["members"], "weights": "",
            "note": f"WILDCARD W1 trial {int(r['trial'])}, k={int(r['k'])}",
            "beats_v4": beats_v4(r["ndcg5"]),
            "ndcg5": float(r["ndcg5"]), "recall1": float(r["recall1"]),
            "recall5": float(r["recall5"]), "mean_booked_rank": float(r["mean_booked_rank"]),
            "n_queries": int(r["n_queries"]), "n_booked": int(r["n_booked"]),
        })

    # ===== W2 random weights on top10 ========================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] W2 random Dirichlet weights on top10 (n={RANDOM_SEARCH_N})…")
    w2 = random_weights(rank_cache, top10, starts, sizes, rel, n_samples=RANDOM_SEARCH_N, seed=RNG_SEED + 1, alpha=1.0)
    w2_top = w2.head(TOP_K_REPORT).copy()
    w2_top.insert(0, "ensemble_id", [f"W2_random_weights_{i:03d}" for i in range(len(w2_top))])
    print(f"  W2 best ndcg5: {w2['ndcg5'].max():.5f}")
    for _, r in w2_top.iterrows():
        all_results.append({
            "ensemble_id": r["ensemble_id"], "method": "WILDCARD_random_weights",
            "n_members": 10, "members": ",".join(top10), "weights": r["weights"],
            "note": f"WILDCARD W2 Dirichlet alpha=1, trial {int(r['trial'])}",
            "beats_v4": beats_v4(r["ndcg5"]),
            "ndcg5": float(r["ndcg5"]), "recall1": float(r["recall1"]),
            "recall5": float(r["recall5"]), "mean_booked_rank": float(r["mean_booked_rank"]),
            "n_queries": int(r["n_queries"]), "n_booked": int(r["n_booked"]),
        })

    # ===== W3 random subset + weights ========================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] W3 random subsets + random weights (n={RANDOM_SEARCH_N})…")
    w3 = random_pool_and_weights(rank_cache, pool_ids, starts, sizes, rel, n_samples=RANDOM_SEARCH_N, seed=RNG_SEED + 2)
    w3_top = w3.head(TOP_K_REPORT).copy()
    w3_top.insert(0, "ensemble_id", [f"W3_random_pool_weights_{i:03d}" for i in range(len(w3_top))])
    print(f"  W3 best ndcg5: {w3['ndcg5'].max():.5f}")
    for _, r in w3_top.iterrows():
        all_results.append({
            "ensemble_id": r["ensemble_id"], "method": "WILDCARD_random_pool_and_weights",
            "n_members": int(r["k"]), "members": r["members"], "weights": r["weights"],
            "note": f"WILDCARD W3 trial {int(r['trial'])}",
            "beats_v4": beats_v4(r["ndcg5"]),
            "ndcg5": float(r["ndcg5"]), "recall1": float(r["recall1"]),
            "recall5": float(r["recall5"]), "mean_booked_rank": float(r["mean_booked_rank"]),
            "n_queries": int(r["n_queries"]), "n_booked": int(r["n_booked"]),
        })

    # Save the combined random_search_top20.csv
    rand_all = pd.concat([
        w1_top.assign(method="W1_random_subsets"),
        w2_top.assign(method="W2_random_weights"),
        w3_top.assign(method="W3_random_pool_and_weights"),
    ], ignore_index=True)
    rand_all.to_csv(out_dir / "random_search_top20.csv", index=False)
    w1.to_csv(out_dir / "W1_random_subsets_all.csv", index=False)
    w2.to_csv(out_dir / "W2_random_weights_all.csv", index=False)
    w3.to_csv(out_dir / "W3_random_pool_and_weights_all.csv", index=False)

    # ===== Persist master ensemble_results.csv ===============================
    df_results = pd.DataFrame(all_results)
    df_results = df_results[[
        "ensemble_id", "method", "n_members", "members", "weights",
        "ndcg5", "recall1", "recall5", "mean_booked_rank", "beats_v4",
        "n_queries", "n_booked", "note",
    ]]
    # Append to experiment_logs/ensemble_results.csv (preserves prior V4 rows)
    master_path = ROOT / "experiment_logs" / "ensemble_results.csv"
    if master_path.exists():
        prior = pd.read_csv(master_path)
        # Align columns and concat
        combined = pd.concat([prior, df_results], ignore_index=True)
    else:
        combined = df_results
    combined.to_csv(master_path, index=False)
    df_results.to_csv(out_dir / "ensemble_results_this_run.csv", index=False)
    print(f"\n  → wrote {len(df_results)} new rows to {master_path}")

    # ===== Final rankings ====================================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === FINAL RANKING (top 20 by NDCG@5) ===")
    rank_view = df_results.sort_values("ndcg5", ascending=False).head(20)
    print(rank_view[["ensemble_id", "method", "n_members", "ndcg5", "recall5", "mean_booked_rank", "beats_v4"]]
          .to_string(index=False))

    best_det = df_results[~df_results["method"].str.startswith("WILDCARD")].sort_values("ndcg5", ascending=False).iloc[0]
    best_wc = df_results[df_results["method"].str.startswith("WILDCARD")].sort_values("ndcg5", ascending=False).iloc[0]
    best_overall = df_results.sort_values("ndcg5", ascending=False).iloc[0]
    print(f"\nBest DETERMINISTIC: {best_det['ensemble_id']} ndcg5={best_det['ndcg5']:.5f} (V4_local={V4_LOCAL:.5f}, Δ={best_det['ndcg5']-V4_LOCAL:+.5f})")
    print(f"Best WILDCARD     : {best_wc['ensemble_id']} ndcg5={best_wc['ndcg5']:.5f} (Δ={best_wc['ndcg5']-V4_LOCAL:+.5f})")
    print(f"Best OVERALL      : {best_overall['ensemble_id']} ndcg5={best_overall['ndcg5']:.5f} (Δ={best_overall['ndcg5']-V4_LOCAL:+.5f})")

    summary = {
        "out_dir": str(out_dir),
        "v4_local": V4_LOCAL,
        "v4_kaggle": V4_KAGGLE,
        "anchor": ANCHOR,
        "n_candidate_pool": len(pool_ids),
        "best_deterministic": {
            "ensemble_id": best_det["ensemble_id"],
            "ndcg5": float(best_det["ndcg5"]),
            "members": best_det["members"],
            "beats_v4": str(best_det["beats_v4"]),
        },
        "best_wildcard": {
            "ensemble_id": best_wc["ensemble_id"],
            "ndcg5": float(best_wc["ndcg5"]),
            "members": best_wc["members"],
            "weights": best_wc["weights"],
            "beats_v4": str(best_wc["beats_v4"]),
        },
        "best_overall": {
            "ensemble_id": best_overall["ensemble_id"],
            "ndcg5": float(best_overall["ndcg5"]),
            "members": best_overall["members"],
            "weights": best_overall["weights"],
            "beats_v4": str(best_overall["beats_v4"]),
        },
        "elapsed_seconds": time.time() - t0,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Out dir: {out_dir}")


if __name__ == "__main__":
    main()
