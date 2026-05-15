"""Promote per-run artifact results into the master experiments/ trackers.

Idempotent: existing rows are skipped (keyed by model_id / exp_id).

Usage:
    uv run --no-sync python scripts/aggregate_results.py \
        --run-id phase2_labelgain --phase P2 --change-summary "Label-gain sweep around V4 bal15"
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
ART_ROOT = ROOT / "artifacts"

V4_LOCAL = 0.42512  # V4 ensemble local NDCG@5 (baseline reference)
V4_KAGGLE = 0.42021


def load_run(run_id):
    art_dir = ART_ROOT / run_id
    res_csv = art_dir / "model_results.csv"
    cfg_json = art_dir / "run_config.json"
    if not res_csv.exists():
        raise SystemExit(f"Missing {res_csv}. Has the run finished?")
    if not cfg_json.exists():
        raise SystemExit(f"Missing {cfg_json}.")
    with open(cfg_json) as f:
        cfg = json.load(f)
    return pd.read_csv(res_csv), cfg, art_dir


def infer_ipw_mode(cfg):
    return "ipw_clipped"  # Phase 2 reuses V4 default; later phases override.


def append_unique(csv_path, new_rows, key_col):
    new_df = pd.DataFrame(new_rows)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        keep_mask = ~new_df[key_col].isin(existing[key_col])
        if not keep_mask.any():
            print(f"  {csv_path.name}: all {len(new_df)} rows already present, nothing to add.")
            return 0
        new_df = new_df[keep_mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    print(f"  {csv_path.name}: added {len(new_df)} row(s).")
    return len(new_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run directory under artifacts/")
    ap.add_argument("--phase", required=True, help="Phase label, e.g. P2")
    ap.add_argument("--change-summary", default="", help="Free-text for experiment_tracker.notes")
    ap.add_argument("--exp-prefix", default=None, help="Override exp_id prefix (default = phase)")
    args = ap.parse_args()

    df, cfg, art_dir = load_run(args.run_id)
    exp_prefix = args.exp_prefix or args.phase

    print(f"Run {args.run_id}: {len(df)} model rows, phase={args.phase}")

    n_features = int(df["n_features"].max()) if "n_features" in df.columns else None
    ipw_mode = infer_ipw_mode(cfg)
    date = cfg.get("date", datetime.utcnow().isoformat() + "Z")
    num_boost_round = cfg.get("num_boost_round")

    # --- experiments/model_results.csv ---
    model_rows = []
    for _, r in df.iterrows():
        cname = r["config_name"]
        model_rows.append({
            "model_id": f"{args.run_id}_{cname}",
            "exp_id": f"{exp_prefix}_{cname}",
            "date": date,
            "model_type": "lightgbm",
            "objective": r.get("objective"),
            "boosting": r.get("boosting"),
            "label_gain": r.get("label_gain"),
            "ipw_mode": ipw_mode,
            "n_features": int(r.get("n_features")) if pd.notna(r.get("n_features")) else None,
            "seed": int(r.get("seed")) if pd.notna(r.get("seed")) else None,
            "num_boost_round": num_boost_round,
            "best_iter": int(r.get("best_iter")) if pd.notna(r.get("best_iter")) else None,
            "local_ndcg5": float(r.get("ndcg5")) if pd.notna(r.get("ndcg5")) else None,
            "recall_at_5": float(r.get("recall5")) if pd.notna(r.get("recall5")) else None,
            "mean_booked_rank": float(r.get("mean_booked_rank")) if pd.notna(r.get("mean_booked_rank")) else None,
            "artifact_path": f"artifacts/{args.run_id}/model_result_{cname}.json",
            "notes": args.change_summary,
        })

    # --- experiments/experiment_tracker.csv ---
    exp_rows = []
    for _, r in df.iterrows():
        cname = r["config_name"]
        ndcg = float(r["ndcg5"]) if pd.notna(r.get("ndcg5")) else None
        exp_rows.append({
            "exp_id": f"{exp_prefix}_{cname}",
            "phase": args.phase,
            "name": f"{args.run_id}/{cname}",
            "date": date,
            "change_summary": args.change_summary or f"single-model, label_gain={r.get('label_gain')}",
            "baseline_ref": "V4_BEST_SINGLE",
            "local_ndcg5": ndcg,
            "kaggle_ndcg5": "",
            "delta_local_vs_v4": (ndcg - V4_LOCAL) if ndcg is not None else None,
            "delta_kaggle_vs_v4": "",
            "kept": "TBD",
            "notes": f"artifact_path=artifacts/{args.run_id}/model_result_{cname}.json",
        })

    print(f"\nWriting to {EXP_DIR}/")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    append_unique(EXP_DIR / "model_results.csv", model_rows, key_col="model_id")
    append_unique(EXP_DIR / "experiment_tracker.csv", exp_rows, key_col="exp_id")
    print("\nDone.")


if __name__ == "__main__":
    main()
