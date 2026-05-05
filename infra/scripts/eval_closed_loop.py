"""Metric: closed-loop repair — Pass@1/2, avg iters, cost per usable minute."""
from __future__ import annotations
import argparse
import json
import csv
import pathlib
from typing import List


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_closed_loop(logs_path: str, out_dir: str,
                        pass_threshold: float = 0.5) -> dict:
    """Compute closed-loop metrics from iteration logs.

    Log record format (one per video per iteration):
    {
      "id": str,
      "iteration": int,          # 0-indexed
      "passed": bool,            # quality check passed
      "duration_s": float,       # video duration in seconds
      "time_s": float,           # wall time for this iteration
      "cost_usd": float          # API/compute cost for this iteration
    }
    """
    logs = load_jsonl(logs_path)

    # Group by video id
    by_id: dict = {}
    for rec in logs:
        by_id.setdefault(rec["id"], []).append(rec)

    pass1_vals, pass2_vals = [], []
    iter_vals, time_vals, cost_per_min_vals = [], [], []
    rows = []

    for vid_id, iters in by_id.items():
        iters_sorted = sorted(iters, key=lambda x: x["iteration"])
        passed_at = None
        total_time = 0.0
        total_cost = 0.0
        vid_dur = iters_sorted[0].get("duration_s", 1.0)

        for rec in iters_sorted:
            total_time += rec.get("time_s", 0.0)
            total_cost += rec.get("cost_usd", 0.0)
            if rec.get("passed") and passed_at is None:
                passed_at = rec["iteration"]

        p1 = 1.0 if (passed_at is not None and passed_at <= 0) else 0.0
        p2 = 1.0 if (passed_at is not None and passed_at <= 1) else 0.0
        n_iters = passed_at + 1 if passed_at is not None else len(iters_sorted)
        cost_per_min = (total_cost / (vid_dur / 60.0)) if vid_dur > 0 else 0.0

        pass1_vals.append(p1)
        pass2_vals.append(p2)
        iter_vals.append(n_iters)
        time_vals.append(total_time)
        if passed_at is not None:
            cost_per_min_vals.append(cost_per_min)

        rows.append({
            "id": vid_id, "pass@1": p1, "pass@2": p2,
            "iters": n_iters, "time_s": round(total_time, 2),
            "cost_per_min": round(cost_per_min, 4),
        })

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    result = {
        "pass_at_1":       _mean(pass1_vals),
        "pass_at_2":       _mean(pass2_vals),
        "avg_iterations":  _mean(iter_vals),
        "avg_time_s":      _mean(time_vals),
        "avg_cost_per_min": _mean(cost_per_min_vals),
        "n_videos":        len(by_id),
    }

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "closed_loop.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "closed_loop.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id","pass@1","pass@2","iters","time_s","cost_per_min"])
        w.writeheader()
        w.writerows(rows)

    print(f"Pass@1={result['pass_at_1']}  Pass@2={result['pass_at_2']}  "
          f"AvgIters={result['avg_iterations']}  "
          f"AvgTime={result['avg_time_s']}s  "
          f"Cost/min=${result['avg_cost_per_min']}")
    print(f"Results written to {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate closed-loop repair.")
    parser.add_argument("--logs", required=True, help="Iteration logs JSONL")
    parser.add_argument("--out",  required=True, help="Output directory")
    parser.add_argument("--pass-threshold", type=float, default=0.5)
    args = parser.parse_args()
    compute_closed_loop(args.logs, args.out, args.pass_threshold)

