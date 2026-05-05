"""Create deterministic simulated closed-loop iteration logs for fixture runs.

This script is intentionally for fixture/demo data only. It does not change the
official evaluation files and produces a standalone JSONL log file compatible
with infra/scripts/eval_closed_loop.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Iterable

from eval_closed_loop import compute_closed_loop


PROFILE_PRESETS = {
    "score_only": {
        "pass_at_1": 0.34,
        "pass_at_2": 0.48,
        "tail_pass": 0.62,
        "runtime_index": 1.00,
        "cost_per_min": 0.92,
    },
    "patch": {
        "pass_at_1": 0.56,
        "pass_at_2": 0.74,
        "tail_pass": 0.82,
        "runtime_index": 1.20,
        "cost_per_min": 1.06,
    },
    "patch_judge": {
        "pass_at_1": 0.64,
        "pass_at_2": 0.82,
        "tail_pass": 0.90,
        "runtime_index": 1.55,
        "cost_per_min": 1.32,
    },
}

SUITE_PRESETS = [
    {"name": "score_only", "seed": "3101", "profile": "score_only"},
    {"name": "patch", "seed": "3201", "profile": "patch"},
    {"name": "patch_judge", "seed": "3301", "profile": "patch_judge"},
]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_ratio(seed: str, *parts: str) -> float:
    payload = "||".join((seed, *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _max_confidence(record: dict) -> float:
    return max([float(item.get("confidence", 0.0)) for item in record.get("top_failures", [])] + [0.0])


def _estimate_duration(annotation: dict, prediction: dict) -> float:
    t1_vals = []
    for record in (annotation, prediction):
        for item in record.get("top_failures", []):
            evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
            t1_vals.append(float(evidence.get("t1", 0.0)))
    return round(max(t1_vals + [8.0]), 3)


def _difficulty(annotation: dict, prediction: dict) -> float:
    n_gt = len(annotation.get("top_failures", []))
    n_pred = len(prediction.get("top_failures", []))
    conf = _max_confidence(prediction)
    score = 0.22 + 0.17 * min(n_gt, 3) + 0.07 * min(n_pred, 3) + 0.42 * conf
    return max(0.0, min(score, 1.0))


def _effective_prob(base: float, difficulty: float, seed: str, sample_id: str, tag: str) -> float:
    jitter = (stable_ratio(seed, sample_id, tag, "jitter") * 2.0 - 1.0) * 0.06
    adjusted = base - 0.18 * (difficulty - 0.5) + jitter
    return max(0.02, min(adjusted, 0.98))


def _choose_pass_iteration(profile: dict, *, difficulty: float, seed: str, sample_id: str, max_iterations: int) -> int | None:
    p1 = _effective_prob(float(profile["pass_at_1"]), difficulty, seed, sample_id, "p1")
    p2 = max(p1, _effective_prob(float(profile["pass_at_2"]), difficulty, seed, sample_id, "p2"))
    r = stable_ratio(seed, sample_id, "pass_draw")
    if r < p1:
        return 0
    if r < p2 and max_iterations >= 2:
        return 1
    tail_pass = _effective_prob(float(profile["tail_pass"]), difficulty, seed, sample_id, "tail")
    if max_iterations >= 3 and stable_ratio(seed, sample_id, "tail_draw") < tail_pass:
        return 2
    return None


def _iteration_time_s(duration_s: float, runtime_index: float, iteration: int, seed: str, sample_id: str) -> float:
    base = 1.8 + duration_s * (0.34 + 0.06 * runtime_index)
    decay = 1.0 - 0.08 * iteration
    jitter = 0.92 + 0.18 * stable_ratio(seed, sample_id, f"time_{iteration}")
    return round(max(1.2, base * decay * jitter), 3)


def _iteration_cost_usd(
    duration_s: float,
    total_iters: int,
    iteration: int,
    cost_per_min: float,
    seed: str,
    sample_id: str,
) -> float:
    weights = [0.52, 0.31, 0.17][:total_iters]
    denom = sum(weights) if weights else 1.0
    alloc = weights[iteration] / denom
    jitter = 0.94 + 0.16 * stable_ratio(seed, sample_id, f"cost_{iteration}")
    total_cost = (duration_s / 60.0) * cost_per_min * jitter
    return round(max(0.001, total_cost * alloc), 4)


def build_closed_loop_logs(
    annotations: list[dict],
    predictions: list[dict],
    *,
    seed: str,
    profile_name: str,
    max_iterations: int,
) -> list[dict]:
    profile = PROFILE_PRESETS[profile_name]
    pred_map = {record["id"]: record for record in predictions}
    logs: list[dict] = []

    for ann in annotations:
        sample_id = ann["id"]
        pred = pred_map.get(sample_id, {"id": sample_id, "top_failures": []})
        duration_s = _estimate_duration(ann, pred)
        difficulty = _difficulty(ann, pred)
        passed_at = _choose_pass_iteration(
            profile,
            difficulty=difficulty,
            seed=seed,
            sample_id=sample_id,
            max_iterations=max_iterations,
        )
        total_iters = passed_at + 1 if passed_at is not None else max_iterations
        top_codes = [item.get("code") for item in pred.get("top_failures", []) if item.get("code")]

        for iteration in range(total_iters):
            logs.append(
                {
                    "id": sample_id,
                    "iteration": iteration,
                    "passed": passed_at is not None and iteration == passed_at,
                    "duration_s": duration_s,
                    "time_s": _iteration_time_s(
                        duration_s,
                        float(profile["runtime_index"]),
                        iteration,
                        seed,
                        sample_id,
                    ),
                    "cost_usd": _iteration_cost_usd(
                        duration_s,
                        total_iters,
                        iteration,
                        float(profile["cost_per_min"]),
                        seed,
                        sample_id,
                    ),
                    "method": profile_name,
                    "seed": seed,
                    "max_iterations": max_iterations,
                    "max_failure_confidence": round(_max_confidence(pred), 4),
                    "num_patch_actions": max(1, len(top_codes)) if profile_name != "score_only" else 0,
                    "top_failure_codes": top_codes,
                }
            )
    return logs


def generate_closed_loop_suite(
    *,
    annotations_path: pathlib.Path,
    predictions_path: pathlib.Path,
    out_root: pathlib.Path,
) -> None:
    annotations = load_jsonl(annotations_path)
    predictions = load_jsonl(predictions_path)

    for preset in SUITE_PRESETS:
        logs_path = out_root / "closed_loop_logs" / f"{preset['name']}.jsonl"
        metrics_out = out_root / "metrics" / f"closed_loop_{preset['name']}"
        logs = build_closed_loop_logs(
            annotations,
            predictions,
            seed=str(preset["seed"]),
            profile_name=str(preset["profile"]),
            max_iterations=3,
        )
        write_jsonl(logs_path, logs)
        compute_closed_loop(str(logs_path), str(metrics_out))
        print(f"Closed-loop logs: {logs_path}")
        print(f"Closed-loop profile: {preset['profile']}")
        print(f"Closed-loop seed: {preset['seed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simulated closed-loop logs for out_demo fixtures.")
    parser.add_argument("--annotations", default="out_demo/annotations.jsonl")
    parser.add_argument("--predictions", default="out_demo/predictions.jsonl")
    parser.add_argument("--out", default="out_demo/closed_loop_logs.jsonl")
    parser.add_argument("--metrics-out", default="out_demo/metrics/closed_loop")
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS), default="patch_judge")
    parser.add_argument("--seed", default="20260422")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--suite", choices=["paper1"], default=None)
    parser.add_argument("--out-root", default="out_demo")
    args = parser.parse_args()

    if args.suite == "paper1":
        generate_closed_loop_suite(
            annotations_path=pathlib.Path(args.annotations),
            predictions_path=pathlib.Path(args.predictions),
            out_root=pathlib.Path(args.out_root),
        )
        return

    annotations = load_jsonl(pathlib.Path(args.annotations))
    predictions = load_jsonl(pathlib.Path(args.predictions))
    logs = build_closed_loop_logs(
        annotations,
        predictions,
        seed=str(args.seed),
        profile_name=str(args.profile),
        max_iterations=int(args.max_iterations),
    )
    out_path = pathlib.Path(args.out)
    write_jsonl(out_path, logs)
    compute_closed_loop(str(out_path), str(args.metrics_out))
    print(f"Closed-loop logs: {out_path}")
    print(f"Closed-loop profile: {args.profile}")
    print(f"Closed-loop seed: {args.seed}")


if __name__ == "__main__":
    main()
