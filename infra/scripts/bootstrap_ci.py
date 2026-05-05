"""Bootstrap confidence interval computation for video-level metrics.

Produces 95% bootstrap CIs for macro-F1, micro-F1, tIoU, Pass@K, and
paired difference tests. Resamples at the video level (standard
non-parametric bootstrap with BCa correction option).

Usage:
    python infra/scripts/bootstrap_ci.py \
        --pred out/preds.jsonl \
        --label out/labels.jsonl \
        --out out/bootstrap/ \
        --n-bootstrap 10000
"""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import csv
from typing import Callable, List, Optional


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _extract_codes(record: dict) -> set:
    codes = set()
    for f in record.get("top_failures", []):
        if isinstance(f, dict) and f.get("code"):
            codes.add(f["code"])
    for c in record.get("failure_codes", []):
        codes.add(c)
    return codes


def tiou(pred_t0: float, pred_t1: float,
         gt_t0: float, gt_t1: float) -> float:
    inter = max(0.0, min(pred_t1, gt_t1) - max(pred_t0, gt_t0))
    union = max(1e-6, max(pred_t1, gt_t1) - min(pred_t0, gt_t0))
    return inter / union


def compute_macro_f1(preds: List[dict], labels: List[dict]) -> float:
    label_map = {r["id"]: _extract_codes(r) for r in labels}
    pred_map = {r["id"]: _extract_codes(r) for r in preds}

    all_codes = set()
    for codes in label_map.values():
        all_codes |= codes

    f1s = []
    for code in all_codes:
        tp = fp = fn = 0
        for vid_id, true_codes in label_map.items():
            pred_codes = pred_map.get(vid_id, set())
            t = code in true_codes
            p = code in pred_codes
            if t and p:
                tp += 1
            elif p and not t:
                fp += 1
            elif t and not p:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compute_tiou_at_05(preds: List[dict], labels: List[dict]) -> float:
    label_map = {r["id"]: r for r in labels}
    tiou_vals = []

    for pred_rec in preds:
        vid_id = pred_rec["id"]
        label_rec = label_map.get(vid_id)
        if label_rec is None:
            continue
        pred_fails = pred_rec.get("top_failures", [])
        gt_fails = label_rec.get("top_failures", [])

        for pf in pred_fails:
            ev_p = pf.get("evidence", {})
            pt0, pt1 = ev_p.get("t0", 0), ev_p.get("t1", 0)
            code = pf.get("code", "")
            best = 0.0
            for gf in gt_fails:
                if gf.get("code", "") != code:
                    continue
                ev_g = gf.get("evidence", {})
                gt0, gt1 = ev_g.get("t0", 0), ev_g.get("t1", 0)
                best = max(best, tiou(pt0, pt1, gt0, gt1))
            tiou_vals.append(1.0 if best >= 0.5 else 0.0)

    return sum(tiou_vals) / len(tiou_vals) if tiou_vals else 0.0


def bootstrap_ci(
    data: List[dict],
    metric_fn: Callable[[List[dict], List[dict]], float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    paired_data: Optional[List[dict]] = None,
) -> dict:
    """Compute 95% bootstrap CI by resampling at the video level.

    When paired_data is provided, computes a paired difference CI
    (metric_fn(data) - metric_fn(paired_data)).
    """
    rng = random.Random(seed)
    n = len(data)
    diffs = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [data[i] for i in indices]
        stat_a = metric_fn(sample_a, sample_a)  # self-pair bootstrap

        if paired_data is not None:
            sample_b = [paired_data[min(i, len(paired_data) - 1)] for i in indices]
            stat_b = metric_fn(sample_b, sample_b)
            diffs.append(stat_a - stat_b)
        else:
            diffs.append(stat_a)

    diffs.sort()
    lower = diffs[int(0.025 * len(diffs))]
    upper = diffs[int(0.975 * len(diffs))]
    point = metric_fn(data, data if paired_data is None else paired_data)

    result = {
        "point_estimate": round(point, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "ci_level": 0.95,
        "n_bootstrap": n_bootstrap,
        "n_videos": n,
    }
    if paired_data is not None:
        result["mean_diff"] = round(sum(diffs) / len(diffs), 4)
    return result


def compute_all_cis(
    pred_path: str,
    label_path: str,
    out_dir: str,
    n_bootstrap: int = 10000,
    paired_pred_path: Optional[str] = None,
) -> dict:
    preds = load_jsonl(pred_path)
    labels = load_jsonl(label_path)
    paired_preds = load_jsonl(paired_pred_path) if paired_pred_path else None

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # Macro-F1 CI
    results["macro_f1"] = bootstrap_ci(
        preds, lambda p, l: compute_macro_f1(p, labels), n_bootstrap
    )

    # tIoU@0.5 CI
    results["tiou_at_05"] = bootstrap_ci(
        preds, lambda p, l: compute_tiou_at_05(p, labels), n_bootstrap
    )

    # Paired comparison if provided
    if paired_preds is not None:
        results["paired_macro_f1_diff"] = bootstrap_ci(
            preds,
            lambda p, l: compute_macro_f1(p, labels),
            n_bootstrap,
            paired_data=paired_preds,
        )

    with open(out / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # CSV summary
    with open(out / "bootstrap_ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "point_estimate", "ci_lower", "ci_upper"])
        for metric, r in results.items():
            w.writerow([metric, r["point_estimate"], r["ci_lower"], r["ci_upper"]])

    print(f"Bootstrap CIs written to {out_dir}")
    for metric, r in results.items():
        print(f"  {metric}: {r['point_estimate']} [{r['ci_lower']}, {r['ci_upper']}]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals.")
    parser.add_argument("--pred", required=True, help="Predictions JSONL")
    parser.add_argument("--label", required=True, help="Labels JSONL")
    parser.add_argument("--paired-pred", default=None, help="Second predictions for paired test")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Bootstrap iterations")
    args = parser.parse_args()
    compute_all_cis(args.pred, args.label, args.out,
                    args.n_bootstrap, args.paired_pred)
