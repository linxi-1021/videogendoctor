"""Metric: evidence localization — tIoU and Top-K keyframe hit rate."""
from __future__ import annotations
import argparse
import json
import csv
import pathlib
from typing import List


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _is_positive_labeled_failure(item: dict) -> bool:
    """Label semantics: verified=false should not be treated as GT positive."""
    return item.get("verified") is not False


def extract_label_failures(record: dict) -> List[dict]:
    """Return only GT-positive failure entries from a label record."""
    top_failures = record.get("top_failures")
    if isinstance(top_failures, list):
        return [
            item for item in top_failures
            if isinstance(item, dict) and item.get("code") and _is_positive_labeled_failure(item)
        ]

    failures = record.get("failures", [])
    return [item for item in failures if isinstance(item, dict) and item.get("code")]


def tiou(pred_t0: float, pred_t1: float,
         gt_t0: float, gt_t1: float) -> float:
    inter_start = max(pred_t0, gt_t0)
    inter_end   = min(pred_t1, gt_t1)
    inter = max(0.0, inter_end - inter_start)
    union = max(1e-6,
                max(pred_t1, gt_t1) - min(pred_t0, gt_t0))
    return inter / union


def keyframe_hit(pred_kfs: List[str],
                 gt_kfs:   List[str], k: int) -> float:
    """Top-K hit: 1 if any of pred_kfs[:k] basename matches gt_kfs basename."""
    pred_names = {pathlib.Path(p).name for p in pred_kfs[:k]}
    gt_names   = {pathlib.Path(p).name for p in gt_kfs}
    return 1.0 if pred_names & gt_names else 0.0


def compute_localization(pred_path: str, label_path: str,
                         out_dir: str) -> dict:
    preds  = load_jsonl(pred_path)
    labels = load_jsonl(label_path)

    label_map = {r["id"]: r for r in labels}

    tiou_03_vals, tiou_05_vals = [], []
    hit1_vals, hit3_vals = [], []

    rows = []
    for pred_rec in preds:
        vid_id = pred_rec["id"]
        label_rec = label_map.get(vid_id)
        if label_rec is None:
            continue

        # Match predicted failures to labeled evidence spans
        pred_fails = pred_rec.get("top_failures",
                     [f for seg in pred_rec.get("segments", [])
                      for f in seg.get("failures", [])])
        gt_fails = extract_label_failures(label_rec)

        for pred_f in pred_fails:
            ev_p = pred_f.get("evidence", {})
            pt0, pt1 = ev_p.get("t0", 0), ev_p.get("t1", 0)
            pkfs = ev_p.get("keyframes", [])
            code = pred_f.get("code", "")

            # Find matching GT failure by code
            best_tiou = 0.0
            best_hit1 = 0.0
            best_hit3 = 0.0
            for gt_f in gt_fails:
                if gt_f.get("code", "") != code:
                    continue
                ev_g = gt_f.get("evidence", {})
                gt0, gt1 = ev_g.get("t0", 0), ev_g.get("t1", 0)
                gkfs = ev_g.get("keyframes", [])
                t = tiou(pt0, pt1, gt0, gt1)
                best_tiou = max(best_tiou, t)
                best_hit1 = max(best_hit1, keyframe_hit(pkfs, gkfs, 1))
                best_hit3 = max(best_hit3, keyframe_hit(pkfs, gkfs, 3))

            tiou_03_vals.append(1.0 if best_tiou >= 0.3 else 0.0)
            tiou_05_vals.append(1.0 if best_tiou >= 0.5 else 0.0)
            hit1_vals.append(best_hit1)
            hit3_vals.append(best_hit3)
            rows.append({"id": vid_id, "code": code,
                         "tiou": round(best_tiou, 4),
                         "hit1": best_hit1, "hit3": best_hit3})

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    result = {
        "tiou_at_03": _mean(tiou_03_vals),
        "tiou_at_05": _mean(tiou_05_vals),
        "top1_hit":   _mean(hit1_vals),
        "top3_hit":   _mean(hit3_vals),
        "n_predictions": len(rows),
    }

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evidence_localization.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "evidence_localization.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","code","tiou","hit1","hit3"])
        w.writeheader()
        w.writerows(rows)

    print(f"tIoU@0.3={result['tiou_at_03']}  tIoU@0.5={result['tiou_at_05']}  "
          f"Top-1={result['top1_hit']}  Top-3={result['top3_hit']}")
    print(f"Results written to {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate evidence localization.")
    parser.add_argument("--pred",  required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()
    compute_localization(args.pred, args.label, args.out)

