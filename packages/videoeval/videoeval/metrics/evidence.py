"""Evidence localization metrics: tIoU and Top-K keyframe hit rate."""

from __future__ import annotations
import pathlib
from typing import Dict, List


def tiou(pred_t0: float, pred_t1: float,
         gt_t0: float, gt_t1: float) -> float:
    """Temporal Intersection-over-Union between two spans."""
    inter = max(0.0, min(pred_t1, gt_t1) - max(pred_t0, gt_t0))
    union = max(1e-6, max(pred_t1, gt_t1) - min(pred_t0, gt_t0))
    return inter / union


def keyframe_hit(pred_kfs: List[str], gt_kfs: List[str], k: int) -> float:
    """Top-K keyframe hit: 1 if any pred_kfs[:k] matches gt_kfs basename."""
    pred_names = {pathlib.Path(p).name for p in pred_kfs[:k]}
    gt_names = {pathlib.Path(p).name for p in gt_kfs}
    return 1.0 if pred_names & gt_names else 0.0


def compute_evidence_localization(
    preds: List[dict],
    labels: List[dict],
    tiou_thresholds: tuple = (0.3, 0.5),
) -> Dict:
    """Compute evidence localization metrics.

    Matches predicted failures to ground-truth failures by code, then computes
    tIoU and Top-K keyframe hit rate. Missing or invalid spans receive no credit.

    Args:
        preds: List of prediction records.
        labels: List of label records (verified failures only).
        tiou_thresholds: tIoU thresholds for hit-rate computation.

    Returns:
        Dict with tIoU and Top-K hit metrics.
    """
    label_map = {r["id"]: r for r in labels}

    tiou_per_threshold = {t: [] for t in tiou_thresholds}
    hit1_vals, hit3_vals = [], []
    rows = []

    for pred_rec in preds:
        vid_id = pred_rec["id"]
        label_rec = label_map.get(vid_id)
        if label_rec is None:
            continue

        pred_fails = pred_rec.get("top_failures", [])
        gt_fails = [
            f for f in label_rec.get("top_failures", [])
            if isinstance(f, dict) and f.get("code") and f.get("verified") is not False
        ]

        for pf in pred_fails:
            ev_p = pf.get("evidence", {})
            pt0, pt1 = ev_p.get("t0", 0), ev_p.get("t1", 0)
            pkfs = ev_p.get("keyframes", [])
            code = pf.get("code", "")

            best_tiou = 0.0
            best_hit1 = 0.0
            best_hit3 = 0.0
            for gf in gt_fails:
                if gf.get("code", "") != code:
                    continue
                ev_g = gf.get("evidence", {})
                gt0, gt1 = ev_g.get("t0", 0), ev_g.get("t1", 0)
                gkfs = ev_g.get("keyframes", [])
                t = tiou(pt0, pt1, gt0, gt1)
                best_tiou = max(best_tiou, t)
                best_hit1 = max(best_hit1, keyframe_hit(pkfs, gkfs, 1))
                best_hit3 = max(best_hit3, keyframe_hit(pkfs, gkfs, 3))

            for thresh in tiou_thresholds:
                tiou_per_threshold[thresh].append(1.0 if best_tiou >= thresh else 0.0)
            hit1_vals.append(best_hit1)
            hit3_vals.append(best_hit3)
            rows.append({
                "id": vid_id, "code": code,
                "tiou": round(best_tiou, 4),
                "hit1": best_hit1, "hit3": best_hit3,
            })

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    result = {}
    for thresh, vals in tiou_per_threshold.items():
        result[f"tiou_at_{int(thresh * 100):02d}"] = _mean(vals)
    result["top1_hit"] = _mean(hit1_vals)
    result["top3_hit"] = _mean(hit3_vals)
    result["n_predictions"] = len(rows)
    result["details"] = rows
    return result
