"""Failure-code detection metrics: macro-F1, micro-F1, per-code precision/recall."""

from __future__ import annotations
from typing import Dict, List


def _extract_codes(record: dict, verified_only: bool = False) -> set:
    codes = set()
    for f in record.get("top_failures", []):
        if isinstance(f, dict):
            code = f.get("code")
            if code and (not verified_only or f.get("verified") is not False):
                codes.add(code)
    for c in record.get("failure_codes", []):
        codes.add(c)
    return codes


def compute_failure_code_f1(
    preds: List[dict],
    labels: List[dict],
) -> Dict:
    """Compute macro-F1 and micro-F1 for multi-label failure-code detection.

    Args:
        preds: List of prediction records, each with an 'id' and 'top_failures'.
        labels: List of label records, each with an 'id' and 'top_failures'.

    Returns:
        Dict with 'macro_f1', 'micro_f1', and 'per_code' breakdown.
    """
    label_map = {r["id"]: _extract_codes(r, verified_only=True) for r in labels}
    pred_map = {r["id"]: _extract_codes(r) for r in preds}

    all_codes: set = set()
    for codes in label_map.values():
        all_codes |= codes

    per_code: Dict[str, Dict] = {}
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
        per_code[code] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    macro_f1 = sum(v["f1"] for v in per_code.values()) / len(per_code) if per_code else 0.0

    total_tp = sum(v["tp"] for v in per_code.values())
    total_fp = sum(v["fp"] for v in per_code.values())
    total_fn = sum(v["fn"] for v in per_code.values())
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_code": per_code,
    }
