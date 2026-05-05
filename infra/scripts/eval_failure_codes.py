"""Metric: failure-code detection F1 (macro + micro + per-code)."""
from __future__ import annotations
import argparse
import json
import csv
import pathlib
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _is_positive_labeled_failure(item: dict) -> bool:
    """Label semantics: verified=false is a negative judgment; missing means legacy positive."""
    return item.get("verified") is not False


def extract_codes(record: dict, *, is_label: bool = False) -> set:
    """Extract set of failure codes from a record."""
    # Supports both pred (top_failures list) and label (failure_codes/top_failures)
    codes = set()
    for f in record.get("top_failures", []):
        if isinstance(f, dict):
            code = f.get("code")
            if not code:
                continue
            if is_label and not _is_positive_labeled_failure(f):
                continue
            codes.add(code)
        elif f:
            codes.add(f)
    for c in record.get("failure_codes", []):
        codes.add(c)
    return codes


def compute_f1(pred_path: str, label_path: str, out_dir: str) -> dict:
    preds = load_jsonl(pred_path)
    labels = load_jsonl(label_path)

    # Index by id
    label_map = {r["id"]: extract_codes(r, is_label=True) for r in labels}
    pred_map  = {r["id"]: extract_codes(r, is_label=False) for r in preds}

    all_codes = set()
    for codes in label_map.values():
        all_codes |= codes

    per_code: Dict[str, Dict] = {}
    for code in all_codes:
        tp = fp = fn = 0
        for vid_id, true_codes in label_map.items():
            pred_codes = pred_map.get(vid_id, set())
            t = code in true_codes
            p = code in pred_codes
            if t and p:  tp += 1
            elif p and not t: fp += 1
            elif t and not p: fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_code[code] = {"precision": prec, "recall": rec, "f1": f1,
                          "tp": tp, "fp": fp, "fn": fn}

    macro_f1 = sum(v["f1"] for v in per_code.values()) / len(per_code) if per_code else 0.0

    # Micro
    total_tp = sum(v["tp"] for v in per_code.values())
    total_fp = sum(v["fp"] for v in per_code.values())
    total_fn = sum(v["fn"] for v in per_code.values())
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) \
                 if (micro_prec + micro_rec) > 0 else 0.0

    result = {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_code": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                     for k, v in per_code.items()},
    }

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "failure_code_f1.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # CSV
    with open(out / "failure_code_f1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "precision", "recall", "f1", "tp", "fp", "fn"])
        for code, v in sorted(per_code.items()):
            w.writerow([code, v["precision"], v["recall"], v["f1"],
                        v["tp"], v["fp"], v["fn"]])

    print(f"Macro-F1: {macro_f1:.4f}  Micro-F1: {micro_f1:.4f}")
    print(f"Results written to {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate failure-code F1.")
    parser.add_argument("--pred",  required=True, help="Predictions JSONL")
    parser.add_argument("--label", required=True, help="Labels JSONL")
    parser.add_argument("--out",   required=True, help="Output directory")
    args = parser.parse_args()
    compute_f1(args.pred, args.label, args.out)

