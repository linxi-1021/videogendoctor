"""Compute inter-annotator agreement for second-pass annotations."""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _normalized_positive_codes(record: dict[str, Any]) -> set[str]:
    positives: set[str] = set()

    for code in record.get("failure_codes", []):
        code_str = str(code or "").strip()
        if code_str:
            positives.add(code_str)

    for failure in record.get("top_failures", []):
        code = str(failure.get("code") or "").strip()
        if not code:
            continue
        verified = failure.get("verified")
        if verified is False:
            continue
        if verified in (True, None) or "verified" not in failure:
            positives.add(code)

    return positives


def _all_mentioned_codes(record: dict[str, Any]) -> set[str]:
    codes = set(_normalized_positive_codes(record))
    for failure in record.get("top_failures", []):
        code = str(failure.get("code") or "").strip()
        if code:
            codes.add(code)
    return codes


def _span_iou(span_a: dict[str, Any], span_b: dict[str, Any]) -> float:
    a0 = float(span_a.get("t0", 0.0))
    a1 = float(span_a.get("t1", 0.0))
    b0 = float(span_b.get("t0", 0.0))
    b1 = float(span_b.get("t1", 0.0))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def _kappa_from_binary_labels(labels_a: list[int], labels_b: list[int]) -> dict[str, float]:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have the same length")

    n = len(labels_a)
    if n == 0:
        return {"kappa": 0.0, "observed_agreement": 0.0, "expected_agreement": 0.0}

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p0 = agree / n

    a_yes = sum(labels_a) / n
    a_no = 1.0 - a_yes
    b_yes = sum(labels_b) / n
    b_no = 1.0 - b_yes
    pe = a_yes * b_yes + a_no * b_no

    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if abs(p0 - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p0 - pe) / (1.0 - pe)

    return {
        "kappa": round(kappa, 4),
        "observed_agreement": round(p0, 4),
        "expected_agreement": round(pe, 4),
    }


def compute_iaa(
    primary_path: pathlib.Path,
    secondary_path: pathlib.Path,
    out_path: pathlib.Path,
    manifest_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    primary_records = {str(r.get("id", "")): r for r in _read_jsonl(primary_path)}
    secondary_records = {str(r.get("id", "")): r for r in _read_jsonl(secondary_path)}

    if manifest_path and manifest_path.exists():
        sample_ids = [str(r.get("id", "")) for r in _read_jsonl(manifest_path)]
    else:
        sample_ids = sorted(set(primary_records) & set(secondary_records))

    labels_a: list[int] = []
    labels_b: list[int] = []
    per_code_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    span_ious: list[float] = []
    missing_primary: list[str] = []
    missing_secondary: list[str] = []
    per_sample: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        primary = primary_records.get(sample_id)
        secondary = secondary_records.get(sample_id)
        if primary is None:
            missing_primary.append(sample_id)
            continue
        if secondary is None:
            missing_secondary.append(sample_id)
            continue

        primary_positive = _normalized_positive_codes(primary)
        secondary_positive = _normalized_positive_codes(secondary)
        candidate_codes = (
            _all_mentioned_codes(primary)
            | _all_mentioned_codes(secondary)
        )

        if not candidate_codes:
            per_sample.append(
                {
                    "id": sample_id,
                    "primary_positive": [],
                    "secondary_positive": [],
                    "exact_match": True,
                }
            )
            continue

        for code in sorted(candidate_codes):
            a = 1 if code in primary_positive else 0
            b = 1 if code in secondary_positive else 0
            labels_a.append(a)
            labels_b.append(b)
            per_code_pairs[code].append((a, b))

        primary_spans = {
            str(f.get("code")): f.get("evidence", {})
            for f in primary.get("top_failures", [])
            if str(f.get("code") or "").strip() in primary_positive
        }
        secondary_spans = {
            str(f.get("code")): f.get("evidence", {})
            for f in secondary.get("top_failures", [])
            if str(f.get("code") or "").strip() in secondary_positive
        }
        shared_codes = sorted(primary_positive & secondary_positive)
        for code in shared_codes:
            if code in primary_spans and code in secondary_spans:
                span_ious.append(_span_iou(primary_spans[code], secondary_spans[code]))

        per_sample.append(
            {
                "id": sample_id,
                "primary_positive": sorted(primary_positive),
                "secondary_positive": sorted(secondary_positive),
                "exact_match": primary_positive == secondary_positive,
                "shared_positive_codes": shared_codes,
            }
        )

    overall = _kappa_from_binary_labels(labels_a, labels_b)
    per_code = {}
    for code, pairs in sorted(per_code_pairs.items()):
        code_a = [a for a, _ in pairs]
        code_b = [b for _, b in pairs]
        stats = _kappa_from_binary_labels(code_a, code_b)
        stats["n"] = len(pairs)
        stats["support_primary"] = sum(code_a)
        stats["support_secondary"] = sum(code_b)
        per_code[code] = stats

    exact_match_rate = 0.0
    if per_sample:
        exact_match_rate = sum(1 for row in per_sample if row["exact_match"]) / len(per_sample)

    mean_span_iou = round(sum(span_ious) / len(span_ious), 4) if span_ious else 0.0

    result = {
        "n_samples_requested": len(sample_ids),
        "n_samples_compared": len(per_sample),
        "missing_primary_ids": missing_primary,
        "missing_secondary_ids": missing_secondary,
        "decision_pairs": len(labels_a),
        "cohen_kappa": overall["kappa"],
        "observed_agreement": overall["observed_agreement"],
        "expected_agreement": overall["expected_agreement"],
        "exact_match_rate": round(exact_match_rate, 4),
        "mean_span_iou_on_shared_positives": mean_span_iou,
        "per_code": per_code,
        "per_sample": per_sample,
    }

    _write_json(out_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for double-annotated records.")
    parser.add_argument("--primary", required=True, help="Primary annotations JSONL")
    parser.add_argument("--secondary", required=True, help="Second-pass annotations JSONL")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--manifest", default=None, help="QC subset manifest JSONL (recommended)")
    args = parser.parse_args()

    result = compute_iaa(
        primary_path=pathlib.Path(args.primary),
        secondary_path=pathlib.Path(args.secondary),
        out_path=pathlib.Path(args.out),
        manifest_path=pathlib.Path(args.manifest) if args.manifest else None,
    )
    print(
        "Cohen's kappa={kappa:.4f}  observed={obs:.4f}  exact_match={exact:.4f}".format(
            kappa=result["cohen_kappa"],
            obs=result["observed_agreement"],
            exact=result["exact_match_rate"],
        )
    )
    print(f"IAA report written to {args.out}")


if __name__ == "__main__":
    main()
