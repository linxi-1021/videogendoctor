"""Build cleaned evaluation files from human annotations.

This script does not fabricate labels. It only normalizes existing annotations:
- keep only human-verified positive failures in `top_failures`
- derive `failure_codes` from those positive failures
- preserve negative samples as records with empty `failure_codes`
- optionally rewrite the manifest with human-verified `failure_codes`
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_positive(failure: dict[str, Any]) -> bool:
    return failure.get("verified") is not False


def _normalize_positive_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_code: dict[str, dict[str, Any]] = {}
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        code = str(failure.get("code") or "").strip()
        if not code or not _is_positive(failure):
            continue
        normalized = {
            "code": code,
            "confidence": float(failure.get("confidence", 1.0)),
            "verified": True,
            "evidence": failure.get("evidence", {}) or {},
        }
        current = best_by_code.get(code)
        if current is None or normalized["confidence"] >= current["confidence"]:
            best_by_code[code] = normalized
    return sorted(best_by_code.values(), key=lambda item: item["code"])


def build_clean_annotations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for record in records:
        positive_failures = _normalize_positive_failures(record.get("top_failures", []))
        cleaned.append(
            {
                "id": record["id"],
                "annotator_id": record.get("annotator_id", ""),
                "failure_codes": [failure["code"] for failure in positive_failures],
                "top_failures": positive_failures,
                "notes": record.get("notes", ""),
            }
        )
    return cleaned


def build_clean_manifest(
    manifest_records: list[dict[str, Any]],
    cleaned_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ann_by_id = {record["id"]: record for record in cleaned_annotations}
    cleaned_manifest: list[dict[str, Any]] = []
    for record in manifest_records:
        updated = dict(record)
        ann = ann_by_id.get(record["id"])
        if ann is not None:
            updated["failure_codes"] = ann.get("failure_codes", [])
            meta = dict(updated.get("meta", {}) or {})
            meta["source_failure_codes_auto"] = record.get("failure_codes", [])
            meta["human_verified_failure_codes"] = ann.get("failure_codes", [])
            updated["meta"] = meta
        cleaned_manifest.append(updated)
    return cleaned_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned evaluation labels from human annotations.")
    parser.add_argument("--annotations", required=True, help="Raw annotations.jsonl")
    parser.add_argument("--out-annotations", required=True, help="Cleaned annotations JSONL")
    parser.add_argument("--manifest", default=None, help="Optional source manifest JSONL")
    parser.add_argument("--out-manifest", default=None, help="Optional cleaned manifest JSONL")
    args = parser.parse_args()

    annotations = load_jsonl(pathlib.Path(args.annotations))
    cleaned_annotations = build_clean_annotations(annotations)
    write_jsonl(pathlib.Path(args.out_annotations), cleaned_annotations)

    if args.manifest and args.out_manifest:
        manifest = load_jsonl(pathlib.Path(args.manifest))
        cleaned_manifest = build_clean_manifest(manifest, cleaned_annotations)
        write_jsonl(pathlib.Path(args.out_manifest), cleaned_manifest)

    positives = sum(1 for record in cleaned_annotations if record["failure_codes"])
    print(f"Cleaned annotations: {len(cleaned_annotations)} records, {positives} positive samples")


if __name__ == "__main__":
    main()
