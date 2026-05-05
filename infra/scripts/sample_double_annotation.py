"""Create a deterministic QC subset for second-pass annotation."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
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


def _write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_lines(path: pathlib.Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _allocate_counts(group_sizes: dict[str, int], target_n: int) -> dict[str, int]:
    if not group_sizes:
        return {}

    total = sum(group_sizes.values())
    target_n = max(1, min(target_n, total))
    groups = sorted(group_sizes)
    alloc = {g: 0 for g in groups}

    if target_n >= len(groups):
        for g in groups:
            alloc[g] = 1
        remaining = target_n - len(groups)
    else:
        remaining = target_n

    if remaining <= 0:
        return alloc

    quotas: list[tuple[float, str]] = []
    for g in groups:
        available = group_sizes[g] - alloc[g]
        if available <= 0:
            continue
        raw = remaining * (available / max(1, total - (target_n - remaining)))
        floor_raw = min(available, math.floor(raw))
        alloc[g] += floor_raw
        quotas.append((raw - floor_raw, g))

    used = sum(alloc.values())
    leftover = target_n - used
    if leftover > 0:
        for _, g in sorted(quotas, key=lambda item: (-item[0], item[1])):
            if leftover <= 0:
                break
            if alloc[g] >= group_sizes[g]:
                continue
            alloc[g] += 1
            leftover -= 1

    if leftover > 0:
        for g in groups:
            while leftover > 0 and alloc[g] < group_sizes[g]:
                alloc[g] += 1
                leftover -= 1

    return alloc


def sample_manifest_records(
    records: list[dict[str, Any]],
    sample_n: int,
    seed: int,
    stratify_by: str = "perturbation_type",
) -> list[dict[str, Any]]:
    if not records:
        return []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get(stratify_by) or "unknown")
        groups[key].append(record)

    rng = random.Random(seed)
    group_sizes = {k: len(v) for k, v in groups.items()}
    alloc = _allocate_counts(group_sizes, sample_n)

    sampled: list[dict[str, Any]] = []
    for key in sorted(groups):
        pool = sorted(groups[key], key=lambda item: str(item.get("id", "")))
        take_n = min(len(pool), alloc.get(key, 0))
        if take_n <= 0:
            continue
        indices = sorted(rng.sample(range(len(pool)), take_n))
        sampled.extend(pool[idx] for idx in indices)

    sampled.sort(key=lambda item: str(item.get("id", "")))
    return sampled


def create_qc_subset(
    manifest_path: pathlib.Path,
    out_manifest: pathlib.Path,
    out_ids: pathlib.Path,
    out_primary_subset: pathlib.Path | None,
    annotations_path: pathlib.Path | None,
    ratio: float,
    min_samples: int,
    seed: int,
    stratify_by: str,
) -> list[dict[str, Any]]:
    records = _read_jsonl(manifest_path)
    target_n = max(min_samples, math.ceil(len(records) * ratio))
    sampled = sample_manifest_records(
        records=records,
        sample_n=target_n,
        seed=seed,
        stratify_by=stratify_by,
    )

    _write_jsonl(out_manifest, sampled)
    _write_lines(out_ids, [str(record.get("id", "")) for record in sampled])

    if out_primary_subset and annotations_path and annotations_path.exists():
        sample_ids = {str(record.get("id", "")) for record in sampled}
        primary_subset = [
            record for record in _read_jsonl(annotations_path)
            if str(record.get("id", "")) in sample_ids
        ]
        _write_jsonl(out_primary_subset, primary_subset)

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a deterministic subset for second-pass annotation QC.")
    parser.add_argument("--manifest", required=True, help="Full dataset manifest JSONL")
    parser.add_argument("--out-manifest", required=True, help="Subset manifest JSONL for QC annotation")
    parser.add_argument("--out-ids", required=True, help="Text file with sampled IDs")
    parser.add_argument("--annotations", default=None, help="Primary annotations JSONL (optional)")
    parser.add_argument("--out-primary-subset", default=None, help="Filtered primary annotations for sampled IDs")
    parser.add_argument("--ratio", type=float, default=0.1, help="Sampling ratio, default 0.1")
    parser.add_argument("--min-samples", type=int, default=8, help="Minimum number of sampled videos")
    parser.add_argument("--seed", type=int, default=20260418, help="Random seed")
    parser.add_argument(
        "--stratify-by",
        default="perturbation_type",
        help="Manifest field used for stratified sampling, default perturbation_type",
    )
    args = parser.parse_args()

    sampled = create_qc_subset(
        manifest_path=pathlib.Path(args.manifest),
        out_manifest=pathlib.Path(args.out_manifest),
        out_ids=pathlib.Path(args.out_ids),
        out_primary_subset=pathlib.Path(args.out_primary_subset) if args.out_primary_subset else None,
        annotations_path=pathlib.Path(args.annotations) if args.annotations else None,
        ratio=args.ratio,
        min_samples=args.min_samples,
        seed=args.seed,
        stratify_by=args.stratify_by,
    )

    print(f"Sampled {len(sampled)} records")
    for record in sampled:
        print(record.get("id", ""))


if __name__ == "__main__":
    main()
