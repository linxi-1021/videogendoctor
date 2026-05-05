"""Build a source-video manifest JSONL from data/source/.

This is a lightweight utility to support the TANBO A2 phase:
- Scan for *.mp4 under data/source/<generator>/
- Emit one JSON object per line with relative, portable paths

Expected layout (recommended):
  data/source/<generator>/<video_id>.mp4
  data/source/<generator>/<video_id>_shotir.json   (optional)

Example:
  python infra/scripts/build_source_manifest.py --out data/source_manifest.jsonl

Notes:
- Paths in the manifest are written as POSIX-style relative paths (forward slashes)
  to make them portable across Windows/Linux and easier to consume by HTML.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable


def _repo_root() -> pathlib.Path:
    # .../infra/scripts/build_source_manifest.py -> repo root is parents[2]
    return pathlib.Path(__file__).resolve().parents[2]


def _iter_videos(source_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    yield from sorted(source_dir.rglob("*.mp4"))


def _rel_posix(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def build_manifest(source_dir: pathlib.Path, out_path: pathlib.Path) -> int:
    root = _repo_root()
    source_dir = source_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    records: list[dict] = []
    for mp4_path in _iter_videos(source_dir):
        rel = mp4_path.relative_to(source_dir)
        # generator is first path component under data/source/
        parts = rel.parts
        generator = parts[0] if len(parts) >= 2 else "unknown"
        video_id = mp4_path.stem

        shotir_candidate = mp4_path.with_name(f"{video_id}_shotir.json")
        shotir_rel = _rel_posix(shotir_candidate, root) if shotir_candidate.exists() else None

        records.append(
            {
                "id": video_id,
                "video_path": _rel_posix(mp4_path, root),
                "shotir_path": shotir_rel,
                "generator": generator,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build source_manifest.jsonl from data/source")
    parser.add_argument(
        "--source",
        default="data/source",
        help="Directory containing per-generator subfolders of mp4 files (default: data/source)",
    )
    parser.add_argument(
        "--out",
        default="data/source_manifest.jsonl",
        help="Output JSONL path (default: data/source_manifest.jsonl)",
    )
    args = parser.parse_args()

    root = _repo_root()
    n = build_manifest(root / args.source, root / args.out)
    print(f"Wrote {n} records to {args.out}")
