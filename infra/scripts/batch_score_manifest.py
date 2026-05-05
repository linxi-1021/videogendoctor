"""Batch score videos listed in a manifest JSONL.

The repo's CLI currently scores one video at a time. For A2/A3/B phases
we often want to run scoring over many videos deterministically.

Input manifest format (JSONL, one object per line):
  {
    "id": "real_cog_001",
    "video_path": "data/source/cogvideox/real_cog_001.mp4",
    "shotir_path": "data/source/cogvideox/real_cog_001_shotir.json",   # optional
    "generator": "cogvideox"                                           # optional
  }

Example:
  python infra/scripts/batch_score_manifest.py \
    --manifest data/source_manifest.jsonl \
    --out-root out/a2_scores

Outputs:
- For each record: <out-root>/<id>/report.json and report.html etc.
- Aggregated: <out-root>/predictions.jsonl
- Errors (if any): <out-root>/errors.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "packages" / "videoeval"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from videoeval.config import load_runtime_config  # type: ignore[import-not-found]
from videoeval.pipeline import run_score  # type: ignore[import-not-found]


def _repo_root() -> pathlib.Path:
    return REPO_ROOT


def _load_config(config_path: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    judge_config = None
    if config_path:
        judge_candidate = pathlib.Path(config_path).resolve().parent / "judge.yaml"
        judge_config = str(judge_candidate)
    return load_runtime_config(config_path, overrides, judge_config_path=judge_config)


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
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def batch_score(manifest_path: pathlib.Path, out_root: pathlib.Path, cfg: dict[str, Any], fail_fast: bool) -> None:
    root = _repo_root()
    records = _read_jsonl(manifest_path)

    out_root.mkdir(parents=True, exist_ok=True)

    preds: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, r in enumerate(records, start=1):
        vid_id = r.get("id")
        if not vid_id:
            msg = "Missing 'id' in manifest record"
            if fail_fast:
                raise ValueError(msg)
            errors.append({"index": idx, "error": msg, "record": r})
            continue

        video_rel = r.get("video_path")
        if not video_rel:
            msg = f"[{vid_id}] Missing 'video_path'"
            if fail_fast:
                raise ValueError(msg)
            errors.append({"id": vid_id, "error": msg, "record": r})
            continue

        video_path = (root / video_rel).resolve()
        shotir_rel = r.get("shotir_path")
        shotir_path = (root / shotir_rel).resolve() if shotir_rel else None

        one_out = out_root / str(vid_id)
        try:
            report = run_score(
                video_path=str(video_path),
                shotir_path=str(shotir_path) if shotir_path else None,
                out_dir=str(one_out),
                cfg=cfg,
            )
        except Exception as e:  # noqa: BLE001
            err = {"id": vid_id, "video_path": video_rel, "error": repr(e)}
            errors.append(err)
            print(f"[{idx}/{len(records)}] {vid_id}: ERROR {repr(e)}")
            if fail_fast:
                raise
            continue

        preds.append(
            {
                "id": vid_id,
                "generator": r.get("generator"),
                "video_path": video_rel,
                "shotir_path": shotir_rel,
                "top_failures": report.get("top_failures", []),
                "report_path": (one_out / "report.json").relative_to(root).as_posix(),
            }
        )
        print(f"[{idx}/{len(records)}] {vid_id}: ok")

    _write_jsonl(out_root / "predictions.jsonl", preds)
    if errors:
        _write_jsonl(out_root / "errors.jsonl", errors)
        print(f"Done with errors: {len(errors)} (see {out_root / 'errors.jsonl'})")
    else:
        print("Done without errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch score videos from a manifest JSONL")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument("--out-root", required=True, help="Output root directory")
    parser.add_argument("--config", default=None, help="Optional videoeval.yaml config")

    parser.add_argument("--seg-len", type=float, default=None, help="Override segment length (seconds)")
    parser.add_argument("--stride", type=float, default=None, help="Override segment stride (seconds)")
    parser.add_argument("--k-frames", type=int, default=None, help="Override keyframes per segment")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLOv8 object detection")
    parser.add_argument("--use-judge", action="store_true", help="Enable Stage-2 VLM judge")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")

    args = parser.parse_args()

    overrides = {
        "seg_len": args.seg_len,
        "stride": args.stride,
        "k_frames": args.k_frames,
        "use_yolo": bool(args.use_yolo),
        "use_judge": bool(args.use_judge),
    }

    cfg = _load_config(args.config, overrides)
    batch_score(
        manifest_path=_repo_root() / args.manifest,
        out_root=_repo_root() / args.out_root,
        cfg=cfg,
        fail_fast=bool(args.fail_fast),
    )
