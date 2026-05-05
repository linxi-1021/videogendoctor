"""Main scoring pipeline for VideoEval."""
from __future__ import annotations
import json
import pathlib
import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


def run_score(video_path: str, shotir_path: str | None,
             out_dir: str, cfg: dict) -> dict:
    """Run the full VideoEval scoring pipeline and return the report dict."""
    import os
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence").mkdir(exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)

    # Save config snapshot
    _save_metadata(out, cfg, video_path, shotir_path)

    from videoeval.features import extract_features
    from videoeval.rules import apply_rules
    from videoeval.patch.compile_patch import compile_patch
    from videoeval.report.render import render_html

    # 1. Extract features + keyframes
    logger.info("Extracting features from %s", video_path)
    seg_data = extract_features(
        video_path=video_path,
        out_dir=str(out),
        seg_len=cfg.get("seg_len", 2.0),
        stride=cfg.get("stride", 2.0),
        k_frames=cfg.get("k_frames", 6),
        use_yolo=cfg.get("use_yolo", False),
    )

    # 2. Load ShotIR spec
    shotir = None
    if shotir_path and pathlib.Path(shotir_path).exists():
        with open(shotir_path) as f:
            shotir = json.load(f)

    # 3. Stage-1 rule engine
    logger.info("Running Stage-1 rule engine")
    segments = apply_rules(seg_data, cfg=cfg, shotir=shotir)

    # 4. Optional Stage-2 VLM judge
    if cfg.get("use_judge"):
        from videoeval.judge_runner import run_judge
        segments = run_judge(segments, cfg=cfg, out_dir=str(out))

    # 5. Aggregate scores and top failures
    report = _build_report(
        video_path=video_path, shotir_path=shotir_path,
        segments=segments, cfg=cfg
    )

    # 6. Compile patch
    patch = compile_patch(report, shotir=shotir)
    report["patch_hints"] = patch

    # 7. Write outputs
    report_path = out / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    html_path = out / "report.html"
    render_html(report, str(html_path))

    logger.info("Done. Report: %s", report_path)
    return report


def _build_report(video_path, shotir_path, segments, cfg):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0

    # Collect all failures across segments, deduplicate by code
    all_failures = []
    for seg in segments:
        for fail in seg.get("failures", []):
            all_failures.append({
                "code": fail["code"],
                "confidence": fail["confidence"],
                "evidence": fail["evidence"],
            })
    # Sort by confidence, take unique top codes
    seen = set()
    top_failures = []
    for f in sorted(all_failures, key=lambda x: -x["confidence"]):
        if f["code"] not in seen:
            top_failures.append(f)
            seen.add(f["code"])

    # Aggregate scores
    clip_drifts = [s.get("features", {}).get("clip_drift", 0.0) for s in segments]
    consistency = float(1.0 - (sum(clip_drifts) / len(clip_drifts) if clip_drifts else 0.0))
    consistency = max(0.0, min(1.0, consistency))

    return {
        "video_meta": {
            "video_path": str(video_path),
            "shotir_path": str(shotir_path) if shotir_path else None,
            "duration_s": round(duration, 3),
            "fps": round(fps, 3),
            "resolution": {"width": w, "height": h},
            "num_frames": n_frames,
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "segments": segments,
        "scores": {
            "consistency": round(consistency, 4),
            "coherence": round(consistency, 4),   # placeholder
            "alignment": 0.5,                      # placeholder until shotir props checked
        },
        "top_failures": top_failures,
        "patch_hints": [],  # filled in after compile_patch
    }


def _save_metadata(out: pathlib.Path, cfg: dict, video_path, shotir_path):
    import platform
    meta = {
        "config": cfg,
        "video_path": str(video_path),
        "shotir_path": str(shotir_path),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
    }
    with open(out / "metadata" / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)

