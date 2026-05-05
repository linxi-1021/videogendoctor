"""Controlled perturbation dataset generator for VideoGenDoctor-Bench-v0."""
from __future__ import annotations
import argparse
import json
import pathlib
import shutil
import random
import logging
import subprocess
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)
FFMPEG_BIN = shutil.which("ffmpeg")

# Maps perturbation type -> failure codes auto-assigned
PERTURB_CODE_MAP = {
    "remove_anchor":                ["ID_FACE_DRIFT", "ID_BODY_DRIFT"],
    "drop_props_required":          ["AL_PROP_MISSING"],
    "change_camera_movement":       ["CA_MOVE_WRONG"],
    "extend_duration_or_merge":     ["MO_SEGMENT_BREAK", "MO_EVENT_MISSING"],
    "temporal_jitter_or_frame_drop":["MO_JITTER", "MO_FRAME_DROP"],
    "compression_artifacts":        ["ST_COMPRESSION_ARTIFACT", "ST_COLOR_SHIFT"],
}


def apply_perturbation(video_path: str, perturb_type: str,
                       out_path: str, seed: int = 42) -> dict:
    """Apply a single perturbation to a video and save to out_path."""
    rng = random.Random(seed)
    np.random.seed(seed)

    if perturb_type == "temporal_jitter_or_frame_drop":
        _apply_temporal_jitter(video_path, out_path, rng)
    elif perturb_type == "compression_artifacts":
        _apply_compression(video_path, out_path, rng)
    elif perturb_type == "remove_anchor":
        _apply_remove_anchor(video_path, out_path, rng)
    elif perturb_type == "extend_duration_or_merge":
        _apply_extend_duration(video_path, out_path, rng)
    elif perturb_type == "drop_props_required":
        _apply_drop_props_required(video_path, out_path, rng)
    elif perturb_type == "change_camera_movement":
        _apply_change_camera_movement(video_path, out_path, rng)
    else:
        # Unknown perturbation: copy video as-is
        shutil.copy2(video_path, out_path)

    return {
        "perturbation_type": perturb_type,
        "failure_codes": PERTURB_CODE_MAP.get(perturb_type, []),
        "seed": seed,
    }


def _open_perturb_writer(dst: str, fps: float, width: int, height: int):
    """Write with OpenCV first, then transcode to a browser-compatible MP4 when ffmpeg exists."""
    dst_path = pathlib.Path(dst)
    raw_path = dst_path
    if FFMPEG_BIN:
        raw_path = dst_path.with_name(f"{dst_path.stem}.__opencv_tmp__.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {raw_path}")
    return writer, raw_path


def _finalize_perturb_writer(writer: cv2.VideoWriter, raw_path: pathlib.Path, dst: str):
    writer.release()

    dst_path = pathlib.Path(dst)
    if raw_path == dst_path:
        if not FFMPEG_BIN:
            logger.warning("ffmpeg not found; keeping OpenCV mp4v output: %s", dst_path)
        return

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(raw_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"ffmpeg transcode failed for {dst_path}: {msg}") from exc
    finally:
        if raw_path.exists():
            raw_path.unlink()


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _clamp_roi_xyxy(roi: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    w = max(1, int(x1 - x0))
    h = max(1, int(y1 - y0))

    # Keep ROI size fixed; only clamp position.
    x0 = _clamp_int(x0, 0, max(0, frame_w - w))
    y0 = _clamp_int(y0, 0, max(0, frame_h - h))
    x1 = x0 + w
    y1 = y0 + h

    x0 = _clamp_int(x0, 0, max(0, frame_w - 1))
    y0 = _clamp_int(y0, 0, max(0, frame_h - 1))
    x1 = _clamp_int(x1, x0 + 1, frame_w)
    y1 = _clamp_int(y1, y0 + 1, frame_h)
    return (x0, y0, x1, y1)


def _roi_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    union = max(1e-9, area_a + area_b - inter)
    return inter / union


def _roi_xyxy_to_xywh(roi: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = roi
    return (float(x0), float(y0), float(max(1, x1 - x0)), float(max(1, y1 - y0)))


def _bbox_xywh_to_roi_xyxy(
    bbox_xywh: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    *,
    min_size: int = 8,
) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox_xywh
    bw_i = max(int(min_size), int(round(float(bw))))
    bh_i = max(int(min_size), int(round(float(bh))))
    x0 = int(round(float(x)))
    y0 = int(round(float(y)))
    return _clamp_roi_xyxy((x0, y0, x0 + bw_i, y0 + bh_i), frame_w, frame_h)


def _try_create_opencv_tracker(name: str):
    """Create a classic OpenCV tracker if available (requires opencv-contrib build)."""
    name = str(name or "").strip().upper()
    if not name:
        return None

    # Trackers live in cv2.legacy for many OpenCV 4.x builds.
    mods = [getattr(cv2, "legacy", None), cv2]
    for mod in mods:
        if mod is None:
            continue

        fn = getattr(mod, f"Tracker{name}_create", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass

        cls = getattr(mod, f"Tracker{name}", None)
        if cls is not None and hasattr(cls, "create"):
            try:
                return cls.create()
            except Exception:
                pass

    return None


def _create_preferred_opencv_tracker(prefer: tuple[str, ...] = ("CSRT", "KCF")) -> tuple[object | None, str]:
    for name in prefer:
        tr = _try_create_opencv_tracker(name)
        if tr is not None:
            return tr, str(name)
    return None, ""


def _tracker_init(tracker: object, frame_bgr: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> bool:
    try:
        bbox = _roi_xyxy_to_xywh(roi_xyxy)
        ret = tracker.init(frame_bgr, bbox)
        return True if ret is None else bool(ret)
    except Exception:
        return False


def _tracker_update(
    tracker: object,
    frame_bgr: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[bool, tuple[int, int, int, int] | None]:
    try:
        ok, bbox = tracker.update(frame_bgr)
    except Exception:
        return False, None
    if not ok:
        return False, None
    try:
        x, y, bw, bh = bbox
    except Exception:
        return False, None
    roi = _bbox_xywh_to_roi_xyxy((float(x), float(y), float(bw), float(bh)), frame_w, frame_h)
    return True, roi


def _init_klt_points(gray: np.ndarray, roi: tuple[int, int, int, int], max_corners: int = 160) -> np.ndarray | None:
    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    pts = cv2.goodFeaturesToTrack(
        patch,
        maxCorners=int(max_corners),
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        useHarrisDetector=False,
    )
    if pts is None:
        return None

    pts = pts.astype(np.float32)
    pts[:, 0, 0] += float(x0)
    pts[:, 0, 1] += float(y0)
    return pts


def _update_roi_by_klt(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: tuple[int, int, int, int],
    pts: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    *,
    min_points: int = 10,
) -> tuple[tuple[int, int, int, int], np.ndarray | None]:
    """Track ROI translation using LK optical flow on feature points.

    Returns updated (roi, pts_on_curr_frame). ROI keeps its original size.
    """
    if pts is None or len(pts) < min_points:
        pts = _init_klt_points(prev_gray, roi)
        if pts is None:
            return _clamp_roi_xyxy(roi, frame_w, frame_h), None

    next_pts, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts is None or st is None:
        return _clamp_roi_xyxy(roi, frame_w, frame_h), None

    st = st.reshape(-1)
    good_prev = pts[st == 1]
    good_next = next_pts[st == 1]
    if good_prev is None or good_next is None or len(good_prev) < min_points:
        return _clamp_roi_xyxy(roi, frame_w, frame_h), None

    disp = good_next - good_prev
    dx = float(np.median(disp[:, 0, 0]))
    dy = float(np.median(disp[:, 0, 1]))

    x0, y0, x1, y1 = roi
    roi_w = int(x1 - x0)
    roi_h = int(y1 - y0)
    new_x0 = int(round(x0 + dx))
    new_y0 = int(round(y0 + dy))
    new_roi = _clamp_roi_xyxy((new_x0, new_y0, new_x0 + roi_w, new_y0 + roi_h), frame_w, frame_h)

    nx0, ny0, nx1, ny1 = new_roi
    inside = (
        (good_next[:, 0, 0] >= nx0)
        & (good_next[:, 0, 0] < nx1)
        & (good_next[:, 0, 1] >= ny0)
        & (good_next[:, 0, 1] < ny1)
    )
    good_next_inside = good_next[inside]

    if good_next_inside is None or len(good_next_inside) < min_points:
        pts2 = _init_klt_points(curr_gray, new_roi)
    else:
        pts2 = good_next_inside.astype(np.float32)
    return new_roi, pts2


def _detect_face_roi(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Best-effort face detection ROI using Haar cascade. Returns (x0,y0,x1,y1)."""
    try:
        cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        if not cascade_dir:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Try multiple cascades (frontal + profile) to improve hit rate.
        cascade_names = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_profileface.xml",
        ]

        best = None
        best_area = -1
        for name in cascade_names:
            cascade_path = pathlib.Path(cascade_dir) / name
            face_cascade = cv2.CascadeClassifier(str(cascade_path))
            if face_cascade.empty():
                continue

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(20, 20),
            )
            if len(faces) <= 0:
                continue

            x, y, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
            area = int(fw) * int(fh)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(x + fw), int(y + fh))

        return best
    except Exception:
        return None


def _detect_person_roi_hog(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Best-effort person detection ROI using OpenCV HOG people detector."""
    try:
        h0, w0 = frame_bgr.shape[:2]
        if h0 <= 0 or w0 <= 0:
            return None

        # HOG is slow on large frames; resize for speed and scale back.
        max_side = max(h0, w0)
        scale = 1.0
        frame = frame_bgr
        if max_side > 720:
            scale = 720.0 / float(max_side)
            nw = max(64, int(round(w0 * scale)))
            nh = max(64, int(round(h0 * scale)))
            frame = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        rects, weights = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if rects is None or len(rects) == 0:
            return None

        weights = np.array(weights).reshape(-1) if weights is not None else np.ones((len(rects),), dtype=np.float32)

        best = None
        best_score = -1e18
        for (x, y, rw, rh), wt in zip(rects, weights):
            score = float(wt) * float(int(rw) * int(rh))
            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(x + rw), int(y + rh))

        if best is None:
            return None

        x0, y0, x1, y1 = best
        if scale != 1.0:
            inv = 1.0 / scale
            x0 = int(round(x0 * inv))
            y0 = int(round(y0 * inv))
            x1 = int(round(x1 * inv))
            y1 = int(round(y1 * inv))

        return _clamp_roi_xyxy((x0, y0, x1, y1), w0, h0)
    except Exception:
        return None


def _expand_face_roi_to_head(face_roi: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Expand a detected face box into a head-centered ROI (less body included)."""
    fx0, fy0, fx1, fy1 = face_roi
    fw = max(1, fx1 - fx0)
    fh = max(1, fy1 - fy0)
    cx = fx0 + fw / 2.0
    cy = fy0 + fh / 2.0

    roi_w = int(fw * 1.8)
    roi_h = int(fh * 2.0)
    x0 = int(cx - roi_w / 2.0)
    y0 = int(cy - roi_h / 2.0)
    return _clamp_roi_xyxy((x0, y0, x0 + roi_w, y0 + roi_h), frame_w, frame_h)


def _expand_face_roi_to_subject(face_roi: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    fx0, fy0, fx1, fy1 = face_roi
    fw = max(1, fx1 - fx0)
    fh = max(1, fy1 - fy0)
    cx = fx0 + fw / 2.0
    cy = fy0 + fh / 2.0

    # Expand around face to include some head/upper-body.
    roi_w = int(fw * 2.2)
    roi_h = int(fh * 2.8)
    x0 = int(cx - roi_w / 2.0)
    y0 = int(cy - roi_h * 0.45)
    return _clamp_roi_xyxy((x0, y0, x0 + roi_w, y0 + roi_h), frame_w, frame_h)


def _approx_face_roi_from_person(person_roi: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    px0, py0, px1, py1 = person_roi
    pw = max(1, px1 - px0)
    ph = max(1, py1 - py0)

    # Approximate head/face region as upper portion of the detected person box.
    roi_w = int(pw * 0.75)
    roi_h = int(ph * 0.55)
    cx = px0 + pw / 2.0
    cy = py0 + ph * 0.22
    x0 = int(cx - roi_w / 2.0)
    y0 = int(cy - roi_h / 2.0)
    return _clamp_roi_xyxy((x0, y0, x0 + roi_w, y0 + roi_h), frame_w, frame_h)


def _score_roi_trackability(gray: np.ndarray, energy: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = roi
    patch_g = gray[y0:y1, x0:x1]
    if patch_g.size == 0:
        return -1e18
    patch_e = energy[y0:y1, x0:x1]
    edge_mean = float(patch_e.mean()) if patch_e.size else 0.0
    var = float(patch_g.var())

    pts = cv2.goodFeaturesToTrack(
        patch_g,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        useHarrisDetector=False,
    )
    n_pts = 0 if pts is None else int(len(pts))

    # Keep score scale stable across videos.
    return edge_mean + 0.30 * float(n_pts) + 0.002 * var


def _score_roi_prop_candidate(
    gray: np.ndarray,
    energy: np.ndarray,
    sat: np.ndarray,
    roi: tuple[int, int, int, int],
) -> float:
    """Score a ROI for looking like a small, distinct prop.

    Heuristic goals:
    - prefer strong-but-sparse edges (object boundaries over texture)
    - prefer some color saturation (often true for handheld props)
    - keep some trackable points for tracking stability
    """
    x0, y0, x1, y1 = roi
    patch_g = gray[y0:y1, x0:x1]
    if patch_g.size == 0:
        return -1e18

    patch_e = energy[y0:y1, x0:x1]
    patch_s = sat[y0:y1, x0:x1]

    edge_mean = float(patch_e.mean()) if patch_e.size else 0.0
    edge_frac = float((patch_e > 18.0).mean()) if patch_e.size else 0.0
    sat_mean = float(patch_s.mean()) if patch_s.size else 0.0
    var = float(patch_g.var())

    pts = cv2.goodFeaturesToTrack(
        patch_g,
        maxCorners=60,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=7,
        useHarrisDetector=False,
    )
    n_pts = 0 if pts is None else int(len(pts))

    # Edge fraction penalty reduces preference for highly textured patches (e.g., wood grain / clothing).
    # Saturation is a weak bonus; many props are colorful but not all.
    return edge_mean + 0.25 * float(n_pts) + 0.002 * var + 0.010 * sat_mean - 35.0 * edge_frac


def _pick_prop_roi_for_drop_props(
    first_bgr: np.ndarray,
    rng: random.Random,
    *,
    face_roi: tuple[int, int, int, int] | None,
    person_roi: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    """Pick a prop ROI for drop_props_required.

    We bias the search towards the hands/table region by extending the detected person box downward.
    This makes it more likely to hit objects like cups/phones on a table (often outside the person box).
    """
    h, w = first_bgr.shape[:2]
    gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.magnitude(gx, gy)

    # Search window: prefer lower part; if person box exists, extend downward into the interaction surface.
    if person_roi is not None:
        px0, py0, px1, py1 = person_roi
        pw = max(1, px1 - px0)
        ph = max(1, py1 - py0)
        sx0 = _clamp_int(int(px0 - 0.20 * pw), 0, max(0, w - 1))
        sx1 = _clamp_int(int(px1 + 0.20 * pw), sx0 + 1, w)
        sy0 = _clamp_int(int(py0 + 0.55 * ph), 0, max(0, h - 1))
        sy1 = _clamp_int(int(py0 + 1.25 * ph), sy0 + 1, h)
        midx0 = int(px0 + 0.45 * pw)
        midx1 = int(px0 + 0.55 * pw)
    else:
        sx0, sx1 = 0, w
        sy0 = int(0.58 * h)
        sy1 = int(0.95 * h)
        midx0, midx1 = int(0.45 * w), int(0.55 * w)

    best = None
    best_score = -1e18

    for _ in range(120):
        rect_w = max(14, int(rng.uniform(0.07, 0.14) * w))
        rect_h = max(14, int(rng.uniform(0.07, 0.16) * h))

        # Prefer left/right side of the torso (hands) rather than the center chest.
        cx_lo_all = sx0 + rect_w / 2.0
        cx_hi_all = (sx1 - rect_w / 2.0)
        if cx_hi_all <= cx_lo_all:
            continue

        side = rng.random() < 0.5
        if side:
            cx_lo = cx_lo_all
            cx_hi = max(cx_lo + 1.0, float(min(midx1, sx1)) - rect_w / 2.0)
        else:
            cx_lo = float(max(midx0, sx0)) + rect_w / 2.0
            cx_hi = cx_hi_all

        if cx_hi <= cx_lo:
            cx_lo, cx_hi = cx_lo_all, cx_hi_all

        cy_lo = sy0 + rect_h / 2.0
        cy_hi = sy1 - rect_h / 2.0
        if cy_hi <= cy_lo:
            continue

        cx = rng.uniform(cx_lo, cx_hi)
        cy = rng.uniform(cy_lo, cy_hi)
        x0 = int(round(cx - rect_w / 2.0))
        y0 = int(round(cy - rect_h / 2.0))
        cand = _clamp_roi_xyxy((x0, y0, x0 + rect_w, y0 + rect_h), w, h)

        if face_roi is not None and _roi_iou(cand, face_roi) > 0.12:
            continue

        sc = _score_roi_prop_candidate(gray, energy, sat, cand)
        if sc > best_score:
            best_score = sc
            best = cand

    if best is not None:
        return _clamp_roi_xyxy(best, w, h)

    # Last resort fallback.
    return _pick_salient_roi_for_prop(first_bgr, rng, avoid_roi=face_roi)


def _pick_salient_roi_for_prop(
    frame_bgr: np.ndarray,
    rng: random.Random,
    *,
    avoid_roi: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pick a prop ROI that looks 'object-like' (high edge energy), then track it.

    This does not require any detector; it only tries to avoid bland regions and avoid face ROI.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.magnitude(gx, gy)

    best = None
    best_score = -1.0

    for _ in range(40):
        rect_w = max(16, int(rng.uniform(0.14, 0.24) * w))
        rect_h = max(16, int(rng.uniform(0.14, 0.24) * h))

        x0 = int(rng.uniform(0.10, 0.90) * max(1, (w - rect_w)))
        y0 = int(rng.uniform(0.45, 0.82) * max(1, (h - rect_h)))
        x0 = _clamp_int(x0, 0, max(0, w - rect_w))
        y0 = _clamp_int(y0, 0, max(0, h - rect_h))
        roi = (x0, y0, x0 + rect_w, y0 + rect_h)

        if avoid_roi is not None and _roi_iou(roi, avoid_roi) > 0.15:
            continue

        score = _score_roi_trackability(gray, energy, roi)
        if score > best_score:
            best_score = score
            best = roi

    if best is not None:
        return _clamp_roi_xyxy(best, w, h)

    # Fallback: original heuristic.
    rect_w = max(16, int(rng.uniform(0.16, 0.26) * w))
    rect_h = max(16, int(rng.uniform(0.16, 0.26) * h))
    x0 = int(rng.uniform(0.20, 0.80) * max(1, (w - rect_w)))
    y0 = int(rng.uniform(0.45, 0.80) * max(1, (h - rect_h)))
    x0 = _clamp_int(x0, 0, max(0, w - rect_w))
    y0 = _clamp_int(y0, 0, max(0, h - rect_h))
    return _clamp_roi_xyxy((x0, y0, x0 + rect_w, y0 + rect_h), w, h)


def _apply_temporal_jitter(src: str, dst: str, rng: random.Random):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw_path = _open_perturb_writer(dst, fps, w, h)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    changed = False
    for i, frame in enumerate(frames):
        # Randomly duplicate or drop frames
        r = rng.random()
        if r < 0.08 and i > 0:    # drop
            changed = True
            continue
        out.write(frame)
        if r < 0.16:              # duplicate
            out.write(frame)
            changed = True
    if frames and not changed:
        # Ensure at least one visible temporal artifact
        mid = frames[len(frames) // 2]
        out.write(mid)
    _finalize_perturb_writer(out, raw_path, dst)


def _apply_compression(src: str, dst: str, rng: random.Random):
    """Re-encode at low bitrate via OpenCV."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw_path = _open_perturb_writer(dst, fps, w, h)

    # Make the degradation seed-dependent so multiple seeds aren't identical.
    div = rng.randint(6, 10)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Simulate compression: downscale then upscale
        small = cv2.resize(frame, (max(1, w // div), max(1, h // div)),
                           interpolation=cv2.INTER_LINEAR)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out.write(degraded)
    cap.release()
    _finalize_perturb_writer(out, raw_path, dst)


def _apply_remove_anchor(src: str, dst: str, rng: random.Random):
    """Simulate identity drift by color-jittering a subject ROI for a contiguous block.

    We try to detect a face once (OpenCV Haar cascade) to anchor the ROI; otherwise
    fall back to a central ROI. This keeps the background mostly unchanged so this
    perturbation is less like a global style/color shift.
    """
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw_path = _open_perturb_writer(dst, fps, w, h)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    n = len(frames)
    if n == 0:
        _finalize_perturb_writer(out, raw_path, dst)
        return

    block_len = max(1, n // 4)

    # Try to anchor the perturbation window at a frame where a face/person is detectable.
    probe = sorted(set([0, n // 6, 2 * n // 6, 3 * n // 6, 4 * n // 6, max(0, n - 1)]))
    anchor_idx = None
    anchor_face = None
    anchor_person = None
    for idx in probe:
        fr = frames[idx]
        fr_face = _detect_face_roi(fr)
        if fr_face is not None:
            anchor_idx = idx
            anchor_face = fr_face
            break
        fr_person = _detect_person_roi_hog(fr)
        if fr_person is not None and anchor_person is None:
            anchor_idx = idx
            anchor_person = fr_person

    if anchor_idx is None:
        anchor_idx = rng.randint(max(0, n // 3), max(0, 2 * n // 3)) if n > 1 else 0

    start = int(max(0, min(n - 1, anchor_idx)))
    start = int(max(0, min(n - block_len, start))) if n > block_len else 0
    end = min(n, start + block_len)

    anchor = frames[start]
    roi = None

    # Prefer face ROI; fall back to person ROI; last resort is center ROI.
    face_roi = _detect_face_roi(anchor) or anchor_face
    if face_roi is not None:
        roi = _expand_face_roi_to_head(face_roi, w, h)
    else:
        person_roi = _detect_person_roi_hog(anchor) or anchor_person
        if person_roi is not None:
            roi = _approx_face_roi_from_person(person_roi, w, h)

    if roi is None:
        roi_w = max(16, int(0.45 * w))
        roi_h = max(16, int(0.65 * h))
        jitter_x = int(rng.uniform(-0.05, 0.05) * w)
        jitter_y = int(rng.uniform(-0.05, 0.05) * h)
        cx = (w // 2) + jitter_x
        cy = int(0.45 * h) + jitter_y
        x0 = int(cx - roi_w / 2)
        y0 = int(cy - roi_h / 2)
        roi = _clamp_roi_xyxy((x0, y0, x0 + roi_w, y0 + roi_h), w, h)

    # Track ROI within the perturbation window so the change follows the person.
    x0, y0, x1, y1 = roi
    roi_hh = max(1, y1 - y0)
    roi_ww = max(1, x1 - x0)

    hue_delta = int(rng.randint(25, 55))
    sat_scale = float(rng.uniform(0.90, 1.20))

    mask = np.zeros((roi_hh, roi_ww), dtype=np.float32)
    cv2.ellipse(mask, (roi_ww // 2, roi_hh // 2), (roi_ww // 2, roi_hh // 2), 0, 0, 360, 1.0, -1)
    sigma = max(3.0, 0.02 * float(min(roi_ww, roi_hh)))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    mask3 = mask[:, :, None]

    cur_roi = roi

    tracker, _tracker_name = _create_preferred_opencv_tracker()
    tracker_ok = False
    if tracker is not None:
        tracker_ok = _tracker_init(tracker, frames[start], cur_roi)
    if not tracker_ok:
        tracker = None

    prev_gray = cv2.cvtColor(frames[start], cv2.COLOR_BGR2GRAY)
    pts = None if tracker is not None else _init_klt_points(prev_gray, cur_roi)
    tpl = prev_gray[cur_roi[1]:cur_roi[3], cur_roi[0]:cur_roi[2]].copy()

    re_detect_every = 6
    tpl_update_every = 4

    for i, frame in enumerate(frames):
        if i < start or i >= end:
            out.write(frame)
            continue

        if i > start:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Preferred: classic tracker update (CSRT/KCF)
            if tracker_ok and tracker is not None:
                ok_t, roi_t = _tracker_update(tracker, frame, w, h)
                if ok_t and roi_t is not None:
                    cur_roi = roi_t
                else:
                    tracker_ok = False
                    tracker = None
                    pts = None

            # Periodic re-detection to re-anchor on face/person.
            if (i - start) % re_detect_every == 0:
                fr_face = _detect_face_roi(frame)
                if fr_face is not None:
                    cur_roi = _expand_face_roi_to_head(fr_face, w, h)
                    tracker, _tracker_name = _create_preferred_opencv_tracker()
                    tracker_ok = tracker is not None and _tracker_init(tracker, frame, cur_roi)
                    if not tracker_ok:
                        tracker = None
                    pts = None
                    tpl = curr_gray[cur_roi[1]:cur_roi[3], cur_roi[0]:cur_roi[2]].copy()
                else:
                    fr_person = _detect_person_roi_hog(frame)
                    if fr_person is not None:
                        cur_roi = _approx_face_roi_from_person(fr_person, w, h)
                        tracker, _tracker_name = _create_preferred_opencv_tracker()
                        tracker_ok = tracker is not None and _tracker_init(tracker, frame, cur_roi)
                        if not tracker_ok:
                            tracker = None
                        pts = None
                        tpl = curr_gray[cur_roi[1]:cur_roi[3], cur_roi[0]:cur_roi[2]].copy()

            # Fallback: KLT + template match (when tracker is unavailable or lost).
            if tracker is None:
                cur_roi, pts = _update_roi_by_klt(prev_gray, curr_gray, cur_roi, pts, w, h)

                # Template match refinement around predicted ROI
                if tpl is not None and tpl.size > 0:
                    rx0, ry0, rx1, ry1 = cur_roi
                    rw = max(8, rx1 - rx0)
                    rh = max(8, ry1 - ry0)
                    margin_x = int(0.6 * rw)
                    margin_y = int(0.6 * rh)
                    sx0 = _clamp_int(rx0 - margin_x, 0, max(0, w - rw))
                    sy0 = _clamp_int(ry0 - margin_y, 0, max(0, h - rh))
                    sx1 = _clamp_int(rx0 + rw + margin_x, sx0 + rw, w)
                    sy1 = _clamp_int(ry0 + rh + margin_y, sy0 + rh, h)

                    search = curr_gray[sy0:sy1, sx0:sx1]
                    if search.size >= tpl.size:
                        res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
                        _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(res)
                        if float(maxv) >= 0.35:
                            mx, my = int(maxloc[0]), int(maxloc[1])
                            cur_roi = _clamp_roi_xyxy((sx0 + mx, sy0 + my, sx0 + mx + rw, sy0 + my + rh), w, h)
                            pts = None

                if pts is None:
                    pts = _init_klt_points(curr_gray, cur_roi)

            if (i - start) % tpl_update_every == 0:
                rx0, ry0, rx1, ry1 = cur_roi
                tpl = curr_gray[ry0:ry1, rx0:rx1].copy()

            prev_gray = curr_gray

        rx0, ry0, rx1, ry1 = cur_roi
        roi_src = frame[ry0:ry1, rx0:rx1]

        # If ROI was clamped at borders, sizes may differ; rebuild a per-frame soft mask.
        hh = max(1, ry1 - ry0)
        ww = max(1, rx1 - rx0)
        if hh != roi_hh or ww != roi_ww:
            m = np.zeros((hh, ww), dtype=np.float32)
            cv2.ellipse(m, (ww // 2, hh // 2), (ww // 2, hh // 2), 0, 0, 360, 1.0, -1)
            sig = max(3.0, 0.02 * float(min(ww, hh)))
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=sig)
            m3 = m[:, :, None]
        else:
            m3 = mask3

        hsv = cv2.cvtColor(roi_src, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * sat_scale, 0, 255).astype(np.int32)
        roi_shift = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        blended = (
            roi_src.astype(np.float32) * (1.0 - m3) +
            roi_shift.astype(np.float32) * m3
        ).astype(np.uint8)

        out_frame = frame.copy()
        out_frame[ry0:ry1, rx0:rx1] = blended
        out.write(out_frame)
    _finalize_perturb_writer(out, raw_path, dst)


def _apply_extend_duration(src: str, dst: str, rng: random.Random):
    """Duplicate a tail chunk of frames to simulate segment extension."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw_path = _open_perturb_writer(dst, fps, w, h)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    # Seed-dependent tail size (roughly 15% ~ 35%)
    if frames:
        tail_ratio = rng.uniform(0.15, 0.35)
        tail_start = max(0, min(len(frames) - 1, int(len(frames) * (1.0 - tail_ratio))))
        tail = frames[tail_start:]
    else:
        tail = []
    for f in frames + tail:
        out.write(f)
    _finalize_perturb_writer(out, raw_path, dst)


def _apply_drop_props_required(src: str, dst: str, rng: random.Random):
    """Simulate a 'missing prop' by removing a static region via inpainting.

    This is a lightweight pixel-level perturbation (no object detector).
    """
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Failed to read video: {src}")

    h, w = first.shape[:2]
    out, raw_path = _open_perturb_writer(dst, fps, w, h)

    face_roi = _detect_face_roi(first)
    person_roi = _detect_person_roi_hog(first)

    roi = _pick_prop_roi_for_drop_props(first, rng, face_roi=face_roi, person_roi=person_roi)

    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    tracker, _tracker_name = _create_preferred_opencv_tracker()
    tracker_ok = False
    if tracker is not None:
        tracker_ok = _tracker_init(tracker, first, roi)
    if not tracker_ok:
        tracker = None

    prev_gray = first_gray
    pts = None if tracker is not None else _init_klt_points(prev_gray, roi)
    tpl = prev_gray[roi[1]:roi[3], roi[0]:roi[2]].copy()

    mask = np.zeros((h, w), dtype=np.uint8)

    def _remove_region(frame_bgr: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = roi_xyxy
        mask.fill(0)
        mask[y0:y1, x0:x1] = 255
        try:
            return cv2.inpaint(frame_bgr, mask, 3, cv2.INPAINT_TELEA)
        except Exception:
            # Fallback: heavy blur on the region
            removed2 = frame_bgr.copy()
            patch = removed2[y0:y1, x0:x1]
            removed2[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (31, 31), 0)
            return removed2

    # Process first frame
    out.write(_remove_region(first, roi))

    tpl_update_every = 4
    tracker_reinit_every = 3

    frame_idx = 1

    # Process rest frames with tracking (KLT + template match refinement)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Preferred: classic tracker update (CSRT/KCF)
        if tracker_ok and tracker is not None:
            ok_t, roi_t = _tracker_update(tracker, frame, w, h)
            if ok_t and roi_t is not None:
                roi = roi_t
            else:
                tracker_ok = False
                tracker = None
                pts = None

        # Fallback: KLT + template match
        if tracker is None:
            roi, pts = _update_roi_by_klt(prev_gray, curr_gray, roi, pts, w, h)

            # Template matching around predicted ROI improves stability for small props.
            if tpl is not None and tpl.size > 0:
                rx0, ry0, rx1, ry1 = roi
                rw = max(8, rx1 - rx0)
                rh = max(8, ry1 - ry0)
                margin_x = int(0.8 * rw)
                margin_y = int(0.8 * rh)
                sx0 = _clamp_int(rx0 - margin_x, 0, max(0, w - rw))
                sy0 = _clamp_int(ry0 - margin_y, 0, max(0, h - rh))
                sx1 = _clamp_int(rx0 + rw + margin_x, sx0 + rw, w)
                sy1 = _clamp_int(ry0 + rh + margin_y, sy0 + rh, h)
                search = curr_gray[sy0:sy1, sx0:sx1]
                if search.size >= tpl.size:
                    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
                    _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(res)
                    if float(maxv) >= 0.35:
                        mx, my = int(maxloc[0]), int(maxloc[1])
                        roi = _clamp_roi_xyxy((sx0 + mx, sy0 + my, sx0 + mx + rw, sy0 + my + rh), w, h)
                        pts = None

            if pts is None:
                pts = _init_klt_points(curr_gray, roi)

            # Try to switch back to classic tracker when possible.
            if (frame_idx % tracker_reinit_every) == 0:
                tracker, _tracker_name = _create_preferred_opencv_tracker()
                tracker_ok = tracker is not None and _tracker_init(tracker, frame, roi)
                if not tracker_ok:
                    tracker = None

        prev_gray = curr_gray

        if (frame_idx % tpl_update_every) == 0:
            rx0, ry0, rx1, ry1 = roi
            tpl = curr_gray[ry0:ry1, rx0:rx1].copy()

        out.write(_remove_region(frame, roi))

        frame_idx += 1

    cap.release()
    _finalize_perturb_writer(out, raw_path, dst)


def _apply_change_camera_movement(src: str, dst: str, rng: random.Random):
    """Simulate a 'wrong camera movement' using smooth digital pan/zoom/roll."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Failed to read video: {src}")

    h, w = first.shape[:2]
    out, raw_path = _open_perturb_writer(dst, fps, w, h)

    mode = rng.choice(["pan", "zoom", "roll"])  # keep it simple & visible
    period = max(24, int(fps * 2.0))

    max_shift = max(2, int(0.06 * min(w, h)))
    zoom_amp = float(rng.uniform(0.05, 0.12))
    max_angle = float(rng.uniform(1.0, 3.0))

    def _warp(frame: np.ndarray, i: int) -> np.ndarray:
        phase = 2.0 * math.pi * (i % period) / period
        dx = int(max_shift * math.sin(phase))
        dy = int((max_shift * 0.5) * math.sin(phase * 1.37))

        if mode == "pan":
            m = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        else:
            if mode == "zoom":
                angle = 0.0
                scale = 1.0 + zoom_amp * math.sin(phase)
            else:  # roll
                angle = max_angle * math.sin(phase)
                scale = 1.0

            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            m[0, 2] += dx
            m[1, 2] += dy

        return cv2.warpAffine(
            frame,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    frame = first
    i = 0
    while True:
        out.write(_warp(frame, i))
        i += 1
        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    _finalize_perturb_writer(out, raw_path, dst)


def generate_dataset(input_manifest: str, out_dir: str,
                     seeds: int = 2) -> None:
    """Generate perturbed dataset from input manifest."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos_dir = out / "videos"
    videos_dir.mkdir(exist_ok=True)

    out_lines = []
    with open(input_manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for entry in entries:
        vid_path = entry["video_path"]
        base_id = entry["id"]
        for perturb_type in PERTURB_CODE_MAP:
            for seed in range(seeds):
                new_id = f"{base_id}__{perturb_type}__s{seed}"
                out_vid = str(videos_dir / f"{new_id}.mp4")
                try:
                    info = apply_perturbation(vid_path, perturb_type,
                                             out_vid, seed=seed)
                    rec = {
                        "id": new_id,
                        "video_path": out_vid,
                        "shotir_path": entry.get("shotir_path"),
                        "perturbation_type": info["perturbation_type"],
                        "failure_codes": info["failure_codes"],
                        "meta": entry.get("meta", {}),
                    }
                    out_lines.append(json.dumps(rec))
                    logger.info("Generated: %s", new_id)
                except Exception as e:
                    logger.error("Failed %s / %s: %s", new_id, perturb_type, e)

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    logger.info("Dataset manifest: %s (%d entries)", manifest_path, len(out_lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    generate_dataset(args.input_manifest, args.out, seeds=args.seeds)

