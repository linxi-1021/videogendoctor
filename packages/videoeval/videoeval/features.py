"""Feature extraction: segmentation, CLIP drift, optical flow, face drift."""
from __future__ import annotations
import pathlib
import logging
import os
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_features(video_path: str, out_dir: str,
                     seg_len: float = 2.0, stride: float = 2.0,
                     k_frames: int = 6, use_yolo: bool = False) -> list[dict]:
    """Segment video, extract keyframes, compute feature streams."""
    out = pathlib.Path(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    seg_frames = max(1, int(seg_len * fps))
    stride_frames = max(1, int(stride * fps))

    # Load CLIP model (graceful fallback)
    clip_model = _load_clip()
    # Load face model (graceful fallback)
    face_model = _load_face()
    # Load YOLO (graceful fallback)
    yolo_model = _load_yolo() if use_yolo else None

    segments = []
    seg_idx = 0
    start_f = 0

    while start_f < n_frames:
        end_f = min(start_f + seg_frames, n_frames)
        t_start = start_f / fps
        t_end = end_f / fps
        seg_id = f"seg_{seg_idx:03d}"

        # Sample k_frames evenly in [start_f, end_f)
        frame_indices = _sample_indices(start_f, end_f, k_frames)
        keyframe_paths = []
        frames_rgb = []

        ev_dir = out / "evidence" / seg_id
        ev_dir.mkdir(parents=True, exist_ok=True)

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
            kf_abs = ev_dir / f"frame_{fi:06d}.jpg"
            cv2.imwrite(str(kf_abs), frame)
            # Store paths relative to the output directory using POSIX separators
            # so they work in HTML reports across OSes.
            keyframe_paths.append(kf_abs.relative_to(out).as_posix())

        # CLIP drift
        clip_drift = _compute_clip_drift(frames_rgb, clip_model)
        # Optical flow
        flow_mean, flow_std = _compute_flow(cap, start_f, end_f, fps)
        # Face drift
        face_drift, has_face = _compute_face_drift(frames_rgb, face_model)

        segments.append({
            "seg_id": seg_id,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "keyframes": keyframe_paths,
            "features": {
                "clip_drift": round(clip_drift, 4),
                "flow_magnitude_mean": round(flow_mean, 4),
                "flow_magnitude_std": round(flow_std, 4),
                "face_drift": round(face_drift, 4) if face_drift is not None else None,
                "has_face": has_face,
            },
            "failures": [],
        })

        start_f += stride_frames
        seg_idx += 1

    cap.release()
    return segments


def _sample_indices(start, end, k):
    if end <= start:
        return [start]
    total = end - start
    step = max(1, total // k)
    return [start + i * step for i in range(k) if start + i * step < end]


def _preferred_torch_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _preferred_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        available = set()

    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available or not providers:
        providers.append("CPUExecutionProvider")
    return providers


@lru_cache(maxsize=1)
def _load_clip():
    try:
        import open_clip
        import torch
        model_name = "ViT-B-32"
        pretrained_tag = "openai"
        # The OpenAI-tagged CLIP weights are trained with QuickGELU.
        # Force QuickGELU so the model config matches the pretrained tag and
        # avoids the open_clip QuickGELU mismatch warning.
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_tag, force_quick_gelu=True
            )
        except TypeError:
            # Compatibility with open_clip versions that don't expose
            # force_quick_gelu in create_model_and_transforms.
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_tag
            )
        device = _preferred_torch_device()
        model = model.to(device)
        model.eval()
        logger.info("OpenCLIP initialized on device=%s", device)
        return {"model": model, "preprocess": preprocess,
                "tokenizer": None, "torch": torch, "device": device}
    except Exception as e:
        logger.warning("OpenCLIP not available (%s); using random drift fallback.", e)
        return None


@lru_cache(maxsize=1)
def _load_face():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        providers = _preferred_onnx_providers()
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        app = FaceAnalysis(providers=providers)
        app.prepare(ctx_id=ctx_id)
        logger.info("InsightFace initialized with providers=%s ctx_id=%s", providers, ctx_id)
        return app
    except Exception as e:
        logger.warning("InsightFace not available (%s); face drift disabled.", e)
        return None


@lru_cache(maxsize=1)
def _load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as e:
        logger.warning("YOLOv8 not available (%s); object detection disabled.", e)
        return None


def _compute_clip_drift(frames: list, clip_pkg) -> float:
    if clip_pkg is None or len(frames) < 2:
        # Fallback: pixel-level drift
        if len(frames) < 2:
            return 0.0
        diffs = []
        for a, b in zip(frames[:-1], frames[1:]):
            diff = float(np.mean(np.abs(a.astype(float) - b.astype(float)))) / 255.0
            diffs.append(diff)
        return float(np.mean(diffs))
    import torch
    from PIL import Image
    model = clip_pkg["model"]
    preprocess = clip_pkg["preprocess"]
    device = clip_pkg.get("device", "cpu")
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            img = Image.fromarray(frame)
            tensor = preprocess(img).unsqueeze(0).to(device)
            emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze(0).cpu().numpy())
    drifts = []
    for a, b in zip(embeddings[:-1], embeddings[1:]):
        cos = float(np.dot(a, b))
        drifts.append(1.0 - cos)
    return float(np.mean(drifts)) if drifts else 0.0


def _compute_flow(cap, start_f, end_f, fps) -> tuple[float, float]:
    """Compute mean/std optical flow magnitude over segment."""
    magnitudes = []
    prev_gray = None
    indices = _sample_indices(start_f, end_f, 8)
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                magnitudes.append(float(np.mean(mag)))
            except Exception:
                pass
        prev_gray = gray
    if not magnitudes:
        return 0.0, 0.0
    return float(np.mean(magnitudes)), float(np.std(magnitudes))


def _compute_face_drift(frames: list, face_app) -> tuple[float | None, bool]:
    if face_app is None:
        return None, False
    embeddings = []
    for frame in frames:
        try:
            faces = face_app.get(frame)
            if faces:
                emb = faces[0].normed_embedding
                embeddings.append(emb)
        except Exception:
            pass
    if len(embeddings) < 2:
        return None, len(embeddings) > 0
    drifts = []
    for a, b in zip(embeddings[:-1], embeddings[1:]):
        cos = float(np.dot(a, b))
        drifts.append(1.0 - cos)
    return float(np.mean(drifts)), True

