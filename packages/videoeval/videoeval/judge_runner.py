"""Run Stage-2 VLM judge on Stage-1 candidate segments."""
from __future__ import annotations
import logging
from pathlib import Path

from videoeval.config import infer_judge_provider

logger = logging.getLogger(__name__)

# Default question templates per code group
_QUESTIONS = [
    {"qid": "Q_IDENTITY_DRIFT", "text": "Does the main character's face or body appearance change inconsistently between frames? Answer Yes or No.", "code_group": "ID_FACE_DRIFT"},
    {"qid": "Q_PROP_MISSING",   "text": "Is there any required object or prop missing from this scene? Answer Yes or No.",  "code_group": "AL_PROP_MISSING"},
    {"qid": "Q_CAMERA_DEV",     "text": "Does the camera movement appear unintended or inconsistent with a planned shot? Answer Yes or No.", "code_group": "CA_MOVE_WRONG"},
    {"qid": "Q_ACTION_PRESENT", "text": "Does a clear action or event occur in this segment? Answer Yes or No.", "code_group": "MO_EVENT_MISSING"},
    {"qid": "Q_STYLE_CONSIST",  "text": "Is the visual style consistent throughout this segment? Answer Yes or No.", "code_group": "ST_COLOR_SHIFT"},
]


def run_judge(segments: list[dict], cfg: dict, out_dir: str | None = None) -> list[dict]:
    """Run VLM judge on top-K candidate segments and update failure scores."""
    judge = _load_judge(cfg)
    top_k = cfg.get("judge_candidate_topk", 3)
    alpha = cfg.get("judge_alpha", 0.6)

    # Score each segment by max stage-1 failure confidence
    scored = sorted(
        [(max((f["confidence"] for f in s.get("failures", [])), default=0.0), i)
         for i, s in enumerate(segments)],
        reverse=True
    )
    candidate_indices = {i for _, i in scored[:top_k]}

    for i, seg in enumerate(segments):
        if i not in candidate_indices:
            continue
        # Load keyframe images
        frames = _load_keyframes(seg.get("keyframes", []), out_dir=out_dir)
        result = judge.judge_segment(frames=frames, questions=_QUESTIONS)
        code_probs = result.get("code_probs", {})

        # Update failure confidences
        for fail in seg.get("failures", []):
            code = fail["code"]
            if code in code_probs:
                s1 = fail["confidence"]
                s2 = code_probs[code]
                fail["confidence"] = round((1 - alpha) * s1 + alpha * s2, 4)
                fail["_judge_answer"] = result["answers"]

        # Re-rank evidence keyframes
        reranked = result.get("evidence_rerank", [])
        if reranked:
            for fail in seg.get("failures", []):
                fail["evidence"]["keyframes"] = reranked[:2]

    return segments


def _load_judge(cfg: dict):
    model_name = cfg.get("judge_model")
    if not model_name:
        logger.info("No judge_model configured; using DummyJudge.")
        from services.worker.judge.dummy_judge import DummyJudge
        return DummyJudge()
    provider = cfg.get("judge_provider") or infer_judge_provider(model_name)
    device = cfg.get("judge_device", "cpu")
    try:
        if provider == "openai":
            from services.worker.judge.openai_judge import OpenAIJudge
            return OpenAIJudge(model_name=model_name)
        from services.worker.judge.transformers_hook import TransformersJudge
        return TransformersJudge(model_name=model_name, device=device)
    except Exception as e:
        logger.warning("Could not load judge provider=%s model=%s: %s; using DummyJudge.", provider, model_name, e)
        from services.worker.judge.dummy_judge import DummyJudge
        return DummyJudge()


def _load_keyframes(paths: list[str], out_dir: str | None = None) -> list:
    import cv2

    base_dir = Path(out_dir).resolve() if out_dir else None
    frames = []
    for p in paths:
        try:
            img_path = _resolve_keyframe_path(p, base_dir=base_dir)
            if img_path is None:
                logger.debug("Skipping missing keyframe: %s", p)
                continue
            img = cv2.imread(str(img_path))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
    return frames


def _resolve_keyframe_path(path_str: str, base_dir: Path | None) -> Path | None:
    path = Path(path_str)
    if path.is_absolute():
        return path if path.exists() else None
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    if path.exists():
        return path.resolve()
    return None

