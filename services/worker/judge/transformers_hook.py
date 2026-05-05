"""Transformers hook judge — pluggable open-source VLM.

If the model is missing, falls back to DummyJudge with a clear message.
"""
from __future__ import annotations
import logging

from services.worker.judge.base import BaseJudge

logger = logging.getLogger(__name__)


def _normalize_generated_text(output) -> str:
    """Flatten HF pipeline outputs into a plain answer string."""
    if output is None:
        return "[no response]"
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = [_normalize_generated_text(item) for item in output]
        parts = [part for part in parts if part and part != "[no response]"]
        return "\n".join(parts) if parts else "[no response]"
    if isinstance(output, dict):
        # Common chat-style payloads:
        # {"role": "assistant", "content": "..."}
        # {"generated_text": [...]}
        # {"text": "..."}
        for key in ("generated_text", "text", "content"):
            if key in output:
                return _normalize_generated_text(output[key])
        return str(output)
    return str(output)


class TransformersJudge(BaseJudge):
    """VLM judge using a HuggingFace transformers model.

    Set model_name in configs/paper1/judge.yaml.
    Falls back to DummyJudge if model is unavailable.
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._fallback = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading VLM judge: %s on %s", self.model_name, self.device)
            self._pipe = hf_pipeline(
                "image-text-to-text",
                model=self.model_name,
                device=self.device,
            )
            logger.info("VLM judge loaded successfully.")
        except Exception as e:
            logger.warning(
                "TransformersJudge: could not load model '%s': %s\n"
                "  Falling back to DummyJudge. "
                "Install transformers + torch and set model_name in judge.yaml.",
                self.model_name, e
            )
            self._fallback = True

    def judge_segment(
        self,
        frames: list,
        questions: list[dict],
        context: dict | None = None,
    ) -> dict:
        if self._fallback:
            from services.worker.judge.dummy_judge import DummyJudge
            return DummyJudge().judge_segment(frames, questions, context)

        from PIL import Image
        import numpy as np

        answers, confidences, code_probs = {}, {}, {}

        for q in questions:
            qid = q["qid"]
            text = q["text"]
            # Use first keyframe for VLM input
            if frames:
                img = Image.fromarray(frames[0]) if isinstance(frames[0], __import__('numpy').ndarray) else frames[0]
            else:
                img = Image.new("RGB", (224, 224))

            try:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ]}]
                result = self._pipe(messages, max_new_tokens=128)
                ans = _normalize_generated_text(
                    result[0].get("generated_text") if result else None
                )
            except Exception as e:
                logger.warning("VLM inference error for %s: %s", qid, e)
                ans = "[inference error]"

            answers[qid] = ans
            # Heuristic confidence from Yes/No answer
            ans_lower = ans.lower()
            conf = 0.85 if ans_lower.startswith("yes") else (
                   0.15 if ans_lower.startswith("no") else 0.5)
            confidences[qid] = conf

            # Map qid to code_prob
            code = q.get("code_group", "").upper()
            if code:
                code_probs[code] = max(code_probs.get(code, 0.0), conf)

        return {
            "answers": answers,
            "confidences": confidences,
            "code_probs": code_probs,
            "evidence_rerank": [],
        }

