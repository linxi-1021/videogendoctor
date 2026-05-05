"""OpenAI-backed VLM judge for Stage-2 verification."""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image

from services.worker.judge.base import BaseJudge

logger = logging.getLogger(__name__)


_MODEL_ALIASES = {
    "gpt-4v": "gpt-4o",
    "gpt4v": "gpt-4o",
    "gpt-4-vision-preview": "gpt-4o",
}


class OpenAIJudge(BaseJudge):
    """Vision judge implemented with the OpenAI Responses API."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.requested_model_name = model_name
        self.model_name = _MODEL_ALIASES.get(model_name, model_name)
        self._client = None
        self._fallback = False
        self._load_client()

    def _load_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI()
            if self.model_name != self.requested_model_name:
                logger.warning(
                    "Mapped legacy OpenAI vision model alias %s to %s.",
                    self.requested_model_name,
                    self.model_name,
                )
            logger.info("OpenAIJudge initialized with model=%s", self.model_name)
        except Exception as e:
            logger.warning(
                "OpenAIJudge unavailable for model '%s': %s. Falling back to DummyJudge.",
                self.requested_model_name,
                e,
            )
            self._fallback = True

    def judge_segment(
        self,
        frames: list,
        questions: list[dict],
        context: dict | None = None,
    ) -> dict:
        if self._fallback or self._client is None:
            from services.worker.judge.dummy_judge import DummyJudge
            return DummyJudge().judge_segment(frames, questions, context)

        image_content = []
        if frames:
            image_content.append({
                "type": "input_image",
                "image_url": _frame_to_data_url(frames[0]),
            })

        answers: dict[str, str] = {}
        confidences: dict[str, float] = {}
        code_probs: dict[str, float] = {}

        for q in questions:
            qid = q["qid"]
            prompt = (
                "You are a strict video quality judge.\n"
                "Answer exactly in one line as either 'Yes' or 'No', then optionally a short reason.\n"
                f"Question: {q['text']}"
            )
            content = [{"type": "input_text", "text": prompt}, *image_content]
            try:
                response = self._client.responses.create(
                    model=self.model_name,
                    input=[{"role": "user", "content": content}],
                    max_output_tokens=80,
                )
                ans = (response.output_text or "").strip() or "[no response]"
            except Exception as e:
                logger.warning("OpenAI judge inference error for %s: %s", qid, e)
                ans = "[inference error]"

            answers[qid] = ans
            conf = _answer_confidence(ans)
            confidences[qid] = conf

            code = q.get("code_group", "").upper()
            if code:
                code_probs[code] = max(code_probs.get(code, 0.0), conf)

        return {
            "answers": answers,
            "confidences": confidences,
            "code_probs": code_probs,
            "evidence_rerank": [],
        }


def _frame_to_data_url(frame: Any) -> str:
    image = Image.fromarray(frame)
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=90)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _answer_confidence(answer: str) -> float:
    normalized = answer.strip().lower()
    if normalized.startswith("yes"):
        return 0.85
    if normalized.startswith("no"):
        return 0.15
    return 0.5
