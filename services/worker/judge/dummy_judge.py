"""Dummy judge — always runnable, returns placeholder outputs."""
from __future__ import annotations
from services.worker.judge.base import BaseJudge


class DummyJudge(BaseJudge):
    """Always-runnable dummy judge for testing and ablations.

    Returns neutral (0.5) confidences and empty answers.
    Safe to use in CI and smoke tests.
    """

    def judge_segment(
        self,
        frames: list,
        questions: list[dict],
        context: dict | None = None,
    ) -> dict:
        answers = {q["qid"]: "[DUMMY] No VLM loaded." for q in questions}
        confidences = {q["qid"]: 0.5 for q in questions}
        code_probs = {}  # no adjustment from dummy
        return {
            "answers": answers,
            "confidences": confidences,
            "code_probs": code_probs,
            "evidence_rerank": [],
        }

