"""Base class for Stage-2 VLM judge."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseJudge(ABC):
    """Abstract VLM judge interface.

    Implementations must override judge_segment().
    """

    @abstractmethod
    def judge_segment(
        self,
        frames: list,           # list of np.ndarray (RGB)
        questions: list[dict],  # [{qid, text, code_group}]
        context: dict | None = None,
    ) -> dict:
        """Judge a segment.

        Returns:
            {
              segment_id: str,
              answers: {qid: str},
              confidences: {qid: float},
              code_probs: {code: float},
              evidence_rerank: [keyframe_path, ...],
            }
        """

