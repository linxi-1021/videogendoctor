"""Runtime config helpers for VideoEval."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_runtime_config(
    config_path: str | None,
    overrides: dict[str, Any] | None = None,
    *,
    judge_config_path: str | None = None,
) -> dict[str, Any]:
    import os
    import yaml

    cfg: dict[str, Any] = {
        "seg_len": 2.0,
        "stride": 2.0,
        "k_frames": 6,
        "use_yolo": False,
        "use_judge": False,
        "clip_drift_threshold": 0.15,
        "flow_jitter_threshold": 5.0,
        "flow_frozen_threshold": 0.01,
        "flow_shake_threshold": 8.0,
        "face_drift_threshold": 0.35,
        "top_k_failures": 3,
        "enable_prop_placeholders": False,
        "judge_provider": None,
        "judge_model": None,
        "judge_candidate_topk": 3,
        "judge_alpha": 0.6,
        "judge_device": "cpu",
        "judge_question_templates": "configs/paper1/judge_questions.yaml",
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})

    resolved_judge_path = _resolve_judge_config_path(config_path, judge_config_path)
    if resolved_judge_path and resolved_judge_path.exists():
        with open(resolved_judge_path, encoding="utf-8") as f:
            judge_cfg = yaml.safe_load(f) or {}
        cfg.update(_normalize_judge_config(judge_cfg))

    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _resolve_judge_config_path(
    config_path: str | None,
    judge_config_path: str | None,
) -> Path | None:
    if judge_config_path:
        return Path(judge_config_path)
    if config_path:
        base = Path(config_path).resolve().parent
        candidate = base / "judge.yaml"
        if candidate.exists():
            return candidate
    return None


def _normalize_judge_config(judge_cfg: dict[str, Any]) -> dict[str, Any]:
    provider = judge_cfg.get("provider")
    model_name = judge_cfg.get("model_name")
    if not provider and isinstance(model_name, str):
        provider = infer_judge_provider(model_name)
    return {
        "judge_provider": provider,
        "judge_model": model_name,
        "judge_candidate_topk": judge_cfg.get("candidate_topk"),
        "judge_alpha": judge_cfg.get("alpha"),
        "judge_device": judge_cfg.get("device"),
        "judge_question_templates": judge_cfg.get("question_templates"),
    }


def infer_judge_provider(model_name: str | None) -> str | None:
    if not model_name:
        return None
    normalized = model_name.strip().lower()
    if normalized in {"gpt-4v", "gpt4v", "gpt-4-vision-preview"}:
        return "openai"
    if normalized.startswith(("gpt-", "o1", "o3", "o4", "gpt4")):
        return "openai"
    return "transformers"
