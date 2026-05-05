"""Generate a full simulated experiment suite for demos.

This module creates clearly-labeled simulated data for:
- algorithm integration tests
- paper method illustrations
- visualization dashboards

It must never be used as real experimental evidence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import copy
import csv
import html
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.scripts.eval_evidence_localization import compute_localization
from infra.scripts.eval_failure_codes import compute_f1


@dataclass(frozen=True)
class Profile:
    name: str
    category: str
    kind: str
    description: str
    recall_scale: float
    fp_scale: float
    quality_weights: tuple[float, float, float, float]
    keyframe_weights: tuple[float, float, float]
    max_failures: int
    confidence_bias: float
    cost_index: float = 1.0
    runtime_index: float = 1.0


MAIN_METHOD_PROFILES: list[Profile] = [
    Profile(
        name="random",
        category="main_methods",
        kind="detector",
        description="Random lower-bound baseline.",
        recall_scale=0.08,
        fp_scale=0.28,
        quality_weights=(0.05, 0.10, 0.25, 0.60),
        keyframe_weights=(0.08, 0.14, 0.78),
        max_failures=3,
        confidence_bias=-0.18,
        cost_index=0.3,
        runtime_index=0.4,
    ),
    Profile(
        name="rule_only",
        category="main_methods",
        kind="detector",
        description="Stage-1 rule engine only; moderate F1 and weak localization.",
        recall_scale=0.66,
        fp_scale=0.20,
        quality_weights=(0.16, 0.28, 0.32, 0.24),
        keyframe_weights=(0.28, 0.25, 0.47),
        max_failures=3,
        confidence_bias=-0.04,
        cost_index=1.0,
        runtime_index=1.0,
    ),
    Profile(
        name="rule_dummy_judge",
        category="main_methods",
        kind="detector",
        description="Judge framework with dummy verifier; nearly identical to rule-only.",
        recall_scale=0.67,
        fp_scale=0.195,
        quality_weights=(0.17, 0.29, 0.31, 0.23),
        keyframe_weights=(0.29, 0.25, 0.46),
        max_failures=3,
        confidence_bias=-0.03,
        cost_index=1.08,
        runtime_index=1.08,
    ),
    Profile(
        name="rule_open_vlm",
        category="main_methods",
        kind="detector",
        description="Rule engine plus open VLM judge.",
        recall_scale=0.80,
        fp_scale=0.13,
        quality_weights=(0.34, 0.35, 0.20, 0.11),
        keyframe_weights=(0.46, 0.28, 0.26),
        max_failures=3,
        confidence_bias=0.04,
        cost_index=1.8,
        runtime_index=1.55,
    ),
    Profile(
        name="rule_gpt4v",
        category="main_methods",
        kind="detector",
        description="Rule engine plus strongest VLM judge.",
        recall_scale=0.90,
        fp_scale=0.07,
        quality_weights=(0.46, 0.32, 0.15, 0.07),
        keyframe_weights=(0.58, 0.24, 0.18),
        max_failures=4,
        confidence_bias=0.10,
        cost_index=4.2,
        runtime_index=1.75,
    ),
]

EXTERNAL_BASELINE_PROFILES: list[Profile] = [
    Profile(
        name="vbench",
        category="external_baselines",
        kind="detector",
        description="General quality baseline with weak fault grounding.",
        recall_scale=0.30,
        fp_scale=0.18,
        quality_weights=(0.10, 0.18, 0.28, 0.44),
        keyframe_weights=(0.12, 0.18, 0.70),
        max_failures=2,
        confidence_bias=-0.08,
        cost_index=0.9,
        runtime_index=0.9,
    ),
    Profile(
        name="evalcrafter",
        category="external_baselines",
        kind="detector",
        description="General video evaluator closer to the task but not fault-native.",
        recall_scale=0.48,
        fp_scale=0.16,
        quality_weights=(0.14, 0.26, 0.30, 0.30),
        keyframe_weights=(0.20, 0.24, 0.56),
        max_failures=3,
        confidence_bias=-0.02,
        cost_index=1.3,
        runtime_index=1.1,
    ),
]

ABLATION_PROFILES: list[Profile] = [
    Profile(
        name="stride_1",
        category="ablations",
        kind="detector",
        description="Finer segmentation for stronger localization at higher cost.",
        recall_scale=0.68,
        fp_scale=0.20,
        quality_weights=(0.26, 0.30, 0.26, 0.18),
        keyframe_weights=(0.42, 0.27, 0.31),
        max_failures=3,
        confidence_bias=0.00,
        cost_index=1.5,
        runtime_index=1.9,
    ),
    Profile(
        name="stride_2",
        category="ablations",
        kind="detector",
        description="Default stride with best overall balance.",
        recall_scale=0.73,
        fp_scale=0.16,
        quality_weights=(0.24, 0.33, 0.27, 0.16),
        keyframe_weights=(0.37, 0.28, 0.35),
        max_failures=3,
        confidence_bias=0.00,
        cost_index=1.0,
        runtime_index=1.0,
    ),
    Profile(
        name="stride_4",
        category="ablations",
        kind="detector",
        description="Coarse stride with lower recall and weaker localization.",
        recall_scale=0.60,
        fp_scale=0.18,
        quality_weights=(0.11, 0.21, 0.31, 0.37),
        keyframe_weights=(0.23, 0.24, 0.53),
        max_failures=3,
        confidence_bias=-0.03,
        cost_index=0.8,
        runtime_index=0.62,
    ),
    Profile(
        name="K_1",
        category="ablations",
        kind="detector",
        description="Judge top-K set to 1; lower cost but misses weak candidates.",
        recall_scale=0.73,
        fp_scale=0.13,
        quality_weights=(0.28, 0.31, 0.24, 0.17),
        keyframe_weights=(0.41, 0.26, 0.33),
        max_failures=2,
        confidence_bias=0.01,
        cost_index=1.4,
        runtime_index=1.2,
    ),
    Profile(
        name="K_3",
        category="ablations",
        kind="detector",
        description="Judge top-K set to 3; best F1/cost trade-off.",
        recall_scale=0.79,
        fp_scale=0.13,
        quality_weights=(0.33, 0.34, 0.22, 0.11),
        keyframe_weights=(0.46, 0.28, 0.26),
        max_failures=3,
        confidence_bias=0.04,
        cost_index=1.8,
        runtime_index=1.55,
    ),
    Profile(
        name="K_5",
        category="ablations",
        kind="detector",
        description="Judge top-K set to 5; marginal gains at higher cost.",
        recall_scale=0.80,
        fp_scale=0.14,
        quality_weights=(0.34, 0.34, 0.21, 0.11),
        keyframe_weights=(0.47, 0.28, 0.25),
        max_failures=4,
        confidence_bias=0.03,
        cost_index=2.3,
        runtime_index=1.95,
    ),
    Profile(
        name="threshold_loose",
        category="ablations",
        kind="detector",
        description="Loose thresholds: better recall, worse precision.",
        recall_scale=0.77,
        fp_scale=0.30,
        quality_weights=(0.15, 0.25, 0.30, 0.30),
        keyframe_weights=(0.28, 0.25, 0.47),
        max_failures=3,
        confidence_bias=-0.02,
        cost_index=1.0,
        runtime_index=1.0,
    ),
    Profile(
        name="threshold_default",
        category="ablations",
        kind="detector",
        description="Default thresholds: strongest overall trade-off.",
        recall_scale=0.72,
        fp_scale=0.15,
        quality_weights=(0.24, 0.33, 0.27, 0.16),
        keyframe_weights=(0.36, 0.27, 0.37),
        max_failures=3,
        confidence_bias=0.00,
        cost_index=1.0,
        runtime_index=1.0,
    ),
    Profile(
        name="threshold_strict",
        category="ablations",
        kind="detector",
        description="Strict thresholds: higher precision, lower recall.",
        recall_scale=0.57,
        fp_scale=0.08,
        quality_weights=(0.24, 0.32, 0.26, 0.18),
        keyframe_weights=(0.38, 0.27, 0.35),
        max_failures=3,
        confidence_bias=0.02,
        cost_index=1.0,
        runtime_index=1.0,
    ),
]

CLOSED_LOOP_PROFILES: list[Profile] = [
    Profile(
        name="closed_loop_score_only",
        category="closed_loop",
        kind="closed_loop",
        description="Repair loop driven by score/report only.",
        recall_scale=0.66,
        fp_scale=0.20,
        quality_weights=(0.16, 0.28, 0.32, 0.24),
        keyframe_weights=(0.28, 0.25, 0.47),
        max_failures=3,
        confidence_bias=-0.04,
        cost_index=1.0,
        runtime_index=1.0,
    ),
    Profile(
        name="closed_loop_patch",
        category="closed_loop",
        kind="closed_loop",
        description="Repair loop with structured patch hints.",
        recall_scale=0.73,
        fp_scale=0.17,
        quality_weights=(0.24, 0.31, 0.26, 0.19),
        keyframe_weights=(0.36, 0.28, 0.36),
        max_failures=3,
        confidence_bias=0.01,
        cost_index=1.35,
        runtime_index=1.2,
    ),
    Profile(
        name="closed_loop_patch_judge",
        category="closed_loop",
        kind="closed_loop",
        description="Repair loop with patch hints and judge verification.",
        recall_scale=0.81,
        fp_scale=0.13,
        quality_weights=(0.32, 0.34, 0.22, 0.12),
        keyframe_weights=(0.46, 0.28, 0.26),
        max_failures=3,
        confidence_bias=0.05,
        cost_index=2.3,
        runtime_index=1.55,
    ),
]

ALL_PROFILES: list[Profile] = (
    MAIN_METHOD_PROFILES
    + EXTERNAL_BASELINE_PROFILES
    + ABLATION_PROFILES
    + CLOSED_LOOP_PROFILES
)

PROFILE_BY_NAME = {profile.name: profile for profile in ALL_PROFILES}

BASE_RECALL_BY_CODE = {
    "CA_MOVE_WRONG": 0.86,
    "ST_COMPRESSION_ARTIFACT": 0.78,
    "AL_PROP_MISSING": 0.78,
    "ID_FACE_DRIFT": 0.8,
    "ID_BODY_DRIFT": 0.68,
    "MO_FRAME_DROP": 0.69,
    "MO_JITTER": 0.62,
    "MO_SEGMENT_BREAK": 0.67,
    "MO_EVENT_MISSING": 0.48,
    "ST_COLOR_SHIFT": 0.4,
}

BASE_FP_POOL = [
    ("AL_PROP_MISSING", 0.24),
    ("ID_FACE_DRIFT", 0.08),
    ("MO_JITTER", 0.09),
    ("ST_COMPRESSION_ARTIFACT", 0.08),
    ("CA_SHAKE", 0.06),
]

BLOCKED_BY_PROFILE = {
    "rule_only": {"CA_MOVE_WRONG": 0.78, "ST_COMPRESSION_ARTIFACT": 0.76, "MO_FRAME_DROP": 0.74, "MO_SEGMENT_BREAK": 0.7, "ID_BODY_DRIFT": 0.72},
    "rule_dummy_judge": {"CA_MOVE_WRONG": 0.75, "ST_COMPRESSION_ARTIFACT": 0.74, "MO_FRAME_DROP": 0.72, "MO_SEGMENT_BREAK": 0.69, "ID_BODY_DRIFT": 0.7},
}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _suite_label(out_root: pathlib.Path) -> str:
    name = out_root.name
    if name == "simulated_experiments":
        return "simulated"
    if name.endswith("_simulated_experiments"):
        prefix = name[: -len("_simulated_experiments")]
        return f"{prefix}_simulated"
    return name.replace(" ", "_")


def _profile_layout(out_root: pathlib.Path, profile_name: str) -> dict[str, pathlib.Path]:
    suite_label = _suite_label(out_root)
    return {
        "suite_root": out_root,
        "profile_root": out_root / "profiles" / profile_name,
        "eval_root": out_root.parent / f"eval_{suite_label}_{profile_name}",
        "metrics_root": out_root.parent / "metrics" / suite_label / profile_name,
    }


def _code_templates(perturbation_type: str, variant: str) -> list[tuple[str, tuple[float, float], float]]:
    if perturbation_type == "remove_anchor":
        base = [("ID_FACE_DRIFT", (0.6, 2.5), 1.0), ("ID_BODY_DRIFT", (0.2, 1.0), 0.75)]
    elif perturbation_type == "drop_props_required":
        base = [("AL_PROP_MISSING", (0.2, 1.0), 0.8)]
    elif perturbation_type == "change_camera_movement":
        base = [("CA_MOVE_WRONG", (0.0, 12.0), 1.0)]
    elif perturbation_type == "extend_duration_or_merge":
        base = [("MO_SEGMENT_BREAK", (9.5, 13.5), 0.82), ("MO_EVENT_MISSING", (10.2, 13.7), 0.46)]
    elif perturbation_type == "temporal_jitter_or_frame_drop":
        base = [("MO_FRAME_DROP", (3.0, 5.5), 0.8), ("MO_JITTER", (2.5, 5.0), 0.56)]
    elif perturbation_type == "compression_artifacts":
        base = [("ST_COMPRESSION_ARTIFACT", (0.0, 12.0), 0.9), ("ST_COLOR_SHIFT", (1.5, 9.5), 0.35)]
    else:
        base = []

    if variant == "s1":
        return [(code, span, min(1.0, prob + 0.05)) for code, span, prob in base]
    return base


def _sample_keyframes(sample_id: str, code: str) -> list[str]:
    stem = f"{sample_id}__{code.lower()}"
    return [
        f"simulated_evidence/{stem}_k1.jpg",
        f"simulated_evidence/{stem}_k2.jpg",
        f"simulated_evidence/{stem}_k3.jpg",
    ]


def simulate_annotations(
    manifest_records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    annotations: list[dict[str, Any]] = []

    for record in manifest_records:
        sample_id = str(record["id"])
        parts = sample_id.split("__")
        perturbation_type = record.get("perturbation_type") or (parts[1] if len(parts) > 1 else "")
        variant = parts[2] if len(parts) > 2 else "s0"

        failures: list[dict[str, Any]] = []
        for code, (t0, t1), prob in _code_templates(str(perturbation_type), variant):
            if rng.random() > prob:
                continue
            at0 = max(0.0, round(t0 + rng.uniform(-0.12, 0.12), 3))
            at1 = max(at0 + 0.15, round(t1 + rng.uniform(-0.18, 0.18), 3))
            failures.append(
                {
                    "code": code,
                    "confidence": round(rng.uniform(0.92, 1.0), 3),
                    "verified": True,
                    "evidence": {
                        "t0": at0,
                        "t1": at1,
                        "keyframes": _sample_keyframes(sample_id, code),
                    },
                }
            )

        annotations.append(
            {
                "id": sample_id,
                "annotator_id": "simulated_demo_generator",
                "failure_codes": [failure["code"] for failure in failures],
                "top_failures": failures,
                "notes": "SIMULATED DEMO DATA ONLY. Not a real experiment result.",
            }
        )

    return annotations


def _profile_recall_prob(profile: Profile, code: str) -> float:
    base = BASE_RECALL_BY_CODE.get(code, 0.62) * profile.recall_scale
    damp = BLOCKED_BY_PROFILE.get(profile.name, {}).get(code, 1.0)
    return max(0.01, min(0.98, base * damp))


def _quality_mode(rng: random.Random, weights: tuple[float, float, float, float]) -> str:
    return rng.choices(
        population=["tight", "usable", "coarse", "borderline"],
        weights=weights,
        k=1,
    )[0]


def _perturb_span(rng: random.Random, gt0: float, gt1: float, quality_mode: str) -> tuple[float, float]:
    span_len = max(0.3, gt1 - gt0)
    if quality_mode == "tight":
        pt0 = max(0.0, round(gt0 + rng.uniform(-0.12, 0.12), 3))
        pt1 = max(pt0 + 0.1, round(gt1 + rng.uniform(-0.15, 0.15), 3))
    elif quality_mode == "usable":
        pt0 = max(0.0, round(gt0 + rng.uniform(-0.35, 0.28), 3))
        pt1 = max(pt0 + 0.1, round(gt1 + rng.uniform(-0.35, 0.35), 3))
    elif quality_mode == "coarse":
        pt0 = max(0.0, round(gt0 + rng.uniform(-0.85, 0.8), 3))
        pt1 = max(pt0 + 0.1, round(gt1 + rng.uniform(-0.8, 0.85), 3))
    else:
        shift = rng.uniform(0.45 * span_len, 0.8 * span_len)
        if rng.random() < 0.5:
            pt0 = max(0.0, round(gt0 + shift, 3))
            pt1 = max(pt0 + 0.1, round(gt1 + shift * rng.uniform(0.72, 1.08), 3))
        else:
            pt0 = max(0.0, round(gt0 - shift, 3))
            pt1 = max(pt0 + 0.1, round(gt1 - shift * rng.uniform(0.72, 1.08), 3))
    return pt0, pt1


def _sample_pred_keyframes(
    rng: random.Random,
    sample_id: str,
    code: str,
    gt_keyframes: list[str],
    weights: tuple[float, float, float],
) -> list[str]:
    if not gt_keyframes:
        return []
    exact = gt_keyframes[: min(3, len(gt_keyframes))]
    distractors = [
        f"simulated_distractor/{sample_id}_{code.lower()}_{idx}.jpg"
        for idx in range(1, 4)
    ]
    mode = rng.choices(
        population=["top1_hit", "top3_only", "miss_all"],
        weights=weights,
        k=1,
    )[0]
    if mode == "top1_hit":
        return exact[:]
    if mode == "top3_only":
        return [distractors[0], exact[0]] + exact[1:2]
    return distractors[:3]


def _sample_confidence(rng: random.Random, quality_mode: str, bias: float) -> float:
    base = {
        "tight": rng.uniform(0.80, 0.95),
        "usable": rng.uniform(0.68, 0.88),
        "coarse": rng.uniform(0.52, 0.78),
        "borderline": rng.uniform(0.38, 0.66),
    }[quality_mode]
    return round(max(0.05, min(0.99, base + bias)), 3)


def simulate_predictions(
    annotations: list[dict[str, Any]],
    manifest_records: list[dict[str, Any]],
    seed: int,
    profile: Profile | None = None,
) -> list[dict[str, Any]]:
    profile = profile or PROFILE_BY_NAME["rule_only"]
    rng = random.Random(seed + 1 + sum(ord(c) for c in profile.name))
    manifest_by_id = {record["id"]: record for record in manifest_records}
    predictions: list[dict[str, Any]] = []

    for ann in annotations:
        sample_id = ann["id"]
        pred_failures: list[dict[str, Any]] = []
        seen_codes: set[str] = set()

        for gt in ann.get("top_failures", []):
            code = gt["code"]
            if rng.random() > _profile_recall_prob(profile, code):
                continue

            ev = gt.get("evidence", {})
            gt0 = float(ev.get("t0", 0.0))
            gt1 = float(ev.get("t1", max(gt0 + 0.5, 1.0)))
            quality_mode = _quality_mode(rng, profile.quality_weights)
            pt0, pt1 = _perturb_span(rng, gt0, gt1, quality_mode)
            pred_keyframes = _sample_pred_keyframes(
                rng,
                sample_id,
                code,
                list(ev.get("keyframes", [])),
                profile.keyframe_weights,
            )
            pred_failures.append(
                {
                    "code": code,
                    "confidence": _sample_confidence(rng, quality_mode, profile.confidence_bias),
                    "evidence": {
                        "t0": pt0,
                        "t1": pt1,
                        "keyframes": pred_keyframes,
                    },
                }
            )
            seen_codes.add(code)

        for code, base_fp in BASE_FP_POOL:
            if code in seen_codes:
                continue
            fp_prob = max(0.0, min(0.95, base_fp * profile.fp_scale))
            if rng.random() > fp_prob:
                continue
            pred_failures.append(
                {
                    "code": code,
                    "confidence": round(max(0.08, min(0.72, rng.uniform(0.30, 0.58) + profile.confidence_bias / 2)), 3),
                    "evidence": {
                        "t0": round(rng.uniform(0.0, 8.0), 3),
                        "t1": round(rng.uniform(8.2, 12.5), 3),
                        "keyframes": [
                            f"simulated_fp/{sample_id}_{code.lower()}_1.jpg",
                            f"simulated_fp/{sample_id}_{code.lower()}_2.jpg",
                        ],
                    },
                }
            )

        pred_failures.sort(key=lambda item: (-item["confidence"], item["code"]))
        predictions.append(
            {
                "id": sample_id,
                "generator": f"simulated_{profile.name}",
                "video_path": manifest_by_id[sample_id].get("video_path"),
                "shotir_path": manifest_by_id[sample_id].get("shotir_path"),
                "top_failures": pred_failures[: profile.max_failures],
                "report_path": None,
            }
        )

    return predictions


def _derive_dummy_judge_predictions(
    base_predictions: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 991)
    predictions = copy.deepcopy(base_predictions)
    for record in predictions:
        record["generator"] = "simulated_rule_dummy_judge"
        for failure in record.get("top_failures", []):
            confidence = float(failure.get("confidence", 0.5))
            failure["confidence"] = round(max(0.05, min(0.99, confidence + rng.uniform(-0.015, 0.015))), 3)
    return predictions


def build_simulated_manifest(
    manifest_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ann_by_id = {record["id"]: record for record in annotations}
    out: list[dict[str, Any]] = []
    for record in manifest_records:
        updated = dict(record)
        ann = ann_by_id[record["id"]]
        updated["failure_codes"] = ann.get("failure_codes", [])
        meta = dict(updated.get("meta", {}) or {})
        meta["simulated"] = True
        meta["usage"] = ["algorithm_test", "paper_method_demo", "visualization_demo"]
        meta["not_for_real_results"] = True
        updated["meta"] = meta
        out.append(updated)
    return out


def _closed_loop_metrics(profile: Profile, summary: dict[str, Any]) -> dict[str, Any]:
    if profile.name == "closed_loop_score_only":
        return {
            "pass_at_1": 0.34,
            "pass_at_2": 0.48,
            "avg_iterations": 2.4,
            "time_to_usable": 16.8,
            "cost_per_usable_minute": 0.92,
        }
    if profile.name == "closed_loop_patch":
        return {
            "pass_at_1": 0.56,
            "pass_at_2": 0.74,
            "avg_iterations": 1.8,
            "time_to_usable": 12.1,
            "cost_per_usable_minute": 1.06,
        }
    return {
        "pass_at_1": 0.64,
        "pass_at_2": 0.82,
        "avg_iterations": 1.6,
        "time_to_usable": 10.9,
        "cost_per_usable_minute": 1.32,
    }


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _simulated_report_segments(top_failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, failure in enumerate(top_failures, start=1):
        evidence = dict(failure.get("evidence", {}) or {})
        t0 = float(evidence.get("t0", 0.0))
        t1 = float(evidence.get("t1", max(t0 + 0.5, 1.0)))
        keyframes = list(evidence.get("keyframes", []))
        segments.append(
            {
                "seg_id": f"sim_seg_{idx:03d}",
                "t_start": t0,
                "t_end": t1,
                "keyframes": keyframes,
                "features": {
                    "clip_drift": round(min(1.0, 0.2 + idx * 0.07), 3),
                    "flow_magnitude_mean": round(0.8 + idx * 0.45, 3),
                    "flow_magnitude_std": round(0.25 + idx * 0.12, 3),
                    "face_drift": None,
                    "has_face": str(failure.get("code", "")).startswith("ID_"),
                },
                "failures": [
                    {
                        "code": failure.get("code"),
                        "confidence": failure.get("confidence", 0.0),
                        "evidence": {
                            "t0": t0,
                            "t1": t1,
                            "keyframes": keyframes,
                        },
                    }
                ],
            }
        )
    return segments


def _simulated_patch_hints(top_failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for failure in top_failures:
        code = str(failure.get("code", "UNKNOWN"))
        evidence = dict(failure.get("evidence", {}) or {})
        hints.append(
            {
                "action": "review_failure_span",
                "field": "top_failures",
                "reason": f"Review failure span for {code}",
                "failure_code": code,
                "confidence": failure.get("confidence", 0.0),
                "evidence_t0": evidence.get("t0", 0.0),
                "evidence_t1": evidence.get("t1", 0.0),
            }
        )
    return hints


def _existing_keyframes(sample_root: pathlib.Path, keyframes: list[str]) -> list[str]:
    kept: list[str] = []
    for keyframe in keyframes:
        relative_path = pathlib.Path(str(keyframe))
        if (sample_root / relative_path).exists() or (REPO_ROOT / relative_path).exists():
            kept.append(str(keyframe))
    return kept


def _build_simulated_report(
    prediction: dict[str, Any],
    profile: Profile,
    sample_id: str,
    sample_root: pathlib.Path,
) -> dict[str, Any]:
    top_failures = copy.deepcopy(list(prediction.get("top_failures", [])))
    for failure in top_failures:
        evidence = dict(failure.get("evidence", {}) or {})
        evidence["keyframes"] = _existing_keyframes(
            sample_root,
            list(evidence.get("keyframes", [])),
        )
        failure["evidence"] = evidence
    segments = _simulated_report_segments(top_failures)
    for segment in segments:
        segment["keyframes"] = _existing_keyframes(sample_root, list(segment.get("keyframes", [])))
        for failure in segment.get("failures", []):
            evidence = dict(failure.get("evidence", {}) or {})
            evidence["keyframes"] = _existing_keyframes(
                sample_root,
                list(evidence.get("keyframes", [])),
            )
            failure["evidence"] = evidence
    duration_s = max(
        [float(f.get("evidence", {}).get("t1", 0.0)) for f in top_failures] + [12.0]
    )
    return {
        "video_meta": {
            "video_path": prediction.get("video_path"),
            "shotir_path": prediction.get("shotir_path"),
            "duration_s": round(duration_s, 3),
            "fps": 24.0,
            "resolution": {"width": 1280, "height": 720},
            "num_frames": max(1, int(round(duration_s * 24.0))),
            "run_id": profile.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "segments": segments,
        "scores": {
            "consistency": round(max(0.05, min(0.99, 0.92 - 0.12 * len(top_failures))), 4),
            "coherence": round(max(0.05, min(0.99, 0.9 - 0.1 * len(top_failures))), 4),
            "alignment": round(max(0.05, min(0.99, 0.88 - 0.1 * len(top_failures))), 4),
        },
        "top_failures": top_failures,
        "patch_hints": _simulated_patch_hints(top_failures),
    }


def _render_simulated_report_html(report: dict[str, Any], sample_id: str, profile: Profile) -> str:
    rows: list[str] = []
    for failure in report.get("top_failures", []):
        evidence = dict(failure.get("evidence", {}) or {})
        keyframes = " ".join(
            f"<code>{html.escape(str(path))}</code>"
            for path in evidence.get("keyframes", [])
        ) or "<span>n/a</span>"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(failure.get('code', '')))}</td>"
            f"<td>{float(failure.get('confidence', 0.0)):.3f}</td>"
            f"<td>{float(evidence.get('t0', 0.0)):.3f}</td>"
            f"<td>{float(evidence.get('t1', 0.0)):.3f}</td>"
            f"<td>{keyframes}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows) or "<tr><td colspan='5'>No predicted failures</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VideoEval Report - {html.escape(sample_id)}</title>
<style>
body {{ font-family: monospace; margin: 24px; background: #f7f7f5; color: #111; }}
h1, h2 {{ margin-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #efefef; }}
code {{ white-space: nowrap; }}
</style>
</head>
<body>
<h1>VideoEval Report</h1>
<p><strong>Sample:</strong> {html.escape(sample_id)}<br><strong>Video:</strong> {html.escape(str(report.get("video_meta", {}).get("video_path", "")))}<br><strong>Profile:</strong> {html.escape(profile.name)}</p>
<h2>Top Failures</h2>
<table>
<tr><th>Code</th><th>Confidence</th><th>t0</th><th>t1</th><th>Keyframes</th></tr>
{rows_html}
</table>
</body>
</html>
"""


def _write_simulated_sample_reports(
    eval_root: pathlib.Path,
    profile: Profile,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for prediction in predictions:
        record = copy.deepcopy(prediction)
        sample_id = str(record["id"])
        sample_root = eval_root / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        record["generator"] = profile.name
        sanitized_failures: list[dict[str, Any]] = []
        for failure in list(record.get("top_failures", [])):
            sanitized_failure = copy.deepcopy(failure)
            evidence = dict(sanitized_failure.get("evidence", {}) or {})
            evidence["keyframes"] = _existing_keyframes(
                sample_root,
                list(evidence.get("keyframes", [])),
            )
            sanitized_failure["evidence"] = evidence
            sanitized_failures.append(sanitized_failure)
        record["top_failures"] = sanitized_failures
        report = _build_simulated_report(record, profile, sample_id, sample_root)
        _write_json(sample_root / "report.json", report)
        (sample_root / "report.html").write_text(
            _render_simulated_report_html(report, sample_id, profile),
            encoding="utf-8",
        )
        record["report_path"] = (sample_root / "report.json").as_posix()
        rewritten.append(record)
    return rewritten


def refresh_eval_reports(eval_root: pathlib.Path, profile_name: str) -> None:
    profile = PROFILE_BY_NAME[profile_name]
    predictions_path = eval_root / "predictions.jsonl"
    predictions = load_jsonl(predictions_path)
    rewritten = _write_simulated_sample_reports(eval_root, profile, predictions)
    write_jsonl(predictions_path, rewritten)


def _write_closed_loop_metrics_files(metrics_root: pathlib.Path, metrics: dict[str, Any]) -> pathlib.Path:
    metrics_root.mkdir(parents=True, exist_ok=True)
    json_path = metrics_root / "closed_loop.json"
    csv_path = metrics_root / "closed_loop.csv"
    _write_json(json_path, metrics)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pass_at_1", "pass_at_2", "avg_iterations", "avg_time_s", "avg_cost_per_min"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pass_at_1": metrics.get("pass_at_1"),
                "pass_at_2": metrics.get("pass_at_2"),
                "avg_iterations": metrics.get("avg_iterations"),
                "avg_time_s": metrics.get("avg_time_s"),
                "avg_cost_per_min": metrics.get("avg_cost_per_min"),
            }
        )
    return json_path


def _write_profile_summary_and_readme(
    *,
    out_root: pathlib.Path,
    profile_root: pathlib.Path,
    eval_root: pathlib.Path,
    metrics_root: pathlib.Path,
    annotations_path: pathlib.Path,
    profile: Profile,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    annotations = load_jsonl(annotations_path)
    summary = _headline_summary(profile, metrics_root, annotations, predictions)
    summary["files"] = {
        "manifest": str((out_root / "manifest.jsonl").as_posix()),
        "annotations": str(annotations_path.as_posix()),
        "eval_dir": str(eval_root.as_posix()),
        "predictions": str((eval_root / "predictions.jsonl").as_posix()),
        "metrics_dir": str(metrics_root.as_posix()),
        "profile_dir": str(profile_root.as_posix()),
    }

    if profile.kind == "closed_loop":
        closed_loop_metrics = {
            "pass_at_1": summary["closed_loop_metrics"]["pass_at_1"],
            "pass_at_2": summary["closed_loop_metrics"]["pass_at_2"],
            "avg_iterations": summary["closed_loop_metrics"]["avg_iterations"],
            "avg_time_s": summary["closed_loop_metrics"]["time_to_usable"],
            "avg_cost_per_min": summary["closed_loop_metrics"]["cost_per_usable_minute"],
        }
        closed_loop_path = _write_closed_loop_metrics_files(metrics_root, closed_loop_metrics)
        summary["files"]["closed_loop"] = str(closed_loop_path.as_posix())

    (profile_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (profile_root / "README.txt").write_text(
        "SIMULATED EXPERIMENT DATA ONLY\n"
        "This directory is generated for algorithm tests, paper method demos, and visualization.\n"
        f"Sample-level outputs: {eval_root.as_posix()}\n"
        f"Metrics authority: {metrics_root.as_posix()}\n"
        "Do NOT report these metrics as real experimental results.\n",
        encoding="utf-8",
    )
    return summary


def _headline_summary(
    profile: Profile,
    metrics_dir: pathlib.Path,
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_metrics = _read_json(metrics_dir / "failure_code_f1.json")
    localization_metrics = _read_json(metrics_dir / "evidence_localization.json")
    positives = [record for record in annotations if record["failure_codes"]]
    avg_pred = statistics.mean(len(record.get("top_failures", [])) for record in predictions) if predictions else 0.0
    summary = {
        "simulated": True,
        "profile": profile.name,
        "category": profile.category,
        "description": profile.description,
        "not_for_real_results": True,
        "headline_metrics": {
            "macro_f1": failure_metrics.get("macro_f1"),
            "micro_f1": failure_metrics.get("micro_f1"),
            "tiou_at_03": localization_metrics.get("tiou_at_03"),
            "tiou_at_05": localization_metrics.get("tiou_at_05"),
            "top1_hit": localization_metrics.get("top1_hit"),
            "top3_hit": localization_metrics.get("top3_hit"),
        },
        "dataset_summary": {
            "n_samples": len(annotations),
            "n_positive_samples": len(positives),
            "avg_predicted_failures_per_sample": round(avg_pred, 4),
        },
        "system_cost": {
            "cost_index": profile.cost_index,
            "runtime_index": profile.runtime_index,
        },
    }
    if profile.kind == "closed_loop":
        summary["closed_loop_metrics"] = _closed_loop_metrics(profile, summary)
    return summary


def write_experiment_outputs(
    out_root: pathlib.Path,
    profile: Profile,
    annotations_path: pathlib.Path,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    layout = _profile_layout(out_root, profile.name)
    profile_root = layout["profile_root"]
    eval_root = layout["eval_root"]
    metrics_root = layout["metrics_root"]
    profile_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    rewritten_predictions = _write_simulated_sample_reports(eval_root, profile, predictions)
    predictions_path = eval_root / "predictions.jsonl"
    write_jsonl(predictions_path, rewritten_predictions)

    compute_f1(str(predictions_path), str(annotations_path), str(metrics_root))
    compute_localization(str(predictions_path), str(annotations_path), str(metrics_root))
    return _write_profile_summary_and_readme(
        out_root=out_root,
        profile_root=profile_root,
        eval_root=eval_root,
        metrics_root=metrics_root,
        annotations_path=annotations_path,
        profile=profile,
        predictions=rewritten_predictions,
    )


def _table_rows(summaries: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    rows = []
    by_name = {summary["profile"]: summary for summary in summaries}
    for name in names:
        summary = by_name[name]
        row = {"profile": name, **summary["headline_metrics"]}
        if "closed_loop_metrics" in summary:
            row.update(summary["closed_loop_metrics"])
        rows.append(row)
    return rows


def build_suite_tables(out_root: pathlib.Path, summaries: list[dict[str, Any]]) -> None:
    suite_summary = {
        "simulated": True,
        "not_for_real_results": True,
        "profiles": summaries,
    }
    (out_root / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_root / "table_main_methods.json").write_text(
        json.dumps(
            _table_rows(summaries, [p.name for p in MAIN_METHOD_PROFILES]),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_root / "table_external_baselines.json").write_text(
        json.dumps(_table_rows(summaries, [p.name for p in EXTERNAL_BASELINE_PROFILES]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_root / "table_ablations.json").write_text(
        json.dumps(_table_rows(summaries, [p.name for p in ABLATION_PROFILES]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_root / "table_closed_loop.json").write_text(
        json.dumps(_table_rows(summaries, [p.name for p in CLOSED_LOOP_PROFILES]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_experiment_suite(
    manifest_records: list[dict[str, Any]],
    out_root: pathlib.Path,
    seed: int,
) -> list[dict[str, Any]]:
    out_root.mkdir(parents=True, exist_ok=True)
    annotations = simulate_annotations(manifest_records, seed=seed)
    manifest_path = out_root / "manifest.jsonl"
    annotations_path = out_root / "annotations.jsonl"
    write_jsonl(manifest_path, build_simulated_manifest(manifest_records, annotations))
    write_jsonl(annotations_path, annotations)
    (out_root / "README.txt").write_text(
        "SIMULATED EXPERIMENT SUITE ONLY\n"
        "Shared manifest/annotations live in this directory.\n"
        "Sample-level outputs are written to out/eval_<suite>_<profile>/.\n"
        "Experiment-level metrics are written to out/metrics/<suite>/<profile>/.\n"
        "Do NOT report these metrics as real experimental results.\n",
        encoding="utf-8",
    )
    summaries: list[dict[str, Any]] = []
    rule_only_predictions: list[dict[str, Any]] | None = None

    for profile in ALL_PROFILES:
        if profile.name == "rule_dummy_judge" and rule_only_predictions is not None:
            predictions = _derive_dummy_judge_predictions(rule_only_predictions, seed=seed)
        else:
            predictions = simulate_predictions(annotations, manifest_records, seed=seed, profile=profile)
            if profile.name == "rule_only":
                rule_only_predictions = copy.deepcopy(predictions)
        summary = write_experiment_outputs(
            out_root=out_root,
            profile=profile,
            annotations_path=annotations_path,
            predictions=predictions,
        )
        summaries.append(summary)

    build_suite_tables(out_root, summaries)
    return summaries


def migrate_legacy_experiment_suite(out_root: pathlib.Path) -> list[str]:
    migrated: list[str] = []
    summaries: list[dict[str, Any]] = []
    legacy_dirs = sorted(
        path for path in out_root.iterdir()
        if path.is_dir() and path.name in PROFILE_BY_NAME and (path / "predictions.jsonl").exists()
    )
    if not legacy_dirs:
        return migrated

    out_root.mkdir(parents=True, exist_ok=True)
    profiles_root = out_root / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)

    shared_manifest = out_root / "manifest.jsonl"
    shared_annotations = out_root / "annotations.jsonl"
    if not shared_manifest.exists():
        shutil.copy2(legacy_dirs[0] / "manifest.jsonl", shared_manifest)
    if not shared_annotations.exists():
        shutil.copy2(legacy_dirs[0] / "annotations.jsonl", shared_annotations)

    for legacy_dir in legacy_dirs:
        profile_name = legacy_dir.name
        profile = PROFILE_BY_NAME[profile_name]
        layout = _profile_layout(out_root, profile_name)
        eval_root = layout["eval_root"]
        metrics_root = layout["metrics_root"]
        profile_root = layout["profile_root"]
        eval_root.mkdir(parents=True, exist_ok=True)
        metrics_root.mkdir(parents=True, exist_ok=True)
        profile_root.mkdir(parents=True, exist_ok=True)

        predictions = load_jsonl(legacy_dir / "predictions.jsonl")
        rewritten_predictions = _write_simulated_sample_reports(eval_root, profile, predictions)
        write_jsonl(eval_root / "predictions.jsonl", rewritten_predictions)

        legacy_metrics = legacy_dir / "metrics"
        if legacy_metrics.exists():
            for item in legacy_metrics.iterdir():
                if item.is_file():
                    shutil.copy2(item, metrics_root / item.name)

        closed_loop_legacy = legacy_dir / "closed_loop_metrics.json"
        if closed_loop_legacy.exists():
            closed_loop_metrics = _read_json(closed_loop_legacy)
            _write_closed_loop_metrics_files(
                metrics_root,
                {
                    "pass_at_1": closed_loop_metrics.get("pass_at_1"),
                    "pass_at_2": closed_loop_metrics.get("pass_at_2"),
                    "avg_iterations": closed_loop_metrics.get("avg_iterations"),
                    "avg_time_s": closed_loop_metrics.get("time_to_usable"),
                    "avg_cost_per_min": closed_loop_metrics.get("cost_per_usable_minute"),
                },
            )

        summary = _write_profile_summary_and_readme(
            out_root=out_root,
            profile_root=profile_root,
            eval_root=eval_root,
            metrics_root=metrics_root,
            annotations_path=shared_annotations,
            profile=profile,
            predictions=rewritten_predictions,
        )
        summaries.append(summary)

        shutil.rmtree(legacy_dir, ignore_errors=True)
        migrated.append(profile_name)

    (out_root / "README.txt").write_text(
        "SIMULATED EXPERIMENT SUITE ONLY\n"
        "Legacy per-profile mixed directories have been migrated.\n"
        "Shared manifest/annotations live in this directory.\n"
        "Sample-level outputs are written to out/eval_<suite>_<profile>/.\n"
        "Experiment-level metrics are written to out/metrics/<suite>/<profile>/.\n",
        encoding="utf-8",
    )
    build_suite_tables(out_root, summaries)
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simulated experiment suite.")
    parser.add_argument("--manifest", default=None, help="Base manifest JSONL")
    parser.add_argument("--out-root", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=20260419, help="Random seed")
    parser.add_argument(
        "--migrate-legacy-layout",
        action="store_true",
        help="Move legacy out/simulated_experiments/<profile> outputs into eval_/metrics/ structure.",
    )
    parser.add_argument(
        "--refresh-eval-dir",
        default=None,
        help="Rewrite report.json/report.html under an existing eval directory.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile name used with --refresh-eval-dir, e.g. rule_gpt4v or K_3.",
    )
    args = parser.parse_args()

    out_root = pathlib.Path(args.out_root)
    if args.refresh_eval_dir:
        if not args.profile:
            raise SystemExit("--profile is required with --refresh-eval-dir.")
        refresh_eval_reports(pathlib.Path(args.refresh_eval_dir), args.profile)
        print(f"Refreshed eval reports in {args.refresh_eval_dir}")
        return

    if args.migrate_legacy_layout:
        migrated = migrate_legacy_experiment_suite(out_root)
        print(f"Migrated {len(migrated)} legacy simulated profiles under {args.out_root}")
        if not args.manifest:
            return

    if not args.manifest:
        raise SystemExit("--manifest is required unless --migrate-legacy-layout is used alone.")

    manifest_records = load_jsonl(pathlib.Path(args.manifest))
    summaries = generate_experiment_suite(
        manifest_records=manifest_records,
        out_root=out_root,
        seed=args.seed,
    )
    print(f"Simulated experiment suite written to {args.out_root} ({len(summaries)} profiles)")


if __name__ == "__main__":
    main()
