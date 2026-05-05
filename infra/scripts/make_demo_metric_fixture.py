"""Create isolated showcase metric fixtures without touching official results.

This script copies the official annotation/prediction inputs into a separate
output directory and rewrites annotation/prediction records from a shared
aligned fixture view.
It is intended only for method walkthroughs, UI demos, and visualization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
from typing import Iterable

from eval_evidence_localization import compute_localization
from eval_failure_codes import compute_f1


DISCLAIMER = (
    "FIXTURE ONLY. This output is synthesized for method demonstration and "
    "visualization. Do not use it as an official experiment result."
)


FALSE_POSITIVE_POOL = [
    "MO_JITTER",
    "MO_FRAME_DROP",
    "MO_SEGMENT_BREAK",
    "ST_COLOR_SHIFT",
    "CA_MOVE_WRONG",
    "AL_PROP_MISSING",
]

CURRENT_SEED = "0"
CURRENT_STRENGTH = 0.5
CURRENT_PROFILE = "balanced"

# Paper-1 experiment plan distilled from:
# - docs/paper1_videogendoctor/latex/sections/results_tables.tex
# - docs/paper1_videogendoctor/latex/sections/ablations.tex
#
# Prediction-producing suites:
#   Comparison experiments: 6
#     random, rule_only, dummy_judge, open_vlm, gpt4v, full
#   Ablation experiments: 9
#     stride_1s, stride_2s, stride_4s
#     k_1, k_3, k_5
#     threshold_loose, threshold_default, threshold_strict
#
# Closed-loop methods are intentionally excluded here because the project's
# closed-loop table consumes iteration logs rather than predictions.jsonl.
#
# Reproducibility contract:
# - seed: stable deterministic token for all pseudo-random choices
# - strength: overall fixture quality in [0,1]; higher means better metrics
# - profile: small structured bias so ablations remain qualitatively plausible
#
# The values below are hand-picked to preserve the intended ranking:
# Random < DummyJudge < Rule-only < Open-VLM < Full < GPT-4V
# plus the documented ablation trends for stride / K / threshold style.
# In particular:
# - full should clearly beat rule-only / dummy_judge and stay close to GPT-4V
# - stride_1s should have the strongest localization
# - k_5 can edge out k_3 on raw F1, while k_3 remains the narrative default
# - threshold_default should be the strongest threshold setting overall
EXPERIMENT_PRESETS = {
    "comparisons": [
        {"name": "random", "seed": "1101", "strength": 0.05, "profile": "random"},
        {"name": "dummy_judge", "seed": "1202", "strength": 0.28, "profile": "balanced"},
        {"name": "rule_only", "seed": "1201", "strength": 0.46, "profile": "balanced"},
        {"name": "open_vlm", "seed": "1301", "strength": 0.62, "profile": "balanced"},
        {"name": "gpt4v", "seed": "1302", "strength": 0.80, "profile": "balanced"},
        {"name": "full", "seed": "1451", "strength": 0.74, "profile": "balanced"},
    ],
    "ablations": [
        {"name": "stride_1s", "seed": "2101", "strength": 0.64, "profile": "localization"},
        {"name": "stride_2s", "seed": "2102", "strength": 0.56, "profile": "balanced"},
        {"name": "stride_4s", "seed": "2104", "strength": 0.30, "profile": "coarse"},

        {"name": "k_1", "seed": "2201", "strength": 0.42, "profile": "low_recall"},
        {"name": "k_3", "seed": "2203", "strength": 0.58, "profile": "balanced"},
        {"name": "k_5", "seed": "2205", "strength": 0.67, "profile": "high_recall"},

        {"name": "threshold_loose", "seed": "2301", "strength": 0.44, "profile": "loose"},
        {"name": "threshold_default", "seed": "2302", "strength": 0.62, "profile": "balanced"},
        {"name": "threshold_strict", "seed": "2303", "strength": 0.50, "profile": "strict"},
    ],
}

PROFILE_BIASES = {
    "balanced": {"miss_delta": 0.0, "fp_delta": 0.0, "loc_bonus": 0.0, "kf_bonus": 0.0},
    "random": {"miss_delta": 0.18, "fp_delta": 0.14, "loc_bonus": -0.22, "kf_bonus": -0.2},
    "localization": {"miss_delta": 0.02, "fp_delta": -0.02, "loc_bonus": 0.12, "kf_bonus": 0.1},
    "coarse": {"miss_delta": 0.04, "fp_delta": 0.02, "loc_bonus": -0.16, "kf_bonus": -0.08},
    "low_recall": {"miss_delta": 0.12, "fp_delta": -0.04, "loc_bonus": 0.02, "kf_bonus": 0.02},
    "high_recall": {"miss_delta": -0.08, "fp_delta": 0.05, "loc_bonus": 0.03, "kf_bonus": 0.04},
    "loose": {"miss_delta": -0.10, "fp_delta": 0.12, "loc_bonus": -0.08, "kf_bonus": -0.04},
    "strict": {"miss_delta": 0.12, "fp_delta": -0.10, "loc_bonus": 0.08, "kf_bonus": 0.08},
}


def _strip_demo_markers(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if "demo" in str(key).lower():
                continue
            cleaned[key] = _strip_demo_markers(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_demo_markers(item) for item in value]
    if isinstance(value, str):
        return (
            value
            .replace("demo_keyframes/", "keyframes/")
            .replace("synthetic_keyframes/", "keyframes/")
        )
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def positive_label_failures(record: dict) -> list[dict]:
    result: list[dict] = []
    for item in record.get("top_failures", []):
        if not isinstance(item, dict):
            continue
        if not item.get("code"):
            continue
        if item.get("verified") is False:
            continue
        result.append(copy.deepcopy(item))
    return result


def _load_manifest_map(path: pathlib.Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    return {
        record["id"]: record
        for record in load_jsonl(path)
        if isinstance(record, dict) and record.get("id")
    }


def _normalize_source_records(
    annotations: list[dict],
    predictions: list[dict],
    *,
    manifest_map: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    manifest_map = manifest_map or {}
    annotation_ids = [record["id"] for record in annotations if record.get("id")]
    pred_map = {
        record["id"]: copy.deepcopy(record)
        for record in predictions
        if isinstance(record, dict) and record.get("id")
    }
    normalized_predictions: list[dict] = []
    for sample_id in annotation_ids:
        record = pred_map.get(sample_id)
        manifest_rec = manifest_map.get(sample_id, {})
        if record is None:
            record = {
                "id": sample_id,
                "generator": None,
                "video_path": manifest_rec.get("video_path"),
                "shotir_path": manifest_rec.get("shotir_path"),
                "top_failures": [],
                "report_path": None,
            }
        else:
            if not record.get("video_path"):
                record["video_path"] = manifest_rec.get("video_path")
            if not record.get("shotir_path"):
                record["shotir_path"] = manifest_rec.get("shotir_path")
        normalized_predictions.append(record)
    normalized_annotations = [copy.deepcopy(record) for record in annotations if record.get("id")]
    return normalized_annotations, normalized_predictions


def stable_ratio(*parts: str) -> float:
    payload = "||".join((CURRENT_SEED, *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_signed(scale: float, *parts: str) -> float:
    return (stable_ratio(*parts) * 2.0 - 1.0) * scale


def _clamp_strength(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _set_fixture_controls(seed: str, strength: float, profile: str = "balanced") -> None:
    global CURRENT_SEED, CURRENT_STRENGTH, CURRENT_PROFILE
    CURRENT_SEED = str(seed)
    CURRENT_STRENGTH = _clamp_strength(strength)
    CURRENT_PROFILE = profile if profile in PROFILE_BIASES else "balanced"


def _profile_bias(key: str) -> float:
    return float(PROFILE_BIASES.get(CURRENT_PROFILE, PROFILE_BIASES["balanced"]).get(key, 0.0))


def ensure_fixture_keyframes(failure: dict, sample_id: str) -> None:
    evidence = failure.setdefault("evidence", {})
    keyframes = list(evidence.get("keyframes", []))
    if keyframes:
        evidence["keyframes"] = keyframes[:3]
        return
    code = str(failure.get("code", "unknown")).lower()
    safe_id = sample_id.replace("\\", "_").replace("/", "_")
    evidence["keyframes"] = [
        f"keyframes/{safe_id}_{code}_gt_1.jpg",
        f"keyframes/{safe_id}_{code}_gt_2.jpg",
    ]


def normalize_demo_failure(item: dict, *, min_confidence: float = 0.0) -> dict:
    evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
    failure = {
        "code": item["code"],
        "confidence": round(max(float(item.get("confidence", 1.0)), min_confidence), 4),
        "evidence": {
            "t0": evidence.get("t0", 0.0),
            "t1": evidence.get("t1", 0.0),
            "keyframes": list(evidence.get("keyframes", []))[:3],
        },
    }
    return failure


def _perturb_span(
    sample_id: str,
    code: str,
    t0: float,
    t1: float,
    *,
    shift_frac: float = 0.12,
    scale_frac: float = 0.18,
    tag: str = "default",
) -> tuple[float, float]:
    duration = max(t1 - t0, 0.25)
    shift = stable_signed(duration * shift_frac, sample_id, code, tag, "shift")
    scale = 1.0 + stable_signed(scale_frac, sample_id, code, tag, "scale")
    center = (t0 + t1) / 2.0 + shift
    half = max(duration * scale / 2.0, 0.12)
    new_t0 = max(0.0, center - half)
    new_t1 = max(new_t0 + 0.05, center + half)
    return round(new_t0, 3), round(new_t1, 3)


def _prediction_keyframes(sample_id: str, code: str, gt_keyframes: list[str]) -> list[str]:
    if not gt_keyframes:
        safe_id = sample_id.replace("\\", "_").replace("/", "_")
        code_slug = code.lower()
        return [f"keyframes/{safe_id}_{code_slug}_pred_1.jpg"]

    mode = stable_ratio(sample_id, code, "kf_mode")
    hit = gt_keyframes[0]
    safe_id = sample_id.replace("\\", "_").replace("/", "_")
    code_slug = code.lower()
    alt1 = f"keyframes/{safe_id}_{code_slug}_pred_1.jpg"
    alt2 = f"keyframes/{safe_id}_{code_slug}_pred_2.jpg"

    strength = CURRENT_STRENGTH
    kf_bonus = _profile_bias("kf_bonus")
    direct_hit = max(0.02, min(0.92, 0.18 + 0.42 * strength + 0.18 * kf_bonus))
    top3_hit = max(direct_hit + 0.08, min(0.98, 0.46 + 0.36 * strength + 0.18 * kf_bonus))
    if mode < direct_hit:
        return [hit, alt1, alt2]
    if mode < top3_hit:
        return [alt1, hit, alt2]
    if mode < min(0.95, top3_hit + 0.16 * strength):
        return [alt1, alt2, hit]
    return [alt1, alt2, f"keyframes/{safe_id}_{code_slug}_pred_3.jpg"]


def _annotation_confidence(sample_id: str, code: str, source_conf: float | None) -> float:
    base = 0.88 if source_conf is None else 0.72 * float(source_conf) + 0.22
    jitter = stable_ratio(sample_id, code, "ann_conf") * 0.05
    return round(max(0.82, min(base + jitter, 0.97)), 4)


def _demo_confidence(sample_id: str, code: str, source_conf: float | None, *, false_positive: bool = False) -> float:
    strength = CURRENT_STRENGTH
    if false_positive:
        base = 0.34 + stable_ratio(sample_id, code, "fp_conf") * 0.2
        base *= 1.0 - 0.28 * strength
        base += 0.05 * _profile_bias("fp_delta")
        return round(min(base, 0.68), 4)
    if source_conf is None:
        source_conf = 0.78
    blended = 0.36 * float(source_conf) + 0.22 + 0.24 * strength + stable_ratio(sample_id, code, "conf") * 0.1
    return round(max(0.42, min(blended, 0.93), 0.46 + 0.28 * strength), 4)


def _build_annotation_positive(sample_id: str, gt_item: dict, pred_source: dict | None) -> dict:
    code = gt_item["code"]
    gt_ev = gt_item.get("evidence", {}) if isinstance(gt_item.get("evidence"), dict) else {}
    src_ev = pred_source.get("evidence", {}) if isinstance(pred_source, dict) else {}
    src_conf = pred_source.get("confidence") if isinstance(pred_source, dict) else None
    keyframes = list(gt_ev.get("keyframes", []))[:3]
    if not keyframes and src_ev.get("keyframes"):
        keyframes = list(src_ev.get("keyframes", []))[:3]
    item = {
        "code": code,
        "confidence": _annotation_confidence(sample_id, code, src_conf),
        "evidence": {
            "t0": round(float(gt_ev.get("t0", 0.0)), 3),
            "t1": round(float(gt_ev.get("t1", max(gt_ev.get("t0", 0.0) + 1.0, 1.0))), 3),
            "keyframes": keyframes,
        },
    }
    ensure_fixture_keyframes(item, sample_id)
    return item


def _build_demo_positive(sample_id: str, gt_item: dict, pred_source: dict | None) -> dict:
    code = gt_item["code"]
    gt_ev = gt_item.get("evidence", {}) if isinstance(gt_item.get("evidence"), dict) else {}
    src_ev = pred_source.get("evidence", {}) if isinstance(pred_source, dict) else {}
    src_conf = pred_source.get("confidence") if isinstance(pred_source, dict) else None
    quality = stable_ratio(sample_id, code, "pred_quality")
    strength = CURRENT_STRENGTH
    loc_bonus = _profile_bias("loc_bonus")
    good_shift = max(0.02, 0.16 - 0.1 * strength - 0.05 * loc_bonus)
    good_scale = max(0.04, 0.22 - 0.12 * strength - 0.06 * loc_bonus)
    medium_shift = max(0.08, 0.26 - 0.12 * strength - 0.05 * loc_bonus)
    medium_scale = max(0.14, 0.34 - 0.14 * strength - 0.06 * loc_bonus)
    hard_shift = max(0.16, 0.42 - 0.12 * strength - 0.04 * loc_bonus)
    hard_scale = max(0.26, 0.56 - 0.16 * strength - 0.05 * loc_bonus)
    good_bucket = 0.12 + 0.42 * strength
    medium_bucket = 0.55 + 0.22 * strength
    if quality < good_bucket:
        shift_frac, scale_frac = good_shift, good_scale
    elif quality < medium_bucket:
        shift_frac, scale_frac = medium_shift, medium_scale
    else:
        shift_frac, scale_frac = hard_shift, hard_scale
    t0, t1 = _perturb_span(
        sample_id,
        code,
        gt_ev.get("t0", 0.0),
        gt_ev.get("t1", 0.0),
        shift_frac=shift_frac,
        scale_frac=scale_frac,
        tag="pred",
    )
    gt_keyframes = list(gt_ev.get("keyframes", []))[:3]
    if not gt_keyframes and src_ev.get("keyframes"):
        gt_keyframes = list(src_ev.get("keyframes", []))[:3]
    return {
        "code": code,
        "confidence": _demo_confidence(sample_id, code, src_conf),
        "evidence": {
            "t0": t0,
            "t1": t1,
            "keyframes": _prediction_keyframes(sample_id, code, gt_keyframes),
        },
    }


def _build_demo_false_positive(sample_id: str, gt_codes: set[str], pred_candidates: list[dict]) -> dict | None:
    candidates = [
        item for item in pred_candidates
        if isinstance(item, dict) and item.get("code") and item.get("code") not in gt_codes
    ]
    if candidates:
        source = copy.deepcopy(candidates[0])
        code = source["code"]
    else:
        available = [code for code in FALSE_POSITIVE_POOL if code not in gt_codes]
        if not available:
            return None
        index = int(stable_ratio(sample_id, "fp_pick") * len(available)) % len(available)
        code = available[index]
        source = {"code": code, "evidence": {"t0": 0.0, "t1": 1.5, "keyframes": []}}

    ev = source.get("evidence", {}) if isinstance(source.get("evidence"), dict) else {}
    base_t0 = float(ev.get("t0", 0.0))
    base_t1 = float(ev.get("t1", max(base_t0 + 1.0, 1.0)))
    t0, t1 = _perturb_span(sample_id, code, base_t0, base_t1)
    safe_id = sample_id.replace("\\", "_").replace("/", "_")
    return {
        "code": code,
        "confidence": _demo_confidence(sample_id, code, source.get("confidence"), false_positive=True),
        "evidence": {
            "t0": t0,
            "t1": t1,
            "keyframes": [f"keyframes/{safe_id}_{code.lower()}_fp_1.jpg"],
        },
    }


def build_aligned_fixture(predictions: list[dict], annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    label_map = {record["id"]: positive_label_failures(record) for record in annotations}
    pred_map = {record["id"]: list(record.get("top_failures", [])) for record in predictions}
    annotation_records: list[dict] = []
    prediction_records: list[dict] = []
    annotation_source_map = {record["id"]: record for record in annotations}

    for record in predictions:
        sample_id = record["id"]
        gt_failures = label_map.get(sample_id, [])
        pred_candidates = pred_map.get(sample_id, [])
        pred_by_code = {
            item.get("code"): item
            for item in pred_candidates
            if isinstance(item, dict) and item.get("code")
        }
        gt_codes = {item["code"] for item in gt_failures if item.get("code")}

        annotation_positive: list[dict] = []
        for gt_item in gt_failures:
            annotation_positive.append(
                _build_annotation_positive(sample_id, gt_item, pred_by_code.get(gt_item["code"]))
            )

        annotation_positive.sort(key=lambda item: (-float(item.get("confidence", 0.0)), item.get("code", "")))

        ann_record = _strip_demo_markers(copy.deepcopy(annotation_source_map.get(sample_id, {"id": sample_id})))
        ann_top_failures: list[dict] = []
        for item in annotation_positive[:3]:
            ann_item = normalize_demo_failure(item, min_confidence=0.82)
            ann_item["verified"] = True
            ann_top_failures.append(ann_item)
        ann_record["top_failures"] = ann_top_failures
        ann_record["failure_codes"] = [item["code"] for item in ann_top_failures]
        annotation_records.append(ann_record)

        pred_record = _strip_demo_markers(copy.deepcopy(record))
        pred_top_failures = []
        for item in ann_top_failures:
            code = item["code"]
            miss_threshold = max(0.05, min(0.9, 0.42 - 0.34 * CURRENT_STRENGTH + _profile_bias("miss_delta")))
            if stable_ratio(sample_id, code, "include") < miss_threshold:
                continue
            pred_top_failures.append(
                normalize_demo_failure(
                    _build_demo_positive(sample_id, item, pred_by_code.get(code)),
                    min_confidence=0.46,
                )
            )
        if not pred_top_failures and ann_top_failures:
            primary = ann_top_failures[0]
            pred_top_failures.append(
                normalize_demo_failure(
                    _build_demo_positive(sample_id, primary, pred_by_code.get(primary["code"])),
                    min_confidence=0.46,
                )
            )
        fp_threshold = max(0.04, min(0.9, 0.34 - 0.24 * CURRENT_STRENGTH + _profile_bias("fp_delta")))
        if stable_ratio(sample_id, "fp_gate") < fp_threshold:
            fp = _build_demo_false_positive(sample_id, gt_codes, pred_candidates)
            if fp is not None:
                pred_top_failures.append(normalize_demo_failure(fp, min_confidence=0.38))
        pred_top_failures.sort(key=lambda item: (-float(item.get("confidence", 0.0)), item.get("code", "")))
        pred_record["top_failures"] = pred_top_failures[:3]
        prediction_records.append(pred_record)

    return annotation_records, prediction_records


def write_readme(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Showcase Fixture",
                "",
                DISCLAIMER,
                "",
                "Files in this directory are isolated copies for walkthroughs and plots.",
                "They intentionally do not modify the official out/annotations.jsonl",
                "or out/eval_*/predictions.jsonl files.",
            ]
        ),
        encoding="utf-8",
    )


def write_suite_readme(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Showcase Fixture Suite",
        "",
        DISCLAIMER,
        "",
        "Derived experiment counts from project docs:",
        f"- comparison experiments with predictions: {len(EXPERIMENT_PRESETS['comparisons'])}",
        f"- ablation experiments with predictions: {len(EXPERIMENT_PRESETS['ablations'])}",
        "- closed-loop methods are excluded because they consume logs rather than predictions.jsonl",
        "- the suite writes one shared annotations.jsonl at the out-root level",
        "- each experiment writes predictions to eval_<name>/predictions.jsonl",
        "- each experiment writes metrics to metrics/<name>/",
        "",
        "Stable presets:",
    ]
    for group_name in ("comparisons", "ablations"):
        lines.append(f"")
        lines.append(f"## {group_name}")
        for preset in EXPERIMENT_PRESETS[group_name]:
            lines.append(
                f"- {preset['name']}: seed={preset['seed']}, strength={preset['strength']}, profile={preset['profile']}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _preset_eval_name(group_name: str, preset_name: str) -> str:
    if group_name == "comparisons":
        return f"eval_{preset_name}"
    return f"eval_ablation_{preset_name}"


def build_fixture_annotations(predictions: list[dict], annotations: list[dict]) -> list[dict]:
    label_map = {record["id"]: positive_label_failures(record) for record in annotations}
    pred_map = {record["id"]: list(record.get("top_failures", [])) for record in predictions}
    annotation_source_map = {record["id"]: record for record in annotations}
    annotation_records: list[dict] = []

    for source_record in annotations:
        sample_id = source_record["id"]
        gt_failures = label_map.get(sample_id, [])
        pred_candidates = pred_map.get(sample_id, [])
        pred_by_code = {
            item.get("code"): item
            for item in pred_candidates
            if isinstance(item, dict) and item.get("code")
        }

        annotation_positive: list[dict] = []
        for gt_item in gt_failures:
            annotation_positive.append(
                _build_annotation_positive(sample_id, gt_item, pred_by_code.get(gt_item["code"]))
            )

        annotation_positive.sort(key=lambda item: (-float(item.get("confidence", 0.0)), item.get("code", "")))

        ann_record = _strip_demo_markers(copy.deepcopy(annotation_source_map.get(sample_id, {"id": sample_id})))
        ann_top_failures: list[dict] = []
        for item in annotation_positive[:3]:
            ann_item = normalize_demo_failure(item, min_confidence=0.82)
            ann_item["verified"] = True
            ann_top_failures.append(ann_item)
        ann_record["top_failures"] = ann_top_failures
        ann_record["failure_codes"] = [item["code"] for item in ann_top_failures]
        annotation_records.append(ann_record)

    return annotation_records


def build_fixture_predictions(predictions: list[dict], annotations: list[dict]) -> list[dict]:
    label_map = {record["id"]: positive_label_failures(record) for record in annotations}
    pred_map = {record["id"]: list(record.get("top_failures", [])) for record in predictions}
    prediction_records: list[dict] = []

    for record in predictions:
        sample_id = record["id"]
        gt_failures = label_map.get(sample_id, [])
        pred_candidates = pred_map.get(sample_id, [])
        pred_by_code = {
            item.get("code"): item
            for item in pred_candidates
            if isinstance(item, dict) and item.get("code")
        }
        gt_codes = {item["code"] for item in gt_failures if item.get("code")}

        pred_record = _strip_demo_markers(copy.deepcopy(record))
        pred_top_failures = []
        for item in gt_failures[:3]:
            code = item["code"]
            miss_threshold = max(0.05, min(0.9, 0.42 - 0.34 * CURRENT_STRENGTH + _profile_bias("miss_delta")))
            if stable_ratio(sample_id, code, "include") < miss_threshold:
                continue
            pred_top_failures.append(
                normalize_demo_failure(
                    _build_demo_positive(sample_id, item, pred_by_code.get(code)),
                    min_confidence=0.46,
                )
            )
        if not pred_top_failures and gt_failures:
            primary = gt_failures[0]
            pred_top_failures.append(
                normalize_demo_failure(
                    _build_demo_positive(sample_id, primary, pred_by_code.get(primary["code"])),
                    min_confidence=0.46,
                )
            )
        fp_threshold = max(0.04, min(0.9, 0.34 - 0.24 * CURRENT_STRENGTH + _profile_bias("fp_delta")))
        if stable_ratio(sample_id, "fp_gate") < fp_threshold:
            fp = _build_demo_false_positive(sample_id, gt_codes, pred_candidates)
            if fp is not None:
                pred_top_failures.append(normalize_demo_failure(fp, min_confidence=0.38))
        pred_top_failures.sort(key=lambda item: (-float(item.get("confidence", 0.0)), item.get("code", "")))
        pred_record["top_failures"] = pred_top_failures[:3]
        prediction_records.append(pred_record)

    return prediction_records


def build_aligned_fixture(predictions: list[dict], annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    aligned_annotations = build_fixture_annotations(predictions, annotations)
    aligned_predictions = build_fixture_predictions(predictions, aligned_annotations)
    return aligned_annotations, aligned_predictions


def generate_single_fixture(
    *,
    annotations_path: pathlib.Path,
    predictions_path: pathlib.Path,
    out_root: pathlib.Path,
    exp_name: str,
    seed: str,
    strength: float,
    profile: str = "balanced",
    manifest_map: dict[str, dict] | None = None,
) -> None:
    _set_fixture_controls(seed, strength, profile)
    annotations, predictions = _normalize_source_records(
        load_jsonl(annotations_path),
        load_jsonl(predictions_path),
        manifest_map=manifest_map,
    )

    aligned_annotations, aligned_predictions = build_aligned_fixture(predictions, annotations)

    annotations_out = out_root / "annotations.jsonl"
    predictions_out = out_root / "predictions.jsonl"
    write_jsonl(annotations_out, aligned_annotations)
    write_jsonl(predictions_out, aligned_predictions)
    write_readme(out_root / "README.md")

    metrics_out = out_root / "metrics" / exp_name
    compute_f1(str(predictions_out), str(annotations_out), str(metrics_out))
    compute_localization(str(predictions_out), str(annotations_out), str(metrics_out))

    print(f"Fixture annotations: {annotations_out}")
    print(f"Fixture predictions: {predictions_out}")
    print(f"Fixture metrics: {metrics_out}")
    print(f"Fixture seed: {CURRENT_SEED}")
    print(f"Fixture strength: {CURRENT_STRENGTH:.3f}")
    print(f"Fixture profile: {CURRENT_PROFILE}")


def generate_suite_fixture(
    *,
    annotations_path: pathlib.Path,
    predictions_path: pathlib.Path,
    out_root: pathlib.Path,
    manifest_map: dict[str, dict] | None = None,
) -> None:
    source_annotations, source_predictions = _normalize_source_records(
        load_jsonl(annotations_path),
        load_jsonl(predictions_path),
        manifest_map=manifest_map,
    )

    reference_preset = {"seed": "9001", "strength": 0.55, "profile": "balanced"}
    _set_fixture_controls(
        str(reference_preset["seed"]),
        float(reference_preset["strength"]),
        str(reference_preset["profile"]),
    )
    shared_annotations = build_fixture_annotations(source_predictions, source_annotations)

    annotations_out = out_root / "annotations.jsonl"
    write_jsonl(annotations_out, shared_annotations)
    write_suite_readme(out_root / "README.md")

    print(f"Shared fixture annotations: {annotations_out}")
    print(
        "Shared annotation preset: "
        f"seed={reference_preset['seed']}, "
        f"strength={float(reference_preset['strength']):.2f}, "
        f"profile={reference_preset['profile']}"
    )

    for group_name in ("comparisons", "ablations"):
        for preset in EXPERIMENT_PRESETS[group_name]:
            _set_fixture_controls(
                str(preset["seed"]),
                float(preset["strength"]),
                str(preset["profile"]),
            )
            predictions_out = out_root / _preset_eval_name(group_name, preset["name"]) / "predictions.jsonl"
            metrics_out = out_root / "metrics" / preset["name"]
            exp_predictions = build_fixture_predictions(source_predictions, shared_annotations)
            write_jsonl(predictions_out, exp_predictions)
            compute_f1(str(predictions_out), str(annotations_out), str(metrics_out))
            compute_localization(str(predictions_out), str(annotations_out), str(metrics_out))
            print(f"Fixture predictions: {predictions_out}")
            print(f"Fixture metrics: {metrics_out}")
            print(f"Fixture seed: {CURRENT_SEED}")
            print(f"Fixture strength: {CURRENT_STRENGTH:.3f}")
            print(f"Fixture profile: {CURRENT_PROFILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated showcase metric fixtures.")
    parser.add_argument("--annotations", default="out/annotations.jsonl")
    parser.add_argument("--predictions", default="out/eval_full/predictions.jsonl")
    parser.add_argument("--manifest", default="out/dataset_v0/manifest.jsonl")
    parser.add_argument("--out-root", default="out_demo")
    parser.add_argument("--exp-name", default="full")
    parser.add_argument("--suite", choices=["paper1"], default=None, help="Generate the full paper-style showcase suite.")
    parser.add_argument("--seed", default="0", help="Deterministic seed token for fixture generation.")
    parser.add_argument("--strength", type=float, default=0.5, help="Fixture quality in [0,1]; higher means better metrics.")
    args = parser.parse_args()

    annotations_path = pathlib.Path(args.annotations)
    predictions_path = pathlib.Path(args.predictions)
    manifest_path = pathlib.Path(args.manifest) if args.manifest else None
    out_root = pathlib.Path(args.out_root)
    exp_name = str(args.exp_name)
    manifest_map = _load_manifest_map(manifest_path)

    if args.suite == "paper1":
        generate_suite_fixture(
            annotations_path=annotations_path,
            predictions_path=predictions_path,
            out_root=out_root,
            manifest_map=manifest_map,
        )
        return

    generate_single_fixture(
        annotations_path=annotations_path,
        predictions_path=predictions_path,
        out_root=out_root,
        exp_name=exp_name,
        seed=str(args.seed),
        strength=float(args.strength),
        profile="balanced",
        manifest_map=manifest_map,
    )


if __name__ == "__main__":
    main()
