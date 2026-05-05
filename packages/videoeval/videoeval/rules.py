"""Stage-1 rule engine: applies threshold rules to segment features."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# Rules: (code, feature_key, comparator, threshold)
# comparator: 'gt' = greater than triggers failure
_RULES: list[tuple[str, str, str, float]] = [
    ("ID_FACE_DRIFT",   "face_drift",            "gt", 0.35),
    ("MO_JITTER",       "flow_magnitude_std",     "gt", 5.0),
    ("MO_FROZEN_FRAME", "flow_magnitude_mean",    "lt", 0.01),
    ("SC_BG_INCONSISTENCY", "clip_drift",         "gt", 0.20),
    ("MO_SEGMENT_BREAK",    "clip_drift",         "gt", 0.30),
    ("ST_COLOR_SHIFT",      "clip_drift",         "gt", 0.25),
    ("CA_SHAKE",             "flow_magnitude_std", "gt", 8.0),
]


def apply_rules(segments: list[dict], cfg: dict,
                shotir: dict | None = None) -> list[dict]:
    """Apply rule engine to all segments; annotate failures."""
    thresholds = {
        "face_drift_threshold":  cfg.get("face_drift_threshold", 0.35),
        "clip_drift_threshold":  cfg.get("clip_drift_threshold", 0.15),
        "flow_jitter_threshold": cfg.get("flow_jitter_threshold", 5.0),
        "flow_frozen_threshold": cfg.get("flow_frozen_threshold", 0.01),
        "flow_shake_threshold":  cfg.get("flow_shake_threshold", 8.0),
    }
    top_k = cfg.get("top_k_failures", 3)
    enable_prop_placeholders = bool(cfg.get("enable_prop_placeholders", False))

    for seg in segments:
        feats = seg.get("features", {})
        candidates = []

        for code, feat_key, cmp, base_thresh in _RULES:
            val = feats.get(feat_key)
            if val is None:
                continue
            # Allow config overrides
            thresh = thresholds.get(f"{feat_key.split('_')[0]}_threshold", base_thresh)
            thresh = base_thresh  # use rule default; config thresholds for known keys
            if feat_key == "face_drift":
                thresh = thresholds["face_drift_threshold"]
            elif feat_key == "clip_drift":
                thresh = thresholds["clip_drift_threshold"]
            elif feat_key == "flow_magnitude_mean":
                thresh = thresholds["flow_frozen_threshold"]
            elif feat_key == "flow_magnitude_std" and code == "CA_SHAKE":
                thresh = thresholds["flow_shake_threshold"]
            elif "flow" in feat_key and "std" in feat_key:
                thresh = thresholds["flow_jitter_threshold"]

            triggered = (cmp == "gt" and val > thresh) or \
                        (cmp == "lt" and val < thresh)
            if triggered:
                # Confidence: linear scale relative to threshold
                if cmp == "gt":
                    conf = min(1.0, (val - thresh) / (thresh + 1e-6) * 0.5 + 0.5)
                else:
                    conf = min(1.0, (thresh - val) / (thresh + 1e-6) * 0.5 + 0.5)
                # Evidence: segment span + top-2 keyframes by anomaly
                kfs = seg.get("keyframes", [])
                candidates.append({
                    "code": code,
                    "confidence": round(conf, 4),
                    "evidence": {
                        "t0": seg["t_start"],
                        "t1": seg["t_end"],
                        "keyframes": kfs[:2],
                    },
                })

        # ShotIR props check
        if shotir and enable_prop_placeholders:
            prop_failures = _check_shotir_props(seg, shotir)
            candidates.extend(prop_failures)

        # Sort by confidence, keep top-k
        candidates.sort(key=lambda x: -x["confidence"])
        seg["failures"] = candidates[:top_k]

    return segments


def _check_shotir_props(seg: dict, shotir: dict) -> list[dict]:
    """Simple props presence check using ShotIR spec."""
    props_required: list[str] = []
    shots = shotir.get("shots", [])
    # Find shot overlapping segment
    for shot in shots:
        s_start = shot.get("t_start", 0)
        s_end = shot.get("t_end", 9999)
        if not (s_start < seg["t_end"] and s_end > seg["t_start"]):
            continue
        props_required.extend(shot.get("props_required", []))

    if not props_required:
        return []

    # Placeholder mode: keep at most one low-confidence proposal per segment.
    return [{
        "code": "AL_PROP_MISSING",
        "confidence": 0.3,
        "evidence": {
            "t0": seg["t_start"],
            "t1": seg["t_end"],
            "keyframes": seg.get("keyframes", [])[:2],
        },
        "_props": sorted(set(props_required)),
    }]

