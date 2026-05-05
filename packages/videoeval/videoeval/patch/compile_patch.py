"""Patch compiler: report.json + optional shotir -> patch actions."""
from __future__ import annotations
import json
import pathlib
import logging

logger = logging.getLogger(__name__)

_PATCH_MAP_PATH = pathlib.Path(__file__).parent.parent / "taxonomy" / "patch_map_v0.1.json"


def _load_patch_map() -> dict:
    with open(_PATCH_MAP_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("patch_map", {})


def compile_patch(report: dict, shotir: dict | None = None) -> list[dict]:
    """Return list of structured patch actions from top_failures."""
    patch_map = _load_patch_map()
    top_failures = report.get("top_failures", [])
    actions = []
    seen_actions = set()

    for failure in top_failures:
        code = failure.get("code", "")
        conf = failure.get("confidence", 0.0)
        evidence = failure.get("evidence", {})
        templates = patch_map.get(code, [])

        if not templates:
            # Fallback generic action
            templates = [{
                "action": "regenerate_segment",
                "field": "segment",
                "reason": f"No specific patch for {code}; regenerate segment."
            }]

        for tmpl in templates:
            action = dict(tmpl)
            action["failure_code"] = code
            action["confidence"] = conf
            action["evidence_t0"] = evidence.get("t0")
            action["evidence_t1"] = evidence.get("t1")

            # ShotIR-aware: add shot reference if available
            if shotir:
                shot = _find_shot(shotir, evidence.get("t0", 0))
                if shot:
                    action["shot_id"] = shot.get("shot_id")

            key = (code, action["action"])
            if key not in seen_actions:
                actions.append(action)
                seen_actions.add(key)

    return actions


def write_patch_output(report: dict, shotir: dict | None,
                       out_dir: str) -> str:
    """Write shotir.diff.json or rerender_plan.json to out_dir."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    actions = compile_patch(report, shotir=shotir)

    if shotir:
        out_path = out / "shotir.diff.json"
        diff = {
            "schema_version": "0.1",
            "source_shotir": report.get("video_meta", {}).get("shotir_path"),
            "diffs": actions,
        }
        with open(out_path, "w") as f:
            json.dump(diff, f, indent=2)
    else:
        out_path = out / "rerender_plan.json"
        plan = {
            "schema_version": "0.1",
            "video_path": report.get("video_meta", {}).get("video_path"),
            "actions": actions,
        }
        with open(out_path, "w") as f:
            json.dump(plan, f, indent=2)

    logger.info("Patch output written to %s", out_path)
    return str(out_path)


def _find_shot(shotir: dict, t: float) -> dict | None:
    for shot in shotir.get("shots", []):
        if shot.get("t_start", 0) <= t < shot.get("t_end", 9999):
            return shot
    return None

