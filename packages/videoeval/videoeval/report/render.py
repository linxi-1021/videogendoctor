"""HTML report renderer using Jinja2."""
from __future__ import annotations
import pathlib
import logging
import os

logger = logging.getLogger(__name__)

_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VideoEval Report — {{ meta.video_path }}</title>
<style>
body { font-family: monospace; background: #0d1117; color: #e6edf3; margin: 2em; }
h1 { color: #58a6ff; } h2 { color: #79c0ff; border-bottom: 1px solid #30363d; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th { background: #161b22; color: #8b949e; padding: 6px 10px; text-align: left; }
td { padding: 5px 10px; border-top: 1px solid #21262d; font-size: 0.85em; }
.high { background: #3d1c1c; } .med { background: #2d2010; } .low { background: #0d2010; }
.code { color: #f0883e; font-weight: bold; } .conf { color: #79c0ff; }
.patch { color: #56d364; } .seg-header { color: #d2a8ff; }
img { max-width: 160px; max-height: 90px; margin: 2px;
      border: 1px solid #30363d; border-radius: 4px; }
.scores span { display: inline-block; margin-right: 2em; }
.score-val { color: #58a6ff; font-weight: bold; }
</style>
</head>
<body>
<h1>VideoEval Report</h1>
<p><b>Video:</b> {{ meta.video_path }}<br>
<b>Duration:</b> {{ meta.duration_s }}s &nbsp; <b>FPS:</b> {{ meta.fps }}<br>
<b>Run ID:</b> {{ meta.run_id }} &nbsp; <b>Time:</b> {{ meta.timestamp }}</p>

<h2>Scores</h2>
<div class="scores">
  <span>Consistency: <span class="score-val">{{ "%.3f"|format(scores.consistency) }}</span></span>
  <span>Coherence: <span class="score-val">{{ "%.3f"|format(scores.coherence) }}</span></span>
  <span>Alignment: <span class="score-val">{{ "%.3f"|format(scores.alignment) }}</span></span>
</div>

<h2>Top Failures</h2>
{% if top_failures %}
<table><tr><th>Code</th><th>Confidence</th><th>t0</th><th>t1</th><th>Keyframes</th></tr>
{% for f in top_failures %}
{% set row_class = 'high' if f.confidence > 0.7 else ('med' if f.confidence > 0.4 else 'low') %}
<tr class="{{ row_class }}">
  <td class="code">{{ f.code }}</td>
  <td class="conf">{{ "%.2f"|format(f.confidence) }}</td>
  <td>{{ f.evidence.t0 }}s</td>
  <td>{{ f.evidence.t1 }}s</td>
  <td>{% for kf in f.evidence.keyframes[:2] %}<img src="{{ kf }}" title="{{ kf }}">{% endfor %}</td>
</tr>
{% endfor %}
</table>
{% else %}<p>No failures detected.</p>{% endif %}

<h2>Patch Hints</h2>
{% if patch_hints %}
<table><tr><th>Action</th><th>Field</th><th>Failure Code</th><th>Reason</th></tr>
{% for p in patch_hints %}
<tr><td class="patch">{{ p.action }}</td><td>{{ p.get('field','') }}</td>
    <td class="code">{{ p.get('failure_code','') }}</td><td>{{ p.reason }}</td></tr>
{% endfor %}
</table>
{% else %}<p>No patch hints.</p>{% endif %}

<h2>Segments ({{ segments|length }})</h2>
<table><tr><th>ID</th><th>t_start</th><th>t_end</th><th>CLIP Drift</th>
<th>Flow Mean</th><th>Face Drift</th><th>Failures</th><th>Keyframes</th></tr>
{% for seg in segments %}
{% set has_fail = seg.failures|length > 0 %}
<tr class="{{ 'med' if has_fail else '' }}">
  <td class="seg-header">{{ seg.seg_id }}</td>
  <td>{{ seg.t_start }}s</td><td>{{ seg.t_end }}s</td>
  <td>{{ "%.3f"|format(seg.features.clip_drift) }}</td>
  <td>{{ "%.3f"|format(seg.features.flow_magnitude_mean) }}</td>
  <td>{{ "%.3f"|format(seg.features.face_drift) if seg.features.face_drift is not none else 'n/a' }}</td>
  <td>{% for f in seg.failures %}<span class="code">{{ f.code }}</span><br>{% endfor %}</td>
  <td>{% for kf in seg.keyframes[:3] %}<img src="{{ kf }}">{% endfor %}</td>
</tr>
{% endfor %}
</table>
</body></html>
"""


def render_html(report: dict, out_path: str) -> None:
    try:
        from jinja2 import Template
    except ImportError:
        logger.warning("Jinja2 not installed; skipping HTML render.")
        return

    out_dir = pathlib.Path(out_path).parent

    def _to_img_url(path_str: str) -> str:
        """Convert a filesystem path (possibly Windows-style) into a URL path
        relative to the directory containing report.html.
        """
        if not path_str:
            return path_str

        s = str(path_str).replace("\\\\", "/")

        # If the path already contains an evidence/ segment, make it relative.
        lower = s.lower()
        idx = lower.find("evidence/")
        if idx != -1:
            return s[idx:]

        p = pathlib.Path(s)

        candidates: list[pathlib.Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(out_dir / p)
            candidates.append(pathlib.Path.cwd() / p)

        abs_path = next((c for c in candidates if c.exists()), candidates[0])
        try:
            rel = abs_path.resolve().relative_to(out_dir.resolve())
            return rel.as_posix()
        except Exception:
            return pathlib.Path(os.path.relpath(str(abs_path), str(out_dir))).as_posix()

    def _normalize_report_paths(r: dict) -> dict:
        for seg in r.get("segments", []) or []:
            if isinstance(seg, dict):
                seg["keyframes"] = [_to_img_url(kf) for kf in seg.get("keyframes", []) or []]
                for fail in seg.get("failures", []) or []:
                    ev = (fail or {}).get("evidence") if isinstance(fail, dict) else None
                    if isinstance(ev, dict):
                        ev["keyframes"] = [_to_img_url(kf) for kf in ev.get("keyframes", []) or []]

        for fail in r.get("top_failures", []) or []:
            if isinstance(fail, dict):
                ev = fail.get("evidence")
                if isinstance(ev, dict):
                    ev["keyframes"] = [_to_img_url(kf) for kf in ev.get("keyframes", []) or []]
        return r

    try:
        report = _normalize_report_paths(report)
        tmpl = Template(_TEMPLATE)
        html = tmpl.render(
            meta=report.get("video_meta", {}),
            scores=report.get("scores", {}),
            top_failures=report.get("top_failures", []),
            patch_hints=report.get("patch_hints", []),
            segments=report.get("segments", []),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML report written to %s", out_path)
    except Exception as e:
        logger.warning("HTML render failed: %s", e)

