"""Generate the full demo experiment suite for Paper 1 under out_demo.

This orchestrates:
- 5 main comparison experiments
- 9 ablation experiments
- 3 closed-loop experiments

Outputs are written under a single out-root with a shared annotations.jsonl and
per-experiment predictions/logs + metrics directories.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from make_demo_closed_loop_logs import SUITE_PRESETS as CLOSED_LOOP_PRESETS
from make_demo_closed_loop_logs import generate_closed_loop_suite
from make_demo_metric_fixture import EXPERIMENT_PRESETS, _load_manifest_map, generate_suite_fixture


def _experiment_index() -> list[dict]:
    rows: list[dict] = []
    for preset in EXPERIMENT_PRESETS["comparisons"]:
        rows.append(
            {
                "name": preset["name"],
                "category": "comparisons",
                "outputs": {
                    "predictions": f"out_demo/eval_{preset['name']}/predictions.jsonl",
                    "metrics": f"out_demo/metrics/{preset['name']}",
                },
                "paper_tables": ["Table 3", "Table 4"],
                "seed": preset["seed"],
                "strength": preset["strength"],
                "profile": preset["profile"],
            }
        )
    for preset in EXPERIMENT_PRESETS["ablations"]:
        rows.append(
            {
                "name": preset["name"],
                "category": "ablations",
                "outputs": {
                    "predictions": f"out_demo/eval_ablation_{preset['name']}/predictions.jsonl",
                    "metrics": f"out_demo/metrics/{preset['name']}",
                },
                "paper_tables": ["Ablation Study", "Table ablation_demo"],
                "seed": preset["seed"],
                "strength": preset["strength"],
                "profile": preset["profile"],
            }
        )
    for preset in CLOSED_LOOP_PRESETS:
        rows.append(
            {
                "name": preset["name"],
                "category": "closed_loop",
                "outputs": {
                    "logs": f"out_demo/closed_loop_logs/{preset['name']}.jsonl",
                    "metrics": f"out_demo/metrics/closed_loop_{preset['name']}",
                },
                "paper_tables": ["Table 5"],
                "seed": preset["seed"],
                "profile": preset["profile"],
            }
        )
    return rows


def write_experiment_index(out_root: pathlib.Path) -> None:
    rows = _experiment_index()
    index_path = out_root / "experiment_index.json"
    md_path = out_root / "EXPERIMENTS.md"
    out_root.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Paper1 Demo Experiments",
        "",
        f"- Total experiments: {len(rows)}",
        "- Shared annotations: out_demo/annotations.jsonl",
        "- Standard layout: eval_* / metrics/* / closed_loop_logs/*",
        "",
        "| Experiment | Category | Output Dir | Paper Table |",
        "|---|---|---|---|",
    ]
    for row in rows:
        metrics_dir = row["outputs"]["metrics"]
        tables = " / ".join(row["paper_tables"])
        lines.append(f"| {row['name']} | {row['category']} | `{metrics_dir}` | {tables} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _cleanup_legacy_layout(out_root: pathlib.Path) -> None:
    import shutil

    for name in ("comparisons", "ablations", "closed_loop"):
        path = out_root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for file_name in ("predictions.jsonl", "closed_loop_logs.jsonl"):
        path = out_root / file_name
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass
    legacy_metrics = out_root / "metrics"
    if legacy_metrics.exists():
        for sub in ("full", "closed_loop"):
            path = legacy_metrics / sub
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the full Paper1 demo suite under out_demo.")
    parser.add_argument("--annotations", default="out/annotations.jsonl")
    parser.add_argument("--predictions", default="out/eval_full/predictions.jsonl")
    parser.add_argument("--manifest", default="out/dataset_v0/manifest.jsonl")
    parser.add_argument("--out-root", default="out_demo")
    args = parser.parse_args()

    out_root = pathlib.Path(args.out_root)
    manifest_map = _load_manifest_map(pathlib.Path(args.manifest))
    _cleanup_legacy_layout(out_root)

    generate_suite_fixture(
        annotations_path=pathlib.Path(args.annotations),
        predictions_path=pathlib.Path(args.predictions),
        out_root=out_root,
        manifest_map=manifest_map,
    )
    generate_closed_loop_suite(
        annotations_path=out_root / "annotations.jsonl",
        predictions_path=out_root / "eval_gpt4v" / "predictions.jsonl",
        out_root=out_root,
    )
    write_experiment_index(out_root)
    print(f"Paper1 demo suite written to {out_root}")
    print(f"Experiment index: {out_root / 'experiment_index.json'}")


if __name__ == "__main__":
    main()
