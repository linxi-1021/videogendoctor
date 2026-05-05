"""VideoEval CLI entry point."""
import click
from videoeval.config import load_runtime_config
from videoeval.pipeline import run_score


@click.group()
def main():
    """VideoEval: evidence-grounded failure diagnosis for video generation."""


@main.command()
@click.option("--video", required=True, help="Path to input video file.")
@click.option("--shotir", default=None, help="Path to optional ShotIR JSON spec.")
@click.option("--out", required=True, help="Output directory.")
@click.option("--config", default=None, help="Path to videoeval.yaml config.")
@click.option("--seg-len", default=2.0, show_default=True, help="Segment length in seconds.")
@click.option("--stride", default=2.0, show_default=True, help="Segment stride in seconds.")
@click.option("--k-frames", default=6, show_default=True, help="Keyframes per segment.")
@click.option("--use-yolo", is_flag=True, default=False, help="Enable YOLOv8 object detection.")
@click.option("--use-judge", is_flag=True, default=False, help="Enable Stage-2 VLM judge.")
def score(video, shotir, out, config, seg_len, stride, k_frames, use_yolo, use_judge):
    """Score a video and produce a diagnosis report."""
    import json, pathlib
    cfg = _load_config(config, seg_len=seg_len, stride=stride,
                       k_frames=k_frames, use_yolo=use_yolo, use_judge=use_judge)
    report = run_score(video_path=video, shotir_path=shotir, out_dir=out, cfg=cfg)
    click.echo(f"Report written to: {pathlib.Path(out) / 'report.json'}")
    click.echo(f"HTML  written to: {pathlib.Path(out) / 'report.html'}")
    top = report.get("top_failures", [])
    if top:
        click.echo("Top failures:")
        for f in top[:5]:
            click.echo(f"  [{f['confidence']:.2f}] {f['code']}  "
                       f"t={f['evidence']['t0']:.1f}s-{f['evidence']['t1']:.1f}s")
    else:
        click.echo("No failures detected.")


def _load_config(config_path, **overrides):
    import pathlib
    judge_config = None
    if config_path:
        judge_candidate = pathlib.Path(config_path).resolve().parent / "judge.yaml"
        judge_config = str(judge_candidate)
    return load_runtime_config(config_path, overrides, judge_config_path=judge_config)


if __name__ == "__main__":
    main()

