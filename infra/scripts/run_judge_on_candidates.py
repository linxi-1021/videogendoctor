"""Demo: run Stage-2 VLM judge on Stage-1 candidate segments from a report."""
import argparse
import json
import logging
import sys
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM judge on top-K candidates from a VideoEval report.")
    parser.add_argument("--report", required=True, help="Path to report.json")
    parser.add_argument("--out",    required=True, help="Output directory")
    parser.add_argument("--model",  default=None,  help="HuggingFace model name (optional)")
    parser.add_argument("--topk",   type=int, default=3, help="Number of candidate segments")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    cfg = {
        "judge_candidate_topk": args.topk,
        "judge_model": args.model,
        "judge_alpha": 0.6,
    }

    from packages.videoeval.videoeval.judge_runner import run_judge
    segments = report.get("segments", [])
    updated_segments = run_judge(segments, cfg=cfg)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report["segments"] = updated_segments
    out_path = out / "report_judged.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Judge output written to %s", out_path)
    print(f"Judge report: {out_path}")


if __name__ == "__main__":
    main()

