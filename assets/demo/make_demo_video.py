"""Generate a short demo.mp4 using ffmpeg test sources.

Cross-platform replacement for make_demo_video.sh (Windows-friendly).
Requires: ffmpeg in PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("demo.mp4")),
        help="Output mp4 path (default: assets/demo/demo.mp4)",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--rate", type=int, default=25)
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ERROR: ffmpeg not found in PATH. Install ffmpeg and re-run."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = args.duration
    size = args.size
    rate = args.rate

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={size}:rate={rate}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(out_path),
    ]

    subprocess.run(cmd, check=True)
    print(f"Demo video created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
