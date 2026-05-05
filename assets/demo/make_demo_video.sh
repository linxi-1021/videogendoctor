#!/usr/bin/env bash
# make_demo_video.sh — generates a short demo.mp4 using ffmpeg testsrc
# Requires: ffmpeg
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/demo.mp4"

if command -v ffmpeg &>/dev/null; then
    ffmpeg -y -f lavfi -i testsrc=duration=8:size=640x360:rate=25 \
           -f lavfi -i sine=frequency=440:duration=8 \
           -c:v libx264 -preset fast -crf 28 \
           -c:a aac -b:a 64k \
           "$OUT"
    echo "Demo video created: $OUT"
else
    echo "ERROR: ffmpeg not found. Please install ffmpeg and re-run."
    exit 1
fi

