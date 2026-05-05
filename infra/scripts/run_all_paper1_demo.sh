#!/usr/bin/env bash
# run_all_paper1_demo.sh — one-command demo pipeline for Paper 1
# Runs: videoeval score -> controlled perturb -> all three metric scripts
# All outputs go under out/
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

VIDEO="assets/demo/demo.mp4"
SHOTIR="assets/demo/demo_shotir.json"
OUT="out"
METRICS="$OUT/metrics"

echo "=== VideoGenDoctor Paper 1 Demo Pipeline ==="

# Step 0: ensure demo video exists
if [ ! -f "$VIDEO" ]; then
  echo "Generating demo video..."
  python assets/demo/make_demo_video.py
fi

# Step 1: run videoeval score
echo "[1/4] Running videoeval score..."
python -m videoeval.cli score \
  --video "$VIDEO" \
  --shotir "$SHOTIR" \
  --out "$OUT/demo_report" \
  --config configs/paper1/videoeval.yaml

echo "Report: $OUT/demo_report/report.json"

# Step 2: generate controlled perturbation dataset
echo "[2/4] Generating perturbation dataset..."
python -c "
import json, pathlib
p = pathlib.Path('out/demo_manifest.jsonl')
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps({'id':'demo_001','video_path':'assets/demo/demo.mp4','meta':{}})+'\n')
"
python -m videoeval.data_gen.controlled_perturb \
  --input_manifest out/demo_manifest.jsonl \
  --out out/dataset_v0 \
  --seeds 1

# Step 3: create stub eval files if missing
python -c "
import json, pathlib
for name, content in [
  ('out/demo_preds.jsonl',  {'id':'demo_001','top_failures':[{'code':'MO_JITTER','confidence':0.75,'evidence':{'t0':0.0,'t1':2.0,'keyframes':[]}}]}),
  ('out/demo_labels.jsonl', {'id':'demo_001','failure_codes':['MO_JITTER'],'top_failures':[{'code':'MO_JITTER','confidence':1.0,'evidence':{'t0':0.0,'t1':2.0,'keyframes':[]}}]}),
  ('out/demo_cl_logs.jsonl',{'id':'demo_001','iteration':0,'passed':True,'duration_s':8.0,'time_s':12.5,'cost_usd':0.004}),
]:
  p = pathlib.Path(name)
  p.parent.mkdir(exist_ok=True)
  if not p.exists():
    p.write_text(json.dumps(content)+'\n')
"

# Step 4: run metric scripts
echo "[3/4] Running metric scripts..."
mkdir -p "$METRICS"

python infra/scripts/eval_failure_codes.py \
  --pred out/demo_preds.jsonl \
  --label out/demo_labels.jsonl \
  --out "$METRICS"

python infra/scripts/eval_evidence_localization.py \
  --pred out/demo_preds.jsonl \
  --label out/demo_labels.jsonl \
  --out "$METRICS"

python infra/scripts/eval_closed_loop.py \
  --logs out/demo_cl_logs.jsonl \
  --out "$METRICS"

# Step 5: generate LaTeX tables
echo "[4/4] Generating LaTeX tables..."
python infra/scripts/make_paper1_tables.py \
  --metrics-dir "$METRICS" \
  --latex-dir docs/paper1_videogendoctor/latex

echo ""
echo "=== DONE ==="
echo "Report:       $OUT/demo_report/report.json"
echo "HTML:         $OUT/demo_report/report.html"
echo "Dataset:      $OUT/dataset_v0/manifest.jsonl"
echo "Metrics:      $METRICS/"
echo "LaTeX macros: docs/paper1_videogendoctor/latex/auto_numbers.tex"

