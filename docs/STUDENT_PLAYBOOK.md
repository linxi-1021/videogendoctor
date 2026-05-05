# Student Playbook — VideoGenDoctor Paper 1
## Step-by-step guide from zero to submission

---

## Step 0: Environment Setup

```bash
# 0a. Python 3.9+
python --version

# 0b. Install core deps
pip install opencv-contrib-python-headless numpy Pillow jinja2 click tqdm jsonschema pyyaml pytest

# 0c. Install videoeval package (editable)
pip install -e packages/videoeval

# 0d. (Optional) Install heavy deps for full features
pip install open_clip_torch insightface torch
pip install ultralytics          # YOLOv8
pip install transformers accelerate  # VLM judge

# 0e. Install ffmpeg (for demo video generation)
# Windows: winget install ffmpeg   OR   choco install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## Step 1: Generate Demo Video and Run Pipeline

```bash
# Generate demo video
python assets/demo/make_demo_video.py

# Run full pipeline
make demo

# If you don't have `make` on Windows, run the equivalent command directly:
python -m videoeval.cli score \
  --video assets/demo/demo.mp4 \
  --shotir assets/demo/demo_shotir.json \
  --out out/demo_report \
  --config configs/paper1/videoeval.yaml

# Open report in browser
# Windows: start out\demo_report\report.html
# Mac: open out/demo_report/report.html
```

Expected output:
- `out/demo_report/report.json` — machine-readable report
- `out/demo_report/report.html` — visual report with keyframes
- `out/demo_report/evidence/` — extracted keyframe JPEGs

---

## Step 2: Understand the Output

Open `out/demo_report/report.json`. Key fields:
- `video_meta` — video info
- `segments[]` — per-segment features and failures
- `scores` — consistency / coherence / alignment
- `top_failures[]` — failure code + confidence + evidence span
- `patch_hints[]` — structured repair actions

Read `packages/videoeval/videoeval/taxonomy/failure_taxonomy_v0.1.json`
to understand what each failure code means.

---

## Step 3: Generate Perturbation Dataset

```bash
# Create a source manifest (one real video per line)
echo '{"id":"vid_001","video_path":"your/video.mp4","meta":{}}' > data/source_manifest.jsonl

# Generate perturbed dataset
python -m videoeval.data_gen.controlled_perturb \
  --input_manifest data/source_manifest.jsonl \
  --out out/dataset_v0 \
  --seeds 3

# Inspect output
cat out/dataset_v0/manifest.jsonl | head -5
```

---

## Step 4: Annotate the Dataset

1. Read `docs/annotation_guide.md` fully.
2. For each video in `out/dataset_v0/manifest.jsonl`:
   - Watch the video
   - Verify the auto-assigned `failure_codes`
   - Add `evidence` spans (t0, t1, keyframes) to `annotations.jsonl`
3. Compute inter-annotator agreement with a second annotator.

---

## Step 5: Run Evaluation Metrics

```bash
# After annotation, create pred/label JSONL files from your reports

# Failure-code F1
python infra/scripts/eval_failure_codes.py \
  --pred out/predictions.jsonl \
  --label out/annotations.jsonl \
  --out out/metrics/

# Evidence localization
python infra/scripts/eval_evidence_localization.py \
  --pred out/predictions.jsonl \
  --label out/annotations.jsonl \
  --out out/metrics/

# Closed-loop repair (requires generator integration)
python infra/scripts/eval_closed_loop.py \
  --logs out/closed_loop_logs.jsonl \
  --out out/metrics/
```

---

## Step 6: Run VLM Judge (Optional)

```bash
# Set model in configs/paper1/judge.yaml:
# model_name: "llava-hf/llava-1.5-7b-hf"

python infra/scripts/run_judge_on_candidates.py \
  --report out/demo_report/report.json \
  --out out/judged/ \
  --model "llava-hf/llava-1.5-7b-hf" \
  --topk 3
```

---

## Step 7: Generate Paper Tables and Submit

```bash
# Generate LaTeX macros from metrics
make paper_tables

# Check all placeholders are replaced
grep -r '\[X\]' docs/paper1_videogendoctor/latex/

# Compile LaTeX (requires texlive)
cd docs/paper1_videogendoctor/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Create repro bundle
make repro_bundle

# Final check
cat docs/submission_checklist.md
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `videoeval: command not found` | Run `pip install -e packages/videoeval` |
| `make: command not found` (Windows) | Install make (e.g., `winget install MSYS2.MSYS2` then `pacman -S make`, or `scoop install make`), or run the equivalent `python -m videoeval.cli score ...` command shown in Step 1 |
| `OpenCLIP not available` | Run `pip install open_clip_torch torch` |
| `InsightFace not available` | Run `pip install insightface` (Linux/Mac only) |
| `demo.mp4 not found` | Run `python assets/demo/make_demo_video.py` |
| `ffmpeg not found` | Install ffmpeg (see Step 0) |
| LaTeX compile error | Check `auto_numbers.tex` for missing macros |

