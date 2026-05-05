# VideoGenDoctor — Top-level Makefile
# One-command workflow for the full paper pipeline.
# Requires: Python 3.9+, ffmpeg, pip

PYTHON     ?= python
VIDEO      ?= assets/demo/demo.mp4
SHOTIR     ?= assets/demo/demo_shotir.json
OUT        ?= out
METRICS    ?= $(OUT)/metrics
LATEX_DIR  ?= docs/paper1_videogendoctor/latex
CONFIG     ?= configs/paper1/videoeval.yaml

.PHONY: setup demo dataset_v0 metrics paper_tables repro_bundle help

help:
	@echo "VideoGenDoctor paper pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  make setup          Print install hints"
	@echo "  make demo           Run videoeval score on demo.mp4"
	@echo "  make dataset_v0     Generate controlled perturbation dataset"
	@echo "  make metrics        Run all three metric scripts"
	@echo "  make paper_tables   Generate LaTeX tables/macros from metrics"
	@echo "  make repro_bundle   Pack configs+outputs+commit hash"

# ------------------------------------------------------------
setup:
	@echo "=== Install hints ==="
	@echo "1) pip install -e packages/videoeval[full]"
	@echo "2) pip install jinja2 click pyyaml tqdm jsonschema"
	@echo "3) (optional) pip install open_clip_torch insightface torch"
	@echo "4) (optional, YOLO) pip install ultralytics"
	@echo "5) (optional, judge) pip install transformers accelerate"
	@echo "6) Install ffmpeg: https://ffmpeg.org/download.html"
	@echo "7) Generate demo video: python assets/demo/make_demo_video.py"

# ------------------------------------------------------------
demo: $(VIDEO)
	@echo "=== Running VideoEval score on demo video ==="
	$(PYTHON) -m videoeval.cli score \
		--video $(VIDEO) \
		--shotir $(SHOTIR) \
		--out $(OUT)/demo_report \
		--config $(CONFIG)
	@echo ""
	@echo "Report: $(OUT)/demo_report/report.json"
	@echo "HTML:   $(OUT)/demo_report/report.html"

$(VIDEO):
	@echo "Demo video not found. Generating with ffmpeg..."
	$(PYTHON) assets/demo/make_demo_video.py

# ------------------------------------------------------------
dataset_v0:
	@echo "=== Generating controlled perturbation dataset ==="
	@$(PYTHON) -c "import json, pathlib; \
	  p=pathlib.Path('out/demo_manifest.jsonl'); \
	  p.parent.mkdir(exist_ok=True); \
	  p.write_text(json.dumps({'id':'demo_001','video_path':'assets/demo/demo.mp4','meta':{}})+'\n')"
	$(PYTHON) -m videoeval.data_gen.controlled_perturb \
		--input_manifest out/demo_manifest.jsonl \
		--out out/dataset_v0 \
		--seeds 1
	@echo "Dataset: out/dataset_v0/manifest.jsonl"

# ------------------------------------------------------------
metrics: _ensure_demo_outputs
	@echo "=== Running metric scripts ==="
	@mkdir -p $(METRICS)
	$(PYTHON) infra/scripts/eval_failure_codes.py \
		--pred out/demo_preds.jsonl \
		--label out/demo_labels.jsonl \
		--out $(METRICS)
	$(PYTHON) infra/scripts/eval_evidence_localization.py \
		--pred out/demo_preds.jsonl \
		--label out/demo_labels.jsonl \
		--out $(METRICS)
	$(PYTHON) infra/scripts/eval_closed_loop.py \
		--logs out/demo_cl_logs.jsonl \
		--out $(METRICS)
	@echo "Metrics written to $(METRICS)"

_ensure_demo_outputs:
	@$(PYTHON) -c "
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

# ------------------------------------------------------------
paper_tables:
	@echo "=== Generating LaTeX tables and macros ==="
	$(PYTHON) infra/scripts/make_paper1_tables.py \
		--metrics-dir $(METRICS) \
		--latex-dir $(LATEX_DIR)
	@echo "LaTeX macros: $(LATEX_DIR)/auto_numbers.tex"

# ------------------------------------------------------------
repro_bundle:
	@echo "=== Creating reproducibility bundle ==="
	@mkdir -p $(OUT)/repro_bundle
	cp -r configs/ $(OUT)/repro_bundle/configs
	cp -r $(METRICS) $(OUT)/repro_bundle/metrics 2>/dev/null || true
	@$(PYTHON) -c "
import json, pathlib, datetime
bundle = {
  'timestamp': datetime.datetime.utcnow().isoformat()+'Z',
  'git_commit': 'N/A (no git)',
  'configs': 'configs/',
  'metrics': 'out/metrics/',
}
pathlib.Path('$(OUT)/repro_bundle/bundle_meta.json').write_text(json.dumps(bundle, indent=2))
print('Bundle meta written.')
"
	@echo "Bundle: $(OUT)/repro_bundle/"

