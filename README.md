# VideoGenDoctor

**Structured Diagnosis, Evidence Localization, and Repair for Controllable Video Generation**

VideoGenDoctor replaces scalar video quality scores with structured diagnosis
records that localize failures in time, name affected targets, and prescribe
repair actions. On VideoGenDoctor-Bench-v0 (324 controlled videos, 2,016
human-verified evidence spans), the default pipeline detects failure codes at
macro-F1 0.911, localizes evidence at tIoU@0.5 of 0.732, and lifts blind human
repair Pass@2 from 27.8% to 76.4%.

> **Paper:** NeurIPS 2026 submission.
> **Authors:** Bo Tan (Sichuan Agricultural University), Yupeng Niu (CAMS & PUMC).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e packages/videoeval
pip install opencv-contrib-python-headless numpy Pillow jinja2 click tqdm jsonschema pyyaml

# 2. (Optional) Install full features
pip install -e "packages/videoeval[full]"

# 3. Generate a demo video (requires ffmpeg)
python assets/demo/make_demo_video.py

# 4. Run diagnosis on the demo
make demo

# 5. Open the HTML report
#    Windows: start out\demo_report\report.html
#    Mac/Linux: open out/demo_report/report.html
```

## CLI Usage

```bash
# Basic: score a video against a ShotIR spec
videoeval score \
  --video path/to/video.mp4 \
  --shotir path/to/spec.json \
  --out out/my_report/ \
  --config configs/paper1/videoeval.yaml

# With Stage-2 VLM judge (OpenAI)
videoeval score \
  --video path/to/video.mp4 \
  --shotir path/to/spec.json \
  --out out/my_report/ \
  --config configs/paper1/videoeval.yaml \
  --use-judge

# With object detection
videoeval score \
  --video path/to/video.mp4 \
  --out out/my_report/ \
  --config configs/paper1/videoeval.yaml \
  --use-yolo
```

## Repository Structure

```
assets/demo/              Demo video + ShotIR spec
configs/
  paper1/                 Experiment configs (videoeval, judge, closed_loop)
  prompts/                VLM baseline prompt templates
docs/
  paper1_videogendoctor/  Paper LaTeX source
  annotation_guide.md     Annotation protocol
infra/scripts/            Evaluation scripts + bootstrap CI
packages/videoeval/       Core Python package (videoeval)
services/worker/judge/    VLM judge implementations
tests/                    Smoke tests
Makefile                  Top-level pipeline targets
CITATION.cff              Citation metadata
LICENSE                   MIT License
```

## Key Outputs

| File | Description |
|------|-------------|
| `out/report.json` | Structured diagnosis report with failure codes, evidence spans, confidence, keyframes |
| `out/report.html` | Visual HTML report with segment table and keyframe thumbnails |
| `out/evidence/seg_XXX/` | Extracted keyframe images per segment |
| `out/shotir.diff.json` | Patch diffs for ShotIR-aware generators |
| `out/rerender_plan.json` | Generator-agnostic repair plan |

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Print install hints |
| `make demo` | Run videoeval score on demo.mp4 |
| `make dataset_v0` | Generate controlled perturbation dataset |
| `make metrics` | Run all three evaluation metric scripts |
| `make paper_tables` | Generate LaTeX tables/macros from metrics |
| `make repro_bundle` | Pack configs + outputs + commit hash |

## Evaluation Scripts

```bash
# Failure-code detection F1
python infra/scripts/eval_failure_codes.py \
  --pred out/preds.jsonl \
  --label out/labels.jsonl \
  --out out/metrics/

# Evidence localization (tIoU, Top-K keyframe hit)
python infra/scripts/eval_evidence_localization.py \
  --pred out/preds.jsonl \
  --label out/labels.jsonl \
  --out out/metrics/

# Closed-loop repair (Pass@K, avg iterations, cost)
python infra/scripts/eval_closed_loop.py \
  --logs out/closed_loop_logs.jsonl \
  --out out/metrics/

# Bootstrap confidence intervals
python infra/scripts/bootstrap_ci.py \
  --pred out/preds.jsonl \
  --label out/labels.jsonl \
  --out out/bootstrap/ \
  --n-bootstrap 10000
```

## Data Availability

### Code (this repository)
The reference implementation, taxonomy schema, patch compiler, evaluation
scripts, VLM baseline prompt templates, and bootstrap CI scripts are released
under the MIT License.

### VideoGenDoctor-Bench-v0 (Zenodo)
The full benchmark dataset — including 324 controlled perturbation videos,
240 real-failure videos, 120 real-normal videos, human annotations, prediction
logs, and diagnostic reports — is deposited on Zenodo:

> **[Zenodo DOI — will be assigned upon acceptance]**
> https://doi.org/10.5281/zenodo.XXXXXXXXX

The 18 clean source clips used to construct the controlled fixture are obtained
from publicly available video datasets under permissive licenses. The four
production generators (CogVideoX, Wan, Stable Video Diffusion, HunyuanVideo)
are publicly available models; checkpoint identifiers are recorded in per-video
JSON manifests.

### Reproducibility
CPU-only recomputation of tables from serialized predictions and annotations
completes in seconds. Full feature extraction and VLM judging depend on GPU
availability and API access.

## Paper

> Bo Tan and Yupeng Niu. "VideoGenDoctor: Structured Diagnosis, Evidence
> Localization, and Repair for Controllable Video Generation." NeurIPS 2026
> submission.

LaTeX source: `docs/paper1_videogendoctor/neurIPS/`.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{tan2026videogendoctor,
  title = {VideoGenDoctor: Structured Diagnosis, Evidence Localization,
           and Repair for Controllable Video Generation},
  author = {Tan, Bo and Niu, Yupeng},
  year = {2026},
  version = {0.1.0},
  license = {MIT},
  repository = {https://github.com/tanbo1217/videogendoctor},
}
```
