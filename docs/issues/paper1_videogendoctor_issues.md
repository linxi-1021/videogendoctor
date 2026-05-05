# GitHub Issues — Paper 1: VideoGenDoctor

Copy these as GitHub issues. Each has: title, scope, files, acceptance criteria, dependencies.

---

## Issue #1: Collect source videos and build benchmark manifest
**Scope:** Data collection
**Files:** `data/source_manifest.jsonl`, `out/dataset_v0/`
**Acceptance criteria:**
- [ ] At least [N] source videos collected with paired ShotIR specs
- [ ] `data/source_manifest.jsonl` validated (all paths exist)
- [ ] `make dataset_v0` runs successfully on full manifest
- [ ] `out/dataset_v0/manifest.jsonl` has expected number of entries
**Dependencies:** None

---

## Issue #2: Human annotation of VideoGenDoctor-Bench-v0
**Scope:** Annotation
**Files:** `out/annotations.jsonl`, `docs/annotation_guide.md`
**Acceptance criteria:**
- [ ] All videos in `out/dataset_v0/` annotated
- [ ] Inter-annotator agreement Cohen's κ ≥ 0.7 on 10% double-annotated subset
- [ ] `out/annotations.jsonl` validated against schema
**Dependencies:** Issue #1

---

## Issue #3: Run Stage-1 evaluation and get real F1 numbers
**Scope:** Experiments
**Files:** `out/metrics/failure_code_f1.json`, `out/predictions.jsonl`
**Acceptance criteria:**
- [ ] `videoeval score` run on all benchmark videos
- [ ] `eval_failure_codes.py` produces macro-F1 and per-code F1
- [ ] Results match or exceed baseline in `tables_plan.md`
- [ ] `auto_numbers.tex` updated with real values
**Dependencies:** Issue #2

---

## Issue #4: Evidence localization evaluation
**Scope:** Experiments
**Files:** `out/metrics/evidence_localization.json`
**Acceptance criteria:**
- [ ] `eval_evidence_localization.py` produces tIoU@0.3/0.5 and Top-1/3 hit
- [ ] Results written to `out/metrics/`
- [ ] `auto_numbers.tex` macros updated
**Dependencies:** Issue #2, Issue #3

---

## Issue #5: Closed-loop repair evaluation
**Scope:** Experiments
**Files:** `out/closed_loop_logs.jsonl`, `out/metrics/closed_loop.json`
**Acceptance criteria:**
- [ ] Generator interface implemented (wraps at least one generator)
- [ ] Repair loop runs for at least [N] videos
- [ ] `eval_closed_loop.py` produces Pass@1, Pass@2, avg iters, cost/min
- [ ] `auto_numbers.tex` macros updated
**Dependencies:** Issue #3

---

## Issue #6: Stage-2 VLM judge integration
**Scope:** System
**Files:** `services/worker/judge/transformers_hook.py`, `configs/paper1/judge.yaml`
**Acceptance criteria:**
- [ ] `judge.yaml` has a valid `model_name`
- [ ] `run_judge_on_candidates.py` runs on demo report without error
- [ ] Stage-1+Judge F1 compared to Stage-1-only in ablation table
**Dependencies:** Issue #3

---

## Issue #7: Expand failure taxonomy to 60+ codes
**Scope:** Taxonomy
**Files:** `packages/videoeval/videoeval/taxonomy/failure_taxonomy_v0.1.json`
**Acceptance criteria:**
- [ ] At least 60 codes total across 6 groups
- [ ] Each code has: definition, evidence procedure, patch_template
- [ ] `patch_map_v0.1.json` updated with new codes
- [ ] `test_taxonomy_loads` passes with updated count
**Dependencies:** None

---

## Issue #8: Figure creation (Fig 1-4)
**Scope:** Paper writing
**Files:** `docs/paper1_videogendoctor/figures_plan.md`
**Acceptance criteria:**
- [ ] Fig 1 (system diagram) created and included in main.tex
- [ ] Fig 2 (taxonomy mindmap) generated from taxonomy JSON
- [ ] Fig 3 (closed-loop curve) generated from `closed_loop_curve.csv`
- [ ] Fig 4 (qualitative examples) selected from `out/evidence/`
**Dependencies:** Issue #5

---

## Issue #9: Replace all LaTeX placeholders and compile
**Scope:** Paper writing
**Files:** `docs/paper1_videogendoctor/latex/`
**Acceptance criteria:**
- [ ] Zero `[X]` placeholders in any `.tex` file
- [ ] `pdflatex main.tex` compiles without errors
- [ ] All citations resolve (no undefined references)
- [ ] `docs/submission_checklist.md` fully checked
**Dependencies:** Issues #3, #4, #5, #8

---

## Issue #10: Reproducibility bundle and dataset release
**Scope:** Release
**Files:** `out/repro_bundle/`, `CITATION.cff`
**Acceptance criteria:**
- [ ] `make repro_bundle` produces complete bundle
- [ ] Dataset uploaded to Zenodo or HuggingFace with DOI
- [ ] `CITATION.cff` updated with real author names and DOI
- [ ] README with one-command reproduce instructions published
**Dependencies:** Issue #9

