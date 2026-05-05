# Paper 1 Outline
## VideoGenDoctor: Evidence-Grounded Failure Diagnosis and Actionable Repair for Controllable Video Generation

**Target venue:** CVPR / NeurIPS / ICCV (CCF-A)
**Submission type:** System + Benchmark paper

---

## Abstract (150 words target)
Highlight: gap (score-only → no recourse), system (VideoGenDoctor closed-loop), benchmark
(VideoGenDoctor-Bench-v0), metrics (F1, tIoU, Pass@1/2), open-source.

---

## 1. Introduction
- Hook: controllable video generation is hard to debug at scale.
- Problem formulation: given (video V, optional spec S), output (failure_codes C, evidence E, patch P).
- 3 Contributions:
  1. **Taxonomy**: 60+ structured failure codes with evidence & patch templates.
  2. **VideoGenDoctor system**: two-stage (rule engine + VLM judge) closed-loop pipeline.
  3. **Benchmark & Metrics**: VideoGenDoctor-Bench-v0 with controlled perturbations; 3 evaluation axes.
- Teaser figure (Fig 1).

## 2. Related Work
- Video quality metrics (VBench, EvalCrafter, VideoScore, T2V-CompBench).
- Controllable generation & ShotIR.
- VLM-as-judge (GPT-4V, InternVL, LLaVA).
- Temporal grounding.
- Novelty: we output code+evidence+patch and measure **closed-loop gains** — not just scores.

## 3. Problem Setup
- Inputs/outputs formalized.
- Failure taxonomy overview (6 groups).
- Metric definitions: failure-code F1, tIoU@0.3/0.5, Top-K keyframe hit, Pass@1/2.

## 4. VideoGenDoctor System
### 4.1 Segmentation & Feature Extraction
- Fixed-stride segmentation, OpenCLIP embedding, optical flow, face embedding.
### 4.2 Stage-1 Rule Engine
- Per-segment scoring, rule-based failure detection, candidate ranking.
### 4.3 Stage-2 VLM Judge (optional)
- Template Q&A, evidence reranking, confidence calibration.
### 4.4 Patch Compiler
- report → shotir.diff.json / rerender_plan.json.
### 4.5 Closed-Loop Integration
- Feedback loop architecture (Fig 3).

## 5. VideoGenDoctor-Bench-v0 (Dataset)
- Source videos + ShotIR specs.
- Controlled perturbation protocol (6 perturbation types).
- Annotation procedure.
- Statistics (Table 2).

## 6. Experiments
### 6.1 Failure-Code Detection (Table 3 — F1)
### 6.2 Evidence Localization (Table 4 — tIoU, Top-K hit)
### 6.3 Closed-Loop Repair (Table 5 — Pass@1/2, cost)
### 6.4 Ablation Study (Table in appendix)

## 7. Results & Analysis
- Qualitative examples (Fig 4).
- Per-group F1 breakdown.
- Stage-2 judge contribution.

## 8. Limitations & Future Work
## 9. Ethics & Responsible Release
## 10. Reproducibility Statement

---

## Appendix
- A: Full taxonomy table.
- B: Judge protocol & question templates.
- C: Patch template examples.
- D: Annotation guide.
- E: Ablation full table.

