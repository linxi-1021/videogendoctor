# Tables Plan — Paper 1
## VideoGenDoctor: Evidence-Grounded Failure Diagnosis and Actionable Repair

---

## Table 1: Comparison with Related Evaluation Frameworks (§2 / §6)
- **Purpose:** Show that VideoGenDoctor is the only system providing code+evidence+patch
  AND measuring closed-loop gains.
- **Columns:** System | Score | Failure Code | Evidence Loc. | Patch Output | Closed-Loop | Open-Source
- **Rows:** VBench | EvalCrafter | VideoScore | T2V-CompBench | VideoGenDoctor (ours)
- **Cell values:** ✓ / ✗ / partial
- **Data artifact:** `out/metrics/comparison_table.json`
- **Script:** `infra/scripts/make_paper1_tables.py --table comparison`
- **LaTeX output:** `docs/paper1_videogendoctor/latex/sections/results_tables.tex`

---

## Table 2: VideoGenDoctor-Bench-v0 Statistics (§5)
- **Purpose:** Describe the benchmark dataset.
- **Columns:** Split | #Videos | #Segments | Avg Duration | #Perturbation Types | #Failure Annotations
- **Rows:** Train | Val | Test | Total
- **Data artifact:** `out/metrics/bench_stats.json`
- **Script:** `infra/scripts/make_paper1_tables.py --table bench_stats`

---

## Table 3: Failure-Code Detection — F1 (§6.1)
- **Purpose:** Main detection result.
- **Columns:** Method | Macro-F1 | Micro-F1 | Identity-F1 | Scene-F1 | Motion-F1 | Camera-F1 | Align-F1 | Style-F1
- **Rows:** Rule-only | Rule+Judge(dummy) | Rule+Judge(VLM) | VideoGenDoctor (full)
- **Placeholders:** All values → [X]
- **Data artifact:** `out/metrics/failure_code_f1.json`
- **Script:** `infra/scripts/eval_failure_codes.py --out out/metrics/`
- **LaTeX output:** `docs/paper1_videogendoctor/latex/sections/results_tables.tex`

---

## Table 4: Evidence Localization (§6.2)
- **Purpose:** Temporal and spatial grounding accuracy.
- **Columns:** Method | tIoU@0.3 | tIoU@0.5 | Top-1 Keyframe Hit | Top-3 Keyframe Hit
- **Rows:** Random | Stage-1 only | Stage-1 + Judge | VideoGenDoctor (full)
- **Placeholders:** All values → [X]
- **Data artifact:** `out/metrics/evidence_localization.json`
- **Script:** `infra/scripts/eval_evidence_localization.py --out out/metrics/`

---

## Table 5: Closed-Loop Repair (§6.3)
- **Purpose:** End-to-end repair effectiveness.
- **Columns:** Method | Pass@1 | Pass@2 | Avg Iters | Time-to-Usable (s) | Cost/Usable-Min (USD)
- **Rows:** No-patch | Random-patch | Score-only | VideoGenDoctor (ours)
- **Placeholders:** All values → [X]
- **Data artifact:** `out/metrics/closed_loop.json`
- **Script:** `infra/scripts/eval_closed_loop.py --out out/metrics/`

