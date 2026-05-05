# Figures Plan — Paper 1
## VideoGenDoctor: Evidence-Grounded Failure Diagnosis and Actionable Repair

---

## Fig 1: System Overview / Teaser (full-width, top of §4)
- **Type:** Architecture pipeline diagram
- **Content:**
  Input (video V + optional ShotIR S) →
  Segmentation & Feature Extraction →
  Stage-1 Rule Engine → top-K candidate segments →
  Stage-2 VLM Judge (optional) →
  Report (failure codes + evidence spans + keyframes) →
  Patch Compiler → shotir.diff / rerender_plan →
  Generator → (closed-loop arrow back)
- **Data artifact:** N/A (static diagram)
- **Script:** `infra/scripts/make_paper1_tables.py` (no data needed; diagram is hand-drawn / TikZ)
- **Caption template:** "VideoGenDoctor closed-loop pipeline. Given a generated video and an optional
  ShotIR specification, the system localises failures to temporal segments, assigns structured
  failure codes, and compiles actionable patches for re-rendering."

---

## Fig 2: Failure Taxonomy Mind-Map (§3 or Appendix A)
- **Type:** Hierarchical tree / sunburst
- **Content:** 6 groups × N codes each:
  Identity | Scene | Motion | Camera | Alignment | Style
  Each leaf node = one failure code (short label).
- **Data artifact:** `out/metrics/taxonomy_stats.json` (code counts per group)
- **Script:** `infra/scripts/make_paper1_tables.py --fig taxonomy`
- **Caption template:** "VideoGenDoctor failure taxonomy v0.1: [N] codes across 6 groups.
  Shading encodes average diagnosis confidence on VideoGenDoctor-Bench-v0."

---

## Fig 3: Closed-Loop Repair Curve (§6.3)
- **Type:** Line chart — Pass rate vs. iteration number
- **X-axis:** Iteration k = 0, 1, 2, 3
- **Y-axis:** Pass@k rate (0–1)
- **Series:** VideoGenDoctor (ours) vs. baselines (no-patch, random-patch, score-only)
- **Data artifact:** `out/metrics/closed_loop_curve.csv`
  Columns: method, iteration, pass_rate
- **Script:** `infra/scripts/eval_closed_loop.py --plot`
- **Caption template:** "Pass rate vs. repair iteration. VideoGenDoctor reaches Pass@1=[P1]%
  and Pass@2=[P2]%, [X]pp above the score-only baseline at iteration 2."

---

## Fig 4: Qualitative Examples (§7)
- **Type:** 2×2 panel grid
- **Content per panel:**
  Row = one failure type (e.g., identity drift, camera deviation)
  Col 1: problematic keyframe with bounding box / highlight
  Col 2: patch diff snippet + re-rendered keyframe (before/after)
- **Data artifact:** `out/evidence/` keyframe jpegs + `out/report.json`
- **Script:** `infra/scripts/make_paper1_tables.py --fig qualitative`
- **Caption template:** "Qualitative diagnosis and repair examples. Red boxes indicate
  evidence regions; green boxes show the corrected output after patch application."
