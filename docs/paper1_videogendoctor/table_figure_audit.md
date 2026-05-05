# Table and Figure Audit

Last updated: 2026-04-24

This audit compares three sources:

- `manuscript.docx`
- `docs/paper1_videogendoctor/latex/main.tex`
- `docs/paper1_videogendoctor/tables_plan.md` and `docs/paper1_videogendoctor/figures_plan.md`

## Manuscript.docx Content Review

The DOCX draft contains useful conceptual material but uses an older experimental story.

Usable content merged into LaTeX:

- Overall feedback-loop architecture figure, exported as `docs/paper1_videogendoctor/figures/manuscript_fig1.png`.
- Discussion/future-work language about modular deployment, sparse intervention, creative tools, and extension to adjacent generative media.

Content intentionally not merged:

- DOCX experimental claims such as ActionSynth with 1,200 videos, three generation backends, Pass@1 = 0.68, Pass@2 = 0.89, and cost = 0.18 USD/min.
- DOCX Table 1 and Table 2, because their numbers conflict with `out/metrics`.
- DOCX Figure 2 and Figure 3, because they visualize old or unsupported synthetic values rather than the current `out` results.
- DOCX numeric references using old code names such as `T-107`, `S-205`, and `P-201`, because the current taxonomy uses codes such as `MO_JITTER`, `AL_PROP_MISSING`, and `ID_FACE_DRIFT`.

## Tables

| Planned item | Current LaTeX status | Consistency |
|---|---|---|
| Table 1: Comparison with related frameworks | Present as `tab:comparison` | Partially consistent. It compares systems and capabilities, but the plan includes an extra Open-Source column that is not in the current table. |
| Table 2: Benchmark statistics | Present as `tab:bench_stats` | Partially consistent. It reports total benchmark statistics, but the plan describes Train/Val/Test/Total rows and average duration. Current data has no split table. |
| Table 3: Failure-code detection | Present as `tab:f1` | Partially consistent. It reports Macro-F1 and Micro-F1 for all comparison methods. The plan also requested per-group F1 columns. |
| Table 4: Evidence localization | Present as `tab:evidence` | Consistent. It reports tIoU@0.3, tIoU@0.5, Top-1, and Top-3. |
| Table 5: Closed-loop repair | Present as `tab:closedloop` | Mostly consistent. It reports Pass@1, Pass@2, iterations, time, and cost. Row names differ from the older plan but are aligned with `out/metrics`. |
| Ablation table | Present as `tab:ablation_demo` | Consistent with current `out/metrics` ablation files. |

Recommendation:

- Treat `tables_plan.md` as stale and update it to match the current data-backed tables, rather than replacing current tables with the DOCX tables.
- Do not merge the DOCX tables unless the corresponding experiments are actually rerun and written to `out/metrics`.

## Figures

| Planned item | Current LaTeX status | Consistency |
|---|---|---|
| Fig 1: System overview / teaser | Added as `fig:system_overview` from DOCX `manuscript_fig1.png` | Mostly consistent. It shows the feedback-loop architecture, although it is simpler than the full plan text. |
| Fig 2: Failure taxonomy mind-map | Missing | Not consistent. No taxonomy figure exists yet. |
| Fig 3: Closed-loop repair curve | Missing | Not consistent. DOCX Figure 3 is not usable because it uses unsupported synthetic values. Need a chart generated from `out/metrics` or closed-loop logs. |
| Fig 4: Qualitative examples | Missing | Not consistent. Needs selected evidence keyframes and before/after repair examples. |

Recommendation:

1. Keep the imported DOCX architecture figure as Fig. 1 for now.
2. Generate Fig. 2 directly from `packages/videoeval/videoeval/taxonomy/failure_taxonomy_v0.1.json`.
3. Generate Fig. 3 from the actual closed-loop metrics/logs under `out/closed_loop*` or `out/metrics/closed_loop*`.
4. Build Fig. 4 from real evidence frames under `out/eval_full/**/evidence/` and matching patch hints.
