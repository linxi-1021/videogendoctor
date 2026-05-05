# HANDOFF — VideoGenDoctor

## Onboarding (Day 1)

1. Clone repo and enter root: `cd VideoGenDoctor`
2. Install deps: `make setup` (read printed hints, then run them)
3. Generate demo video: `python assets/demo/make_demo_video.py`
4. Run full demo: `make demo`
5. Open `out/demo_report/report.html` in browser — verify segments + failures render.
6. Read `docs/STUDENT_PLAYBOOK.md` end-to-end.

## Repository Map

```
VideoGenDoctor/
  assets/demo/            demo video + ShotIR spec
  configs/paper1/         experiment configs (videoeval / judge / closed_loop)
  docs/
    drafts/               raw manuscript (paste yours here)
    paper1_videogendoctor/ paper writing kit + LaTeX
    HANDOFF.md            this file
    STUDENT_PLAYBOOK.md   step-by-step student guide
    annotation_guide.md   annotation schema
    submission_checklist.md pre-submission checklist
    issues/               GitHub issues list
  infra/scripts/          eval scripts + demo pipeline
  packages/videoeval/     core Python package
    videoeval/
      cli.py              CLI entry point
      pipeline.py         main scoring pipeline
      features.py         feature extraction
      rules.py            Stage-1 rule engine
      judge_runner.py     Stage-2 VLM judge runner
      data_gen/           controlled perturbation generator
      patch/              patch compiler
      report/             schema + HTML renderer
      taxonomy/           failure taxonomy + patch map
  services/worker/judge/  VLM judge interface + implementations
  tests/                  smoke tests
  Makefile                one-command workflow
  CITATION.cff
```

## Week-by-Week Plan

### Week 1-2: Infrastructure
- [ ] Install all deps (`make setup`)
- [ ] Run `make demo` successfully
- [ ] Run `make dataset_v0` — inspect `out/dataset_v0/manifest.jsonl`
- [ ] Run `make metrics` — inspect `out/metrics/*.json`
- [ ] Run `make paper_tables` — inspect `auto_numbers.tex`
- [ ] Read taxonomy JSON, understand all 6 groups

### Week 3-4: Data Collection
- [ ] Collect [N] source videos with ShotIR specs
- [ ] Create `data/source_manifest.jsonl`
- [ ] Run `make dataset_v0` on real data
- [ ] Annotate dataset using `docs/annotation_guide.md`
- [ ] Compute inter-annotator agreement (κ)

### Week 5-6: Experiments
- [ ] Run `videoeval score` on all benchmark videos
- [ ] Run `eval_failure_codes.py` — get real F1 numbers
- [ ] Run `eval_evidence_localization.py` — get real tIoU numbers
- [ ] Set up closed-loop generator interface
- [ ] Run `eval_closed_loop.py` — get real Pass@1/2

### Week 7-8: VLM Judge
- [ ] Choose VLM model (set in `configs/paper1/judge.yaml`)
- [ ] Run `infra/scripts/run_judge_on_candidates.py` on subset
- [ ] Compare Stage-1 vs Stage-1+Judge F1
- [ ] Tune `alpha` weighting

### Week 9-10: Paper Writing
- [ ] Replace all `[X]` placeholders in LaTeX with real numbers
- [ ] Run `make paper_tables` to regenerate macros
- [ ] Complete all figures (Fig 1-4)
- [ ] Internal review pass
- [ ] Run `make repro_bundle`
- [ ] Check `docs/submission_checklist.md` — all boxes ticked

## Definition of Done (DoD)

- [ ] All `[X]` placeholders replaced with real experimental results
- [ ] `make demo` runs end-to-end without errors
- [ ] `pytest tests/` passes (including `test_pipeline_on_demo_video`)
- [ ] `make repro_bundle` produces a complete bundle
- [ ] `docs/submission_checklist.md` fully checked
- [ ] `citation_audit.md` all `to_verify` entries resolved
- [ ] LaTeX compiles without errors
- [ ] Dataset released / DOI obtained

## Key Commands

```bash
make demo                          # full pipeline on demo video
make dataset_v0                    # generate perturbation dataset
make metrics                       # run all eval scripts
make paper_tables                  # regenerate LaTeX macros
make repro_bundle                  # pack for submission
pytest tests/ -v                   # run smoke tests
python infra/scripts/run_judge_on_candidates.py --report out/demo_report/report.json --out out/judged
```

