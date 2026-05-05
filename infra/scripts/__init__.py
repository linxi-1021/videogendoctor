"""Infrastructure scripts for VideoGenDoctor evaluation and demo pipelines.

Key scripts:
    eval_failure_codes.py      — Macro/micro F1 for failure-code detection
    eval_evidence_localization.py — tIoU and Top-K keyframe hit rate
    eval_closed_loop.py        — Pass@K, cost, avg iterations
    bootstrap_ci.py            — Bootstrap 95% confidence intervals
    build_source_manifest.py   — Construct source video manifest
    controlled_perturb.py      — Generate VideoGenDoctor-Bench-v0
    sample_double_annotation.py — Sample for dual-annotation reliability
    compute_iaa.py             — Inter-annotator agreement
"""
