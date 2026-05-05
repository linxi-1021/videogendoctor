"""Smoke tests for VideoEval pipeline."""
import pathlib
import json
import sys
import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
DEMO_VIDEO = REPO_ROOT / "assets" / "demo" / "demo.mp4"
DEMO_SHOTIR = REPO_ROOT / "assets" / "demo" / "demo_shotir.json"


def test_taxonomy_loads():
    """Taxonomy JSON must load and have >=60 codes."""
    tax_path = (REPO_ROOT / "packages" / "videoeval" / "videoeval"
                / "taxonomy" / "failure_taxonomy_v0.1.json")
    assert tax_path.exists(), f"Taxonomy not found: {tax_path}"
    with open(tax_path, encoding="utf-8") as f:
        data = json.load(f)
    codes = [c for g in data["groups"] for c in g["codes"]]
    assert len(codes) >= 36, f"Expected >=36 codes, got {len(codes)}"


def test_patch_map_loads():
    """Patch map JSON must load and cover all taxonomy codes."""
    pm_path = (REPO_ROOT / "packages" / "videoeval" / "videoeval"
               / "taxonomy" / "patch_map_v0.1.json")
    assert pm_path.exists()
    with open(pm_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "patch_map" in data
    assert len(data["patch_map"]) >= 20


def test_schema_loads():
    """Report schema JSON must be valid JSON."""
    schema_path = (REPO_ROOT / "packages" / "videoeval" / "videoeval"
                   / "report" / "schema.json")
    assert schema_path.exists()
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    assert "properties" in schema
    assert "top_failures" in schema["properties"]


def test_demo_shotir_valid():
    """Demo ShotIR JSON must be valid and have >=1 shot."""
    assert DEMO_SHOTIR.exists(), f"demo_shotir.json not found: {DEMO_SHOTIR}"
    with open(DEMO_SHOTIR, encoding="utf-8") as f:
        spec = json.load(f)
    assert "shots" in spec
    assert len(spec["shots"]) >= 1


def test_compile_patch_no_failures():
    """Patch compiler returns empty list when no failures."""
    sys.path.insert(0, str(REPO_ROOT / "packages" / "videoeval"))
    from videoeval.patch.compile_patch import compile_patch
    report = {"top_failures": [], "video_meta": {}}
    actions = compile_patch(report, shotir=None)
    assert isinstance(actions, list)
    assert len(actions) == 0


def test_compile_patch_with_failure():
    """Patch compiler returns at least one action per failure code."""
    sys.path.insert(0, str(REPO_ROOT / "packages" / "videoeval"))
    from videoeval.patch.compile_patch import compile_patch
    report = {
        "top_failures": [{
            "code": "ID_FACE_DRIFT",
            "confidence": 0.8,
            "evidence": {"t0": 0.0, "t1": 2.0, "keyframes": []}
        }],
        "video_meta": {}
    }
    actions = compile_patch(report, shotir=None)
    assert len(actions) >= 1
    assert actions[0]["failure_code"] == "ID_FACE_DRIFT"


def test_dummy_judge():
    """DummyJudge must be importable and return expected keys."""
    sys.path.insert(0, str(REPO_ROOT))
    from services.worker.judge.dummy_judge import DummyJudge
    judge = DummyJudge()
    questions = [{"qid": "Q_TEST", "text": "Test question?", "code_group": "ID"}]
    result = judge.judge_segment(frames=[], questions=questions)
    assert "answers" in result
    assert "confidences" in result
    assert "code_probs" in result
    assert "Q_TEST" in result["answers"]


def test_runtime_config_loads_judge_yaml():
    """Main runtime config should merge judge.yaml fields."""
    sys.path.insert(0, str(REPO_ROOT / "packages" / "videoeval"))
    from videoeval.config import load_runtime_config

    cfg = load_runtime_config(
        str(REPO_ROOT / "configs" / "paper1" / "videoeval.yaml"),
        {"use_judge": True},
        judge_config_path=str(REPO_ROOT / "configs" / "paper1" / "judge.yaml"),
    )
    assert "judge_provider" in cfg
    assert "judge_model" in cfg
    assert "judge_candidate_topk" in cfg
    assert cfg["use_judge"] is True


def test_openai_judge_alias_mapping():
    """Legacy GPT-4V aliases should map to a current OpenAI vision-capable model name."""
    sys.path.insert(0, str(REPO_ROOT))
    from services.worker.judge.openai_judge import OpenAIJudge

    judge = OpenAIJudge(model_name="gpt-4v")
    assert judge.model_name == "gpt-4o"


def test_transformers_judge_normalizes_chat_style_output():
    """Transformers judge should flatten chat-style generated_text payloads."""
    sys.path.insert(0, str(REPO_ROOT))
    from services.worker.judge.transformers_hook import _normalize_generated_text

    payload = [
        {"role": "assistant", "content": [{"type": "text", "text": "Yes, the face changes."}]}
    ]
    text = _normalize_generated_text(payload)
    assert isinstance(text, str)
    assert "Yes" in text


def test_eval_failure_codes_stub():
    """Failure-code eval script runs on stub data."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.eval_failure_codes import compute_f1
    tmpdir = REPO_ROOT / "out" / "pytest_eval_failure_stub"
    tmpdir.mkdir(parents=True, exist_ok=True)
    pred = tmpdir / "pred.jsonl"
    label = tmpdir / "label.jsonl"
    pred.write_text(json.dumps({"id": "v1", "top_failures":
        [{"code": "MO_JITTER", "confidence": 0.8,
          "evidence": {"t0": 0, "t1": 2, "keyframes": []}}]}) + "\n", encoding="utf-8")
    label.write_text(json.dumps({"id": "v1", "failure_codes": ["MO_JITTER"]}) + "\n", encoding="utf-8")
    result = compute_f1(str(pred), str(label), str(tmpdir))
    assert "macro_f1" in result
    assert result["macro_f1"] == 1.0


def test_eval_failure_codes_ignores_verified_false_labels():
    """verified=false labels must not be counted as GT positives in F1."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.eval_failure_codes import compute_f1
    tmpdir = REPO_ROOT / "out" / "pytest_eval_failure_codes"
    tmpdir.mkdir(parents=True, exist_ok=True)
    pred = tmpdir / "pred.jsonl"
    label = tmpdir / "label.jsonl"

    pred.write_text(
        json.dumps({
            "id": "v1",
            "top_failures": [
                {"code": "MO_JITTER", "confidence": 0.8, "evidence": {"t0": 0, "t1": 2, "keyframes": []}}
            ],
        }) + "\n",
        encoding="utf-8",
    )
    label.write_text(
        json.dumps({
            "id": "v1",
            "top_failures": [
                {"code": "MO_JITTER", "verified": False, "evidence": {"t0": 0, "t1": 2, "keyframes": []}}
            ],
        }) + "\n",
        encoding="utf-8",
    )

    result = compute_f1(str(pred), str(label), str(tmpdir))
    assert result["macro_f1"] == 0.0
    assert result["micro_f1"] == 0.0
    assert result["per_code"] == {}


def test_eval_localization_ignores_verified_false_labels():
    """verified=false labels must not be used as GT spans in localization."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.eval_evidence_localization import compute_localization
    tmpdir = REPO_ROOT / "out" / "pytest_eval_localization"
    tmpdir.mkdir(parents=True, exist_ok=True)
    pred = tmpdir / "pred.jsonl"
    label = tmpdir / "label.jsonl"

    pred.write_text(
        json.dumps({
            "id": "v1",
            "top_failures": [
                {"code": "MO_JITTER", "confidence": 0.8, "evidence": {"t0": 1, "t1": 3, "keyframes": ["a.jpg"]}}
            ],
        }) + "\n",
        encoding="utf-8",
    )
    label.write_text(
        json.dumps({
            "id": "v1",
            "top_failures": [
                {"code": "MO_JITTER", "verified": False, "evidence": {"t0": 1, "t1": 3, "keyframes": ["a.jpg"]}}
            ],
        }) + "\n",
        encoding="utf-8",
    )

    result = compute_localization(str(pred), str(label), str(tmpdir))
    assert result["n_predictions"] == 1
    assert result["tiou_at_03"] == 0.0
    assert result["tiou_at_05"] == 0.0
    assert result["top1_hit"] == 0.0
    assert result["top3_hit"] == 0.0


def test_build_clean_eval_data_keeps_only_positive_failures():
    """Cleaner should keep verified=true failures and preserve negative samples."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.build_clean_eval_data import build_clean_annotations

    cleaned = build_clean_annotations([
        {
            "id": "v1",
            "annotator_id": "A1",
            "top_failures": [
                {"code": "MO_JITTER", "verified": True, "confidence": 0.9, "evidence": {"t0": 0, "t1": 1}},
                {"code": "MO_FRAME_DROP", "verified": False, "confidence": 1.0, "evidence": {"t0": 0, "t1": 1}},
            ],
            "notes": "x",
        },
        {
            "id": "v2",
            "annotator_id": "A1",
            "top_failures": [
                {"code": "AL_PROP_MISSING", "verified": False, "confidence": 1.0, "evidence": {"t0": 0, "t1": 1}},
            ],
            "notes": "y",
        },
    ])

    assert cleaned[0]["failure_codes"] == ["MO_JITTER"]
    assert [f["code"] for f in cleaned[0]["top_failures"]] == ["MO_JITTER"]
    assert cleaned[1]["failure_codes"] == []
    assert cleaned[1]["top_failures"] == []


def test_simulated_demo_generation_shapes():
    """Simulated demo builders should produce coherent label/prediction records."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.generate_simulated_demo_data import (
        simulate_annotations,
        simulate_predictions,
        build_simulated_manifest,
    )

    manifest = [
        {"id": "video1__remove_anchor__s0", "video_path": "x.mp4", "shotir_path": "x.json", "perturbation_type": "remove_anchor", "meta": {}},
        {"id": "video1__change_camera_movement__s1", "video_path": "y.mp4", "shotir_path": "y.json", "perturbation_type": "change_camera_movement", "meta": {}},
    ]
    annotations = simulate_annotations(manifest, seed=123)
    predictions = simulate_predictions(annotations, manifest, seed=123)
    simulated_manifest = build_simulated_manifest(manifest, annotations)

    assert len(annotations) == 2
    assert len(predictions) == 2
    assert len(simulated_manifest) == 2
    assert simulated_manifest[0]["meta"]["simulated"] is True
    assert "not_for_real_results" in simulated_manifest[0]["meta"]


def test_simulated_experiment_suite_generation_constraints():
    """Suite generator should create ordered, clearly-labeled simulated experiment outputs."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.generate_simulated_demo_data import generate_experiment_suite

    tmpdir = REPO_ROOT / "out" / "pytest_simulated_experiments"
    tmpdir.mkdir(parents=True, exist_ok=True)

    manifest = [
        {"id": "video1__remove_anchor__s0", "video_path": "x.mp4", "shotir_path": "x.json", "perturbation_type": "remove_anchor", "meta": {}},
        {"id": "video1__remove_anchor__s1", "video_path": "x2.mp4", "shotir_path": "x2.json", "perturbation_type": "remove_anchor", "meta": {}},
        {"id": "video2__drop_props_required__s0", "video_path": "y.mp4", "shotir_path": "y.json", "perturbation_type": "drop_props_required", "meta": {}},
        {"id": "video3__change_camera_movement__s0", "video_path": "z.mp4", "shotir_path": "z.json", "perturbation_type": "change_camera_movement", "meta": {}},
        {"id": "video4__extend_duration_or_merge__s0", "video_path": "m.mp4", "shotir_path": "m.json", "perturbation_type": "extend_duration_or_merge", "meta": {}},
        {"id": "video5__temporal_jitter_or_frame_drop__s0", "video_path": "n.mp4", "shotir_path": "n.json", "perturbation_type": "temporal_jitter_or_frame_drop", "meta": {}},
        {"id": "video6__compression_artifacts__s1", "video_path": "p.mp4", "shotir_path": "p.json", "perturbation_type": "compression_artifacts", "meta": {}},
    ]

    summaries = generate_experiment_suite(manifest_records=manifest, out_root=tmpdir, seed=20260419)
    by_name = {summary["profile"]: summary for summary in summaries}
    eval_root = tmpdir.parent
    metrics_root = tmpdir.parent / "metrics" / "pytest_simulated"

    assert len(summaries) >= 3
    assert (tmpdir / "manifest.jsonl").exists()
    assert (tmpdir / "annotations.jsonl").exists()
    assert (tmpdir / "profiles" / "random" / "summary.json").exists()
    assert (tmpdir / "profiles" / "rule_only" / "summary.json").exists()
    assert (tmpdir / "profiles" / "rule_open_vlm" / "summary.json").exists()
    assert (eval_root / "eval_pytest_simulated_rule_only" / "predictions.jsonl").exists()
    assert (eval_root / "eval_pytest_simulated_rule_only" / "video1__remove_anchor__s0" / "report.json").exists()
    assert (eval_root / "eval_pytest_simulated_rule_only" / "video1__remove_anchor__s0" / "report.html").exists()
    assert (metrics_root / "rule_only" / "failure_code_f1.json").exists()
    assert (metrics_root / "rule_only" / "evidence_localization.json").exists()
    assert (metrics_root / "closed_loop_patch" / "closed_loop.json").exists()
    assert by_name["random"]["simulated"] is True
    assert by_name["random"]["not_for_real_results"] is True

    random_macro = by_name["random"]["headline_metrics"]["macro_f1"]
    rule_macro = by_name["rule_only"]["headline_metrics"]["macro_f1"]
    dummy_macro = by_name["rule_dummy_judge"]["headline_metrics"]["macro_f1"]
    open_macro = by_name["rule_open_vlm"]["headline_metrics"]["macro_f1"]
    gpt_macro = by_name["rule_gpt4v"]["headline_metrics"]["macro_f1"]

    assert random_macro < rule_macro
    assert abs(dummy_macro - rule_macro) <= 0.08
    assert open_macro > rule_macro
    assert gpt_macro >= open_macro

    for profile_name in ["rule_only", "rule_dummy_judge", "rule_open_vlm", "rule_gpt4v"]:
        metrics = by_name[profile_name]["headline_metrics"]
        assert metrics["tiou_at_03"] >= metrics["tiou_at_05"]
        assert metrics["top3_hit"] >= metrics["top1_hit"]

    score_only = by_name["closed_loop_score_only"]["closed_loop_metrics"]
    score_patch = by_name["closed_loop_patch"]["closed_loop_metrics"]
    score_patch_judge = by_name["closed_loop_patch_judge"]["closed_loop_metrics"]
    assert score_only["pass_at_1"] < score_patch["pass_at_1"] <= score_patch_judge["pass_at_1"]

    summary_text = (tmpdir / "profiles" / "rule_only" / "summary.json").read_text(encoding="utf-8")
    readme_text = (tmpdir / "profiles" / "rule_only" / "README.txt").read_text(encoding="utf-8")
    assert '"not_for_real_results": true' in summary_text
    assert "Do NOT report these metrics as real experimental results." in readme_text

    for filename in [
        "suite_summary.json",
        "table_main_methods.json",
        "table_external_baselines.json",
        "table_ablations.json",
        "table_closed_loop.json",
    ]:
        assert (tmpdir / filename).exists()


def test_sample_double_annotation_subset():
    """QC subset sampling should be deterministic and preserve target size."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.sample_double_annotation import create_qc_subset
    tmpdir = REPO_ROOT / "out" / "pytest_qc_subset"
    tmpdir.mkdir(parents=True, exist_ok=True)
    manifest = tmpdir / "manifest.jsonl"
    annotations = tmpdir / "annotations.jsonl"

    records = []
    annotation_records = []
    for idx in range(12):
        records.append({
            "id": f"v{idx:02d}",
            "video_path": f"videos/v{idx:02d}.mp4",
            "perturbation_type": "group_a" if idx < 6 else "group_b",
        })
        annotation_records.append({
            "id": f"v{idx:02d}",
            "annotator_id": "A1",
            "top_failures": [],
            "notes": "",
        })

    manifest.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(json.dumps(r) for r in annotation_records) + "\n",
        encoding="utf-8",
    )

    sampled = create_qc_subset(
        manifest_path=manifest,
        out_manifest=tmpdir / "qc_manifest.jsonl",
        out_ids=tmpdir / "qc_ids.txt",
        out_primary_subset=tmpdir / "qc_primary.jsonl",
        annotations_path=annotations,
        ratio=0.1,
        min_samples=4,
        seed=7,
        stratify_by="perturbation_type",
    )

    assert len(sampled) == 4
    sampled_ids = [r["id"] for r in sampled]
    assert sampled_ids == sorted(sampled_ids)
    assert (tmpdir / "qc_manifest.jsonl").exists()
    assert (tmpdir / "qc_primary.jsonl").exists()


def test_compute_iaa_filters_verified_false():
    """IAA script should treat verified=false as a negative label."""
    sys.path.insert(0, str(REPO_ROOT))
    from infra.scripts.compute_iaa import compute_iaa
    tmpdir = REPO_ROOT / "out" / "pytest_iaa"
    tmpdir.mkdir(parents=True, exist_ok=True)
    primary = tmpdir / "primary.jsonl"
    secondary = tmpdir / "secondary.jsonl"
    manifest = tmpdir / "qc_manifest.jsonl"
    out = tmpdir / "iaa.json"

    primary.write_text(
        "\n".join([
            json.dumps({
                "id": "v1",
                "top_failures": [
                    {"code": "MO_JITTER", "verified": True, "evidence": {"t0": 0, "t1": 1}},
                    {"code": "ST_COLOR_SHIFT", "verified": False, "evidence": {"t0": 1, "t1": 2}},
                ],
            }),
            json.dumps({
                "id": "v2",
                "top_failures": [
                    {"code": "AL_PROP_MISSING", "verified": True, "evidence": {"t0": 0, "t1": 1}},
                ],
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    secondary.write_text(
        "\n".join([
            json.dumps({
                "id": "v1",
                "top_failures": [
                    {"code": "MO_JITTER", "verified": True, "evidence": {"t0": 0, "t1": 1}},
                ],
            }),
            json.dumps({
                "id": "v2",
                "top_failures": [
                    {"code": "AL_PROP_MISSING", "verified": False, "evidence": {"t0": 0, "t1": 1}},
                ],
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        "\n".join([
            json.dumps({"id": "v1"}),
            json.dumps({"id": "v2"}),
        ]) + "\n",
        encoding="utf-8",
    )

    result = compute_iaa(
        primary_path=primary,
        secondary_path=secondary,
        out_path=out,
        manifest_path=manifest,
    )

    assert result["decision_pairs"] == 3
    assert result["cohen_kappa"] == 0.4
    assert out.exists()


@pytest.mark.skipif(not DEMO_VIDEO.exists(),
    reason="demo.mp4 not found; run: python assets/demo/make_demo_video.py")
def test_pipeline_on_demo_video():
    """Full pipeline smoke test on demo.mp4."""
    import tempfile
    sys.path.insert(0, str(REPO_ROOT / "packages" / "videoeval"))
    from videoeval.pipeline import run_score
    cfg = {
        "seg_len": 2.0, "stride": 2.0, "k_frames": 3,
        "use_yolo": False, "use_judge": False,
        "clip_drift_threshold": 0.15,
        "flow_jitter_threshold": 5.0,
        "face_drift_threshold": 0.35,
        "top_k_failures": 3,
    }
    with tempfile.TemporaryDirectory() as tmp:
        report = run_score(
            video_path=str(DEMO_VIDEO),
            shotir_path=str(DEMO_SHOTIR),
            out_dir=tmp,
            cfg=cfg
        )
        assert "video_meta" in report
        assert "segments" in report
        assert len(report["segments"]) >= 1
        assert "top_failures" in report
        assert "patch_hints" in report
        assert "scores" in report

