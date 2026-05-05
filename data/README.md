# Data Directory

This directory contains source videos and specifications used to construct
VideoGenDoctor-Bench-v0. The full benchmark dataset is deposited on Zenodo.

## Contents

| File | Description |
|------|-------------|
| `source_manifest.jsonl` | Index of source videos and their ShotIR specs |
| `video{1..6}.mp4` | Clean source clips for demo and testing |
| `video{1..6}_shotir.json` | ShotIR specifications for each source clip |

## Full Benchmark Dataset (Zenodo)

The complete VideoGenDoctor-Bench-v0 is available on Zenodo:

> **[Zenodo DOI — will be assigned upon acceptance]**
> https://doi.org/10.5281/zenodo.XXXXXXXXX

The Zenodo deposit includes:
- `controlled_fixture/` — 324 perturbed videos (9 perturbation types × 18 source clips × 2 seeds)
- `real_failure_subset/` — 240 naturally occurring failure videos (4 generators × 60)
- `real_normal_subset/` — 120 normal videos (4 generators × 30)
- `annotations/` — Human-verified annotations in JSONL format
- `predictions/` — Prediction logs from all compared methods
- `manifests/` — Per-video JSON manifests with generator checkpoints, seeds, and metadata

## Source Clip Attribution

The 18 clean source clips are obtained from publicly available video datasets
under permissive licenses. Sources are recorded in the construction manifest.

## Generator Checkpoints

The four production generators used for the real-failure subset are publicly
available:

| Generator | Version | HuggingFace Model |
|-----------|---------|-------------------|
| CogVideoX | CogVideoX1.5-5B | THUDM/CogVideoX1.5-5B |
| Wan | Wan2.1-T2V-14B | Wan-AI/Wan2.1-T2V-14B |
| Stable Video Diffusion | SVD-XT-1.1 | stabilityai/stable-video-diffusion-img2vid-xt-1-1 |
| HunyuanVideo | HunyuanVideo-1.5 | tencent/HunyuanVideo |
