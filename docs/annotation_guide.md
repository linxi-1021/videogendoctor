# VideoGenDoctor Annotation Guide

This document describes the annotation protocol used to construct
VideoGenDoctor-Bench-v0, the controlled diagnostic fixture for the
VideoGenDoctor paper (NeurIPS 2026 submission).

## Overview

Annotators review each perturbed video together with the original specification
(ShotIR JSON) and the auto-assigned candidate failure codes. For each candidate,
annotators determine whether the failure is visually present, assign the tightest
temporal span that contains the visible deviation, and select representative
keyframes when available. Annotators may also add missing failures if the
perturbation produces an additional visible artifact.

## Prerequisites

- Media player that displays timestamps (VLC, mpv, or the annotation interface).
- Copy of the failure taxonomy: `packages/videoeval/videoeval/taxonomy/failure_taxonomy_v0.1.json`.
- The video file and its paired ShotIR specification JSON.

## Annotation Procedure

1. **Load the sample**: Open the video and its paired ShotIR specification.
2. **Review candidate failures**: For each auto-assigned candidate code:
   a. Watch the segment indicated by the temporal span.
   b. Determine if the failure is visually observable.
   c. If YES: adjust the temporal span to the tightest interval that contains
      the visible deviation.
   d. If NO: mark as `verified: false`.
   e. Select 1-3 representative keyframes that best illustrate the failure.
3. **Add missed failures**: If additional visible artifacts are present, add them
   with the appropriate taxonomy code, temporal span, and keyframes.
4. **Ambiguous cases**: If a deviation is not visually identifiable, mark the
   candidate as not verified. Do not guess. A failure is retained in the
   benchmark only when the annotator can verify it from the rendered video and
   specification, without relying on perturbation metadata alone.

## Annotation Record Format

One JSON object per video, stored as a line in `annotations.jsonl`:

```json
{
  "id": "vid_001__temporal_jitter_or_frame_drop__s0",
  "annotator_id": "A1",
  "failure_codes": ["MO_JITTER", "MO_FRAME_DROP"],
  "top_failures": [
    {
      "code": "MO_JITTER",
      "confidence": 1.0,
      "verified": true,
      "evidence": {
        "t0": 2.0,
        "t1": 4.0,
        "keyframes": ["frame_000050.jpg", "frame_000075.jpg"]
      }
    }
  ],
  "notes": "Jitter visible at t=2.5s; frame drop at t=3.1s"
}
```

## Failure Code Reference

The full 38-code taxonomy is defined in
`packages/videoeval/videoeval/taxonomy/failure_taxonomy_v0.1.json`.

The 12 codes observed in the controlled fixture are:

| Code | Group | Definition |
|------|-------|------------|
| `ID_FACE_DRIFT` | Identity | Face identity changes inconsistently |
| `ID_BODY_DRIFT` | Identity | Body/clothing appearance changes |
| `CA_MOVE_WRONG` | Camera | Camera movement deviates from spec |
| `CA_SHOT_TYPE_WRONG` | Camera | Shot type/framing is incorrect |
| `CA_SHAKE` | Camera | Unintended camera shake |
| `MO_JITTER` | Motion | Unnatural frame-to-frame jitter |
| `MO_FROZEN_FRAME` | Motion | Video frozen when motion expected |
| `MO_SEGMENT_BREAK` | Motion | Visual discontinuity at segment boundary |
| `MO_EVENT_MISSING` | Motion | Required action/event not visible |
| `AL_PROP_MISSING` | Alignment | Required prop not visible |
| `ST_COMPRESSION_ARTIFACT` | Style | Visible compression/blocking artifacts |
| `SC_BG_INCONSISTENCY` | Scene | Background changes unexpectedly |

## Quality Control

- Every 10th video is double-annotated for inter-annotator agreement (IAA).
- A 120-video reliability subset (72 controlled + 48 real-failure) with 768
  candidate evidence spans is dual-annotated before adjudication.
- Disagreements are resolved by a third annotator.
- Code-level agreement target: Cohen's κ ≥ 0.8.
- Span-level agreement target: tIoU ≥ 0.65.

## Reliability Results

| Subset | #Videos | #Spans | Code κ | Span tIoU |
|--------|---------|--------|--------|-----------|
| Controlled | 72 | 468 | 0.858 | 0.716 |
| Real-failure | 48 | 300 | 0.803 | 0.641 |
| Combined | 120 | 768 | 0.831 | 0.681 |

Three-annotator extension (120 videos, 768 spans): Fleiss' κ = 0.813,
mean pairwise tIoU = 0.666, average boundary disagreement = 0.45s.

## IRB

The annotation protocol was reviewed and approved by the relevant institutional
review board. No personally identifiable information was collected. Annotators
were compensated according to institutional guidelines.

## Time Estimate

~3-5 minutes per video for experienced annotators.
