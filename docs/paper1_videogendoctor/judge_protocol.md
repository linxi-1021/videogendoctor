# VLM Judge Protocol — Paper 1

## Overview
The Stage-2 VLM Judge receives a list of keyframe images and a set of structured questions
for a candidate segment. It returns answers, confidence scores, per-code probability
distributions, and optionally re-ranks evidence keyframes.

The judge runs ONLY on Stage-1 top-K candidate segments (candidate pruning).
Default: top-3 segments per video.

---

## Input Schema
```json
{
  "segment_id": "seg_000",
  "t_start": 0.0,
  "t_end": 2.0,
  "keyframes": ["evidence/seg_000/frame_000.jpg", "..."],
  "questions": [
    {"qid": "Q_IDENTITY_DRIFT", "text": "...", "code_group": "Identity"}
  ],
  "context": {
    "shotir_props_required": [],
    "character_ids": []
  }
}
```

## Output Schema
```json
{
  "segment_id": "seg_000",
  "answers": {
    "Q_IDENTITY_DRIFT": "Yes, the face identity changes noticeably at frame 3."
  },
  "confidences": {"Q_IDENTITY_DRIFT": 0.87},
  "code_probs": {"IDENTITY_DRIFT": 0.85, "STYLE_SHIFT": 0.12},
  "evidence_rerank": ["frame_003.jpg", "frame_001.jpg", "frame_000.jpg"]
}
```

---

## Question Templates by Code Group

### Group: Identity
- **Q_IDENTITY_DRIFT:** "Does the main character's face or body appearance change
  inconsistently between frames? Answer Yes/No and identify the frame index where
  the change is most visible."
- **Q_IDENTITY_CONSISTENCY:** "Is the clothing or distinctive visual attribute of the
  main character consistent throughout this segment? Answer Yes/No."

### Group: Props / Alignment
- **Q_PROP_MISSING:** "Is the required prop [PROP_NAME] visible in this segment?
  Answer Yes/No and describe its location if visible."
- **Q_PROP_PLACEMENT:** "Is [PROP_NAME] placed in the expected region [REGION]?
  Answer Yes/No."

### Group: Camera
- **Q_CAMERA_DEVIATION:** "Does the camera movement in this segment match a
  [EXPECTED_MOVEMENT] (e.g., static, pan-left, zoom-in)? Answer Yes/No and describe
  the actual movement observed."
- **Q_SHOT_TYPE:** "Is this segment shot as a [EXPECTED_SHOT_TYPE] (e.g., close-up,
  wide shot, medium shot)? Answer Yes/No."

### Group: Action / Event
- **Q_ACTION_PRESENT:** "Does the action '[ACTION_DESC]' occur in this segment?
  Answer Yes/No and identify the approximate frame where it begins."
- **Q_ACTION_ORDER:** "Do actions occur in the expected order: [ACTION_A] before
  [ACTION_B]? Answer Yes/No."

### Group: Style
- **Q_STYLE_CONSISTENCY:** "Is the visual style (lighting, color palette, art style)
  consistent throughout this segment? Answer Yes/No and describe any noticeable shift."
- **Q_QUALITY:** "Are there visible compression artifacts, blurring, or quality
  degradation in this segment? Answer Yes/No."

---

## Candidate Pruning Rule
1. Stage-1 computes a `candidate_score` for each segment = max failure confidence.
2. Top-K segments (default K=3) are passed to the judge.
3. Judge re-ranks evidence keyframes and updates `code_probs`.
4. Final `top_failures` list merges Stage-1 codes with Stage-2 `code_probs`
   (weighted: 0.4 × stage1 + 0.6 × stage2; weights configurable in judge.yaml).

---

## Reproducibility
- All judge calls must log: model name, prompt, raw response, token count, cost.
- Log path: `out/judge_logs/seg_{id}_judge.json`
- For open-source VLMs: log model path and quantization settings.

