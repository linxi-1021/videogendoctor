# Rebuttal Cheatsheet — Paper 1

## Pre-emptive Responses to Expected Reviewer Concerns

---

### R1: "The failure taxonomy is ad-hoc / not validated."
**Response:**
- Taxonomy was derived from [N] real production failures in controllable video generation.
- Each code has an operational definition, evidence procedure, and patch template.
- Inter-annotator agreement: Cohen's κ = [K] on [N] annotated segments (Table A in appendix).
- We will release full annotation guide and raw annotations.

---

### R2: "Baselines are weak / no comparison to GPT-4V directly."
**Response:**
- We compare: rule-only, rule+dummy-judge, rule+VLM-judge (full VideoGenDoctor).
- GPT-4V baseline included in Table 3 (row: "GPT-4V zero-shot") — see §6.1.
- Our system is complementary: VLM judge is pluggable (§4.3); any VLM can be inserted.
- We measure a fundamentally different output: code+evidence+patch, not just a score.

---

### R3: "Benchmark is too small / not diverse enough."
**Response:**
- VideoGenDoctor-Bench-v0: [N] videos, [M] segments, 6 perturbation types × [K] seeds each.
- We provide the controlled perturbation generator so any lab can extend the benchmark.
- We will release benchmark + generator + annotation tools upon acceptance.

---

### R4: "Closed-loop evaluation depends on a specific generator; not general."
**Response:**
- We evaluate on [G] generators (Table 5 rows).
- Patch compiler outputs both shotir.diff.json (ShotIR-aware) and rerender_plan.json (generator-agnostic).
- The closed-loop protocol is defined as a standard interface; any generator implementing
  the spec can be plugged in.

---

### R5: "Stage-2 VLM judge is expensive / not reproducible."
**Response:**
- Stage-2 runs ONLY on Stage-1 top-K candidates (candidate pruning, §4.2).
- We report cost per usable minute (Table 5) with full cost breakdown in appendix.
- Dummy judge is always runnable (zero cost) for ablation; VLM is plug-in only.
- We provide a local open-source VLM option (transformers_hook.py).

---

### R6: "No real numbers; placeholders are not convincing."
**Response (pre-submission):**
- ALL placeholders must be replaced with real experimental results before submission.
- See submission_checklist.md §3 for the complete list of required numbers.
- This cheatsheet is for *after* experiments are run.

---

## Score Anchors
| Score | Meaning | Our target |
|-------|---------|------------|
| 6 (Weak Accept) | Minor concerns | Minimum acceptable |
| 7 (Accept) | Good paper | Target |
| 8 (Strong Accept) | Top-[X]% | Stretch 
