# Coordinate Boundary Testing for VLM Grounding (Exp-103) — Design

**Date:** 2026-06-22
**Status:** Design (feasibility de-risked by spike; pending plan)
**Thesis:** *Semantics Preserving Boundary Testing in Vision-Language Models* — generalizes the method to a second output modality.

## 1. Motivation

The framework currently drives search toward *classification* decision boundaries
(output = class A vs B; boundary at `|P(A)−P(B)| ≈ 0`). This design extends it to
**coordinate / bounding-box outputs** from a VLM doing visual grounding, demonstrating
that the same boundary-search method generalizes beyond discrete classification.

Two payoffs: (a) it shows the method is output-modality-general, strengthening the
thesis's external validity; (b) it is a genuine research gap — no 2024–2026 work applies
search/boundary-value testing to VLM grounding *coordinates*. The boundary-crossing
*difficulty* finding (`Elements/Boundary-Crossing-Difficulty`) recurs in a new output space.

## 2. Core insight (the design pivot)

**The boundary is the teacher-forced probability balance, not geometry.** Exactly as
today's `TargetedBalance = |lp(class_A) − lp(class_B)|` comes from teacher-forcing the two
class strings, the coordinate boundary is:

> **`TgtBal_coord = |lp(box_A) − lp(box_B)|`**, where `box_A`, `box_B` are the two
> candidate *coordinate strings* (the two referents' boxes), each scored by teacher forcing.

The model is at the boundary when it is equally likely to emit box A or box B. IoU/DIoU is
**not** the boundary — at most a secondary output-distance/progress signal (optional, §6).

Consequence: `VLMScorer.score_categories` already teacher-forces *arbitrary* candidate
strings and returns length-normalized log-probs. Feeding it the two box strings as the
"categories" yields the boundary objective **with no change to the scoring core**.

## 3. Architecture

- **SUT:** `Qwen3.5-4B` (the repo's existing SUT; natively multimodal; grounding-confirmed).
  - Coordinate space: **normalized [0,1000]** in JSON `{"bbox_2d":[x1,y1,x2,y2]}` (verified
    empirically on a real image; *must be re-verified per model* — Qwen2.5-VL uses absolute
    smart-resized pixels instead).
  - Thinking model: close the `<think>` block before scoring so the forced answer is the
    immediate continuation (clean per-coordinate-token log-probs).
- **Boundary objective:** `TargetedBalance`, unchanged. Candidates = `(box_A_str, box_B_str)`.
- **Candidate construction:** two same-category referents from **RefCOCO+** (location words
  forbidden ⇒ appearance-driven flips), each with its annotated box → the A/B pair.
- **Manipulation axis:** reuse existing image (VQGAN) + text (MLM) manipulators via the
  `modality` flag. Perturb the image and/or the referring expression to drive the model from
  box A to box B. Default decision deferred (§7).
- **Pipeline:** evolutionary (AGE-MOEA-II) — it owns `TargetedBalance`. PDQ untouched
  (available later for the AutoBVA output-discontinuity angle).

## 4. Component inventory

**Reuse as-is:** `VLMScorer.score_categories` (`scorer.py:241`); `VLMSUT.process_input`
(`vlm_sut.py:131`, returns `Tensor(n_candidates)`); `TargetedBalance` (`targeted_balance.py:30`);
`CriterionCollection.evaluate_all(**kwargs)`; image+text manipulators (`src/manipulator/`);
evolutionary engine / optimizer / genotype→phenotype / seeds / `ParquetBuffer`; Qwen3.5-4B
adapter; Redis inference cache.

**Adapt (small):** grounding prompt via a new `answer_format`; candidate construction (two box
strings instead of two class names); trace/stats columns (`build_trace_rows`); runner objective
wiring (new coordinate `modality`/objectives key in `run_boundary_test.py`).

**New (small / some optional):** RefCOCO+ seed adapter (two-referent image + two boxes);
`<think>`-block handling; **[optional]** soft-box / DIoU progress objective + per-digit-logprob
exposure (only if a second objective beyond the teacher-forced boundary is wanted — the
per-token logprobs exist but are currently discarded at `scorer.py:308–319`).

## 5. Experiment protocol (Exp-103)

1. Select ambiguous RefCOCO+ items (two same-category referents, appearance-distinguishable).
2. **Reuse the Gen-0 margin predictor** to pre-screen crossability (median gen-0 `TgtBal_coord`),
   per `Elements/Boundary-Crossing-Difficulty` — only search items predicted to cross.
3. Short (20–50 gen) searches perturbing image and/or expression to drive `TgtBal_coord → 0`
   (flip which referent the model localizes).
4. Harvest boundary stimuli; characterize the boundary scale vs. the class-case label walls.

## 6. Optional second objective (soft-box progress)

If a continuous *progress* signal beyond the teacher-forced boundary is needed: decode the
expected coordinate per digit-position from the per-digit-token softmax (LLMTIME-style),
forming a "soft box," and use **DIoU over soft boxes**. Gives a gradient even between argmax
flips. Requires exposing the discarded per-token logprobs. Treat as a follow-on, not v1.

## 7. Open parameters

- Manipulation axis default: image-only, text-only, or joint (mirrors the existing modality flag).
- Soft-box progress objective: include in v1 or defer (§6).
- Dataset: RefCOCO+ primary (appearance-only) vs RefCOCO (allows positional expression
  perturbation); RefCOCOg-UMD as companion. COCO image licensing for any public artifact.
- Experiment identity: **Exp-103 provisional** — confirm number/series.

## 8. Risks

- **Coordinate-space normalization is per-model** — verify empirically each SUT (done for
  Qwen3.5-4B = [0,1000]); rescale via `image_grid_thw` if a model uses absolute pixels.
- **Thinking model** — must suppress/close `<think>` for clean teacher-forced scoring.
- **Boundary scale** — coordinate `TgtBal` magnitudes (~0.2–0.7 nats observed) differ from the
  class-case walls (~2.7–4.5 nats); the predictor threshold needs per-task recalibration.
- **Grounding quality on perturbed images** — VQGAN round-trips/abstract images localize poorly;
  use real images; validate grounding survives perturbation.
- **RefCOCO+ image licensing** (COCO/Flickr ToU) for any redistributable stimulus set.

## 9. Spike evidence (2026-06-22)

On `Qwen3.5-4B`, MPS, no new download (cached):
- Grounding works; emits valid `bbox_2d` JSON; coords normalized [0,1000] on a real image.
- Digits tokenize one-per-token (digit-distribution premise holds).
- A/B referent flip is real (two-cats image: left vs right cat, IoU=0).
- **Teacher-forced boundary signal is prompt-responsive:** `|lp(box_L)−lp(box_R)|` =
  0.51 (ambiguous) → 0.66 (specific-left, far from boundary) → **0.17 (specific-right,
  collapsing toward the boundary)**. This is `TargetedBalance` for coordinates, validated.

## 10. Build sequence (feeds the implementation plan)

1. SUT: coordinate grounding mode in `VLMScorer`/`VLMSUT` (prompt, `<think>` handling,
   `score_categories` with box-string candidates, coordinate-space normalization).
2. Seed adapter: RefCOCO+ two-referent items → `(image, box_A_str, box_B_str)` candidate pairs.
3. Runner wiring: coordinate `modality`/objectives key; trace/stats columns.
4. Validate `TgtBal_coord` end-to-end on a handful of items (no search).
5. Gen-0 predictor screen + short flip searches; harvest + characterize.
6. [optional] soft-box DIoU progress objective.

## References

LLMTIME (Gruver 2023, arXiv:2310.07820) · LLM Processes (Requeima 2024, arXiv:2405.12856) ·
Regression-aware Inference (Lukasik 2024, arXiv:2403.04182) · DIoU (Zheng 2020,
arXiv:1911.08287) · DETR (Carion 2020, arXiv:2005.12872) · Qwen2.5-VL (arXiv:2502.13923) ·
Qwen3-VL (arXiv:2511.21631) · RefCOCO/+ (arXiv:1608.00272) · Ref-L4 (arXiv:2406.16866) ·
AutoBVA (Dobslaw 2022, arXiv:2207.09065).
