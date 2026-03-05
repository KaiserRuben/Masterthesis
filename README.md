# VLM Boundary Testing

> Systematic boundary testing for Vision-Language Models under black-box and white-box constraints.

## Overview

This project develops a search-based testing framework for finding decision boundaries in VLMs — minimal input perturbations (text and image) that cause behavioral change in model outputs.

**Approach:**
- Multimodal manipulators for text (POS-aware synonym replacement) and image (VQGAN with discrete codebook)
- Multi-objective optimization (AGE-MOEA-2) over a combined text+image genotype
- Two boundary detection strategies: white-box (Liang et al. 2025, log-likelihood gap) and black-box (frontier pair detection via PDQ)

**SUT:** Open VLMs — Qwen3-VL (8B), Ministral-3 (14B), Gemma3, Alpamayo-R1 (based on Qwen3-VL).

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Optimizer                            │
│                      (AGE-MOEA-2)                           │
│                                                             │
│  Genotype: [synonym_idx₁..synonym_idxₙ | code₁..codeₘ]    │
│            ├── Text: |W| dims, each ∈{0..Kᵢ}              │
│            └── Image: |P| dims, each ∈{0..N-1}            │
│                                                             │
│  Objectives:                                                │
│    ω₁: minimize text perturbation distance                  │
│    ω₂: minimize image perturbation distance                 │
│    ω₃: maximize boundary closeness (Liang) / detect flip   │
├──────────────┬──────────────────────────┬───────────────────┤
│  Text Manip. │     Image Manipulator    │       SUT         │
│              │                          │                   │
│  POS-aware   │  VQGAN (pretrained       │  VLM (Qwen3-VL,  │
│  synonym     │  codebook, e.g.          │  Gemma3, etc.)    │
│  replacement │  imagenet_f16_16384)     │                   │
│  via spaCy + │                          │  Tasks:           │
│  fastText    │  Patch selection via     │  - Classification │
│              │  spatial frequency       │  - BBox           │
│  Seed expan- │                          │  - VQA            │
│  sion via    │  Codebook reduction via  │                   │
│  Pegasus     │  embedding distance      │                   │
└──────────────┴──────────────────────────┴───────────────────┘
```

## Boundary Detection Strategies

| Strategy | Access | Signal | Reference |
|----------|--------|--------|-----------|
| **Liang et al. 2025** | White-box (logits) | Log-likelihood gap between top-2 output sequences → 0 = boundary | [arXiv:2510.03271](https://arxiv.org/abs/2510.03271) |
| **PDQ (Dobslaw et al.)** | Black-box | Frontier pair detection — y(X) ≠ y(X') with minimal distance | Dobslaw & Feldt 2023 |

## VLM Task Taxonomy

| Task | Output Type | Boundary Definition |
|------|------------|---------------------|
| Classification | Discrete | Output label flip (clean, no threshold) |
| BBox / Segmentation | Continuous | Threshold-dependent (IoU, validity) |
| VQA | Free-form | Embedding-based semantic distance |

Initial focus: **Classification** (cleanest boundary definition), then generalization to other tasks.

## Project Structure

```
├── experiments/
│   ├── imagenet_vlm_probing/    # VLM classification accuracy on ImageNet (forced decoding)
│   ├── question_rewrite/        # Text manipulation prototypes (Pegasus, DIPPER, Parrot)
│   └── Archive/                 # Previous experiments (Phases 0–4, trajectory-based)
├── tools/
│   ├── VLTest/                  # VLTest framework (reference implementation)
│   ├── smoo/                    # SMOO framework (modular SUT/Manipulator/Optimizer)
│   ├── vlm/                     # VLM inference queue (Ollama-based)
│   ├── scene/                   # Classification schemas & prompts
│   └── alpamayo/                # NVIDIA Alpamayo-R1 trajectory model (submodule)
├── pipeline/                    # [ARCHIVED] Trajectory-based boundary analysis pipeline
├── infrastructure/
│   ├── docker/                  # Cloud GPU deployment
│   ├── workstation/             # NVIDIA GPU setup
│   └── local/                   # Apple Silicon (MPS) setup
├── data/                        # Run outputs (gitignored)
└── vlm_config.yaml              # VLM endpoint configuration (Ollama)
```

## Tools

### VLTest (Reference)

Black-box VLM testing framework using VQGAN codebook mutation and POS-aware text perturbation. Primary reference for the image manipulation methodology.

### SMOO (Reference)

Modular search-based testing framework (SUT → Manipulator → Optimizer → Objectives). Architectural reference — the thesis pipeline follows this structure.

### vlm/

VLM inference abstraction over Ollama with work-stealing queue for distributed inference.

```python
from vlm import load_config, SyncRequestQueue, Message
config = load_config("vlm_config.yaml")
with SyncRequestQueue(config) as queue:
    result = queue.submit(model="qwen3-vl:8b", messages=[...])
```

### scene/

Classification schema definitions with prompts and Pydantic response models for 24 semantic keys.

## Active Experiments

### imagenet_vlm_probing

Evaluates VLM forced-decoding classification accuracy on ImageNet. Measures per-class P(correct_label | image, prompt), approximate rank of correct label among 1000 classes, and top-1 prediction. Used to establish classification baselines for boundary testing.

### question_rewrite

Prototyping text manipulation strategies: Pegasus paraphrase, DIPPER, and Parrot models. Evaluates rewrite quality and diversity for seed expansion in the testing pipeline.

## Hardware

| Setup | Use Case | Details |
|-------|----------|---------|
| Apple Silicon (MPS) | Development, small experiments | See `infrastructure/local/` |
| NVIDIA GPU (24GB+) | Full inference, batch experiments | See `infrastructure/workstation/` |
| Cloud GPU (Docker) | Large-scale runs | See `infrastructure/docker/` |

## Key References

- Liang et al. 2025 — "Decision Boundary Testing for VLMs" ([arXiv:2510.03271](https://arxiv.org/abs/2510.03271))
- Weißl et al. — MIMICRY: Targeted DL System Boundary Testing
- Weißl et al. — SMOO: Modular Testing Framework
- Dobslaw & Feldt 2023 — PDQ: Partition Distance Quantification
- VLTest — Semantics-Preserving Multimodal Mutations for VLM Testing

## Supervision

| Role | Name | Affiliation |
|------|------|-------------|
| Supervisor | Prof. Andrea Stocco | TUM / fortiss |
| Co-Supervisor | Oliver Weißl | fortiss |

## License

Copyright (c) 2026 Ruben Kaiser. All rights reserved. See [LICENSE](LICENSE).
