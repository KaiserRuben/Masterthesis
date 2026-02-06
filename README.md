# Semantic Boundary Testing for VLMs in Autonomous Driving

> Where do VLM decision boundaries lie, and do they predict trajectory model failures?

## Overview

This project investigates how semantic attributes affect trajectory prediction in vision-language models (VLMs) for autonomous driving. Rather than pixel-level perturbations, we discretize the input manifold into semantic classes and test how transitions between them affect prediction error.

**Approach:**
- Discretize input space into semantic classes (weather, road type, etc.)
- Test how semantic transitions affect trajectory prediction error
- Map decision landscape anisotropy and asymmetry

**Key Finding:** Embedding-level boundaries and behavioral boundaries are structurally different. Testing strategies cannot use embedding distance as a proxy for behavioral risk.

## Research Questions

| RQ | Question | Status |
|----|----------|--------|
| **RQ1** | Can decision boundaries be systematically mapped through semantic perturbation? | ✅ Yes — 13,043 matched pairs across 6 keys |
| **RQ2** | Is the decision landscape anisotropic (some axes more brittle)? | ✅ Moderate support (α = 1.41) |
| **RQ3** | Do boundary characteristics predict real-world failure modes? | ⚠️ Key-level: yes. Scene-level: no |

## Pipeline

5-step reproducible workflow for boundary sensitivity analysis:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║   Step 0: Sample     →  scenes.parquet (PhysicalAI-AV dataset)            ║
║   Step 1: Embed      →  embeddings.npz (OpenCLIP ViT-bigG-14, 1280-dim)   ║
║   Step 2: Classify   →  6 keys via k-NN propagation from anchors          ║
║   Step 3: Infer      →  ADE + trajectory classes (Alpamayo-R1-10B)        ║
║   Step 4: Analyze    →  stability map, boundary metrics                   ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**6 Canonical Classification Keys:**
- `weather` — clear, cloudy, rainy, foggy
- `time_of_day` — day, dawn/dusk, night
- `depth_complexity` — flat, layered, complex
- `occlusion_level` — none, minimal, moderate, severe
- `road_type` — highway, urban, residential, intersection, rural
- `required_action` — none, slow, stop, evade

See [`pipeline/README.md`](pipeline/README.md) for detailed documentation.

## Quick Start

### Prerequisites

- Python 3.12+
- GPU: NVIDIA (CUDA) or Apple Silicon (MPS)
- HuggingFace account with access to `nvidia/PhysicalAI-Autonomous-Vehicles`

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd Masterarbeit

# Install dependencies
pip install -r pipeline/requirements.txt

# Install Alpamayo package (requires HuggingFace access)
pip install -e tools/alpamayo

# Login to HuggingFace
huggingface-cli login
```

### Run Pipeline

```bash
cd pipeline

# Step 0: Sample scenes from dataset
python step_0_sample.py --n 2600 --seed 42

# Step 1: Compute OpenCLIP embeddings
python step_1_embed.py --batch-size 4

# Step 2: Propagate labels from anchors
python step_2_classify.py

# Step 3: Run trajectory inference (requires 24GB+ VRAM)
python step_3_infer.py --resume

# Step 4: Analyze boundaries
python step_4_analyze.py
```

## Project Structure

```
├── pipeline/                 # 5-step analysis pipeline
│   ├── step_0_sample.py      # Sample scenes from dataset
│   ├── step_1_embed.py       # Compute OpenCLIP embeddings
│   ├── step_2_classify.py    # Label propagation from anchors
│   ├── step_3_infer.py       # Alpamayo trajectory inference
│   ├── step_4_analyze.py     # Boundary sensitivity analysis
│   ├── lib/                  # Pipeline library modules
│   ├── notebooks/            # Interactive analysis
│   └── config.yaml           # Pipeline configuration
├── tools/
│   ├── vlm/                  # VLM inference queue (Ollama)
│   ├── scene/                # Classification schemas & prompts
│   └── alpamayo/             # Trajectory model interface (submodule)
├── infrastructure/
│   ├── docker/               # Cloud GPU deployment
│   ├── workstation/          # NVIDIA GPU workstation setup
│   └── local/                # Apple Silicon (MPS) setup
├── experiments/              # Research experiments by phase
│   ├── Phase-0_Infrastructure/
│   ├── Phase-1_Classification/
│   ├── Phase-2_Embeddings/
│   ├── Phase-3_Boundaries/
│   └── Phase-4_Validation/
├── data/                     # Run outputs (gitignored)
└── vlm_config.yaml           # VLM endpoint configuration
```

## Dataset

**PhysicalAI-AV** (NVIDIA) — requires HuggingFace registration.

- ~10k driving scenes from autonomous vehicle data
- 4-camera composite images (front, left, right, rear)
- 5s history + 5s prediction horizon
- Ground truth trajectories for ADE computation

Request access: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

## Hardware

### GPU Workstation (Recommended)

For running Alpamayo-R1 with NVIDIA GPU (24GB+ VRAM):

```bash
cd infrastructure/workstation
./setup.sh
conda activate alpamayo
```

See [`infrastructure/workstation/README.md`](infrastructure/workstation/README.md).

### Apple Silicon (Development)

For local development on Mac with MPS:

```bash
cd infrastructure/local
./setup.sh
conda activate alpamayo-local
```

See [`infrastructure/local/README.md`](infrastructure/local/README.md).

### Docker (Cloud Providers)

For cloud GPU providers (RunPod, Vast.ai, Lambda, Modal):

```bash
docker build -f infrastructure/docker/Dockerfile.alpamayo -t alpamayo-inference .
docker run --gpus all -e HF_TOKEN=$HF_TOKEN alpamayo-inference
```

See [`infrastructure/docker/README.md`](infrastructure/docker/README.md).

## Experiments

Research experiments organized by phase:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase-0** | Infrastructure & prompt tuning | Completed |
| **Phase-1** | Batch scene classification (100 anchors) | Completed |
| **Phase-2** | Latent navigation & embeddings | Completed |
| **Phase-3** | Data-first boundary detection | Completed |
| **Phase-4** | Validation experiments | In Progress |

**Key Results (v0.1.1):**
- 66.7% of semantic boundary crossings cause trajectory class changes
- Moderate anisotropy (α = 1.41), ranking stable across experiments
- Directional asymmetry confirmed (cloudy→foggy ≠ foggy→cloudy)
- 402 danger zones identified, spatially coherent in UMAP

See [`experiments/`](experiments/) for detailed experiment documentation.

## Tools

### vlm/

VLM inference abstraction supporting Ollama. Provides work-stealing queue for distributed classification.

```python
from vlm import load_config, SyncRequestQueue, Message

config = load_config("vlm_config.yaml")
with SyncRequestQueue(config) as queue:
    result = queue.submit(model="qwen3-vl:8b", messages=[...])
```

### scene/

Classification schema definitions with prompts and Pydantic response models for each of the 24 semantic keys.

```python
from scene import get_prompt, get_schema, get_response_model, KEYS

prompt = get_prompt("weather")
model = get_response_model("weather")
```

### alpamayo/

Interface to NVIDIA's Alpamayo-R1-10B trajectory prediction model (git submodule).

## License

Copyright (c) 2026 Ruben Kaiser. All rights reserved. See [LICENSE](LICENSE).
