# VLM Scene Classification

Vision-language model pipeline for autonomous vehicle scene classification using the PhysicalAI-AV dataset. This is a Master's thesis project investigating how semantic attributes affect trajectory prediction in autonomous vehicles.

## Overview

This project implements a two-stage classification approach:

1. **Stage 1 (Vision)**: VLM generates detailed scene reasoning from a 4-camera composite image
2. **Stage 2 (Text)**: Per-key classification from the reasoning text using structured JSON outputs

The classified scenes are then used for boundary analysis to understand which semantic transitions cause the largest changes in trajectory prediction error.

## Project Structure

```
├── pipeline/             # Unified 5-step analysis pipeline
│   ├── step_0_sample.py  # Sample scenes from dataset
│   ├── step_1_embed.py   # Compute OpenCLIP embeddings
│   ├── step_2_classify.py # Label propagation from anchors
│   ├── step_3_infer.py   # Run Alpamayo trajectory inference
│   ├── step_4_analyze.py # Boundary sensitivity analysis
│   ├── notebooks/        # Interactive analysis & hypothesis testing
│   └── config.yaml       # Pipeline configuration
├── tools/
│   ├── vlm/              # VLM provider abstraction (Ollama)
│   ├── scene/            # Scene classification keys, prompts, response models
│   └── alpamayo/         # Alpamayo-R1 model wrapper (submodule)
├── infrastructure/
│   ├── docker/           # Docker setup for cloud GPU inference
│   ├── workstation/      # NVIDIA GPU workstation setup
│   └── local/            # Apple Silicon (MPS) setup
├── experiments/          # Research experiments by phase
│   ├── Phase-0_Infrastructure/
│   ├── Phase-1_Classification/
│   ├── Phase-2_Embeddings/
│   ├── Phase-3_Boundaries/
│   └── Phase-4_Validation/
├── data/                 # Run outputs (gitignored)
└── vlm_config.yaml       # Model and endpoint configuration
```

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recurse-submodules <repo-url>
cd Masterarbeit
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install pandas pillow torch pydantic pyyaml ollama

# Alpamayo dataset loader (requires HuggingFace access to nvidia/PhysicalAI-Autonomous-Vehicles)
pip install -e tools/alpamayo
```

### 3. Start Ollama

```bash
ollama serve
ollama pull qwen3-vl:8b
```

### 4. Run Classification

```bash
python experiments/batch_scene_classification.py --num-scenes 10
```

## Configuration

Edit `vlm_config.yaml` to configure endpoints and model tiers:

```yaml
endpoints:
  default:
    url: http://localhost:11434
    max_concurrent: 1

models:
  qwen3-vl:8b:
    endpoint: default
    tier: medium

key_mapping:
  stage1: large      # Scene reasoning needs vision
  weather: small     # Simple classification
  safety_criticality: large  # Complex reasoning
```

## Run Management

Each run creates a self-contained folder in `data/runs/`:

```
data/runs/classification_20260120_143022/
├── config.json              # Run configuration + selected clips
├── progress.json            # Incremental progress (resumable)
├── images/                  # Composite images
└── scene_classifications.json  # Final results
```

### Resume Interrupted Run

```bash
python batch_scene_classification.py --run-id classification_20260120_143022
```

## Classification Keys

Keys are organized by category:

| Category | Keys |
|----------|------|
| Scene Context | `road_type`, `weather`, `time_of_day`, `traffic_situation` |
| Object Detection | `pedestrians_present`, `cyclists_present`, `construction_activity`, `traffic_signals_visible`, `vehicle_count`, `notable_elements` |
| Spatial Reasoning | `occlusion_level`, `depth_complexity`, `nearest_vehicle_distance`, `spatial_relations` |
| Perceptual Challenge | `visual_degradation`, `similar_object_confusion`, `edge_case_objects` |
| Safety Critical | `safety_criticality`, `vulnerable_road_users`, `immediate_hazards`, `required_action` |
| Counting | `pedestrian_count`, `vehicle_count_by_type` |
| Attribute Binding | `traffic_light_states`, `lane_marking_type` |

## Tools

### VLM Package

```python
from vlm import load_config, SyncRequestQueue, Message

config = load_config("vlm_config.yaml")
with SyncRequestQueue(config) as queue:
    result = queue.submit(
        model="qwen3-vl:8b",
        messages=[Message("user", "Describe this.", images=(img_b64,))]
    )
```

### Scene Package

```python
from scene import get_prompt, get_schema, get_response_model, KEYS

prompt = get_prompt("weather")
schema = get_schema("weather")
model = get_response_model("weather")

# Parse LLM output to Pydantic model
response = model.model_validate_json(llm_output)
```

## Unified Analysis Pipeline

The `pipeline/` directory contains a reproducible 5-step workflow for boundary sensitivity analysis:

```bash
cd pipeline

# Step 0: Sample 2600 scenes (includes all anchors)
python step_0_sample.py --n 2600

# Step 1: Compute OpenCLIP embeddings (ViT-bigG-14)
python step_1_embed.py

# Step 2: Propagate labels from anchors via k-NN
python step_2_classify.py

# Step 3: Run Alpamayo trajectory inference
python step_3_infer.py

# Step 4: Build k-NN graph and analyze boundaries
python step_4_analyze.py
```

See [`pipeline/README.md`](pipeline/README.md) for detailed documentation.

### Interactive Analysis

The `pipeline/notebooks/` directory contains Jupyter notebooks for hypothesis testing:

- **`boundary_explorer.ipynb`** - 3D embedding space exploration with semantic coloring
- **`unified_analysis.ipynb`** - Hypothesis evaluation (H1-H3) with publication-ready figures

Key hypotheses tested:
- **H1**: Boundary-error correlation (scenes near class boundaries have higher ADE)
- **H2**: Anisotropy (sensitivity varies by semantic dimension)
- **H3**: Transition asymmetry (A→B vs B→A have different error profiles)

## Research Experiments

Experiments are organized by research phase:

| Phase | ID | Description | Status |
|-------|-----|-------------|--------|
| 0 | INF-001/002 | Infrastructure & prompt tuning | Completed |
| 1 | CLS-001 | Batch scene classification (100 anchors) | Completed |
| 2 | EMB-001 | Latent navigation & text alignment | Completed |
| 3 | BND-002 | Data-first boundary detection | Completed |
| 4 | VAL-* | Validation experiments | In Progress |

### Key Findings (BND-002)

From analyzing 224 scene pairs with single-key semantic differences:

| Rank | Semantic Key | Relative ΔADE |
|------|--------------|---------------|
| 1 | weather | 96% |
| 2 | required_action | 95% |
| 3 | depth_complexity | 89% |
| 4 | road_type | 70% |
| 5 | occlusion_level | 64% |

**Note**: Rankings are sensitive to sample size. Embedding similarity does NOT predict trajectory error (r = 0.058).

## Infrastructure

### GPU Workstation (Recommended)

For running Alpamayo-R1 locally with NVIDIA GPU:

```bash
cd infrastructure/workstation
bash setup.sh
conda activate alpamayo
```

See [`infrastructure/workstation/README.md`](infrastructure/workstation/README.md) for details.

### Docker (Cloud Providers)

For running on cloud GPU providers (RunPod, Vast.ai, Lambda, Modal):

```bash
# Build image
docker build -f infrastructure/docker/Dockerfile.alpamayo -t alpamayo-inference .

# Run with GPU
docker run --gpus all -e HF_TOKEN=$HF_TOKEN alpamayo-inference
```

See [`infrastructure/docker/README.md`](infrastructure/docker/README.md) for cloud deployment instructions.

### Apple Silicon (Development)

For local development on Mac with MPS acceleration:

```bash
cd infrastructure/local
bash setup.sh
```

See [`infrastructure/local/README.md`](infrastructure/local/README.md) for details.

## License

Copyright (c) 2026 Ruben Kaiser. All rights reserved. See [LICENSE](LICENSE).
