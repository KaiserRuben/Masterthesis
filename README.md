# VLM Scene Classification

Vision-language model pipeline for autonomous vehicle scene classification using the PhysicalAI-AV dataset.

## Overview

This project implements a two-stage classification approach:

1. **Stage 1 (Vision)**: VLM generates detailed scene reasoning from a 4-camera composite image
2. **Stage 2 (Text)**: Per-key classification from the reasoning text using structured JSON outputs

## Project Structure

```
├── tools/
│   ├── vlm/              # VLM provider abstraction (Ollama)
│   ├── scene/            # Scene classification keys, prompts, response models
│   └── alpamayo/         # PhysicalAI-AV dataset loader (submodule)
├── experiments/
│   ├── batch_scene_classification.py   # Main batch processing script
│   ├── prompt_tuning/                  # Prompt variant experiments
│   └── workstation/                    # Remote GPU inference setup
├── docker/               # Containerized Ollama deployment
├── data/                 # Run outputs (gitignored)
│   └── runs/             # Each run is self-contained
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

## Docker Deployment

For GPU inference on remote servers:

```bash
cd docker
cp .env.example .env
make up
```

See `docker/README.md` for details.
