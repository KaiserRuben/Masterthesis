# Pipeline

5-step reproducible workflow for boundary sensitivity analysis in autonomous vehicle trajectory prediction.

## Overview

This pipeline investigates which semantic attributes cause the largest changes in trajectory prediction error when varied between similar scenes.

```
Sample → Embed → Classify → Infer → Analyze
```

## Quick Start

```bash
python step_0_sample.py --n 2600 --seed 42
python step_1_embed.py --batch-size 4
python step_2_classify.py
python step_3_infer.py --resume
python step_4_analyze.py
```

## Pipeline Steps

### Step 0: Sample Scenes

Sample N scenes from the PhysicalAI-AV dataset, always including all anchor scenes (VLM-classified).

```bash
python step_0_sample.py --n 2600 --seed 42   # Sample 2600 scenes
python step_0_sample.py --n 3000 --extend    # Add more scenes to existing
python step_0_sample.py --force              # Start fresh, delete existing
```

**CLI Flags:**
| Flag | Description |
|------|-------------|
| `--n N` | Total number of scenes (default: from config) |
| `--seed SEED` | Random seed (default: from config) |
| `--extend` | Keep existing scenes, add more to reach new N |
| `--force` | Delete existing data and start fresh |
| `--data-dir DIR` | Use different output directory |
| `--config PATH` | Path to config file |

**Output:** `data/pipeline/scenes.parquet`

### Step 1: Compute Embeddings

Generate OpenCLIP embeddings (ViT-bigG-14, 1280-dim) for all scenes, plus text vocabulary embeddings for classification keys.

```bash
python step_1_embed.py                    # Default settings
python step_1_embed.py --batch-size 8     # Increase for more GPU memory
python step_1_embed.py --device cuda      # Force specific device
python step_1_embed.py --skip-text        # Skip text vocabulary embeddings
python step_1_embed.py --text-only        # Only generate text embeddings
```

**CLI Flags:**
| Flag | Description |
|------|-------------|
| `--device DEVICE` | Device (cuda, mps, cpu). Auto-detected if not specified |
| `--batch-size N` | Batch size for embedding (default: from config) |
| `--skip-text` | Skip text vocabulary embedding generation |
| `--text-only` | Only generate text vocabulary embeddings |
| `--config PATH` | Path to config file |

**Output:** `data/pipeline/embeddings.npz`

### Step 2: Classify (Label Propagation)

Propagate semantic labels from anchor scenes (VLM-classified) to all scenes using nearest centroid in embedding space.

```bash
python step_2_classify.py                      # Default settings
python step_2_classify.py --min-confidence 0.5 # Set confidence threshold
python step_2_classify.py --reclassify         # Re-propagate ALL non-anchor labels
```

**CLI Flags:**
| Flag | Description |
|------|-------------|
| `--min-confidence FLOAT` | Minimum confidence threshold (default: from config) |
| `--reclassify` | Re-propagate ALL non-anchor labels |
| `--config PATH` | Path to config file |

**Output:** Updates `scenes.parquet` with classification columns

### Step 3: Trajectory Inference

Run Alpamayo-R1 trajectory prediction on scenes. Requires 24GB+ VRAM.

```bash
python step_3_infer.py                    # Process all scenes
python step_3_infer.py --resume           # Skip already-processed scenes
python step_3_infer.py --max-scenes 100   # Limit number of scenes
```

**CLI Flags:**
| Flag | Description |
|------|-------------|
| `--max-scenes N` | Maximum number of scenes to process |
| `--resume` | Resume from checkpoint (skips already-processed scenes) |
| `--config PATH` | Path to config file |

**Output:** `data/pipeline/results/inference_*.json`

For cloud GPU inference, see [`../infrastructure/docker/README.md`](../infrastructure/docker/README.md).

### Step 4: Analyze Boundaries

Build k-NN graph, find single-key-diff pairs, and compute stability map.

```bash
python step_4_analyze.py                        # Default settings
python step_4_analyze.py --k 20 --max-key-diff 1
python step_4_analyze.py --snapshot v0.1.1      # Archive current results first
```

**CLI Flags:**
| Flag | Description |
|------|-------------|
| `--k N` | Number of k-NN neighbors (default: from config) |
| `--max-key-diff N` | Maximum Hamming distance for pairs (default: from config) |
| `--min-confidence FLOAT` | Minimum label confidence threshold |
| `--snapshot NAME` | Archive current results before re-analyzing |
| `--config PATH` | Path to config file |

**Output:** `data/pipeline/results/` (stability map, pairs, figures)

## Configuration

Edit `config.yaml` to customize pipeline behavior:

```yaml
dataset:
  default_n: 2600
  default_seed: 42
  t0_us: 5000000              # 5 seconds into clip

embedding:
  model: "ViT-bigG-14"
  pretrained: "laion2b_s39b_b160k"
  dim: 1280
  batch_size: 4

classification:
  keys:                        # Top-6 from EMB-001 text alignment
    - weather
    - time_of_day
    - depth_complexity
    - occlusion_level
    - road_type
    - required_action

inference:
  model_id: "nvidia/Alpamayo-R1-10B"
  checkpoint_interval: 10

analysis:
  k_neighbors: 20
  max_key_diff: 1              # Strict single-key-diff
  min_confidence: 0.0

paths:
  data_dir: "data/pipeline"
  scenes_file: "data/pipeline/scenes.parquet"
  embeddings_file: "data/pipeline/embeddings.npz"
  results_dir: "data/pipeline/results"
  anchor_file: "data/CLS-001/scene_classifications.json"
  image_cache: "data/pipeline/image_cache"
```

## Library Modules

### lib/schema.py

DataFrame I/O with schema validation for `scenes.parquet`.

```python
from lib.schema import load_scenes, save_scenes, update_scenes, COLUMNS, CLASSIFICATION_KEYS
```

**Functions:**
- `load_scenes(path)` — Load parquet with correct dtypes
- `save_scenes(df, path)` — Save with schema validation
- `update_scenes(updates, path)` — Merge updates by clip_id

### lib/io.py

Configuration and file utilities.

```python
from lib.io import load_config, load_embeddings, append_embeddings, merge_inference_results
```

**Functions:**
- `load_config(path)` — Load and validate config.yaml
- `load_embeddings(path)` — Load embeddings.npz
- `append_embeddings(path, new_embeddings, new_clip_ids)` — Add new embeddings
- `merge_inference_results(results_dir)` — Combine inference JSON files

### lib/models.py

Pydantic models for typed data contracts.

```python
from lib.models import (
    TrajectoryClassification,  # Trajectory class result (direction, speed, lateral)
    InferenceResult,           # Single scene inference result
    PipelineConfig,            # Root config (from config.yaml)
)
```

**Models:**
- `TrajectoryClassification` — 36 possible classes (4×3×3)
- `InferenceResult` — clip_id, ade, trajectory classes, timestamps
- `PipelineConfig` — Validated configuration with nested models

### lib/composites.py

4-camera composite image generation.

```python
from lib.composites import create_composite, ensure_composites
```

**Functions:**
- `create_composite(clip_id, t0_us)` — Create 2×2 camera grid image
- `ensure_composites(clip_ids, cache_dir)` — Batch create with caching

### lib/trajectory.py

Trajectory classification from raw predictions.

```python
from lib.trajectory import classify_trajectory
```

**Classification thresholds:**
- Direction: `turn_left` (Δθ > 30°), `turn_right` (Δθ < -30°), `straight` (|Δθ| < 10°), `slight_curve`
- Speed: `accelerate` (Δv > 2 m/s), `decelerate` (Δv < -2 m/s), `constant`
- Lateral: `lane_change_left` (Δy > 3m), `lane_change_right` (Δy < -3m), `lane_keep`

## Data Schema

### scenes.parquet

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `clip_id` | string | Step 0 | Unique scene identifier |
| `is_anchor` | boolean | Step 0 | True if VLM-classified |
| `sample_seed` | Int64 | Step 0 | Random seed used for sampling |
| `emb_index` | Int64 | Step 1 | Index into embeddings matrix |
| `has_embedding` | boolean | Step 1 | Whether embedding exists |
| `weather` | string | Step 2 | clear, cloudy, rainy, foggy |
| `time_of_day` | string | Step 2 | day, dawn_dusk, night |
| `depth_complexity` | string | Step 2 | flat, layered, complex |
| `occlusion_level` | string | Step 2 | none, minimal, moderate, severe |
| `road_type` | string | Step 2 | highway, urban, residential, intersection, rural |
| `required_action` | string | Step 2 | none, slow, stop, evade |
| `label_source` | string | Step 2 | "vlm" or "propagated" |
| `label_confidence` | Float64 | Step 2 | Confidence score (1.0 for anchors) |
| `ade` | Float64 | Step 3 | Average Displacement Error (meters) |
| `coc_reasoning` | string | Step 3 | Chain-of-Causation reasoning text |
| `has_ade` | boolean | Step 3 | Whether inference completed |
| `inference_timestamp` | string | Step 3 | ISO timestamp of inference |
| `traj_direction` | string | Step 3 | turn_left, turn_right, straight, slight_curve |
| `traj_speed` | string | Step 3 | accelerate, decelerate, constant |
| `traj_lateral` | string | Step 3 | lane_change_left, lane_change_right, lane_keep |

### embeddings.npz

```python
data = np.load("embeddings.npz")
embeddings = data["embeddings"]  # Shape: (n_scenes, 1280)
clip_ids = data["clip_ids"]      # Shape: (n_scenes,)
```

## Notebooks

Interactive analysis notebooks in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `boundary_explorer.ipynb` | 3D embedding visualization with semantic coloring |
| `unified_analysis.ipynb` | Hypothesis testing (H1-H3) with publication figures |

### Hypothesis Testing Utilities

The `notebooks/utils/` module provides reusable analysis functions:

```python
from pipeline.notebooks.utils import (
    load_pipeline_data,
    classify_ade,
    compute_boundary_margin,
    create_h1_correlation_plot,
)
```

## Output Structure

```
data/pipeline/
├── scenes.parquet           # Scene metadata + classifications
├── embeddings.npz           # OpenCLIP embeddings (n × 1280)
├── image_cache/             # Cached composite images
└── results/
    ├── inference_*.json     # Trajectory predictions
    ├── stability_map.json   # Sensitivity rankings
    └── figures/             # Generated visualizations
```

## Requirements

See `requirements.txt` for full list:

```
pandas>=2.0
numpy>=1.24
torch>=2.0
open_clip_torch
transformers>=4.48
physical_ai_av
plotly
scikit-learn
networkx
pyarrow
pydantic
pyyaml
```
