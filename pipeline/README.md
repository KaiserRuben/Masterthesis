# Analysis Pipeline

Unified 5-step workflow for boundary sensitivity analysis in autonomous vehicle trajectory prediction.

## Overview

This pipeline investigates which semantic attributes cause the largest changes in trajectory prediction error when varied between similar scenes.

```
Sample → Embed → Classify → Infer → Analyze
```

## Quick Start

```bash
# Run all steps
python step_0_sample.py --n 2600
python step_1_embed.py
python step_2_classify.py
python step_3_infer.py
python step_4_analyze.py
```

## Pipeline Steps

### Step 0: Sample Scenes

Sample N scenes from the PhysicalAI-AV dataset, always including all anchor scenes (VLM-classified).

```bash
python step_0_sample.py --n 2600 --seed 42
python step_0_sample.py --n 3000 --extend  # Add more scenes
python step_0_sample.py --force             # Start fresh
```

**Output**: `data/pipeline/scenes.parquet`

### Step 1: Compute Embeddings

Generate OpenCLIP embeddings (ViT-bigG-14, 1280-dim) for all scenes.

```bash
python step_1_embed.py
python step_1_embed.py --batch-size 8  # Adjust for GPU memory
```

**Output**: `data/pipeline/embeddings.npz`

### Step 2: Classify (Label Propagation)

Propagate semantic labels from anchor scenes to all scenes using k-NN in embedding space.

```bash
python step_2_classify.py
python step_2_classify.py --k 10  # Adjust neighbor count
```

**Output**: Updates `scenes.parquet` with classification columns

### Step 3: Trajectory Inference

Run Alpamayo-R1 trajectory prediction on scenes. Requires 24GB+ VRAM.

```bash
python step_3_infer.py
python step_3_infer.py --checkpoint-interval 5  # Save more frequently
```

**Output**: `data/pipeline/results/inference_*.json`

For cloud GPU inference, see [`infrastructure/docker/README.md`](../infrastructure/docker/README.md).

### Step 4: Analyze Boundaries

Build k-NN graph, find single-key-diff pairs, and compute stability map.

```bash
python step_4_analyze.py
python step_4_analyze.py --k 20 --max-key-diff 1
```

**Output**: `data/pipeline/results/` (stability map, pairs, figures)

## Configuration

Edit `config.yaml` to customize:

```yaml
dataset:
  default_n: 2600
  default_seed: 42

embedding:
  model: "ViT-bigG-14"
  pretrained: "laion2b_s39b_b160k"
  dim: 1280

classification:
  keys:  # Top-6 from EMB-001 text alignment
    - weather
    - time_of_day
    - depth_complexity
    - occlusion_level
    - road_type
    - required_action

analysis:
  k_neighbors: 20
  max_key_diff: 1
```

## Notebooks

Interactive analysis notebooks in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `boundary_explorer.ipynb` | 3D embedding visualization with semantic coloring |
| `unified_analysis.ipynb` | Hypothesis testing (H1-H3) with publication figures |

### Hypothesis Testing Utilities

The `notebooks/utils/` module provides:

```python
from pipeline.notebooks.utils import (
    # Data loading
    load_pipeline_data,
    classify_ade,

    # H1: Boundary-error correlation
    compute_boundary_margin,
    create_h1_correlation_plot,
    create_h1_perkey_correlation_plot,

    # H2: Anisotropy
    compute_anisotropy_vector,
    create_h2_anisotropy_plot,

    # H3: Asymmetry
    compute_transition_asymmetry,
    create_h3_asymmetry_heatmap,

    # Visualization
    create_3d_explorer,
    create_stability_map,
    export_figure_for_print,
)
```

## Data Schema

### scenes.parquet

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | str | Unique scene identifier |
| `is_anchor` | bool | True if VLM-classified |
| `emb_index` | int | Index into embeddings matrix |
| `weather` | str | Classification value |
| `weather_confidence` | float | Label confidence (1.0 for anchors) |
| `ade` | float | Average Displacement Error (if inferred) |
| ... | ... | Other classification keys |

### embeddings.npz

```python
data = np.load("embeddings.npz")
embeddings = data["embeddings"]  # Shape: (n_scenes, 1280)
clip_ids = data["clip_ids"]      # Shape: (n_scenes,)
```

## Library Modules

### lib/schema.py

DataFrame I/O with schema validation:

```python
from lib.schema import load_scenes, save_scenes, COLUMNS
```

### lib/io.py

Configuration and path utilities:

```python
from lib.io import load_config, resolve_path, get_repo_root
```

### lib/trajectory.py

Alpamayo inference wrapper:

```python
from lib.trajectory import AlpamayoInference, run_batch_inference
```

## Output Structure

```
data/pipeline/
├── scenes.parquet           # Scene metadata + classifications
├── embeddings.npz           # OpenCLIP embeddings
├── image_cache/             # Cached composite images
└── results/
    ├── inference_*.json     # Trajectory predictions
    ├── knn_graph.pkl        # NetworkX graph
    ├── stability_map.json   # Sensitivity rankings
    └── figures/             # Generated visualizations
```

## Requirements

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
```

See `requirements.txt` for full list.
