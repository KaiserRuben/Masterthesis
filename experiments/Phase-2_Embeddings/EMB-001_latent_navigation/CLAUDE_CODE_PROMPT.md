# EMB-001: Latent Space Navigation & Semantic Structure Validation

## Context

Master Thesis: VLM Boundary Testing für Autonomous Driving. Kernidee: Navigation im semantischen Raum entlang interpretierbarer Achsen (weather, traffic, occlusion, ...).

**Konzeptionelle Trennung:**
- **Input Manifold**: Echte Driving-Szenen (Bilder) — "on-manifold" = echte Datenpunkte
- **Embedding Space**: Repräsentation durch Vision-Modell — Werkzeug zum Navigieren
- **Decision Model**: Alpamayo (später) — was wir testen

**Forschungsfrage:** Welches Embedding-Modell strukturiert den Input Space am besten für semantische Navigation?

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EMB-001 Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Provider   │    │   Provider   │    │   Provider   │     │
│   │   EVA-02-E   │    │  SigLIP 2    │    │    ...       │     │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│          │                   │                   │              │
│          └───────────────────┼───────────────────┘              │
│                              ▼                                  │
│                    ┌─────────────────┐                         │
│                    │  run_experiment │                         │
│                    │    (per model)  │                         │
│                    └────────┬────────┘                         │
│                             │                                   │
│                             ▼                                   │
│                    data/EMB-001/{model_name}/                  │
│                    ├── embeddings.npz                          │
│                    ├── analysis.json                           │
│                    ├── navigation_graph.pkl                    │
│                    └── visualization.html                      │
│                                                                  │
│                             │                                   │
│                             ▼                                   │
│                    ┌─────────────────┐                         │
│                    │ compare.ipynb   │  ← Alle Modelle         │
│                    │ (analysis)      │    vergleichen          │
│                    └─────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Models to Compare

| ID | Model | Params | Dim | Res | Hypothesis |
|----|-------|--------|-----|-----|------------|
| `eva02_e` | EVA-02-CLIP-E/14+ | 4.4B | 1024 | 224 | Best overall (MIM + size) |
| `openclip_bigg` | OpenCLIP ViT-bigG/14 | 2.5B | 1280 | 224 | Best pure CLIP (no MIM) |
| `siglip2_so400m` | SigLIP 2 SO400M/14-384 | 400M | 1152 | 384 | Best text-anchor alignment |
| `openai_clip_l` | OpenAI CLIP ViT-L/14@336 | 428M | 768 | 336 | Baseline |
| `eva02_l` | EVA-02-CLIP-L/14@336 | 428M | 1024 | 336 | MIM vs no-MIM (same size) |

**Vergleichsachsen:**
- Size: 400M → 4.4B
- Pretraining: Pure Contrastive vs MIM + Contrastive  
- Loss: Softmax vs Sigmoid
- Resolution: 224 vs 336 vs 384

---

## Data Strategy — IMPORTANT

### The Core Idea

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPERSET (N = configurable)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○   │
│   ○ ○ ● ○ ○ ○ ● ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○   │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ○ ○ ○   │
│   ○ ● ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○   │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○   │
│                                                                  │
│   ○ = Unlabeled scene (embedded, clustered, in k-NN graph)      │
│   ● = Labeled anchor (has 24 semantic keys) — 100 total         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Clustering:  ALL scenes (dense structure discovery)
Metrics:     ONLY anchors (need labels for ARI, alignment)
Navigation:  ALL scenes (any real clip is on-manifold)
```

### Data Sources

| Source | Location | Count | Has Labels |
|--------|----------|-------|------------|
| Classified anchors | `data/CLS-001/` | 100 | ✅ 24 keys |
| Pre-cached clips | Same dataset as key generation experiments | N - 100 | ❌ |

**IMPORTANT:** The superset scenes come from the SAME pre-cached dataset used throughout the experiments (key generation, etc.). Look in the existing codebase for how scenes are loaded — there should be a cache or data loader already in place.

### Target Size

```python
# PRODUCTION RUN
SUPERSET_SIZE = 10_000  # Target: 10,000+ scenes

# This will run overnight (user expectation: ~1h per model × 5 models = 5h+)
```

### ⚠️ VALIDATION-FIRST WORKFLOW — CRITICAL

```
┌─────────────────────────────────────────────────────────────────┐
│                    MANDATORY VALIDATION STEPS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Smoke Test (N=100, anchors only)                       │
│  ├── Verify all 5 providers load correctly                      │
│  ├── Verify embedding shapes match expected dims                │
│  ├── Verify text embedding works                                │
│  ├── Verify analysis pipeline runs without errors               │
│  └── Verify visualization generates                             │
│                                                                  │
│  STEP 2: Small Validation Run (N=500)                           │
│  ├── Run ONE model end-to-end (fastest: siglip2_so400m)        │
│  ├── Verify all outputs are generated                           │
│  ├── Verify metrics are plausible (not NaN, not all zeros)     │
│  ├── Verify visualization is interactive and correct            │
│  └── Estimate time per model for full run                       │
│                                                                  │
│  STEP 3: Report to User                                         │
│  ├── Show sample metrics from validation run                    │
│  ├── Show estimated runtime for full 10k run                    │
│  └── ASK USER TO CONFIRM before starting overnight run          │
│                                                                  │
│  STEP 4: Full Run (N=10,000)                                    │
│  └── Only after user confirmation                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**DO NOT** ask user to start the main 10k run until validation passes and results look reasonable.

### Finding the Pre-Cached Data

Explore the existing codebase to find:
1. Where scene images are cached
2. How scenes were loaded in previous experiments (CLS-001, key generation)
3. The data loader or cache mechanism

Likely locations to check:
- `data/` directory structure
- `tools/` for data loading utilities
- Previous experiment scripts for how they accessed scenes

---

## Existing Code Structure

```
/Users/kaiser/Projects/Masterarbeit/
├── data/
│   └── CLS-001/
│       ├── scene_classifications.json   # 100 scenes mit 24 semantic keys
│       └── images/                       # Scene composite images (.jpg)
├── tools/
│   ├── scene/
│   │   ├── keys.py                      # Semantic key definitions
│   │   └── enums.py                     # Key value enums
│   └── vlm/
│       └── ...
└── experiments/
    └── EMB-001_latent_navigation/       # Target directory
```

---

## Task: Implementation

### 1. `providers/base.py` — Abstract Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class EmbeddingResult:
    embeddings: np.ndarray      # (N, D)
    scene_ids: list[str]
    model_name: str
    embedding_dim: int
    
class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider"""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass
    
    @abstractmethod
    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Embed batch of images -> (N, D)"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed batch of texts -> (N, D)"""
        pass
```

### 2. `providers/` — One file per model

Implementiere 5 Provider:

**`providers/eva02_e.py`**
```python
# Uses open_clip
# Model: EVA02-E-14-plus, pretrained: laion2b_s9b_b144k
```

**`providers/openclip_bigg.py`**
```python
# Uses open_clip  
# Model: ViT-bigG-14, pretrained: laion2b_s39b_b160k
```

**`providers/siglip2_so400m.py`**
```python
# Uses transformers
# Model: google/siglip2-so400m-patch14-384
```

**`providers/openai_clip_l.py`**
```python
# Uses transformers
# Model: openai/clip-vit-large-patch14-336
```

**`providers/eva02_l.py`**
```python
# Uses open_clip
# Model: EVA02-L-14-336, pretrained: merged2b_s6b_b61k
```

**Wichtig:**
- Alle Provider nutzen `mps` device (Apple Silicon)
- Batch processing für Effizienz
- Normalisierte Embeddings (L2 norm = 1)

### 3. `providers/__init__.py` — Registry

```python
from .eva02_e import EVA02EProvider
from .openclip_bigg import OpenCLIPBigGProvider
from .siglip2_so400m import SigLIP2Provider
from .openai_clip_l import OpenAICLIPProvider
from .eva02_l import EVA02LProvider

PROVIDERS = {
    "eva02_e": EVA02EProvider,
    "openclip_bigg": OpenCLIPBigGProvider,
    "siglip2_so400m": SigLIP2Provider,
    "openai_clip_l": OpenAICLIPProvider,
    "eva02_l": EVA02LProvider,
}

def get_provider(name: str) -> EmbeddingProvider:
    return PROVIDERS[name]()
```

### 4. `run_experiment.py` — Per-Model Pipeline

```python
def run_experiment(provider_name: str, output_dir: Path):
    """
    Runs full experiment pipeline for one model.
    
    Steps:
    1. Load scenes + classifications
    2. Embed all images
    3. Embed text anchors (semantic key values as text)
    4. Run structure analysis (PCA, UMAP, HDBSCAN)
    5. Build navigation graph (k-NN)
    6. Compute alignment metrics
    7. Generate visualization
    8. Save all outputs
    """
```

**CLI:**
```bash
python run_experiment.py --provider eva02_e
python run_experiment.py --provider siglip2_so400m
python run_experiment.py --provider all  # runs all 5
```

### 5. `analysis.py` — Shared Analysis Functions

```python
def compute_pca(embeddings: np.ndarray, variance_threshold: float = 0.95) -> tuple[np.ndarray, int]:
    """PCA reduction, returns (reduced, n_components)"""

def compute_umap_3d(embeddings: np.ndarray) -> np.ndarray:
    """UMAP to 3D for visualization"""

def compute_clusters(embeddings: np.ndarray) -> np.ndarray:
    """HDBSCAN clustering, returns labels"""

def compute_alignment_metrics(
    embeddings: np.ndarray,
    classifications: dict[str, dict],
    cluster_labels: np.ndarray
) -> dict:
    """
    Returns:
    - per_key_ari: ARI between clusters and each semantic key
    - per_key_silhouette: Silhouette score grouped by key values
    - intra_inter_ratio: For each key, ratio of intra-key to inter-key distances
    """

def build_navigation_graph(
    embeddings: np.ndarray,
    scene_ids: list[str],
    classifications: dict,
    k: int = 10
) -> nx.Graph:
    """
    k-NN graph with edge attributes:
    - distance
    - keys_changed: list of keys that differ between nodes
    - is_single_key_diff: bool
    """

def compute_text_anchor_alignment(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    classifications: dict,
    key_name: str,
    key_values: list[str]
) -> dict:
    """
    For each image, compute distance to each text anchor.
    Returns alignment score: are images closer to their "correct" text anchor?
    """
```

### 6. `visualize.py` — Interactive Plotly

```python
def create_visualization(
    coords_3d: np.ndarray,
    scene_ids: list[str],
    classifications: dict,
    cluster_labels: np.ndarray,
    graph: nx.Graph,
    text_anchors: dict[str, np.ndarray],  # {key_value: 3d_coord}
    output_path: Path
):
    """
    Interactive 3D Plotly visualization:
    - Dropdown: Color by semantic key OR by cluster
    - Points: Scenes (hover shows all key values)
    - Stars: Text anchors (optional toggle)
    - Edges: k-NN connections (optional toggle)
    - Edges colored: single-key-diff edges highlighted
    """
```

### 7. `compare.ipynb` — Cross-Model Analysis Notebook

Sections:

1. **Setup** — Load all model results
2. **Cluster Quality** — Compare cluster counts, silhouette scores
3. **Key Alignment** — Bar chart: ARI per key, per model
4. **Text Anchor Alignment** — Which model aligns best with text descriptions?
5. **Navigation Coverage** — Single-key-diff pairs per model
6. **Size vs Performance** — Does bigger = better?
7. **Visualizations Side-by-Side** — Embed 3D plots for comparison
8. **Conclusion** — Which model for which purpose?

---

## Output Structure

```
data/EMB-001/
├── eva02_e/
│   ├── embeddings.npz         # embeddings, scene_ids, model_name
│   ├── text_anchors.npz       # text embeddings for semantic key values
│   ├── analysis.json          # all metrics
│   ├── umap_coords.npz        # 3D coordinates
│   ├── navigation_graph.pkl   # NetworkX graph
│   └── visualization.html     # Interactive plot
├── openclip_bigg/
│   └── ...
├── siglip2_so400m/
│   └── ...
├── openai_clip_l/
│   └── ...
├── eva02_l/
│   └── ...
└── comparison_summary.json    # Generated by compare.ipynb
```

---

## Metrics to Compute (per model)

### Cluster Quality
- `n_clusters`: Number of HDBSCAN clusters
- `noise_ratio`: Fraction of points labeled as noise
- `silhouette_score`: Overall silhouette

### Per-Key Alignment
For each of the 24 semantic keys:
- `ari`: Adjusted Rand Index (clusters vs key values)
- `silhouette`: Silhouette score (grouped by key value)
- `intra_inter_ratio`: Mean intra-key distance / mean inter-key distance

### Text Anchor Alignment
For each key:
- `accuracy`: % of images closer to correct text anchor than others
- `mean_rank`: Mean rank of correct anchor (1 = best)

### Navigation Graph
- `n_edges`: Total k-NN edges
- `n_single_key_diff`: Edges where exactly one key differs
- `coverage`: % of (scene, key) pairs with a single-key-diff neighbor
- `graph_connected`: Is the graph connected?

---

## Technical Requirements

- Python 3.12
- Dependencies: 
  - `open_clip_torch` (for EVA, OpenCLIP models)
  - `transformers` (for SigLIP, OpenAI CLIP)
  - `torch` (MPS backend)
  - `numpy`, `pandas`, `scikit-learn`
  - `umap-learn`, `hdbscan`
  - `plotly`, `networkx`
- Device: `mps` (Apple Silicon, 32GB RAM)
- No venv setup needed

---

## Research Questions

After running all 5 models:

1. **Does size matter?** EVA-E (4.4B) vs EVA-L (428M) vs SigLIP (400M)
2. **Does MIM help?** EVA-L vs OpenAI CLIP (same size, different pretraining)
3. **Does Sigmoid loss help?** SigLIP vs OpenAI CLIP (different loss functions)
4. **Which keys cluster best?** Consistent across models or model-dependent?
5. **Text anchors useful?** Does text-image alignment correlate with cluster quality?
6. **Best model for navigation?** Highest single-key-diff coverage = most navigable
