# VLM Boundary Testing - Notebooks

Threshold optimization and boundary detection for Vision-Language Models using referring expression grounding.

---

## 📁 Project Structure

```
notebooks/
├── 01_dataset_exploration.ipynb       # RefCOCO dataset analysis
├── 02_threshold_optimization.ipynb    # MI-based threshold optimization (mock + real data)
├── 03_vlm_pipeline.ipynb             # VLM integration & perturbation experiments
│
├── refcoco_loader.py                 # RefCOCO dataset utilities
├── vlm_mock_data.py                  # Realistic mock data generation
├── ollama_proxy.py                   # Ollama API wrapper (reduced context)
│
├── test_vlm_components.py            # Component validation (18/18 ✓)
├── test_ollama_api.py                # API integration tests (10/10 ✓)
│
├── README.md                         # This file
└── VALIDATION_RESULTS.md             # Validation matrix & findings
```

---

## 🎯 Workflow

### **Phase 1: Dataset Understanding** → `01_dataset_exploration.ipynb`
- Load lmms-lab/RefCOCO (8,811 validation samples)
- Analyze bbox distributions, expression patterns
- Create balanced subsets (pilot: 10, small: 50, validation: 100)
- **Output:** Subset indices for experiments

### **Phase 2: Threshold Optimization** → `02_threshold_optimization.ipynb`
- Validate MI optimization on improved mock data
- Test on real RefCOCO samples
- Compare geometric vs semantic signal optimization
- **Goal:** Confirm optimal thresholds differ by signal type

### **Phase 3: VLM Boundary Detection** → `03_vlm_pipeline.ipynb`
- Integrate Qwen3-VL via Ollama
- Run perturbation experiments (brightness, contrast, blur, noise)
- Compute dual-space boundary intensity (β_geo, β_sem)
- Detect boundary samples via class transitions
- **Output:** Boundary detection results, visualizations

---

## 🔬 Key Validation Results

### ✅ Component Tests (18/18 passed)
- Perturbation functions ✓
- IoU computation ✓
- Bbox parsing ✓
- Ollama service ✓

### ✅ API Integration (10/10 passed)
- Qwen3-VL bbox prediction: **IoU=0.535** ✓
- Embedding extraction: 4096-dim ✓
- Inference speed: ~5 sec/sample (8b model) ✓

### ✅ Dataset Access
- **lmms-lab/RefCOCO** loaded successfully
- Format: Multiple referring expressions + bbox per sample
- Split names: 'val', 'test', 'testA', 'testB'

---

## 📊 Experiment Plan

| Phase | Samples | Perturbations | Runtime | Purpose |
|-------|---------|---------------|---------|---------|
| Pilot | 10 | 5×5 (25) | ~12 min | Feasibility check |
| Small-scale | 50 | 5×5 (25) | ~60 min | Threshold optimization |
| Validation | 100 | 10×10 (100) | ~8 hours | Final results |

**Expected findings:**
- Boundary rate: 20-40% (realistic)
- MI improvement: >5% over uniform thresholds
- Geo-sem correlation: 0.3-0.7 (moderate coupling)

---

## 🚀 Quick Start

```bash
# 1. Explore dataset
jupyter notebook 01_dataset_exploration.ipynb

# 2. Run threshold optimization
jupyter notebook 02_threshold_optimization.ipynb

# 3. Run VLM experiments (requires Ollama with qwen3-vl:8b)
ollama pull qwen3-vl:8b
jupyter notebook 03_vlm_pipeline.ipynb
```

---

## 🔧 Utilities

### `refcoco_loader.py`
```python
from refcoco_loader import load_refcoco, get_sample_info

# Load dataset
ds = load_refcoco('val', num_samples=10)

# Get sample info
info = get_sample_info(ds[0])
# Returns: image, expressions, bbox_pixels, bbox_normalized, etc.
```

### `vlm_mock_data.py`
```python
from vlm_mock_data import generate_mock_vlm_data

# Generate realistic mock data
data = generate_mock_vlm_data(num_samples=100, num_perturbations_per_sample=10)
# Returns: iou_original, iou_perturbed, perturbation_magnitude, etc.
```

### `ollama_proxy.py`
```python
import ollama_proxy as ollama

# Automatically uses reduced context window (16k tokens)
response = ollama.chat(model='qwen3-vl:8b', messages=[...])
```

---

## 📖 Research Context

See **[CONTEXT.md](../CONTEXT.md)** for:
- Problem formalization
- Design decisions (why bbox, why MI, dual-space analysis)
- Function signatures & data shapes
- Research questions

See **[VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)** for:
- Detailed validation matrix
- Literature review findings
- Critical assumptions & tests

---

## 📝 Notes

### Dataset Format (lmms-lab/RefCOCO)
```python
{
    'question_id': str,
    'image': PIL.Image,
    'question': str,  # Generic prompt
    'answer': list[str],  # Multiple referring expressions
    'bbox': [x, y, width, height],  # Pixels
    'segmentation': list[float],  # Polygon points
    'file_name': str
}
```

### Model Selection
- **Qwen3-VL 8b:** Fast (2-4 sec/inference), use for testing
- **Qwen3-VL 30b:** Better accuracy, use for final results (5-10 sec/inference)

### Known Issues
- Voxel51/RefCOCO-M: Rate limited (use lmms-lab instead)
- flickr30k: Deprecated dataset scripts
- Mock data boundary rate was 92% → Fixed in `vlm_mock_data.py` (now ~30%)

---

## 🎓 Thesis Information

**Topic:** Boundary-Guided Hallucination Elicitation for Vision-Language Models

**Supervisors:** Prof. Andrea Stocco, Oliver Weißl (TUM/fortiss)

**Core Question:** Where are VLMs unstable? Find input regions where small perturbations cause large behavioral changes.
