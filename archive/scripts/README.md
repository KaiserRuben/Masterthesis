# Notebooks Directory Structure

## Folder Organization

```
notebooks/
├── 01_dataset_exploration.ipynb
├── 02_threshold_optimization.ipynb
├── 03_vlm_pipeline.ipynb
├── data/                          # All .npz data files
│   ├── threshold_optimization_results.npz  ⭐ Main results
│   ├── optimization_results.npz
│   └── pilot_test_results.npz
├── figures/                       # All visualizations
│   ├── boundary_detection_analysis.png
│   ├── threshold_optimization_comparison.png
│   └── model_comparison_analysis.png
├── scripts/                       # Utility modules
│   ├── ollama_proxy.py           ⭐ VLM interface
│   ├── refcoco_loader.py         ⭐ Dataset loader
│   └── run_optimization_from_existing_data.py
└── tests/                         # Test scripts
    ├── test_*.py
    └── pilot_test.py
```

## How to Use Notebooks After Reorganization

### Step 1: Add Import Fix to First Cell

Add this to the **FIRST cell** of each notebook (before any imports):

```python
import sys
sys.path.insert(0, 'scripts')
```

### Step 2: Update Data Paths

Change file loading/saving paths:

**Before:**
```python
data = np.load('threshold_optimization_results.npz')
plt.savefig('my_plot.png')
```

**After:**
```python
data = np.load('data/threshold_optimization_results.npz')
plt.savefig('figures/my_plot.png')
```

### Step 3: Imports Work as Before

```python
import ollama_proxy as ollama
from refcoco_loader import load_refcoco, get_sample_info
```

## Quick Reference: File Locations

| What | Before | After |
|------|--------|-------|
| Load data | `np.load('file.npz')` | `np.load('data/file.npz')` |
| Save data | `np.savez('file.npz', ...)` | `np.savez('data/file.npz', ...)` |
| Save figure | `plt.savefig('fig.png')` | `plt.savefig('figures/fig.png')` |
| Import module | `import ollama_proxy` | Same (add path fix) |

## Running Helper Scripts

```bash
# From notebooks/ directory:
python scripts/run_optimization_from_existing_data.py
```

## Notebooks Status

- ✅ **Notebook 01**: Dataset exploration
- ✅ **Notebook 02**: Threshold optimization (data ready)
- ⏳ **Notebook 03**: Ready to run validation pipeline

## Key Files for Notebook 03

Required:
- `data/threshold_optimization_results.npz` ✓
  - Contains: `optimal_thresholds_geo`, `optimal_thresholds_sem`
- `scripts/ollama_proxy.py` ✓
- `scripts/refcoco_loader.py` ✓
