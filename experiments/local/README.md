# Local (Mac/MPS) Experiment Setup

Setup for running Alpamayo-R1 experiments on Apple Silicon Macs using MPS backend.

## Requirements

- Apple Silicon Mac (M1/M2/M3) with 32GB+ unified memory recommended
- macOS 13+ (Ventura or later)
- Python 3.12
- HuggingFace account with access to `nvidia/PhysicalAI-Autonomous-Vehicles`

## Comparison with Workstation

| Feature | Local (Mac/MPS) | Workstation (CUDA) |
|---------|-----------------|---------------------|
| GPU Backend | MPS (Metal) | CUDA |
| Attention | eager | flash-attn |
| Model dtype | float32 | bfloat16 |
| Inference Speed | Slower (~5-10x) | Baseline |
| VRAM/Memory | Unified (32GB+) | 24GB+ dedicated |

## Quick Start

### 1. Run setup script

```bash
cd /path/to/Masterarbeit/experiments/local
chmod +x setup.sh
./setup.sh
```

The script will:
- Install Miniconda (if needed)
- Create `alpamayo-local` conda environment with Python 3.12
- Install PyTorch with MPS support
- Install Alpamayo-R1 package (with eager attention)
- Prompt for HuggingFace login
- Verify MPS access

### 2. Activate environment

```bash
conda activate alpamayo-local
```

### 3. Run experiments

**Basic inference test (5 scenes):**
```bash
python experiments/local/basic_inference_test.py
```

**Interactive notebook:**
```bash
jupyter notebook experiments/local/interactive_inference.ipynb
```

## Files

| File | Description |
|------|-------------|
| `setup.sh` | Automated setup script for Mac |
| `basic_inference_test.py` | Test model on random scenes |
| `interactive_inference.ipynb` | Interactive exploration notebook |

## CLI Options

```bash
# Basic usage
python basic_inference_test.py                    # 5 random scenes
python basic_inference_test.py -n 10              # 10 random scenes
python basic_inference_test.py --clip-id abc123   # Specific clip

# Device options
python basic_inference_test.py --cpu              # Force CPU (if MPS fails)
python basic_inference_test.py --dtype float16    # Try float16 (may not work)

# Output
python basic_inference_test.py -o results.json    # Custom output file
python basic_inference_test.py --list-clips       # Show available clips
```

## Key Differences from Workstation

1. **No flash-attn**: Uses `eager` attention instead
2. **float32 default**: MPS has limited bfloat16 support
3. **No autocast**: MPS autocast is experimental, disabled by default
4. **Slower inference**: Expect 5-10x slower than NVIDIA GPU

## Troubleshooting

### MPS not available
```bash
# Check macOS version (need 13+)
sw_vers

# Verify PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Out of memory
- The 10B model needs ~40GB for inference
- Close other applications
- Use `--cpu` flag as fallback (very slow)
- Consider running on workstation instead

### Model download slow
First run downloads ~20GB model weights. This is normal.

### HuggingFace access denied
1. Go to https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
2. Request access
3. Run `huggingface-cli login` again

## Output Files

Results are saved to:
```
data/alpamayo_inference/
└── inference_local_YYYYMMDD_HHMMSS.json
```

Contains:
- Chain-of-Causation (CoC) reasoning text
- Predicted trajectories
- Ground truth trajectories
- minADE metrics
- Timing information
- Device info (MPS/CPU)

## Performance Notes

On Apple M1 Max (32GB):
- Model load: ~2-3 minutes
- Per-scene inference: ~30-60 seconds (vs ~5-10s on RTX 4090)

For production/batch experiments, use the workstation setup with NVIDIA GPU.
