# Workstation Experiment Setup

Self-contained setup for running Alpamayo-R1 experiments on a GPU workstation.

## Requirements

- NVIDIA GPU with 24GB+ VRAM (RTX 4090 ✓)
- Linux (tested) or WSL2
- Internet connection (for model download)
- HuggingFace account with access to `nvidia/PhysicalAI-Autonomous-Vehicles`

## Quick Start

### 1. Transfer files to workstation

```bash
# From your local machine
scp -r /path/to/Masterarbeit user@workstation:/path/to/
```

Or use git, rsync, USB, etc.

### 2. Run setup script

```bash
cd /path/to/Masterarbeit/experiments/workstation
chmod +x setup.sh
./setup.sh
```

The script will:
- Install Miniconda (if needed)
- Create `alpamayo` conda environment with Python 3.12
- Install PyTorch with CUDA 12.4
- Install Alpamayo-R1 package
- Prompt for HuggingFace login
- Verify GPU access

### 3. Activate environment

```bash
conda activate alpamayo
```

### 4. Run experiments

**Basic inference test (5 scenes):**
```bash
python experiments/workstation/basic_inference_test.py
```

**Interactive notebook:**
```bash
jupyter notebook experiments/workstation/interactive_inference.ipynb
```

## Files

| File | Description |
|------|-------------|
| `setup.sh` | Automated setup script |
| `basic_inference_test.py` | Test model on 5 random scenes |
| `interactive_inference.ipynb` | Interactive exploration notebook |

## Troubleshooting

### CUDA not found
```bash
# Check NVIDIA driver
nvidia-smi

# If not installed, install drivers first
```

### HuggingFace dataset access denied
1. Go to https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
2. Request access (may require agreeing to terms)
3. Run `huggingface-cli login` again

### Out of memory
- The model requires ~20GB VRAM
- Close other GPU applications
- Reduce `num_traj_samples` in the code if needed

### Model download slow
First run downloads ~20GB model weights. This is normal.

## Output Files

Results are saved to:
```
experiments/workstation/inference_results/
└── basic_inference_results.json
```

Contains:
- Chain-of-Causation (CoC) reasoning text
- Predicted trajectories
- Ground truth trajectories
- minADE metrics
- Timing information
