#!/bin/bash
# =============================================================================
# Local (Mac/MPS) Setup Script
# Setup for Alpamayo-R1 experiments on Apple Silicon Macs
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "ALPAMAYO-R1 LOCAL (MPS) SETUP"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. Check for conda
# -----------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Detect architecture
    if [[ $(uname -m) == "arm64" ]]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O /tmp/miniconda.sh
    else
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O /tmp/miniconda.sh
    fi

    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
    echo "Conda installed. Please restart your shell and run this script again."
    exit 0
fi

# -----------------------------------------------------------------------------
# 2. Create environment
# -----------------------------------------------------------------------------
ENV_NAME="alpamayo-local"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Using existing environment."
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -n $ENV_NAME python=3.12 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Active environment: $CONDA_DEFAULT_ENV"

# -----------------------------------------------------------------------------
# 3. Install PyTorch for MPS (Apple Silicon)
# -----------------------------------------------------------------------------
echo ""
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Verify MPS
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# -----------------------------------------------------------------------------
# 4. Install Alpamayo dependencies (no flash-attn on Mac)
# -----------------------------------------------------------------------------
echo ""
echo "Installing Alpamayo-R1 dependencies..."

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

# Install alpamayo package (--no-deps since we install dependencies manually)
pip install -e tools/alpamayo/ --no-deps

# Core dependencies
pip install \
    "transformers==4.57.1" \
    "accelerate>=1.12.0" \
    huggingface-hub \
    "einops>=0.8.1" \
    "hydra-core>=1.3.2" \
    "hydra-colorlog>=1.2.0" \
    "physical_ai_av>=0.1.0" \
    "av>=16.0.1" \
    "pillow>=12.0.0"

# Additional dependencies for experiments
pip install pandas numpy matplotlib jupyter ipykernel ipywidgets tqdm rich datasets mediapy

# -----------------------------------------------------------------------------
# 5. HuggingFace login (for gated dataset)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "HUGGINGFACE LOGIN"
echo "=============================================="
echo "The dataset requires HuggingFace authentication."
echo "You need a HuggingFace account with access to: nvidia/PhysicalAI-Autonomous-Vehicles"
echo ""

if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to HuggingFace:"
    huggingface-cli login
else
    echo "Already logged in as: $(huggingface-cli whoami)"
fi

# -----------------------------------------------------------------------------
# 6. Verify setup
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "VERIFICATION"
echo "=============================================="

python << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    # Test MPS with a simple tensor operation
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device="mps")
        y = x * 2
        print(f"MPS test: OK ({y.device})")
    except Exception as e:
        print(f"MPS test: FAILED - {e}")

try:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    print("Alpamayo-R1: OK")
except ImportError as e:
    print(f"Alpamayo-R1: FAILED - {e}")

try:
    import pandas
    print("Pandas: OK")
except ImportError:
    print("Pandas: FAILED")
EOF

echo ""
echo "=============================================="
echo "SETUP COMPLETE"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run basic inference test:"
echo "  python experiments/local/basic_inference_test.py"
echo ""
echo "To start Jupyter notebook:"
echo "  jupyter notebook experiments/local/interactive_inference.ipynb"
echo ""
echo "NOTE: This setup uses 'eager' attention instead of flash-attn."
echo "      Model inference will be slower than on NVIDIA GPUs."
echo ""
