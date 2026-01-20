#!/bin/bash
# =============================================================================
# Container Entrypoint
# Validates environment and runs experiments
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Colors for output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  ALPAMAYO-R1 EXPERIMENT RUNNER"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
info "Running environment checks..."
CHECKS_PASSED=true

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$GPU_NAME" ]; then
        success "GPU: $GPU_NAME ($GPU_MEM)"
    else
        error "GPU: nvidia-smi found but no GPU detected"
        CHECKS_PASSED=false
    fi
else
    error "GPU: nvidia-smi not found - is nvidia-container-toolkit installed?"
    error "     Run: docker run --gpus all ..."
    CHECKS_PASSED=false
fi

# Check CUDA in PyTorch
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "True" ]; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    success "PyTorch: $TORCH_VERSION (CUDA enabled)"
else
    error "PyTorch: CUDA not available"
    CHECKS_PASSED=false
fi

# Check HuggingFace auth
if [ -n "$HF_TOKEN" ]; then
    success "HuggingFace: Token provided via HF_TOKEN"
    # Login with token
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>/dev/null
elif [ -f "$HF_HOME/token" ]; then
    success "HuggingFace: Token found in cache"
else
    warn "HuggingFace: No token found"
    warn "     Set HF_TOKEN env var or mount ~/.cache/huggingface"
    warn "     Dataset download may fail for gated repos"
fi

# Check directories
for dir in "$DATA_DIR" "$CACHE_DIR" "$RESULTS_DIR"; do
    if [ -w "$dir" ]; then
        success "Directory writable: $dir"
    else
        error "Directory not writable: $dir"
        CHECKS_PASSED=false
    fi
done

# Check alpamayo package
if python -c "from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1" 2>/dev/null; then
    success "Alpamayo-R1: Package installed"
else
    error "Alpamayo-R1: Package not found"
    CHECKS_PASSED=false
fi

echo ""

# -----------------------------------------------------------------------------
# Abort if checks failed
# -----------------------------------------------------------------------------
if [ "$CHECKS_PASSED" = false ]; then
    error "Environment validation failed. See errors above."
    echo ""
    echo "Common fixes:"
    echo "  - GPU not detected: docker run --gpus all ..."
    echo "  - HF token missing: docker run -e HF_TOKEN=hf_xxx ..."
    echo "  - Permission issues: check volume mount permissions"
    echo ""
    exit 1
fi

success "All checks passed!"
echo ""

# -----------------------------------------------------------------------------
# Execute command
# -----------------------------------------------------------------------------
if [ "$1" = "test" ]; then
    shift  # Remove 'test' from args
    info "Running basic inference test..."
    exec python /app/experiments/workstation/basic_inference_test.py "$@"

elif [ "$1" = "notebook" ]; then
    info "Starting Jupyter notebook..."
    exec jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/app/experiments/workstation

elif [ "$1" = "classify" ]; then
    shift
    info "Running batch scene classification..."
    exec python /app/experiments/batch_scene_classification.py "$@"

elif [ "$1" = "bash" ] || [ -z "$1" ]; then
    info "Starting interactive shell..."
    exec /bin/bash

else
    # Run arbitrary command
    exec "$@"
fi
