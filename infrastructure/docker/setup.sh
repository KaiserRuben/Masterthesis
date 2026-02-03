#!/bin/bash
# Setup script for Alpamayo-R1 inference on cloud GPU

set -e

echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing other dependencies ==="
pip install -r requirements.txt

echo "=== Setup complete ==="
