#!/bin/bash
# Build and optionally push Alpamayo inference Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

IMAGE_NAME="${DOCKER_USER:-alpamayo}-inference"
TAG="${1:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Context: ${PROJECT_ROOT}"

cd "$PROJECT_ROOT"

docker build \
    -f infrastructure/docker/Dockerfile.alpamayo \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo ""
echo "✓ Built: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To run locally:"
echo "  docker run --gpus all -e HF_TOKEN=\$HF_TOKEN -v \$(pwd)/data:/app/data ${IMAGE_NAME}:${TAG}"
echo ""
echo "To push to Docker Hub:"
echo "  docker tag ${IMAGE_NAME}:${TAG} yourusername/${IMAGE_NAME}:${TAG}"
echo "  docker push yourusername/${IMAGE_NAME}:${TAG}"
