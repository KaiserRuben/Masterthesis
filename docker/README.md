# Docker Setup for Alpamayo-R1 Experiments

Self-contained Docker environment for running experiments on any GPU workstation.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with 24GB+ VRAM
- HuggingFace account with access to `nvidia/PhysicalAI-Autonomous-Vehicles`

### Install NVIDIA Container Toolkit (if needed)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

```bash
cd docker/

# 1. Setup environment
make setup                    # Creates .env from template
nano .env                     # Add your HF_TOKEN

# 2. Build image
make build

# 3. Run experiments
make test                     # Basic inference test (5 scenes)
make notebook                 # Start Jupyter (http://localhost:8888)
make shell                    # Interactive bash
```

## Directory Structure

```
docker/
├── Dockerfile                # Main image definition
├── docker-compose.yml        # Service orchestration
├── Makefile                  # Convenience commands
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .env                      # Your config (gitignored)
├── scripts/
│   └── entrypoint.sh         # Container entrypoint with validation
└── README.md                 # This file
```

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for gated dataset |
| `JUPYTER_PORT` | No | Jupyter port (default: 8888) |

### Volumes

| Volume | Purpose |
|--------|---------|
| `alpamayo-hf-cache` | Cached HuggingFace models (~20GB) |
| `alpamayo-torch-cache` | Cached Torch models |
| `./inference_results` | Experiment outputs |

Model weights are cached in Docker volumes, so they only download once.

## Commands

### Using Make

```bash
make build       # Build Docker image
make test        # Run basic inference test
make notebook    # Start Jupyter notebook
make shell       # Interactive bash shell
make clean       # Remove containers/images
make clean-all   # Also remove cached models
```

### Using Docker Compose directly

```bash
docker-compose up test        # Run test
docker-compose up notebook    # Start notebook
docker-compose run shell      # Interactive shell
docker-compose down           # Stop all
```

### Run arbitrary commands

```bash
docker-compose run shell python my_script.py
docker-compose run shell nvidia-smi
```

## Troubleshooting

### GPU not detected

```
[ERROR] GPU: nvidia-smi not found
```

**Fix:** Ensure NVIDIA Container Toolkit is installed and Docker is restarted:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### HuggingFace authentication failed

```
[WARN] HuggingFace: No token found
```

**Fix:** Add your token to `.env`:
```bash
echo "HF_TOKEN=hf_your_token_here" >> .env
```

Get token from: https://huggingface.co/settings/tokens

### Permission denied on results directory

**Fix:** Check ownership of mounted directories:
```bash
sudo chown -R $USER:$USER ../experiments/workstation/inference_results
```

### Out of memory

The model requires ~20GB VRAM. If you get OOM errors:
1. Close other GPU applications
2. Reduce batch size or `num_traj_samples` in the code

### Slow first run

First run downloads ~20GB of model weights. Subsequent runs use cached volumes and start much faster.

## Development

To mount source code for development (edit files outside container):

```yaml
# Uncomment in docker-compose.yml:
volumes:
  - ../alpamayo:/app/alpamayo
  - ../experiments:/app/experiments
```

Then rebuild: `make build`
