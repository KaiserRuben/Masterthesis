# Alpamayo-R1 Docker Inference

Run trajectory prediction on cloud GPU providers.

## Quick Start

### 1. Build Image

```bash
cd /Users/kaiser/Projects/Masterarbeit
docker build -f infrastructure/docker/Dockerfile.alpamayo -t alpamayo-inference .
```

### 2. Run Locally (with NVIDIA GPU)

```bash
export HF_TOKEN="your_huggingface_token"

docker run --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/data/BND-002:/app/data/BND-002:ro \
  -v $(pwd)/data/alpamayo_outputs:/app/data/alpamayo_outputs \
  alpamayo-inference \
  python /app/run_inference.py \
    --clip-ids /app/data/BND-002/missing_clips.txt \
    -o /app/data/alpamayo_outputs/docker_inference.json
```

### 3. Using Docker Compose

```bash
export HF_TOKEN="your_huggingface_token"
cd infrastructure/docker
docker-compose up
```

---

## Cloud Provider Instructions

### RunPod

1. **Push image to Docker Hub:**
   ```bash
   docker tag alpamayo-inference:latest yourusername/alpamayo-inference:latest
   docker push yourusername/alpamayo-inference:latest
   ```

2. **Create Pod:**
   - Template: `yourusername/alpamayo-inference:latest`
   - GPU: RTX 4090 or A100 (24GB+ VRAM required)
   - Container Disk: 50GB
   - Volume: Mount for output

3. **Run:**
   ```bash
   # SSH into pod, then:
   export HF_TOKEN="your_token"
   python /app/run_inference.py \
     --clip-ids /app/missing_clips.txt \
     -o /workspace/inference_output.json
   ```

### Vast.ai

1. Search for: `RTX 4090` or `A100`
2. Use Docker image: `yourusername/alpamayo-inference:latest`
3. Set environment: `HF_TOKEN=your_token`
4. Run command as above

### Lambda Labs

1. Launch instance with A100
2. Install Docker + NVIDIA Container Toolkit
3. Pull and run image as shown above

### Modal.com (Serverless)

```python
# modal_inference.py
import modal

app = modal.App("alpamayo-inference")

image = modal.Image.from_dockerfile(
    "infrastructure/docker/Dockerfile.alpamayo",
    context_mount=modal.Mount.from_local_dir(".", remote_path="/app")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_inference(clip_ids: list[str]):
    import subprocess
    # Write clip_ids to file
    with open("/tmp/clips.txt", "w") as f:
        f.write("\n".join(clip_ids))

    subprocess.run([
        "python", "/app/run_inference.py",
        "--clip-ids", "/tmp/clips.txt",
        "-o", "/tmp/output.json"
    ])

    with open("/tmp/output.json") as f:
        return f.read()
```

---

## Hardware Requirements

| GPU | VRAM | Est. Time (47 clips) |
|-----|------|----------------------|
| RTX 4090 | 24GB | ~1.5-2 hours |
| A100 40GB | 40GB | ~1 hour |
| A100 80GB | 80GB | ~45 min |
| H100 | 80GB | ~30 min |

---

## Files

- `Dockerfile.alpamayo` - Docker image definition
- `run_inference.py` - Inference script
- `docker-compose.yml` - Local compose config
- `missing_clips.txt` - 47 clip IDs to process (copied from BND-002)

---

## Output

Results are saved to JSON with structure:
```json
{
  "metadata": {
    "model_id": "nvidia/Alpamayo-R1-10B",
    "num_samples": 47,
    "gpu": "NVIDIA A100",
    "timestamp": "2026-02-02T..."
  },
  "results": [
    {
      "clip_id": "abc123...",
      "min_ade": 1.234,
      "predicted_trajectory": [[x1,y1], ...],
      "coc_reasoning": "The vehicle should..."
    }
  ]
}
```

After inference, merge with existing data:
```bash
python experiments/Phase-3_Boundaries/BND-002_data_first_transitions/run_missing_inference.py \
  --merge data/alpamayo_outputs/docker_inference.json
```
