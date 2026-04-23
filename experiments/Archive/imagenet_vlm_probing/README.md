# ImageNet VLM Probing

Evaluates VLM forced-decoding classification accuracy on ImageNet validation set.

## Purpose

Establishes classification baselines for boundary testing by measuring how well VLMs perform on standard image classification when forced to decode specific labels. This informs which classes are "easy" vs. "hard" and where decision boundaries are likely concentrated.

## Metrics

For each image with a known ground-truth label:
- **P(correct_label | image, prompt)** via teacher-forced log-prob
- **Approximate rank** of correct label among all 1000 ImageNet classes (first-token stage)
- **Top-1 prediction** — what the model predicts instead

## Model

`Qwen/Qwen3.5-9B` with prompt: `"What is the main object in this image? Answer with just the object name:"`

## Usage

```bash
python run.py --device mps                      # full run
python run.py --max-images 100 --device mps     # quick test
python run.py --max-per-class 5 --device mps    # 5 per class
python run.py --resume results/run_001.csv      # continue from checkpoint
```

## Output

CSV in `results/` with per-image classification results for downstream analysis (per-class difficulty, outliers, boundary candidates).
