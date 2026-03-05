#!/usr/bin/env python3
"""Evaluate VLM forced-decoding accuracy on ImageNet.

Streams ImageNet validation from HuggingFace (no local download needed).
For each image with a known ground-truth label, measures:
  - P(correct_label | image, prompt)  via teacher-forced log-prob
  - Approximate rank of correct label among all 1000 classes (first-token)
  - What the model predicts instead (top-1 from first-token stage)

Outputs a CSV for downstream analysis (per-class difficulty, outliers, etc.)

Usage:
  python run.py --device mps
  python run.py --max-images 100 --device mps          # quick test
  python run.py --max-per-class 5 --device mps         # 5 per class
  python run.py --resume results/run_001.csv            # continue
"""

import argparse
import csv
import json
import math
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3.5-9B"
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)
PROMPT = "What is the main object in this image? Answer with just the object name:"

CSV_COLUMNS = [
    "image_idx",
    "ground_truth_idx",
    "ground_truth_label",
    "correct_log_prob",
    "correct_log_prob_norm",
    "correct_n_tokens",
    "correct_rank_approx",
    "top1_label",
    "top1_first_token_lp",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_imagenet_labels(cache_dir: Path) -> list[str]:
    cache_file = cache_dir / "imagenet_labels.json"
    if not cache_file.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, cache_file)
    return json.loads(cache_file.read_text())


def stream_imagenet_val(
    max_images: int | None = None,
    max_per_class: int | None = None,
) -> list[dict]:
    """Stream ImageNet validation from HuggingFace. Returns list of {image, label}."""
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    samples = []
    class_counts: dict[int, int] = defaultdict(int)
    for sample in tqdm(ds, desc="Streaming ImageNet val", total=50_000):
        label_idx = sample["label"]

        if max_per_class and class_counts[label_idx] >= max_per_class:
            continue
        class_counts[label_idx] += 1

        samples.append({
            "image": sample["image"].convert("RGB"),
            "idx": label_idx,
        })

        if max_images and len(samples) >= max_images:
            break

    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return processor, model


def make_messages(image: Image.Image, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


THINK_END_TOKEN = "</think>"


def _find_think_end(token_ids, tokenizer) -> int | None:
    """Return index of the </think> token in generated ids, or None."""
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    for i, tid in enumerate(token_ids):
        if tid.item() == think_end_id:
            return i
    return None


def score_image(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    correct_label: str,
    all_labels: list[str],
    device: str,
    enable_thinking: bool = True,
    max_thinking_tokens: int = 2000,
) -> dict:
    """Score a single image: free generation + teacher-forced P(correct).

      1. Free generation → model's predicted label + first-token ranking
         With thinking: generate until </think>, then score first answer token.
         Without thinking: model answers directly.
      2. Full teacher-forced pass → exact P(correct_label)
    """
    messages = make_messages(image, prompt)

    # --- Pass 1: free generation → model's prediction ---
    prefix_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(device)
    prefix_len = prefix_inputs["input_ids"].shape[1]

    max_gen = max_thinking_tokens + 50 if enable_thinking else 50
    with torch.no_grad():
        gen_ids = model.generate(**prefix_inputs, max_new_tokens=max_gen)

    generated_tokens = gen_ids[0][prefix_len:]

    if enable_thinking:
        think_end_pos = _find_think_end(generated_tokens, processor.tokenizer)
        if think_end_pos is not None:
            # Answer starts after </think>\n\n
            answer_tokens = generated_tokens[think_end_pos + 1:]
        else:
            # Thinking didn't finish — use whatever we got
            answer_tokens = generated_tokens
        # Build the context up to (and including) </think> for scoring
        if think_end_pos is not None:
            answer_start = prefix_len + think_end_pos + 1
        else:
            answer_start = prefix_len
    else:
        answer_tokens = generated_tokens
        answer_start = prefix_len

    generated_text = processor.tokenizer.decode(
        answer_tokens, skip_special_tokens=True
    ).strip()

    # Match against label list
    top1_label = generated_text
    generated_lower = generated_text.lower()
    for lbl in all_labels:
        if lbl.lower() == generated_lower or lbl.lower() in generated_lower:
            top1_label = lbl
            break

    # First-token log-prob of the answer (after thinking if enabled)
    # Use the full generated sequence up to the answer start as context
    context_ids = gen_ids[0][:answer_start].unsqueeze(0)
    with torch.no_grad():
        context_logits = model(input_ids=context_ids).logits[0, -1, :]
    log_probs_first = F.log_softmax(context_logits, dim=-1)

    if len(answer_tokens) > 0:
        top1_lp = log_probs_first[answer_tokens[0]].item()
    else:
        top1_lp = float("-inf")

    # Approximate rank: score all labels by first token
    label_first_scores = []
    for lbl in all_labels:
        toks = processor.tokenizer.encode(lbl, add_special_tokens=False)
        if toks:
            label_first_scores.append((lbl, log_probs_first[toks[0]].item()))
    label_first_scores.sort(key=lambda x: x[1], reverse=True)
    rank = next(
        (i + 1 for i, (lbl, _) in enumerate(label_first_scores) if lbl == correct_label),
        len(label_first_scores),
    )

    # --- Pass 2: full teacher-forced scoring of correct label ---
    # Re-encode clean prefix (without generated thinking) for consistent scoring
    prefix_for_forced = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)
    forced_prefix_len = prefix_for_forced["input_ids"].shape[1]

    messages_with_answer = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": correct_label}]}
    ]
    full_inputs = processor.apply_chat_template(
        messages_with_answer,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)
    full_ids = full_inputs["input_ids"]

    with torch.no_grad():
        logits = model(**full_inputs).logits

    n_new = full_ids.shape[1] - forced_prefix_len
    n_label_tokens = max(n_new - 1, 1)  # exclude end-of-turn token

    total_lp = 0.0
    for i in range(n_label_tokens):
        target = full_ids[0, forced_prefix_len + i]
        lp = F.log_softmax(logits[0, forced_prefix_len + i - 1, :], dim=-1)
        total_lp += lp[target].item()

    return {
        "correct_log_prob": total_lp,
        "correct_log_prob_norm": total_lp / n_label_tokens,
        "correct_n_tokens": n_label_tokens,
        "correct_rank_approx": rank,
        "top1_label": top1_label,
        "top1_first_token_lp": top1_lp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results_path: Path):
    """Print per-class and overall statistics from the results CSV."""
    rows = []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No results to summarize.")
        return

    n = len(rows)
    top1_correct = sum(1 for r in rows if r["top1_label"] == r["ground_truth_label"])
    top5_correct = sum(1 for r in rows if int(r["correct_rank_approx"]) <= 5)
    mean_lp = sum(float(r["correct_log_prob_norm"]) for r in rows) / n
    mean_prob = sum(math.exp(float(r["correct_log_prob_norm"])) for r in rows) / n

    print(f"\n{'=' * 65}")
    print(f"  Overall  ({n} images)")
    print(f"{'=' * 65}")
    print(f"  Top-1 accuracy (first-token approx):  {top1_correct/n:.1%}")
    print(f"  Top-5 accuracy (first-token approx):  {top5_correct/n:.1%}")
    print(f"  Mean normalized log-prob of correct:   {mean_lp:+.4f}")
    print(f"  Mean P(correct) (approx):             {mean_prob:.4f}")

    # Per-class stats
    by_class: dict[str, list] = defaultdict(list)
    for r in rows:
        by_class[r["ground_truth_label"]].append(r)

    class_stats = []
    for lbl, class_rows in by_class.items():
        cn = len(class_rows)
        c_mean_lp = sum(float(r["correct_log_prob_norm"]) for r in class_rows) / cn
        c_top1 = sum(1 for r in class_rows if r["top1_label"] == lbl) / cn
        c_mean_rank = sum(int(r["correct_rank_approx"]) for r in class_rows) / cn
        class_stats.append((lbl, cn, c_mean_lp, c_top1, c_mean_rank))

    # Hardest classes
    class_stats.sort(key=lambda x: x[2])
    print(f"\n  Hardest 10 classes (lowest mean log-prob):")
    for lbl, cn, mlp, t1, mr in class_stats[:10]:
        print(f"    {lbl:30s}  n={cn:3d}  mean_lp={mlp:+.4f}  top1={t1:.0%}  mean_rank={mr:.0f}")

    # Easiest classes
    print(f"\n  Easiest 10 classes:")
    for lbl, cn, mlp, t1, mr in class_stats[-10:][::-1]:
        print(f"    {lbl:30s}  n={cn:3d}  mean_lp={mlp:+.4f}  top1={t1:.0%}  mean_rank={mr:.0f}")

    # Biggest outlier images (lowest P(correct))
    rows_sorted = sorted(rows, key=lambda r: float(r["correct_log_prob_norm"]))
    print(f"\n  10 worst outlier images:")
    for r in rows_sorted[:10]:
        print(
            f"    idx={r['image_idx']:>6s}  "
            f"gt={r['ground_truth_label']:20s}  "
            f"pred={r['top1_label']:20s}  "
            f"lp={float(r['correct_log_prob_norm']):+.4f}  "
            f"rank={r['correct_rank_approx']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM forced-decoding accuracy on ImageNet"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-images", type=int, default=None, help="Limit total images")
    parser.add_argument("--max-per-class", type=int, default=None, help="Limit images per class")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable model thinking (default: enabled)")
    parser.add_argument("--max-thinking-tokens", type=int, default=2000,
                        help="Max tokens for thinking phase (default: 2000)")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from existing CSV")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    cache_dir = Path(__file__).parent / ".cache"
    labels = load_imagenet_labels(cache_dir)
    print(f"Loaded {len(labels)} ImageNet labels")

    # Stream from HuggingFace
    print("Streaming ImageNet validation from HuggingFace ...")
    samples = stream_imagenet_val(
        max_images=args.max_images,
        max_per_class=args.max_per_class,
    )
    print(f"Loaded {len(samples)} images")

    # Resume support
    done_indices: set[int] = set()
    if args.resume and args.resume.exists():
        with open(args.resume) as f:
            for row in csv.DictReader(f):
                done_indices.add(int(row["image_idx"]))
        print(f"Resuming: {len(done_indices)} images already done")

    # Output file
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.resume if args.resume else args.out_dir / f"run_{timestamp}.csv"
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    # Load model
    print(f"Loading {args.model} on {args.device} ...")
    processor, model = load_model(args.model, args.device)

    # Run evaluation
    with open(out_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, sample in enumerate(tqdm(samples, desc="Scoring")):
            if i in done_indices:
                continue

            gt_idx = sample["idx"]
            gt_label = labels[gt_idx]

            try:
                result = score_image(
                    processor, model, sample["image"], args.prompt,
                    gt_label, labels, args.device,
                    enable_thinking=args.thinking,
                    max_thinking_tokens=args.max_thinking_tokens,
                )
                writer.writerow({
                    "image_idx": i,
                    "ground_truth_idx": gt_idx,
                    "ground_truth_label": gt_label,
                    **result,
                })
                csvfile.flush()
            except Exception as e:
                print(f"\n  ERROR image {i} (class {gt_label}): {e}")

    print(f"\nResults saved to {out_path}")
    print_summary(out_path)


if __name__ == "__main__":
    main()
