#!/usr/bin/env python3
"""Evaluate VLM forced-decoding accuracy on ImageNet (closed-set).

Streams ImageNet validation from HuggingFace, filtered to 10 categories.
For each image, the model is given the category options in the prompt and:
  1. Free generation (with or without thinking) → raw model prediction
  2. Teacher-forced scoring of every category → exact P(label) and ranking

Two modes via --thinking / --no-thinking:
  --thinking (default):  Model reasons in <think>...</think> before answering.
  --no-thinking:         Model answers directly. Faster.

Outputs two parquet tables per run for downstream analysis:
  images.parquet  — one row per image (ground truth, generation, reasoning)
  scores.parquet  — one row per (image × candidate) (log-probs, ranks)

Usage:
  python run.py --device mps                            # default (thinking on)
  python run.py --no-thinking --device mps              # direct answers, faster
  python run.py --max-images 100 --device mps           # quick test
  python run.py --max-per-class 5 --device mps          # 5 per class
  python run.py --max-thinking-tokens 4000 --device mps # longer reasoning
  python run.py --resume results/run_20260306_130830    # continue from run dir
"""

import argparse
import gc
import json
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
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
CATEGORIES = [
    "tench", "goldfish", "great white shark", "tiger shark",
    "hammerhead shark", "electric ray", "stingray", "cock", "hen", "ostrich",
]
PROMPT = (
    "What is the main object in this image? Answer with exactly one of these options: "
    + ", ".join(CATEGORIES)
    + "."
)

THINK_END_TOKEN = "</think>"


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
    labels: list[str],
    categories: list[str],
    max_images: int | None = None,
    max_per_class: int | None = None,
) -> list[dict]:
    """Stream ImageNet validation from HuggingFace, filtered to categories."""
    category_indices = {labels.index(c) for c in categories}
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    samples = []
    class_counts: dict[int, int] = defaultdict(int)
    for sample in tqdm(ds, desc="Streaming ImageNet val", total=50_000):
        label_idx = sample["label"]

        if label_idx not in category_indices:
            continue

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
# VLM Scorer
# ---------------------------------------------------------------------------

class VLMScorer(ABC):
    """Abstract VLM scorer following smoo SUT conventions."""

    _model: AutoModelForImageTextToText
    _processor: AutoProcessor
    _device: torch.device
    _enable_thinking: bool
    _max_thinking_tokens: int

    def __init__(
        self,
        model_id: str,
        device: str,
        enable_thinking: bool = True,
        max_thinking_tokens: int = 2000,
        dtype: Optional[torch.dtype] = None,
        max_pixels: Optional[int] = None,
    ) -> None:
        self._device = torch.device(device)
        self._enable_thinking = enable_thinking
        self._max_thinking_tokens = max_thinking_tokens

        proc_kwargs = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        self._processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype or torch.float16,
            device_map=device,
        )
        self._model.eval()

    @abstractmethod
    def _prepare_inputs(
        self, image: Image.Image, prompt: str, enable_thinking: bool,
    ) -> dict:
        """Tokenize messages into model inputs on device."""
        ...

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    def _find_think_end(self, token_ids) -> Optional[int]:
        think_end_id = self.tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
        for i, tid in enumerate(token_ids):
            if tid.item() == think_end_id:
                return i
        return None

    def generate(
        self, image: Image.Image, prompt: str,
    ) -> tuple[str, Optional[str], Optional[torch.Tensor]]:
        """Generate a response.

        Returns (answer_text, thinking_text, thinking_ids):
          - answer_text: the model's answer after any thinking
          - thinking_text: decoded thinking trace, or None
          - thinking_ids: raw thinking token ids for score_categories, or None
        """
        inputs = self._prepare_inputs(image, prompt, self._enable_thinking)
        prefix_len = inputs["input_ids"].shape[1]

        max_gen = self._max_thinking_tokens + 50 if self._enable_thinking else 50
        with torch.no_grad():
            gen_ids = self._model.generate(**inputs, max_new_tokens=max_gen)

        generated_tokens = gen_ids[0][prefix_len:]
        answer_tokens = generated_tokens
        thinking_ids = None
        thinking_text = None

        if self._enable_thinking:
            think_end_pos = self._find_think_end(generated_tokens)
            if think_end_pos is not None:
                thinking_ids = generated_tokens[: think_end_pos + 1]
                answer_tokens = generated_tokens[think_end_pos + 1:]
                thinking_text = self.tokenizer.decode(
                    thinking_ids, skip_special_tokens=True,
                ).strip()

        answer_text = self.tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        ).strip()
        return answer_text, thinking_text, thinking_ids

    def score_categories(
        self, image: Image.Image, prompt: str, categories: list[str],
        thinking_ids: Optional[torch.Tensor] = None,
    ) -> list[tuple[str, float, float, int]]:
        """Force-score each category.

        If thinking_ids is provided (from generate()), the thinking trace is
        appended to the prompt so scoring is conditioned on the same context
        as free generation. Image features are always properly included.
        """
        if thinking_ids is not None and self._enable_thinking:
            inputs = self._prepare_inputs(image, prompt, enable_thinking=True)
            n_think = len(thinking_ids)
            # Append thinking tokens to input_ids and extend sequence-length tensors
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], thinking_ids.unsqueeze(0)], dim=1,
            )
            for key in ("attention_mask", "mm_token_type_ids"):
                if key in inputs:
                    pad_val = 1 if key == "attention_mask" else 0
                    extra = torch.full(
                        (1, n_think), pad_val,
                        device=self._device,
                        dtype=inputs[key].dtype,
                    )
                    inputs[key] = torch.cat([inputs[key], extra], dim=1)
            with torch.no_grad():
                prefix_out = self._model(**inputs, use_cache=True)
        else:
            inputs = self._prepare_inputs(image, prompt, enable_thinking=False)
            with torch.no_grad():
                prefix_out = self._model(**inputs, use_cache=True)

        prefix_kvs = prefix_out.past_key_values
        last_logits = prefix_out.logits[0, -1, :]

        scored = []
        for lbl in categories:
            label_tok_ids = self.tokenizer.encode(lbl, add_special_tokens=False)
            label_ids = torch.tensor(label_tok_ids, device=self._device)
            n_tokens = len(label_tok_ids)

            if n_tokens == 0:
                scored.append((lbl, float("-inf"), float("-inf"), 0))
                continue

            total_lp = F.log_softmax(last_logits, dim=-1)[label_ids[0]].item()

            if n_tokens > 1:
                with torch.no_grad():
                    cont_out = self._model(
                        input_ids=label_ids[:-1].unsqueeze(0),
                        past_key_values=prefix_kvs,
                    )
                for i in range(n_tokens - 1):
                    lp = F.log_softmax(cont_out.logits[0, i, :], dim=-1)
                    total_lp += lp[label_ids[i + 1]].item()

            norm_lp = total_lp / n_tokens
            scored.append((lbl, total_lp, norm_lp, n_tokens))

        return sorted(scored, key=lambda x: x[2], reverse=True)

    def score_image(
        self,
        image: Image.Image,
        prompt: str,
        categories: list[str],
    ) -> tuple[str, Optional[str], list[tuple[str, float, float, int]]]:
        """Return (generated_text, thinking_text, scored)."""
        generated_text, thinking_text, thinking_ids = self.generate(image, prompt)
        scored = self.score_categories(image, prompt, categories, thinking_ids=thinking_ids)
        return generated_text, thinking_text, scored

    def cleanup(self) -> None:
        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()


class Qwen3VLScorer(VLMScorer):
    """Scorer for Qwen3-VL models (standard attention)."""

    def __init__(self, model_id, device, enable_thinking=True,
                 max_thinking_tokens=2000, dtype=None, max_pixels=None):
        super().__init__(
            model_id, device, enable_thinking, max_thinking_tokens, dtype,
            max_pixels=max_pixels if max_pixels is not None else 512 * 28 * 28,
        )

    def _prepare_inputs(
        self, image: Image.Image, prompt: str, enable_thinking: bool,
    ) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        return self._processor(
            text=[text], images=[image], return_tensors="pt",
        ).to(self._device)


class Qwen35Scorer(VLMScorer):
    """Scorer for Qwen3.5 models (DeltaNet architecture).

    Defaults max_pixels to 512*28*28 (~400 image tokens) to avoid
    memory blowup in the DeltaNet fallback implementation on MPS/CPU.
    """

    def __init__(self, model_id, device, enable_thinking=True,
                 max_thinking_tokens=2000, dtype=None, max_pixels=None):
        super().__init__(
            model_id, device, enable_thinking, max_thinking_tokens, dtype,
            max_pixels=max_pixels if max_pixels is not None else 512 * 28 * 28,
        )

    def _prepare_inputs(
        self, image: Image.Image, prompt: str, enable_thinking: bool,
    ) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        return self._processor(
            text=[text], images=[image], return_tensors="pt",
        ).to(self._device)


SCORER_REGISTRY: dict[str, type[VLMScorer]] = {
    "qwen3-vl": Qwen3VLScorer,
    "qwen3.5": Qwen35Scorer,
}


def create_scorer(
    model_id: str,
    device: str,
    enable_thinking: bool = True,
    max_thinking_tokens: int = 2000,
) -> VLMScorer:
    model_lower = model_id.lower()
    for key, cls in SCORER_REGISTRY.items():
        if key in model_lower:
            return cls(model_id, device, enable_thinking, max_thinking_tokens)
    raise ValueError(
        f"No scorer for model '{model_id}'. "
        f"Known prefixes: {list(SCORER_REGISTRY.keys())}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load images and scores tables from a run directory."""
    return (
        pd.read_parquet(run_dir / "images.parquet"),
        pd.read_parquet(run_dir / "scores.parquet"),
    )


def print_summary(run_dir: Path):
    """Print per-class and overall statistics from a run directory."""
    images, scores = load_run(run_dir)
    if images.empty:
        print("No results to summarize.")
        return

    n_images = len(images)

    # Join scores with ground truth for convenience
    gt_scores = scores.merge(
        images[["image_idx", "ground_truth_label"]], on="image_idx",
    )
    # Ground-truth scores: where candidate == ground truth
    gt = gt_scores[gt_scores["candidate_label"] == gt_scores["ground_truth_label"]]

    # Top-1 and top-2 per image
    top1 = gt_scores[gt_scores["candidate_rank"] == 1]
    top2 = gt_scores[gt_scores["candidate_rank"] == 2]
    top1_forced = (top1["candidate_label"] == top1["ground_truth_label"]).sum()

    # Free generation accuracy
    gen_correct = images["generated_text"].str.lower().eq(
        images["ground_truth_label"].str.lower()
    ).sum()

    # Top-1 vs top-2 margin per image
    margins = (
        top1.set_index("image_idx")["log_prob_norm"]
        .subtract(top2.set_index("image_idx")["log_prob_norm"])
    )

    top3_correct = (gt["candidate_rank"] <= 3).sum()
    mean_lp = gt["log_prob_norm"].mean()
    mean_prob = gt["log_prob_norm"].apply(lambda x: 2.718281828 ** x).mean()

    print(f"\n{'=' * 75}")
    print(f"  Overall  ({n_images} images)")
    print(f"{'=' * 75}")
    print(f"  Top-1 accuracy (free generation):     {gen_correct/n_images:.1%}")
    print(f"  Top-1 accuracy (teacher-forced):       {top1_forced/n_images:.1%}")
    print(f"  Top-3 accuracy (teacher-forced):       {top3_correct/n_images:.1%}")
    print(f"  Mean normalized log-prob of correct:   {mean_lp:+.4f}")
    print(f"  Mean P(correct) (approx):             {mean_prob:.4f}")
    print(f"  Mean top1-top2 margin:                {margins.mean():+.4f}")

    # Per-class stats
    class_stats = gt.groupby("ground_truth_label").agg(
        n=("image_idx", "count"),
        mean_lp=("log_prob_norm", "mean"),
        mean_rank=("candidate_rank", "mean"),
    )
    class_top1 = top1.copy()
    class_top1["correct"] = class_top1["candidate_label"] == class_top1["ground_truth_label"]
    class_stats["top1"] = class_top1.groupby("ground_truth_label")["correct"].mean()
    margins_df = margins.reset_index()
    margins_df.columns = ["image_idx", "margin"]
    margins_df = margins_df.merge(
        images[["image_idx", "ground_truth_label"]].drop_duplicates(), on="image_idx",
    )
    class_stats["mean_margin"] = margins_df.groupby("ground_truth_label")["margin"].mean()
    class_stats = class_stats.sort_values("mean_lp")

    print(f"\n  Hardest 10 classes (lowest mean log-prob):")
    for lbl, row in class_stats.head(10).iterrows():
        print(f"    {lbl:30s}  n={row['n']:3.0f}  mean_lp={row['mean_lp']:+.4f}  top1={row['top1']:.0%}  mean_rank={row['mean_rank']:.0f}  margin={row['mean_margin']:+.4f}")

    print(f"\n  Easiest 10 classes:")
    for lbl, row in class_stats.tail(10).iloc[::-1].iterrows():
        print(f"    {lbl:30s}  n={row['n']:3.0f}  mean_lp={row['mean_lp']:+.4f}  top1={row['top1']:.0%}  mean_rank={row['mean_rank']:.0f}  margin={row['mean_margin']:+.4f}")

    # Narrowest decision boundaries (correct predictions, smallest margin)
    correct_imgs = set(gt[gt["candidate_rank"] == 1]["image_idx"])
    narrow = margins.loc[margins.index.isin(correct_imgs)].sort_values().head(10)
    if not narrow.empty:
        print(f"\n  10 narrowest boundaries (correct predictions, smallest margin):")
        for img_idx, margin in narrow.items():
            img_scores = gt_scores[gt_scores["image_idx"] == img_idx].sort_values("candidate_rank")
            gt_lbl = img_scores.iloc[0]["ground_truth_label"]
            top2_lbl = img_scores.iloc[1]["candidate_label"]
            top1_lp = img_scores.iloc[0]["log_prob_norm"]
            print(f"    idx={img_idx:>6d}  gt={gt_lbl:20s}  top1_lp={top1_lp:+.4f}  top2={top2_lbl:20s}  margin={margin:+.4f}")

    # Worst outlier images
    worst = gt.nsmallest(10, "log_prob_norm")
    print(f"\n  10 worst outlier images:")
    for _, r in worst.iterrows():
        img_scores = gt_scores[gt_scores["image_idx"] == r["image_idx"]].sort_values("candidate_rank")
        top1_lbl = img_scores.iloc[0]["candidate_label"]
        top2_lbl = img_scores.iloc[1]["candidate_label"]
        margin = img_scores.iloc[0]["log_prob_norm"] - img_scores.iloc[1]["log_prob_norm"]
        print(
            f"    idx={r['image_idx']:>6d}  "
            f"gt={r['ground_truth_label']:20s}  "
            f"pred={top1_lbl:20s}  "
            f"lp={r['log_prob_norm']:+.4f}  "
            f"rank={r['candidate_rank']}  "
            f"top2={top2_lbl}  margin={margin:+.4f}"
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
    parser.add_argument("--resume", type=Path, default=None, help="Resume from run directory")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    cache_dir = Path(__file__).parent / ".cache"
    labels = load_imagenet_labels(cache_dir)
    print(f"Loaded {len(labels)} ImageNet labels")

    # Stream from HuggingFace (filtered to categories)
    print(f"Streaming ImageNet validation for {len(CATEGORIES)} categories ...")
    samples = stream_imagenet_val(
        labels=labels,
        categories=CATEGORIES,
        max_images=args.max_images,
        max_per_class=args.max_per_class,
    )
    print(f"Loaded {len(samples)} images")

    # Resume support
    done_indices: set[int] = set()
    existing_images: Optional[pd.DataFrame] = None
    existing_scores: Optional[pd.DataFrame] = None
    if args.resume and args.resume.is_dir():
        existing_images, existing_scores = load_run(args.resume)
        done_indices = set(existing_images["image_idx"])
        print(f"Resuming: {len(done_indices)} images already done")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.resume if args.resume else args.out_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    meta = {
        "model": args.model,
        "device": args.device,
        "prompt": args.prompt,
        "thinking": args.thinking,
        "max_thinking_tokens": args.max_thinking_tokens,
        "max_images": args.max_images,
        "max_per_class": args.max_per_class,
        "categories": CATEGORIES,
        "timestamp": timestamp,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Reproducibility
    torch.manual_seed(42)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(42)

    # Load model
    print(f"Loading {args.model} on {args.device} ...")
    scorer = create_scorer(
        args.model, args.device, args.thinking, args.max_thinking_tokens,
    )

    # Run evaluation
    image_rows: list[dict] = []
    score_rows: list[dict] = []
    n_correct = 0
    n_forced_top1 = 0
    n_done = 0
    sum_lp = 0.0

    pbar = tqdm(enumerate(samples), total=len(samples), desc="Scoring")
    for i, sample in pbar:
        if i in done_indices:
            continue

        gt_idx = sample["idx"]
        gt_label = labels[gt_idx]

        try:
            generated_text, thinking_text, scored = scorer.score_image(
                sample["image"], args.prompt, CATEGORIES,
            )
            scorer.cleanup()

            top1_label = scored[0][0]
            gt_rank = next(
                rank + 1 for rank, (lbl, *_) in enumerate(scored) if lbl == gt_label
            )
            gt_norm_lp = next(n for lbl, _, n, _ in scored if lbl == gt_label)

            image_rows.append({
                "image_idx": i,
                "ground_truth_idx": gt_idx,
                "ground_truth_label": gt_label,
                "generated_text": generated_text,
                "thinking_text": thinking_text,
            })

            for rank, (lbl, total_lp, norm_lp, n_tokens) in enumerate(scored, 1):
                score_rows.append({
                    "image_idx": i,
                    "candidate_label": lbl,
                    "candidate_rank": rank,
                    "log_prob": total_lp,
                    "log_prob_norm": norm_lp,
                    "n_tokens": n_tokens,
                })

            n_done += 1
            n_forced_top1 += gt_rank == 1
            n_correct += gt_label.lower() in generated_text.lower()
            sum_lp += gt_norm_lp

            pbar.set_postfix_str(
                f"gen={n_correct/n_done:.0%} "
                f"top1={n_forced_top1/n_done:.0%} "
                f"lp={sum_lp/n_done:+.2f} "
                f"last={gt_label}→{top1_label}"
            )

        except Exception as e:
            print(f"\n  ERROR image {i} (class {gt_label}): {e}")

    # Save results
    images_df = pd.DataFrame(image_rows)
    scores_df = pd.DataFrame(score_rows)
    if existing_images is not None:
        images_df = pd.concat([existing_images, images_df], ignore_index=True)
        scores_df = pd.concat([existing_scores, scores_df], ignore_index=True)
    images_df.to_parquet(run_dir / "images.parquet", index=False)
    scores_df.to_parquet(run_dir / "scores.parquet", index=False)
    print(f"\nResults saved to {run_dir}/")
    print(f"  images.parquet: {len(images_df)} rows")
    print(f"  scores.parquet: {len(scores_df)} rows ({len(images_df)} images × {len(CATEGORIES)} candidates)")
    print_summary(run_dir)


if __name__ == "__main__":
    main()
