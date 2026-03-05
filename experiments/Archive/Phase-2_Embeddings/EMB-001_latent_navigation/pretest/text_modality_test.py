"""
Pre-test: Text Modality Impact on Image-Text Alignment

Research question: Which text format produces best-aligned embeddings?

Templates tested:
- bare: "rainy"
- key_value: "weather: rainy"
- simple: "rainy weather"
- sentence: "A driving scene with rainy weather"
- photo_prompt: "A photo of a driving scene in rainy weather"

Metrics:
- accuracy: % of images where correct anchor has highest similarity
- mean_rank: average rank of correct anchor (1 = best)
- margin: mean(sim_correct - max_sim_incorrect)
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# Add tools to path
sys.path.insert(0, str(Path(__file__).parents[3]))
from tools.scene.enums import Weather, RoadType, TimeOfDay, TrafficSituation, OcclusionLevel

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path(__file__).parents[3] / "data" / "CLS-001"
OUTPUT_DIR = Path(__file__).parent / "results"

# Keys to test (categorical with clean enum values)
TEST_KEYS = {
    "weather": [e.value for e in Weather],
    "road_type": [e.value for e in RoadType],
    "time_of_day": [e.value for e in TimeOfDay],
    "traffic_situation": [e.value for e in TrafficSituation],
    "occlusion_level": [e.value for e in OcclusionLevel],
}

# Text templates - {key} and {value} are placeholders
TEXT_TEMPLATES = {
    "bare": "{value}",
    "key_value": "{key}: {value}",
    "simple": "{value_clean} {key_clean}",
    "sentence": "A driving scene with {value_clean} {key_clean}",
    "photo_prompt": "A photo of a driving scene with {value_clean} {key_clean}",
}

# Human-readable transformations
VALUE_TRANSFORMS = {
    # weather
    "clear": "clear",
    "cloudy": "cloudy",
    "rainy": "rainy",
    "foggy": "foggy",
    "snowy": "snowy",
    # road_type
    "highway": "highway",
    "urban_street": "urban street",
    "residential": "residential",
    "intersection": "intersection",
    "parking_lot": "parking lot",
    "construction_zone": "construction zone",
    "rural": "rural road",
    # time_of_day
    "day": "daytime",
    "dawn_dusk": "dawn or dusk",
    "night": "nighttime",
    # traffic_situation
    "simple": "simple traffic",
    "moderate": "moderate traffic",
    "complex": "complex traffic",
    "critical": "critical traffic",
    # occlusion_level
    "none": "no occlusion",
    "minimal": "minimal occlusion",
    "moderate": "moderate occlusion",
    "severe": "severe occlusion",
}

KEY_TRANSFORMS = {
    "weather": "weather",
    "road_type": "road type",
    "time_of_day": "lighting",
    "traffic_situation": "conditions",
    "occlusion_level": "visibility",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_classifications() -> dict[str, dict]:
    """Load scene classifications, extracting key values."""
    with open(DATA_DIR / "scene_classifications.json") as f:
        data = json.load(f)

    # Extract classifications list
    classifications_list = data.get("classifications", [])

    result = {}
    for item in classifications_list:
        clip_id = item["clip_id"]
        cls = item["classification"]

        # Extract simple values from complex responses
        extracted = {}
        for key in TEST_KEYS:
            if key in cls:
                val = cls[key]
                # Handle nested response objects
                if isinstance(val, dict):
                    # Look for the key name in the response (e.g., "weather" in WeatherResponse)
                    if key in val:
                        extracted[key] = val[key]
                    elif "category" in val:  # traffic_situation uses "category"
                        extracted[key] = val["category"]
                    else:
                        # Try to find the main value
                        for k, v in val.items():
                            if k not in ["reasoning", "points", "total"]:
                                extracted[key] = v
                                break
                else:
                    extracted[key] = val

        result[clip_id] = extracted

    return result


def get_image_paths(classifications: dict[str, dict]) -> list[Path]:
    """Get paths to all scene images."""
    image_dir = DATA_DIR / "images"
    paths = []
    for clip_id in classifications:
        img_path = image_dir / f"{clip_id}.jpg"
        if img_path.exists():
            paths.append(img_path)
    return paths


# =============================================================================
# MODEL
# =============================================================================

@dataclass
class EVA02EProvider:
    """EVA02-E-14-plus embedding provider."""

    model: Any = None
    preprocess: Any = None
    tokenizer: Any = None
    device: str = "mps"

    def __post_init__(self):
        print(f"Loading EVA02-E-14-plus (4.4B params) to {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "EVA02-E-14-plus",
            pretrained="laion2b_s9b_b144k",
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer("EVA02-E-14-plus")
        self.model.eval()

        # Cache embedding dim (inferred from dummy forward pass)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._embedding_dim = self.model.encode_image(dummy).shape[-1]
        print(f"Model loaded. Embedding dim: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.no_grad()
    def embed_images(self, image_paths: list[Path], batch_size: int = 4) -> np.ndarray:
        """Embed images in batches."""
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = torch.stack([
                self.preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(self.device)

            emb = self.model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text strings."""
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
        return emb.cpu().numpy()


# =============================================================================
# TEXT ANCHOR GENERATION
# =============================================================================

def generate_text_anchor(key: str, value: str, template_name: str) -> str:
    """Generate text anchor using specified template."""
    template = TEXT_TEMPLATES[template_name]

    value_clean = VALUE_TRANSFORMS.get(value, value.replace("_", " "))
    key_clean = KEY_TRANSFORMS.get(key, key.replace("_", " "))

    return template.format(
        key=key,
        value=value,
        key_clean=key_clean,
        value_clean=value_clean,
    )


def generate_all_anchors(template_name: str) -> dict[str, dict[str, str]]:
    """Generate all text anchors for a template."""
    anchors = {}
    for key, values in TEST_KEYS.items():
        anchors[key] = {v: generate_text_anchor(key, v, template_name) for v in values}
    return anchors


# =============================================================================
# METRICS
# =============================================================================

def compute_alignment_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: dict[str, np.ndarray],  # {value: embedding}
    ground_truth: list[str],  # correct value per image
    values: list[str],  # all possible values
) -> dict:
    """Compute alignment metrics for one key."""

    # Stack text embeddings in order
    text_matrix = np.stack([text_embeddings[v] for v in values])  # (V, D)

    # Compute similarities: (N, V)
    similarities = image_embeddings @ text_matrix.T

    # Get predictions and ground truth indices
    pred_indices = similarities.argmax(axis=1)
    gt_indices = np.array([values.index(gt) for gt in ground_truth])

    # Accuracy
    accuracy = (pred_indices == gt_indices).mean()

    # Mean rank of correct answer (1-indexed)
    ranks = []
    for i, gt_idx in enumerate(gt_indices):
        sims = similarities[i]
        rank = (sims > sims[gt_idx]).sum() + 1  # 1 = best
        ranks.append(rank)
    mean_rank = np.mean(ranks)

    # Margin: sim_correct - max_sim_incorrect
    margins = []
    for i, gt_idx in enumerate(gt_indices):
        sims = similarities[i]
        correct_sim = sims[gt_idx]
        incorrect_sims = np.concatenate([sims[:gt_idx], sims[gt_idx+1:]])
        if len(incorrect_sims) > 0:
            margin = correct_sim - incorrect_sims.max()
            margins.append(margin)
    mean_margin = np.mean(margins) if margins else 0.0

    return {
        "accuracy": float(accuracy),
        "mean_rank": float(mean_rank),
        "mean_margin": float(mean_margin),
        "n_samples": len(ground_truth),
    }


# =============================================================================
# MAIN
# =============================================================================

def run_pretest():
    """Run the text modality pre-test."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading classifications...")
    classifications = load_classifications()
    print(f"Loaded {len(classifications)} scenes")

    # Get image paths
    image_paths = get_image_paths(classifications)
    print(f"Found {len(image_paths)} images")

    # Filter to scenes with all required keys
    valid_scenes = []
    for path in image_paths:
        clip_id = path.stem
        cls = classifications.get(clip_id, {})
        if all(key in cls for key in TEST_KEYS):
            valid_scenes.append((path, clip_id, cls))

    print(f"Valid scenes with all keys: {len(valid_scenes)}")

    if not valid_scenes:
        print("ERROR: No valid scenes found!")
        # Debug: show what keys are present
        for clip_id, cls in list(classifications.items())[:3]:
            print(f"  {clip_id}: {list(cls.keys())}")
        return

    # Load model
    provider = EVA02EProvider()

    # Embed images
    paths = [s[0] for s in valid_scenes]
    image_embeddings = provider.embed_images(paths)
    print(f"Image embeddings shape: {image_embeddings.shape}")

    # Test each template
    results = {}

    for template_name in TEXT_TEMPLATES:
        print(f"\n{'='*60}")
        print(f"Testing template: {template_name}")
        print(f"{'='*60}")

        # Generate anchors
        anchors = generate_all_anchors(template_name)

        # Show sample anchors
        print("\nSample anchors:")
        for key in list(TEST_KEYS.keys())[:2]:
            for value in list(TEST_KEYS[key])[:2]:
                print(f"  {key}/{value}: \"{anchors[key][value]}\"")

        # Embed all text anchors
        text_embeddings = {}
        for key, value_texts in anchors.items():
            texts = list(value_texts.values())
            embs = provider.embed_texts(texts)
            text_embeddings[key] = {v: embs[i] for i, v in enumerate(value_texts.keys())}

        # Compute metrics per key
        template_results = {}
        for key, values in TEST_KEYS.items():
            ground_truth = [s[2][key] for s in valid_scenes]

            # Filter to samples where ground truth is in known values
            valid_indices = [i for i, gt in enumerate(ground_truth) if gt in values]
            if not valid_indices:
                print(f"  {key}: No valid samples (ground truth values not in enum)")
                continue

            filtered_embeddings = image_embeddings[valid_indices]
            filtered_gt = [ground_truth[i] for i in valid_indices]

            metrics = compute_alignment_metrics(
                filtered_embeddings,
                text_embeddings[key],
                filtered_gt,
                values,
            )
            template_results[key] = metrics

            print(f"  {key}: acc={metrics['accuracy']:.3f}, "
                  f"rank={metrics['mean_rank']:.2f}, "
                  f"margin={metrics['mean_margin']:.3f} "
                  f"(n={metrics['n_samples']})")

        results[template_name] = template_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Mean accuracy across keys per template")
    print(f"{'='*60}")

    summary = {}
    for template_name, template_results in results.items():
        if template_results:
            mean_acc = np.mean([m["accuracy"] for m in template_results.values()])
            mean_rank = np.mean([m["mean_rank"] for m in template_results.values()])
            mean_margin = np.mean([m["mean_margin"] for m in template_results.values()])
            summary[template_name] = {
                "mean_accuracy": float(mean_acc),
                "mean_rank": float(mean_rank),
                "mean_margin": float(mean_margin),
            }
            print(f"  {template_name:15s}: acc={mean_acc:.3f}, rank={mean_rank:.2f}, margin={mean_margin:.3f}")

    # Save results
    output = {
        "model": "EVA02-E-14-plus",
        "pretrained": "laion2b_s9b_b144k",
        "n_scenes": len(valid_scenes),
        "keys_tested": list(TEST_KEYS.keys()),
        "templates": list(TEXT_TEMPLATES.keys()),
        "results_per_template": results,
        "summary": summary,
    }

    output_path = OUTPUT_DIR / "text_modality_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Best template
    if summary:
        best = max(summary.items(), key=lambda x: x[1]["mean_accuracy"])
        print(f"\nBest template: {best[0]} (acc={best[1]['mean_accuracy']:.3f})")


if __name__ == "__main__":
    run_pretest()
