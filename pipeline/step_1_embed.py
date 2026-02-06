#!/usr/bin/env python3
"""
Step 1: Embed Scenes

Generates OpenCLIP bigG embeddings for all scenes (image + text vocabulary).
- Image embeddings: stored indexed by emb_index in scenes.parquet
- Text embeddings: vocabulary of all classification key values

Usage:
    python pipeline/step_1_embed.py
    python pipeline/step_1_embed.py --device cuda --batch-size 8
    python pipeline/step_1_embed.py --skip-text  # Skip text vocabulary generation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import load_scenes, save_scenes, CLASSIFICATION_KEYS
from lib.composites import create_composite, ensure_composites
from lib.io import (
    load_config, load_embeddings, append_embeddings, save_embeddings,
    load_text_vocabulary, get_repo_root
)


# =============================================================================
# TEXT VOCABULARY DEFINITIONS
# =============================================================================

# Human-readable descriptions for each classification value
# Used to generate text embeddings that capture semantic meaning
# Covers all categorical keys from CLS-001 classifications
TEXT_VOCABULARY: dict[str, dict[str, str]] = {
    # --- Scene Context ---
    "weather": {
        "clear": "clear sunny weather with blue sky",
        "cloudy": "cloudy overcast weather conditions",
        "rainy": "rainy weather with wet roads",
        "foggy": "foggy weather with reduced visibility",
        "snowy": "snowy winter weather conditions",
    },
    "time_of_day": {
        "day": "daytime with bright natural lighting",
        "dawn_dusk": "dawn or dusk with low sun angle",
        "night": "nighttime with artificial lighting",
    },
    "road_type": {
        "highway": "highway road with multiple lanes and high speed traffic",
        "urban_street": "urban city street environment with buildings",
        "residential": "quiet residential neighborhood road with houses",
        "intersection": "road intersection with traffic controls and turn lanes",
        "parking_lot": "parking lot with parked vehicles and pedestrians",
        "construction_zone": "construction zone with workers and equipment",
        "rural": "rural countryside road with open fields",
    },
    "traffic_situation": {
        "simple": "simple traffic with few vehicles and clear road",
        "moderate": "moderate traffic requiring some attention",
        "complex": "complex traffic with many vehicles and hazards",
        "critical": "critical traffic situation requiring maximum vigilance",
    },

    # --- Spatial Reasoning ---
    "depth_complexity": {
        "flat": "flat scene with single depth plane and clear view",
        "layered": "layered scene with objects at multiple depth zones",
        "complex": "complex scene with many overlapping objects at different distances",
    },
    "occlusion_level": {
        "none": "no occlusion with all objects fully visible",
        "minimal": "minimal occlusion with few partially hidden objects",
        "moderate": "moderate occlusion with some objects partially hidden",
        "severe": "severe occlusion with many critical objects hidden",
    },

    # --- Perceptual Challenges ---
    "visual_degradation": {
        "none": "no visual degradation with clear image quality",
        "glare": "sun glare causing bright reflections and overexposure",
        "low_light": "low light conditions with dark underexposed areas",
        "motion_blur": "motion blur affecting moving objects",
        "rain_artifacts": "rain droplets and wet surface reflections",
        "fog_haze": "fog or haze reducing visibility",
        "sensor_artifact": "sensor artifacts like lens flare or dirt",
    },

    # --- Safety Critical ---
    "required_action": {
        "none": "no action required, maintain current speed and course",
        "slow": "slow down and prepare to stop",
        "stop": "stop the vehicle completely",
        "evade": "evasive steering maneuver required",
    },
    "safety_criticality": {
        "tier1_catastrophic": "catastrophic risk with vulnerable road users in path",
        "tier2_severe": "severe risk requiring signal compliance or lane management",
        "tier3_moderate": "moderate risk requiring speed and distance judgment",
        "tier4_minor": "minor risk with minimal consequences",
    },

    # --- Attribute Binding ---
    "lane_marking_type": {
        "solid_white": "solid white lane markings",
        "dashed_white": "dashed white lane markings allowing lane changes",
        "solid_yellow": "solid yellow lane markings",
        "dashed_yellow": "dashed yellow lane markings",
        "double_yellow": "double yellow center line no passing zone",
        "none": "no visible lane markings",
        "unknown": "unknown or unclear lane markings",
    },

    # --- Boolean Presence Keys (as binary text) ---
    "pedestrians_present": {
        "true": "pedestrians are present and visible in the scene",
        "false": "no pedestrians visible in the scene",
    },
    "cyclists_present": {
        "true": "cyclists are present and visible in the scene",
        "false": "no cyclists visible in the scene",
    },
    "construction_activity": {
        "true": "active construction with workers and equipment",
        "false": "no construction activity in the scene",
    },
    "traffic_signals_visible": {
        "true": "traffic signals and signs are visible",
        "false": "no traffic signals visible in the scene",
    },
    "similar_object_confusion": {
        "true": "multiple similar objects that could be confused",
        "false": "no similar object confusion risk",
    },
}


def get_device(requested: str | None = None) -> str:
    """Get best available device."""
    if requested:
        return requested

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class OpenCLIPBigGProvider:
    """OpenCLIP ViT-bigG/14 embedding provider (2.5B params)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._embedding_dim = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return

        import open_clip

        print(f"Loading ViT-bigG-14 to {self.device}...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14",
            pretrained="laion2b_s39b_b160k",
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
        self._model.eval()

        # Get embedding dim
        if hasattr(self._model.visual, 'output_dim'):
            self._embedding_dim = self._model.visual.output_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                self._embedding_dim = self._model.encode_image(dummy).shape[-1]

        print(f"Model loaded. Embedding dim: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._embedding_dim

    @torch.no_grad()
    def embed_images(
        self,
        images: list[Image.Image | Path | str],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed images in batches.

        Args:
            images: List of PIL Images or paths to images
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            L2-normalized embeddings, shape (N, dim)
        """
        self._load_model()
        embeddings = []

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding images")

        for i in iterator:
            batch_items = images[i:i + batch_size]

            # Load and preprocess images
            batch_images = []
            for item in batch_items:
                if isinstance(item, (str, Path)):
                    img = Image.open(item).convert("RGB")
                else:
                    img = item.convert("RGB")
                batch_images.append(self._preprocess(img))

            # Stack and encode
            batch_tensor = torch.stack(batch_images).to(self.device)
            emb = self._model.encode_image(batch_tensor)

            # L2 normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)

    @torch.no_grad()
    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed texts in batches using CLIP text encoder.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            L2-normalized embeddings, shape (N, dim)
        """
        self._load_model()
        embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding texts")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize and encode
            tokens = self._tokenizer(batch_texts).to(self.device)
            emb = self._model.encode_text(tokens)

            # L2 normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)


def generate_text_vocabulary_embeddings(
    provider: OpenCLIPBigGProvider,
    vocabulary: dict[str, dict[str, str]] = TEXT_VOCABULARY,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """
    Generate embeddings for the text vocabulary.

    Args:
        provider: OpenCLIP provider
        vocabulary: {key: {value: description}} mapping

    Returns:
        Tuple of (embeddings array, vocabulary index mapping)
        - embeddings: shape (V, dim) where V is total vocabulary size
        - vocab_map: {key: {value: index}} for looking up embeddings
    """
    texts = []
    vocab_map = {}
    current_idx = 0

    # Embed ALL keys in vocabulary, not just CLASSIFICATION_KEYS
    for key in sorted(vocabulary.keys()):
        vocab_map[key] = {}
        for value, description in vocabulary[key].items():
            # Prefix with context for better CLIP alignment
            text = f"A driving scene with {description}"
            texts.append(text)
            vocab_map[key][value] = current_idx
            current_idx += 1

    print(f"Generating embeddings for {len(texts)} vocabulary entries across {len(vocab_map)} keys...")
    embeddings = provider.embed_texts(texts, show_progress=True)

    return embeddings, vocab_map


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate embeddings for scenes (image + text vocabulary)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda, mps, cpu). Auto-detected if not specified."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for embedding (default: from config)"
    )
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Skip text vocabulary embedding generation"
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Only generate text vocabulary embeddings (skip images)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)

    # Resolve parameters
    device = get_device(args.device)
    batch_size = args.batch_size or config.embedding.batch_size
    t0_us = config.dataset.t0_us

    # Resolve paths
    repo_root = get_repo_root()
    scenes_file = repo_root / config.paths.scenes_file
    embeddings_file = repo_root / config.paths.embeddings_file
    image_cache = repo_root / config.paths.image_cache

    print("=" * 60)
    print("STEP 1: EMBED SCENES")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input: {scenes_file}")

    # Load embedding model
    print("\nLoading embedding model...")
    provider = OpenCLIPBigGProvider(device=device)

    # =========================================================================
    # PART A: Text Vocabulary Embeddings
    # =========================================================================
    if not args.skip_text:
        print("\n" + "-" * 60)
        print("PART A: TEXT VOCABULARY EMBEDDINGS")
        print("-" * 60)

        # Check if text embeddings already exist
        existing_text_emb, existing_vocab = load_text_vocabulary(embeddings_file)

        if len(existing_text_emb) > 0 and not args.text_only:
            print(f"Text vocabulary already exists ({len(existing_text_emb)} entries)")
            print("  Use --text-only to regenerate")
        else:
            # Generate text vocabulary embeddings
            text_embeddings, vocab_map = generate_text_vocabulary_embeddings(provider)
            print(f"Text vocabulary shape: {text_embeddings.shape}")
            print(f"Keys: {list(vocab_map.keys())}")

            # Save text embeddings (preserves existing image embeddings)
            save_embeddings(
                embeddings_file,
                text_embeddings=text_embeddings,
                text_vocab_map=vocab_map,
            )
            print(f"Saved text vocabulary to: {embeddings_file}")

    if args.text_only:
        print("\n--text-only specified, skipping image embeddings.")
        return 0

    # =========================================================================
    # PART B: Image Embeddings
    # =========================================================================
    print("\n" + "-" * 60)
    print("PART B: IMAGE EMBEDDINGS")
    print("-" * 60)

    # Load scenes
    if not scenes_file.exists():
        print("\nError: scenes.parquet not found. Run step_0_sample.py first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"Total scenes: {len(df)}")

    # Filter scenes without embeddings
    to_embed = df[df["has_embedding"] != True].copy()
    print(f"To embed: {len(to_embed)}")

    if len(to_embed) == 0:
        print("\nAll scenes already embedded. Nothing to do.")
        return 0

    # Ensure composite images exist
    print("\nEnsuring composite images exist...")
    clip_ids = to_embed["clip_id"].tolist()
    composite_paths = ensure_composites(
        clip_ids,
        cache_dir=image_cache,
        t0_us=t0_us,
        num_workers=4,
        show_progress=True,
    )

    # Check for missing composites
    missing = [cid for cid in clip_ids if cid not in composite_paths]
    if missing:
        print(f"Warning: Could not generate composites for {len(missing)} scenes")
        # Remove missing from to_embed
        to_embed = to_embed[~to_embed["clip_id"].isin(missing)]
        clip_ids = to_embed["clip_id"].tolist()

    if len(to_embed) == 0:
        print("\nNo scenes to embed after filtering failures.")
        return 1

    # Get composite paths in order
    image_paths = [composite_paths[cid] for cid in clip_ids]

    # Generate embeddings
    print(f"\nEmbedding {len(image_paths)} scenes...")
    new_embeddings = provider.embed_images(
        image_paths,
        batch_size=batch_size,
        show_progress=True,
    )

    print(f"Embedding shape: {new_embeddings.shape}")

    # Append to embeddings file
    start_index = append_embeddings(embeddings_file, new_embeddings)
    print(f"Appended embeddings starting at index {start_index}")

    # Update scenes DataFrame
    for i, clip_id in enumerate(clip_ids):
        idx = df[df["clip_id"] == clip_id].index[0]
        df.loc[idx, "emb_index"] = start_index + i
        df.loc[idx, "has_embedding"] = True

    # Save updated scenes
    save_scenes(df, scenes_file)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Image embeddings: {len(to_embed)} new scenes embedded")
    print(f"Total with embeddings: {df['has_embedding'].sum()}/{len(df)}")

    # Load and report text vocabulary
    text_emb, vocab_map = load_text_vocabulary(embeddings_file)
    if len(text_emb) > 0:
        print(f"Text vocabulary: {len(text_emb)} entries across {len(vocab_map)} keys")

    print(f"\nEmbeddings file: {embeddings_file}")
    print(f"Scenes file: {scenes_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
