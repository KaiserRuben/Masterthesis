#!/usr/bin/env python3
"""
Step 1: Embed Scenes

Generates OpenCLIP bigG embeddings for all scenes.
Embeddings stored in embeddings.npz, indexed by emb_index in scenes.parquet.

Usage:
    python pipeline/step_1_embed.py
    python pipeline/step_1_embed.py --device cuda --batch-size 8
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

from lib.schema import load_scenes, save_scenes
from lib.composites import create_composite, ensure_composites
from lib.io import load_config, load_embeddings, append_embeddings, get_repo_root


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


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate embeddings for scenes",
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
        "--config", type=str, default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)

    # Resolve parameters
    device = get_device(args.device)
    batch_size = args.batch_size or config["embedding"]["batch_size"]
    t0_us = config["dataset"]["t0_us"]

    # Resolve paths
    repo_root = get_repo_root()
    scenes_file = repo_root / config["paths"]["scenes_file"]
    embeddings_file = repo_root / config["paths"]["embeddings_file"]
    image_cache = repo_root / config["paths"]["image_cache"]

    print("=" * 60)
    print("STEP 1: EMBED SCENES")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input: {scenes_file}")

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

    # Load embedding model
    print("\nLoading embedding model...")
    provider = OpenCLIPBigGProvider(device=device)

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

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Embedded: {len(to_embed)} scenes")
    print(f"Total with embeddings: {df['has_embedding'].sum()}/{len(df)}")
    print(f"\nEmbeddings: {embeddings_file}")
    print(f"Scenes: {scenes_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
