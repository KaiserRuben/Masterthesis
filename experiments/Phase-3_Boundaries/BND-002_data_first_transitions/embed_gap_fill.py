#!/usr/bin/env python3
"""
Embed the 47 gap-fill scenes using OpenCLIP bigG.

These are scenes that have ADE data (from Alpamayo) but were never embedded
because they were inferenced after EMB-001 ran.

After embedding:
1. Merge with existing embeddings
2. Re-run BND-002b analysis with expanded ADE coverage
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add EMB-001 to path for providers and data loader
EMB001_DIR = Path(__file__).parents[2] / "Phase-2_Embeddings" / "EMB-001_latent_navigation"
sys.path.insert(0, str(EMB001_DIR))

from providers.openclip_bigg import OpenCLIPBigGProvider
from data_loader import load_clip_composite, CACHE_DIR

DATA_DIR = Path(__file__).parents[3] / "data"
BND002_DIR = DATA_DIR / "BND-002"
EMB001_OUTPUT_DIR = DATA_DIR / "EMB-001" / "v2" / "openclip_bigg_all_20260129_015339"


def main():
    # Load gap-fill scene IDs
    with open(BND002_DIR / "gap_fill_scene_ids.json") as f:
        gap_fill = json.load(f)

    scene_ids = gap_fill["scene_ids"]
    print(f"Embedding {len(scene_ids)} gap-fill scenes...")

    # Load composite images for each scene
    # These will be cached in the image_cache directory
    cache_dir = DATA_DIR / "EMB-001" / "image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading composite images...")
    images = []
    valid_ids = []
    failed = []

    for scene_id in scene_ids:
        try:
            img = load_clip_composite(scene_id, cache_dir=cache_dir)
            images.append(img)
            valid_ids.append(scene_id)
        except Exception as e:
            print(f"  Failed to load {scene_id}: {e}")
            failed.append(scene_id)

    print(f"Loaded {len(images)} images, {len(failed)} failed")

    if not images:
        print("ERROR: No images loaded. Exiting.")
        return

    # Save images temporarily for embedding
    temp_dir = BND002_DIR / "temp_gap_fill_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for scene_id, img in zip(valid_ids, images):
        path = temp_dir / f"{scene_id}.jpg"
        img.save(path, "JPEG", quality=85)
        image_paths.append(str(path))

    print(f"Saved {len(image_paths)} images to {temp_dir}")

    # Initialize provider and embed
    print("\nInitializing OpenCLIP bigG...")
    provider = OpenCLIPBigGProvider(device="mps")

    print("Embedding images...")
    embeddings = provider.embed_images(image_paths, batch_size=4)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save gap-fill embeddings
    gap_fill_output = BND002_DIR / "gap_fill_embeddings.npz"
    np.savez(
        gap_fill_output,
        embeddings=embeddings,
        scene_ids=np.array(valid_ids),
        model_name="openclip_bigg",
        embedding_dim=embeddings.shape[1],
    )
    print(f"Saved gap-fill embeddings to {gap_fill_output}")

    # Merge with existing embeddings
    print("\nMerging with existing EMB-001 embeddings...")
    existing = np.load(EMB001_OUTPUT_DIR / "embeddings.npz", allow_pickle=True)

    existing_embeddings = existing["embeddings"]
    existing_ids = list(existing["scene_ids"])

    # Concatenate
    merged_embeddings = np.vstack([existing_embeddings, embeddings])
    merged_ids = existing_ids + valid_ids

    print(f"Merged: {existing_embeddings.shape[0]} + {embeddings.shape[0]} = {merged_embeddings.shape[0]} scenes")

    # Save merged embeddings
    merged_output = BND002_DIR / "embeddings_merged_2647.npz"
    np.savez(
        merged_output,
        embeddings=merged_embeddings,
        scene_ids=np.array(merged_ids),
        model_name="openclip_bigg",
        embedding_dim=merged_embeddings.shape[1],
    )
    print(f"Saved merged embeddings to {merged_output}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gap-fill scenes requested: {len(scene_ids)}")
    print(f"Successfully embedded: {len(valid_ids)}")
    print(f"Failed: {len(failed)}")
    print(f"")
    print(f"Original EMB-001 scenes: {existing_embeddings.shape[0]}")
    print(f"Merged total: {merged_embeddings.shape[0]}")
    print(f"")
    print(f"Output files:")
    print(f"  Gap-fill only: {gap_fill_output}")
    print(f"  Merged: {merged_output}")

    # Clean up temp images
    print("\nCleaning up temp images...")
    for p in temp_dir.glob("*.jpg"):
        p.unlink()
    temp_dir.rmdir()
    print("Done!")


if __name__ == "__main__":
    main()
