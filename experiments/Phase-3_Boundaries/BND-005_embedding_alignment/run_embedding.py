#!/usr/bin/env python3
"""
BND-005: Embed coc_reasoning text via Ollama qwen3-embedding.

Generates architecture-aligned embeddings by encoding each scene's
chain-of-thought reasoning through a Qwen3-based embedding model.

Usage:
    python run_embedding.py
    python run_embedding.py --host http://my-gpu-server:11434
    python run_embedding.py --batch-size 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.lib.schema import load_scenes

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "pipeline"
OUTPUT_DIR = PROJECT_ROOT / "data" / "BND-005"

MODEL = "qwen3-embedding"
DEFAULT_BATCH_SIZE = 8


# =============================================================================
# EMBEDDING
# =============================================================================

def embed_texts(
    texts: list[str],
    clip_ids: list[str],
    host: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Embed texts using Ollama qwen3-embedding.

    Args:
        texts: List of coc_reasoning strings to embed
        clip_ids: Corresponding clip_id for each text
        host: Ollama server URL
        batch_size: Number of texts per API call

    Returns:
        (embeddings array, clip_ids list) -- aligned by index
    """
    from ollama import Client

    client = Client(host=host)

    all_embeddings = []
    valid_clip_ids = []
    n_failures = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = clip_ids[i : i + batch_size]

        try:
            response = client.embed(model=MODEL, input=batch_texts)
            embeddings = response["embeddings"]

            for emb, cid in zip(embeddings, batch_ids):
                all_embeddings.append(emb)
                valid_clip_ids.append(cid)

        except Exception as e:
            n_failures += len(batch_texts)
            print(f"\nBatch {i // batch_size} failed: {e}")

    if n_failures > 0:
        print(f"Total failures: {n_failures}/{len(texts)}")

    # Stack into array and L2-normalize
    emb_array = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    emb_array = emb_array / norms

    return emb_array, valid_clip_ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BND-005: Embed coc_reasoning via qwen3-embedding",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embedding API calls (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BND-005: EMBED COC_REASONING (qwen3-embedding)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Host: {args.host}")
    print(f"Batch size: {args.batch_size}")

    # Load scenes
    scenes_file = DATA_DIR / "scenes.parquet"
    if not scenes_file.exists():
        print(f"\nError: {scenes_file} not found. Run pipeline first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"\nTotal scenes: {len(df)}")

    # Filter: non-empty coc_reasoning AND has_ade
    mask = (
        (df["has_ade"] == True)
        & df["coc_reasoning"].notna()
        & (df["coc_reasoning"].str.len() > 0)
    )
    df_valid = df[mask].reset_index(drop=True)
    print(f"Scenes with ADE + coc_reasoning: {len(df_valid)}")

    if len(df_valid) == 0:
        print("\nError: No valid scenes found.")
        return 1

    texts = df_valid["coc_reasoning"].tolist()
    clip_ids = df_valid["clip_id"].tolist()

    # Embed
    print(f"\nEmbedding {len(texts)} scenes...")
    embeddings, valid_ids = embed_texts(
        texts, clip_ids, host=args.host, batch_size=args.batch_size
    )

    print(f"\nEmbedded: {len(valid_ids)} scenes")
    print(f"Embedding dim: {embeddings.shape[1]}")

    # Verify L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}]")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "qwen3_embeddings.npz"

    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        clip_ids=np.array(valid_ids, dtype=str),
    )

    print(f"\nSaved: {output_file}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  clip_ids: {len(valid_ids)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
