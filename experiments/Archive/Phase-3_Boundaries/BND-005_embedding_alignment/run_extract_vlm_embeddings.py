#!/usr/bin/env python3
"""
BND-005b: Extract Alpamayo VLM Hidden States as Scene Embeddings.

Runs Alpamayo's Qwen3-VL backbone forward pass (no generation, no expert,
no diffusion) and extracts the last hidden layer as a scene embedding.
This produces a 3584-dim vector per scene that represents how the VLM
*actually perceives* each driving scene.

Data flow (mirrors step_3_infer.py):
    1. load_physical_aiavdataset(clip_id, t0_us)
    2. helper.create_message(image_frames)
    3. processor.apply_chat_template(messages, ...)
    4. model.fuse_traj_tokens(input_ids, traj_data)
    5. model.vlm(input_ids, pixel_values, ..., output_hidden_states=True)
    6. Extract last hidden state -> mean pool -> L2 normalize -> (3584,)

Usage:
    python run_extract_vlm_embeddings.py --device mps
    python run_extract_vlm_embeddings.py --device cuda
    python run_extract_vlm_embeddings.py --device mps --max-scenes 50
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from archive.pipeline.lib.schema import load_scenes
from archive.pipeline.lib.io import load_config

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "pipeline"
OUTPUT_DIR = PROJECT_ROOT / "data" / "BND-005"
DEFAULT_T0_US = 5_000_000


# =============================================================================
# ALPAMAYO SETUP (reused from step_3_infer.py)
# =============================================================================

def patch_physical_ai_av():
    """Monkey-patch physical_ai_av.egomotion to fix read-only array issue with scipy>=1.16."""
    pd.options.mode.copy_on_write = False

    from physical_ai_av import egomotion
    import scipy.spatial.transform as spt

    @classmethod
    def patched_from_egomotion_df(cls, egomotion_df):
        return cls(
            pose=spt.RigidTransform.from_components(
                rotation=spt.Rotation.from_quat(
                    egomotion_df[["qx", "qy", "qz", "qw"]].to_numpy().copy()
                ),
                translation=egomotion_df[["x", "y", "z"]].to_numpy().copy(),
            ),
            velocity=egomotion_df[["vx", "vy", "vz"]].to_numpy().copy(),
            acceleration=egomotion_df[["ax", "ay", "az"]].to_numpy().copy(),
            curvature=egomotion_df[["curvature"]].to_numpy().copy(),
        )

    egomotion.EgomotionState.from_egomotion_df = patched_from_egomotion_df
    print("Applied scipy>=1.16 compatibility patch")


def setup_alpamayo():
    """Ensure alpamayo source is on sys.path."""
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        return
    except ImportError:
        pass

    # Try local tools directory first
    alpamayo_src = PROJECT_ROOT / "tools" / "alpamayo" / "src"
    if alpamayo_src.exists():
        if str(alpamayo_src) not in sys.path:
            sys.path.insert(0, str(alpamayo_src))
        return

    print("Cloning Alpamayo-R1...")
    clone_dir = Path("alpamayo")
    if not clone_dir.exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/NVlabs/Alpamayo.git", str(clone_dir),
        ])

    src = (clone_dir / "src").resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def setup_huggingface():
    """Setup HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated")
    else:
        print("WARNING: HF_TOKEN not set. Model download may fail.")


# =============================================================================
# VLM FORWARD PASS
# =============================================================================

def extract_vlm_embedding(
    model,
    processor,
    clip_id: str,
    device: str,
    t0_us: int = DEFAULT_T0_US,
) -> np.ndarray:
    """
    Run VLM forward pass on a single scene and extract hidden state embedding.

    Returns:
        L2-normalized mean-pooled last hidden state, shape (hidden_dim,)
    """
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    # 1. Load scene data
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)

    # 2. Create message from image frames
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    # 3. Tokenize via processor
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 4. Fuse trajectory tokens
    input_ids = inputs.pop("input_ids")
    traj_data = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    traj_data = helper.to_device(traj_data, device)
    input_ids = helper.to_device(input_ids, device)
    input_ids = model.fuse_traj_tokens(input_ids, traj_data)

    # Move remaining inputs to device
    inputs = helper.to_device(inputs, device)

    # 5. VLM forward pass (no generation)
    autocast_device = "cuda" if device.startswith("cuda") else device
    with torch.autocast(autocast_device, dtype=torch.bfloat16):
        outputs = model.vlm(
            input_ids=input_ids,
            output_hidden_states=True,
            **inputs,
        )

    # 6. Extract last hidden state, mean pool, L2 normalize
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    # Get attention mask for proper pooling (exclude padding)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        pooled = last_hidden.mean(dim=1)  # (1, hidden_dim)

    pooled = pooled.squeeze(0).float()  # (hidden_dim,)
    embedding = pooled / pooled.norm().clamp(min=1e-10)

    return embedding.cpu().numpy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BND-005b: Extract Alpamayo VLM hidden states as scene embeddings",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device to run on (default: mps). Also supports cuda.",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Maximum number of scenes to process",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to pipeline config file",
    )
    args = parser.parse_args()

    device = args.device

    print("=" * 60)
    print("BND-005b: EXTRACT VLM HIDDEN STATES")
    print("=" * 60)

    # Load pipeline config for model_id
    config_path = args.config or (PROJECT_ROOT / "pipeline" / "config.yaml")
    config = load_config(config_path)
    model_id = config.inference.model_id
    t0_us = config.dataset.t0_us

    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"t0_us: {t0_us}")

    # Load scenes
    scenes_file = DATA_DIR / "scenes.parquet"
    if not scenes_file.exists():
        print(f"\nError: {scenes_file} not found. Run pipeline first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"\nTotal scenes: {len(df)}")

    # Filter: scenes with ADE (same as the other embedding scripts)
    mask = df["has_ade"] == True
    df_valid = df[mask].reset_index(drop=True)
    print(f"Scenes with ADE: {len(df_valid)}")

    if len(df_valid) == 0:
        print("\nError: No scenes with ADE found.")
        return 1

    if args.max_scenes and args.max_scenes < len(df_valid):
        df_valid = df_valid.head(args.max_scenes)
        print(f"Limited to {len(df_valid)} scenes")

    # Check for existing output — support resume
    output_file = OUTPUT_DIR / "vlm_embeddings.npz"
    done_ids = set()
    existing_embeddings = []
    existing_clip_ids = []

    if output_file.exists():
        existing = np.load(output_file, allow_pickle=True)
        existing_clip_ids = list(existing["clip_ids"])
        existing_embeddings = list(existing["embeddings"])
        done_ids = set(existing_clip_ids)
        print(f"Resuming: {len(done_ids)} scenes already embedded")

    clip_ids = df_valid["clip_id"].tolist()
    to_process = [cid for cid in clip_ids if cid not in done_ids]
    print(f"To process: {len(to_process)}")

    if len(to_process) == 0:
        print("\nAll scenes already embedded. Nothing to do.")
        return 0

    # Apply patches and setup
    patch_physical_ai_av()
    setup_alpamayo()
    setup_huggingface()

    # Import alpamayo after setup
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    # Load model with SDPA attention (MPS-compatible, no flash_attention_2)
    print(f"\nLoading model: {model_id}...")
    model_load_start = time.time()

    model_config = AlpamayoR1Config.from_pretrained(model_id)
    model_config.attn_implementation = "sdpa"

    model = AlpamayoR1.from_pretrained(
        model_id, config=model_config, torch_dtype=torch.bfloat16,
    )

    # Delete components we don't need — only VLM + traj tokenizer are used.
    # Expert (~2B params), diffusion, action projections are dead weight.
    del model.expert
    del model.diffusion
    del model.action_space
    del model.action_in_proj
    del model.action_out_proj
    import gc; gc.collect()

    model = model.to(device)
    model.eval()

    from alpamayo_r1 import helper

    processor = helper.get_processor(model.tokenizer)

    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Process scenes one at a time
    all_embeddings = list(existing_embeddings)
    all_clip_ids = list(existing_clip_ids)
    failed = []
    checkpoint_interval = 10

    print(f"\nProcessing {len(to_process)} scenes...")
    print("-" * 60)

    for i, clip_id in enumerate(tqdm(to_process, desc="Extracting VLM embeddings")):
        try:
            with torch.no_grad():
                emb = extract_vlm_embedding(
                    model, processor, clip_id, device, t0_us=t0_us,
                )

            all_embeddings.append(emb)
            all_clip_ids.append(clip_id)

            # Checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                _save_checkpoint(output_file, all_embeddings, all_clip_ids)

        except Exception as e:
            import traceback
            tqdm.write(f"  ERROR [{clip_id}]: {e}")
            traceback.print_exc()
            failed.append((clip_id, str(e)))

    # Final save
    _save_checkpoint(output_file, all_embeddings, all_clip_ids)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    emb_array = np.array(all_embeddings)
    print(f"Total embedded: {len(all_clip_ids)}")
    print(f"Embedding shape: {emb_array.shape}")
    print(f"Failed: {len(failed)}")

    if len(all_embeddings) > 0:
        norms = np.linalg.norm(emb_array, axis=1)
        print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}]")

    if failed:
        print("\nFailed scenes:")
        for clip_id, error in failed[:10]:
            print(f"  {clip_id}: {error}")

    print(f"\nOutput: {output_file}")
    return 0


def _save_checkpoint(output_file: Path, embeddings: list, clip_ids: list):
    """Save current progress to disk."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        embeddings=np.array(embeddings, dtype=np.float32),
        clip_ids=np.array(clip_ids, dtype=str),
    )


if __name__ == "__main__":
    sys.exit(main())
