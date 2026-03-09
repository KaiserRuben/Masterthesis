"""VQGAN model loading from HuggingFace or taming-transformers checkpoints.

Uses the self-contained ``vqgan.VQModel`` — no dependency on taming,
pytorch-lightning, einops, or omegaconf.

Usage::

    # From HuggingFace
    model = load_huggingface_vqgan("thomwolf/vqgan_imagenet_f16_1024")

    # From taming-transformers checkpoint
    model = load_checkpoint_vqgan(config, checkpoint_path)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .vqgan import VQModel


# ---------------------------------------------------------------------------
# HuggingFace loading
# ---------------------------------------------------------------------------


def load_huggingface_vqgan(repo_id: str) -> nn.Module:
    """Load a VQGAN from a HuggingFace repository.

    Expects ``config.json`` (architecture params) and
    ``pytorch_model.bin`` (weights) in the repo.

    Args:
        repo_id: e.g. ``"thomwolf/vqgan_imagenet_f16_1024"``.

    Returns:
        ``VQModel`` in eval mode on CPU.
    """
    from huggingface_hub import hf_hub_download

    config_path = Path(hf_hub_download(repo_id, "config.json"))
    weights_path = Path(hf_hub_download(repo_id, "pytorch_model.bin"))

    with open(config_path) as f:
        hf_config = json.load(f)

    model = _vqmodel_from_hf_config(hf_config)
    _load_weights(model, weights_path, source="bin")

    return model.eval()


def _vqmodel_from_hf_config(cfg: dict[str, Any]) -> VQModel:
    """Construct a VQModel from HuggingFace config.json fields."""
    return VQModel(
        n_codes=cfg.get("n_embed", cfg.get("num_embeddings", 1024)),
        embed_dim=cfg.get("embed_dim", cfg.get("quantized_embed_dim", 256)),
        z_channels=cfg.get("z_channels", 256),
        resolution=cfg.get("resolution", 256),
        in_channels=cfg.get("in_channels", 3),
        out_ch=cfg.get("out_ch", 3),
        ch=cfg.get("ch", cfg.get("hidden_channels", 128)),
        ch_mult=tuple(cfg.get("ch_mult", cfg.get("channel_mult", [1, 1, 2, 2, 4]))),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [16])),
        dropout=cfg.get("dropout", 0.0),
        double_z=cfg.get("double_z", False),
    )


# ---------------------------------------------------------------------------
# Taming-transformers checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint_vqgan(
    config: dict[str, Any],
    checkpoint_path: Path,
) -> nn.Module:
    """Load a VQGAN from architecture config dict + checkpoint file.

    Args:
        config: Architecture parameters (same keys as HuggingFace config.json
            or a taming-style ddconfig dict with ``n_embed`` and ``embed_dim``).
        checkpoint_path: Path to ``.ckpt`` file.

    Returns:
        ``VQModel`` in eval mode on CPU.
    """
    model = _vqmodel_from_hf_config(config)
    _load_weights(model, checkpoint_path, source="ckpt")

    return model.eval()


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


_FILTER_PREFIXES = ("loss.", "discriminator.")


def _load_weights(
    model: nn.Module,
    path: Path,
    source: str,
) -> None:
    """Load weights from .bin or .ckpt, filtering non-inference keys.

    Args:
        model: Target model.
        path: Weights file.
        source: ``"bin"`` for ``pytorch_model.bin`` (flat state dict),
                ``"ckpt"`` for taming ``.ckpt`` (nested under ``"state_dict"``).
    """
    if source == "bin":
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)

    filtered = {
        k: v for k, v in sd.items()
        if not k.startswith(_FILTER_PREFIXES)
    }

    result = model.load_state_dict(filtered, strict=False)

    if result.missing_keys:
        warnings.warn(
            f"VQGAN checkpoint missing keys: {result.missing_keys}",
            stacklevel=3,
        )
    if result.unexpected_keys:
        warnings.warn(
            f"VQGAN checkpoint unexpected keys: {result.unexpected_keys}",
            stacklevel=3,
        )
