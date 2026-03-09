"""VQGAN model loading: presets, HuggingFace repos, and checkpoints.

Uses the self-contained ``vqgan.VQModel`` — no dependency on taming,
pytorch-lightning, einops, or omegaconf.

Usage::

    # By preset name (recommended)
    model = load_vqgan("f8-16384")

    # From HuggingFace
    model = load_huggingface_vqgan("thomwolf/vqgan_imagenet_f16_1024")

    # From taming-transformers checkpoint
    model = load_checkpoint_vqgan(config, checkpoint_path)
"""

from __future__ import annotations

import json
import logging
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from .vqgan import VQModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Presets for known models
# ---------------------------------------------------------------------------


class _Preset(NamedTuple):
    """Architecture config + weight location for a known VQGAN."""

    config: dict[str, Any]
    location: str  # HuggingFace repo_id or URL


PRESETS: dict[str, _Preset] = {
    "f16-1024": _Preset(
        config=dict(
            n_embed=1024, embed_dim=256, z_channels=256,
            ch=128, ch_mult=[1, 1, 2, 2, 4],
            num_res_blocks=2, attn_resolutions=[16],
            resolution=256, in_channels=3, out_ch=3,
            dropout=0.0, double_z=False,
        ),
        location="thomwolf/vqgan_imagenet_f16_1024",
    ),
    "f16-16384": _Preset(
        config=dict(
            n_embed=16384, embed_dim=256, z_channels=256,
            ch=128, ch_mult=[1, 1, 2, 2, 4],
            num_res_blocks=2, attn_resolutions=[16],
            resolution=256, in_channels=3, out_ch=3,
            dropout=0.0, double_z=False,
        ),
        location="dalle-mini/vqgan_imagenet_f16_16384",
    ),
    "f8-16384": _Preset(
        config=dict(
            n_embed=16384, embed_dim=4, z_channels=4,
            ch=128, ch_mult=[1, 2, 2, 4],
            num_res_blocks=2, attn_resolutions=[32],
            resolution=256, in_channels=3, out_ch=3,
            dropout=0.0, double_z=False,
        ),
        location="https://ommer-lab.com/files/latent-diffusion/vq-f8.zip",
    ),
}


def load_vqgan(name: str) -> nn.Module:
    """Load a VQGAN by preset name.

    Available presets:

    ========== ===== =========== ======= =========
    Name       f     Codebook    Grid    Val rFID
    ========== ===== =========== ======= =========
    f16-1024   16    1 024       16x16   7.94
    f16-16384  16    16 384      16x16   4.98
    f8-16384   8     16 384      32x32   1.14
    ========== ===== =========== ======= =========

    Returns:
        ``VQModel`` in eval mode on CPU.
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Choose from: {available}")

    preset = PRESETS[name]
    model = _vqmodel_from_hf_config(preset.config)

    if _is_url(preset.location):
        path = _download_checkpoint(preset.location)
        _load_weights(model, path, source="ckpt")
    else:
        path = _download_hf_weights(preset.location)
        _load_weights(model, path, source="bin")

    return model.eval()


def _is_url(location: str) -> bool:
    return location.startswith("http://") or location.startswith("https://")


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
# Download helpers
# ---------------------------------------------------------------------------


def _download_hf_weights(repo_id: str) -> Path:
    """Download pytorch_model.bin from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, "pytorch_model.bin"))


def _download_checkpoint(url: str) -> Path:
    """Download a checkpoint from a URL (zip or ckpt), with local caching."""
    cache_dir = Path(torch.hub.get_dir()) / "vqgan"
    # Derive a stable cache key from the URL filename
    url_stem = Path(urllib.request.urlparse(url).path).stem  # e.g. "vq-f8"
    dest_dir = cache_dir / url_stem

    # Check for cached .ckpt
    cached = _find_ckpt(dest_dir)
    if cached is not None:
        log.info("Using cached checkpoint: %s", cached)
        return cached

    dest_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s ...", url)
    tmp_path, _ = urllib.request.urlretrieve(url)
    tmp = Path(tmp_path)

    if zipfile.is_zipfile(tmp):
        log.info("Extracting to %s", dest_dir)
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dest_dir)
        tmp.unlink(missing_ok=True)
    else:
        # Assume it's a bare .ckpt file
        final = dest_dir / "model.ckpt"
        tmp.rename(final)

    ckpt = _find_ckpt(dest_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No .ckpt file found after downloading {url}"
        )
    return ckpt


def _find_ckpt(directory: Path) -> Path | None:
    """Find the first .ckpt file in a directory tree."""
    if not directory.exists():
        return None
    for p in directory.rglob("*.ckpt"):
        return p
    return None


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
