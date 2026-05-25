"""StyleGAN-XL checkpoint loading + SHA hashing.

Wraps :func:`smoo.manipulator.style_gan_manipulator.load_stylegan` with a
checkpoint cache so the runner downloads the pickle once per machine.
Also exposes :func:`checkpoint_sha256_hex` so the class-target /
accepted-origin caches can key their entries on the *content* of the
loaded weights, not the URL.

No defensive try/except wrappers — a missing local pickle plus a missing
URL is a configuration error and should fail loudly.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution + download
# ---------------------------------------------------------------------------


def resolve_checkpoint_path(path: Path) -> Path:
    """Expand ``~`` and resolve to an absolute path without touching disk."""
    return Path(path).expanduser().resolve()


def ensure_checkpoint(url: str, path: Path) -> Path:
    """Return ``path`` if present, otherwise download from ``url``.

    Creates parent directories as needed. The download is atomic: writes
    to ``<path>.partial`` then renames on completion, so a partial file
    never masquerades as a complete checkpoint.

    :param url: Source URL for the NVlabs StyleGAN-XL pickle.
    :param path: Local destination. Created if missing.
    :returns: The resolved on-disk path.
    :raises ValueError: If ``url`` is empty and ``path`` does not exist.
    """
    resolved = resolve_checkpoint_path(path)
    if resolved.exists():
        logger.info("StyleGAN checkpoint cached at %s", resolved)
        return resolved
    if not url:
        raise ValueError(
            f"StyleGAN checkpoint {resolved} is missing and no "
            f"checkpoint_url was configured."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_suffix(resolved.suffix + ".partial")
    logger.info("Downloading StyleGAN checkpoint from %s -> %s", url, resolved)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted URL from config
    tmp.replace(resolved)
    return resolved


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def checkpoint_sha256_hex(path: Path, *, prefix_chars: int = 16) -> str:
    """Compute SHA-256 of an on-disk checkpoint, truncated for cache keys.

    The full digest is 64 hex chars; we truncate to ``prefix_chars`` (16
    by default) for terseness in Redis keys. Collision probability at
    this length is negligible for the handful of checkpoints we will
    realistically load.

    :param path: On-disk file path.
    :param prefix_chars: Hex characters to keep (default 16).
    :returns: Lowercase hex prefix of the SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_chars]


# ---------------------------------------------------------------------------
# Generator loading (thin wrapper over SMOO)
# ---------------------------------------------------------------------------


_SMOO_SG_ROOT = "smoo.manipulator.style_gan_manipulator"
_PICKLE_SG_ROOT = "src.manipulator.style_gan_manipulator"


def _install_legacy_unpickle_aliases() -> None:
    """Alias SMOO's StyleGAN package tree under the path NVlabs pickles expect.

    Two flavours of alias are needed before unpickling an NVlabs-trained
    StyleGAN-XL checkpoint:

    1. ``smoo.manipulator.style_gan_manipulator.*`` →
       ``src.manipulator.style_gan_manipulator.*`` — pickles were saved with
       module paths rooted at ``src.*`` (SMOO's pre-rename layout); SMOO
       installs them under ``smoo.*`` so the original path no longer
       resolves.
    2. ``timm.models.layers.*`` → ``timm.layers.*`` — newer timm versions
       moved layer modules; older NVlabs pickles still reference the old
       path. We forward whichever ``timm.layers`` submodules the pickle is
       likely to demand.
    """
    import importlib  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415
    import sys  # noqa: PLC0415

    root = importlib.import_module(_SMOO_SG_ROOT)
    queue = [root]
    while queue:
        mod = queue.pop()
        alias = mod.__name__.replace(_SMOO_SG_ROOT, _PICKLE_SG_ROOT, 1)
        sys.modules.setdefault(alias, mod)
        if hasattr(mod, "__path__"):
            for _, sub, _ in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    queue.append(importlib.import_module(sub))
                except Exception:  # noqa: BLE001 - some submodules import on first use
                    continue

    import src.manipulator as _our_manipulator  # noqa: PLC0415
    if not hasattr(_our_manipulator, "style_gan_manipulator"):
        _our_manipulator.style_gan_manipulator = root

    # timm.models.layers → timm.layers shim. Mirror the package itself plus
    # every submodule reachable from ``timm.layers`` so the unpickler can
    # resolve any deep reference.
    timm_layers = importlib.import_module("timm.layers")
    sys.modules.setdefault("timm.models.layers", timm_layers)
    if hasattr(timm_layers, "__path__"):
        for _, sub, _ in pkgutil.iter_modules(timm_layers.__path__, timm_layers.__name__ + "."):
            try:
                submod = importlib.import_module(sub)
            except Exception:  # noqa: BLE001 - tolerate optional deps
                continue
            shim = sub.replace("timm.layers", "timm.models.layers", 1)
            sys.modules.setdefault(shim, submod)

    # Newer timm (≥1.0) renamed several internal helpers with a leading
    # underscore; older NVlabs pickles still reference the public names.
    # Forward each public alias to its underscored counterpart when present.
    timm_models = importlib.import_module("timm.models")
    if hasattr(timm_models, "__path__"):
        for _, sub, _ in pkgutil.iter_modules(timm_models.__path__, timm_models.__name__ + "."):
            base = sub.rsplit(".", 1)[-1]
            if not base.startswith("_"):
                continue
            try:
                submod = importlib.import_module(sub)
            except Exception:  # noqa: BLE001
                continue
            public_name = sub.rsplit(".", 1)[0] + "." + base.lstrip("_")
            sys.modules.setdefault(public_name, submod)


def load_generator(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a StyleGAN-XL generator from a NVlabs pickle file.

    Delegates to :func:`smoo.manipulator.style_gan_manipulator.load_stylegan`
    which itself wraps NVlabs' legacy ``load_network_pkl``. The returned
    module is moved to ``device`` and switched to eval mode.

    :param checkpoint_path: Local path to the pickle file. Must already
        exist; use :func:`ensure_checkpoint` to download on demand.
    :param device: Torch device the generator should live on.
    :returns: The ``G_ema`` generator submodule, ready for inference.
    """
    _install_legacy_unpickle_aliases()
    # Imported lazily so unit tests that mock the generator do not need
    # the SMOO StyleGAN tree on the import path.
    from smoo.manipulator.style_gan_manipulator import (  # noqa: PLC0415
        _load_stylegan,
    )
    generator = _load_stylegan.load_stylegan(str(checkpoint_path))
    generator.to(device)
    generator.eval()
    return generator


__all__ = [
    "checkpoint_sha256_hex",
    "ensure_checkpoint",
    "load_generator",
    "resolve_checkpoint_path",
]
