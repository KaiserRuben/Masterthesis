"""VQGAN codec: encode images to discrete code grids, decode back.

Model-agnostic — works with any ``nn.Module`` that exposes the
taming-transformers VQModel interface:

    encode(x)  → (z_q, loss, (_, _, indices))
    decode(z_q) → x_reconstructed
    quantize.embedding.weight       → codebook tensor
    quantize.get_codebook_entry(indices, shape) → z_q
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms import functional as TF

from .types import CodeGrid


class VQGANCodec:
    """Encode/decode images through a pretrained VQGAN.

    Handles all tensor bookkeeping: PIL ↔ tensor conversion,
    VQGAN normalization ([-1, 1]), resize/crop, device placement,
    and gradient suppression.

    The codec is stateless after construction — safe for concurrent
    reads from multiple threads (GIL-protected tensor ops).
    """

    __slots__ = (
        "_model",
        "_device",
        "_resolution",
        "_codebook",
        "_n_codes",
        "_embed_dim",
        "_grid_size",
    )

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        resolution: int = 256,
    ) -> None:
        self._device = torch.device(device)
        self._resolution = resolution
        self._model = model.eval().to(self._device)

        # Extract codebook once — immutable after init
        weight = model.quantize.embedding.weight
        self._codebook: NDArray[np.float32] = (
            weight.detach().cpu().numpy().copy()
        )
        self._n_codes, self._embed_dim = self._codebook.shape

        # Derive grid dimensions from a probe encode
        self._grid_size = self._probe_grid_size()

    def _probe_grid_size(self) -> tuple[int, int]:
        """Run one encode on a zero image to learn the spatial grid shape."""
        r = self._resolution
        dummy = torch.zeros(1, 3, r, r, device=self._device)
        with torch.no_grad():
            _, _, (_, _, indices) = self._model.encode(2.0 * dummy - 1.0)
        n = indices.numel()
        side = int(n**0.5)
        if side * side != n:
            raise ValueError(
                f"Expected square grid, got {n} tokens "
                f"(sqrt ≈ {n**0.5:.2f})"
            )
        return (side, side)

    # -- properties ----------------------------------------------------------

    @property
    def codebook(self) -> NDArray[np.float32]:
        """(n_codes, embed_dim) codebook matrix. Read-only view."""
        view = self._codebook.view()
        view.flags.writeable = False
        return view

    @property
    def n_codes(self) -> int:
        return self._n_codes

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def grid_size(self) -> tuple[int, int]:
        """Spatial dimensions of the code grid, e.g. (16, 16)."""
        return self._grid_size

    @property
    def resolution(self) -> int:
        """Pixel resolution the codec operates at (square)."""
        return self._resolution

    # -- preprocessing -------------------------------------------------------

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Resize and center-crop to the codec's resolution.

        Returns a PIL image at exactly (resolution × resolution),
        matching what the encoder sees. Useful for visual comparisons.
        """
        img = image.convert("RGB")
        img = TF.resize(
            img,
            self._resolution,
            interpolation=TF.InterpolationMode.LANCZOS,
            antialias=True,
        )
        return TF.center_crop(img, [self._resolution, self._resolution])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image → VQGAN input tensor [1, 3, H, W] in [-1, 1]."""
        img = self.preprocess(image)
        t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        t = 2.0 * t - 1.0  # → [-1, 1]
        return t.unsqueeze(0).to(self._device)

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """VQGAN output tensor [1, 3, H, W] in [-1, 1] → PIL image."""
        t = tensor.squeeze(0).clamp(-1.0, 1.0)
        t = (t + 1.0) / 2.0  # → [0, 1]
        return TF.to_pil_image(t.cpu())

    # -- encode / decode -----------------------------------------------------

    def encode(self, image: Image.Image) -> CodeGrid:
        """Encode a PIL image to a discrete code grid."""
        x = self._preprocess(image)
        with torch.no_grad():
            _, _, (_, _, indices) = self._model.encode(x)
        h, w = self._grid_size
        grid = indices.cpu().numpy().reshape(h, w).astype(np.int64)
        return CodeGrid(grid)

    def decode(self, grid: CodeGrid) -> Image.Image:
        """Decode a code grid back to a PIL image."""
        h, w = self._grid_size
        indices = torch.from_numpy(
            grid.indices.ravel().copy()
        ).long().to(self._device)
        shape = (1, h, w, self._embed_dim)
        with torch.no_grad():
            z_q = self._model.quantize.get_codebook_entry(indices, shape)
            x = self._model.decode(z_q)
        return self._postprocess(x)

    def reconstruct(self, image: Image.Image) -> Image.Image:
        """Encode → decode roundtrip. Shows VQGAN reconstruction quality."""
        return self.decode(self.encode(image))
