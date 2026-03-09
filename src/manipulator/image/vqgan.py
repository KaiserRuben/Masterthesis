"""Inference-only VQGAN: Encoder, Decoder, VectorQuantizer, VQModel.

Self-contained reimplementation of the taming-transformers VQModel
with identical architecture and state_dict key naming. No dependency
on taming, pytorch-lightning, einops, or omegaconf.

Weight-compatible with:
  - CompVis/taming-transformers checkpoints (.ckpt)
  - HuggingFace repos (thomwolf/vqgan_imagenet_f16_1024, etc.)

Only the forward path is implemented. Training-specific code (loss,
discriminator, commitment loss, gradient straight-through) is omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class _ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 0,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        h = _swish(self.norm1(x))
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(_swish(temb))[:, :, None, None]
        h = _swish(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h_dim, w_dim = q.shape
        q = q.reshape(b, c, h_dim * w_dim).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, h_dim * w_dim)                     # (b, c, hw)
        w = torch.bmm(q, k) * (c ** -0.5)                      # (b, hw, hw)
        w = F.softmax(w, dim=2)

        v = v.reshape(b, c, h_dim * w_dim)                     # (b, c, hw)
        h = torch.bmm(v, w.permute(0, 2, 1))                   # (b, c, hw)
        h = h.reshape(b, c, h_dim, w_dim)
        h = self.proj_out(h)

        return x + h


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """VQGAN encoder: image → continuous latent grid."""

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        **ignore_kwargs: Any,
    ) -> None:
        super().__init__()
        num_resolutions = len(ch_mult)

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(_ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=0, dropout=dropout,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != num_resolutions - 1:
                down.downsample = _Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=0, dropout=dropout,
        )
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=0, dropout=dropout,
        )

        # End
        self.norm_out = _normalize(block_in)
        out_z = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_z, 3, stride=1, padding=1)

        self._num_resolutions = num_resolutions
        self._num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self._num_resolutions):
            for i_block in range(self._num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], None)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        h = _swish(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    """VQGAN decoder: quantized latent grid → image."""

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        **ignore_kwargs: Any,
    ) -> None:
        super().__init__()
        num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (num_resolutions - 1)

        # z → block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=0, dropout=dropout,
        )
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=0, dropout=dropout,
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(_ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=0, dropout=dropout,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        # End
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

        self._num_resolutions = num_resolutions
        self._num_res_blocks = num_res_blocks
        self._give_pre_end = give_pre_end

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        for i_level in reversed(range(self._num_resolutions)):
            for i_block in range(self._num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        if self._give_pre_end:
            return h

        h = _swish(self.norm_out(h))
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Vector Quantizer (inference-only)
# ---------------------------------------------------------------------------


class VectorQuantizer(nn.Module):
    """Nearest-neighbor codebook lookup.

    At inference: finds the nearest codebook vector for each spatial
    position and returns the quantized latent grid plus the indices.

    Compatible with taming-transformers ``VectorQuantizer2`` weights.
    The ``embedding`` parameter name matches the checkpoint key
    ``quantize.embedding.weight``.
    """

    def __init__(self, n_codes: int, embed_dim: int) -> None:
        super().__init__()
        self.n_codes = n_codes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_codes, embed_dim)

    def forward(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[None, None, torch.Tensor]]:
        # z: (B, C, H, W) → (B, H, W, C) → (B*H*W, C)
        z_bhwc = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_bhwc.view(-1, self.embed_dim)

        # L2 distance to each codebook entry: ||z - e||² = ||z||² + ||e||² - 2·z·eᵀ
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2.0 * z_flat @ self.embedding.weight.T
        )

        min_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_indices).view(z_bhwc.shape)

        # Straight-through (gradient passes through as if z_q == z)
        z_q = z + (z_q.permute(0, 3, 1, 2).contiguous() - z).detach()

        # Commitment loss (not used at inference, but keeps interface compatible)
        loss = torch.tensor(0.0, device=z.device)

        return z_q, loss, (None, None, min_indices)

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Look up codebook vectors by index and reshape to (B, C, H, W)."""
        z_q = self.embedding(indices)
        z_q = z_q.view(shape)
        return z_q.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# VQModel (top-level)
# ---------------------------------------------------------------------------


class VQModel(nn.Module):
    """VQGAN: Encoder → Quantizer → Decoder.

    Inference-only. Exposes ``encode()``, ``decode()``, and
    ``quantize.get_codebook_entry()`` — the three operations
    the codec needs.

    State dict keys are identical to taming-transformers, so
    pretrained weights load with ``strict=True``.
    """

    def __init__(
        self,
        *,
        n_codes: int,
        embed_dim: int,
        z_channels: int = 256,
        resolution: int = 256,
        in_channels: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.0,
        double_z: bool = False,
    ) -> None:
        super().__init__()

        ddconfig = dict(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            dropout=dropout, resamp_with_conv=True, in_channels=in_channels,
            resolution=resolution, z_channels=z_channels, double_z=double_z,
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_codes, embed_dim)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

    def encode(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[None, None, torch.Tensor]]:
        """Encode image tensor to quantized latent grid + codebook indices.

        Args:
            x: (B, 3, H, W) in [-1, 1].

        Returns:
            (z_q, loss, (None, None, indices)) where
            z_q is (B, embed_dim, Hq, Wq) and indices is (B*Hq*Wq,).
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantize(h)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent grid to image tensor.

        Args:
            z_q: (B, embed_dim, Hq, Wq).

        Returns:
            (B, 3, H, W) in approximately [-1, 1].
        """
        z_q = self.post_quant_conv(z_q)
        return self.decoder(z_q)
