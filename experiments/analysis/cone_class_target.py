#!/usr/bin/env python3
"""Per-patch class-conditional target — compare modal (A) vs weighted mean (B).

For each grid position (i, j) and target class T, build a histogram
h_ij[c] over the m exemplars and derive two target representations:

  A) modal codeword:  p_t[i,j] = codebook[argmax_c h_ij[c]]
  B) weighted mean :  p_t[i,j] = Σ_c h_ij[c] · codebook[c]

For one origin image, run the cone filter at fixed α against each
target and compare:

  * survivor-count |B| distribution per method
  * decoded τ=0.5 midline image per method
  * decoded synthetic "class target" image per method
    (A directly; B by snapping to the nearest codebook entry per position)
  * modal-confidence per position (how peaky the histogram is)

Usage::

    python experiments/analysis/cone_class_target.py \\
        --origin junco --target-class macaw --m 50 --alpha 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import ImageConfig  # noqa: E402
from src.data.imagenet import ImageNetCache  # noqa: E402
from src.manipulator.image import ImageManipulator  # noqa: E402
from src.manipulator.image.cone_candidates import filter_and_order  # noqa: E402
from src.manipulator.image.types import CodeGrid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--origin", default="junco")
    p.add_argument("--target-class", default="macaw")
    p.add_argument("--m", type=int, default=50)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--preset", default="f8-16384")
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "experiments" / "analysis" / "output"),
    )
    return p.parse_args()


def build_position_histogram(target_grids: list[np.ndarray], n_codes: int) -> np.ndarray:
    """Per-position codeword histogram. Returns shape (n_pos, n_codes), values in [0, 1]."""
    n_pos = target_grids[0].size
    m = len(target_grids)
    counts = np.zeros((n_pos, n_codes), dtype=np.float64)
    for g in target_grids:
        flat = g.ravel()
        np.add.at(counts, (np.arange(n_pos), flat), 1.0)
    return counts / m


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) / (
        f"cone_class_target_{args.origin}_to_{args.target_class}_m{args.m}_a{int(args.alpha)}"
    ).replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] out dir: {out_dir}")

    config = ImageConfig(preset=args.preset)
    manipulator = ImageManipulator.from_preset(device=args.device, config=config)
    codec = manipulator.codec
    codebook = np.asarray(codec.codebook, dtype=np.float32)
    n_codes = codebook.shape[0]
    h, w = codec.grid_size
    n_pos = h * w
    print(f"[probe] codebook {codebook.shape}, grid {(h, w)}")

    print(f"[probe] fetching {args.origin}=1, {args.target_class}=m={args.m}")
    cache = ImageNetCache()
    origin_samples = cache.load_samples([args.origin], n_per_class=1)
    target_samples = cache.load_samples([args.target_class], n_per_class=args.m)
    if len(target_samples) < args.m:
        print(f"[probe] WARNING: only got {len(target_samples)} of {args.m} requested")

    print("[probe] encoding origin + targets")
    origin_grid = codec.encode(origin_samples[0].image).indices
    target_grids = [codec.encode(s.image).indices for s in target_samples]
    m_eff = len(target_grids)

    # Position histogram h_ij[c]
    print(f"[probe] building per-position histogram from m={m_eff} targets")
    h_ij = build_position_histogram(target_grids, n_codes)  # (n_pos, n_codes)

    # Modal-confidence per position
    modal_codes = h_ij.argmax(axis=1)  # (n_pos,)
    modal_freq = h_ij.max(axis=1)
    print(
        f"[probe] modal confidence — median {np.median(modal_freq):.3f}, "
        f"p25 {np.percentile(modal_freq, 25):.3f}, p75 {np.percentile(modal_freq, 75):.3f}"
    )

    # Targets per position
    p_t_modal = codebook[modal_codes]  # (n_pos, d), on-codebook
    p_t_mean = (h_ij @ codebook).astype(np.float32)  # (n_pos, d), off-codebook

    # Origin embeddings
    emb_o = codebook[origin_grid.ravel()]  # (n_pos, d)

    # Cone filter per position for each target
    alpha_rad = np.deg2rad(args.alpha)
    print(f"[probe] cone-filtering n_pos={n_pos} positions at α={args.alpha}°")

    sizes_A = np.zeros(n_pos, dtype=np.int64)
    sizes_B = np.zeros(n_pos, dtype=np.int64)
    survivors_A: list[np.ndarray] = []
    survivors_B: list[np.ndarray] = []

    for i in range(n_pos):
        sA = filter_and_order(emb_o[i], p_t_modal[i], codebook, alpha_rad=alpha_rad)
        sB = filter_and_order(emb_o[i], p_t_mean[i], codebook, alpha_rad=alpha_rad)
        survivors_A.append(sA)
        survivors_B.append(sB)
        sizes_A[i] = sA.size
        sizes_B[i] = sB.size

    def report(label: str, sizes: np.ndarray) -> None:
        nz = sizes[sizes > 0]
        med = float(np.median(sizes))
        p25 = float(np.percentile(sizes, 25))
        p75 = float(np.percentile(sizes, 75))
        frac_empty = float((sizes == 0).mean())
        print(
            f"  {label}: min={sizes.min()} p25={p25:.0f} med={med:.0f} p75={p75:.0f} "
            f"max={sizes.max()} mean={sizes.mean():.1f} empty={100*frac_empty:.1f}%"
        )

    print("[probe] |B| distributions:")
    report("A modal     ", sizes_A)
    report("B mean      ", sizes_B)

    # Histogram plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, sizes, label in zip(
        axes, [sizes_A, sizes_B], ["A: modal codeword target", "B: weighted-mean target"]
    ):
        nz = sizes[sizes > 0]
        ax.hist(nz if nz.size else sizes, bins=40, color="steelblue", edgecolor="k", linewidth=0.3)
        ax.set_title(f"{label}\n(median |B|={np.median(sizes):.0f}, empty={100*(sizes==0).mean():.1f}%)")
        ax.set_xlabel("|B_i| survivors")
        ax.set_yscale("log")
    axes[0].set_ylabel("positions (log)")
    fig.suptitle(f"{args.origin} → {args.target_class}  α={args.alpha}°  m={m_eff}")
    fig.tight_layout()
    hist_path = out_dir / "B_distribution.png"
    fig.savefig(hist_path, dpi=120)
    print(f"[probe] wrote {hist_path}")

    # Modal-confidence histogram
    fig2, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(modal_freq, bins=30, color="seagreen", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("per-position modal frequency (h_ij[argmax])")
    ax.set_ylabel("positions")
    ax.set_title(f"Modal confidence of class={args.target_class}, m={m_eff}")
    fig2.tight_layout()
    fig2.savefig(out_dir / "modal_confidence.png", dpi=120)

    # Decoded composite "target image" — A and B
    # A: per-patch modal codeword grid → decode directly
    target_grid_A = modal_codes.reshape(h, w).astype(np.int64)
    img_target_A = codec.decode(CodeGrid(target_grid_A))
    img_target_A.save(out_dir / "target_A_modal.png")

    # B: snap mean to nearest codebook entry per position, decode
    diffs = p_t_mean[:, None, :] - codebook[None, :, :]
    dists2 = (diffs * diffs).sum(axis=2)
    nearest = dists2.argmin(axis=1).astype(np.int64)
    target_grid_B = nearest.reshape(h, w)
    img_target_B = codec.decode(CodeGrid(target_grid_B))
    img_target_B.save(out_dir / "target_B_meanSnap.png")
    print(f"[probe] wrote target_A_modal.png / target_B_meanSnap.png")

    # τ=0.5 midline for each
    mid_A = origin_grid.copy()
    mid_B = origin_grid.copy()
    n_fill_A = 0
    n_fill_B = 0
    for i in range(n_pos):
        if survivors_A[i].size:
            mid_A.ravel()[i] = int(survivors_A[i][survivors_A[i].size // 2])
            n_fill_A += 1
        if survivors_B[i].size:
            mid_B.ravel()[i] = int(survivors_B[i][survivors_B[i].size // 2])
            n_fill_B += 1
    img_mid_A = codec.decode(CodeGrid(mid_A))
    img_mid_B = codec.decode(CodeGrid(mid_B))
    img_mid_A.save(out_dir / f"midline_A_modal_alpha{int(args.alpha)}.png")
    img_mid_B.save(out_dir / f"midline_B_mean_alpha{int(args.alpha)}.png")
    print(f"[probe] midline A: filled {n_fill_A}/{n_pos}; midline B: filled {n_fill_B}/{n_pos}")

    # Origin + canonical-target reference reconstructions
    codec.decode_batch([codec.encode(origin_samples[0].image)])[0].save(out_dir / "origin_recon.png")
    codec.decode_batch([codec.encode(target_samples[0].image)])[0].save(out_dir / "target_sample0_recon.png")

    # Side-by-side comparison panel
    fig3, axs = plt.subplots(2, 3, figsize=(11, 7.5))
    panels = [
        (axs[0, 0], codec.decode_batch([codec.encode(origin_samples[0].image)])[0], f"origin: {args.origin}"),
        (axs[0, 1], img_target_A, f"target A (modal codeword)\nm={m_eff}"),
        (axs[0, 2], img_target_B, f"target B (mean → nearest codebook)\nm={m_eff}"),
        (axs[1, 0], codec.decode_batch([codec.encode(target_samples[0].image)])[0], f"target sample[0]: {args.target_class}"),
        (axs[1, 1], img_mid_A, f"midline A τ=0.5  α={args.alpha}°\nfilled {n_fill_A}/{n_pos}"),
        (axs[1, 2], img_mid_B, f"midline B τ=0.5  α={args.alpha}°\nfilled {n_fill_B}/{n_pos}"),
    ]
    for ax, im, title in panels:
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig3.suptitle(
        f"Per-patch class-conditional target: A modal vs B mean   "
        f"(α={args.alpha}°, m={m_eff})"
    )
    fig3.tight_layout()
    panel_path = out_dir / "panel.png"
    fig3.savefig(panel_path, dpi=120)
    print(f"[probe] wrote {panel_path}")

    np.savez_compressed(
        out_dir / "data.npz",
        sizes_A=sizes_A,
        sizes_B=sizes_B,
        modal_codes=modal_codes,
        modal_freq=modal_freq,
        nearest_B=nearest,
        alpha=args.alpha,
        m=m_eff,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
