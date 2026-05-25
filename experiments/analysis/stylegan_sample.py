#!/usr/bin/env python3
"""Quick StyleGAN-XL sample probe.

Downloads the NVlabs ImageNet256 checkpoint (if not cached), generates a
class-conditional sample for one ImageNet class, and saves the image. Used
to eyeball that the StyleGAN backend produces sensible class samples before
running a full experiment.

Usage::

    python experiments/analysis/stylegan_sample.py --class-name junco
    python experiments/analysis/stylegan_sample.py --class-idx 13 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.imagenet import ImageNetCache  # noqa: E402
from src.manipulator.image_stylegan.loading import ensure_checkpoint, load_generator  # noqa: E402


DEFAULT_URL = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
DEFAULT_PATH = "~/.cache/stylegan_xl/imagenet256.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--class-name", default="junco")
    g.add_argument("--class-idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=4, help="Samples to generate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-url", default=DEFAULT_URL)
    p.add_argument("--checkpoint-path", default=DEFAULT_PATH)
    p.add_argument("--trunc-psi", type=float, default=1.0)
    p.add_argument("--trunc-cutoff", type=int, default=0)
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "experiments" / "analysis" / "output" / "stylegan_sample"),
    )
    return p.parse_args()


def resolve_class_idx(args: argparse.Namespace) -> tuple[int, str]:
    cache = ImageNetCache()
    labels = cache.labels()
    if args.class_idx is not None:
        return args.class_idx, labels[args.class_idx]
    target = args.class_name
    idx = labels.index(target)
    return idx, target


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_idx, class_name = resolve_class_idx(args)
    print(f"[sample] class={class_name!r} idx={class_idx} seed={args.seed} n={args.n}")

    ckpt = ensure_checkpoint(args.checkpoint_url, Path(args.checkpoint_path))
    device = torch.device(args.device)
    print(f"[sample] loading generator from {ckpt} onto {device}")
    G = load_generator(ckpt, device)
    G.eval()

    # Class-conditional w via StyleGAN-XL mapping network
    torch.manual_seed(args.seed)
    z = torch.randn(args.n, G.z_dim, device=device)
    c = torch.zeros(args.n, G.c_dim, device=device)
    c[:, class_idx] = 1.0
    with torch.no_grad():
        w = G.mapping(z=z, c=c, truncation_psi=args.trunc_psi, truncation_cutoff=args.trunc_cutoff)
        imgs = G.synthesis(w, noise_mode="random", force_fp32=False)

    # imgs: (n, 3, H, W) in approx [-1, 1] (StyleGAN-XL output convention)
    imgs = imgs.clamp(-1.0, 1.0)
    imgs = (imgs + 1.0) / 2.0
    imgs = (imgs * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
    imgs = imgs.permute(0, 2, 3, 1).numpy()

    for i in range(args.n):
        path = out_dir / f"{class_name.replace(' ', '_')}_seed{args.seed}_{i}.png"
        Image.fromarray(imgs[i]).save(path)
        print(f"[sample] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
