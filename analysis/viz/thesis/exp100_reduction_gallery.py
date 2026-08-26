#!/usr/bin/env python3
r"""Thesis figure: exp100-reduction-gallery -- one mapping-stage walk in images.

Four boundary-search states of ONE Exp-100 run, rendered like the Chapter-4
manipulation gallery (fig:method:gallery): the decoded input on top, below it
the amplified channel-mean difference to the codec baseline, and below that
the difference to the stage-1 flip.

    col 1  codec baseline     all-zero genotype: the codec roundtrip of the
                              seed photo, unmodified prompt.
    col 2  stage-1 flip       dense_uniform probe that flipped the answer
                              (junco -> boa constrictor), all 241 genes active.
    col 3  reduction walk     the state after the first 6 accepted resets of
                              the zero pass (the six largest-rank genes).
    col 4  minimized flip     the walk's end state after 30 attempted resets
                              (27 accepted, 3 reverted the flip and were
                              rolled back).

Run / flip choice.  seed_0117_1781060711 is the thesis running-example cell
("bird" vs "musical instrument", the junco photo of fig:res:boundary-map);
flip_id 0 is its first dense_uniform stage-1 flip.  Its walk removes 98.4% of
the rank-sum distance to the anchor (70,489 -> 1,153) -- the walk never
touches a text gene (all 30 targets are image genes), so the flip's
manipulated prompt is carried unchanged through columns 2-4.

Everything is reconstructed offline from the run's own artifacts and
validated against them before rendering:

  * ``codec.encode(origin.png)`` must reproduce ``image_original_codes`` at
    all 222 stored patch positions (context.json),
  * replaying the accepted stage-2 edits onto ``genotype_flipped`` must end
    exactly at ``genotype_min`` (archive.parquet),
  * recomputed rank-sum deltas must equal ``d_i_primary`` and the trace,
  * the decoded anchor and stage-1 flip must match the stored
    ``anchor_544.png`` / ``flip_0000_stage1.png`` within re-decode tolerance
    (report printed; the run decoded on another host).

Diff convention = fig:method:gallery: channel-mean absolute difference in
percent of the pixel range, clipped at 100/gain with gain 8; the printed RMS
is ``MatrixDistance(norm="fro")`` itself (channel-mean of per-channel RMS),
imported from ``src.objectives`` so the number cannot drift from the
objective the search minimizes.

Produces
    figures/results/exp100-reduction-gallery.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/exp100_reduction_gallery.py
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# Set THESIS_DIR to the thesis checkout to emit into it; otherwise figures
# land inside the repository. See docs/REPRODUCTION.md.
THESIS = Path(os.environ.get("THESIS_DIR", REPO / "analysis" / "outputs" / "thesis"))

RUN = REPO / "runs/Exp-100/poc_boundary_pair/seed_0117_1781060711"
OUT = THESIS / "figures/results/exp100-reduction-gallery.png"

FLIP_ID = 0
ANCHOR_IDX = 544          # pareto_idx of the flip's anchor
N_IMG, N_TXT = 222, 19
WALK_CUT = 5              # intermediate state = after accepted steps 0..5
GAIN = 8.0
DPI = 600

# fig:method:gallery conventions
DIFF_RAMP = ("#FFFFFF", "#A8C8DC", "#2274A5", "#123A54")
INK, INK_MUTED = "#1A1A1A", "#555555"
F_TITLE, F_ROW, F_ANN, F_CBAR = 9.5, 9.0, 7.2, 7.5

PROMPT_ZERO = "What is the main subject in this image?"


def load_states():
    """Reconstruct the four genotypes and every number the figure quotes."""
    import pandas as pd

    ctx = json.load(open(RUN / "pdq/context.json"))
    arc = pd.read_parquet(RUN / "pdq/archive.parquet")
    row = arc[arc.flip_id == FLIP_ID].iloc[0]
    assert int(row.pareto_idx) == ANCHOR_IDX, row.pareto_idx

    g_anchor = np.asarray(row.genotype_anchor, dtype=np.int64)
    g_flip = np.asarray(row.genotype_flipped, dtype=np.int64)
    g_min = np.asarray(row.genotype_min, dtype=np.int64)
    assert g_anchor.shape == g_flip.shape == g_min.shape == (N_IMG + N_TXT,)
    assert row.label_anchor == "junco" and row.label_flipped == "boa constrictor"
    assert row.label_min == "boa constrictor"

    # replay the walk: accepted edits, in step order, onto the flip genotype
    s2 = pd.read_parquet(RUN / "pdq/stage2_trajectories.parquet")
    walk = s2[s2.flip_id == FLIP_ID].sort_values("step")
    assert len(walk) == 30 and int(walk.accepted.sum()) == 27
    assert (walk.target_gene < N_IMG).all(), "walk touches a text gene"
    g = g_flip.copy()
    g_mid = None
    for _, st in walk.iterrows():
        if not st.accepted:
            assert st.label_after == "junco", "rejected step must revert"
            continue
        assert g[st.target_gene] == st.old_value, "replay out of sync"
        g[st.target_gene] = st.new_value
        if st.step == WALK_CUT:
            g_mid = g.copy()
    assert g_mid is not None and (g == g_min).all(), "replay must end at genotype_min"

    # rank-sum deltas to the anchor: the walk's own d_i quantity
    def delta(x):
        return int(np.abs(x - g_anchor).sum())

    d_flip, d_mid, d_min = delta(g_flip), delta(g_mid), delta(g_min)
    assert d_flip == int(walk.rank_sum_before.iloc[0]), d_flip
    assert d_min == int(row.d_i_primary), (d_min, row.d_i_primary)
    assert d_mid == int(walk[walk.step == WALK_CUT].rank_sum_after.iloc[0])

    # the flip's manipulated prompt, carried unchanged through the walk
    s1 = pd.read_parquet(RUN / "pdq/stage1_flips.parquet")
    cand_id = int(s1[s1.flip_id == FLIP_ID].candidate_id.iloc[0])
    cand = pd.read_parquet(RUN / "pdq/candidates.parquet")
    prompt = str(cand[cand.candidate_id == cand_id].rendered_text.iloc[0])
    assert (g_flip[N_IMG:] == g_min[N_IMG:]).all()

    genos = {"zero": np.zeros_like(g_flip), "flip": g_flip,
             "mid": g_mid, "min": g_min}
    stats = {"d_flip": d_flip, "d_mid": d_mid, "d_min": d_min,
             "n_reset_mid": 6, "n_reset_min": 27, "prompt": prompt,
             "p_a_anchor": 0.830}
    return ctx, genos, g_anchor, stats


def decode_all(ctx, genos, g_anchor):
    """Encode the origin, validate, decode every state; return PIL images."""
    from PIL import Image

    from src.manipulator.image.codec import VQGANCodec
    from src.manipulator.image.loading import load_vqgan
    from src.manipulator.image.manipulator import apply_genotype
    from src.manipulator.image.types import PatchSelection

    codec = VQGANCodec(load_vqgan("f8-16384"), device="cpu", resolution=256)
    origin = Image.open(RUN / "evolutionary/origin.png")
    grid = codec.encode(origin)

    positions = np.asarray(ctx["image_patch_positions"], dtype=np.intp)
    original_codes = np.asarray(ctx["image_original_codes"], dtype=np.int64)
    got = grid.indices[positions[:, 0], positions[:, 1]]
    n_ok = int((got == original_codes).sum())
    assert n_ok == N_IMG, f"origin encode reproduces {n_ok}/{N_IMG} codes"

    selection = PatchSelection(
        positions=positions,
        candidates=tuple(np.asarray(c, dtype=np.int64)
                         for c in ctx["image_candidates"]),
        original_codes=original_codes,
    )

    keys = list(genos)
    grids = [apply_genotype(grid, selection, genos[k][:N_IMG]) for k in keys]
    grids.append(apply_genotype(grid, selection, g_anchor[:N_IMG]))
    images = codec.decode_batch(grids)
    out = dict(zip(keys + ["anchor"], images))

    # cross-host re-decode check against the stored run images
    for key, ref in (("flip", RUN / f"pdq/flips/flip_{FLIP_ID:04d}_stage1.png"),
                     ("anchor", RUN / f"pdq/anchors/anchor_{ANCHOR_IDX}.png")):
        a = np.asarray(out[key], dtype=np.float64) / 255.0
        b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64) / 255.0
        rms = float(np.sqrt(((a - b) ** 2).mean(axis=(0, 1))).mean())
        print(f"re-decode check {key}: RMS vs stored PNG = {rms * 100:.3f}%")
        assert rms < 0.005, f"{key}: re-decode deviates by {rms:.4f}"
    return out


def matrix_distance_fn():
    """The image objective itself, as ``d(image, baseline)`` on [0,1] HWC."""
    import torch

    from src.objectives import MatrixDistance

    criterion = MatrixDistance()

    def chw(arr):
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    def distance(image, baseline) -> float:
        return float(criterion.evaluate(images=[chw(image), chw(baseline)]))

    return distance


def render(images, stats) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    matplotlib.rcParams["pdf.fonttype"] = 42

    dist = matrix_distance_fn()
    arr = {k: np.asarray(v, dtype=np.float64) / 255.0 for k, v in images.items()}
    cols = ["zero", "flip", "mid", "min"]
    titles = [
        "codec baseline\n(all-zero genotype)",
        "stage-1 flip\n(241 genes active)",
        f"reduction walk\n({stats['n_reset_mid']} of 27 resets)",
        f"minimized flip\n({stats['n_reset_min']} of 27 resets)",
    ]
    answers = ["junco", "boa constrictor", "boa constrictor", "boa constrictor"]
    dranks = [None, stats["d_flip"], stats["d_mid"], stats["d_min"]]

    cmap = mcolors.LinearSegmentedColormap.from_list("diff_blue", DIFF_RAMP)
    vmax_pct = 100.0 / GAIN
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax_pct)
    label_box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82)

    # -- geometry (inches) --------------------------------------------------
    fig_w = 6.7
    M_L, M_R_CB, CBAR_W = 0.78, 0.62, 0.06
    M_TOP, M_BOT = 0.58, 0.66
    GC, GR = 0.07, 0.24
    T = (fig_w - M_L - M_R_CB - CBAR_W - 3 * GC) / 4
    fig_h = M_TOP + 3 * T + 2 * GR + M_BOT

    fig = plt.figure(figsize=(fig_w, fig_h))

    def ax_at(x, y, w, h):
        ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.75")
        return ax

    xs = [M_L + j * (T + GC) for j in range(4)]
    ys = [fig_h - M_TOP - (r + 1) * T - r * GR for r in range(3)]

    row_labels = ["decoded input",
                  f"$|$input $-$ baseline$|$\n($\\times$ {GAIN:g})",
                  f"$|$input $-$ flip$|$\n($\\times$ {GAIN:g})"]
    for r, lab in enumerate(row_labels):
        fig.text((M_L - 0.10) / fig_w, (ys[r] + T / 2) / fig_h, lab,
                 ha="right", va="center", fontsize=F_ROW, color=INK,
                 linespacing=1.35)

    for j, key in enumerate(cols):
        img = arr[key]
        # row 1: the decoded input
        ax = ax_at(xs[j], ys[0], T, T)
        ax.imshow(images[key], interpolation="antialiased")
        head = titles[j] + ("" if dranks[j] is None
                            else f"\n$\\Delta$rank {dranks[j]:,}")
        fig.text((xs[j] + T / 2) / fig_w, (ys[0] + T + 0.06) / fig_h, head,
                 ha="center", va="bottom", fontsize=F_TITLE, color=INK,
                 linespacing=1.3)
        ax.text(0.5, 0.025, answers[j], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=F_ANN,
                color=INK if j == 0 else "#7A1E1E",
                fontstyle="italic", bbox=label_box)
        # rows 2-3: differences
        for r, ref in ((1, arr["zero"]), (2, arr["flip"])):
            axd = ax_at(xs[j], ys[r], T, T)
            diff_pct = 100.0 * np.abs(img - ref).mean(axis=2)
            axd.imshow(diff_pct, cmap=cmap, norm=norm,
                       interpolation="antialiased")
            axd.text(0.5, 0.025, f"RMS {dist(img, ref) * 100:.2f}%",
                     transform=axd.transAxes, ha="center", va="bottom",
                     fontsize=F_ANN, color=INK, bbox=label_box)

    # -- colourbar ----------------------------------------------------------
    cax = ax_at(xs[3] + T + 0.14, ys[2], CBAR_W, 2 * T + GR)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cax.tick_params(labelsize=F_CBAR, length=2.0, pad=1.6)
    cax.set_yticks([0, 4, 8, 12])
    cax.set_yticklabels(["0%", "4%", "8%", "12%"])
    fig.text((xs[3] + T + 0.14 + CBAR_W / 2) / fig_w,
             (ys[2] + 2 * T + GR + 0.06) / fig_h,
             "$|$diff$|$", ha="center", va="bottom", fontsize=F_CBAR,
             color=INK_MUTED)

    # -- prompt lines -------------------------------------------------------
    fig.text(M_L / fig_w, (M_BOT - 0.18) / fig_h,
             f"prompt, col 1:  “{PROMPT_ZERO}”",
             ha="left", va="top", fontsize=F_ANN, color=INK_MUTED)
    fig.text(M_L / fig_w, (M_BOT - 0.36) / fig_h,
             f"prompt, cols 2–4 (18 of 19 text genes active, "
             f"untouched by the walk):  “{stats['prompt']}”",
             ha="left", va="top", fontsize=F_ANN, color=INK_MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}  ({fig_w:.2f} x {fig_h:.2f} in at {DPI} dpi)")


def main() -> None:
    ctx, genos, g_anchor, stats = load_states()
    print(f"rank-sum delta to anchor: flip {stats['d_flip']:,} -> "
          f"mid {stats['d_mid']:,} -> min {stats['d_min']:,} "
          f"({100 * (1 - stats['d_min'] / stats['d_flip']):.1f}% removed)")
    print(f"prompt of the flip: {stats['prompt']!r}")
    images = decode_all(ctx, genos, g_anchor)
    render(images, stats)


if __name__ == "__main__":
    main()
