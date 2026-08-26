r"""Thesis figure: exp-init-coverage — what the initial population decides.

Two panels, one claim each, both isolating the *initialization sampler* while
budget, operators, objectives, SUT and seed stay frozen.

(a) Exp-09 vs Exp-10, great white shark vs tiger shark, seed 5.  One point per
    final Pareto-front solution.  x is the number of active image sites of that
    solution (image genes != 0; gene 0 means "leave this patch alone"), y its
    targeted balance |lp_A - lp_B| in nats on a log axis.  Both runs are
    200 generations x 30 individuals over the full 16383-word codebook and
    differ only in ``optimizer.sampling.mode`` (``uniform`` vs ``sparse``,
    p_active = 0.03).  The uniform front is a single vertical strip: all 94 of
    its solutions have all 228 image sites active, and the sparse half of the
    space holds none of them.  The sparse-prior front spreads over 0..42 active
    sites, puts 96 of 174 solutions at or below 10 sites, and reaches a floor
    two orders of magnitude lower.

(b) Exp-22 / 22b / 22c, junco vs chickadee, seed 83.  Best targeted balance so
    far against generation, log y.  Three configs that share pair, seed, SUT,
    300 x 30 budget, full-codebook image space and the ``full_stack`` text
    operator profile; the only difference in the yaml is the block
    ``optimizer.sampling``.  The floor moves 2.4311 -> 2.0768 -> 1.8481 and the
    generation at which each run first gets under 2.5 nats moves 159 -> 24 -> 15.

Both x axes are square-root-scaled, the convention the Exp-100 budget figure
already uses: in (a) the sparse-prior front lives in the first 20 of 228 site
counts, in (b) two of the three ladder rungs clear 2.5 nats inside 25 of 300
generations, and on a linear axis both stories are a smear against the left
spine.  A log axis cannot be used in either panel because 0 active sites and
generation 0 are real, load-bearing values.

The run identity of each panel is a grey second title line, not an in-axes
note, because (a) has no empty corner left once the legend is placed and (b)
would have to put it under a falling curve.  In (b) the arrival generation
rides in each curve's legend entry for the same reason: two of the three
crossings are 9 generations apart and all three sit where the curves are
steepest, so an in-plot number would have to be pushed away from the marker it
names.  The marker on the 2.5 line stays, since that placement is unambiguous.

Colours.  The three ladder rungs are one purple hue in three lightness steps,
because the ladder is ordered (single-p sparse -> multi-tier -> score-guided);
lightness carries that order into print greyscale, and the dash patterns carry
it again for readers who lose the hue.  Panel (a) reuses the first rung for the
sparse-prior run, so the same sampler has the same colour in both panels, and
puts the uniform baseline in neutral near-black.  The SUT colours and the
blue/vermillion answer coding are deliberately not borrowed: nothing here is a
system comparison and nothing here is an answer.

Data sources (all read-only, no aggregate csv in between)
    runs/Exp-09/exp09_M0_n16383_shark_seed_5_1776512034/
    runs/Exp-10/exp10_phase1_shark_n16383_seed_5_1776620110/
        pareto_<i>.json  -> genotype (228 image genes + 3 text genes) and
                            fitness [MatrixDistance_fro, TextDist, TgtBal]
        convergence.parquet -> n_pareto, used only to assert that the
                            pareto_*.json set is the final front
    runs/Exp-22/exp22_mlm_composite_junco_chickadee_seed_83_1777365666/
    runs/Exp-22/exp22b_multitier_junco_chickadee_seed_83_1777390834/
    runs/Exp-22/exp22c_pattern_junco_chickadee_seed_83_1777404499/
        convergence.parquet -> pareto_min_TgtBal, which is exactly the running
                            minimum of pop_min_TgtBal (asserted below), i.e.
                            the best population minimum up to that generation
    configs/Archive/Exp-09/M0_n16383.yaml, configs/Exp-10/phase1_shark_n16383.yaml
    configs/Exp-22/mlm_composite_junco_chickadee{,_multitier,_score_pattern}.yaml
        -> asserted equal on budget, population, codebook size and text profile,
           and different only in ``optimizer.sampling``

Produces
    figures/results/exp-init-coverage.pdf
    figures/results/exp-init-coverage.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/exp_init_coverage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from _common import FS_ANN, FS_LABEL, FS_TICK, W_FULL, rect, save, setup

REPO = Path("/Users/kaiser/Projects/Masterarbeit")

# --- panel (a): Exp-09 uniform vs Exp-10 sparse prior, same pair and seed ----
RUN_UNIFORM = REPO / "runs/Exp-09/exp09_M0_n16383_shark_seed_5_1776512034"
RUN_SPARSE = REPO / "runs/Exp-10/exp10_phase1_shark_n16383_seed_5_1776620110"
CFG_UNIFORM = REPO / "configs/Archive/Exp-09/M0_n16383.yaml"
CFG_SPARSE = REPO / "configs/Exp-10/phase1_shark_n16383.yaml"

N_IMG_SHARK = 228       # image genes; genotype is 228 image + 3 text = 231
N_TXT = 3
SPARSE_CUT = 10         # "sparse" region of the site-count axis

# --- panel (b): the initialization-sampler ladder, one pair, frozen operators -
LADDER = (
    ("sparse", "sparse, $p=0.03$", "sparse",
     "runs/Exp-22/exp22_mlm_composite_junco_chickadee_seed_83_1777365666",
     "configs/Exp-22/mlm_composite_junco_chickadee.yaml"),
    ("multitier", "multi-tier", "sparse_multitier",
     "runs/Exp-22/exp22b_multitier_junco_chickadee_seed_83_1777390834",
     "configs/Exp-22/mlm_composite_junco_chickadee_multitier.yaml"),
    ("pattern", "score-guided (pattern)", "sparse_score_guided",
     "runs/Exp-22/exp22c_pattern_junco_chickadee_seed_83_1777404499",
     "configs/Exp-22/mlm_composite_junco_chickadee_score_pattern.yaml"),
)
REACH = 2.5             # nats; the level whose arrival time the panel marks

# --- colours ----------------------------------------------------------------
# One purple hue in three lightness steps: the ladder is ordered, so the
# encoding has to be ordered too, in colour and in greyscale.
C_LADDER = {"sparse": "#B0559F", "multitier": "#77439B", "pattern": "#3A2A6B"}
DASH = {"sparse": (0, (1.6, 1.5)), "multitier": (0, (4.2, 1.8)),
        "pattern": (0, ())}
C_UNIFORM = "#1A1A1A"   # the no-prior baseline, neutral by construction
C_SPARSE_EDGE = "#8A3A7E"
BAND_FC = "#EFEFEF"
BAND_EC = "#C4C4C4"

# White stroke behind in-plot numbers, so a label stays readable where a curve
# or a marker passes under it without hiding either (same device as the
# Exp-104 prior map).
HALO = [pe.withStroke(linewidth=1.9, foreground="white")]

# --- inch budget ------------------------------------------------------------
W = W_FULL
H = 3.18
L = 0.60                # y label + y ticks, panel (a)
MID = 0.58              # y label + y ticks, panel (b)
R_PAD = 0.10
B_TICKS, B_LABEL, B_PAD = 0.16, 0.20, 0.05
T_TITLE, T_PAD = 0.42, 0.03   # two title lines: name, then run identity
AXW = (W - L - MID - R_PAD) / 2.0
BOTTOM = B_PAD + B_LABEL + B_TICKS
AXH = H - BOTTOM - T_TITLE - T_PAD
X0_A, X0_B = L, L + AXW + MID

# Square-root axis shifted by one so that x = 0 is a drawable, visible value:
# 0 active sites and generation 0 are both real observations here.
def _fwd(x):
    return np.sqrt(np.clip(np.asarray(x, dtype=float) + 1.0, 0.0, None))


def _inv(y):
    return np.asarray(y, dtype=float) ** 2 - 1.0


SQRT1 = (_fwd, _inv)


# ---------------------------------------------------------------------------
# loading + provenance
# ---------------------------------------------------------------------------
def front(run: Path, n_img: int) -> tuple[np.ndarray, np.ndarray]:
    """Final Pareto front of one run: active image sites and TgtBal per solution.

    ``pareto_<i>.json`` is written once at the end of a run, so the file set is
    the final front; that is asserted against ``n_pareto`` of the last row of
    convergence.parquet.  A genotype entry of 0 means "keep the original code
    at this site", so the active-site count is the number of non-zero genes in
    the leading image block.  ``fitness`` is
    ``[MatrixDistance_fro, TextDist, TgtBal]`` in the order the convergence
    columns list the criteria.
    """
    files = sorted(run.glob("pareto_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    idx = [int(p.stem.split("_")[1]) for p in files]
    assert idx == list(range(len(files))), f"{run.name}: gap in pareto_*.json"

    conv = pd.read_parquet(run / "convergence.parquet",
                           columns=["generation", "n_pareto"])
    assert conv.generation.iloc[0] == 0 and len(conv) == 200, \
        f"{run.name}: expected 200 generations indexed from 0"
    assert int(conv.n_pareto.iloc[-1]) == len(files), \
        (f"{run.name}: {len(files)} pareto files vs n_pareto="
         f"{int(conv.n_pareto.iloc[-1])} at the last generation")

    n_act, tgt = [], []
    for p in files:
        d = json.loads(p.read_text())
        g = np.asarray(d["genotype"], dtype=np.int64)
        assert len(g) == n_img + N_TXT, f"{p.name}: genotype length {len(g)}"
        assert len(d["fitness"]) == 3, f"{p.name}: expected 3 objectives"
        n_act.append(int(np.count_nonzero(g[:n_img])))
        tgt.append(float(d["fitness"][2]))
    return np.asarray(n_act), np.asarray(tgt)


def _cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def check_panel_a() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load both fronts and assert everything the caption claims."""
    cu, cs = _cfg(CFG_UNIFORM), _cfg(CFG_SPARSE)
    for name, c in (("Exp-09", cu), ("Exp-10", cs)):
        assert c["generations"] == 200 and c["pop_size"] == 30, \
            f"{name}: budget is not 200 x 30"
        assert c["image"]["n_candidates"] == 16383, f"{name}: codebook changed"
        assert c["sut"]["model_id"] == cu["sut"]["model_id"], f"{name}: SUT differs"
        assert "text" not in c, f"{name}: has a text-operator override"
    assert cu["optimizer"]["sampling"]["mode"] == "uniform"
    assert cs["optimizer"]["sampling"]["mode"] == "sparse"
    assert cs["optimizer"]["sampling"]["p_active"] == 0.03

    nu, tu = front(RUN_UNIFORM, N_IMG_SHARK)
    ns, ts = front(RUN_SPARSE, N_IMG_SHARK)

    assert (len(nu), len(ns)) == (94, 174), (len(nu), len(ns))
    assert nu.min() == nu.max() == N_IMG_SHARK, \
        f"uniform front is not pinned at {N_IMG_SHARK} sites: {nu.min()}..{nu.max()}"
    assert int((nu <= SPARSE_CUT).sum()) == 0, "uniform front entered the sparse region"
    assert int((ns <= SPARSE_CUT).sum()) == 96, int((ns <= SPARSE_CUT).sum())
    assert ns.min() == 0 and ns.max() == 42, (ns.min(), ns.max())
    assert abs(tu.min() - 0.209828) < 5e-7, tu.min()
    assert abs(ts.min() - 0.001651) < 5e-7, ts.min()
    return nu, tu, ns, ts


def check_panel_b() -> dict[str, pd.DataFrame]:
    """Load the three trajectories and assert the ladder is init-only."""
    cfgs = {k: _cfg(REPO / cp) for k, _, _, _, cp in LADDER}
    base = cfgs["sparse"]
    for k, _, mode, _, _ in LADDER:
        c = cfgs[k]
        assert c["generations"] == 300 and c["pop_size"] == 30, f"{k}: budget"
        assert c["image"] == base["image"], f"{k}: image space differs"
        assert c["text"] == base["text"], f"{k}: text operator profile differs"
        assert c["sut"] == base["sut"] and c["seeds"] == base["seeds"], f"{k}: pair/SUT"
        assert c["optimizer"]["early_stop"] == base["optimizer"]["early_stop"], \
            f"{k}: early-stop policy differs"
        assert c["optimizer"]["sampling"]["mode"] == mode, f"{k}: sampler mode"
        assert set(c["optimizer"]) == {"sampling", "early_stop"}, \
            f"{k}: optimizer block carries more than sampling"

    want_floor = {"sparse": 2.4311, "multitier": 2.0768, "pattern": 1.8481}
    want_reach = {"sparse": 159, "multitier": 24, "pattern": 15}
    out = {}
    for k, _, _, rp, _ in LADDER:
        c = pd.read_parquet(REPO / rp / "convergence.parquet",
                            columns=["generation", "pareto_min_TgtBal",
                                     "pop_min_TgtBal"])
        assert len(c) == 300 and c.generation.iloc[0] == 0 \
            and c.generation.iloc[-1] == 299, f"{k}: unexpected generation index"
        # The plotted series is the best population minimum so far.  That is
        # what pareto_min_TgtBal already is -- assert it rather than trust it.
        assert np.allclose(c.pareto_min_TgtBal,
                           np.minimum.accumulate(c.pop_min_TgtBal)), \
            f"{k}: pareto_min_TgtBal is not the running min of pop_min_TgtBal"
        floor = float(c.pareto_min_TgtBal.iloc[-1])
        assert abs(round(floor, 4) - want_floor[k]) < 5e-5, (k, floor)
        hit = c.generation[c.pareto_min_TgtBal <= REACH]
        assert len(hit) and int(hit.min()) == want_reach[k], (k, list(hit[:1]))
        out[k] = c
    return out


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def panel_a(ax, nu, tu, ns, ts) -> None:
    ax.set_xscale("function", functions=SQRT1)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 244)
    # Headroom above the data (max 1.65) for a one-row legend, so the legend
    # never has to sit on the point cloud or in the empty lower-right corner
    # that is half of what the panel is about.
    ax.set_ylim(8e-4, 12.0)

    ax.axvspan(-0.6, SPARSE_CUT, facecolor=BAND_FC, lw=0, zorder=0)
    ax.axvline(SPARSE_CUT, color=BAND_EC, lw=0.7, zorder=0.4)
    ax.grid(True, which="major", color="0.88", lw=0.5, zorder=0.2)
    ax.set_axisbelow(True)

    ax.scatter(ns, ts, s=15, marker="o", facecolors=C_LADDER["sparse"],
               edgecolors=C_SPARSE_EDGE, linewidths=0.4, alpha=0.85, zorder=3)
    ax.scatter(nu, tu, s=13, marker="o", facecolors=C_UNIFORM,
               edgecolors="none", alpha=0.45, zorder=2.6)

    xt = [0, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:d}" for t in xt])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xlabel("active image sites (square-root axis)", fontsize=FS_LABEL,
                  labelpad=3)
    ax.set_ylabel("targeted balance (nats)", fontsize=FS_LABEL, labelpad=3)
    ax.tick_params(labelsize=FS_TICK, length=2.2, pad=2.0)

    # Band label sits at the foot of the band, which no solution reaches.
    ax.text(SPARSE_CUT * 0.5, 0.018, f"$\\leq${SPARSE_CUT} sites",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=FS_ANN, color="0.42", zorder=4)

    # The two floors, each written next to the point that sets it.
    ax.annotate(f"{ts.min():.5f}", xy=(ns[ts.argmin()], ts.min()),
                xytext=(7, -1), textcoords="offset points", ha="left",
                va="center", fontsize=FS_ANN, color=C_SPARSE_EDGE,
                path_effects=HALO, zorder=4)
    ax.annotate(f"{tu.min():.4f}", xy=(N_IMG_SHARK, tu.min()),
                xytext=(-5, -5), textcoords="offset points", ha="right",
                va="top", fontsize=FS_ANN, color=C_UNIFORM,
                path_effects=HALO, zorder=4)
    ax.annotate(f"all {N_IMG_SHARK} sites", xy=(N_IMG_SHARK, 0.75),
                xytext=(-5, 0), textcoords="offset points", ha="right",
                va="center", fontsize=FS_ANN, color=C_UNIFORM,
                path_effects=HALO, zorder=4)

    handles = [
        (Line2D([], [], ls="none", marker="o", ms=4.0, mfc=C_UNIFORM,
                mec="none", alpha=0.55),
         f"uniform ({len(nu)})"),
        (Line2D([], [], ls="none", marker="o", ms=4.0,
                mfc=C_LADDER["sparse"], mec=C_SPARSE_EDGE, mew=0.4),
         f"sparsity prior ({len(ns)})"),
    ]
    ax.legend([h for h, _ in handles], [t for _, t in handles],
              loc="upper left", bbox_to_anchor=(0.010, 0.995), ncol=2,
              frameon=True, framealpha=0.95, edgecolor="0.80", fancybox=False,
              fontsize=FS_ANN, handlelength=1.1, handletextpad=0.5,
              columnspacing=1.1, borderpad=0.45
              ).get_frame().set_linewidth(0.5)


def panel_b(ax, traces) -> None:
    ax.set_xscale("function", functions=SQRT1)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 306)
    ax.set_ylim(1.74, 3.95)

    ax.grid(True, which="major", color="0.88", lw=0.5, zorder=0.2)
    ax.set_axisbelow(True)
    ax.axhline(REACH, color="0.50", lw=0.7, ls=(0, (4, 2.4)), zorder=0.9)

    for k, label, _, _, _ in LADDER:
        c = traces[k]
        ax.plot(c.generation, c.pareto_min_TgtBal, color=C_LADDER[k], lw=1.3,
                ls=DASH[k], solid_joinstyle="round", zorder=3)

    # Arrival at 2.5 nats: a marker on the curve, and the generation in that
    # curve's legend entry.  Two of the three crossings are 9 generations
    # apart and all three sit in the busiest part of the panel, so an in-plot
    # number would have to be pushed so far from its marker that it stops
    # naming it.
    reach_gen = {}
    for k, _, _, _, _ in LADDER:
        c = traces[k]
        g = int(c.generation[c.pareto_min_TgtBal <= REACH].min())
        reach_gen[k] = g
        ax.plot([g], [REACH], marker="o", ms=3.4, mfc=C_LADDER[k],
                mec="white", mew=0.6, ls="none", zorder=4)

    # Floors, written under the right end of each curve (above would put the
    # topmost one on the 2.5 line).
    for k, _, _, _, _ in LADDER:
        f = float(traces[k].pareto_min_TgtBal.iloc[-1])
        ax.annotate(f"{f:.4f}", xy=(299, f), xytext=(-2, -4),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=FS_ANN, color=C_LADDER[k], path_effects=HALO,
                    zorder=4)

    xt = [0, 10, 25, 50, 100, 200, 300]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:d}" for t in xt])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([1.9, 2.0, REACH, 3.0, 3.5])
    ax.set_yticklabels(["1.9", "2.0", "2.5", "3.0", "3.5"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("generation (square-root axis)", fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel("best targeted balance so far (nats)", fontsize=FS_LABEL,
                  labelpad=3)
    ax.tick_params(labelsize=FS_TICK, length=2.2, pad=2.0)

    # Legend into the empty upper right: every curve is monotone downward, so
    # that corner is clear by construction.
    handles = [(Line2D([], [], color=C_LADDER[k], lw=1.3, ls=DASH[k]),
                f"{label} · gen {reach_gen[k]}")
               for k, label, _, _, _ in LADDER]
    handles.append((Line2D([], [], color="0.50", lw=0.7, ls=(0, (4, 2.4))),
                    f"{REACH} nats; gen = first crossing"))
    ax.legend([h for h, _ in handles], [t for _, t in handles],
              loc="upper right", bbox_to_anchor=(0.990, 0.990), frameon=True,
              framealpha=0.95, edgecolor="0.80", fancybox=False,
              fontsize=FS_ANN, handlelength=1.9, handletextpad=0.55,
              labelspacing=0.36, borderpad=0.45
              ).get_frame().set_linewidth(0.5)


# ---------------------------------------------------------------------------
def main() -> None:
    setup()
    nu, tu, ns, ts = check_panel_a()
    traces = check_panel_b()

    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes(rect(X0_A, BOTTOM, AXW, AXH, W=W, H=H))
    ax_b = fig.add_axes(rect(X0_B, BOTTOM, AXW, AXH, W=W, H=H))
    panel_a(ax_a, nu, tu, ns, ts)
    panel_b(ax_b, traces)

    y_name = (BOTTOM + AXH + 0.215) / H
    y_ident = (BOTTOM + AXH + 0.055) / H
    titles = (
        (X0_A, "(a)", "uniform vs. sparsity-prior init",
         "great white vs. tiger shark, seed 5 · 200 gen × 30 pop"),
        (X0_B, "(b)", "initialization-sampler ladder",
         "junco vs. chickadee, seed 83 · 300 gen × 30 pop"),
    )
    for x0, tag, name, ident in titles:
        fig.text(x0 / W, y_name, tag, fontsize=FS_LABEL, fontweight="bold",
                 ha="left", va="bottom")
        fig.text((x0 + 0.24) / W, y_name, name, fontsize=FS_LABEL,
                 ha="left", va="bottom", color="0.20")
        fig.text(x0 / W, y_ident, ident, fontsize=FS_ANN, ha="left",
                 va="bottom", color="0.42")

    save(fig, "exp-init-coverage")

    # --- ground truth, printed so the caption never has to be guessed --------
    print("\n[a] shark pair, seed 5, 200 gen x 30 pop, codebook 16383, "
          "identical operators")
    for tag, n, t in (("uniform (Exp-09)", nu, tu),
                      ("sparse   (Exp-10)", ns, ts)):
        print(f"  {tag}: {len(n)} front solutions   n_active "
              f"[{n.min()}, {n.max()}] median {int(np.median(n))}   "
              f"<= {SPARSE_CUT} sites: {int((n <= SPARSE_CUT).sum())}/{len(n)}   "
              f"TgtBal [{t.min():.6f}, {t.max():.4f}]")
    print(f"  floor ratio uniform/sparse = {tu.min() / ts.min():.1f}x   "
          f"sparse solutions below the uniform floor: "
          f"{int((ts < tu.min()).sum())}/{len(ts)}")

    print("\n[b] junco vs chickadee, seed 83, 300 gen x 30 pop, "
          "full_stack text profile, sampler is the only difference")
    for k, label, mode, _, _ in LADDER:
        c = traces[k]
        g25 = int(c.generation[c.pareto_min_TgtBal <= REACH].min())
        print(f"  {label:<22s} mode={mode:<20s} gen0="
              f"{c.pareto_min_TgtBal.iloc[0]:.4f}  floor="
              f"{c.pareto_min_TgtBal.iloc[-1]:.4f}  first <= {REACH}: gen {g25}")


if __name__ == "__main__":
    main()
