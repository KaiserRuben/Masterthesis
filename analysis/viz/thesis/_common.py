"""Shared setup for the thesis results figures (Master Thesis v0.5.0).

Every figure script in this package imports ``setup()`` from here, so all four
figures share one rcParam state, one palette and one save path.

Sizing contract
---------------
The thesis body width is 483.7pt = 6.69in and every figure is included at
``scale 1.0``.  Canvas inches are therefore *rendered* inches and font sizes in
points are *rendered* points: nothing here may go below 7pt.  Figures are
either full width (``W_FULL``) or half width (``W_HALF``).

Layout is manual throughout (``fig.add_axes`` with figure-fraction rectangles
derived from inch budgets via :func:`rect`).  No ``tight_layout`` and no
``bbox_inches="tight"``: both silently rescale the canvas, which would break
the "canvas inches == rendered inches" contract above.

Outputs go to ``$THESIS_DIR/figures/results/`` as ``<slug>.pdf`` (600 dpi, the
file LaTeX includes) plus ``<slug>.png`` (150 dpi preview).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Set THESIS_DIR to the thesis checkout to emit into it; otherwise figures land
# inside the repository. See docs/REPRODUCTION.md.
THESIS = Path(os.environ.get("THESIS_DIR", REPO / "analysis" / "outputs" / "thesis"))
OUT = THESIS / "figures/results"

# --- data sources ----------------------------------------------------------
AGG_PARQUET = REPO / "experiments/analysis/output/exp100_poc_aggregate.parquet"
RUN_DIR = REPO / "runs/Exp-100/poc_boundary_pair"
EXP101_PER_SEED = REPO / "experiments/analysis/output/exp101/exp101_per_seed.csv"

# --- sizing ----------------------------------------------------------------
W_FULL = 6.69   # 483.7pt thesis text width
W_HALF = 3.30

FS_LABEL = 8.5  # axis labels
FS_TICK = 7.5   # tick labels
FS_ANN = 7.0    # in-plot annotations (hard floor)

# --- palette ---------------------------------------------------------------
# Project answer coding, carried over from the SRC-ASE2026 boundary maps:
# blue = the answer the unmodified input gets, vermillion = the flip.
BLUE = "#2274A5"
RED = "#D64933"
GREY = "#5A5A5A"

# Sequential scale for "how hard is this cell": magma_r is perceptually uniform
# and colour-vision-deficiency safe.  The ends are trimmed so that the lightest
# cell is still distinguishable from the white page and the darkest still
# admits white text.
HARD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hardness", plt.get_cmap("magma_r")(np.linspace(0.04, 0.90, 256)), N=256)

# No-data texture.  A grey fill would sit at the same lightness as the middle
# of the sequential scale; a hatch cannot be misread as a value.
HATCH = "///"


def setup() -> None:
    """rcParams for every thesis figure.  Call once per script."""
    from analysis.core.style import apply_style

    apply_style()
    # AFTER apply_style, which resets rcParams: matplotlib's PDF default is
    # Type 3, which no thesis/print pipeline should be handed.  42 = TrueType.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANN,
        "font.size": FS_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "hatch.color": "0.60",
        "hatch.linewidth": 0.5,
    })


def rect(x, y, w, h, *, W, H):
    """Inch rectangle -> figure-fraction rectangle for ``fig.add_axes``."""
    return [x / W, y / H, w / W, h / H]


def save(fig, slug: str) -> None:
    """Write ``<slug>.pdf`` (600 dpi) and ``<slug>.png`` (150 dpi preview)."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{slug}.pdf", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{slug}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / (slug + '.pdf')}")
    print(f"wrote {OUT / (slug + '.png')}")


def sci(v: float) -> str:
    """``0.000389`` -> ``4e-4`` — one significant digit, compact exponent."""
    s = f"{v:.0e}"                      # 4e-04
    m, e = s.split("e")
    return f"{m}e{int(e)}"


def wrap_word(w: str) -> str:
    """Break a two-word prompt label onto two lines so tick labels stay narrow."""
    return w.replace(" ", "\n")
