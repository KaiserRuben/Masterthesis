"""The atlas's shared visual language.

Every figure speaks the same dialect, so a reader can move between maps
without relearning the legend:

claim
    The suptitle states the geometric finding in plain words.
method
    A grey line beneath it carries the jargon: prompt regime, data source,
    normalization, masking rules. The claim line stays clean.
cells
    Label pairs are named by their prompt words ('sparrow' vs 'snake') and
    tagged WALL (dark red) or EASY (dark blue).

Numbers and wordings that all figures must agree on — the repeat-noise
floor, the class palette, the axis phrasings — live here and nowhere else.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

# Fixed hue per ImageNet class, used wherever a prediction or a search
# target is colored. Chosen against analysis.core.style.PIPELINE.
CLASS_COLORS = {
    "junco": "#937860",
    "ostrich": "#E6A817",
    "green iguana": "#55A868",
    "boa constrictor": "#C44E52",
    "cello": "#4C72B0",
    "marimba": "#CCB974",
}

# The two boundary characters of Exp-100.
WALL_COLOR = "#8C2D04"   # dark red — a cell the search never crossed
EASY_COLOR = "#1A5E8A"   # dark blue — a cell with a reachable boundary

# Repeating an evaluation moves the pair margin by up to 0.38 lp (q90).
# Any claim about a crossing must clear this band.
NOISE_LP = 0.38                         # in log-prob units
NOISE_G = float(np.tanh(NOISE_LP / 2))  # the same band in g units ≈ 0.188


class AXIS:
    """One phrasing per plotted quantity, atlas-wide."""

    img_sem = "image distance from seed  (per-seed q99 norm.)"
    txt_sem = "text distance from seed  (per-seed q99 norm.)"
    img_strength = "image manipulation strength  (rank-sum, norm.)"
    txt_strength = "text manipulation strength  (rank-sum, norm.)"
    g = "median  g = P(anchor word) − P(target word)"


def header(fig: Figure, claim: str, method: str, *,
           claim_y: float = 0.99, method_y: float = 0.95) -> None:
    """Set the figure's claim suptitle and the grey method line below it."""
    fig.suptitle(claim, fontsize=14.5, fontweight="bold", y=claim_y)
    fig.text(0.5, method_y, method, ha="center", va="top",
             fontsize=9.5, color="0.38", linespacing=1.5)
