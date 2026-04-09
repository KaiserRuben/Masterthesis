"""Shared visual language for all thesis visualizations.

Ensures SMOO and PDQ plots are visually comparable:
same palette, same axis conventions, same figure sizing.

Usage:
    from analysis.style import apply_style, COLORS, save_fig
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Palette — colorblind-safe, print-friendly
# ---------------------------------------------------------------------------

# Pipeline identity
PIPELINE = {
    "smoo": "#2274A5",   # steel blue
    "pdq": "#D64933",    # vermillion
}

# Per-anchor-class (consistent across pipelines)
ANCHOR = {
    "goldfish": "#E6A817",
    "hammerhead shark": "#4A7C59",
    "brambling": "#C44E52",
    "monarch butterfly": "#8172B3",
    "stingray": "#55A868",
    "fire salamander": "#DD8452",
    "indigo bunting": "#4C72B0",
    "junco": "#937860",
    "default": "#999999",
}

# PDQ strategies
STRATEGY = {
    "dense_uniform": "#4C72B0",
    "bituniform_density": "#55A868",
    "sparsity_sweep": "#C44E52",
    "modality_image": "#8172B3",
    "modality_text": "#CCB974",
    "max_rank": "#64B5CD",
    "sparse_small": "#CCCCCC",
}

# SMOO objectives
OBJECTIVE = {
    "fitness_MatrixDistance_fro": "#4C72B0",
    "fitness_TextDist": "#C44E52",
    "fitness_TgtBal": "#55A868",
    "fitness_Conc": "#8172B3",
    "fitness_ArchiveSparsity": "#CCCCCC",
}

# Short names for display
OBJ_LABELS = {
    "fitness_MatrixDistance_fro": "Image dist",
    "fitness_TextDist": "Text dist",
    "fitness_TgtBal": "Targeted balance",
    "fitness_Conc": "Concentration",
    "fitness_ArchiveSparsity": "Archive sparsity",
}

# Stage 2 passes
PASS = {
    "zero": "#4C72B0",
    "rank": "#C44E52",
    "random_subset": "#55A868",
}


def anchor_color(label: str) -> str:
    return ANCHOR.get(label, ANCHOR["default"])


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply thesis-wide matplotlib defaults.  Call once at script start."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.8,
        "font.family": "sans-serif",
        "font.size": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

ASSET_ROOT = Path.home() / "Obsidian/Notizen/01 - Active Projects/Master Thesis/Diary/assets"


def asset_dir(name: str) -> Path:
    """Return (and create) an asset sub-directory."""
    d = ASSET_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig: Figure, path: Path, *, tight: bool = True) -> Path:
    """Save figure, print confirmation, return path."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path.name}")
    return path


def subplot_label(ax, label: str, x: float = -0.08, y: float = 1.06) -> None:
    """Add (a), (b), ... to a subplot corner."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
