r"""Emitter for the Chapter 6 results figure package (LaTeX-native pgfplots).

Every figure in ``figures/results/`` is a ``.tex`` file that the thesis
``\input``s inside a float.  This module writes them.  The data behind each
figure is loaded by the matplotlib scripts that sit next to this package --
those scripts carry the ground-truth assertions and stay the single source of
the numbers; only the renderer changed.

Size grid
---------
The thesis text width is 483.70pt = 17.00cm and text height 702.78pt.  Every
figure spans the full text width and takes one of three heights:

    T1  17.0 x 5.5cm   single panel
    T2  17.0 x 7.0cm   panel row (2-3 panels)
    T3  17.0 x 9.5cm   field map or dense grid

``width``/``height`` in a pgfplots axis are nominal: pgfplots reserves room
for tick labels and axis labels inside them, but nodes placed with
``clip=false`` outside the axis box are not counted.  :func:`measure` builds
the emitted figure on its own and reports the true typeset box, which is how
each figure was fitted to the grid.

Design
------
``figures/results/results-style.tex`` extends the HS-01 package's style file,
so the two families share one grayscale ramp, one marker set and one axis
appearance.  Colour appears only on the three field maps, where the sign of
the decision margin is the content.

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python -m analysis.viz.thesis.pgf            # all figures
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python -m analysis.viz.thesis.pgf budget     # one figure
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
THESIS = Path("/Users/kaiser/Desktop/Uni/Masterarbeit/Master Thesis v0.6.0")
OUT = THESIS / "figures/results"

# --- size grid (cm) --------------------------------------------------------
TW = 17.00      # \textwidth = 483.70pt
TH = 24.70      # \textheight = 702.78pt
T1, T2, T3 = 5.5, 7.0, 9.5

TIER_NAME = {T1: "T1", T2: "T2", T3: "T3"}


# --- number formatting -----------------------------------------------------

def n(x: float, k: int = 4) -> str:
    """Shortest fixed-point form of ``x`` at ``k`` decimals, no trailing zeros."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        raise ValueError(f"non-finite coordinate: {x}")
    s = f"{x:.{k}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def coords(xs, ys, k: int = 4) -> str:
    """``(x,y) (x,y) ...`` for an \\addplot coordinates list."""
    return " ".join(f"({n(x, k)},{n(y, k)})" for x, y in zip(xs, ys))


def table(rows, k: int = 4) -> str:
    """Inline ``table {...}`` body: one whitespace-separated row per line."""
    body = "\n".join(" ".join(n(v, k) for v in r) for r in rows)
    return body


def tex(s: str) -> str:
    """Escape a data string for LaTeX text (labels, prompt words, class names)."""
    out = (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
             .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
             .replace("}", r"\}").replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))
    return out.replace("->", r"$\to$").replace("--", "--")


def sci(v: float) -> str:
    """``0.000389`` -> ``4e-4``: one significant digit, compact exponent."""
    m, e = f"{v:.0e}".split("e")
    return f"{m}e{int(e)}"


def mathsci(v: float) -> str:
    r"""``0.000389`` -> ``$4{\cdot}10^{-4}$`` for axis and legend text."""
    m, e = f"{v:.0e}".split("e")
    return rf"$10^{{{int(e)}}}$" if m == "1" else rf"${m}{{\cdot}}10^{{{int(e)}}}$"


# --- cache -----------------------------------------------------------------
# Some figures walk 119 run directories to reconstruct their curves.  The data
# functions stay authoritative; this only keeps their result on disk so that
# re-rendering a figure after a layout change costs seconds instead of minutes.
# Delete .cache/ (or pass fresh=True) whenever the underlying runs change.

CACHE = Path(__file__).resolve().parent / ".cache"


def cached(key: str, fn, *, fresh: bool = False):
    import pickle
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{key}.pkl"
    if f.exists() and not fresh:
        return pickle.loads(f.read_bytes())
    val = fn()
    f.write_bytes(pickle.dumps(val))
    return val


# --- figure writer ---------------------------------------------------------

class Fig:
    """Accumulates the body of one ``figures/results/<slug>.tex``."""

    def __init__(self, slug: str, what: str, source: str, tier: float,
                 generator: str) -> None:
        self.slug = slug
        self.what = what
        self.source = source
        self.tier = tier
        self.generator = generator
        self.lines: list[str] = []

    def __call__(self, *lines: str) -> "Fig":
        self.lines.extend(lines)
        return self

    # -- body -------------------------------------------------------------
    def head(self) -> str:
        src = "\n".join(f"%%   {ln}" for ln in self.source.strip().splitlines())
        return (
            "%% ---------------------------------------------------------------\n"
            f"%% {self.slug}.tex\n"
            f"%% {self.what}\n"
            "%% Source:\n"
            f"{src}\n"
            f"%% Generated by analysis/viz/thesis/pgf/{self.generator} "
            "-- do not edit by hand.\n"
            f"%% Size grid: {TIER_NAME[self.tier]}  {n(TW, 2)} x {n(self.tier, 2)}cm "
            "(see figures/results/results-style.tex).\n"
            "%% Preamble must contain (once):\n"
            "%%   \\usetikzlibrary{patterns,patterns.meta}\n"
            "%%   \\usepgfplotslibrary{groupplots,statistics,colormaps}\n"
            "%% Use inside a figure environment:  "
            f"\\input{{figures/results/{self.slug}}}\n"
            "%% ---------------------------------------------------------------\n"
            "\\input{figures/results/results-style}%\n"
        )

    def text(self) -> str:
        return self.head() + "\n".join(self.lines) + "\n"

    def save(self, *, check: bool = True, quiet: bool = False) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{self.slug}.tex"
        path.write_text(self.text(), encoding="utf-8")
        if quiet:
            return path
        msg = f"wrote {path}  ({len(self.lines)} lines)"
        if check:
            w, h = measure(self.slug)
            fit = "" if abs(w - TW) < 0.35 and h <= self.tier + 0.35 else "  <-- OFF GRID"
            msg += (f"  typeset {n(w, 2)} x {n(h, 2)}cm  "
                    f"[{TIER_NAME[self.tier]} {n(TW, 2)} x {n(self.tier, 2)}]{fit}")
        print(msg)
        return path


# --- build check -----------------------------------------------------------
#
# The measurement document does NOT use tumbook: the class hangs pdflatex when
# it is loaded outside the thesis (verified 2026-08-25 -- a bare
# `\documentclass[a4paper,thesis=student,...]{tumbook}` + "Hello" spins at 90%
# CPU after the font defaults check, with or without coverpage/titlepage).
# scrbook at the same 11pt base with helvet scaled and the thesis geometry
# reproduces the body font exactly (both resolve to T1/phv/m/n at 10.95pt) and
# the same \textwidth/\textheight, so figure boxes measure identically.

MEASURE_DOC = r"""\documentclass[a4paper,11pt,english,oneside]{scrbook}
\usepackage[T1]{fontenc}
\usepackage[scaled]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{geometry}
\geometry{textwidth=483.6969pt, textheight=702.78313pt}
\usepackage{amsmath}\usepackage{amssymb}
\usepackage{graphicx}\usepackage{xcolor}
\usepackage{tikz}\usetikzlibrary{arrows.meta,calc,positioning,fit,matrix}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usetikzlibrary{patterns,patterns.meta}
\usepgfplotslibrary{groupplots,statistics,colormaps}
\pagestyle{empty}
\begin{document}
\newsavebox{\resbox}
\sbox{\resbox}{\input{figures/results/SLUG}}
\typeout{RESMEASURE width=\the\wd\resbox\space height=\the\dimexpr\ht\resbox+\dp\resbox\relax}
\usebox{\resbox}
\end{document}
"""


def measure(slug: str) -> tuple[float, float]:
    """Typeset ``<slug>.tex`` alone and return its true (width, height) in cm.

    Raises with the LaTeX error context if the figure does not compile, which
    is the point: no figure is written to the thesis without a build behind it.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "m.tex").write_text(MEASURE_DOC.replace("SLUG", slug), encoding="utf-8")
        # run from the thesis root so every relative \input and \includegraphics
        # in the figure resolves exactly as it does in the real build; keep the
        # aux files in the temp dir so nothing lands next to main.tex
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={d}", str(d / "m.tex")],
            cwd=THESIS, capture_output=True, text=True,
            env=dict(os.environ), timeout=600)
        log = (d / "m.log").read_text(encoding="utf-8", errors="replace") \
            if (d / "m.log").exists() else r.stdout
        if r.returncode != 0:
            err = "\n".join(ln for ln in log.splitlines()
                            if ln.startswith("!") or ln.startswith("l."))
            raise RuntimeError(f"{slug}: LaTeX build failed\n{err[:2000]}")
        m = re.search(r"RESMEASURE width=([\d.]+)pt height=([\d.]+)pt", log)
        if not m:
            raise RuntimeError(f"{slug}: no measurement in log")
        over = [ln for ln in log.splitlines() if "Overfull" in ln]
        if over:
            print(f"  {slug}: {len(over)} overfull box(es) in the isolated build")
        pt_cm = 28.45274
        return float(m.group(1)) / pt_cm, float(m.group(2)) / pt_cm


def fit(slug: str, make, tier: float, *, w0: float | None = None,
        h0: float | None = None, iters: int = 5, tol: float = 0.04) -> "Fig":
    """Size a figure onto the grid by building it and correcting the residual.

    ``make(axw, axh)`` returns the :class:`Fig` for a given *axis area* -- the
    figure sets ``scale only axis=true``, so ``axw``/``axh`` describe the plot
    rectangle and the tick labels, axis labels and outside nodes add to it.
    The mapping from axis area to typeset box is affine, so correcting by the
    measured residual converges in two or three builds.  Every iteration is a
    real pdflatex run of the emitted file: a figure that does not compile never
    reaches the thesis.
    """
    w = TW - 2.2 if w0 is None else w0
    h = tier - 1.6 if h0 is None else h0
    W = H = float("nan")
    for i in range(iters):
        fig = make(w, h)
        fig.save(check=False, quiet=True)
        W, H = measure(slug)
        dw, dh = TW - W, tier - H
        if abs(dw) <= tol and abs(dh) <= tol:
            break
        w, h = w + dw, h + dh
        if w <= 0.5 or h <= 0.5:
            raise RuntimeError(f"{slug}: axis area collapsed at iteration {i}")
    off = "" if abs(TW - W) <= 0.12 and abs(tier - H) <= 0.12 else "  <-- OFF GRID"
    print(f"wrote {OUT / (slug + '.tex')}  ({len(fig.lines)} lines)  "
          f"axis {n(w, 3)} x {n(h, 3)}cm  ->  typeset {n(W, 2)} x {n(H, 2)}cm  "
          f"[{TIER_NAME[tier]} {n(TW, 2)} x {n(tier, 2)}]{off}")
    return fig


def raster(fig, slug: str, dpi: int = 600) -> Path:
    """Save a bare matplotlib field (no axes, no margins) for \\addplot graphics."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{slug}.png"
    fig.savefig(path, dpi=dpi, facecolor="white", pad_inches=0,
                bbox_inches="tight")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":  # pragma: no cover
    print(f"text width {TW}cm, tiers {T1}/{T2}/{T3}cm, out {OUT}", file=sys.stderr)
