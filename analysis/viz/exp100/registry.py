"""Figure registry — one decorator, one ledger, one way to render.

Every atlas figure declares its place exactly once::

    @figure(4, "compass")
    def compass() -> Figure: ...

The output filename (``exp100_atlas_04_compass.png``), the save settings,
and the CLI lookup all derive from that declaration, so a figure can never
drift out of step with the atlas index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure

from analysis.core.style import save_fig

Builder = Callable[[], Figure]

FIGURES: dict[str, "FigureSpec"] = {}


@dataclass(frozen=True)
class FigureSpec:
    """One registered atlas figure: its index entry and how to build it."""

    number: int
    slug: str
    build: Builder

    @property
    def filename(self) -> str:
        return f"exp100_atlas_{self.number:02d}_{self.slug}.png"

    def render(self, out: Path) -> Path:
        """Build the figure and write it to ``out`` under its atlas name.

        Atlas figures position their headers and legends with explicit
        ``subplots_adjust`` calls, so they are saved without tight_layout.
        """
        return save_fig(self.build(), out / self.filename, tight=False)


def figure(number: int, slug: str) -> Callable[[Builder], Builder]:
    """Register a builder as atlas figure ``number`` under ``slug``."""

    def register(build: Builder) -> Builder:
        if slug in FIGURES or number in {s.number for s in FIGURES.values()}:
            raise ValueError(
                f"figure {number:02d}/{slug!r} collides with an existing entry")
        FIGURES[slug] = FigureSpec(number, slug, build)
        return build

    return register
