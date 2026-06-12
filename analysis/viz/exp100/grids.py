"""Binned 2D fields — the projection toolkit behind every map.

A :class:`Field` is a rectangular survey grid over two scatter coordinates
with one aggregated value per bin and an honest record of how many
evaluations support it. Bins below ``min_n`` are NaN: unsampled territory
renders blank, never interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb

Extent = tuple[float, float, float, float]


@dataclass(frozen=True)
class Field:
    """One aggregated value per bin over a regular 2D grid."""

    values: np.ndarray   # (nx, ny), indexed [ix, iy]; NaN = unsampled
    counts: np.ndarray   # evaluations per bin (within-extent only)
    xe: np.ndarray       # bin edges
    ye: np.ndarray

    @property
    def xc(self) -> np.ndarray:
        return 0.5 * (self.xe[:-1] + self.xe[1:])

    @property
    def yc(self) -> np.ndarray:
        return 0.5 * (self.ye[:-1] + self.ye[1:])

    @property
    def img(self) -> np.ndarray:
        """(ny, nx) view for pcolormesh / imshow / contour."""
        return self.values.T

    @property
    def spans_zero(self) -> bool:
        """True if the field has values on both sides of zero — i.e. the
        boundary runs through surveyed territory."""
        finite = self.values[np.isfinite(self.values)]
        return finite.size > 0 and finite.min() < 0 < finite.max()

    def boundary(self, ax: Axes, *, color: str = "black",
                 lw: float = 1.6) -> None:
        """Draw the zero contour, if the field crosses zero at all.

        Walls don't: their fields have no zero level in surveyed
        territory, so nothing is drawn — absence of the line is the
        finding, not a rendering gap.
        """
        if self.spans_zero:
            ax.contour(self.xc, self.yc, self.img, levels=[0.0],
                       colors=color, linewidths=lw)


def field(x, y, values, *, nbins: int, extent: Extent,
          stat: str = "median", min_n: int = 0) -> Field:
    """Aggregate ``values`` over a regular (x, y) grid.

    ``counts`` come from a strict histogram over the extent, while values
    beyond the extent are clipped into the edge bins — so an edge bin is
    only considered "mapped" (vs ``min_n``) by in-extent evaluations.
    """
    x, y, v = (np.asarray(a, dtype=float) for a in (x, y, values))
    xe = np.linspace(extent[0], extent[1], nbins + 1)
    ye = np.linspace(extent[2], extent[3], nbins + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    ix = np.clip(np.digitize(x, xe) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, nbins - 1)
    grid = np.full((nbins, nbins), np.nan)
    for (i, j), val in pd.Series(v).groupby([ix, iy]).agg(stat).items():
        grid[i, j] = val
    if min_n:
        grid[counts < min_n] = np.nan
    return Field(grid, counts, xe, ye)


def majority_rgba(x, y, labels, *, xe: np.ndarray, ye: np.ndarray,
                  palette: dict[str, str], min_n: int = 4,
                  alpha_floor: float = 0.25,
                  alpha_min: float = 0.15) -> np.ndarray:
    """Per-bin majority-class image (ny, nx, 4): hue = the winning class,
    opacity = its share. Bins under ``min_n`` stay fully transparent."""
    nx, ny = len(xe) - 1, len(ye) - 1
    img = np.zeros((ny, nx, 4))
    ix = np.clip(np.digitize(x, xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, ny - 1)
    bins = pd.DataFrame({"ix": ix, "iy": iy, "lbl": np.asarray(labels)})
    for (bx, by), sub in bins.groupby(["ix", "iy"]):
        if len(sub) < min_n:
            continue
        votes = sub.lbl.value_counts()
        share = votes.iloc[0] / len(sub)
        r, g, b = to_rgb(palette[votes.index[0]])
        a = alpha_floor + (1 - alpha_floor) * (share - 1 / 3) / (2 / 3)
        img[by, bx] = (r, g, b, np.clip(a, alpha_min, 1.0))
    return img
