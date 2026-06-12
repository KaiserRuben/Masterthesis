"""Exp-100 boundary-geometry atlas — 17 figures, one visual language.

Renders the 809k-evaluation cartography survey of the junco decision
boundary (store: ``experiments/analysis/output/cartography/exp100``) into
a coherent atlas. Distilled from the explore_maps/m1–m5 exploration round.

Two chapters:

:mod:`~analysis.viz.exp100.maps` — where the boundary runs
    01 wall_shape             an easy crossing has a boundary line; a wall
                              is a plateau that never descends
    02 region_map             junco island, boa sea — class territory with
                              surveyed border stakes
    03 margin_relief          the junco trench: margin terrain + sea level
    04 compass                boundary radius by direction of perturbation
    05 output_space           in the model's own coordinates the boundary
                              is a straight line
    06 conquest               earliest visit / earliest crossing per bin
    07 field_over_time        the boundary doesn't move — the search's
                              picture of it does
    08 descent                descent profiles + boundary-touch survival
    09 walk_flow              shrink walks: recrossing vs one-sided

:mod:`~analysis.viz.exp100.fabric` — what it is made of
    10 wall_taxonomy          two wall species, one per prompt regime
    11 anatomy                boundary stakes by gene and modality
    12 sharpness              crossings are slopes, not cliffs
    13 folding                roughness tracks attractor subdominance
    14 junco_slab             the junco region is a slab, not a ball
    15 escape                 escape vs active text genes
    16 flipreach_hardness     regime decoupling scatter
    17 projection_benchmark   which axes make the boundary crisp (methods)

Support: :mod:`~analysis.viz.exp100.language` (shared visual language),
:mod:`~analysis.viz.exp100.data` (store access + cell vocabulary),
:mod:`~analysis.viz.exp100.grids` (binned-field toolkit),
:mod:`~analysis.viz.exp100.registry` (the ``@figure`` ledger).

Rendering (figures land in the Obsidian diary asset dir, ``exp100/``)::

    conda run -n uni python -m analysis.viz.exp100              # everything
    conda run -n uni python -m analysis.viz.exp100 compass 11   # a subset
"""

import matplotlib

matplotlib.use("Agg")

from .registry import FIGURES, FigureSpec, figure  # noqa: E402
from . import maps  # noqa: E402,F401  — importing a chapter registers its figures
from . import fabric  # noqa: E402,F401

__all__ = ["FIGURES", "FigureSpec", "figure"]
