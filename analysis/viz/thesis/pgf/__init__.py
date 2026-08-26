"""LaTeX-native (pgfplots) renderers for the Chapter 6 results figures.

The matplotlib scripts in the parent directory stay the source of the data --
they carry the ground-truth assertions -- and each module here imports their
data functions and emits a ``.tex`` figure instead of a PDF.  See
``emit.py`` for the size grid and the writer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_THESIS_VIZ = _HERE.parent                     # analysis/viz/thesis
_REPO = _THESIS_VIZ.parents[2]                 # repo root

for _p in (str(_THESIS_VIZ), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["emit"]
