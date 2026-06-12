"""Command-line renderer for the atlas.

Run from the repo root::

    conda run -n uni python -m analysis.viz.exp100              # everything
    conda run -n uni python -m analysis.viz.exp100 compass 11   # by slug or number
"""

from __future__ import annotations

import sys

from analysis.core.style import apply_style, asset_dir

from . import FIGURES


def _atlas() -> list:
    return sorted(FIGURES.values(), key=lambda s: s.number)


def _pick(tokens: list[str]) -> list:
    if not tokens:
        return _atlas()
    by_number = {s.number: s for s in FIGURES.values()}
    chosen = []
    for t in tokens:
        spec = FIGURES.get(t) or by_number.get(int(t) if t.isdigit() else -1)
        if spec is None:
            index = "\n".join(f"  {s.number:02d}  {s.slug}" for s in _atlas())
            sys.exit(f"unknown figure {t!r} — the atlas:\n{index}")
        chosen.append(spec)
    return chosen


def main(argv: list[str]) -> None:
    specs = _pick(argv)
    apply_style()
    out = asset_dir("exp100")
    for spec in specs:
        print(f"[{spec.number:02d} {spec.slug}]")
        spec.render(out)


if __name__ == "__main__":
    main(sys.argv[1:])
