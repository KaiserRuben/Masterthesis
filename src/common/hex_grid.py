"""Hex-colour grid construction + tokenizer validation.

A *hex grid* is an evenly spaced arc of HSV hues rendered as uppercase
``#RRGGBB`` strings.  It is used as a dense, ordered contrast set for
slot scans (Exp-105): the slot filler walks a perceptual continuum
while the carrier sentence stays byte-identical, so the shared prefix
cancels exactly in the length-normalised log-prob.

That cancellation argument only holds if every code costs the SUT the
*same number of tokens*.  Constant string length does **not** guarantee
a constant BPE token count — ``#00FF00`` and ``#00FF20`` can split
differently depending on which byte pairs the merge table happens to
contain.  :func:`validate_token_counts` / :func:`filter_equal_token_count`
make that check mandatory before a hex scan runs; the caller logs what
was dropped.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from typing import Any, Iterable

__all__ = [
    "build_hex_grid",
    "validate_token_counts",
    "filter_equal_token_count",
]


def build_hex_grid(
    hue_start: float = 120.0,
    hue_end: float = 240.0,
    steps: int = 17,
    s: float = 1.0,
    v: float = 1.0,
) -> list[str]:
    """Build an evenly spaced HSV hue arc as uppercase hex codes.

    The arc is closed on both ends: ``codes[0]`` is *hue_start* and
    ``codes[-1]`` is *hue_end*.  With the defaults (green→blue, full
    saturation and value) the endpoints round-trip exactly to
    ``"#00FF00"`` and ``"#0000FF"``.

    :param hue_start: First hue in degrees.
    :param hue_end: Last hue in degrees (inclusive).
    :param steps: Number of grid points (``>= 2``).
    :param s: HSV saturation for every point, in ``[0, 1]``.
    :param v: HSV value for every point, in ``[0, 1]``.
    :returns: ``steps`` uppercase ``"#RRGGBB"`` strings, ordered by hue.
    :raises ValueError: If *steps* < 2 or *s* / *v* are out of range.
    """
    if steps < 2:
        raise ValueError(f"hex grid needs at least 2 steps; got {steps}")
    if not (0.0 <= s <= 1.0) or not (0.0 <= v <= 1.0):
        raise ValueError(
            f"hex grid s/v must lie in [0, 1]; got s={s}, v={v}"
        )

    span = hue_end - hue_start
    codes: list[str] = []
    for i in range(steps):
        hue = hue_start + span * i / (steps - 1)
        r, g, b = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, s, v)
        codes.append(
            "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))
        )
    return codes


def validate_token_counts(
    codes: Iterable[str],
    tokenizer: Any,
) -> dict[str, int]:
    """Tokenize every code and report its token count.

    :param codes: Hex codes (or any strings) to measure.
    :param tokenizer: Object exposing HuggingFace's
        ``encode(text, add_special_tokens=False) -> list[int]``.
    :returns: Mapping ``code -> n_tokens`` in input order.
    """
    return {
        code: len(tokenizer.encode(code, add_special_tokens=False))
        for code in codes
    }


def filter_equal_token_count(
    codes: Iterable[str],
    tokenizer: Any,
) -> tuple[list[str], int]:
    """Keep only the codes whose token count equals the modal count.

    Ties between equally frequent counts are resolved toward the count
    seen first (``collections.Counter`` preserves insertion order), which
    makes the result deterministic for a given code ordering.

    :param codes: Hex codes to filter.
    :param tokenizer: Object exposing
        ``encode(text, add_special_tokens=False)``.
    :returns: ``(kept_codes, modal_token_count)``.  The caller is
        expected to log the dropped complement.
    :raises ValueError: If *codes* is empty.
    """
    counts = validate_token_counts(codes, tokenizer)
    if not counts:
        raise ValueError("filter_equal_token_count got an empty code list")
    modal, _ = Counter(counts.values()).most_common(1)[0]
    kept = [code for code, n in counts.items() if n == modal]
    return kept, modal
