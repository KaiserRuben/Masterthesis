"""Tests for the hex-colour grid builder + tokenizer validation.

No model, no network: the token-count checks run against a stub
tokenizer whose per-string token lists are defined inline.
"""

from __future__ import annotations

import colorsys
import re

import pytest

from src.common.hex_grid import (
    build_hex_grid,
    filter_equal_token_count,
    validate_token_counts,
)

_HEX_RE = re.compile(r"^#[0-9A-F]{6}$")


class StubTokenizer:
    """Minimal HF-tokenizer surface: ``encode(text, add_special_tokens=)``.

    Token counts come from *table*; anything unlisted gets *default*
    tokens.  Ids are arbitrary — only the length is ever read.
    """

    def __init__(self, table: dict[str, int] | None = None, default: int = 4):
        self._table = table or {}
        self._default = default
        self.calls: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False, (
            "hex codes must be measured without special tokens"
        )
        self.calls.append(text)
        return list(range(self._table.get(text, self._default)))


# ---------------------------------------------------------------------------
# build_hex_grid
# ---------------------------------------------------------------------------


class TestBuildHexGrid:
    def test_default_length(self) -> None:
        assert len(build_hex_grid()) == 17

    def test_endpoints_exact(self) -> None:
        """The green/blue anchors must round-trip exactly, not approximately."""
        codes = build_hex_grid()
        assert codes[0] == "#00FF00"
        assert codes[-1] == "#0000FF"

    def test_all_uppercase_hex(self) -> None:
        assert all(_HEX_RE.match(c) for c in build_hex_grid())

    def test_hue_is_monotone(self) -> None:
        """Decoding each code back to HSV yields a strictly rising hue."""
        hues = []
        for code in build_hex_grid():
            r, g, b = (int(code[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
            hues.append(colorsys.rgb_to_hsv(r, g, b)[0])
        assert all(a < b for a, b in zip(hues, hues[1:]))
        assert hues[0] == pytest.approx(120.0 / 360.0, abs=1e-3)
        assert hues[-1] == pytest.approx(240.0 / 360.0, abs=1e-3)

    def test_distinct_codes(self) -> None:
        codes = build_hex_grid()
        assert len(set(codes)) == len(codes)

    def test_custom_steps_and_arc(self) -> None:
        codes = build_hex_grid(hue_start=0.0, hue_end=120.0, steps=3)
        assert codes == ["#FF0000", "#FFFF00", "#00FF00"]

    def test_value_scales_channels(self) -> None:
        assert build_hex_grid(steps=2, v=0.5)[0] == "#008000"

    def test_too_few_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 steps"):
            build_hex_grid(steps=1)

    def test_out_of_range_sv_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            build_hex_grid(s=1.5)


# ---------------------------------------------------------------------------
# Token-count validation
# ---------------------------------------------------------------------------


class TestTokenCounts:
    def test_validate_returns_count_per_code(self) -> None:
        tok = StubTokenizer({"#00FF00": 5, "#0000FF": 3}, default=4)
        counts = validate_token_counts(["#00FF00", "#0000FF", "#00FF20"], tok)
        assert counts == {"#00FF00": 5, "#0000FF": 3, "#00FF20": 4}
        assert tok.calls == ["#00FF00", "#0000FF", "#00FF20"]

    def test_filter_keeps_modal_count(self) -> None:
        codes = ["#00FF00", "#00FF20", "#0000FF", "#0080FF"]
        tok = StubTokenizer({"#00FF00": 5, "#0000FF": 6}, default=4)
        kept, modal = filter_equal_token_count(codes, tok)
        assert modal == 4
        assert kept == ["#00FF20", "#0080FF"]

    def test_filter_all_equal_keeps_everything(self) -> None:
        codes = build_hex_grid(steps=5)
        kept, modal = filter_equal_token_count(codes, StubTokenizer(default=4))
        assert kept == codes
        assert modal == 4

    def test_filter_tie_resolves_to_first_seen(self) -> None:
        """Deterministic tiebreak — Counter preserves insertion order."""
        codes = ["#AAAAAA", "#BBBBBB"]
        tok = StubTokenizer({"#AAAAAA": 3, "#BBBBBB": 7})
        kept, modal = filter_equal_token_count(codes, tok)
        assert (kept, modal) == (["#AAAAAA"], 3)

    def test_filter_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty code list"):
            filter_equal_token_count([], StubTokenizer())

    def test_exported_from_src_common(self) -> None:
        from src.common import (
            build_hex_grid as exported_build,
            filter_equal_token_count as exported_filter,
            validate_token_counts as exported_validate,
        )

        assert exported_build is build_hex_grid
        assert exported_filter is filter_equal_token_count
        assert exported_validate is validate_token_counts
