"""Tests for the origin->target double-cone candidate filter.

The pipeline in :mod:`src.manipulator.image.cone_candidates` is purely
functional and operates on NumPy arrays — no VQGAN, no Redis, no
filesystem. Cases mirror the design session's toy 2-D geometry plus the
edge conditions that motivated specific implementation choices.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.manipulator.image.cone_candidates import (
    ConeCandidateFilter,
    axis_vector,
    cone_mask,
    endpoint_cosines,
    filter_and_order,
    order_by_axis,
    projection_tau,
    segment_mask,
)


# ---------------------------------------------------------------------------
# Toy 2-D codebook used across multiple tests
# ---------------------------------------------------------------------------


def _toy_codebook() -> np.ndarray:
    """Codebook from the design session: 7 codewords in R^2.

    Indices and intent:
        0: origin (0, 0)
        1: target (10, 0)
        2: on-axis midpoint (5, 0)
        3: slightly off-axis above mid (5, 1)  — kept by alpha>=12 deg
        4: far off-axis above mid (5, 3)       — kept by alpha>=31 deg only
        5: behind origin (-1, 0)                — beyond segment endpoint
        6: past target (11, 0)                  — beyond segment endpoint
    """
    return np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 0.0],
            [5.0, 1.0],
            [5.0, 3.0],
            [-1.0, 0.0],
            [11.0, 0.0],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Pure pipeline steps
# ---------------------------------------------------------------------------


class TestAxisVector:
    def test_axis_norm_zero_on_degenerate(self) -> None:
        p_c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        axis, norm = axis_vector(p_c, p_c.copy())
        assert norm == pytest.approx(0.0)
        np.testing.assert_array_equal(axis, np.zeros(3))

    def test_axis_norm_non_degenerate(self) -> None:
        p_c = np.array([0.0, 0.0], dtype=np.float32)
        p_t = np.array([3.0, 4.0], dtype=np.float32)
        axis, norm = axis_vector(p_c, p_t)
        assert norm == pytest.approx(5.0)
        np.testing.assert_allclose(axis, [3.0, 4.0])


class TestProjectionTau:
    def test_tau_zero_at_origin_one_at_target(self) -> None:
        cb = _toy_codebook()
        p_c, p_t = cb[0], cb[1]
        axis, norm = axis_vector(p_c, p_t)
        tau = projection_tau(cb, p_c, axis, norm)
        assert tau[0] == pytest.approx(0.0)
        assert tau[1] == pytest.approx(1.0)
        assert tau[2] == pytest.approx(0.5)
        # On-axis behaviour: rows 5 and 6 must lie outside [0, 1].
        assert tau[5] < 0.0
        assert tau[6] > 1.0


class TestSegmentMask:
    def test_endpoints_admitted(self) -> None:
        tau = np.array([-1e-9, 0.0, 0.5, 1.0, 1.0 + 1e-9, 1.1], dtype=np.float64)
        mask = segment_mask(tau)
        # Default tolerance 1e-6 admits both endpoints (including tiny overshoot).
        np.testing.assert_array_equal(
            mask, [True, True, True, True, True, False],
        )

    def test_tolerance_absorbs_float32_roundoff(self) -> None:
        # The tolerance must be large enough to keep a tau computed as
        # exactly 1.0 in float64 — but our concern is the float32 path
        # where roundoff can push tau marginally past 1.
        tau = np.array([1.0 + 1e-7], dtype=np.float64)
        assert segment_mask(tau)[0]


class TestEndpointCosines:
    def test_endpoint_codewords_force_alignment(self) -> None:
        cb = _toy_codebook()
        p_c, p_t = cb[0], cb[1]
        axis, norm = axis_vector(p_c, p_t)
        cos_c, cos_t = endpoint_cosines(cb, p_c, p_t, axis, norm)
        # Codeword 0 == p_c → diff_c is zero → cos_c set to 1.0
        assert cos_c[0] == pytest.approx(1.0)
        # Codeword 1 == p_t → diff_t is zero → cos_t set to 1.0
        assert cos_t[1] == pytest.approx(1.0)
        # On-axis midpoint sees perfect alignment from both endpoints
        assert cos_c[2] == pytest.approx(1.0)
        assert cos_t[2] == pytest.approx(1.0)


class TestConeMask:
    def test_admits_only_within_half_angle(self) -> None:
        cos_c = np.array([1.0, 0.9, 0.5], dtype=np.float64)
        cos_t = np.array([1.0, 0.9, 0.5], dtype=np.float64)
        # alpha = 20 deg → cos_alpha ~= 0.94
        mask = cone_mask(cos_c, cos_t, math.radians(20))
        np.testing.assert_array_equal(mask, [True, False, False])


class TestOrderByAxis:
    def test_ascending_tau_with_distance_tiebreak(self) -> None:
        idx = np.array([0, 1, 2], dtype=np.int64)
        tau = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        # Same tau for 0 and 1 → tiebreak by distance from origin.
        dist = np.array([1.0, 0.1, 0.0], dtype=np.float64)
        ordered = order_by_axis(idx, tau, dist)
        # Smallest tau first (2), then 1 (closer of the tau-tied pair), then 0.
        np.testing.assert_array_equal(ordered, [2, 1, 0])


# ---------------------------------------------------------------------------
# Top-level pipeline (filter_and_order / ConeCandidateFilter facade)
# ---------------------------------------------------------------------------


class TestFilterAndOrderToyGeometry:
    """Reproduces the design-session 2-D scenarios."""

    def setup_method(self) -> None:
        self.cb = _toy_codebook()
        self.p_c = self.cb[0]
        self.p_t = self.cb[1]

    def test_alpha_10_keeps_on_axis_only(self) -> None:
        result = filter_and_order(
            self.p_c, self.p_t, self.cb, math.radians(10),
        )
        # On-axis codewords (0, 2, 1) survive; off-axis (3, 4) and out-of-segment
        # (5, 6) do not. Order is by tau ascending: 0, 2, 1.
        np.testing.assert_array_equal(result, [0, 2, 1])

    def test_alpha_30_admits_mild_off_axis(self) -> None:
        result = filter_and_order(
            self.p_c, self.p_t, self.cb, math.radians(30),
        )
        # Codeword 3 (5, 1) is ~11.3 deg off axis at midpoint → admitted.
        # Codeword 4 (5, 3) is ~31 deg off axis at midpoint → not admitted at 30 deg.
        np.testing.assert_array_equal(result, [0, 2, 3, 1])

    def test_alpha_45_admits_more_off_axis(self) -> None:
        result = filter_and_order(
            self.p_c, self.p_t, self.cb, math.radians(45),
        )
        # Codeword 4 (5, 3) is ~31 deg → admitted at 45 deg.
        np.testing.assert_array_equal(result, [0, 2, 3, 4, 1])

    def test_out_of_segment_dropped(self) -> None:
        # Codewords 5 and 6 lie strictly past the endpoints — never returned
        # regardless of alpha.
        for deg in (10, 30, 90):
            result = filter_and_order(
                self.p_c, self.p_t, self.cb, math.radians(deg),
            )
            assert 5 not in result.tolist()
            assert 6 not in result.tolist()


class TestFilterAndOrderEdgeCases:
    def test_degenerate_axis_returns_empty(self) -> None:
        cb = _toy_codebook()
        p = cb[0].copy()
        result = filter_and_order(p, p, cb, math.radians(45))
        assert result.size == 0
        assert result.dtype == np.int64

    def test_float32_roundoff_keeps_target(self) -> None:
        # Two nearly-equal float32 vectors; computing tau for p_t in
        # float32 routinely overshoots 1.0 by a few ulps. The segment_mask
        # tolerance must catch this so the target codeword survives.
        p_c = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        p_t = np.array([0.10001, 0.20002, 0.30003], dtype=np.float32)
        codebook = np.stack([p_c, p_t]).astype(np.float32)
        result = filter_and_order(p_c, p_t, codebook, math.radians(89))
        # Both codewords (origin + target) must be in the result.
        assert 0 in result.tolist()
        assert 1 in result.tolist()

    def test_no_segment_restriction_drops_segment_mask(self) -> None:
        # restrict_to_segment=False removes the tau in [0, 1] gate; codewords
        # that still satisfy BOTH endpoint cones (i.e., would never be ruled
        # out for geometric reasons) are kept regardless of where on the
        # axis they project.
        cb = _toy_codebook()
        p_c, p_t = cb[0], cb[1]
        loose = set(
            filter_and_order(
                p_c, p_t, cb, math.radians(45), restrict_to_segment=False,
            ).tolist()
        )
        strict = set(
            filter_and_order(
                p_c, p_t, cb, math.radians(45), restrict_to_segment=True,
            ).tolist()
        )
        # Strict result is a subset of loose (segment mask only removes
        # codewords; never adds them).
        assert strict <= loose


class TestConeCandidateFilterFacade:
    """Configuration-bound facade defers to ``filter_and_order``."""

    def test_callable_matches_functional_form(self) -> None:
        cb = _toy_codebook()
        p_c, p_t = cb[0], cb[1]
        flt = ConeCandidateFilter(alpha_deg=20.0)
        functional = filter_and_order(p_c, p_t, cb, flt.alpha_rad)
        np.testing.assert_array_equal(flt(p_c, p_t, cb), functional)

    def test_alpha_rad_consistent_with_degrees(self) -> None:
        flt = ConeCandidateFilter(alpha_deg=30.0)
        assert flt.alpha_rad == pytest.approx(math.pi / 6)

    def test_restrict_to_segment_propagates(self) -> None:
        cb = _toy_codebook()
        p_c, p_t = cb[0], cb[1]
        flt_strict = ConeCandidateFilter(
            alpha_deg=45.0, restrict_to_segment=True,
        )
        flt_loose = ConeCandidateFilter(
            alpha_deg=45.0, restrict_to_segment=False,
        )
        strict_arr = flt_strict(p_c, p_t, cb)
        loose_arr = flt_loose(p_c, p_t, cb)
        # The functional call with the same flag must match the facade.
        np.testing.assert_array_equal(
            strict_arr,
            filter_and_order(
                p_c, p_t, cb, flt_strict.alpha_rad, restrict_to_segment=True,
            ),
        )
        np.testing.assert_array_equal(
            loose_arr,
            filter_and_order(
                p_c, p_t, cb, flt_loose.alpha_rad, restrict_to_segment=False,
            ),
        )
