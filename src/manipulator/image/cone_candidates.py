"""Origin→target double-cone filter over a VQGAN codebook.

For one patch position with origin embedding ``p_c`` and target embedding
``p_t`` (both rows of the codebook), return codebook indices that lie inside
the double-cone of half-angle ``alpha`` around the segment, sorted by axis
projection ``tau`` (origin → target).

The pipeline is functional. Each step is a pure NumPy operation; the
``ConeCandidateFilter`` class merely binds configuration.

Geometry::

      p_c ------------- p_t           axis = p_t - p_c
       \\       α      /
        \\     /-\\    /
         \\   /   \\  /              cone half-angle α from each endpoint
          \\ /     \\/
           X       X                 codewords inside BOTH cones survive
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


_EPS = 1e-12


# ---------------------------------------------------------------------------
# Pure functional steps
# ---------------------------------------------------------------------------


def axis_vector(p_c: NDArray[np.floating], p_t: NDArray[np.floating]) -> tuple[NDArray[np.float64], float]:
    """Return (axis, axis_norm) where axis = p_t - p_c. Computed in float64."""
    axis = (p_t.astype(np.float64) - p_c.astype(np.float64))
    return axis, float(np.linalg.norm(axis))


def projection_tau(
    codebook: NDArray[np.floating],
    p_c: NDArray[np.floating],
    axis: NDArray[np.float64],
    axis_norm: float,
) -> NDArray[np.float64]:
    """Per-codeword projection fraction along axis. τ=0 at p_c, τ=1 at p_t."""
    diff_c = codebook.astype(np.float64) - p_c.astype(np.float64)
    return (diff_c @ axis) / (axis_norm * axis_norm)


def endpoint_cosines(
    codebook: NDArray[np.floating],
    p_c: NDArray[np.floating],
    p_t: NDArray[np.floating],
    axis: NDArray[np.float64],
    axis_norm: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cosines of angles seen from each endpoint toward the other.

    ``cos_c[i]`` = cosine of angle between (codebook[i] − p_c) and axis.
    ``cos_t[i]`` = cosine of angle between (codebook[i] − p_t) and −axis.

    For the endpoint codewords themselves the difference is zero and the
    cosine is undefined; both are set to 1.0 (i.e. perfectly aligned, in
    cone) so the endpoints survive any filter.
    """
    cb64 = codebook.astype(np.float64)
    diff_c = cb64 - p_c.astype(np.float64)
    diff_t = cb64 - p_t.astype(np.float64)
    norm_c = np.linalg.norm(diff_c, axis=1)
    norm_t = np.linalg.norm(diff_t, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_c = (diff_c @ axis) / (norm_c * axis_norm)
        cos_t = (diff_t @ (-axis)) / (norm_t * axis_norm)

    cos_c = np.where(norm_c < _EPS, 1.0, cos_c)
    cos_t = np.where(norm_t < _EPS, 1.0, cos_t)
    return cos_c, cos_t


def cone_mask(
    cos_c: NDArray[np.floating],
    cos_t: NDArray[np.floating],
    alpha_rad: float,
) -> NDArray[np.bool_]:
    """Boolean mask: codewords inside BOTH endpoint cones of half-angle α."""
    cos_alpha = math.cos(alpha_rad)
    return (cos_c >= cos_alpha) & (cos_t >= cos_alpha)


def segment_mask(tau: NDArray[np.floating], tol: float = 1e-6) -> NDArray[np.bool_]:
    """Boolean mask: codewords whose projection falls inside [0, 1] ± tol.

    Tolerance absorbs float roundoff at the endpoints — without it, the
    target codeword may end up at τ = 1 + ε and be dropped.
    """
    return (tau >= -tol) & (tau <= 1.0 + tol)


def order_by_axis(
    indices: NDArray[np.int64],
    tau: NDArray[np.floating],
    origin_distance: NDArray[np.floating],
) -> NDArray[np.int64]:
    """Sort indices ascending by tau; tiebreak by ascending origin distance."""
    keys_tau = tau[indices]
    keys_dist = origin_distance[indices]
    order = np.lexsort((keys_dist, keys_tau))
    return indices[order]


def filter_and_order(
    p_c: NDArray[np.float32],
    p_t: NDArray[np.float32],
    codebook: NDArray[np.float32],
    alpha_rad: float,
    restrict_to_segment: bool = True,
) -> NDArray[np.int64]:
    """Top-level functional pipeline.

    Args:
        p_c: Origin patch embedding, shape (d,).
        p_t: Target patch embedding, shape (d,).
        codebook: Full codebook, shape (n_codes, d). Any row order.
        alpha_rad: Cone half-angle in radians.
        restrict_to_segment: If True (default), drop codewords whose
            projection τ falls outside [0, 1] — i.e., that lie beyond
            either endpoint along the axis.

    Returns:
        Codeword indices (rows of ``codebook``) that survive the cone
        filter, sorted ascending by projection ``τ`` (origin → target).
        Tiebreak: ascending L2 distance from ``p_c``.

        Returns an empty array (dtype int64) if the segment is
        degenerate (``|p_t − p_c| ≈ 0``).
    """
    axis, axis_norm = axis_vector(p_c, p_t)
    if axis_norm < _EPS:
        return np.array([], dtype=np.int64)

    tau = projection_tau(codebook, p_c, axis, axis_norm)
    cos_c, cos_t = endpoint_cosines(codebook, p_c, p_t, axis, axis_norm)

    keep = cone_mask(cos_c, cos_t, alpha_rad)
    if restrict_to_segment:
        keep &= segment_mask(tau)

    kept = np.where(keep)[0].astype(np.int64)
    if kept.size == 0:
        return kept

    origin_distance = np.linalg.norm(codebook.astype(np.float64) - p_c.astype(np.float64), axis=1)
    return order_by_axis(kept, tau, origin_distance)


# ---------------------------------------------------------------------------
# Configuration-bound facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConeCandidateFilter:
    """Bound configuration for the cone candidate pipeline.

    Use when α is fixed across many (p_c, p_t) pairs — typical at run
    construction. For one-off use, call ``filter_and_order`` directly.
    """

    alpha_deg: float
    restrict_to_segment: bool = True

    @property
    def alpha_rad(self) -> float:
        return math.radians(self.alpha_deg)

    def __call__(
        self,
        p_c: NDArray[np.float32],
        p_t: NDArray[np.float32],
        codebook: NDArray[np.float32],
    ) -> NDArray[np.int64]:
        return filter_and_order(
            p_c,
            p_t,
            codebook,
            alpha_rad=self.alpha_rad,
            restrict_to_segment=self.restrict_to_segment,
        )


__all__ = [
    "ConeCandidateFilter",
    "axis_vector",
    "cone_mask",
    "endpoint_cosines",
    "filter_and_order",
    "order_by_axis",
    "projection_tau",
    "segment_mask",
]
