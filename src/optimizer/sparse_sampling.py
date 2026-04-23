"""Sparse init-population sampler for the SMOO boundary optimizer.

Replaces PyMoo's default ``IntegerRandomSampling`` with a three-way mixture
designed to seed the population near the identity genotype while still
admitting full-codebook exploration:

1. **Zero anchors** — exact ``[0, ..., 0]`` in the image block. Guarantees the
   no-op solution is in the initial Pareto set and protects against drift
   away from identity.
2. **Uniform-sparse** — Bernoulli mask ``(p_active)`` over image genes;
   active genes draw depth uniformly from ``[1, bound]``. Diversity
   insurance against over-aggressive geometric bias.
3. **Geometric-sparse** — Bernoulli mask ``(p_active)`` over image genes;
   active genes draw depth from a truncated geometric favouring small
   values. The primary distribution — biases toward shallow-sparse
   perturbations, which is the untested Pareto corner diagnosed in Exp-09.

Text-block genes are drawn uniform-random across all sub-samplers
(text has no sparsity problem — cf. 2026-04-16 feedback).

See ``docs/diary/2026-04-19-Saliency-Validation-And-Phase1-Design.md``
for the empirical rationale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling


class SparseSampling(Sampling):
    """Bernoulli-gated sparse sampler with geometric depth over image block.

    The genotype is ``[image_genes | text_genes]`` with constant
    ``text_dim`` across seeds. ``image_dim`` is derived from
    ``problem.n_var - text_dim`` at each ``_do`` invocation, so one
    instance works for all seeds in a run.

    :param text_dim: Number of text genes in the genotype (trailing block).
    :param p_active: Bernoulli probability an image gene is active per
        individual. ``E[n_active] ≈ image_dim × p_active``.
    :param geometric_rate: Rate parameter of the truncated geometric
        depth distribution for active genes. Higher rate → shallower
        depths. Must be in ``(0, 1]``.
    :param zero_anchor_fraction: Fraction of the population emitted as
        exact zero vectors in the image block (text block remains
        uniform-random).
    :param uniform_fallback_fraction: Fraction of the population that
        uses uniform depth instead of geometric. Insurance against
        over-aggressive geometric bias.
    :param seed: Optional RNG seed for reproducibility.

    :raises ValueError: if parameters are out of range or fractions sum > 1.
    """

    def __init__(
        self,
        text_dim: int,
        p_active: float = 0.03,
        geometric_rate: float = 0.5,
        zero_anchor_fraction: float = 0.05,
        uniform_fallback_fraction: float = 0.10,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if text_dim < 0:
            raise ValueError(f"text_dim must be ≥ 0, got {text_dim}")
        if not 0.0 <= p_active <= 1.0:
            raise ValueError(f"p_active must be in [0, 1], got {p_active}")
        if not 0.0 < geometric_rate <= 1.0:
            raise ValueError(
                f"geometric_rate must be in (0, 1], got {geometric_rate}"
            )
        if not 0.0 <= zero_anchor_fraction <= 1.0:
            raise ValueError(
                f"zero_anchor_fraction must be in [0, 1], got {zero_anchor_fraction}"
            )
        if not 0.0 <= uniform_fallback_fraction <= 1.0:
            raise ValueError(
                f"uniform_fallback_fraction must be in [0, 1], "
                f"got {uniform_fallback_fraction}"
            )
        if zero_anchor_fraction + uniform_fallback_fraction > 1.0:
            raise ValueError(
                "zero_anchor_fraction + uniform_fallback_fraction must be ≤ 1.0; "
                f"got {zero_anchor_fraction + uniform_fallback_fraction}"
            )

        self.text_dim = text_dim
        self.p_active = p_active
        self.geometric_rate = geometric_rate
        self.zero_anchor_fraction = zero_anchor_fraction
        self.uniform_fallback_fraction = uniform_fallback_fraction
        self._rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray[np.int64]:
        n_var = int(problem.n_var)
        if n_var <= self.text_dim:
            raise ValueError(
                f"problem.n_var ({n_var}) must be > text_dim ({self.text_dim})"
            )

        n_image = n_var - self.text_dim
        xu = np.asarray(problem.xu, dtype=np.int64)  # inclusive upper bounds
        image_xu = xu[:n_image]
        text_xu = xu[n_image:]

        # Partition population into three sub-samplers
        n_zero = int(round(n_samples * self.zero_anchor_fraction))
        n_uniform = int(round(n_samples * self.uniform_fallback_fraction))
        n_zero = min(n_zero, n_samples)
        n_uniform = min(n_uniform, n_samples - n_zero)
        n_geometric = n_samples - n_zero - n_uniform

        samples = np.zeros((n_samples, n_var), dtype=np.int64)

        # -- Image block -----------------------------------------------------
        # Zero anchors: rows [0, n_zero) — image already zero from np.zeros

        cursor = n_zero

        # Uniform-sparse block: Bernoulli mask × uniform depth in [1, bound]
        if n_uniform > 0:
            mask = (
                self._rng.random((n_uniform, n_image)) < self.p_active
            )
            unif_rand = self._rng.random((n_uniform, n_image))
            unif_depth = 1 + np.floor(unif_rand * image_xu).astype(np.int64)
            samples[cursor:cursor + n_uniform, :n_image] = mask * unif_depth
            cursor += n_uniform

        # Geometric-sparse block: Bernoulli mask × truncated geometric depth
        if n_geometric > 0:
            mask = (
                self._rng.random((n_geometric, n_image)) < self.p_active
            )
            geo = self._rng.geometric(
                p=self.geometric_rate, size=(n_geometric, n_image)
            ).astype(np.int64)
            # Truncate per-column to each gene's inclusive bound
            geo = np.minimum(geo, image_xu[np.newaxis, :])
            samples[cursor:cursor + n_geometric, :n_image] = mask * geo

        # -- Text block ------------------------------------------------------
        if self.text_dim > 0:
            text_rand = self._rng.random((n_samples, self.text_dim))
            text_samples = np.floor(
                text_rand * (text_xu + 1)
            ).astype(np.int64)
            samples[:, n_image:] = text_samples

        return samples
