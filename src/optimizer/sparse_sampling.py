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

Additionally, :class:`MultiTierSparseSampling` provides explicit
multi-tier coverage of the image-activation regime. Motivated by the
Exp-22 search-space-coverage audit which showed the image-genome
activation range ``[30, 222]`` was never visited (max 25/222 = 11.3 %).
Each tier defines its own ``p_active`` and population fraction,
guaranteeing spike + medium + heavy individuals coexist from gen 0.
Persistence on the Pareto front is then governed by AGE-MOEA-II's own
selection pressure rather than by init-distribution lock-in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling

_logger = logging.getLogger(__name__)


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
        if n_var < self.text_dim:
            raise ValueError(
                f"problem.n_var ({n_var}) must be >= text_dim ({self.text_dim})"
            )

        n_image = n_var - self.text_dim
        xu = np.asarray(problem.xu, dtype=np.int64)  # inclusive upper bounds
        image_xu = xu[:n_image]
        text_xu = xu[n_image:]

        samples = np.zeros((n_samples, n_var), dtype=np.int64)

        # Image block — skipped entirely when n_image == 0 (modality=text_only).
        if n_image > 0:
            # Partition population into three sub-samplers
            n_zero = int(round(n_samples * self.zero_anchor_fraction))
            n_uniform = int(round(n_samples * self.uniform_fallback_fraction))
            n_zero = min(n_zero, n_samples)
            n_uniform = min(n_uniform, n_samples - n_zero)
            n_geometric = n_samples - n_zero - n_uniform

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


class MultiTierSparseSampling(Sampling):
    """Multi-tier sparse sampler for explicit image-activation regime coverage.

    Population is partitioned into a small set of tiers, each with its own
    Bernoulli ``p_active``. The tier list is configured by the caller and
    typically spans spike → medium → heavy regimes (e.g. 0.005 / 0.03 /
    0.10 / 0.30, giving expected n_active of ~1, ~7, ~22, ~67 on a 222-
    position image genome).

    Active genes draw their codeword uniformly in ``[1, bound]``; an
    optional zero-anchor block emits exact-zero image vectors, matching
    :class:`SparseSampling` for the trivial-perturbation Pareto corner.
    Text-block genes are drawn uniform-random across all individuals.

    :param text_dim: Number of text genes in the genotype (trailing block).
    :param tiers: Sequence of ``(p_active, fraction)`` tuples. Tier
        fractions are interpreted relative to the non-zero-anchor share
        of the population and normalised internally — they need not sum
        to any particular value. The last tier absorbs rounding residue.
    :param zero_anchor_fraction: Fraction of the population emitted as
        exact zero in the image block.
    :param seed: Optional RNG seed for reproducibility.

    :raises ValueError: if ``tiers`` is empty, fractions are out of
        range, or all tier fractions sum to zero.
    """

    def __init__(
        self,
        text_dim: int,
        tiers: list[tuple[float, float]],
        zero_anchor_fraction: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if text_dim < 0:
            raise ValueError(f"text_dim must be ≥ 0, got {text_dim}")
        if not tiers:
            raise ValueError("tiers must contain at least one entry")
        if not 0.0 <= zero_anchor_fraction <= 1.0:
            raise ValueError(
                f"zero_anchor_fraction must be in [0, 1], got {zero_anchor_fraction}"
            )

        for i, (p, frac) in enumerate(tiers):
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"tier {i}: p_active must be in [0, 1], got {p}"
                )
            if frac < 0.0:
                raise ValueError(
                    f"tier {i}: fraction must be ≥ 0, got {frac}"
                )

        if sum(f for _, f in tiers) <= 0.0:
            raise ValueError("at least one tier must have a positive fraction")

        self.text_dim = text_dim
        self.tiers = list(tiers)
        self.zero_anchor_fraction = zero_anchor_fraction
        self._rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray[np.int64]:
        n_var = int(problem.n_var)
        if n_var < self.text_dim:
            raise ValueError(
                f"problem.n_var ({n_var}) must be >= text_dim ({self.text_dim})"
            )

        n_image = n_var - self.text_dim
        xu = np.asarray(problem.xu, dtype=np.int64)
        image_xu = xu[:n_image]
        text_xu = xu[n_image:]

        samples = np.zeros((n_samples, n_var), dtype=np.int64)

        # Image block — skipped entirely when n_image == 0 (modality=text_only).
        if n_image > 0:
            n_zero = int(round(n_samples * self.zero_anchor_fraction))
            n_zero = min(n_zero, n_samples)
            cursor = n_zero

            # Allocate counts per tier; last tier absorbs rounding remainder.
            remaining = n_samples - n_zero
            used = 0
            tier_counts: list[int] = []
            for idx, (_, frac) in enumerate(self.tiers):
                if idx == len(self.tiers) - 1:
                    tier_counts.append(max(remaining - used, 0))
                else:
                    count = int(round(remaining * frac / max(
                        sum(f for _, f in self.tiers), 1e-12
                    )))
                    count = min(count, remaining - used)
                    tier_counts.append(count)
                    used += count

            for (p_active, _), count in zip(self.tiers, tier_counts):
                if count <= 0:
                    continue
                mask = self._rng.random((count, n_image)) < p_active
                depth_rand = self._rng.random((count, n_image))
                depth = 1 + np.floor(depth_rand * image_xu).astype(np.int64)
                samples[cursor:cursor + count, :n_image] = mask * depth
                cursor += count

        if self.text_dim > 0:
            text_rand = self._rng.random((n_samples, self.text_dim))
            text_samples = np.floor(
                text_rand * (text_xu + 1)
            ).astype(np.int64)
            samples[:, n_image:] = text_samples

        return samples


class ScoreGuidedMultiTierSampling(Sampling):
    """Multi-tier sparse sampler with per-position score-bias.

    Phase-2 extension to :class:`MultiTierSparseSampling`. Each tier
    still defines a Bernoulli ``p_active`` and population fraction; the
    target n_active per individual is drawn from
    ``Binomial(n_image, p_active)``. **Which** positions become active is
    then chosen by binary tournament weighted by an externally-computed
    importance ``score`` (see :mod:`src.optimizer.position_scoring`):
    pick two random unused positions, activate the one with the lower
    score (= more important).

    Codeword values for active positions are still drawn uniformly in
    ``[1, image_xu[i]]``; text genes are uniform-random across all
    individuals.

    :param text_dim: Number of text genes in the genotype (trailing block).
    :param tiers: As in :class:`MultiTierSparseSampling`.
    :param score: ``(n_image,)`` per-position importance score; lower
        = more important. Length must match the image-block dimension
        at ``_do`` time, otherwise :class:`ValueError` is raised.
    :param zero_anchor_fraction: Fraction emitted as exact zero in the
        image block.
    :param seed: Optional RNG seed.

    :raises ValueError: as in :class:`MultiTierSparseSampling`, or if
        ``score`` length mismatches the image-block dimension.
    """

    def __init__(
        self,
        text_dim: int,
        tiers: list[tuple[float, float]],
        score: NDArray[np.float64],
        zero_anchor_fraction: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if text_dim < 0:
            raise ValueError(f"text_dim must be ≥ 0, got {text_dim}")
        if not tiers:
            raise ValueError("tiers must contain at least one entry")
        if not 0.0 <= zero_anchor_fraction <= 1.0:
            raise ValueError(
                f"zero_anchor_fraction must be in [0, 1], got {zero_anchor_fraction}"
            )

        for i, (p, frac) in enumerate(tiers):
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"tier {i}: p_active must be in [0, 1], got {p}"
                )
            if frac < 0.0:
                raise ValueError(
                    f"tier {i}: fraction must be ≥ 0, got {frac}"
                )
        if sum(f for _, f in tiers) <= 0.0:
            raise ValueError("at least one tier must have a positive fraction")

        score = np.asarray(score, dtype=np.float64)
        if score.ndim != 1:
            raise ValueError(f"score must be 1-D; got shape {score.shape}")

        self.text_dim = text_dim
        self.tiers = list(tiers)
        self.score = score
        self.zero_anchor_fraction = zero_anchor_fraction
        self._rng = np.random.default_rng(seed)

    def _tournament_pick(self, available: list[int]) -> int:
        """Binary tournament on ``self.score``: pick lower (= more important)."""
        if len(available) == 1:
            return available[0]
        a_idx, b_idx = self._rng.integers(0, len(available), size=2)
        a, b = available[a_idx], available[b_idx]
        return a if self.score[a] <= self.score[b] else b

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray[np.int64]:
        n_var = int(problem.n_var)
        if n_var < self.text_dim:
            raise ValueError(
                f"problem.n_var ({n_var}) must be >= text_dim ({self.text_dim})"
            )
        if n_var == self.text_dim:
            raise ValueError(
                "ScoreGuidedMultiTierSampling requires image_dim > 0; "
                "modality=text_only must use mode=sparse_multitier or "
                "mode=uniform"
            )

        n_image = n_var - self.text_dim
        if self.score.shape[0] != n_image:
            raise ValueError(
                f"score length {self.score.shape[0]} != image-block dim {n_image}"
            )

        xu = np.asarray(problem.xu, dtype=np.int64)
        image_xu = xu[:n_image]
        text_xu = xu[n_image:]

        n_zero = int(round(n_samples * self.zero_anchor_fraction))
        n_zero = min(n_zero, n_samples)

        samples = np.zeros((n_samples, n_var), dtype=np.int64)
        cursor = n_zero

        remaining = n_samples - n_zero
        used = 0
        tier_counts: list[int] = []
        for idx, (_, frac) in enumerate(self.tiers):
            if idx == len(self.tiers) - 1:
                tier_counts.append(max(remaining - used, 0))
            else:
                count = int(round(remaining * frac / max(
                    sum(f for _, f in self.tiers), 1e-12
                )))
                count = min(count, remaining - used)
                tier_counts.append(count)
                used += count

        for (p_active, _), count in zip(self.tiers, tier_counts):
            if count <= 0:
                continue
            for _k in range(count):
                target_n = int(self._rng.binomial(n_image, p_active))
                target_n = min(target_n, n_image)
                if target_n <= 0:
                    cursor += 1
                    continue
                available = list(range(n_image))
                active: list[int] = []
                for _ in range(target_n):
                    pos = self._tournament_pick(available)
                    active.append(pos)
                    available.remove(pos)
                idx_arr = np.asarray(active, dtype=np.int64)
                depth = 1 + np.floor(
                    self._rng.random(target_n) * image_xu[idx_arr]
                ).astype(np.int64)
                samples[cursor, idx_arr] = depth
                cursor += 1

        if self.text_dim > 0:
            text_rand = self._rng.random((n_samples, self.text_dim))
            text_samples = np.floor(
                text_rand * (text_xu + 1)
            ).astype(np.int64)
            samples[:, n_image:] = text_samples

        return samples


class DiversityFPSMultiTierSampling(MultiTierSparseSampling):
    """Multi-tier sampler with embedding-FPS-spread codes per active position.

    Phase-3 extension to :class:`MultiTierSparseSampling`. Tier-based
    activation-mask logic is identical to the parent — population is split
    across (zero-anchor + N tiers), and each individual's image-block mask
    is drawn Bernoulli with that tier's ``p_active``.

    The novel part: the **gene values at active positions** are no longer
    uniform-random ranks ``[1, image_xu]``. Instead, for each position
    ``p`` whose active mask is set in some subset ``A_p`` of the population,
    Farthest-Point-Sampling is run over the codebook subset
    ``candidates_per_position[p]`` (= the KNN-ordered codeword list at
    position ``p``) to pick ``|A_p|`` codes whose VQGAN embeddings are
    maximally far apart. The resulting per-individual code-rank is
    written to ``samples[i, p]`` as ``rank + 1`` (1-indexed; rank 0 in
    the candidate list = nearest neighbour, encoded as gene-value 1).

    This intervenes only at ``init`` — the genome encoding, the GA
    operators, and the search space are all unchanged. Each individual
    can still mutate to any code in its position's KNN-ordered pool;
    FPS only seeds the population with broad embedding-space coverage
    so the optimizer doesn't have to do that exploration via random walk.

    :param text_dim: Number of text genes in the genotype (trailing block).
    :param tiers: As in :class:`MultiTierSparseSampling`.
    :param codebook: ``(n_codes, d_z)`` VQGAN codebook embedding matrix.
        Used by :func:`farthest_point_sampling` as the embedding space.
    :param candidates_per_position: Tuple of length ``n_image``; entry
        ``p`` is a 1-D int array of code-IDs ordered by ascending cosine
        distance from ``origin_code[p]`` (= the per-position KNN-pool
        as built by :func:`select_candidates`). Length of each entry
        must equal the position's ``image_xu`` upper-bound.
    :param zero_anchor_fraction: As in parent.
    :param fps_subset_size: Lazy-FPS candidate cap per pick step. None =
        full pool (slow on 16384-pool); 512 is a good speed/quality
        balance for VQGAN-f8-16384.
    :param fps_metric: ``"cosine"`` (default) or ``"l2"`` over codebook
        embeddings. Cosine matches the existing KNN-build distance.
    :param seed: Optional RNG seed.

    :raises ValueError: as in parent, or if codebook / candidates shapes
        are inconsistent at ``_do`` time.
    """

    def __init__(
        self,
        text_dim: int,
        tiers: list[tuple[float, float]],
        codebook: NDArray[np.float32],
        candidates_per_position: tuple[NDArray[np.int64], ...],
        zero_anchor_fraction: float = 0.05,
        fps_subset_size: int | None = 512,
        fps_metric: str = "cosine",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            text_dim=text_dim,
            tiers=tiers,
            zero_anchor_fraction=zero_anchor_fraction,
            seed=seed,
        )

        codebook_arr = np.asarray(codebook, dtype=np.float32)
        if codebook_arr.ndim != 2:
            raise ValueError(
                f"codebook must be 2-D; got shape {codebook_arr.shape}"
            )

        self.codebook = codebook_arr
        self.candidates_per_position = tuple(
            np.asarray(c, dtype=np.int64) for c in candidates_per_position
        )
        self.fps_subset_size = fps_subset_size
        self.fps_metric = fps_metric

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray[np.int64]:
        from .diversity_fps import farthest_point_sampling

        n_var = int(problem.n_var)
        if n_var < self.text_dim:
            raise ValueError(
                f"problem.n_var ({n_var}) must be >= text_dim ({self.text_dim})"
            )
        if n_var == self.text_dim:
            raise ValueError(
                "DiversityFPSMultiTierSampling requires image_dim > 0; "
                "modality=text_only must use mode=sparse_multitier or "
                "mode=uniform"
            )

        n_image = n_var - self.text_dim
        if len(self.candidates_per_position) != n_image:
            raise ValueError(
                f"candidates_per_position length "
                f"{len(self.candidates_per_position)} != image-block dim "
                f"{n_image}"
            )

        xu = np.asarray(problem.xu, dtype=np.int64)
        image_xu = xu[:n_image]
        text_xu = xu[n_image:]

        samples = np.zeros((n_samples, n_var), dtype=np.int64)

        # 1. Tier-allocation masks (same logic as parent _do, but we keep
        #    only the boolean activation mask — code values come from FPS).
        n_zero = int(round(n_samples * self.zero_anchor_fraction))
        n_zero = min(n_zero, n_samples)
        cursor = n_zero

        remaining = n_samples - n_zero
        used = 0
        tier_counts: list[int] = []
        for idx, (_, frac) in enumerate(self.tiers):
            if idx == len(self.tiers) - 1:
                tier_counts.append(max(remaining - used, 0))
            else:
                count = int(round(remaining * frac / max(
                    sum(f for _, f in self.tiers), 1e-12
                )))
                count = min(count, remaining - used)
                tier_counts.append(count)
                used += count

        active_mask = np.zeros((n_samples, n_image), dtype=bool)
        for (p_active, _), count in zip(self.tiers, tier_counts):
            if count <= 0:
                continue
            tier_mask = self._rng.random((count, n_image)) < p_active
            active_mask[cursor:cursor + count] = tier_mask
            cursor += count

        # 2. Per-position FPS over the KNN-pool candidates.
        for p in range(n_image):
            indiv = np.flatnonzero(active_mask[:, p])
            if len(indiv) == 0:
                continue

            pool_codes = self.candidates_per_position[p]
            pool_size = len(pool_codes)
            if pool_size == 0:
                continue

            n_picks = min(len(indiv), pool_size)
            pool_E = self.codebook[pool_codes]      # (pool_size, d_z)

            picks = farthest_point_sampling(
                pool_E,
                n_picks=n_picks,
                rng=self._rng,
                candidate_subset=self.fps_subset_size,
                metric=self.fps_metric,
            )

            # Shuffle which individual gets which pick to avoid pick-order
            # correlation with population-tier order (e.g. tier-5 heavy
            # individuals always landing on the first FPS pick).
            order = self._rng.permutation(len(indiv))

            # Gene values are 1-indexed ranks into the candidate list.
            # picks[k] is a row index into pool_codes ∈ [0, pool_size).
            # The candidate list itself is already KNN-ordered, so the
            # row-index *is* the rank: rank-0 = nearest neighbour.
            # Genome encodes "use candidate[gene-1]", so gene = pick + 1.
            gene_values = (picks.astype(np.int64) + 1)

            # Clip just-in-case to the per-position xu (some configs may
            # cap candidates below the codebook size).
            gene_values = np.minimum(gene_values, image_xu[p])

            samples[indiv[order[:n_picks]], p] = gene_values

            # If more individuals were active at this position than picks
            # (rare: only when pool_size < n_individuals), fall back to
            # uniform-random for the overflow individuals.
            if len(indiv) > n_picks:
                overflow = indiv[order[n_picks:]]
                fill = 1 + np.floor(
                    self._rng.random(len(overflow)) * image_xu[p]
                ).astype(np.int64)
                samples[overflow, p] = fill

        # 3. Text block — uniform random as in parent.
        if self.text_dim > 0:
            text_rand = self._rng.random((n_samples, self.text_dim))
            text_samples = np.floor(
                text_rand * (text_xu + 1)
            ).astype(np.int64)
            samples[:, n_image:] = text_samples

        return samples


def build_sampler_from_config(
    sampling_cfg,
    text_dim: int,
    *,
    codebook: NDArray[np.float32] | None = None,
    candidates_per_position: tuple[NDArray[np.int64], ...] | None = None,
) -> Sampling | None:
    """Dispatch a :class:`Sampling` instance from a :class:`SamplingConfig`.

    Returns ``None`` for ``mode="uniform"`` (callers should leave PyMoo's
    own ``IntegerRandomSampling`` in place). For ``sparse_score_guided``
    with a missing ``score_path`` on disk, falls back to
    :class:`MultiTierSparseSampling` with a single warning — never
    crashes, never silently degrades to plain sparse.

    :param sampling_cfg: ``SamplingConfig`` instance.
    :param text_dim: Number of text genes in the genotype.
    :param codebook: VQGAN codebook embeddings, required for
        ``sparse_multitier_fps`` mode. Ignored otherwise.
    :param candidates_per_position: Per-position KNN-ordered candidate
        code lists, required for ``sparse_multitier_fps``.
    :raises ValueError: for unknown modes or missing required fields.
    """
    mode = sampling_cfg.mode
    if mode == "uniform":
        return None

    if mode == "sparse":
        return SparseSampling(
            text_dim=text_dim,
            p_active=sampling_cfg.p_active,
            geometric_rate=sampling_cfg.geometric_rate,
            zero_anchor_fraction=sampling_cfg.zero_anchor_fraction,
            uniform_fallback_fraction=sampling_cfg.uniform_fallback_fraction,
        )

    if mode == "sparse_multitier":
        if not sampling_cfg.tiers:
            raise ValueError(
                "sampling.mode='sparse_multitier' requires non-empty "
                "sampling.tiers"
            )
        return MultiTierSparseSampling(
            text_dim=text_dim,
            tiers=[(t.p_active, t.fraction) for t in sampling_cfg.tiers],
            zero_anchor_fraction=sampling_cfg.zero_anchor_fraction,
        )

    if mode == "sparse_multitier_fps":
        if not sampling_cfg.tiers:
            raise ValueError(
                "sampling.mode='sparse_multitier_fps' requires non-empty "
                "sampling.tiers"
            )
        if codebook is None or candidates_per_position is None:
            raise ValueError(
                "sampling.mode='sparse_multitier_fps' requires the caller "
                "to pass `codebook` and `candidates_per_position` (per-seed "
                "KNN-ordered candidate lists). Wire these from the image "
                "manipulator's prepared context."
            )
        return DiversityFPSMultiTierSampling(
            text_dim=text_dim,
            tiers=[(t.p_active, t.fraction) for t in sampling_cfg.tiers],
            codebook=codebook,
            candidates_per_position=candidates_per_position,
            zero_anchor_fraction=sampling_cfg.zero_anchor_fraction,
            fps_subset_size=sampling_cfg.fps_subset_size,
            fps_metric=sampling_cfg.fps_metric,
        )

    if mode == "sparse_score_guided":
        if not sampling_cfg.tiers:
            raise ValueError(
                "sampling.mode='sparse_score_guided' requires non-empty "
                "sampling.tiers"
            )
        if not sampling_cfg.score_path:
            raise ValueError(
                "sampling.mode='sparse_score_guided' requires "
                "sampling.score_path"
            )
        score_path = Path(sampling_cfg.score_path).expanduser()
        if not score_path.exists():
            _logger.warning(
                "sampling.score_path %s missing on disk; falling back to "
                "sparse_multitier with the configured tiers", score_path,
            )
            return MultiTierSparseSampling(
                text_dim=text_dim,
                tiers=[(t.p_active, t.fraction) for t in sampling_cfg.tiers],
                zero_anchor_fraction=sampling_cfg.zero_anchor_fraction,
            )
        score = np.load(score_path)
        return ScoreGuidedMultiTierSampling(
            text_dim=text_dim,
            tiers=[(t.p_active, t.fraction) for t in sampling_cfg.tiers],
            score=score,
            zero_anchor_fraction=sampling_cfg.zero_anchor_fraction,
        )

    raise ValueError(
        f"Unknown sampling mode {mode!r}; expected 'uniform', 'sparse', "
        f"'sparse_multitier', 'sparse_multitier_fps', or "
        f"'sparse_score_guided'"
    )
