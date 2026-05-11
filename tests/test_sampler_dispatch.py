"""Tests for :func:`src.optimizer.sparse_sampling.build_sampler_from_config`.

Covers the four init modes plus the score-guided fallback when
``score_path`` is missing on disk. Mirrors the dispatch wiring in
:class:`src.evolutionary.vlm_boundary_tester.VLMBoundaryTester` so that
config → sampler routing is a single, testable function.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    DEFAULT_MULTITIER_TIERS,
    SamplingConfig,
    SamplingTier,
)
from src.optimizer.sparse_sampling import (
    DiversityFPSMultiTierSampling,
    MultiTierSparseSampling,
    ScoreGuidedMultiTierSampling,
    SparseSampling,
    build_sampler_from_config,
)


def test_uniform_returns_none() -> None:
    cfg = SamplingConfig(mode="uniform")
    assert build_sampler_from_config(cfg, text_dim=3) is None


def test_sparse_dispatches_sparse_sampling() -> None:
    cfg = SamplingConfig(mode="sparse", p_active=0.03)
    sampler = build_sampler_from_config(cfg, text_dim=3)
    assert isinstance(sampler, SparseSampling)
    assert sampler.text_dim == 3
    assert sampler.p_active == 0.03


def test_multitier_dispatches_multitier_sampling() -> None:
    cfg = SamplingConfig(mode="sparse_multitier")
    sampler = build_sampler_from_config(cfg, text_dim=4)
    assert isinstance(sampler, MultiTierSparseSampling)
    assert sampler.text_dim == 4
    assert len(sampler.tiers) == len(DEFAULT_MULTITIER_TIERS)


def test_score_guided_dispatches_score_guided_sampling(tmp_path: Path) -> None:
    score_path = tmp_path / "pattern_seed83.npy"
    np.save(score_path, np.linspace(0.0, 1.0, 222))
    cfg = SamplingConfig(
        mode="sparse_score_guided",
        score_path=str(score_path),
    )
    sampler = build_sampler_from_config(cfg, text_dim=4)
    assert isinstance(sampler, ScoreGuidedMultiTierSampling)
    assert sampler.text_dim == 4
    assert sampler.score.shape == (222,)


def test_score_guided_falls_back_when_path_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    missing = tmp_path / "no_such_score.npy"
    cfg = SamplingConfig(
        mode="sparse_score_guided",
        score_path=str(missing),
    )
    with caplog.at_level(logging.WARNING, logger="src.optimizer.sparse_sampling"):
        sampler = build_sampler_from_config(cfg, text_dim=2)
    assert isinstance(sampler, MultiTierSparseSampling)
    assert not isinstance(sampler, ScoreGuidedMultiTierSampling)
    assert any(
        "missing on disk" in rec.message and "falling back" in rec.message
        for rec in caplog.records
    ), "expected a single fallback warning"


def test_unknown_mode_raises() -> None:
    cfg = SamplingConfig(mode="bogus")
    with pytest.raises(ValueError, match="Unknown sampling mode"):
        build_sampler_from_config(cfg, text_dim=1)


def test_score_guided_requires_score_path() -> None:
    cfg = SamplingConfig(mode="sparse_score_guided", score_path=None)
    with pytest.raises(ValueError, match="score_path"):
        build_sampler_from_config(cfg, text_dim=1)


def test_multitier_requires_non_empty_tiers() -> None:
    cfg = SamplingConfig(mode="sparse_multitier", tiers=())
    with pytest.raises(ValueError, match="tiers"):
        build_sampler_from_config(cfg, text_dim=1)


def test_default_mode_is_sparse_multitier() -> None:
    """Absent-mode config load instantiates multi-tier with documented tiers."""
    cfg = SamplingConfig()
    assert cfg.mode == "sparse_multitier"
    assert cfg.tiers == DEFAULT_MULTITIER_TIERS
    sampler = build_sampler_from_config(cfg, text_dim=1)
    assert isinstance(sampler, MultiTierSparseSampling)


def test_fps_dispatches_diversity_fps_sampler() -> None:
    """sparse_multitier_fps mode dispatches FPS sampler when codebook is wired."""
    rng = np.random.default_rng(0)
    n_codes = 64
    codebook = rng.standard_normal((n_codes, 4)).astype(np.float32)
    n_image = 12
    candidates = tuple(
        rng.permutation(n_codes).astype(np.int64) for _ in range(n_image)
    )
    cfg = SamplingConfig(mode="sparse_multitier_fps")
    sampler = build_sampler_from_config(
        cfg,
        text_dim=2,
        codebook=codebook,
        candidates_per_position=candidates,
    )
    assert isinstance(sampler, DiversityFPSMultiTierSampling)
    assert sampler.text_dim == 2
    assert sampler.codebook.shape == (n_codes, 4)
    assert len(sampler.candidates_per_position) == n_image


def test_fps_requires_codebook_and_candidates() -> None:
    cfg = SamplingConfig(mode="sparse_multitier_fps")
    with pytest.raises(ValueError, match="codebook"):
        build_sampler_from_config(cfg, text_dim=2)


def test_fps_sampler_active_genes_use_distinct_ranks() -> None:
    """At a position activated by N individuals, FPS picks N distinct ranks."""
    rng = np.random.default_rng(42)
    n_codes = 256
    d_z = 8
    codebook = rng.standard_normal((n_codes, d_z)).astype(np.float32)
    n_image = 6
    n_text = 0
    pool_per_pos = 64
    candidates = tuple(
        rng.choice(n_codes, size=pool_per_pos, replace=False).astype(np.int64)
        for _ in range(n_image)
    )

    sampler = DiversityFPSMultiTierSampling(
        text_dim=n_text,
        tiers=[(0.5, 1.0)],
        codebook=codebook,
        candidates_per_position=candidates,
        zero_anchor_fraction=0.0,
        seed=7,
    )

    class StubProblem:
        n_var = n_image
        xu = np.full(n_image, pool_per_pos, dtype=np.int64)

    n_samples = 12
    samples = sampler._do(StubProblem(), n_samples)

    assert samples.shape == (n_samples, n_image)
    assert (samples >= 0).all()
    assert (samples <= pool_per_pos).all()

    # For each position, gene values at active rows should be distinct
    # (FPS-pick yields no repeats unless the pool is exhausted).
    for p in range(n_image):
        col = samples[:, p]
        active_vals = col[col > 0]
        assert len(set(active_vals.tolist())) == len(active_vals), (
            f"position {p}: FPS picked duplicate ranks {active_vals.tolist()}"
        )


def test_fps_sampler_text_only_rejected() -> None:
    """FPS sampler refuses text_only modality (n_image == 0)."""
    rng = np.random.default_rng(0)
    codebook = rng.standard_normal((16, 4)).astype(np.float32)
    sampler = DiversityFPSMultiTierSampling(
        text_dim=4,
        tiers=[(0.5, 1.0)],
        codebook=codebook,
        candidates_per_position=(),
        seed=0,
    )

    class StubProblem:
        n_var = 4
        xu = np.full(4, 5, dtype=np.int64)

    with pytest.raises(ValueError, match="image_dim > 0"):
        sampler._do(StubProblem(), 4)


def test_custom_tiers_passed_through() -> None:
    cfg = SamplingConfig(
        mode="sparse_multitier",
        tiers=(
            SamplingTier(p_active=0.01, fraction=0.5),
            SamplingTier(p_active=0.20, fraction=0.5),
        ),
    )
    sampler = build_sampler_from_config(cfg, text_dim=0)
    assert isinstance(sampler, MultiTierSparseSampling)
    assert sampler.tiers == [(0.01, 0.5), (0.20, 0.5)]
