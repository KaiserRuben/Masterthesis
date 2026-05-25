"""Tests for :class:`StyleGANImageManipulator`.

All tests use a fake SMOO ``StyleGANManipulator`` and a fake class-target
builder — no real generator, no real SUT, no real checkpoint. The
manipulator's correctness here hinges on:

* :meth:`prepare` pulls the origin w + target w from the class-target
  builder and builds a context with the right ``num_ws`` /
  ``kappa_quant_levels`` / ``gene_bounds``.
* :meth:`apply` (and :meth:`apply_batch`) feed κ = gene / Q to the
  underlying ``manipulate()`` and return PIL images.
* Boundary gene values reproduce origin / target (the apply step's
  contract for the optimizer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image

from src.manipulator.image.manipulator import ImageConfig, StyleGANConfig
from src.manipulator.image_stylegan.manipulator import StyleGANImageManipulator
from src.manipulator.image_stylegan.types import StyleGANManipulationContext


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeGenerator(torch.nn.Module):
    """Bare-minimum object that exposes ``num_ws`` for introspection."""

    def __init__(self, num_ws: int = 6) -> None:
        super().__init__()
        self.num_ws = num_ws


class _FakeSMOOManipulator:
    """Stub for SMOO's StyleGANManipulator.

    ``manipulate`` returns a deterministic image batch whose first-pixel
    red channel encodes the input genotype sum — lets the test verify
    that the right batch entries came out without rendering real images.
    Captures the call args so we can inspect cond / weights / candidates.
    """

    def __init__(self, num_ws: int = 6) -> None:
        self._generator = _FakeGenerator(num_ws=num_ws)
        self.calls: list[dict] = []

    def manipulate(self, *, candidates, cond, weights):
        self.calls.append({
            "cond": cond.copy(),
            "weights": weights.copy(),
            "n_w0": len(candidates.w0_candidates),
            "n_wn": len(candidates.wn_candidates),
        })
        batch_size = weights.shape[0]
        # Construct an image batch where image[i] has red channel
        # encoding the mean of weights[i, :] (so tests can decode it).
        imgs = np.zeros((batch_size, 3, 4, 4), dtype=np.float32)
        for i in range(batch_size):
            avg = float(weights[i].mean()) if weights[i].size else 0.0
            imgs[i, 0, :, :] = max(0.0, min(1.0, avg))  # clamp to [0, 1]
        return torch.from_numpy(imgs)


class _FakeClassTargetBuilder:
    """Stub for :class:`StyleGANClassTargetBuilder`.

    Returns hand-crafted w-tensors and a predetermined image; records
    which classes were queried so the test can verify cache traffic.
    """

    def __init__(self, num_ws: int = 6, w_dim: int = 4) -> None:
        self._num_ws = num_ws
        self._w_dim = w_dim
        self.modal_calls: list[str] = []
        self.origin_calls: list[tuple[str, str]] = []

    def ensure_modal_w(self, class_name: str) -> torch.Tensor:
        self.modal_calls.append(class_name)
        # Target w: every entry is 1.0 so we can sanity-check
        # interpolation later.
        return torch.full((1, self._num_ws, self._w_dim), 1.0, dtype=torch.float32)

    def ensure_origin(
        self, origin_class: str, target_class: str,
    ) -> tuple[int, torch.Tensor, Image.Image]:
        self.origin_calls.append((origin_class, target_class))

        # Origin w: every entry is 0.0 — paired with target 1.0 gives a
        # clean (1 - κ)·0 + κ·1 = κ interpolation profile.
        origin_w = torch.full(
            (1, self._num_ws, self._w_dim), 0.0, dtype=torch.float32,
        )
        # Pretend the precheck accepted seed=5 with a tiny synthetic origin.
        img = Image.new("RGB", (4, 4), (42, 0, 0))
        return 5, origin_w, img

    def populate_modal_w(self, classes) -> None:
        for c in classes:
            self.ensure_modal_w(c)

    def populate_origins(self, pairs) -> None:
        for origin, target in pairs:
            self.ensure_origin(origin, target)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_manip(
    *,
    num_ws: int = 6,
    kappa_levels: int = 10,
    categories: tuple[str, ...] = ("junco", "chickadee"),
) -> tuple[StyleGANImageManipulator, _FakeSMOOManipulator, _FakeClassTargetBuilder]:
    smoo = _FakeSMOOManipulator(num_ws=num_ws)
    builder = _FakeClassTargetBuilder(num_ws=num_ws, w_dim=4)
    cfg = ImageConfig(
        backend="stylegan_xl",
        stylegan=StyleGANConfig(kappa_quant_levels=kappa_levels),
    )
    m = StyleGANImageManipulator(
        smoo_manipulator=smoo,
        class_target_builder=builder,
        config=cfg,
        device=torch.device("cpu"),
        categories=categories,
    )
    return m, smoo, builder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_num_ws_inferred(self) -> None:
        m, _, _ = _make_manip(num_ws=8)
        assert m.num_ws == 8

    def test_backend_mismatch_raises(self) -> None:
        smoo = _FakeSMOOManipulator()
        builder = _FakeClassTargetBuilder()
        cfg = ImageConfig(backend="vqgan_codebook")  # wrong backend
        with pytest.raises(ValueError, match="stylegan_xl"):
            StyleGANImageManipulator(
                smoo_manipulator=smoo,
                class_target_builder=builder,
                config=cfg,
                device=torch.device("cpu"),
                categories=("a",),
            )


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_returns_stylegan_context(self) -> None:
        m, _, builder = _make_manip(num_ws=6, kappa_levels=20)
        ctx = m.prepare(
            Image.new("RGB", (4, 4)),
            target_class="junco",
            origin_class="chickadee",
        )
        assert isinstance(ctx, StyleGANManipulationContext)
        assert ctx.target_class == "junco"
        assert ctx.origin_class == "chickadee"
        assert ctx.num_ws == 6
        assert ctx.kappa_quant_levels == 20
        assert ctx.genotype_dim == 6
        np.testing.assert_array_equal(
            ctx.gene_bounds,
            np.full(6, 21, dtype=np.int64),  # Q + 1 = 21
        )

    def test_target_class_required(self) -> None:
        m, _, _ = _make_manip()
        with pytest.raises(ValueError, match="target_class"):
            m.prepare(Image.new("RGB", (4, 4)))

    def test_origin_class_required(self) -> None:
        # Pairwise precheck has no boundary when origin is missing.
        m, _, _ = _make_manip()
        with pytest.raises(ValueError, match="origin_class"):
            m.prepare(Image.new("RGB", (4, 4)), target_class="junco")

    def test_same_class_origin_rejected(self) -> None:
        m, _, _ = _make_manip()
        with pytest.raises(ValueError, match="origin_class"):
            m.prepare(
                Image.new("RGB", (4, 4)),
                target_class="junco",
                origin_class="junco",
            )

    def test_origin_class_distinct_from_target(self) -> None:
        m, _, builder = _make_manip()
        m.prepare(
            Image.new("RGB", (4, 4)),
            target_class="junco",
            origin_class="chickadee",
        )
        assert "junco" in builder.modal_calls
        assert ("chickadee", "junco") in builder.origin_calls

    def test_candidate_strategy_is_stylegan_interp(self) -> None:
        m, _, _ = _make_manip()
        ctx = m.prepare(
            Image.new("RGB", (4, 4)),
            target_class="junco",
            origin_class="chickadee",
        )
        assert ctx.candidate_strategy == "stylegan_interp"

    def test_zero_genotype_dim_matches_num_ws(self) -> None:
        m, _, _ = _make_manip(num_ws=10)
        ctx = m.prepare(
            Image.new("RGB", (4, 4)),
            target_class="junco",
            origin_class="chickadee",
        )
        assert len(ctx.zero_genotype()) == 10


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_single_apply_raises_not_implemented(self) -> None:
        # StyleGAN does not support batch=1 through SMOO (the underlying
        # manipulate asserts cond[1] and IndexErrors). apply() is on the
        # protocol for VQGAN; for StyleGAN it raises so callers route
        # through apply_batch (batched) or baseline_image (κ=0 cached).
        m, _, _ = _make_manip(num_ws=4, kappa_levels=10)
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        with pytest.raises(NotImplementedError, match="batch >= 2"):
            m.apply(ctx, np.zeros(4, dtype=np.int64))

    def test_baseline_image_returns_origin_without_synthesis(self) -> None:
        m, smoo, _ = _make_manip(num_ws=4, kappa_levels=10)
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        before = len(smoo.calls)
        img = m.baseline_image(ctx)
        # No SMOO call should occur — origin image is cached on the context.
        assert len(smoo.calls) == before
        assert img is ctx.origin_image

    def test_apply_zero_gene_pulls_full_origin(self) -> None:
        m, smoo, _ = _make_manip(num_ws=4, kappa_levels=10)
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        gene = np.zeros((1, 4), dtype=np.int64)  # κ = 0 everywhere
        out = m.apply_batch(ctx, gene)
        assert len(out) == 1
        call = smoo.calls[-1]
        # weights = gene / Q = 0 everywhere → fully origin.
        np.testing.assert_allclose(call["weights"], np.zeros((1, 4)))
        np.testing.assert_array_equal(call["cond"], np.zeros((1, 4), dtype=np.int64))

    def test_apply_max_gene_pulls_full_target(self) -> None:
        m, smoo, _ = _make_manip(num_ws=4, kappa_levels=10)
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        # Max valid gene = kappa_quant_levels = 10 → κ = 1.0.
        gene = np.full((1, 4), ctx.kappa_quant_levels, dtype=np.int64)
        m.apply_batch(ctx, gene)
        call = smoo.calls[-1]
        np.testing.assert_allclose(call["weights"], np.ones((1, 4)))

    def test_apply_mid_gene_interpolates(self) -> None:
        m, smoo, _ = _make_manip(num_ws=4, kappa_levels=20)
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        # gene = 10 / 20 = 0.5.
        gene = np.full((2, 4), 10, dtype=np.int64)
        m.apply_batch(ctx, gene)
        call = smoo.calls[-1]
        np.testing.assert_allclose(call["weights"][0], np.full(4, 0.5))

    def test_apply_batch_returns_n_images(self) -> None:
        m, _, _ = _make_manip(num_ws=4, kappa_levels=10)
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        gts = np.zeros((3, 4), dtype=np.int64)
        gts[1] = 5
        gts[2] = 10
        out = m.apply_batch(ctx, gts)
        assert len(out) == 3
        assert all(isinstance(img, Image.Image) for img in out)

    def test_apply_batch_empty(self) -> None:
        m, smoo, _ = _make_manip()
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        gts = np.zeros((0, m.num_ws), dtype=np.int64)
        out = m.apply_batch(ctx, gts)
        assert out == []
        # No manipulate calls for empty batches.
        assert len(smoo.calls) == 0

    def test_apply_uses_single_w0_and_single_wn(self) -> None:
        m, smoo, _ = _make_manip()
        ctx = m.prepare(Image.new("RGB", (4, 4)), target_class="junco", origin_class="chickadee")
        gts = np.full((2, m.num_ws), 5, dtype=np.int64)
        m.apply_batch(ctx, gts)
        call = smoo.calls[-1]
        assert call["n_w0"] == 1
        assert call["n_wn"] == 1


# ---------------------------------------------------------------------------
# Precompute targets
# ---------------------------------------------------------------------------


class TestPrecomputeTargets:
    def test_target_classes_populate_modal_w(self) -> None:
        m, _, builder = _make_manip()
        m.precompute_targets(("junco", "chickadee"))
        assert builder.modal_calls == ["junco", "chickadee"]
        # No pairs supplied → no origin precheck.
        assert builder.origin_calls == []

    def test_origin_pairs_populate_pairwise(self) -> None:
        m, _, builder = _make_manip()
        m.precompute_targets(
            ("junco",),
            origin_pairs=(("chickadee", "junco"),),
        )
        assert builder.modal_calls == ["junco"]
        assert builder.origin_calls == [("chickadee", "junco")]

    def test_empty_targets_is_noop(self) -> None:
        m, _, builder = _make_manip()
        m.precompute_targets(())
        assert builder.modal_calls == []
        assert builder.origin_calls == []

    def test_attach_modal_builder_is_noop(self) -> None:
        m, _, _ = _make_manip()
        # No-op — StyleGAN owns its builder internally.
        m.attach_modal_builder(object())


# ---------------------------------------------------------------------------
# StyleGANManipulationContext (zero / random / bounds)
# ---------------------------------------------------------------------------


class TestStyleGANManipulationContext:
    def test_zero_genotype_all_zero(self) -> None:
        ctx = StyleGANManipulationContext(
            origin_w=torch.zeros((1, 6, 4)),
            target_w=torch.ones((1, 6, 4)),
            origin_image=Image.new("RGB", (4, 4)),
            origin_class="junco",
            target_class="chickadee",
            kappa_quant_levels=20,
            num_ws=6,
        )
        z = ctx.zero_genotype()
        assert z.dtype == np.int64
        assert z.shape == (6,)
        assert (z == 0).all()

    def test_gene_bounds_is_q_plus_one(self) -> None:
        ctx = StyleGANManipulationContext(
            origin_w=torch.zeros((1, 6, 4)),
            target_w=torch.ones((1, 6, 4)),
            origin_image=Image.new("RGB", (4, 4)),
            origin_class="junco",
            target_class="chickadee",
            kappa_quant_levels=15,
            num_ws=6,
        )
        np.testing.assert_array_equal(
            ctx.gene_bounds,
            np.full(6, 16, dtype=np.int64),
        )

    def test_random_genotype_within_bounds(self) -> None:
        ctx = StyleGANManipulationContext(
            origin_w=torch.zeros((1, 6, 4)),
            target_w=torch.ones((1, 6, 4)),
            origin_image=Image.new("RGB", (4, 4)),
            origin_class="junco",
            target_class="chickadee",
            kappa_quant_levels=10,
            num_ws=6,
        )
        rng = np.random.default_rng(42)
        for _ in range(50):
            g = ctx.random_genotype(rng)
            assert (g >= 0).all()
            assert (g <= 10).all()
            assert g.dtype == np.int64
