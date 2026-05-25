"""Tests for the image-backend dispatch layer.

Covers:

* :class:`ImageConfig` accepting the new ``backend`` field and rejecting
  unknown values.
* The Protocol :class:`ImageBackend` being satisfied by both backends.
* :func:`build_image_backend` dispatching correctly and refusing
  StyleGAN without a SUT / prompt.
* :func:`build_context_meta` falling back to the StyleGAN branch when
  the context shape signals StyleGAN.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.common.seed_context import build_context_meta
from src.manipulator.image.manipulator import (
    ConeFilterConfig,
    ImageConfig,
    ImageManipulator,
    StyleGANConfig,
)
from src.manipulator.image_backend import ImageBackend
from src.manipulator.image_factory import build_image_backend
from src.manipulator.image_stylegan.manipulator import StyleGANImageManipulator
from src.manipulator.image_stylegan.types import StyleGANManipulationContext


# ---------------------------------------------------------------------------
# ImageConfig backend field
# ---------------------------------------------------------------------------


class TestImageConfigBackendField:
    def test_default_is_vqgan(self) -> None:
        cfg = ImageConfig()
        assert cfg.backend == "vqgan_codebook"

    def test_stylegan_xl_accepted(self) -> None:
        cfg = ImageConfig(backend="stylegan_xl")
        assert cfg.backend == "stylegan_xl"

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="image.backend"):
            ImageConfig(backend="bogus")

    def test_stylegan_sub_config_default(self) -> None:
        cfg = ImageConfig()
        assert isinstance(cfg.stylegan, StyleGANConfig)
        assert cfg.stylegan.kappa_quant_levels == 20

    def test_dacite_yaml_with_stylegan_section(self) -> None:
        import dacite

        from src.config import ExperimentConfig

        raw = {
            "name": "stylegan_test",
            "categories": ["a", "b"],
            "image": {
                "backend": "stylegan_xl",
                "stylegan": {
                    "kappa_quant_levels": 8,
                    "target_m": 50,
                    "sut_precheck_max_attempts": 75,
                },
            },
        }
        exp = dacite.from_dict(
            ExperimentConfig, raw,
            config=dacite.Config(cast=[tuple, frozenset]),
        )
        assert exp.image.backend == "stylegan_xl"
        assert exp.image.stylegan.kappa_quant_levels == 8
        assert exp.image.stylegan.target_m == 50
        assert exp.image.stylegan.sut_precheck_max_attempts == 75

    def test_dacite_yaml_without_backend_keeps_legacy_default(self) -> None:
        import dacite

        from src.config import ExperimentConfig

        raw = {
            "name": "legacy_test",
            "categories": ["a", "b"],
            "image": {"patch_ratio": 0.25},
        }
        exp = dacite.from_dict(
            ExperimentConfig, raw,
            config=dacite.Config(cast=[tuple, frozenset]),
        )
        assert exp.image.backend == "vqgan_codebook"
        assert exp.image.patch_ratio == 0.25
        # Cone filter default still disabled.
        assert not exp.image.cone_filter.enabled


# ---------------------------------------------------------------------------
# ImageBackend protocol satisfaction
# ---------------------------------------------------------------------------


class TestImageBackendProtocol:
    """Verify both concrete backends satisfy the :class:`ImageBackend` protocol.

    The protocol is non-runtime-checkable (no
    ``@runtime_checkable``-required isinstance), but structurally
    typing.Protocol satisfaction at type-check time is what matters
    here. We exercise the methods to make sure their signatures match.
    """

    def test_image_manipulator_has_required_methods(self) -> None:
        # Construct with a minimal stub codec so we don't load VQGAN.
        from tests.test_image_manipulator import _FakeCodec, _stripe_codebook
        from src.manipulator.image.types import CodeGrid

        codec = _FakeCodec(
            _stripe_codebook(8),
            CodeGrid(np.array([[0, 1], [2, 3]], dtype=np.int64)),
        )
        m = ImageManipulator(codec, ImageConfig())
        # Required methods present.
        assert callable(getattr(m, "prepare", None))
        assert callable(getattr(m, "apply", None))
        assert callable(getattr(m, "apply_batch", None))
        assert callable(getattr(m, "attach_modal_builder", None))
        assert callable(getattr(m, "precompute_targets", None))

    def test_stylegan_manipulator_has_required_methods(self) -> None:
        from tests.test_stylegan_manipulator import (
            _FakeClassTargetBuilder,
            _FakeSMOOManipulator,
        )

        smoo = _FakeSMOOManipulator(num_ws=4)
        builder = _FakeClassTargetBuilder(num_ws=4)
        cfg = ImageConfig(backend="stylegan_xl")
        m = StyleGANImageManipulator(
            smoo_manipulator=smoo,
            class_target_builder=builder,
            config=cfg,
            device=torch.device("cpu"),
            categories=("a",),
        )
        assert callable(getattr(m, "prepare", None))
        assert callable(getattr(m, "apply", None))
        assert callable(getattr(m, "apply_batch", None))
        assert callable(getattr(m, "attach_modal_builder", None))
        assert callable(getattr(m, "precompute_targets", None))


# ---------------------------------------------------------------------------
# build_image_backend dispatch
# ---------------------------------------------------------------------------


class TestBuildImageBackendDispatch:
    def test_stylegan_without_sut_raises(self) -> None:
        cfg = ImageConfig(backend="stylegan_xl")
        with pytest.raises(ValueError, match="sut"):
            build_image_backend(
                image_config=cfg,
                device="cpu",
                categories=("a",),
                sut=None,
                prompt="x",
            )

    def test_stylegan_without_prompt_raises(self) -> None:
        cfg = ImageConfig(backend="stylegan_xl")

        class _SutMock:
            scorer = None

        with pytest.raises(ValueError, match="prompt"):
            build_image_backend(
                image_config=cfg,
                device="cpu",
                categories=("a",),
                sut=_SutMock(),
                prompt=None,
            )

    def test_unknown_backend_raises(self) -> None:
        # ImageConfig validates in __post_init__; we have to construct
        # by bypassing __post_init__ to get a malformed value into
        # build_image_backend.
        cfg = ImageConfig()
        object.__setattr__(cfg, "backend", "nonsense")
        with pytest.raises(ValueError, match="Unknown image.backend"):
            build_image_backend(image_config=cfg, device="cpu")


# ---------------------------------------------------------------------------
# build_context_meta dispatches on context type
# ---------------------------------------------------------------------------


class TestBuildContextMeta:
    def test_vqgan_branch(self) -> None:
        from conftest import FakeCompositeTextManipulator, FakeImageManipulator
        from src.manipulator.vlm_manipulator import VLMManipulator

        vlm = VLMManipulator(FakeImageManipulator(), FakeCompositeTextManipulator())
        vlm.prepare(Image.new("RGB", (4, 4)), "The quick brown fox")
        meta = build_context_meta(vlm)
        assert meta["image_backend"] == "vqgan_codebook"
        assert "image_patch_positions" in meta
        assert "image_original_codes" in meta
        assert "image_candidates" in meta
        # Text meta still present.
        assert "text_op_order" in meta

    def test_stylegan_branch(self) -> None:
        from conftest import FakeCompositeTextManipulator
        from src.manipulator.vlm_manipulator import VLMManipulator
        from tests.test_stylegan_manipulator import (
            _FakeClassTargetBuilder,
            _FakeSMOOManipulator,
        )

        smoo = _FakeSMOOManipulator(num_ws=4)
        builder = _FakeClassTargetBuilder(num_ws=4)
        cfg = ImageConfig(
            backend="stylegan_xl",
            stylegan=StyleGANConfig(kappa_quant_levels=12),
        )
        image_manip = StyleGANImageManipulator(
            smoo_manipulator=smoo,
            class_target_builder=builder,
            config=cfg,
            device=torch.device("cpu"),
            categories=("junco", "chickadee"),
        )
        vlm = VLMManipulator(image_manip, FakeCompositeTextManipulator())
        vlm.prepare(
            Image.new("RGB", (4, 4)),
            "The quick brown fox",
            target_class="junco",
            origin_class="chickadee",
        )
        meta = build_context_meta(vlm)
        assert meta["image_backend"] == "stylegan_xl"
        assert meta["image_num_ws"] == 4
        assert meta["image_kappa_quant_levels"] == 12
        assert meta["image_target_class"] == "junco"
        assert meta["image_origin_class"] == "chickadee"
        assert meta["image_candidate_strategy"] == "stylegan_interp"
        # Image-specific VQGAN fields must be absent.
        assert "image_patch_positions" not in meta
        assert "image_original_codes" not in meta
        # Text meta still present.
        assert "text_op_order" in meta


# ---------------------------------------------------------------------------
# VLMManipulator accepts both backend types
# ---------------------------------------------------------------------------


class TestVLMManipulatorAcceptsStyleGAN:
    def test_prepare_apply_with_stylegan_backend(self) -> None:
        from conftest import FakeCompositeTextManipulator
        from src.manipulator.vlm_manipulator import VLMManipulator
        from tests.test_stylegan_manipulator import (
            _FakeClassTargetBuilder,
            _FakeSMOOManipulator,
        )

        smoo = _FakeSMOOManipulator(num_ws=3)
        builder = _FakeClassTargetBuilder(num_ws=3)
        cfg = ImageConfig(
            backend="stylegan_xl",
            stylegan=StyleGANConfig(kappa_quant_levels=10),
        )
        image_manip = StyleGANImageManipulator(
            smoo_manipulator=smoo,
            class_target_builder=builder,
            config=cfg,
            device=torch.device("cpu"),
            categories=("junco", "chickadee"),
        )
        vlm = VLMManipulator(image_manip, FakeCompositeTextManipulator())
        vlm.prepare(
            Image.new("RGB", (4, 4)),
            "The quick brown fox",
            target_class="junco",
            origin_class="chickadee",
        )

        # Genotype dim = num_ws + text_dim (2 stub tokens).
        assert vlm.image_dim == 3
        assert vlm.text_dim == 2
        assert vlm.genotype_dim == 5

        # Manipulate with all-zero genotype (κ = 0 everywhere → origin image).
        weights = vlm.zero_genotype().reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)
        assert len(images) == 1
        assert len(texts) == 1
        assert isinstance(images[0], Image.Image)
        # Text unchanged on zero genotype.
        assert texts[0] == "The quick brown fox"
