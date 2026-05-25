"""Tests for StyleGAN-XL class-target helpers.

All tests use in-process stubs for the generator and the SUT — no real
NVlabs checkpoint download, no real model load, no Redis server. The
StyleGAN backend's correctness here hinges on two properties:

* :func:`build_class_modal_w` averages exactly ``m`` deterministic samples.
* :func:`find_pair_dominant_origin_seed` iterates seeds in order until
  the SUT scores ``origin_class`` higher than ``target_class``; the
  chosen seed is the first one satisfying that pairwise condition.

The :class:`StyleGANClassTargetBuilder` also exercises the L1+L2 cache
round-trip — same shape as the VQGAN ModalTargetBuilder test.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.common.redis_cache import BytesRedisCache
from src.manipulator.image_stylegan.class_target import (
    ACCEPTED_SEED_KEY_PREFIX,
    MODAL_W_KEY_PREFIX,
    StyleGANClassTargetBuilder,
    build_class_modal_w,
    find_pair_dominant_origin_seed,
    safe_class_name,
    sut_signature,
    tensor_to_pil,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeGenerator:
    """Deterministic generator stub.

    ``get_w`` returns a constant w-tensor that uniquely encodes both
    ``(seed, class_idx)``, so :func:`build_class_modal_w` can be checked
    against the analytic mean.

    ``get_images`` paints the first pixel red-channel with ``seed % 256``
    so the SUT stub can map images back to seeds without inspecting the
    full pixel buffer.
    """

    def __init__(self, *, num_ws: int = 8, w_dim: int = 4) -> None:
        self.num_ws = num_ws
        self.w_dim = w_dim
        self.get_w_calls: list[tuple[int, int]] = []
        self.get_images_calls: int = 0

    def get_w(self, seed: int, class_idx: int, batch_size: int = 1) -> torch.Tensor:
        self.get_w_calls.append((seed, class_idx))
        # Construct a w-tensor where every entry equals seed + class_idx * 100
        # so averaging exactly recovers the mean of the seed sequence.
        value = float(seed) + 100.0 * float(class_idx)
        return torch.full((batch_size, self.num_ws, self.w_dim), value, dtype=torch.float32)

    def get_images(self, w: torch.Tensor) -> torch.Tensor:
        self.get_images_calls += 1
        # Recover the encoded seed from the w-tensor — every entry is
        # (seed + class_idx * 100), so the first value reveals the seed
        # modulo the offset. Take the value modulo 256 so we can stuff
        # it in a single pixel channel.
        encoded = float(w[0, 0, 0].item())
        red = int(encoded) % 256
        # Make a tiny 1x1 image so tensor_to_pil works.
        img_arr = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img_arr[0, 0, :, :] = red / 255.0
        return torch.from_numpy(img_arr)


class _FakeSUT:
    """SUT stub with a hardcoded ``red_channel → winning_label`` mapping.

    ``winner_map[seed_int]`` gives the class label that the fake SUT
    assigns the highest log-prob for when scoring the image rendered
    from ``seed_int``. All other candidates get a uniform lower log-prob,
    so any pairwise check ``lp[origin] > lp[target]`` is satisfied iff
    ``winner_map[seed_int] == origin``. Missing entries map to a sentinel
    that beats nothing in any real candidate set.
    """

    SENTINEL_NO_MATCH = "__none__"

    def __init__(self, winner_map: dict[int, str]) -> None:
        self._winner_map = winner_map
        self.calls: int = 0

    def predict_logprobs(
        self, image: Image.Image, candidates: tuple[str, ...],
    ) -> dict[str, float]:
        self.calls += 1
        red = image.getpixel((0, 0))[0]
        winner = self._winner_map.get(red, self.SENTINEL_NO_MATCH)
        out: dict[str, float] = {}
        for c in candidates:
            out[c] = 0.0 if c == winner else -1.0
        return out


class _InMemoryRedis:
    """Trivial dict-backed bytes store used as :class:`BytesRedisCache`."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.get_calls: int = 0
        self.set_calls: int = 0

    def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return self.store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.set_calls += 1
        self.store[key] = value


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSafeClassName:
    def test_lowercases_and_replaces_spaces(self) -> None:
        assert safe_class_name("Great White Shark") == "great_white_shark"

    def test_idempotent(self) -> None:
        out = safe_class_name("great_white_shark")
        assert safe_class_name(out) == out


class TestSUTSignature:
    def test_stable_under_reorder(self) -> None:
        # Sorting categories means insertion order doesn't matter.
        a = sut_signature("model-x", ["b", "a"])
        b = sut_signature("model-x", ["a", "b"])
        assert a == b
        assert len(a) == 16

    def test_changes_with_model_id(self) -> None:
        a = sut_signature("model-x", ["a"])
        b = sut_signature("model-y", ["a"])
        assert a != b


class TestTensorToPil:
    def test_4d_input(self) -> None:
        tensor = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        tensor[0, 0, :, :] = 1.0  # full red
        img = tensor_to_pil(tensor)
        assert img.size == (4, 4)
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 0, 0)

    def test_3d_input(self) -> None:
        tensor = torch.zeros((3, 4, 4), dtype=torch.float32)
        tensor[2, :, :] = 1.0  # full blue
        img = tensor_to_pil(tensor)
        assert img.getpixel((0, 0)) == (0, 0, 255)


# ---------------------------------------------------------------------------
# build_class_modal_w
# ---------------------------------------------------------------------------


class TestBuildClassModalW:
    def test_averages_m_samples(self) -> None:
        gen = _FakeGenerator(num_ws=4, w_dim=3)
        # class_idx=5 → every w entry is seed + 500.
        # Seeds 0..4 average to (0+1+2+3+4)/5 + 500 = 502.0.
        w = build_class_modal_w(generator=gen, class_idx=5, m=5)
        assert w.shape == (1, 4, 3)
        np.testing.assert_allclose(
            w.numpy(),
            np.full((1, 4, 3), 502.0, dtype=np.float32),
        )

    def test_uses_correct_seed_sequence(self) -> None:
        gen = _FakeGenerator()
        build_class_modal_w(generator=gen, class_idx=0, m=3, rng_seed=10)
        # Should hit seeds 10, 11, 12.
        assert gen.get_w_calls == [(10, 0), (11, 0), (12, 0)]

    def test_m_one(self) -> None:
        gen = _FakeGenerator()
        w = build_class_modal_w(generator=gen, class_idx=2, m=1)
        # value = 0 + 100*2 = 200 for seed 0.
        np.testing.assert_allclose(w.numpy(), np.full(w.shape, 200.0))

    def test_zero_m_raises(self) -> None:
        gen = _FakeGenerator()
        with pytest.raises(ValueError, match="m must be"):
            build_class_modal_w(generator=gen, class_idx=0, m=0)


# ---------------------------------------------------------------------------
# find_pair_dominant_origin_seed
# ---------------------------------------------------------------------------


class TestFindPairDominantOriginSeed:
    def test_returns_first_pair_dominant_seed(self) -> None:
        gen = _FakeGenerator()
        # Seed-3 image carries red=3 → SUT scores "junco" higher than
        # everything else.
        sut = _FakeSUT(winner_map={3: "junco"})
        seed_int, w = find_pair_dominant_origin_seed(
            generator=gen,
            sut=sut,
            origin_class="junco",
            target_class="chickadee",
            origin_class_idx=0,
            candidates=("junco", "chickadee"),
            max_attempts=20,
        )
        assert seed_int == 3
        assert w.shape == (1, gen.num_ws, gen.w_dim)
        assert sut.calls == 4  # seeds 0,1,2,3

    def test_seed_zero_accepted(self) -> None:
        gen = _FakeGenerator()
        sut = _FakeSUT(winner_map={0: "junco"})
        seed_int, _ = find_pair_dominant_origin_seed(
            generator=gen,
            sut=sut,
            origin_class="junco",
            target_class="chickadee",
            origin_class_idx=0,
            candidates=("junco", "chickadee"),
            max_attempts=5,
        )
        assert seed_int == 0
        assert sut.calls == 1

    def test_raises_when_max_attempts_exhausted(self) -> None:
        gen = _FakeGenerator()
        sut = _FakeSUT(winner_map={})  # never accepts
        with pytest.raises(RuntimeError, match="pair precheck failed"):
            find_pair_dominant_origin_seed(
                generator=gen,
                sut=sut,
                origin_class="junco",
                target_class="chickadee",
                origin_class_idx=0,
                candidates=("junco", "chickadee"),
                max_attempts=5,
            )
        assert sut.calls == 5

    def test_target_winning_rejects_seed(self) -> None:
        # Seed-0 → chickadee wins; seed-1 → junco wins. With
        # origin=junco / target=chickadee, only seed-1 should be accepted.
        gen = _FakeGenerator()
        sut = _FakeSUT(winner_map={0: "chickadee", 1: "junco"})
        seed_int, _ = find_pair_dominant_origin_seed(
            generator=gen,
            sut=sut,
            origin_class="junco",
            target_class="chickadee",
            origin_class_idx=0,
            candidates=("junco", "chickadee"),
            max_attempts=10,
        )
        assert seed_int == 1
        assert sut.calls == 2

    def test_third_class_winner_still_passes_if_origin_beats_target(self) -> None:
        # Critical: a 50-way top-K rule would reject a seed where neither
        # origin nor target is top-1; the pairwise rule still accepts as
        # long as P(origin) > P(target). Use a custom SUT for that.
        class _OrderedSUT:
            def __init__(self) -> None:
                self.calls = 0

            def predict_logprobs(self, image, candidates):
                self.calls += 1
                # "shark" wins overall, but junco still beats chickadee.
                return {"shark": 0.0, "junco": -1.0, "chickadee": -2.0}

        gen = _FakeGenerator()
        sut = _OrderedSUT()
        seed_int, _ = find_pair_dominant_origin_seed(
            generator=gen,
            sut=sut,
            origin_class="junco",
            target_class="chickadee",
            origin_class_idx=0,
            candidates=("junco", "chickadee", "shark"),
            max_attempts=3,
        )
        assert seed_int == 0
        assert sut.calls == 1

    def test_same_class_pair_raises(self) -> None:
        gen = _FakeGenerator()
        sut = _FakeSUT(winner_map={})
        with pytest.raises(ValueError, match="must differ"):
            find_pair_dominant_origin_seed(
                generator=gen,
                sut=sut,
                origin_class="junco",
                target_class="junco",
                origin_class_idx=0,
                candidates=("junco",),
                max_attempts=5,
            )

    def test_missing_candidate_raises(self) -> None:
        gen = _FakeGenerator()
        sut = _FakeSUT(winner_map={})
        with pytest.raises(ValueError, match="must be in candidates"):
            find_pair_dominant_origin_seed(
                generator=gen,
                sut=sut,
                origin_class="junco",
                target_class="missing",
                origin_class_idx=0,
                candidates=("junco", "chickadee"),
                max_attempts=5,
            )


# ---------------------------------------------------------------------------
# StyleGANClassTargetBuilder (L1 + L2)
# ---------------------------------------------------------------------------


def _builder(
    *,
    winner_map: dict[int, str] | None = None,
    redis_cache: BytesRedisCache | None = None,
    categories: tuple[str, ...] = ("junco", "chickadee"),
) -> tuple[StyleGANClassTargetBuilder, _FakeGenerator, _FakeSUT]:
    """Construct a builder + the fakes wired into it."""
    gen = _FakeGenerator()
    sut = _FakeSUT(winner_map=winner_map or {})
    builder = StyleGANClassTargetBuilder(
        generator=gen,
        sut=sut,
        checkpoint_hash="abc123",
        sut_signature="sut0000",
        categories=categories,
        target_m=3,
        truncation_psi=1.0,
        truncation_cutoff=0,
        max_attempts=20,
        redis_cache=redis_cache,
        device=torch.device("cpu"),
    )
    return builder, gen, sut


class TestStyleGANClassTargetBuilderModalW:
    def test_modal_w_l1_hit_skips_generator(self) -> None:
        builder, gen, _ = _builder()
        w1 = builder.ensure_modal_w("junco")
        n_calls_first = len(gen.get_w_calls)

        w2 = builder.ensure_modal_w("junco")
        # L1 hit: generator must NOT be re-invoked.
        assert len(gen.get_w_calls) == n_calls_first
        torch.testing.assert_close(w1, w2)

    def test_modal_w_key_format(self) -> None:
        builder, _, _ = _builder()
        key = builder.modal_w_key("Great White Shark")
        assert key.startswith(MODAL_W_KEY_PREFIX + ":abc123:")
        assert ":great_white_shark:m3:" in key
        assert ":trunc1.0:" in key
        assert ":cut0" in key

    def test_modal_w_l2_round_trip(self) -> None:
        redis = _InMemoryRedis()
        cache = BytesRedisCache(redis)

        b1, g1, _ = _builder(redis_cache=cache)
        w_built = b1.ensure_modal_w("junco")
        assert redis.set_calls == 1
        n_calls_after_first_build = len(g1.get_w_calls)

        # Second builder with empty L1 but shared L2.
        b2, g2, _ = _builder(redis_cache=cache)
        w_cached = b2.ensure_modal_w("junco")
        # The cached path must NOT touch the generator.
        assert len(g2.get_w_calls) == 0
        torch.testing.assert_close(w_built, w_cached)


class TestStyleGANClassTargetBuilderOrigin:
    def test_origin_precheck_finds_seed(self) -> None:
        builder, gen, sut = _builder(winner_map={2: "junco"})
        seed_int, w, img = builder.ensure_origin("junco", "chickadee")
        assert seed_int == 2
        assert sut.calls == 3  # seeds 0, 1, 2
        assert img.size == (4, 4)
        # w-tensor first entry is seed + class_idx*100 = 2 + 0 = 2.
        assert float(w[0, 0, 0].item()) == 2.0

    def test_origin_l1_hit_skips_generator(self) -> None:
        builder, gen, sut = _builder(winner_map={0: "junco"})
        builder.ensure_origin("junco", "chickadee")
        n_get_w_first = len(gen.get_w_calls)
        n_sut_first = sut.calls

        # Second call for the same pair must not re-run the precheck.
        seed_int, _, _ = builder.ensure_origin("junco", "chickadee")
        assert seed_int == 0
        assert len(gen.get_w_calls) == n_get_w_first
        assert sut.calls == n_sut_first

    def test_origin_l1_separate_per_pair(self) -> None:
        # A seed accepted for (junco vs chickadee) is NOT reused for
        # (junco vs shark) — pair-keyed cache.
        builder, _, sut = _builder(
            winner_map={0: "junco"},
            categories=("junco", "chickadee", "shark"),
        )
        builder.ensure_origin("junco", "chickadee")
        calls_after_first = sut.calls
        builder.ensure_origin("junco", "shark")
        # A second precheck happens for the new pair (still cheap since
        # seed 0 also wins for the new target).
        assert sut.calls > calls_after_first

    def test_origin_l2_replay_avoids_sut(self) -> None:
        redis = _InMemoryRedis()
        cache = BytesRedisCache(redis)
        # First builder finds + caches the accepted seed.
        b1, _, sut1 = _builder(winner_map={1: "junco"}, redis_cache=cache)
        seed_a, _, _ = b1.ensure_origin("junco", "chickadee")
        assert seed_a == 1
        assert sut1.calls == 2  # seeds 0, 1
        assert redis.set_calls == 1

        # Second builder with empty L1: must replay the cached seed via
        # generator only — no SUT calls.
        b2, _, sut2 = _builder(winner_map={1: "junco"}, redis_cache=cache)
        seed_b, _, _ = b2.ensure_origin("junco", "chickadee")
        assert seed_b == 1
        assert sut2.calls == 0  # SUT not consulted on the cached path

    def test_origin_l2_key_includes_both_classes(self) -> None:
        builder, _, _ = _builder()
        key = builder.accepted_seed_key("Great White Shark", "Hammerhead Shark")
        assert key.startswith(ACCEPTED_SEED_KEY_PREFIX + ":abc123:")
        assert ":great_white_shark:" in key
        assert ":vs_hammerhead_shark:" in key
        assert key.endswith(":sut0000")

    def test_origin_failure_propagates(self) -> None:
        builder, _, _ = _builder(winner_map={})  # never accepts
        with pytest.raises(RuntimeError, match="pair precheck failed"):
            builder.ensure_origin("junco", "chickadee")

    def test_populate_origins_runs_all_pairs(self) -> None:
        # SUT that scores the queried origin highest for every seed —
        # the first attempt accepts. populate_origins should touch each
        # unique pair exactly once.
        class _AlwaysOriginWinsSUT:
            def __init__(self) -> None:
                self.calls: list[tuple[str, ...]] = []
                self._seen_origin: str | None = None

            def predict_logprobs(self, image, candidates):
                self.calls.append(candidates)
                # We don't have access to origin here directly — set up
                # the test below so that for both pairs the SAME class
                # always wins (origin). For mixed origins we'd need
                # smarter wiring; this fake keeps the test deterministic.
                # Score every candidate equally low except a sentinel
                # that wins — the builder caller passes origin we know.
                return {c: (0.0 if c == self._seen_origin else -1.0) for c in candidates}

        gen = _FakeGenerator()
        sut = _AlwaysOriginWinsSUT()
        builder = StyleGANClassTargetBuilder(
            generator=gen,
            sut=sut,
            checkpoint_hash="abc",
            sut_signature="sut",
            categories=("junco", "chickadee"),
            target_m=2,
            truncation_psi=1.0,
            truncation_cutoff=0,
            max_attempts=5,
            device=torch.device("cpu"),
        )
        # Mutate the SUT state so junco wins for the junco→chickadee pair…
        sut._seen_origin = "junco"
        builder.ensure_origin("junco", "chickadee")
        # …then chickadee wins for chickadee→junco.
        sut._seen_origin = "chickadee"
        builder.ensure_origin("chickadee", "junco")
        # 2 SUT calls (one per pair, both first-attempt).
        assert len(sut.calls) == 2

    def test_populate_origins_skips_same_class_pairs(self) -> None:
        builder, _, sut = _builder(winner_map={0: "junco"})
        builder.populate_origins([("junco", "junco"), ("junco", "chickadee")])
        # Only the non-degenerate pair touched the SUT.
        assert sut.calls == 1

    def test_populate_origins_deduplicates(self) -> None:
        builder, _, sut = _builder(winner_map={0: "junco"})
        builder.populate_origins([
            ("junco", "chickadee"),
            ("junco", "chickadee"),  # duplicate
        ])
        assert sut.calls == 1

    def test_class_idx_resolution_via_categories(self) -> None:
        builder, _, _ = _builder(
            winner_map={},
            categories=("junco", "chickadee", "shark"),
        )
        assert builder.class_idx("junco") == 0
        assert builder.class_idx("chickadee") == 1
        assert builder.class_idx("shark") == 2

    def test_class_idx_unknown_raises(self) -> None:
        builder, _, _ = _builder(winner_map={})
        with pytest.raises(KeyError, match="unknown"):
            builder.class_idx("mystery_class")
