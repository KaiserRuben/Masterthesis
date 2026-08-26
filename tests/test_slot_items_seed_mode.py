"""Exp-105 ``slot_items`` seed mode: config validation + seed generation.

No model, no network, no ImageNet cache — slot items are fully specified
in the config, so the generator only resolves paths and instantiates the
carrier sentences. The SUT / data_source arguments are passed as ``None``
(the generator must not touch them).
"""
import json
from pathlib import Path

import dacite
import pytest
import yaml
from PIL import Image

import src.common.pipeline_bootstrap as pb
from experiments.runners.run_boundary_test import _DACITE_CONFIG, load_config
from src.common.slot_items_seed_generator import (
    build_seed_triples,
    instantiate,
    item_candidates,
    item_pair,
    resolve_item_image,
    slot_items_seeds,
)
from src.config import (
    ExperimentConfig,
    ImageConfig,
    SeedConfig,
    SlotItem,
    SlotItemsConfig,
    apply_modality,
)
from src.evolutionary.vlm_boundary_tester import build_stats, effective_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]


def _item(**over) -> SlotItem:
    kwargs = dict(
        image=Path("experiments/data/exp105_seeds/house_green.png"),
        template="The house is {slot}.",
        fillers=("green", "blue"),
        pair=("green", "blue"),
    )
    kwargs.update(over)
    return SlotItem(**kwargs)


def _tiny_png(tmp_path: Path, name: str = "seed.png") -> Path:
    p = tmp_path / name
    Image.new("RGB", (12, 10), (10, 200, 40)).save(p)
    return p


# ---------------------------------------------------------------------------
# Config parsing + validation
# ---------------------------------------------------------------------------


class TestSlotItemValidation:
    def test_valid_item_parses(self) -> None:
        it = _item()
        assert it.fillers == ("green", "blue")
        assert it.pair == ("green", "blue")

    def test_template_without_slot_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must contain '\{slot\}'"):
            _item(template="The house is green.")

    def test_pair_entry_not_in_fillers_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in `fillers`"):
            _item(fillers=("green", "blue"), pair=("green", "red"))

    def test_pair_must_have_exactly_two_entries(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 `pair` entries"):
            _item(fillers=("green", "blue", "red"), pair=("green", "blue", "red"))
        with pytest.raises(ValueError, match="exactly 2 `pair` entries"):
            _item(pair=("green",))

    def test_pair_entries_must_differ(self) -> None:
        with pytest.raises(ValueError, match="identical `pair` entries"):
            _item(pair=("green", "green"))

    def test_empty_fillers_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty `fillers`"):
            _item(fillers=(), pair=("green", "blue"))

    def test_duplicate_fillers_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate fillers"):
            _item(fillers=("green", "blue", "green"))

    def test_missing_image_file_is_not_a_load_error(self) -> None:
        """Existence is a seed-generation concern; config load stays IO-free."""
        _item(image=Path("/nonexistent/nope.png"))


class TestSeedConfigDispatch:
    def test_slot_items_mode_requires_block(self) -> None:
        with pytest.raises(ValueError, match="requires a seeds.slot_items"):
            SeedConfig(mode="slot_items")

    def test_slot_items_mode_ok(self) -> None:
        sc = SeedConfig(
            mode="slot_items", slot_items=SlotItemsConfig(items=(_item(),)),
        )
        assert sc.mode == "slot_items"
        assert len(sc.slot_items.items) == 1

    def test_rejects_conflicting_blocks(self) -> None:
        from src.config import GapFilterConfig

        with pytest.raises(ValueError, match="drop one"):
            SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(_item(),)),
                gap_filter=GapFilterConfig(),
            )

    def test_other_modes_reject_a_stray_slot_items_block(self) -> None:
        with pytest.raises(ValueError, match="drop one"):
            SeedConfig(
                mode="gap_filter",
                slot_items=SlotItemsConfig(items=(_item(),)),
            )

    def test_empty_items_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one item"):
            SlotItemsConfig(items=())

    def test_unknown_mode_message_lists_slot_items(self) -> None:
        with pytest.raises(ValueError, match="'slot_items'"):
            SeedConfig(mode="bogus")

    def test_existing_modes_unaffected(self) -> None:
        assert SeedConfig().mode == "gap_filter"
        assert SeedConfig().gap_filter is not None  # default filled in
        assert SeedConfig().slot_items is None


class TestYamlParsing:
    def test_parses_from_yaml_dict(self) -> None:
        exp = dacite.from_dict(
            ExperimentConfig,
            yaml.safe_load(
                """
                seeds:
                  mode: slot_items
                  slot_items:
                    items:
                      - image: experiments/data/exp105_seeds/house_green.png
                        template: "The house is {slot}."
                        fillers: [green, blue]
                        pair: [green, blue]
                """
            ),
            config=_DACITE_CONFIG,
        )
        (item,) = exp.seeds.slot_items.items
        assert isinstance(item.image, Path)
        assert item.fillers == ("green", "blue")

    def test_shipped_exp105_config_loads(self) -> None:
        """Acceptance: the real config loader accepts the Exp-105 step-1 YAML.

        The seed image does not have to exist at load time.
        """
        cfg_path = REPO_ROOT / "configs/Exp-105/exp105_step1_house_qwen_raw.yaml"
        exp = apply_modality(load_config(yaml.safe_load(cfg_path.read_text())))
        assert exp.seeds.mode == "slot_items"
        assert exp.modality == "image_only"
        (item,) = exp.seeds.slot_items.items
        assert item_candidates(item) == (
            "The house is green.", "The house is blue.",
        )

    def test_shipped_exp105_step4_config_loads(self) -> None:
        cfg_path = REPO_ROOT / "configs/Exp-105/exp105_step4_gamma0_qwen_raw.yaml"
        exp = apply_modality(load_config(yaml.safe_load(cfg_path.read_text())))
        assert exp.seeds.mode == "slot_items"
        assert len(exp.seeds.slot_items.items) == 3


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiate_is_literal_replace(self) -> None:
        assert instantiate("a {slot} b", "X") == "a X b"
        # Other braces are left alone (no str.format).
        assert instantiate('{"k": {slot}}', "1") == '{"k": 1}'

    def test_candidates_follow_filler_order(self) -> None:
        it = _item(fillers=("blue", "green", "red"), pair=("red", "blue"))
        assert item_candidates(it) == (
            "The house is blue.",
            "The house is green.",
            "The house is red.",
        )

    def test_pair_maps_fillers_to_sentences(self) -> None:
        it = _item(fillers=("blue", "green", "red"), pair=("red", "blue"))
        assert item_pair(it) == ("The house is red.", "The house is blue.")


class TestGenerator:
    def test_one_seed_per_item_with_pair_and_candidates(self, tmp_path) -> None:
        png = _tiny_png(tmp_path)
        exp = ExperimentConfig(
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(
                    _item(image=png),
                    _item(
                        image=png,
                        template="The person in the picture is {slot}.",
                        fillers=("a man", "a woman", "a child"),
                        pair=("a woman", "a man"),
                    ),
                )),
            ),
        )
        seeds = slot_items_seeds(None, exp, None)

        assert len(seeds) == 2
        first, second = seeds
        assert first.class_a == "The house is green."
        assert first.class_b == "The house is blue."
        assert first.metadata["candidates"] == [
            "The house is green.", "The house is blue.",
        ]
        assert first.metadata["image_path"] == str(png)
        assert first.metadata["item_idx"] == 0
        assert first.image.size == (12, 10)
        assert first.image.mode == "RGB"

        # pair order follows `pair`, candidates follow `fillers`
        assert second.class_a == "The person in the picture is a woman."
        assert second.class_b == "The person in the picture is a man."
        assert second.metadata["candidates"] == [
            "The person in the picture is a man.",
            "The person in the picture is a woman.",
            "The person in the picture is a child.",
        ]
        assert second.metadata["pair_fillers"] == ["a woman", "a man"]
        assert second.metadata["template"] == "The person in the picture is {slot}."

    def test_missing_image_fails_fast_with_paths(self, tmp_path) -> None:
        exp = ExperimentConfig(
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(
                    _item(image=tmp_path / "does_not_exist.png"),
                )),
            ),
        )
        with pytest.raises(FileNotFoundError, match="does_not_exist.png"):
            slot_items_seeds(None, exp, None)

    def test_relative_path_resolves_against_repo_root(self) -> None:
        rel = Path("experiments/data/exp105_seeds/house_green.png")
        if not (REPO_ROOT / rel).is_file():
            pytest.skip("Exp-105 house seed image not present")
        assert resolve_item_image(rel) == REPO_ROOT / rel

    def test_wrong_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="seeds.mode='slot_items'"):
            slot_items_seeds(None, ExperimentConfig(), None)

    def test_cone_filter_rejected(self, tmp_path) -> None:
        from src.manipulator.image.manipulator import ConeFilterConfig

        exp = ExperimentConfig(
            image=ImageConfig(cone_filter=ConeFilterConfig(enabled=True)),
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(
                    items=(_item(image=_tiny_png(tmp_path)),),
                ),
            ),
        )
        with pytest.raises(ValueError, match="cone_filter"):
            slot_items_seeds(None, exp, None)

    def test_build_seed_triples_is_pure(self) -> None:
        img = Image.new("RGB", (4, 4))
        (seed,) = build_seed_triples([_item()], [img])
        assert seed.image is img
        assert seed.class_a == "The house is green."


class TestBootstrapDispatch:
    def test_prepare_pipeline_seeds_routes_to_slot_items(self, monkeypatch) -> None:
        fake = [object()]
        monkeypatch.setattr(
            pb, "slot_items_seeds", lambda sut, cfg, ds: fake, raising=False,
        )
        exp = ExperimentConfig(
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(_item(),)),
            ),
        )

        class _C:
            sut = None
            data_source = None

        assert pb.prepare_pipeline_seeds(_C(), exp) is fake


# ---------------------------------------------------------------------------
# Tester wiring: per-seed candidates drive scoring scope + answer suffix
# ---------------------------------------------------------------------------


class TestTesterWiring:
    def test_effective_candidates_reads_metadata(self, tmp_path) -> None:
        exp = ExperimentConfig(
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(
                    _item(
                        image=_tiny_png(tmp_path),
                        fillers=("green", "blue", "red"),
                        pair=("green", "blue"),
                    ),
                )),
            ),
        )
        (seed,) = slot_items_seeds(None, exp, None)
        cands = effective_candidates(seed)
        assert cands == (
            "The house is green.", "The house is blue.", "The house is red.",
        )
        # class_a / class_b must be locatable inside the scored list —
        # this is what the tester uses for target_classes.
        assert (cands.index(seed.class_a), cands.index(seed.class_b)) == (0, 1)

    def test_effective_candidates_none_for_other_modes(self) -> None:
        from src.config import SeedTriple

        plain = SeedTriple(image=Image.new("RGB", (4, 4)), class_a="a", class_b="b")
        assert effective_candidates(plain) is None
        grounding = SeedTriple(
            image=Image.new("RGB", (4, 4)), class_a="a", class_b="b",
            metadata={"prompt_template": "Locate the dog."},
        )
        assert effective_candidates(grounding) is None

    def test_end_to_end_loop_scores_candidate_sentences(self, tmp_path) -> None:
        """Full (fake-SUT) search loop: prompt + scored categories = candidates."""
        import numpy as np
        import pandas as pd
        import torch

        from conftest import FakeCompositeTextManipulator, FakeImageManipulator
        from src.evolutionary.vlm_boundary_tester import VLMBoundaryTester
        from src.manipulator.vlm_manipulator import VLMManipulator
        from src.objectives import (
            CriterionCollection,
            MatrixDistance,
            TargetedBalance,
        )
        from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer

        calls: list[tuple[str, tuple[str, ...]]] = []

        class RecordingSUT:
            text_embedder = None
            device_str = "cpu"
            last_call_cached = False
            cache_stats = {"hits": 0, "misses": 0}

            def process_input(self, image, text=None, categories=None):
                calls.append((text, tuple(categories)))
                # Vary with the (fake) manipulated image so the Pareto front
                # is non-degenerate — a constant front trips AGE-MOEA-II.
                r = image.getpixel((0, 0))[0] / 255.0
                return torch.tensor([-0.5 - r, -1.5 + r, -4.0])

        exp = ExperimentConfig(
            prompt_template="Look at the image and complete the statement truthfully.",
            generations=2,
            pop_size=6,
            seed=17,  # deterministic init/GA stream — no unseeded-pymoo flake
            save_dir=tmp_path / "runs",
            name="slot_e2e",
            modality="image_only",
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(
                    _item(
                        image=_tiny_png(tmp_path),
                        fillers=("green", "blue", "red"),
                        pair=("green", "blue"),
                    ),
                )),
            ),
        )
        exp = apply_modality(exp)
        seeds = slot_items_seeds(None, exp, None)

        # image_only → MatrixDistance + TargetedBalance (what the runner wires).
        objectives = CriterionCollection(MatrixDistance(), TargetedBalance())
        tester = VLMBoundaryTester(
            sut=RecordingSUT(),
            manipulator=VLMManipulator(
                image_manipulator=FakeImageManipulator(),
                text_manipulator=FakeCompositeTextManipulator(),
            ),
            optimizer=DiscretePymooOptimizer(
                gene_bounds=np.ones(1, dtype=np.int64) * 2,
                num_objectives=objectives.num_objectives,
                pop_size=exp.pop_size,
                seed=exp.seed,
            ),
            objectives=objectives,
            config=exp,
        )
        tester.test(seeds)

        assert calls, "SUT was never called"
        prompt, cats = calls[0]
        assert cats == (
            "The house is green.", "The house is blue.", "The house is red.",
        )
        # Answer suffix lists the full candidate sentences. (The prompt body
        # itself comes from the stub text manipulator, not the template.)
        assert prompt.endswith(
            exp.answer_format.format(categories=", ".join(cats))
        )
        for cand in cats:
            assert cand in prompt

        run_dir = next((tmp_path / "runs").iterdir())
        stats = json.loads((run_dir / "stats.json").read_text())
        assert stats["categories"] == list(cats)
        assert stats["target_classes"] == [0, 1]
        assert stats["seed_selection_mode"] == "slot_items"
        df = pd.read_parquet(run_dir / "trace.parquet")
        assert len(df) >= exp.pop_size
        assert set(df["predicted_class"]) <= set(cats)

    def test_stats_record_slot_items_provenance(self, tmp_path) -> None:
        import numpy as np

        class _FakeManip:
            gene_bounds = np.zeros(1)
            image_dim = 1
            text_dim = 0

            def __getattr__(self, n):
                return 0

        exp = ExperimentConfig(
            seeds=SeedConfig(
                mode="slot_items",
                slot_items=SlotItemsConfig(items=(_item(image=_tiny_png(tmp_path)),)),
            ),
        )
        (seed,) = slot_items_seeds(None, exp, None)
        cands = tuple(seed.metadata["candidates"])
        stats = build_stats(
            0, seed, exp, _FakeManip(), 0, 0.0, cands,
            (seed.class_a, seed.class_b), (0, 1), {"hits": 0, "misses": 0},
        )
        assert stats["seed_selection_mode"] == "slot_items"
        assert stats["slot_items_n_items"] == 1
        assert stats["categories"] == list(cands)
        assert stats["seed_metadata"]["template"] == "The house is {slot}."
