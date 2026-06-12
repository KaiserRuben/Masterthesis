"""Modality postprocessing in the boundary-pair pipeline.

Regression coverage for the bug where ``BoundaryPairRunner.run`` accepted
``evolutionary.modality`` on its config but never applied the modality
postprocessing (:func:`src.config.apply_modality`), so a boundary-pair
config with ``modality: image_only`` silently kept the full text
manipulation stack active (and vice versa for ``text_only``).

Everything here operates on configs only — no SUT model is loaded and
no search runs.  The runner-level tests stub out ``ImageNetCache`` /
``init_shared_components`` / ``prepare_pipeline_seeds`` and capture the
evolutionary config exactly as it reaches component initialisation.
"""

from __future__ import annotations

from pathlib import Path

from src.boundary_pair.config import (
    BoundaryPairExperimentConfig,
    EvolutionaryStageConfig,
    to_evolutionary_config,
)
from src.config import apply_modality
from src.manipulator.text.profiles import load_profile_library, resolve_profile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROFILE_LIBRARY = _REPO_ROOT / "configs" / "templates" / "text_profiles.yaml"


def _bp_config(modality: str) -> BoundaryPairExperimentConfig:
    return BoundaryPairExperimentConfig(
        evolutionary=EvolutionaryStageConfig(modality=modality),
    )


# ---------------------------------------------------------------------------
# apply_modality on the evolutionary projection
# ---------------------------------------------------------------------------


class TestApplyModalityOnProjection:
    def test_image_only_forces_noop_text_profile(self) -> None:
        evo = apply_modality(to_evolutionary_config(_bp_config("image_only")))
        assert evo.text.composite.profile == "noop"
        assert evo.text.composite.operators == ()
        assert not evo.text.composite.overrides
        # Image channel untouched.
        assert evo.image.patch_ratio == _bp_config("image_only").image.patch_ratio

    def test_noop_profile_resolves_to_zero_operators(self) -> None:
        """text genotype dim is the sum of per-operator gene dims —
        an empty operator stack means text_dim == 0 at composite.prepare,
        verifiable without loading spacy or any MLM."""
        library = load_profile_library(_PROFILE_LIBRARY)
        specs = resolve_profile(library, profile_name="noop")
        assert specs == ()

    def test_text_only_zeroes_patch_ratio(self) -> None:
        evo = apply_modality(to_evolutionary_config(_bp_config("text_only")))
        assert evo.image.patch_ratio == 0.0
        # Text channel untouched.
        assert (
            evo.text.composite.profile
            == _bp_config("text_only").text.composite.profile
        )

    def test_joint_is_passthrough(self) -> None:
        evo = to_evolutionary_config(_bp_config("joint"))
        assert apply_modality(evo) is evo


# ---------------------------------------------------------------------------
# BoundaryPairRunner.run applies modality before component init
# ---------------------------------------------------------------------------


class _StubDataSource:
    """ImageNetCache stand-in — labels only, no disk access."""

    def __init__(self, dirs=None, **_: object) -> None:
        del dirs

    def labels(self) -> list[str]:
        return ["junco", "boa constrictor"]


class TestRunnerAppliesModality:
    """Capture the evo config exactly as it reaches init_shared_components."""

    @staticmethod
    def _evo_cfg_seen_by_components(monkeypatch, cfg):
        import src.boundary_pair.runner as runner_mod

        captured: dict[str, object] = {}

        def fake_init_shared_components(evo_cfg, data_source):
            captured["evo_cfg"] = evo_cfg
            return object()

        def fake_prepare_pipeline_seeds(components, evo_cfg):
            return []  # empty pool → run() exits right after capture

        monkeypatch.setattr(runner_mod, "ImageNetCache", _StubDataSource)
        monkeypatch.setattr(
            runner_mod, "init_shared_components", fake_init_shared_components,
        )
        monkeypatch.setattr(
            runner_mod, "prepare_pipeline_seeds", fake_prepare_pipeline_seeds,
        )

        runner_mod.BoundaryPairRunner(cfg).run()
        return captured["evo_cfg"]

    def test_image_only_disables_text_channel(self, monkeypatch) -> None:
        evo_cfg = self._evo_cfg_seen_by_components(
            monkeypatch, _bp_config("image_only"),
        )
        assert evo_cfg.modality == "image_only"
        assert evo_cfg.text.composite.profile == "noop"
        assert evo_cfg.text.composite.operators == ()

    def test_text_only_disables_image_channel(self, monkeypatch) -> None:
        evo_cfg = self._evo_cfg_seen_by_components(
            monkeypatch, _bp_config("text_only"),
        )
        assert evo_cfg.modality == "text_only"
        assert evo_cfg.image.patch_ratio == 0.0

    def test_joint_keeps_both_channels(self, monkeypatch) -> None:
        cfg = _bp_config("joint")
        evo_cfg = self._evo_cfg_seen_by_components(monkeypatch, cfg)
        assert evo_cfg.modality == "joint"
        assert evo_cfg.text.composite.profile == cfg.text.composite.profile
        assert evo_cfg.image.patch_ratio == cfg.image.patch_ratio
