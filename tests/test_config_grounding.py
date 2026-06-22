import dacite
import pytest
from src.config import ExperimentConfig, GroundingConfig, apply_modality, DEFAULT_ANSWER_FORMAT
from experiments.runners.run_boundary_test import _DACITE_CONFIG  # dacite Config used by load_config


def _parse(d):
    return dacite.from_dict(ExperimentConfig, d, config=_DACITE_CONFIG)


def test_grounding_modality_accepted_with_block():
    exp = _parse({
        "modality": "grounding",
        "grounding": {"coordinate_space": "norm_1000", "bbox_format": "bare_array"},
    })
    assert exp.modality == "grounding"
    assert isinstance(exp.grounding, GroundingConfig)
    assert exp.grounding.coordinate_space == "norm_1000"
    assert "[x1, y1, x2, y2]" in exp.grounding.answer_format


def test_grounding_defaults_when_block_absent():
    exp = _parse({"modality": "joint"})
    assert exp.grounding.coordinate_space == "norm_1000"  # default_factory


def test_bad_modality_still_rejected():
    with pytest.raises(ValueError, match="modality must be one of"):
        _parse({"modality": "bogus"})


def test_apply_modality_grounding_sets_answer_format():
    """apply_modality on grounding config must propagate grounding.answer_format to top level."""
    exp = _parse({
        "modality": "grounding",
        "grounding": {"coordinate_space": "norm_1000", "bbox_format": "bare_array"},
    })
    result = apply_modality(exp)
    assert result.answer_format == result.grounding.answer_format
    assert "[x1, y1, x2, y2]" in result.answer_format
    assert "{categories}" not in result.answer_format


def test_apply_modality_joint_leaves_answer_format_unchanged():
    """apply_modality on a joint config must not touch answer_format."""
    exp = _parse({"modality": "joint"})
    result = apply_modality(exp)
    assert result.answer_format == DEFAULT_ANSWER_FORMAT
