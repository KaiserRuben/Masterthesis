"""Verify every YAML config field overrides the dataclass default.

For each field in ExperimentConfig (and nested sub-configs):
1. Creates a YAML dict with ONE non-default value
2. Calls load_config() from experiments/run_boundary_test.py
3. Asserts the resulting ExperimentConfig has the overridden value
4. Asserts fields NOT overridden still have their defaults
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_boundary_test import load_config
from src.config import ExperimentConfig
from src.manipulator.image.types import CandidateStrategy, PatchStrategy

DEFAULT = ExperimentConfig()


def _resolve(cfg: ExperimentConfig, path: str) -> object:
    """Resolve a dot-separated path to its value on cfg."""
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _check_defaults(cfg: ExperimentConfig, *, skip_path: str) -> list[str]:
    """Check that every field except skip_path matches the default."""
    failures = []

    def _check(obj, default_obj, prefix: str):
        if not hasattr(obj, "__dataclass_fields__"):
            return
        for name in obj.__dataclass_fields__:
            full = f"{prefix}.{name}" if prefix else name
            if full == skip_path:
                continue
            val = getattr(obj, name)
            dval = getattr(default_obj, name)
            if hasattr(val, "__dataclass_fields__"):
                _check(val, dval, full)
            elif val != dval:
                failures.append(f"{full} = {val!r}, expected {dval!r}")

    _check(cfg, DEFAULT, "")
    return failures


OVERRIDE_CASES = [
    # (field_path, yaml_dict, expected_value)
    ("device", {"device": "cuda"}, "cuda"),
    ("categories", {"categories": ["cat", "dog"]}, ("cat", "dog")),
    ("prompt_template", {"prompt_template": "Describe this image."}, "Describe this image."),
    ("answer_format", {"answer_format": " Pick one: {categories}."}, " Pick one: {categories}."),
    ("name", {"name": "test_experiment"}, "test_experiment"),
    ("save_dir", {"save_dir": "/tmp/test_runs"}, Path("/tmp/test_runs")),
    ("generations", {"generations": 200}, 200),
    ("pop_size", {"pop_size": 100}, 100),
    ("sut.model_id", {"sut": {"model_id": "meta-llama/Llama-3-8B"}}, "meta-llama/Llama-3-8B"),
    ("sut.enable_thinking", {"sut": {"enable_thinking": True}}, True),
    ("sut.max_thinking_tokens", {"sut": {"max_thinking_tokens": 5000}}, 5000),
    ("sut.max_pixels", {"sut": {"max_pixels": 1024}}, 1024),
    ("image.preset", {"image": {"preset": "f16-1024"}}, "f16-1024"),
    ("image.patch_ratio", {"image": {"patch_ratio": 0.5}}, 0.5),
    ("image.patch_strategy", {"image": {"patch_strategy": "ALL"}}, PatchStrategy.ALL),
    ("image.n_candidates", {"image": {"n_candidates": 50}}, 50),
    ("image.candidate_strategy", {"image": {"candidate_strategy": "UNIFORM"}}, CandidateStrategy.UNIFORM),
    ("image.resolution", {"image": {"resolution": 512}}, 512),
    ("image.knn_cache_path", {"image": {"knn_cache_path": "/tmp/knn_cache.npy"}}, Path("/tmp/knn_cache.npy")),
    ("text.spacy_model", {"text": {"spacy_model": "en_core_web_lg"}}, "en_core_web_lg"),
    ("text.embedding_model", {"text": {"embedding_model": "glove-wiki-gigaword-100"}}, "glove-wiki-gigaword-100"),
    ("text.n_candidates", {"text": {"n_candidates": 10}}, 10),
    ("seeds.n_per_class", {"seeds": {"n_per_class": 10}}, 10),
    ("seeds.max_logprob_gap", {"seeds": {"max_logprob_gap": 5.0}}, 5.0),
]


@pytest.mark.parametrize(
    "field_path, yaml_dict, expected",
    OVERRIDE_CASES,
    ids=[case[0] for case in OVERRIDE_CASES],
)
def test_config_override(field_path, yaml_dict, expected):
    """Each YAML field correctly overrides the dataclass default."""
    cfg = load_config(yaml_dict)

    actual = _resolve(cfg, field_path)
    assert actual == expected, f"Override mismatch for {field_path}: got {actual!r}"

    drift = _check_defaults(cfg, skip_path=field_path)
    assert not drift, f"Non-overridden defaults drifted:\n" + "\n".join(drift)
