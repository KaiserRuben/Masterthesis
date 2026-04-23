"""PDQ / AutoBVA-style VLM boundary testing pipeline.

A complementary pipeline to the SMOO/AGE-MOEA-II runner that uses
two-stage directed search (flip discovery → minimisation) instead of
multi-objective evolution.

Key classes:

- :class:`~src.pdq.config.PDQExperimentConfig` — complete experiment config
- :class:`~src.pdq.runner.PDQRunner` — main orchestrator
- :class:`~src.pdq.sut_adapter.SUTAdapter` — SUT call tracker
- :class:`~src.pdq.artifacts.SeedLogger` — per-seed artifact writer

Entry point::

    python experiments/runners/run_pdq_test.py configs/templates/pdq_template.yaml
"""

from .config import PDQExperimentConfig, validate_config, resolve_categories, load_pdq_config
from .runner import PDQRunner
from .sut_adapter import SUTAdapter
from .artifacts import SeedLogger

__all__ = [
    "PDQExperimentConfig",
    "PDQRunner",
    "SeedLogger",
    "SUTAdapter",
    "load_pdq_config",
    "resolve_categories",
    "validate_config",
]
