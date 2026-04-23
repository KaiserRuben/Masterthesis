"""VLM system-under-test -- SMOO-compatible teacher-forced scorer."""

from .preflight import preflight_cost_check
from .vlm_sut import VLMSUT

__all__ = ["VLMSUT", "preflight_cost_check"]
