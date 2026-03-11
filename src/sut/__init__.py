"""VLM system-under-test -- SMOO-compatible teacher-forced scorer."""

from .config import DEFAULT_CATEGORIES, DEFAULT_PROMPT_TEMPLATE, VLMSUTConfig
from .vlm_sut import VLMSUT

__all__ = [
    "DEFAULT_CATEGORIES",
    "DEFAULT_PROMPT_TEMPLATE",
    "VLMSUT",
    "VLMSUTConfig",
]
