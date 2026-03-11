"""Shared pytest configuration.

Adds ``tools/`` to sys.path so that packages living there (scene, vlm, …)
are importable during test collection and execution.
"""

import sys
from pathlib import Path

_tools = str(Path(__file__).resolve().parent.parent / "tools")
if _tools not in sys.path:
    sys.path.insert(0, _tools)
