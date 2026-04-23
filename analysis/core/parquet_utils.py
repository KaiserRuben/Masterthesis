"""Shared Parquet reading helpers used by the SMOO and PDQ loaders."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


def read_parquet_metadata(path: Path) -> dict[str, str]:
    """Read file-level key/value metadata from a Parquet file.

    Returns a plain ``{str: str}`` dict (keys/values decoded from UTF-8).
    Missing or unreadable metadata yields an empty dict — callers then
    fall back to content inspection or an adjacent ``config.json``.
    """
    try:
        raw = pq.read_schema(path).metadata or {}
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        try:
            out[k.decode("utf-8")] = v.decode("utf-8")
        except Exception:  # noqa: BLE001
            pass
    return out
