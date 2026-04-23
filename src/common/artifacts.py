"""Shared on-disk artifact helpers for both pipelines.

Hosts the Parquet schema-version constants and the incremental writer
used by ``src.evolutionary`` and ``src.pdq`` alike.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# Each pipeline maintains its own schema-version counter. They happen to
# be at the same value today but can evolve independently — a bump to
# one does not imply a bump to the other.
EVOLUTIONARY_SCHEMA_VERSION = 2
PDQ_SCHEMA_VERSION = 2


class ParquetBuffer:
    """Incremental crash-safe Parquet writer.

    Rows are accumulated in memory. When the buffer reaches
    ``flush_interval`` entries (or :meth:`close` is called), they are
    appended to the parquet file as a new row group.

    Schema is inferred from the first batch; subsequent batches must
    be schema-compatible. The underlying :class:`pyarrow.parquet.ParquetWriter`
    is kept open between flushes so row groups accumulate in a single file.

    File-level key/value metadata (e.g. ``schema_version``) may be
    supplied via ``file_metadata``; it is stamped into the
    :class:`pyarrow.parquet.ParquetWriter` on first open so it persists
    across row groups.

    :param path: Output Parquet file path.
    :param compression: Parquet compression codec (e.g. ``"zstd"``).
    :param flush_interval: Flush after this many :meth:`append` calls.
    :param file_metadata: Optional bytes→bytes mapping stamped into the
        parquet file-level schema metadata.
    """

    def __init__(
        self,
        path: Path,
        compression: str = "zstd",
        flush_interval: int = 100,
        file_metadata: dict[bytes, bytes] | None = None,
    ) -> None:
        self._path = path
        self._compression = compression
        self._flush_interval = flush_interval
        self._buf: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self._total_rows: int = 0
        self._file_metadata: dict[bytes, bytes] | None = file_metadata

    def append(self, row: dict[str, Any]) -> None:
        """Append one row. Triggers a flush when the buffer is full."""
        self._buf.append(row)
        if len(self._buf) >= self._flush_interval:
            self._flush()

    def append_many(self, rows: list[dict[str, Any]]) -> None:
        """Append multiple rows, flushing when the buffer fills."""
        for row in rows:
            self.append(row)

    def _flush(self) -> None:
        """Write buffered rows as one row group."""
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf)
        if self._file_metadata:
            existing = dict(table.schema.metadata or {})
            existing.update(self._file_metadata)
            table = table.replace_schema_metadata(existing)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self._path),
                table.schema,
                compression=self._compression,
            )
        self._writer.write_table(table)
        self._total_rows += len(self._buf)
        self._buf = []
        logger.debug("Flushed %d rows to %s", self._total_rows, self._path.name)

    def close(self) -> None:
        """Flush remaining rows and close the writer (idempotent)."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def total_rows(self) -> int:
        """Rows flushed to disk so far (excludes in-memory buffer)."""
        return self._total_rows

    @property
    def buffered_rows(self) -> int:
        """Rows in memory awaiting next flush."""
        return len(self._buf)

    def __enter__(self) -> ParquetBuffer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
