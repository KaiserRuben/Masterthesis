"""PDQ artifact writers: parquet buffers and per-seed artifact writers.

Each seed run gets a :class:`SeedLogger` that owns all parquet files and
image/JSON artifacts for that seed.  Parquet files are written
incrementally via :class:`ParquetBuffer` — each flush appends a new row
group to the file, providing crash-safe persistence.

Column schemas are defined as module-level constants so callers can
produce correctly-typed rows without guessing field names.

Validity classes (``archive.parquet`` ``validity`` column)
----------------------------------------------------------
Every archived flip is assigned a two-letter validity code:

``VV`` — Valid-Valid
    The Stage-1 flip genotype still flips the label *and* the Stage-2
    minimised genotype also still flips the label.  Both endpoints are
    confirmed boundary crossings.

``VE`` — Valid-Empty
    The Stage-1 flip is valid but Stage-2 could not tighten it (the
    minimised genotype reverted to the anchor; ``g_min == g_0``).  The
    flip distance is the Stage-1 distance, which is an upper bound.

``EE`` — Empty-Empty
    Neither genotype flips the label (should not appear in the archive
    under normal operation; present only for diagnostic rows written when
    a cached flip is later invalidated by a subsequent SUT call).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parquet column-name constants
# ---------------------------------------------------------------------------

SUT_CALLS_COLUMNS: list[str] = [
    "call_id", "candidate_id", "stage", "prompt_hash", "image_hash",
    "cache_hit", "logprobs", "categories",
    "top1_idx", "top1_label", "top2_idx", "top2_label",
    "logprob_gap_1v2", "wall_time_s", "wall_time_cumulative_s",
    "sut_call_cumulative", "cache_miss_cumulative",
]

CANDIDATES_COLUMNS: list[str] = [
    "candidate_id", "stage", "parent_candidate_id", "operation",
    "target_gene", "old_value", "new_value", "genotype",
    "img_sparsity", "txt_sparsity", "total_sparsity",
    "img_rank_sum", "txt_rank_sum", "total_rank_sum",
    "hamming_to_anchor", "rendered_text", "image_pixel_L2",
    "text_cosine_sum", "image_path", "sut_call_id", "label",
    "flipped_vs_anchor", "accepted",
]

STAGE1_FLIPS_COLUMNS: list[str] = [
    "flip_id", "candidate_id", "discovery_sut_call",
    "discovery_wall_time_s", "operation", "L_anchor", "L_target",
    "genotype_flipped", "total_sparsity_at_discovery",
    "total_rank_sum_at_discovery", "is_first_for_target",
    "stage2_refined_candidate_id",
]

STAGE2_TRAJECTORIES_COLUMNS: list[str] = [
    "flip_id", "step", "pass_name", "target_gene",
    "old_value", "new_value", "candidate_id_before", "candidate_id_after",
    "label_before", "label_after", "still_flipped", "accepted",
    "sparsity_before", "sparsity_after", "rank_sum_before", "rank_sum_after",
    "sut_call_id", "wall_time_cumulative_s",
]

ARCHIVE_COLUMNS: list[str] = [
    "pipeline", "run_id", "seed_id", "flip_id",
    "genotype_anchor", "genotype_flipped", "genotype_min",
    "label_anchor", "label_flipped", "label_min",
    "logprobs_anchor", "logprobs_flipped", "logprobs_min",
    "sparsity_flipped", "sparsity_min",
    "rank_sum_flipped", "rank_sum_min",
    "image_pixel_L2_min", "text_cosine_sum_min",
    "d_i_primary",
    "d_o_label_mismatch", "d_o_label_edit",
    "d_o_label_embedding", "d_o_wordnet_path",
    "pdq", "validity",
    "stage1_sut_calls", "stage2_sut_calls", "sut_calls_total",
    "found_by",
]

CONVERGENCE_COLUMNS: list[str] = [
    "sut_call_id", "stage", "wall_time_s",
    "n_flips_discovered", "n_distinct_targets",
    "min_sparsity_over_archive", "min_rank_sum_over_archive",
    "mean_pdq_over_archive",
]

# Map: filename → ordered column list
ALL_PARQUET_FILES: dict[str, list[str]] = {
    "sut_calls.parquet": SUT_CALLS_COLUMNS,
    "candidates.parquet": CANDIDATES_COLUMNS,
    "stage1_flips.parquet": STAGE1_FLIPS_COLUMNS,
    "stage2_trajectories.parquet": STAGE2_TRAJECTORIES_COLUMNS,
    "archive.parquet": ARCHIVE_COLUMNS,
    "convergence.parquet": CONVERGENCE_COLUMNS,
}


# ---------------------------------------------------------------------------
# Parquet buffer (crash-safe incremental writer)
# ---------------------------------------------------------------------------


class ParquetBuffer:
    """Incremental crash-safe parquet writer.

    Rows are accumulated in memory.  When the buffer reaches
    ``flush_interval`` entries (or :meth:`close` is called), they are
    appended to the parquet file as a new row group.

    Schema is inferred from the first batch; subsequent batches must
    be schema-compatible.  The underlying :class:`pyarrow.parquet.ParquetWriter`
    is kept open between flushes so row groups accumulate in a single file.

    :param path: Output parquet file path.
    :param compression: Parquet compression codec (e.g. ``"zstd"``).
    :param flush_interval: Flush after this many :meth:`append` calls.
    """

    def __init__(
        self,
        path: Path,
        compression: str = "zstd",
        flush_interval: int = 100,
    ) -> None:
        self._path = path
        self._compression = compression
        self._flush_interval = flush_interval
        self._buf: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self._total_rows: int = 0

    def append(self, row: dict[str, Any]) -> None:
        """Append one row.  Triggers a flush when buffer is full.

        :param row: Dict of column-name → value for this row.
        """
        self._buf.append(row)
        if len(self._buf) >= self._flush_interval:
            self._flush()

    def append_many(self, rows: list[dict[str, Any]]) -> None:
        """Append multiple rows, flushing when the buffer fills.

        :param rows: List of row dicts.
        """
        for row in rows:
            self.append(row)

    def _flush(self) -> None:
        """Write buffered rows as one row group."""
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf)
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
        """Flush remaining rows and close the writer.

        Safe to call multiple times (idempotent after first call).
        """
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def total_rows(self) -> int:
        """Total rows written to disk (excludes unflushed buffer)."""
        return self._total_rows

    @property
    def buffered_rows(self) -> int:
        """Rows in memory awaiting next flush."""
        return len(self._buf)


# ---------------------------------------------------------------------------
# Per-seed artifact writer
# ---------------------------------------------------------------------------


class SeedLogger:
    """Manages all artifacts for one seed run.

    Creates the seed directory on construction, opens all parquet
    buffers, and provides helpers for writing images and JSON metadata.

    Use as a context manager — :meth:`__exit__` flushes and closes all
    parquet writers::

        with SeedLogger(seed_dir, compression, flush_interval) as sl:
            sl.append_sut_call(record)
            ...

    :param seed_dir: Seed output directory (created on construction).
    :param compression: Parquet compression codec.
    :param flush_interval: Parquet flush cadence (rows).
    """

    def __init__(
        self,
        seed_dir: Path,
        compression: str = "zstd",
        flush_interval: int = 100,
    ) -> None:
        self._dir = seed_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "flips").mkdir(exist_ok=True)

        self._compression = compression
        self._bufs: dict[str, ParquetBuffer] = {
            fname: ParquetBuffer(
                seed_dir / fname,
                compression=compression,
                flush_interval=flush_interval,
            )
            for fname in ALL_PARQUET_FILES
        }
        # Initialise all parquet files with a zero-row stub so every file
        # exists on disk with the right column schema from the moment the
        # seed dir is created (crash safety + downstream schema discovery).
        self._write_empty_parquets()

    # -- empty initialisation --------------------------------------------

    def _write_empty_parquets(self) -> None:
        """Write zero-row stub files for every parquet table.

        Ensures all files exist on disk immediately after the seed dir is
        created.  Each stub is overwritten the first time the buffer flushes
        real rows (the ParquetWriter appends row groups to the same path).

        Uses pandas (not the ParquetBuffer) so the file is created even
        before any rows arrive.
        """
        for fname, columns in ALL_PARQUET_FILES.items():
            path = self._dir / fname
            if not path.exists():
                pd.DataFrame(columns=columns).to_parquet(
                    path,
                    engine="pyarrow",
                    compression=self._compression,
                    index=False,
                )

    # -- context manager -------------------------------------------------

    def __enter__(self) -> SeedLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close all parquet writers."""
        for buf in self._bufs.values():
            buf.close()

    # -- parquet append helpers ------------------------------------------

    def append_sut_call(self, record: dict[str, Any]) -> None:
        """Append one row to ``sut_calls.parquet``."""
        self._bufs["sut_calls.parquet"].append(record)

    def append_sut_calls(self, records: list[dict[str, Any]]) -> None:
        """Append multiple rows to ``sut_calls.parquet``."""
        self._bufs["sut_calls.parquet"].append_many(records)

    def append_candidate(self, record: dict[str, Any]) -> None:
        """Append one row to ``candidates.parquet``."""
        self._bufs["candidates.parquet"].append(record)

    def append_stage1_flip(self, record: dict[str, Any]) -> None:
        """Append one row to ``stage1_flips.parquet``."""
        self._bufs["stage1_flips.parquet"].append(record)

    def append_stage2_step(self, record: dict[str, Any]) -> None:
        """Append one row to ``stage2_trajectories.parquet``."""
        self._bufs["stage2_trajectories.parquet"].append(record)

    def append_archive_row(self, record: dict[str, Any]) -> None:
        """Append one row to ``archive.parquet``."""
        self._bufs["archive.parquet"].append(record)

    def append_convergence(self, record: dict[str, Any]) -> None:
        """Append one row to ``convergence.parquet``."""
        self._bufs["convergence.parquet"].append(record)

    def flush_all(self) -> None:
        """Force-flush all parquet buffers (crash-safety checkpoint)."""
        for buf in self._bufs.values():
            buf._flush()  # noqa: SLF001  — intentional internal flush

    # -- image / text helpers --------------------------------------------

    def save_anchor_original(self, image: Image.Image) -> Path:
        """Write the original (pre-VQGAN) seed image."""
        path = self._dir / "anchor_original.png"
        image.save(path)
        return path

    def save_anchor_baseline(self, image: Image.Image) -> Path:
        """Write the VQGAN-reconstructed anchor (zero-genotype) image."""
        path = self._dir / "anchor_baseline.png"
        image.save(path)
        return path

    def save_anchor_prompt(self, prompt: str) -> Path:
        """Write the full anchor prompt text."""
        path = self._dir / "anchor_prompt.txt"
        path.write_text(prompt, encoding="utf-8")
        return path

    def save_flip_image(
        self,
        flip_id: int,
        stage: str,
        image: Image.Image,
        meta: dict[str, Any],
    ) -> Path:
        """Write a flip image and its companion JSON to ``flips/``."""
        stem = f"flip_{flip_id:04d}_{stage}"
        img_path = self._dir / "flips" / f"{stem}.png"
        json_path = self._dir / "flips" / f"{stem}.json"
        image.save(img_path)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        return img_path

    # -- JSON metadata helpers -------------------------------------------

    def write_config_json(self, cfg_dict: dict[str, Any]) -> None:
        """Serialise the experiment config."""
        with open(self._dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    def write_context_json(self, ctx: dict[str, Any]) -> None:
        """Serialise the manipulation context (patch/word selections)."""
        with open(self._dir / "context.json", "w") as f:
            json.dump(ctx, f, indent=2)

    def write_stats_json(self, stats: dict[str, Any]) -> None:
        """Serialise per-seed stats."""
        with open(self._dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def write_rng_state_json(self, state: dict[str, Any]) -> None:
        """Serialise the numpy RNG state for reproducibility."""
        with open(self._dir / "rng_state.json", "w") as f:
            json.dump(state, f, indent=2)

    # -- directory / row-count info ---------------------------------------

    @property
    def seed_dir(self) -> Path:
        """The seed output directory."""
        return self._dir

    def row_counts(self) -> dict[str, int]:
        """Return total-on-disk row counts for all parquet files."""
        return {
            fname: buf.total_rows
            for fname, buf in self._bufs.items()
        }
