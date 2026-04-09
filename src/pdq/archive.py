"""PDQ archive write-path.

The archive is the canonical output of the PDQ pipeline: one row per
confirmed flip, capturing the best-known minimised genotype and all
distance metrics.

Two write paths exist:

``append_to_archive``
    Phase-2 path: writes a ``VE`` row using the Stage-1 flip genotype as
    the initial ``genotype_min`` placeholder.  Superseded by Phase 3.

``build_archive_row_stage2`` / ``append_archive_row_stage2``
    Phase-3 path: writes a ``VV`` row with the Stage-2 minimised genotype,
    updated d_i_min, and recomputed PDQ.  Called after
    :func:`~src.pdq.search.stage2.minimise_flip` completes for each flip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .artifacts import ParquetBuffer

if TYPE_CHECKING:
    from .search.stage2 import Stage2Result
from .distances.output import label_mismatch, string_edit
from .search.base import ScoredCandidate


def build_archive_row(
    sc: ScoredCandidate,
    flip_id: int,
    seed_id: str,
    run_id: str,
    anchor_geno_list: list[int],
    anchor_logprobs: list[float],
    anchor_label: str,
    stage1_sut_calls: int,
) -> dict[str, Any]:
    """Build a row dict for ``archive.parquet`` from a Stage-1 flip.

    Since Stage 2 has not run, ``genotype_min = genotype_flipped`` and
    ``validity = "VE"``.  Phase-3 distance stubs (embedding, wordnet) are
    stored as ``None``.

    :param sc: A Stage-1 ``ScoredCandidate`` with ``flipped=True``.
    :param flip_id: 0-based flip index within this seed.
    :param seed_id: Seed identifier string (e.g. ``"seed_0000"``).
    :param run_id: Experiment name / run identifier.
    :param anchor_geno_list: Anchor genotype as a plain list (JSON-safe).
    :param anchor_logprobs: Logprobs from the anchor SUT call.
    :param anchor_label: VLM prediction on the anchor.
    :param stage1_sut_calls: Total SUT calls consumed in Stage 1 so far.
    :returns: Dict with keys matching ``ARCHIVE_COLUMNS`` exactly.
    """
    geno_flipped = sc.candidate.genotype.tolist()

    # All d_o variants — phase-3 stubs stored as None.
    d_o_mismatch = label_mismatch(anchor_label, sc.label)
    d_o_edit = string_edit(anchor_label, sc.label)

    return {
        "pipeline": "pdq",
        "run_id": run_id,
        "seed_id": seed_id,
        "flip_id": flip_id,
        # Genotypes
        "genotype_anchor": anchor_geno_list,
        "genotype_flipped": geno_flipped,
        "genotype_min": geno_flipped,          # Phase 2: no Stage-2 minimisation
        # Labels
        "label_anchor": anchor_label,
        "label_flipped": sc.label,
        "label_min": sc.label,                 # Phase 2: min = flipped
        # Logprobs
        "logprobs_anchor": anchor_logprobs,
        "logprobs_flipped": sc.logprobs,
        "logprobs_min": sc.logprobs,           # Phase 2: min = flipped
        # Sparsity / rank (flipped == min in Phase 2)
        "sparsity_flipped": sc.total_sparsity,
        "sparsity_min": sc.total_sparsity,
        "rank_sum_flipped": sc.total_rank_sum,
        "rank_sum_min": sc.total_rank_sum,
        # Image / text distances (flipped == min in Phase 2)
        "image_pixel_L2_min": sc.image_pixel_L2,
        "text_cosine_sum_min": sc.text_cosine_sum,
        # Input distance (primary)
        "d_i_primary": sc.d_i,
        # Output distances
        "d_o_label_mismatch": d_o_mismatch,
        "d_o_label_edit": d_o_edit,
        "d_o_label_embedding": None,           # TODO(phase3)
        "d_o_wordnet_path": None,              # TODO(phase3)
        # PDQ metric and validity
        "pdq": sc.pdq_score,
        "validity": "VE",                      # Phase 2: flip valid, no minimisation
        # SUT call accounting
        "stage1_sut_calls": stage1_sut_calls,
        "stage2_sut_calls": 0,
        "sut_calls_total": stage1_sut_calls,
        # Provenance
        "found_by": sc.candidate.strategy,
    }


def append_to_archive(
    buffer: ParquetBuffer,
    sc: ScoredCandidate,
    flip_id: int,
    seed_id: str,
    run_id: str,
    anchor_geno_list: list[int],
    anchor_logprobs: list[float],
    anchor_label: str,
    stage1_sut_calls: int,
) -> None:
    """Append one flip to the archive parquet buffer.

    Thin wrapper around :func:`build_archive_row` that writes directly to
    the buffer — callers don't need to import the row builder separately.

    :param buffer: The ``archive.parquet`` :class:`ParquetBuffer`.
    :param sc: Stage-1 flip candidate.
    :param flip_id: 0-based flip index within this seed.
    :param seed_id: Seed identifier string.
    :param run_id: Experiment name / run identifier.
    :param anchor_geno_list: Anchor genotype as plain list.
    :param anchor_logprobs: Logprobs from the anchor SUT call.
    :param anchor_label: VLM prediction on the anchor.
    :param stage1_sut_calls: Total Stage-1 SUT calls at time of write.
    """
    row = build_archive_row(
        sc=sc,
        flip_id=flip_id,
        seed_id=seed_id,
        run_id=run_id,
        anchor_geno_list=anchor_geno_list,
        anchor_logprobs=anchor_logprobs,
        anchor_label=anchor_label,
        stage1_sut_calls=stage1_sut_calls,
    )
    buffer.append(row)


# ---------------------------------------------------------------------------
# Phase-3 write path: Stage-2 minimised archive row
# ---------------------------------------------------------------------------


def build_archive_row_stage2(
    sc: ScoredCandidate,
    flip_id: int,
    seed_id: str,
    run_id: str,
    anchor_geno_list: list[int],
    anchor_logprobs: list[float],
    anchor_label: str,
    stage1_sut_calls: int,
    stage2_result: Stage2Result,
    eps: float = 1e-9,
) -> dict[str, Any]:
    """Build a ``VV`` archive row using Stage-2 minimised genotype.

    :param sc: The Stage-1 :class:`~src.pdq.search.base.ScoredCandidate`
        (provides flip genotype, label, logprobs, sparsity at discovery).
    :param flip_id: 0-based flip index within this seed.
    :param seed_id: Seed identifier string.
    :param run_id: Experiment name / run identifier.
    :param anchor_geno_list: Anchor genotype as plain list.
    :param anchor_logprobs: Logprobs from the anchor SUT call.
    :param anchor_label: VLM prediction on the anchor.
    :param stage1_sut_calls: Total SUT calls consumed by Stage 1.
    :param stage2_result: Result from
        :func:`~src.pdq.search.stage2.minimise_flip`.
    :param eps: Division guard for PDQ computation.
    :returns: Dict with keys matching ``ARCHIVE_COLUMNS`` exactly.
    """
    geno_min = stage2_result.genotype_min
    geno_min_list = geno_min.tolist()
    geno_flipped = sc.candidate.genotype.tolist()

    d_i_min = stage2_result.d_i_min
    d_o = float(sc.d_o)
    pdq_min = d_o / (d_i_min + eps)

    # Validity: VV — Stage-1 flip confirmed AND Stage-2 min confirmed.
    # Since passes only accept steps that preserve the flip, genotype_min
    # always still flips the label.
    validity = "VV"

    label_min = stage2_result.final_label or sc.label
    d_o_mismatch = label_mismatch(anchor_label, label_min)
    d_o_edit = string_edit(anchor_label, label_min)

    return {
        "pipeline": "pdq",
        "run_id": run_id,
        "seed_id": seed_id,
        "flip_id": flip_id,
        # Genotypes
        "genotype_anchor": anchor_geno_list,
        "genotype_flipped": geno_flipped,
        "genotype_min": geno_min_list,
        # Labels
        "label_anchor": anchor_label,
        "label_flipped": sc.label,
        "label_min": label_min,
        # Logprobs — Stage-2 does not do a final re-score; use Stage-1 flip
        # logprobs as proxy for logprobs_min (same label, close genotype).
        "logprobs_anchor": anchor_logprobs,
        "logprobs_flipped": sc.logprobs,
        "logprobs_min": sc.logprobs,
        # Sparsity / rank at flip and after minimisation
        "sparsity_flipped": sc.total_sparsity,
        "sparsity_min": int(np.count_nonzero(geno_min)),
        "rank_sum_flipped": sc.total_rank_sum,
        "rank_sum_min": int(np.sum(geno_min)),
        # Pixel / text distances (minimised values approximate via Stage-1)
        "image_pixel_L2_min": sc.image_pixel_L2,
        "text_cosine_sum_min": sc.text_cosine_sum,
        # Input distance — minimised value
        "d_i_primary": d_i_min,
        # Output distances
        "d_o_label_mismatch": d_o_mismatch,
        "d_o_label_edit": d_o_edit,
        "d_o_label_embedding": None,
        "d_o_wordnet_path": None,
        # PDQ metric — recomputed with d_i_min
        "pdq": pdq_min,
        "validity": validity,
        # SUT call accounting
        "stage1_sut_calls": stage1_sut_calls,
        "stage2_sut_calls": stage2_result.sut_calls_used,
        "sut_calls_total": stage1_sut_calls + stage2_result.sut_calls_used,
        # Provenance
        "found_by": sc.candidate.strategy,
    }


def append_archive_row_stage2(
    buffer: ParquetBuffer,
    sc: ScoredCandidate,
    flip_id: int,
    seed_id: str,
    run_id: str,
    anchor_geno_list: list[int],
    anchor_logprobs: list[float],
    anchor_label: str,
    stage1_sut_calls: int,
    stage2_result: Stage2Result,
) -> None:
    """Append one Stage-2 minimised archive row to the buffer.

    :param buffer: The ``archive.parquet`` :class:`ParquetBuffer`.
    :param sc: Stage-1 flip candidate.
    :param flip_id: Flip index within this seed.
    :param seed_id: Seed identifier string.
    :param run_id: Experiment name.
    :param anchor_geno_list: Anchor genotype as plain list.
    :param anchor_logprobs: Anchor SUT logprobs.
    :param anchor_label: Anchor VLM label.
    :param stage1_sut_calls: Total Stage-1 SUT calls.
    :param stage2_result: Stage-2 minimisation result.
    """
    row = build_archive_row_stage2(
        sc=sc,
        flip_id=flip_id,
        seed_id=seed_id,
        run_id=run_id,
        anchor_geno_list=anchor_geno_list,
        anchor_logprobs=anchor_logprobs,
        anchor_label=anchor_label,
        stage1_sut_calls=stage1_sut_calls,
        stage2_result=stage2_result,
    )
    buffer.append(row)
