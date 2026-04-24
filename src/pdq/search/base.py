"""Shared types and the core scoring function for the PDQ search loop.

``Candidate`` records what was generated and by which strategy.
``ScoredCandidate`` wraps a Candidate with all evaluation results needed
to write ``candidates.parquet``, ``stage1_flips.parquet``, and
``archive.parquet`` without further computation in the runner.

Design note: these dataclasses hold ``np.ndarray`` genotypes, so they
cannot be ``frozen=True`` (numpy arrays are not hashable).  Treat them
as immutable by convention — never mutate fields after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from PIL import Image

from ..distances.input import hamming, image_pixel_l2, rank_sum, sparsity
from ..metric import pdq as pdq_metric

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Candidate:
    """A genotype proposed by a Stage-1 strategy.

    :param genotype: Integer genotype ``[img_genes | txt_genes]``.
    :param strategy: Name of the generating strategy (e.g. "dense_uniform").
    :param seed_idx: 0-based seed index within the run.
    :param gen_step: Position of this candidate within the strategy's output
        (0-based; used for reproducibility bookkeeping).
    """

    genotype: np.ndarray
    strategy: str
    seed_idx: int
    gen_step: int


@dataclass(eq=False)
class ScoredCandidate:
    """A Candidate after SUT evaluation and distance computation.

    Carries every field needed to populate ``candidates.parquet``,
    ``stage1_flips.parquet``, and ``archive.parquet`` directly — the
    runner just converts to dicts, no re-computation required.

    :param candidate: The generating Candidate.
    :param candidate_id: Globally unique ID for this seed (= sut_call_id
        in Stage 1 where there is a 1:1 call-per-candidate correspondence).
    :param label: Top-1 predicted label from the VLM.
    :param logprobs: Log-probability vector (one entry per category).
    :param sut_call_id: SUT adapter call_id (links to sut_calls.parquet).
    :param rendered_text: Prompt text produced by the text manipulator.
    :param rendered_image: PIL image produced by the image manipulator
        (held in memory; saved to disk only for flips when configured).
    :param discovery_wall_time_cum: Cumulative SUT wall time at discovery.
    :param d_i: Primary input distance (from ``d_i_primary`` config).
    :param d_o: Primary output distance (from ``d_o_primary`` config).
    :param pdq_score: PDQ = d_o / (d_i + eps).
    :param flipped: Whether predicted label differs from anchor label.
    :param img_sparsity: Non-zero image genes.
    :param txt_sparsity: Non-zero text genes.
    :param total_sparsity: img_sparsity + txt_sparsity.
    :param img_rank_sum: Sum of image gene values.
    :param txt_rank_sum: Sum of text gene values.
    :param total_rank_sum: img_rank_sum + txt_rank_sum.
    :param hamming_to_anchor: Positions differing from the zero anchor.
    :param image_pixel_L2: Pixel L2 vs the VQGAN-reconstructed anchor image.
    :param text_cosine_sum: Cosine distance between the manipulated prompt
        and the anchor prompt, computed in the SUT's sentence-embedding
        space (mean-pooled last-hidden-state of the Qwen text backbone).
        Name retained for parquet-schema compatibility; semantics replaced
        (was: sum of per-word fasttext cosine distances).
    """

    candidate: Candidate
    candidate_id: int
    label: str
    logprobs: list[float]
    sut_call_id: int
    rendered_text: str
    rendered_image: Image.Image
    discovery_wall_time_cum: float
    d_i: float
    d_o: float
    pdq_score: float
    flipped: bool
    img_sparsity: int
    txt_sparsity: int
    total_sparsity: int
    img_rank_sum: int
    txt_rank_sum: int
    total_rank_sum: int
    hamming_to_anchor: int
    image_pixel_L2: float
    text_cosine_sum: float


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def score_candidate(
    cand: Candidate,
    anchor_label: str,
    anchor_geno: np.ndarray,
    anchor_image_arr: np.ndarray,
    text_distance_fn: Callable[[str], float],
    image_dim: int,
    categories: tuple[str, ...],
    sut_call_fn: Callable[[np.ndarray], tuple[list[float], int, Image.Image, str, float]],
    input_distance_fn: Callable[[np.ndarray, np.ndarray], float],
    output_distance_fn: Callable[[str, str], float],
) -> ScoredCandidate:
    """Evaluate one candidate genotype and return a fully scored result.

    Calling this function makes exactly one SUT call via *sut_call_fn*.
    All derived metrics (sparsity, rank_sum, pixel L2, PDQ) are computed
    from the returned logprobs and rendered outputs — nothing is re-queried.

    :param cand: Candidate to evaluate.
    :param anchor_label: VLM prediction on the zero-genotype anchor.
    :param anchor_geno: Zero genotype (used for hamming distance).
    :param anchor_image_arr: Pixel array of the anchor image (for pixel L2).
    :param text_distance_fn: Closure ``(rendered_text) → float`` returning
        cosine distance in the SUT's sentence-embedding space between the
        manipulated prompt and the anchor prompt (cached by the embedder).
    :param image_dim: Number of image genes (split point in genotype).
    :param categories: Category tuple in the same order as logprobs.
    :param sut_call_fn: Closure ``(genotype) → (logprobs, sut_call_id,
        rendered_image, rendered_text, wall_time_cum)`` — makes one SUT call.
    :param input_distance_fn: ``(g, anchor_geno) → float``.
    :param output_distance_fn: ``(label_a, label_b) → float``.
    :returns: Fully populated :class:`ScoredCandidate`.
    """
    g = cand.genotype
    img_g = g[:image_dim]
    txt_g = g[image_dim:]

    # -- SUT call -----------------------------------------------------------
    logprobs_list, sut_call_id, rendered_image, rendered_text, wall_time_cum = (
        sut_call_fn(g)
    )

    label = categories[int(np.argmax(logprobs_list))]

    # -- Sparsity / rank ----------------------------------------------------
    img_sp = int(np.count_nonzero(img_g))
    txt_sp = int(np.count_nonzero(txt_g))
    img_rs = int(np.sum(img_g))
    txt_rs = int(np.sum(txt_g))

    # -- Distances ----------------------------------------------------------
    candidate_arr = np.array(rendered_image)
    pixel_l2 = image_pixel_l2(candidate_arr, anchor_image_arr)
    text_cos = float(text_distance_fn(rendered_text))

    d_i = input_distance_fn(g, anchor_geno)
    flipped = label != anchor_label
    d_o = float(output_distance_fn(label, anchor_label))
    pdq_val = pdq_metric(d_i, d_o)

    return ScoredCandidate(
        candidate=cand,
        candidate_id=sut_call_id,
        label=label,
        logprobs=logprobs_list,
        sut_call_id=sut_call_id,
        rendered_text=rendered_text,
        rendered_image=rendered_image,
        discovery_wall_time_cum=wall_time_cum,
        d_i=d_i,
        d_o=d_o,
        pdq_score=pdq_val,
        flipped=flipped,
        img_sparsity=img_sp,
        txt_sparsity=txt_sp,
        total_sparsity=img_sp + txt_sp,
        img_rank_sum=img_rs,
        txt_rank_sum=txt_rs,
        total_rank_sum=img_rs + txt_rs,
        hamming_to_anchor=hamming(g, anchor_geno),
        image_pixel_L2=pixel_l2,
        text_cosine_sum=text_cos,
    )
