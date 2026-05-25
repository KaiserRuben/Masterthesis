"""Stage 2 flip minimisation — three passes + orchestrator.

Given a Stage-1 flip genotype ``g_f``, Stage 2 finds the smallest
perturbation ``g_min`` such that the VLM still predicts a label different
from the anchor label.

The "anchor" / "target" of minimisation is configurable via the
``target`` parameter of each pass (and ``anchor_geno`` on
:func:`minimise_flip`). The default behaviour minimises toward the
**zero genotype** — the canonical PDQ semantics. When the combined
pipeline feeds a non-zero anchor (an evolutionary balanced individual),
passing that anchor as ``target`` drives Stage 2 toward minimising
``|g − anchor|`` instead.

Three passes run in order A → B → C (C disabled by default):

    Pass A — greedy zero-toward-target
        Try setting each gene that differs from *target* directly to
        ``target[i]``.  Keep if the flip survives.  Reduces the number
        of differing positions ("sparsity" in delta-from-target sense).
        Processes genes in ``"by_gene_value_desc"`` order (largest abs
        diff from target first) so each accepted step yields the largest
        d_i reduction per call.

    Pass B — rank step-toward-target
        For each gene that still differs from *target*, move the value
        toward ``target[i]`` by ``step_size`` (default 1) while the flip
        is preserved.  Stops a gene when a step breaks the flip or the
        gene reaches ``target[i]``.  Reduces rank_sum without necessarily
        eliminating the difference at every position.

    Pass C — random-subset zero-toward-target (optional, default disabled)
        Sample random subsets of differing positions and set them all to
        ``target[subset]`` in one call.  Aggressive shortcut when many
        genes are "jointly snappable" but individually load-bearing.

Each pass is a pure function returning ``(best_genotype, trajectory)``.
The trajectory is a list of per-step dicts ready for conversion to
``stage2_trajectories.parquet`` rows.

``CheckResult`` is the internal return type of the SUT check closure.
It carries the label and SUT adapter state so trajectory rows can
include ``label_after``, ``sut_call_id``, and ``wall_time_cumulative_s``
without an extra adapter reference inside the pass functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
from tqdm import tqdm

from ..config import Stage2Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class CheckResult(NamedTuple):
    """Result of one Stage-2 SUT check call.

    :param still_flipped: Whether the tested genotype crosses the class
        boundary (VLM label ≠ anchor label).
    :param label: Top-1 label returned by the VLM for the tested genotype.
    :param sut_call_id: Monotonic call ID from the SUT adapter.
    :param wall_time_cum: Cumulative SUT wall time at call completion (s).
    """

    still_flipped: bool
    label: str
    sut_call_id: int
    wall_time_cum: float


@dataclass(eq=False)
class Stage2Result:
    """Result of one Stage-2 minimisation run.

    :param genotype_min: Minimal-perturbation genotype that still flips
        the VLM label.  Equals ``genotype_flipped`` when no minimisation
        was possible within the budget.
    :param d_i_min: Input distance of ``genotype_min`` (same metric as
        Stage 1 ``d_i_primary``).
    :param sut_calls_used: Total SUT calls consumed across all passes.
    :param trajectory_per_pass: Mapping from pass name (``"zero"``,
        ``"rank"``, ``"random_subset"``) to a list of per-step dicts.
        Each dict contains all fields needed for a
        ``stage2_trajectories.parquet`` row.
    :param stopped_reason: Human-readable stop reason (``"all_passes_complete"``,
        ``"budget_exhausted"``, ``"no_nonzero_genes"``,
        ``"no_passes_enabled"``).
    :param final_label: VLM label of ``genotype_min`` — last accepted label
        from the trajectory, or the Stage-1 flip label when no steps were
        accepted.
    """

    genotype_min: np.ndarray
    d_i_min: float
    sut_calls_used: int
    trajectory_per_pass: dict[str, list[dict[str, Any]]]
    stopped_reason: str
    final_label: str


# ---------------------------------------------------------------------------
# Ordering helper
# ---------------------------------------------------------------------------


def _ordered_differs(
    g: np.ndarray, target: np.ndarray, order: str,
) -> np.ndarray:
    """Return indices of genes that differ from *target* in the requested order.

    When *target* is the zero genotype this reduces to "non-zero gene
    indices" (legacy behaviour).  When *target* is a non-zero anchor
    (e.g. an evolutionary balanced individual) it returns the positions
    of the *delta* — exactly the positions Stage 2 can productively
    touch when minimising ``|g − target|``.

    :param g: Genotype array.
    :param target: Reference genotype.  Indices where ``g == target`` are
        excluded (no work to do there).
    :param order: ``"by_gene_value_desc"`` (largest ``|g − target|``
        first) or any other string (ascending position order).
    :returns: Array of gene indices where ``g != target``.
    """
    indices = np.where(g != target)[0]
    if order == "by_gene_value_desc" and len(indices) > 0:
        diffs = np.abs(g[indices].astype(np.int64) - target[indices].astype(np.int64))
        return indices[np.argsort(diffs)[::-1]]
    return indices


def _resolve_target(
    target: np.ndarray | None, like: np.ndarray,
) -> np.ndarray:
    """Return *target* or a zero array shaped like *like*.

    Default ``target=None`` preserves the legacy "minimise toward zero"
    behaviour without forcing callers to construct a zero array.
    """
    if target is None:
        return np.zeros_like(like)
    return np.asarray(target, dtype=like.dtype)


def _delta_rank_sum(g: np.ndarray, target: np.ndarray) -> int:
    """Sum of ``|g − target|`` — anchor-aware rank_sum."""
    return int(np.sum(np.abs(g.astype(np.int64) - target.astype(np.int64))))


def _delta_sparsity(g: np.ndarray, target: np.ndarray) -> int:
    """Number of positions where ``g`` differs from *target*."""
    return int(np.sum(g != target))


# ---------------------------------------------------------------------------
# Pass A — greedy zeroing
# ---------------------------------------------------------------------------


def _update_pbar(
    pbar: tqdm | None,
    pass_name: str,
    rank_sum: int,
    sparsity: int,
    trajectory: list[dict[str, Any]],
) -> None:
    if not pbar:
        return
    # Compute 50-step accept rate.
    recent = trajectory[-50:]
    acc_rate = (
        sum(1 for s in recent if s.get("accepted")) / len(recent) if recent else 0.0
    )
    pbar.set_postfix({
        "pass": pass_name,
        "rs": rank_sum,
        "sp": sparsity,
        "acc50": f"{acc_rate:.1%}",
    })
    pbar.update(1)


def greedy_zeroing(
    flipped_geno: np.ndarray,
    flip_check_fn: Callable[[np.ndarray], CheckResult],
    budget: int,
    order: str = "by_gene_value_desc",
    full_sweep_only: bool = True,  # noqa: ARG001 — kept for config parity
    pbar: tqdm | None = None,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Pass A — snap each differing gene to ``target[i]`` in one step.

    Iterates positions where ``flipped_geno != target`` in *order*
    (default: largest absolute delta first).  At each position the gene
    is tentatively set to ``target[i]``; if the flip survives the change
    is kept, otherwise it is reverted.

    ``target=None`` (default) reduces to the legacy "zero each non-zero
    gene" behaviour — the canonical PDQ pass.  Passing a non-zero anchor
    (e.g. an evolutionary balanced individual) instead snaps each
    differing gene to that anchor's value, driving the partner toward
    the anchor while preserving the flip.

    The rank_sum / sparsity reported in the trajectory are measured as
    delta-from-target (``Σ|g − target|`` and ``Σ[g != target]``) so the
    numbers remain meaningful for both anchor modes.

    :param flipped_geno: Stage-1 flip genotype to minimise.
    :param flip_check_fn: ``(genotype) → CheckResult`` — one SUT call.
    :param budget: Maximum SUT calls for this pass.
    :param order: Gene traversal order.
    :param full_sweep_only: Unused; retained for API consistency with
        ``ZeroPassConfig``.
    :param pbar: Optional tqdm bar to update.
    :param target: Reference genotype.  ``None`` = zeros (legacy PDQ);
        non-zero = anchor-aware minimisation (combined pipeline).
    :returns: ``(best_geno, trajectory)``.
    """
    tgt = _resolve_target(target, flipped_geno)
    g = flipped_geno.copy()
    trajectory: list[dict[str, Any]] = []
    calls_used = 0

    cur_rank_sum = _delta_rank_sum(g, tgt)
    cur_sparsity = _delta_sparsity(g, tgt)

    for step_num, i in enumerate(_ordered_differs(flipped_geno, tgt, order)):
        if calls_used >= budget:
            break
        if g[i] == tgt[i]:
            continue  # already snapped to target by a prior accepted step

        old_val = int(g[i])
        target_val = int(tgt[i])
        g_try = g.copy()
        g_try[i] = target_val

        result = flip_check_fn(g_try)
        calls_used += 1

        rs_before = cur_rank_sum
        sp_before = cur_sparsity
        accepted = result.still_flipped
        if accepted:
            g = g_try
            cur_rank_sum -= abs(old_val - target_val)
            cur_sparsity -= 1  # position now matches target

        trajectory.append({
            "step": step_num,
            "pass_name": "zero",
            "target_gene": int(i),
            "old_value": old_val,
            "new_value": target_val,
            "still_flipped": result.still_flipped,
            "accepted": accepted,
            "label_after": result.label,
            "sut_call_id": result.sut_call_id,
            "wall_time_cumulative_s": result.wall_time_cum,
            "rank_sum_before": rs_before,
            "rank_sum_after": cur_rank_sum,
            "sparsity_before": sp_before,
            "sparsity_after": cur_sparsity,
        })

        _update_pbar(pbar, "zero", cur_rank_sum, cur_sparsity, trajectory)

        logger.debug(
            "Pass A step %d: gene=%d  %d→%d  accepted=%s  delta_rs=%d",
            step_num, i, old_val, target_val, accepted, cur_rank_sum,
        )

    return g, trajectory


# ---------------------------------------------------------------------------
# Pass B — rank reduction
# ---------------------------------------------------------------------------


def rank_reduction(
    flipped_geno: np.ndarray,
    flip_check_fn: Callable[[np.ndarray], CheckResult],
    budget: int,
    order: str = "by_gene_value_desc",
    step_size: int = 1,
    pbar: tqdm | None = None,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Pass B — step each differing gene toward ``target[i]`` by ``step_size``.

    For each gene where ``g[i] != target[i]`` visited in *order*, attempts
    repeated steps of size ``step_size`` toward ``target[i]`` while the
    flip is preserved.  Stops a gene when a step breaks the flip or when
    the gene reaches ``target[i]`` exactly.  When a step reaches the
    target value and is accepted the position joins the "snapped" set —
    this complements Pass A by cleaning up genes whose direct snap broke
    the flip but a partial step survives.

    ``target=None`` reduces to the legacy decrement-toward-zero behaviour.
    With a non-zero anchor *target* the step direction is
    ``sign(target[i] − g[i])`` so the gene moves toward the anchor from
    either side — increment when the partner is below the anchor,
    decrement when above.

    :param flipped_geno: Genotype to minimise (typically output of Pass A).
    :param flip_check_fn: ``(genotype) → CheckResult`` — one SUT call.
    :param budget: Maximum SUT calls for this pass.
    :param order: Gene traversal order.
    :param step_size: Magnitude per step (``RankPassConfig.step``).
    :param pbar: Optional tqdm bar to update.
    :param target: Reference genotype.  ``None`` = zeros (legacy PDQ);
        non-zero = anchor-aware minimisation.
    :returns: ``(best_geno, trajectory)``.
    """
    tgt = _resolve_target(target, flipped_geno)
    g = flipped_geno.copy()
    trajectory: list[dict[str, Any]] = []
    calls_used = 0
    step_num = 0

    cur_rank_sum = _delta_rank_sum(g, tgt)
    cur_sparsity = _delta_sparsity(g, tgt)

    for i in _ordered_differs(flipped_geno, tgt, order):
        if calls_used >= budget:
            break
        if g[i] == tgt[i]:
            continue

        # Step gene i toward target while flip survives.
        while g[i] != tgt[i] and calls_used < budget:
            old_val = int(g[i])
            target_val = int(tgt[i])
            diff = target_val - old_val
            direction = 1 if diff > 0 else -1
            step = direction * min(step_size, abs(diff))
            new_val = old_val + step
            g_try = g.copy()
            g_try[i] = new_val

            result = flip_check_fn(g_try)
            calls_used += 1

            rs_before = cur_rank_sum
            sp_before = cur_sparsity
            accepted = result.still_flipped
            if accepted:
                g = g_try
                cur_rank_sum -= abs(step)
                if new_val == target_val:
                    cur_sparsity -= 1

            trajectory.append({
                "step": step_num,
                "pass_name": "rank",
                "target_gene": int(i),
                "old_value": old_val,
                "new_value": new_val,
                "still_flipped": result.still_flipped,
                "accepted": accepted,
                "label_after": result.label,
                "sut_call_id": result.sut_call_id,
                "wall_time_cumulative_s": result.wall_time_cum,
                "rank_sum_before": rs_before,
                "rank_sum_after": cur_rank_sum,
                "sparsity_before": sp_before,
                "sparsity_after": cur_sparsity,
            })
            step_num += 1

            _update_pbar(pbar, "rank", cur_rank_sum, cur_sparsity, trajectory)

            if not accepted:
                break  # Gene is as close to target as it can get; try next gene

    return g, trajectory


# ---------------------------------------------------------------------------
# Pass C — random-subset zeroing
# ---------------------------------------------------------------------------


def random_subset(
    flipped_geno: np.ndarray,
    flip_check_fn: Callable[[np.ndarray], CheckResult],
    budget: int,
    rng: np.random.Generator,
    subset_sizes: tuple[int, ...] = (2, 3, 5),
    n_trials_per_size: int = 20,
    pbar: tqdm | None = None,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Pass C — snap random subsets of differing genes to ``target[subset]``.

    Samples subsets of positions where ``g != target`` and tries snapping
    the entire subset to ``target`` in one SUT call.  If the flip survives,
    every position in the subset matches the target afterward.  Aggressive
    shortcut for clusters of genes that are individually load-bearing
    but jointly snappable.

    ``target=None`` reduces to the legacy "zero a random subset" behaviour.

    :param flipped_geno: Genotype to minimise (typically output of Passes A+B).
    :param flip_check_fn: ``(genotype) → CheckResult`` — one SUT call.
    :param budget: Maximum SUT calls for this pass.
    :param rng: Seeded random generator.
    :param subset_sizes: Subset sizes to try (from
        ``RandomSubsetPassConfig.subset_sizes``).
    :param n_trials_per_size: Trials per size (from
        ``RandomSubsetPassConfig.n_trials_per_size``).
    :param pbar: Optional tqdm bar to update.
    :param target: Reference genotype.  ``None`` = zeros (legacy PDQ);
        non-zero = anchor-aware minimisation.
    :returns: ``(best_geno, trajectory)``.
    """
    tgt = _resolve_target(target, flipped_geno)
    g = flipped_geno.copy()
    trajectory: list[dict[str, Any]] = []
    calls_used = 0
    step_num = 0

    cur_rank_sum = _delta_rank_sum(g, tgt)
    cur_sparsity = _delta_sparsity(g, tgt)

    for k in subset_sizes:
        for _ in range(n_trials_per_size):
            if calls_used >= budget:
                break
            differing_idx = np.where(g != tgt)[0]
            if len(differing_idx) < k:
                break

            chosen = rng.choice(differing_idx, size=k, replace=False)
            old_vals = g[chosen].copy()
            g_try = g.copy()
            g_try[chosen] = tgt[chosen]

            result = flip_check_fn(g_try)
            calls_used += 1

            rs_before = cur_rank_sum
            sp_before = cur_sparsity
            accepted = result.still_flipped
            if accepted:
                g = g_try
                cur_rank_sum = _delta_rank_sum(g, tgt)
                cur_sparsity = _delta_sparsity(g, tgt)

            trajectory.append({
                "step": step_num,
                "pass_name": "random_subset",
                "target_gene": chosen.tolist(),
                "old_value": old_vals.tolist(),
                "new_value": tgt[chosen].tolist(),
                "still_flipped": result.still_flipped,
                "accepted": accepted,
                "label_after": result.label,
                "sut_call_id": result.sut_call_id,
                "wall_time_cumulative_s": result.wall_time_cum,
                "rank_sum_before": rs_before,
                "rank_sum_after": cur_rank_sum,
                "sparsity_before": sp_before,
                "sparsity_after": cur_sparsity,
            })
            step_num += 1

            _update_pbar(
                pbar, "subset", cur_rank_sum, cur_sparsity, trajectory
            )

    return g, trajectory


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def minimise_flip(
    flipped_geno: np.ndarray,
    sut_check_fn: Callable[[np.ndarray], CheckResult],
    anchor_label: str,
    budget_total: int,
    rng: np.random.Generator,
    cfg: Stage2Config,
    input_distance_fn: Callable[[np.ndarray, np.ndarray], float],
    stage1_flip_label: str = "",
    seed_idx: int = 0,
    flip_id: int = 0,
    anchor_geno: np.ndarray | None = None,
) -> Stage2Result:
    """Run Stage-2 minimisation passes A → B → C on a Stage-1 flip.

    Budget is distributed evenly across enabled passes; the last enabled
    pass receives the remainder.  Budget is also tracked globally so that
    if Pass A consumes fewer calls than its allocation, the surplus rolls
    forward to Pass B (remaining-budget semantics).

    ``anchor_geno=None`` (default) minimises toward the zero genotype —
    the canonical PDQ semantics.  Passing a non-zero anchor (e.g. an
    evolutionary balanced individual) switches every pass to anchor-aware
    mode, minimising ``|g − anchor_geno|`` instead.

    :param flipped_geno: Stage-1 flip genotype to minimise.
    :param sut_check_fn: ``(genotype) → CheckResult`` — one SUT call per
        invocation.  Must update the adapter's call counter correctly.
    :param anchor_label: VLM anchor label (used only for logging).
    :param budget_total: Total SUT budget across all passes for this flip.
    :param rng: Seeded random generator (Pass C only).
    :param cfg: :class:`~src.pdq.config.Stage2Config` with pass flags and
        parameters.
    :param input_distance_fn: ``(g, anchor_geno) → float`` — computes d_i
        for the minimised genotype.
    :param stage1_flip_label: The Stage-1 flip label (used as fallback when
        no Stage-2 step was accepted).
    :param seed_idx: 0-based seed index for logging.
    :param flip_id: 0-based flip index within the seed.
    :param anchor_geno: Reference genotype to minimise toward.  ``None``
        = zero genotype (canonical PDQ).  Combined-pipeline callers pass
        the evolutionary anchor here.
    :returns: :class:`Stage2Result` with the minimised genotype, d_i_min,
        call count, per-pass trajectories, stop reason, and final label.
    """
    tgt = _resolve_target(anchor_geno, flipped_geno)

    # -- Determine enabled passes and their budget allocations --------------
    enabled: list[str] = []
    if cfg.passes.zero.enabled:
        enabled.append("zero")
    if cfg.passes.rank.enabled:
        enabled.append("rank")
    if cfg.passes.random_subset.enabled:
        enabled.append("random_subset")

    if not enabled:
        d_i = input_distance_fn(flipped_geno, tgt)
        return Stage2Result(
            genotype_min=flipped_geno.copy(),
            d_i_min=d_i,
            sut_calls_used=0,
            trajectory_per_pass={},
            stopped_reason="no_passes_enabled",
            final_label=stage1_flip_label,
        )

    if int(np.sum(flipped_geno != tgt)) == 0:
        d_i = input_distance_fn(flipped_geno, tgt)
        return Stage2Result(
            genotype_min=flipped_geno.copy(),
            d_i_min=d_i,
            sut_calls_used=0,
            trajectory_per_pass={},
            stopped_reason="no_nonzero_genes",
            final_label=stage1_flip_label,
        )

    # Even split; last pass gets remainder.
    n = len(enabled)
    base = budget_total // n
    pass_budgets: dict[str, int] = {}
    allocated = 0
    for idx, name in enumerate(enabled):
        if idx == n - 1:
            pass_budgets[name] = max(1, budget_total - allocated)
        else:
            pass_budgets[name] = max(1, base)
            allocated += pass_budgets[name]

    # -- Run passes sequentially, tracking remaining global budget ----------
    g = flipped_geno.copy()
    remaining = budget_total
    total_calls = 0
    trajectory_per_pass: dict[str, list[dict[str, Any]]] = {}
    stopped_reason = "all_passes_complete"

    pbar = tqdm(
        total=budget_total,
        desc=f"  S2 seed_{seed_idx:04d} flip_{flip_id}",
        unit="call",
        leave=False,
        position=1,
    )

    if cfg.passes.zero.enabled and remaining > 0:
        b = min(pass_budgets["zero"], remaining)
        g, traj = greedy_zeroing(
            g,
            sut_check_fn,
            budget=b,
            order=cfg.passes.zero.order,
            full_sweep_only=cfg.passes.zero.full_sweep_only,
            pbar=pbar,
            target=tgt,
        )
        trajectory_per_pass["zero"] = traj
        used = len(traj)
        total_calls += used
        remaining -= used
        logger.debug(
            "Stage2 Pass A: %d calls  delta_rank_sum=%d  delta_sparsity=%d",
            used, _delta_rank_sum(g, tgt), _delta_sparsity(g, tgt),
        )

    if cfg.passes.rank.enabled and remaining > 0:
        rank_traj: list[dict[str, Any]] = []
        for _sweep in range(cfg.passes.rank.max_sweeps):
            if remaining <= 0:
                stopped_reason = "budget_exhausted"
                break
            b = min(pass_budgets["rank"], remaining)
            g_before_sweep = g.copy()
            g, traj = rank_reduction(
                g,
                sut_check_fn,
                budget=b,
                order=cfg.passes.rank.order,
                step_size=cfg.passes.rank.step,
                pbar=pbar,
                target=tgt,
            )
            rank_traj.extend(traj)
            used = len(traj)
            total_calls += used
            remaining -= used
            if np.array_equal(g, g_before_sweep):
                break  # No progress — further sweeps won't help
        trajectory_per_pass["rank"] = rank_traj
        logger.debug(
            "Stage2 Pass B: %d calls  delta_rank_sum=%d  delta_sparsity=%d",
            len(rank_traj), _delta_rank_sum(g, tgt), _delta_sparsity(g, tgt),
        )

    if cfg.passes.random_subset.enabled and remaining > 0:
        b = min(pass_budgets.get("random_subset", remaining), remaining)
        g, traj = random_subset(
            g,
            sut_check_fn,
            budget=b,
            rng=rng,
            subset_sizes=cfg.passes.random_subset.subset_sizes,
            n_trials_per_size=cfg.passes.random_subset.n_trials_per_size,
            pbar=pbar,
            target=tgt,
        )
        trajectory_per_pass["random_subset"] = traj
        used = len(traj)
        total_calls += used
        remaining -= used
        logger.debug(
            "Stage2 Pass C: %d calls  delta_rank_sum=%d  delta_sparsity=%d",
            used, _delta_rank_sum(g, tgt), _delta_sparsity(g, tgt),
        )

    pbar.close()

    if remaining <= 0 and stopped_reason == "all_passes_complete":
        stopped_reason = "budget_exhausted"

    # -- Compute final d_i and label ----------------------------------------
    d_i_min = input_distance_fn(g, tgt)
    final_label = _last_accepted_label(trajectory_per_pass, stage1_flip_label)

    logger.debug(
        "Stage2 flip minimised: anchor=%s  d_i %s→%.1f  calls=%d  reason=%s",
        anchor_label,
        "?" if total_calls == 0 else str(int(input_distance_fn(flipped_geno, tgt))),
        d_i_min,
        total_calls,
        stopped_reason,
    )

    return Stage2Result(
        genotype_min=g,
        d_i_min=d_i_min,
        sut_calls_used=total_calls,
        trajectory_per_pass=trajectory_per_pass,
        stopped_reason=stopped_reason,
        final_label=final_label,
    )


def _last_accepted_label(
    trajectory_per_pass: dict[str, list[dict[str, Any]]],
    fallback: str,
) -> str:
    """Return the label from the last accepted trajectory step.

    Scans passes in reverse order (random_subset → rank → zero) and
    within each pass in reverse step order.  Falls back to *fallback*
    (Stage-1 flip label) if no step was accepted.

    :param trajectory_per_pass: Pass-name → list of step dicts.
    :param fallback: Label to return when no accepted step exists.
    :returns: Last accepted VLM label or *fallback*.
    """
    for pass_name in ("random_subset", "rank", "zero"):
        for step in reversed(trajectory_per_pass.get(pass_name, [])):
            if step.get("accepted"):
                lbl = step.get("label_after", "")
                if lbl:
                    return lbl
    return fallback
