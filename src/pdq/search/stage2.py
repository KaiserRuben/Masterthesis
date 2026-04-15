"""Stage 2 flip minimisation — three passes + orchestrator.

Given a Stage-1 flip genotype ``g_f``, Stage 2 finds the smallest
perturbation ``g_min`` such that the VLM still predicts a label different
from the anchor label.

Three passes run in order A → B → C (C disabled by default):

    Pass A — greedy zeroing
        Try setting each non-zero gene to 0.  Keep if the flip survives.
        Reduces genotype sparsity (number of active genes).  Processes
        genes in ``"by_gene_value_desc"`` order (highest rank first) so
        each accepted zeroing yields the largest d_i reduction per call.

    Pass B — rank reduction
        For each remaining non-zero gene, decrement the value by
        ``step_size`` (default 1) while the flip is preserved.  Stops
        reducing a gene when a decrement breaks the flip or the gene
        reaches 0.  Reduces rank_sum (magnitude) without necessarily
        eliminating genes.

    Pass C — random-subset zeroing (optional, default disabled)
        Sample random subsets of non-zero genes and zero them all at
        once.  Aggressive shortcut when many genes are "jointly removable"
        but individually load-bearing.

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


def _ordered_nonzero(g: np.ndarray, order: str) -> np.ndarray:
    """Return indices of non-zero genes in the requested order.

    :param g: Genotype array.
    :param order: ``"by_gene_value_desc"`` (largest value first) or
        any other string (ascending position order).
    :returns: Array of non-zero gene indices.
    """
    indices = np.where(g != 0)[0]
    if order == "by_gene_value_desc" and len(indices) > 0:
        return indices[np.argsort(g[indices])[::-1]]
    return indices


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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Greedy zeroing pass: try setting each non-zero gene to 0.

    Iterates non-zero gene positions in *order* (default: highest gene
    value first).  At each position the gene is tentatively set to 0; if
    the flip survives the change is kept, otherwise it is reverted.

    *full_sweep_only* is accepted for config-field parity but does not
    alter behaviour in this implementation: every gene is visited at most
    once (one sweep through non-zero positions).

    :param flipped_geno: Stage-1 flip genotype to minimise.
    :param flip_check_fn: ``(genotype) → CheckResult`` — one SUT call.
    :param budget: Maximum SUT calls for this pass.
    :param order: Gene traversal order.
    :param full_sweep_only: Unused; retained for API consistency with
        ``ZeroPassConfig``.
    :param pbar: Optional tqdm bar to update.
    :returns: ``(best_geno, trajectory)``.
    """
    g = flipped_geno.copy()
    trajectory: list[dict[str, Any]] = []
    calls_used = 0

    cur_rank_sum = int(np.sum(g))
    cur_sparsity = int(np.count_nonzero(g))

    for step_num, i in enumerate(_ordered_nonzero(flipped_geno, order)):
        if calls_used >= budget:
            break
        if g[i] == 0:
            continue  # already zeroed by a prior accepted step

        old_val = int(g[i])
        g_try = g.copy()
        g_try[i] = 0

        result = flip_check_fn(g_try)
        calls_used += 1

        rs_before = cur_rank_sum
        sp_before = cur_sparsity
        accepted = result.still_flipped
        if accepted:
            g = g_try
            cur_rank_sum -= old_val
            cur_sparsity -= 1

        trajectory.append({
            "step": step_num,
            "pass_name": "zero",
            "target_gene": int(i),
            "old_value": old_val,
            "new_value": 0,
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
            "Pass A step %d: gene=%d  %d→0  accepted=%s  rank_sum=%d",
            step_num, i, old_val, accepted, cur_rank_sum,
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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rank-reduction pass: decrement each non-zero gene by *step_size*.

    For each non-zero gene visited in *order*, attempts to reduce its value
    by *step_size* while the flip is preserved.  Stops reducing a gene
    when a decrement breaks the flip or the gene reaches 0.  If a decrement
    reaches 0 and is accepted, the gene is effectively zeroed — this
    complements Pass A by cleaning up low-rank genes that Pass A left
    because zeroing them individually was too aggressive.

    :param flipped_geno: Genotype to minimise (typically output of Pass A).
    :param flip_check_fn: ``(genotype) → CheckResult`` — one SUT call.
    :param budget: Maximum SUT calls for this pass.
    :param order: Gene traversal order.
    :param step_size: Decrement per attempt (``RankPassConfig.step``).
    :param pbar: Optional tqdm bar to update.
    :returns: ``(best_geno, trajectory)``.
    """
    g = flipped_geno.copy()
    trajectory: list[dict[str, Any]] = []
    calls_used = 0
    step_num = 0

    cur_rank_sum = int(np.sum(g))
    cur_sparsity = int(np.count_nonzero(g))

    for i in _ordered_nonzero(flipped_geno, order):
        if calls_used >= budget:
            break
        if g[i] == 0:
            continue

        # Reduce gene i step by step while flip survives.
        while g[i] > 0 and calls_used < budget:
            old_val = int(g[i])
            new_val = max(0, old_val - step_size)
            g_try = g.copy()
            g_try[i] = new_val

            result = flip_check_fn(g_try)
            calls_used += 1

            rs_before = cur_rank_sum
            sp_before = cur_sparsity
            accepted = result.still_flipped
            if accepted:
                g = g_try
                cur_rank_sum -= (old_val - new_val)
                if new_val == 0:
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
                break  # Gene is as low as it can go; try next gene

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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Random-subset zeroing pass: zero random groups of non-zero genes.

    Samples subsets of active gene positions and tries zeroing the entire
    subset simultaneously.  If the flip survives, all genes in the subset
    are zeroed.  This is an aggressive shortcut that can eliminate clusters
    of genes that are individually load-bearing but jointly dispensable.

    :param flipped_geno: Genotype to minimise (typically output of Passes A+B).
    :param flip_check_fn: ``(genotype) → CheckResult`` — one SUT call.
    :param budget: Maximum SUT calls for this pass.
    :param rng: Seeded random generator.
    :param subset_sizes: Subset sizes to try (from
        ``RandomSubsetPassConfig.subset_sizes``).
    :param n_trials_per_size: Trials per size (from
        ``RandomSubsetPassConfig.n_trials_per_size``).
    :param pbar: Optional tqdm bar to update.
    :returns: ``(best_geno, trajectory)``.
    """
    g = flipped_geno.copy()
    trajectory: list[dict[str, Any]] = []
    calls_used = 0
    step_num = 0

    cur_rank_sum = int(np.sum(g))
    cur_sparsity = int(np.count_nonzero(g))

    for k in subset_sizes:
        for _ in range(n_trials_per_size):
            if calls_used >= budget:
                break
            nonzero_idx = np.where(g != 0)[0]
            if len(nonzero_idx) < k:
                break

            chosen = rng.choice(nonzero_idx, size=k, replace=False)
            old_vals = g[chosen].copy()
            g_try = g.copy()
            g_try[chosen] = 0

            result = flip_check_fn(g_try)
            calls_used += 1

            rs_before = cur_rank_sum
            sp_before = cur_sparsity
            accepted = result.still_flipped
            if accepted:
                g = g_try
                cur_rank_sum = int(np.sum(g))
                cur_sparsity = int(np.count_nonzero(g))

            trajectory.append({
                "step": step_num,
                "pass_name": "random_subset",
                "target_gene": chosen.tolist(),
                "old_value": old_vals.tolist(),
                "new_value": 0,
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
) -> Stage2Result:
    """Run Stage-2 minimisation passes A → B → C on a Stage-1 flip.

    Budget is distributed evenly across enabled passes; the last enabled
    pass receives the remainder.  Budget is also tracked globally so that
    if Pass A consumes fewer calls than its allocation, the surplus rolls
    forward to Pass B (remaining-budget semantics).

    :param flipped_geno: Stage-1 flip genotype to minimise.
    :param sut_check_fn: ``(genotype) → CheckResult`` — one SUT call per
        invocation.  Must update the adapter's call counter correctly.
    :param anchor_label: VLM anchor label (used only for logging).
    :param budget_total: Total SUT budget across all passes for this flip.
    :param rng: Seeded random generator (Pass C only).
    :param cfg: :class:`~src.pdq.config.Stage2Config` with pass flags and
        parameters.
    :param input_distance_fn: ``(g, zero_anchor) → float`` — computes d_i
        for the minimised genotype.
    :param stage1_flip_label: The Stage-1 flip label (used as fallback when
        no Stage-2 step was accepted).
    :param seed_idx: 0-based seed index for logging.
    :param flip_id: 0-based flip index within the seed.
    :returns: :class:`Stage2Result` with the minimised genotype, d_i_min,
        call count, per-pass trajectories, stop reason, and final label.
    """
    anchor_geno = np.zeros_like(flipped_geno)

    # -- Determine enabled passes and their budget allocations --------------
    enabled: list[str] = []
    if cfg.passes.zero.enabled:
        enabled.append("zero")
    if cfg.passes.rank.enabled:
        enabled.append("rank")
    if cfg.passes.random_subset.enabled:
        enabled.append("random_subset")

    if not enabled:
        d_i = input_distance_fn(flipped_geno, anchor_geno)
        return Stage2Result(
            genotype_min=flipped_geno.copy(),
            d_i_min=d_i,
            sut_calls_used=0,
            trajectory_per_pass={},
            stopped_reason="no_passes_enabled",
            final_label=stage1_flip_label,
        )

    if int(np.count_nonzero(flipped_geno)) == 0:
        d_i = input_distance_fn(flipped_geno, anchor_geno)
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
        )
        trajectory_per_pass["zero"] = traj
        used = len(traj)
        total_calls += used
        remaining -= used
        logger.debug(
            "Stage2 Pass A: %d calls  rank_sum=%d  sparsity=%d",
            used, int(np.sum(g)), int(np.count_nonzero(g)),
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
            )
            rank_traj.extend(traj)
            used = len(traj)
            total_calls += used
            remaining -= used
            if np.array_equal(g, g_before_sweep):
                break  # No progress — further sweeps won't help
        trajectory_per_pass["rank"] = rank_traj
        logger.debug(
            "Stage2 Pass B: %d calls  rank_sum=%d  sparsity=%d",
            len(rank_traj), int(np.sum(g)), int(np.count_nonzero(g)),
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
        )
        trajectory_per_pass["random_subset"] = traj
        used = len(traj)
        total_calls += used
        remaining -= used
        logger.debug(
            "Stage2 Pass C: %d calls  rank_sum=%d  sparsity=%d",
            used, int(np.sum(g)), int(np.count_nonzero(g)),
        )

    pbar.close()

    if remaining <= 0 and stopped_reason == "all_passes_complete":
        stopped_reason = "budget_exhausted"

    # -- Compute final d_i and label ----------------------------------------
    d_i_min = input_distance_fn(g, anchor_geno)
    final_label = _last_accepted_label(trajectory_per_pass, stage1_flip_label)

    logger.debug(
        "Stage2 flip minimised: anchor=%s  d_i %s→%.1f  calls=%d  reason=%s",
        anchor_label,
        "?" if total_calls == 0 else str(int(input_distance_fn(flipped_geno, anchor_geno))),
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
