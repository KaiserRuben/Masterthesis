"""Per-generation early-stopping monitor for the VLM boundary tester.

A self-contained state machine that consumes Pareto-front fitness
arrays each generation and decides whether the optimizer should stop
early. Used by the EXP-08 screening pipeline to cut wasted compute
on runs that either (a) already found a boundary flip, (b) stalled,
or (c) never improved over the random seed matrix.

Four OR-composed triggers are evaluated in order each generation;
the first one to fire is returned:

1. **Flip** — ``pareto_min_TgtBal`` dropped at or below
   ``np.finfo(dtype).tiny + epsilon_margin``. The decision boundary
   has been crossed; there is nothing more to search for.
2. **Plateau** — no Pareto hypervolume improvement for
   ``plateau_patience`` consecutive generations. Requires a
   user-supplied fixed reference point; otherwise the trigger is
   skipped entirely.
3. **No-improvement-since-seed** — after ``no_improvement_warmup``
   generations, ``pareto_min_TgtBal`` has not improved on the value
   recorded at generation 0 (the post-seed state). If the random
   seed matrix alone already found the min and evolution is not
   helping, there is nothing to learn by continuing.
4. **Hard cap** — ``generation >= max_generations``. Safety net that
   matches the pre-early-stop behaviour of the tester loop.

The monitor has no dependency on any stateful repo module; it is fed
plain NumPy arrays by the caller, so analysis code can replay a
finished run's convergence trace through a fresh checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - import guard
    from pymoo.indicators.hv import HV as _HV

    _HAS_PYMOO_HV = True
except ImportError:  # pragma: no cover - pymoo is a hard dep in prod
    _HV = None  # type: ignore[assignment]
    _HAS_PYMOO_HV = False


TriggerName = Literal["flip", "plateau", "no_improvement", "hard_cap"]


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarlyStopConfig:
    """Configuration for per-generation early-stopping checks.

    :param epsilon_margin: Additive slack on top of ``np.finfo(dtype).tiny``
        used as the flip threshold. The effective epsilon is therefore
        ``np.finfo(dtype).tiny + epsilon_margin``, tied to the precision
        of the fitness array's dtype. Default ``1e-30`` (comfortably
        above FP32 tiny, still far below any non-flipped TgtBal value).
    :param plateau_patience: ``K`` — window (in generations) of no
        hypervolume improvement after which the plateau trigger fires.
    :param no_improvement_warmup: Number of initial generations during
        which the no-improvement trigger is suppressed. Needed because
        generation 0 is itself the reference point; the trigger only
        makes sense once some evolution budget has been spent.
    :param hypervolume_reference: Fixed reference point for Pareto
        hypervolume, shape ``(n_objectives,)``. For a minimization
        problem all Pareto points should lie component-wise below this
        reference. ``None`` disables the plateau trigger entirely
        (cheap escape hatch; we never roll our own HV).
    :param max_generations: Hard cap; the checker always returns a
        ``hard_cap`` trigger at ``generation >= max_generations``.
    """

    epsilon_margin: float = 1e-30
    plateau_patience: int = 30
    no_improvement_warmup: int = 20
    hypervolume_reference: tuple[float, ...] | None = None
    max_generations: int = 200


@dataclass(frozen=True)
class EarlyStopTrigger:
    """Result payload returned when a trigger fires.

    :param trigger: Identifier of the trigger that fired.
    :param generation: Generation index at which the trigger fired.
    :param details: Diagnostic key/value pairs (e.g.
        ``{"pareto_min_TgtBal": 5.3e-38, "epsilon": 1.17e-38}``).
        Used by the Tier-3 analysis phase to reason about why a run
        stopped where it did.
    """

    trigger: TriggerName
    generation: int
    details: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class EarlyStopChecker:
    """Stateful early-stop monitor; call :meth:`update` each generation.

    Internal state kept between calls:

    * ``_tgtbal_history`` — ``pareto_min_TgtBal`` per generation (or
      ``nan`` for generations with an empty Pareto front).
    * ``_hv_history`` — Pareto hypervolume per generation under the
      configured reference point, or ``nan`` when the Pareto front is
      empty or when no reference point is configured.
    * ``_gen_history`` — generation indices seen so far, parallel to
      the two arrays above.
    * ``_plateau_counter`` — consecutive generations without HV
      improvement; reset to zero on any strict improvement.
    * ``_best_hv`` — running max of observed hypervolume values.

    The checker is stateless with respect to *which* fitness column is
    the flip metric — ``tgtbal_index`` is passed explicitly to
    :meth:`update` so the same checker can be reused across seeds
    with potentially different objective orderings.

    :param config: Immutable configuration.
    """

    def __init__(self, config: EarlyStopConfig) -> None:
        self._config = config

        self._tgtbal_history: list[float] = []
        self._hv_history: list[float] = []
        self._gen_history: list[int] = []

        self._plateau_counter: int = 0
        self._best_hv: float = float("-inf")

        self._hv_indicator = self._build_hv_indicator(
            config.hypervolume_reference,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hv_indicator(
        reference: tuple[float, ...] | None,
    ) -> object | None:
        """Construct the pymoo HV indicator once, or return ``None``.

        :param reference: Reference point or ``None``.
        :returns: A ``pymoo.indicators.hv.HV`` instance, or ``None`` if
            no reference was supplied or pymoo is unavailable.
        :raises RuntimeError: If a reference point is configured but
            pymoo is not importable.
        """
        if reference is None:
            return None
        if not _HAS_PYMOO_HV:  # pragma: no cover - pymoo is a hard dep
            raise RuntimeError(
                "pymoo.indicators.hv.HV is required for hypervolume-based"
                " plateau detection but could not be imported."
            )
        ref = np.asarray(reference, dtype=np.float64)
        return _HV(ref_point=ref)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        generation: int,
        pareto_fitness: NDArray,
        tgtbal_index: int,
    ) -> EarlyStopTrigger | None:
        """Check all four triggers; return the first one that fires.

        :param generation: 0-based generation index of the call.
        :param pareto_fitness: Non-dominated fitness matrix of shape
            ``(n_pareto, n_objectives)``. An empty front (rows==0) is
            accepted — history is extended with ``nan`` and no trigger
            except ``hard_cap`` can fire on that call.
        :param tgtbal_index: Column index of the TargetedBalance
            objective in ``pareto_fitness``. Used for the flip and
            no-improvement-since-seed triggers.
        :returns: The first :class:`EarlyStopTrigger` that fired, or
            ``None`` if the optimizer should continue.
        :raises ValueError: If ``pareto_fitness`` is not 2D, or if
            ``tgtbal_index`` is out of range for a non-empty front.
        """
        pareto_fitness = np.asarray(pareto_fitness)
        if pareto_fitness.ndim != 2:
            raise ValueError(
                "pareto_fitness must be 2D of shape (n_pareto, n_objectives);"
                f" got shape {pareto_fitness.shape!r}."
            )

        n_pareto = pareto_fitness.shape[0]
        non_empty = n_pareto > 0

        if non_empty and not (
            0 <= tgtbal_index < pareto_fitness.shape[1]
        ):
            raise ValueError(
                "tgtbal_index out of range: got"
                f" {tgtbal_index} for {pareto_fitness.shape[1]} objectives."
            )

        # -- Update per-generation history (even on empty fronts) ------
        tgtbal_min = (
            float(pareto_fitness[:, tgtbal_index].min())
            if non_empty
            else float("nan")
        )
        hv_value = (
            self._compute_hv(pareto_fitness)
            if non_empty and self._hv_indicator is not None
            else float("nan")
        )

        self._gen_history.append(int(generation))
        self._tgtbal_history.append(tgtbal_min)
        self._hv_history.append(hv_value)

        # Update plateau bookkeeping. An empty front contributes a nan
        # HV — we treat that as "no improvement" without incrementing
        # the counter, so an empty generation doesn't poison plateau
        # detection.
        if not np.isnan(hv_value):
            if hv_value > self._best_hv:
                self._best_hv = hv_value
                self._plateau_counter = 0
            else:
                self._plateau_counter += 1

        # -- Trigger evaluation (order matters: flip > plateau >
        #    no-improvement > hard cap) ------------------------------
        if non_empty:
            flip = self._check_flip(generation, pareto_fitness, tgtbal_index)
            if flip is not None:
                return flip

            plateau = self._check_plateau(generation)
            if plateau is not None:
                return plateau

            no_imp = self._check_no_improvement(generation)
            if no_imp is not None:
                return no_imp

        if generation >= self._config.max_generations - 1:
            # Emit at the boundary: generation counts are 0-based, so
            # the cap fires on the last allowed generation.
            return EarlyStopTrigger(
                trigger="hard_cap",
                generation=int(generation),
                details={
                    "generation": float(generation),
                    "max_generations": float(self._config.max_generations),
                },
            )

        return None

    @property
    def history(self) -> dict[str, list[float]]:
        """Return the per-generation history.

        :returns: Dict with three parallel lists ``gen``,
            ``pareto_min_tgtbal`` and ``hypervolume``. All three have
            identical length — one entry per :meth:`update` call.
            ``pareto_min_tgtbal`` and ``hypervolume`` may contain
            ``nan`` for generations with an empty Pareto front (or
            when no HV reference is configured).
        """
        return {
            "gen": [float(g) for g in self._gen_history],
            "pareto_min_tgtbal": list(self._tgtbal_history),
            "hypervolume": list(self._hv_history),
        }

    # ------------------------------------------------------------------
    # Trigger implementations
    # ------------------------------------------------------------------

    def _check_flip(
        self,
        generation: int,
        pareto_fitness: NDArray,
        tgtbal_index: int,
    ) -> EarlyStopTrigger | None:
        """Flip trigger: TgtBal minimum at/under dtype-tied epsilon."""
        col = pareto_fitness[:, tgtbal_index]
        tgtbal_min = float(col.min())
        epsilon = self._epsilon_for(col.dtype)
        if tgtbal_min <= epsilon:
            return EarlyStopTrigger(
                trigger="flip",
                generation=int(generation),
                details={
                    "pareto_min_TgtBal": tgtbal_min,
                    "epsilon": float(epsilon),
                    "dtype_tiny": float(np.finfo(col.dtype).tiny),
                    "epsilon_margin": float(self._config.epsilon_margin),
                },
            )
        return None

    def _check_plateau(self, generation: int) -> EarlyStopTrigger | None:
        """Plateau trigger: no HV gain for plateau_patience generations."""
        if self._hv_indicator is None:
            return None
        if self._plateau_counter < self._config.plateau_patience:
            return None
        return EarlyStopTrigger(
            trigger="plateau",
            generation=int(generation),
            details={
                "plateau_counter": float(self._plateau_counter),
                "plateau_patience": float(self._config.plateau_patience),
                "best_hypervolume": float(self._best_hv),
                "current_hypervolume": float(self._hv_history[-1]),
            },
        )

    def _check_no_improvement(
        self,
        generation: int,
    ) -> EarlyStopTrigger | None:
        """No-improvement trigger: min TgtBal never beat generation 0.

        Activates only after ``no_improvement_warmup`` generations and
        only if at least generation 0 has a finite TgtBal value.
        """
        if generation < self._config.no_improvement_warmup:
            return None
        # Need at least one historical entry (generation 0) to compare
        # against the current one.
        if len(self._tgtbal_history) < 2:
            return None

        seed_value = self._tgtbal_history[0]
        current_value = self._tgtbal_history[-1]
        if not np.isfinite(seed_value) or not np.isfinite(current_value):
            return None

        # Strict improvement required: equal means no improvement.
        if current_value < seed_value:
            return None

        return EarlyStopTrigger(
            trigger="no_improvement",
            generation=int(generation),
            details={
                "pareto_min_TgtBal_gen0": float(seed_value),
                "pareto_min_TgtBal_current": float(current_value),
                "warmup": float(self._config.no_improvement_warmup),
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _epsilon_for(self, dtype: np.dtype) -> float:
        """Compute the flip epsilon for a given floating dtype.

        Non-floating dtypes (e.g. int, bool) fall back to FP64 tiny
        so the caller gets a sensible, extremely small threshold
        rather than a crash.

        :param dtype: NumPy dtype of the TgtBal fitness column.
        :returns: ``np.finfo(dtype).tiny + epsilon_margin`` as a Python
            float.
        """
        try:
            tiny = float(np.finfo(dtype).tiny)
        except ValueError:
            tiny = float(np.finfo(np.float64).tiny)
        return tiny + float(self._config.epsilon_margin)

    def _compute_hv(self, pareto_fitness: NDArray) -> float:
        """Call the pymoo HV indicator; guarded for degenerate fronts."""
        assert self._hv_indicator is not None  # guarded by caller
        try:
            value = float(self._hv_indicator(pareto_fitness.astype(np.float64)))
        except Exception:  # pragma: no cover - defensive
            return float("nan")
        return value
