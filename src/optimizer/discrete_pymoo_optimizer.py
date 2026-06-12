"""Discrete PyMoo optimizer for integer genotypes with per-gene bounds.

Wraps PyMoo's AGEMOEA2 (or any compatible genetic algorithm) for
integer-valued search spaces where each gene has its own upper bound.
This is needed when different genes represent indices into candidate
lists of varying size -- e.g. codebook indices for image patches vs.
synonym indices for text tokens.
"""

from __future__ import annotations

import logging
from typing import Any, Type

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.problems.static import StaticProblem
from smoo.optimizer import Optimizer
from smoo.optimizer.auxiliary_components import OptimizerCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default operator parameters (historical hardcoded values)
# ---------------------------------------------------------------------------

DEFAULT_MUTATION_ETA = 3.0
DEFAULT_CROSSOVER_PROB = 0.9
DEFAULT_CROSSOVER_ETA = 3.0

_DEFAULT_SAMPLING = IntegerRandomSampling()


def build_mutation(
    prob: float | None = None,
    eta: float = DEFAULT_MUTATION_ETA,
) -> PM:
    """Build the integer-aware PM operator.

    :param prob: Per-gene mutation probability (PyMoo ``prob_var``).
        ``None`` keeps PyMoo's adaptive default ``min(0.5, 1/n_var)``.
    :param eta: PM distribution index (smaller → larger jumps).
    :returns: PM operator with rounding repair, matching the historical
        default ``PM(eta=3.0)`` when called with no arguments.
    """
    kwargs: dict[str, Any] = {}
    if prob is not None:
        kwargs["prob_var"] = prob
    return PM(eta=eta, vtype=float, repair=RoundingRepair(), **kwargs)


def build_crossover(
    prob: float = DEFAULT_CROSSOVER_PROB,
    eta: float = DEFAULT_CROSSOVER_ETA,
) -> SBX:
    """Build the integer-aware SBX operator.

    :param prob: Per-mating crossover application probability.
    :param eta: SBX distribution index (smaller → offspring further
        from parents).
    :returns: SBX operator with rounding repair, matching the historical
        default ``SBX(prob=0.9, eta=3.0)`` when called with no arguments.
    """
    return SBX(prob=prob, eta=eta, vtype=float, repair=RoundingRepair())


class DiscretePymooOptimizer(Optimizer):
    """PyMoo optimizer for discrete search spaces with per-gene bounds.

    Operates directly in integer space.  Each gene has its own upper bound,
    enabling mixed search spaces (e.g. codebook indices + synonym indices).

    Uses AGEMOEA2 by default with integer-aware operators:

    - :class:`IntegerRandomSampling` for initial population
    - :class:`SBX` crossover + :class:`PM` mutation with :class:`RoundingRepair`

    :param gene_bounds: 1-D array of exclusive upper bounds per gene.
        Gene *i* takes values in ``[0, gene_bounds[i])``.
    :param num_objectives: Number of objectives to minimize.
    :param pop_size: Population size for the genetic algorithm.
    :param algorithm: PyMoo algorithm class (default :class:`AGEMOEA2`).
    :param algo_params: Extra keyword arguments forwarded to the algorithm
        constructor.  Keys ``sampling``, ``crossover``, and ``mutation``
        override the default integer-aware operators (including any
        ``mutation_*`` / ``crossover_*`` scalar settings below).
    :param mutation_prob: Per-gene mutation probability (PyMoo
        ``prob_var``). ``None`` (default) keeps PyMoo's adaptive default
        ``min(0.5, 1/n_var)``.
    :param mutation_eta: PM distribution index (default ``3.0``,
        the historical hardcoded value).
    :param crossover_prob: SBX per-mating application probability
        (default ``0.9``, the historical hardcoded value).
    :param crossover_eta: SBX distribution index (default ``3.0``,
        the historical hardcoded value).
    """

    _pymoo_algo: GeneticAlgorithm
    _problem: Problem
    _pop_current: Population
    _gene_bounds: NDArray
    _pop_size: int
    _algorithm_cls: Type[GeneticAlgorithm]
    _algo_params: dict[str, Any]
    _mutation_op: PM
    _crossover_op: SBX

    def __init__(
        self,
        gene_bounds: NDArray,
        num_objectives: int,
        pop_size: int = 50,
        algorithm: Type[GeneticAlgorithm] = AGEMOEA2,
        algo_params: dict[str, Any] | None = None,
        mutation_prob: float | None = None,
        mutation_eta: float = DEFAULT_MUTATION_ETA,
        crossover_prob: float = DEFAULT_CROSSOVER_PROB,
        crossover_eta: float = DEFAULT_CROSSOVER_ETA,
    ) -> None:
        super().__init__(num_objectives)

        self._gene_bounds = np.asarray(gene_bounds, dtype=np.int64)
        self._pop_size = pop_size
        self._algorithm_cls = algorithm
        self._algo_params = dict(algo_params) if algo_params else {}
        self._mutation_op = build_mutation(prob=mutation_prob, eta=mutation_eta)
        self._crossover_op = build_crossover(
            prob=crossover_prob, eta=crossover_eta
        )
        self._n_var = len(self._gene_bounds)

        # Defer initialization when bounds are placeholder zeros —
        # _init_algorithm will be called via update_gene_bounds with real bounds.
        if self._gene_bounds.any():
            self._init_algorithm()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_params(self) -> dict[str, Any]:
        """Merge user params with integer-aware defaults.

        :returns: Keyword dict ready for the algorithm constructor.
        """
        params: dict[str, Any] = {
            "pop_size": self._pop_size,
            "sampling": _DEFAULT_SAMPLING,
            "crossover": self._crossover_op,
            "mutation": self._mutation_op,
        }
        params.update(self._algo_params)
        return params

    def _init_algorithm(self) -> None:
        """Create the PyMoo problem and algorithm, then draw initial pop."""
        xl = np.zeros(self._n_var, dtype=np.int64)
        xu = self._gene_bounds - 1  # PyMoo upper bounds are inclusive

        self._problem = Problem(
            n_var=self._n_var,
            n_obj=self._num_objectives,
            xl=xl,
            xu=xu,
            vtype=int,
        )

        params = self._build_params()
        self._pymoo_algo = self._algorithm_cls(**params)
        self._pymoo_algo.setup(self._problem, termination=NoTermination())

        # Draw the first population.
        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._sanitize(self._pop_current.get("X"))

        self._best_candidates = [
            OptimizerCandidate(
                solution=self._x_current[0].copy(),
                fitness=[np.inf] * self._num_objectives,
            )
        ]
        self._previous_best = self._best_candidates.copy()

    def _sanitize(self, x: NDArray) -> NDArray:
        """Round, clip, and cast a population matrix to valid integers.

        :param x: Raw population matrix of shape ``(pop_size, n_var)``.
        :returns: Cleaned integer array with every gene within bounds.
        """
        x = np.round(x).astype(np.int64)
        x = np.clip(x, 0, self._gene_bounds - 1)
        return x

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Advance one generation: tell fitness, ask for new population."""
        logger.debug("Telling fitness and asking for next generation.")
        static = StaticProblem(self._problem, F=np.column_stack(self._fitness))
        Evaluator().eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._sanitize(self._pop_current.get("X"))

    def get_x_current(self) -> NDArray:
        """Return the current population as an integer array.

        :returns: Array of shape ``(pop_size, n_var)`` with dtype int64.
        """
        return self._x_current

    def reset(self) -> None:
        """Re-initialize the algorithm and draw a fresh population."""
        self._init_algorithm()

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def update_gene_bounds(self, new_bounds: NDArray) -> None:
        """Replace gene bounds and re-build the problem and algorithm.

        Use this when switching to a different seed whose genotype has
        different dimensions or candidate counts.

        :param new_bounds: New exclusive upper-bound array.
        """
        self._gene_bounds = np.asarray(new_bounds, dtype=np.int64)
        self._n_var = len(self._gene_bounds)
        self._init_algorithm()

    def set_sampling(self, sampling: Any) -> None:
        """Replace the sampling strategy for subsequent population draws.

        Takes effect at the next call to :meth:`update_gene_bounds` (or any
        other path that triggers ``_init_algorithm``). The sampling object
        may be any PyMoo-compatible ``Sampling`` instance *or* a
        pre-generated ``(n, n_var)`` integer matrix.

        :param sampling: New sampling strategy.
        """
        self._algo_params = {**self._algo_params, "sampling": sampling}

    def set_initial_population(
        self,
        sampling: NDArray,
        pop_size: int | None = None,
    ) -> None:
        """Re-initialize the algorithm with an explicit initial population.

        Used by the EXP-08 screening pipeline to inject fuzzy / precise /
        Pareto-init seed matrices after gene bounds are known for the
        current seed. Supersedes any ``sampling`` entry in
        ``algo_params``; the effective ``pop_size`` becomes
        ``len(sampling)`` unless explicitly overridden.

        :param sampling: Integer ``(n, n_var)`` array. Each row becomes
            one individual in the initial population.
        :param pop_size: Optional override for the algorithm's steady-state
            pop size. If ``None``, defaults to ``len(sampling)``.
        """
        sampling = np.asarray(sampling, dtype=np.int64)
        if sampling.ndim != 2 or sampling.shape[1] != self._n_var:
            raise ValueError(
                f"sampling must have shape (n, {self._n_var}); "
                f"got {sampling.shape}"
            )
        effective_pop = pop_size if pop_size is not None else len(sampling)
        self._pop_size = effective_pop
        self._algo_params = {
            **self._algo_params,
            "sampling": sampling,
            "pop_size": effective_pop,
        }
        self._init_algorithm()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gene_bounds(self) -> NDArray:
        """Per-gene exclusive upper bounds.

        :returns: 1-D int64 array of length ``n_var``.
        """
        return self._gene_bounds

    @property
    def pop_size(self) -> int:
        """Current population size.

        :returns: Number of individuals per generation.
        """
        return self._pop_size
