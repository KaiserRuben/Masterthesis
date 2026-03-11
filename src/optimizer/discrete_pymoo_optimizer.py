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
# Default operator factories
# ---------------------------------------------------------------------------

_DEFAULT_CROSSOVER = SBX(prob=0.9, eta=3.0, vtype=float, repair=RoundingRepair())
_DEFAULT_MUTATION = PM(eta=3.0, vtype=float, repair=RoundingRepair())
_DEFAULT_SAMPLING = IntegerRandomSampling()


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
        override the default integer-aware operators.
    """

    _pymoo_algo: GeneticAlgorithm
    _problem: Problem
    _pop_current: Population
    _gene_bounds: NDArray
    _pop_size: int
    _algorithm_cls: Type[GeneticAlgorithm]
    _algo_params: dict[str, Any]

    def __init__(
        self,
        gene_bounds: NDArray,
        num_objectives: int,
        pop_size: int = 50,
        algorithm: Type[GeneticAlgorithm] = AGEMOEA2,
        algo_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(num_objectives)

        self._gene_bounds = np.asarray(gene_bounds, dtype=np.int64)
        self._pop_size = pop_size
        self._algorithm_cls = algorithm
        self._algo_params = dict(algo_params) if algo_params else {}
        self._n_var = len(self._gene_bounds)

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
            "crossover": _DEFAULT_CROSSOVER,
            "mutation": _DEFAULT_MUTATION,
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
        logger.info("Telling fitness and asking for next generation.")
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
