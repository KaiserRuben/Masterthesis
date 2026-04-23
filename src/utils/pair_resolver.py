"""Resolve a class-pair name into ``SeedConfig.filter_indices`` positions.

Takes a human-readable pair string such as ``"junco->chickadee"`` (or the
shell-friendlier ``"great_white_shark-leatherback_sea_turtle"``), generates
the seed pool implied by the supplied :class:`ExperimentConfig`, and
returns the first ``replicates`` 0-based pool indices that surface the
requested pair.

The module owns an in-process cache keyed by
``(categories, n_per_class, max_logprob_gap, model_id)`` so a CLI that
resolves several pairs back-to-back only generates the pool once.

A future optimisation could persist the parquet emitted by
``experiments/preprocessing/generate_class_similarity.py`` and look up pool indices
directly, skipping the :func:`~src.evolutionary.generate_seeds` call entirely.
That layer is intentionally not here — this module makes the slow-but-
correct path robust first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config import ExperimentConfig, SeedTriple
from src.common import generate_seeds

if TYPE_CHECKING:
    from src.data import DataSource
    from src.sut.vlm_sut import VLMSUT


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    """A parsed (class_a, class_b) pair with both raw and normalised forms.

    :param class_a: The first half of the pair as typed on the CLI.
    :param class_b: The second half of the pair as typed on the CLI.
    :param class_a_normalised: :attr:`class_a` with underscores replaced
        by spaces (shell-friendly input for multi-word labels).
    :param class_b_normalised: :attr:`class_b` with underscores replaced
        by spaces.
    """

    class_a: str
    class_b: str
    class_a_normalised: str
    class_b_normalised: str

    def matches(self, seed_a: str, seed_b: str) -> bool:
        """Return ``True`` iff the pair matches a seed's (class_a, class_b).

        Tries the exact string first (preserves legitimate underscores in
        class names), then the underscore-normalised form.

        :param seed_a: Candidate class_a from a :class:`SeedTriple`.
        :param seed_b: Candidate class_b from a :class:`SeedTriple`.
        :returns: ``True`` on match.
        """
        if seed_a == self.class_a and seed_b == self.class_b:
            return True
        if (
            seed_a == self.class_a_normalised
            and seed_b == self.class_b_normalised
        ):
            return True
        return False


def _parse_pair(pair: str) -> PairSpec:
    """Split a ``class_a->class_b`` or ``class_a-class_b`` string.

    The arrow form ``->`` is checked first so hyphenated class names
    (e.g. ``"band-tailed pigeon"``) are not accidentally split. If the
    string contains neither separator, a :class:`ValueError` is raised.

    :param pair: Pair string as typed on the CLI.
    :returns: :class:`PairSpec` with raw and normalised halves.
    :raises ValueError: If the pair string cannot be split.
    """
    if "->" in pair:
        left, right = pair.split("->", 1)
    elif "-" in pair:
        left, right = pair.split("-", 1)
    else:
        raise ValueError(
            f"Pair {pair!r} is missing a separator. "
            f"Use 'class_a->class_b' or 'class_a-class_b'."
        )

    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(
            f"Pair {pair!r} has an empty class name. "
            f"Use 'class_a->class_b' or 'class_a-class_b'."
        )
    return PairSpec(
        class_a=left,
        class_b=right,
        class_a_normalised=left.replace("_", " "),
        class_b_normalised=right.replace("_", " "),
    )


# ---------------------------------------------------------------------------
# String similarity for "did-you-mean" hints
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Plain iterative DP — dependency-free, O(len(a) * len(b)) time and
    O(len(b)) memory. Used only for ``n_pairs * 5`` candidate scoring
    in error messages, so performance is not critical.

    :param a: First string.
    :param b: Second string.
    :returns: Minimum number of single-character insertions, deletions,
        or substitutions to turn *a* into *b*.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = curr
    return prev[-1]


def _closest_pairs(
    spec: PairSpec,
    pool_pairs: list[tuple[str, str]],
    k: int = 5,
) -> list[tuple[str, str]]:
    """Return the *k* pool pairs closest to *spec* by Levenshtein distance.

    Distance is computed against the concatenation ``class_a->class_b``
    on both sides (using the spec's normalised halves, so underscore-
    typos still map to space-separated pool entries). Ties are broken
    by the pool's original order, making the result deterministic.

    :param spec: Parsed pair spec.
    :param pool_pairs: All unique ``(class_a, class_b)`` tuples in the pool,
        in pool order (first-appearance).
    :param k: Maximum number of suggestions to return.
    :returns: Up to *k* pairs, closest first.
    """
    target = f"{spec.class_a_normalised}->{spec.class_b_normalised}"
    scored = [
        (_levenshtein(target, f"{a}->{b}"), idx, (a, b))
        for idx, (a, b) in enumerate(pool_pairs)
    ]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [pair for _, _, pair in scored[:k]]


# ---------------------------------------------------------------------------
# Pool cache
# ---------------------------------------------------------------------------


# Key: (categories, n_per_class, max_logprob_gap, model_id)
#   * categories is a tuple (hashable, order-preserving — mirrors the
#     deterministic order generate_seeds iterates in)
#   * max_logprob_gap is a float — hashed as-is (config values are
#     typed, so NaN is not a concern here)
_PoolCacheKey = tuple[tuple[str, ...], int, float, str]

_POOL_CACHE: dict[_PoolCacheKey, list[SeedTriple]] = {}


def _pool_cache_key(config: ExperimentConfig) -> _PoolCacheKey:
    """Build the cache key for *config*'s pool-generation parameters.

    :param config: Experiment config. Its ``categories`` must be
        resolved (non-empty) — :func:`generate_seeds` requires this.
    :returns: Hashable key identifying the pool.
    """
    return (
        tuple(config.categories),
        config.seeds.n_per_class,
        float(config.seeds.max_logprob_gap),
        config.sut.model_id,
    )


def _get_or_build_pool(
    sut: VLMSUT,
    config: ExperimentConfig,
    data_source: DataSource,
) -> list[SeedTriple]:
    """Return the seed pool for *config*, caching the result in-process.

    Subsequent calls with the same (categories, n_per_class, max_gap,
    model_id) key skip the expensive :func:`generate_seeds` call.

    :param sut: VLMSUT used for seed scoring.
    :param config: Experiment config with resolved categories.
    :param data_source: Data source feeding :func:`generate_seeds`.
    :returns: List of :class:`SeedTriple`, in pool order.
    """
    key = _pool_cache_key(config)
    pool = _POOL_CACHE.get(key)
    if pool is None:
        pool = generate_seeds(sut, config, data_source)
        _POOL_CACHE[key] = pool
    return pool


def clear_pool_cache() -> None:
    """Drop every cached pool. Use in tests or long-running daemons.

    :returns: ``None``.
    """
    _POOL_CACHE.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_pair(
    pair: str,
    config: ExperimentConfig,
    sut: VLMSUT,
    data_source: DataSource,
    replicates: int,
) -> list[int]:
    """Return filter_indices that surface *pair* under the given pool config.

    Generates the seed pool via :func:`src.evolutionary.generate_seeds`,
    enumerates seeds whose ``(class_a, class_b)`` match the requested
    pair, and returns the first ``replicates`` 0-based pool indices
    sorted by ``pool_idx`` (stable, reproducible).

    Matching is case-sensitive against
    :attr:`~src.config.SeedTriple.class_a` / ``class_b``. Underscores in
    the pair string are treated as shell-friendly spaces, but the exact
    (underscore-preserving) string is tried first so labels that
    legitimately contain underscores still resolve.

    :param pair: Pair string in the form ``"class_a->class_b"`` or
        ``"class_a-class_b"``. Whitespace around the halves is stripped.
    :param config: Experiment config whose ``seeds.*`` and ``sut.model_id``
        fields define the pool. Categories must be resolved.
    :param sut: :class:`VLMSUT` used by :func:`generate_seeds`.
    :param data_source: :class:`DataSource` used by :func:`generate_seeds`.
    :param replicates: Number of pool indices to return. Must be ≥ 1.
    :returns: List of 0-based pool indices, ascending, length
        ``replicates``.
    :raises ValueError: If ``replicates`` is not positive, the pair string
        is malformed, the pair does not appear in the pool, or fewer than
        ``replicates`` pool seeds match the pair. Error messages name the
        pool parameters and suggest a remedy.
    """
    if replicates < 1:
        raise ValueError(
            f"replicates must be ≥ 1, got {replicates!r}."
        )

    spec = _parse_pair(pair)
    pool = _get_or_build_pool(sut, config, data_source)

    matching_indices = [
        i for i, s in enumerate(pool) if spec.matches(s.class_a, s.class_b)
    ]

    pool_params = (
        f"n_categories={len(config.categories)}, "
        f"n_per_class={config.seeds.n_per_class}, "
        f"max_logprob_gap={config.seeds.max_logprob_gap:g}, "
        f"model={config.sut.model_id}"
    )

    if not matching_indices:
        # Pair absent from pool — build a "did-you-mean" list from unique
        # (class_a, class_b) tuples in pool-first-seen order.
        seen: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[str, str]] = []
        for s in pool:
            key = (s.class_a, s.class_b)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(key)

        if not unique_pairs:
            raise ValueError(
                f"Pair {pair!r} not found: the seed pool is empty. "
                f"Pool params: {pool_params}. "
                f"Try increasing seeds.max_logprob_gap or seeds.n_per_class."
            )

        suggestions = _closest_pairs(spec, unique_pairs, k=5)
        suggestion_text = "\n  ".join(f"{a}->{b}" for a, b in suggestions)
        raise ValueError(
            f"Pair {pair!r} not found in seed pool ({len(unique_pairs)} "
            f"unique pairs, {pool_params}). Did you mean one of these?\n  "
            f"{suggestion_text}\n"
            f"If not, try increasing seeds.max_logprob_gap or "
            f"seeds.n_per_class to widen the pool."
        )

    if len(matching_indices) < replicates:
        raise ValueError(
            f"Pair {pair!r} has only {len(matching_indices)} pool seed(s), "
            f"but replicates={replicates} requested "
            f"({pool_params}). Try --replicates N where "
            f"N <= {len(matching_indices)}, or increase "
            f"seeds.max_logprob_gap / seeds.n_per_class to surface more."
        )

    return matching_indices[:replicates]


__all__ = [
    "PairSpec",
    "clear_pool_cache",
    "resolve_pair",
]
