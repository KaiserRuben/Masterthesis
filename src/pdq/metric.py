"""PDQ metric and distance function registries.

The PDQ score measures boundary sharpness:

    PDQ(g_0, g_min) = d_o(L_anchor, L_min) / (d_i(g_0, g_min) + eps)

A high PDQ means a large output change was achieved with a small input
perturbation — a sharp, easy-to-cross boundary.  A low PDQ means the
boundary required heavy perturbation to cross.

Registries
----------
``INPUT_DISTANCES`` and ``OUTPUT_DISTANCES`` map config string names to
callables.  The validator in ``src/pdq/config.py`` checks names against
``VALID_D_I`` / ``VALID_D_O``; adding an entry here is not required for
validation, but it IS required for the runner to resolve the function.

Signature conventions
---------------------
- Input distance: ``(g: np.ndarray, anchor_geno: np.ndarray) -> float``
- Output distance: ``(label_a: str, label_b: str) -> float``
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .distances.input import (
    hamming,
    image_pixel_l2,
    rank_sum,
    sparsity,
    weighted_content,
)
from .distances.output import (
    embedding_cosine,
    label_mismatch,
    string_edit,
    wordnet_path,
)


def pdq(d_i: float, d_o: float, eps: float = 1e-9) -> float:
    """PDQ score: output distance divided by input distance.

    The eps guard prevents division-by-zero when the anchor genotype is
    identical to the tested genotype (zero perturbation).

    :param d_i: Input-space distance (≥ 0).
    :param d_o: Output-space distance (≥ 0).
    :param eps: Small constant to avoid division by zero.
    :returns: PDQ score ≥ 0.
    """
    return d_o / (d_i + eps)


# ---------------------------------------------------------------------------
# Input-distance registry
# ---------------------------------------------------------------------------

# Signature: (g: np.ndarray, anchor_geno: np.ndarray) -> float
# ``image_pixel_L2`` cannot be used as d_i_primary via this registry because
# it requires rendered pixel arrays rather than raw genotypes.  It raises
# NotImplementedError to give a clear message if misconfigured.

INPUT_DISTANCES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rank_sum": lambda g, anchor: float(rank_sum(g)),
    "sparsity": lambda g, anchor: float(sparsity(g)),
    "hamming": lambda g, anchor: float(hamming(g, anchor)),
    "weighted_content": lambda g, anchor: weighted_content(g, np.ones(len(g), dtype=np.float32)),
    "image_pixel_L2": lambda g, anchor: (_ for _ in ()).throw(  # type: ignore[misc]
        NotImplementedError(
            "image_pixel_L2 cannot be used as d_i_primary via the registry; "
            "it requires rendered images.  Use rank_sum or sparsity instead."
        )
    ),
}

# ---------------------------------------------------------------------------
# Output-distance registry
# ---------------------------------------------------------------------------

# Signature: (label_a: str, label_b: str) -> float
# Phase-3 stubs raise NotImplementedError when called; they exist in the
# registry so the config validator accepts their names.

OUTPUT_DISTANCES: dict[str, Callable[[str, str], float]] = {
    "label_mismatch": label_mismatch,
    "label_edit": string_edit,
    "label_embedding": embedding_cosine,
    "wordnet_path": wordnet_path,
}
