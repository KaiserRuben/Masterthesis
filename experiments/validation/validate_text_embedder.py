"""Smoke test for TextEmbedder: correctness + caching.

Loads a real VLMSUT (Qwen3.5-4B or whatever the config names), encodes
the motivating sentence pairs, verifies:
    * identical sentence -> distance 0
    * synonym distance < negation distance (the reason we switched backends)
    * second call on same text is a cache hit (no new miss recorded)

Intentionally minimal — verifies the wiring, not the separation quality
(that is what validate_text_distance.py already measured).
"""

from __future__ import annotations

import numpy as np

from src.config import ExperimentConfig, SUTConfig
from src.sut.vlm_sut import VLMSUT

ORIG    = "What is the main subject of this image"
NEGATED = "What is the non-main subject of this image"
SYNONYM = "What is the primary subject of this image"


def main() -> None:
    cfg = ExperimentConfig(sut=SUTConfig(model_id="Qwen/Qwen3.5-4B"))
    sut = VLMSUT(cfg)
    emb = sut.text_embedder

    # Cold pass.
    stats0 = dict(emb.cache_stats)
    dists = emb.cosine_distances_to(
        emb.embed(ORIG), [ORIG, NEGATED, SYNONYM],
    )
    stats1 = dict(emb.cache_stats)
    print(f"distances (cold): identical={dists[0]:.6f}  "
          f"negation={dists[1]:.4f}  synonym={dists[2]:.4f}")
    print(f"cache delta cold: hits +{stats1['hits']-stats0['hits']}  "
          f"misses +{stats1['misses']-stats0['misses']}")

    # Warm pass — should be all hits (Redis) or LRU hits.
    dists2 = emb.cosine_distances_to(
        emb.embed(ORIG), [ORIG, NEGATED, SYNONYM],
    )
    stats2 = dict(emb.cache_stats)
    print(f"distances (warm): identical={dists2[0]:.6f}  "
          f"negation={dists2[1]:.4f}  synonym={dists2[2]:.4f}")
    print(f"cache delta warm: hits +{stats2['hits']-stats1['hits']}  "
          f"misses +{stats2['misses']-stats1['misses']}")

    # Assertions.
    assert abs(dists[0]) < 1e-4, f"identical sentence should be ~0, got {dists[0]}"
    assert dists[2] < dists[1], (
        f"negation ({dists[1]:.4f}) should be farther than synonym "
        f"({dists[2]:.4f}) in the SUT's embedding space"
    )
    # Warm pass should record only cache hits for the three distance calls
    # (plus an embed() for the anchor, which may also be a hit).
    new_misses = stats2["misses"] - stats1["misses"]
    assert new_misses == 0, f"expected 0 new misses on warm pass, got {new_misses}"
    print("OK")


if __name__ == "__main__":
    main()
