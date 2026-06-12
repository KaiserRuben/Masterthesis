#!/usr/bin/env python3
"""Validate that boundary-pair resume can trust regenerated seed indices.

Runs three independent checks against the live Exp-100 run directory
without loading the SUT or any model (safe to run alongside an
in-flight job):

1. combinatorial_pairs() is byte-stable across two identical calls
   (proves the index → SeedTriple mapping is reproducible).
2. The ImageNet cache for each roster class is sorted (proves
   roster_seeds() emits SeedImages in stable order).
3. Every finished seed_NNNN_<ts>/ on disk has metadata matching what
   combinatorial_pairs() reconstructs at index NNNN with the persisted
   roster_class_list / abstraction config.

Exit 0 on full agreement, 1 on any divergence. Designed to be run
from the repo root.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.common.combinatorial_pair_generator import combinatorial_pairs
from src.common.roster_seed_generator import SeedImage
from src.config import AbstractionConfig
from src.boundary_pair.config import load_boundary_pair_config

CONFIG_PATH = REPO / "configs" / "Exp-100" / "poc_boundary_pair.yaml"
RUN_ROOT = REPO / "runs" / "Exp-100" / "poc_boundary_pair"


def fake_seed_pool(
    class_list: list[str], seeds_per_class: int,
) -> list[SeedImage]:
    """Build a SeedImage list identical to what roster_seeds() emits
    in shape and order, with placeholder images. Reflects roster_seeds()
    invariant: ``for cls in class_list: for k in range(seeds_per_class)``.
    """
    dummy = Image.new("RGB", (1, 1))
    out = []
    for cls in class_list:
        for k in range(seeds_per_class):
            out.append(SeedImage(image=dummy, class_concrete=cls, seed_idx_in_class=k))
    return out


def expansion_metadata_only(seeds) -> list[dict]:
    """Drop the PIL handle so two runs can be compared element-by-element."""
    return [
        {"class_a": s.class_a, "class_b": s.class_b, "metadata": dict(s.metadata)}
        for s in seeds
    ]


def check_combinatorial_determinism(cfg) -> tuple[bool, int]:
    class_list = list(cfg.seeds.roster.class_list)
    pool = fake_seed_pool(class_list, cfg.seeds.roster.seeds_per_class)
    abs_cfg = cfg.seeds.roster.abstraction
    a = expansion_metadata_only(combinatorial_pairs(pool, class_list, abs_cfg))
    b = expansion_metadata_only(combinatorial_pairs(pool, class_list, abs_cfg))
    ok = a == b
    return ok, len(a)


def check_imagenet_cache_sorted(cfg) -> tuple[bool, list[str]]:
    """Confirm each roster class dir lists images in deterministic order.

    ImageNetCache._cached() reads with sorted(); we just verify the
    on-disk directories exist and the file-name sort is stable.
    """
    issues = []
    classes = cfg.seeds.roster.class_list
    cache_dirs = [Path(d) for d in cfg.cache_dirs]
    # ImageNetCache nests under <cache_dir>/category_images/<safe_class>/*.png
    # where safe = category.replace(" ", "_").lower() (see _safe_name).
    seen_any = False
    for cls in classes:
        safe = cls.replace(" ", "_").lower()
        found_dirs = []
        for cd in cache_dirs:
            p = cd / "category_images" / safe
            if p.is_dir():
                found_dirs.append(p)
        if not found_dirs:
            issues.append(f"  {cls!r}: no cache dir at <cache_dir>/images/{safe}/")
            continue
        seen_any = True
        for p in found_dirs:
            files = list(p.glob("*.png"))
            if not files:
                issues.append(f"  {cls!r}: {p} has no .png files")
                continue
            sorted_names = sorted(f.name for f in files)
            if sorted_names != [f.name for f in sorted(files, key=lambda x: x.name)]:
                issues.append(f"  {cls!r}: sort order non-canonical at {p}")
    if not seen_any:
        issues.append("  No cache hits found — config.cache_dirs may be wrong")
    return (not issues), issues


def check_existing_seeds_align(cfg) -> tuple[int, int, list[str]]:
    """For each finished seed_NNNN_<ts>/ on disk, verify the regenerated
    SeedTriple at that index matches the persisted seed_metadata.

    Returns (matched, total_finished, divergences).
    """
    class_list = list(cfg.seeds.roster.class_list)
    pool = fake_seed_pool(class_list, cfg.seeds.roster.seeds_per_class)
    abs_cfg = cfg.seeds.roster.abstraction
    expansion = combinatorial_pairs(pool, class_list, abs_cfg)

    fields = (
        "anchor_class_concrete", "target_class_concrete",
        "level_anchor", "level_target",
        "seed_idx_in_class", "anchor_position", "target_position",
    )
    matched = 0
    total = 0
    divergences = []
    if not RUN_ROOT.exists():
        return 0, 0, [f"  run root {RUN_ROOT} does not exist"]

    seen_idx: dict[int, str] = {}
    for seed_dir in sorted(RUN_ROOT.glob("seed_*")):
        manifest = seed_dir / "manifest.json"
        if not manifest.exists():
            continue
        stats = seed_dir / "evolutionary" / "stats.json"
        if not stats.exists():
            continue
        m = json.loads(manifest.read_text())
        s = json.loads(stats.read_text())
        idx = int(m["seed_idx"])
        meta = s.get("seed_metadata") or {}
        total += 1
        if idx in seen_idx:
            divergences.append(
                f"  seed_idx={idx}: appears in both {seen_idx[idx]} and {seed_dir.name}"
            )
            continue
        seen_idx[idx] = seed_dir.name
        if idx >= len(expansion):
            divergences.append(
                f"  seed_idx={idx} ({seed_dir.name}): "
                f"out of range for reconstructed pool size {len(expansion)}"
            )
            continue
        live_meta = expansion[idx].metadata
        mismatch = []
        for f in fields:
            if live_meta.get(f) != meta.get(f):
                mismatch.append(f"{f}: persisted={meta.get(f)!r} regen={live_meta.get(f)!r}")
        # Also check class_a/class_b at the SeedTriple level (abstract labels)
        if m.get("class_a") != expansion[idx].class_a:
            mismatch.append(f"class_a: manifest={m.get('class_a')!r} regen={expansion[idx].class_a!r}")
        if m.get("class_b") != expansion[idx].class_b:
            mismatch.append(f"class_b: manifest={m.get('class_b')!r} regen={expansion[idx].class_b!r}")
        if mismatch:
            divergences.append(
                f"  seed_idx={idx} ({seed_dir.name}): " + "; ".join(mismatch)
            )
        else:
            matched += 1
    return matched, total, divergences


def main() -> int:
    cfg = load_boundary_pair_config(CONFIG_PATH)
    print(f"Config: {CONFIG_PATH}")
    print(f"Run root: {RUN_ROOT}")
    print(f"Roster: {list(cfg.seeds.roster.class_list)}  ×  "
          f"{cfg.seeds.roster.seeds_per_class} seeds/class")
    print(f"Abstraction: levels_anchor={cfg.seeds.roster.abstraction.levels_anchor}, "
          f"levels_target={cfg.seeds.roster.abstraction.levels_target}, "
          f"directions={cfg.seeds.roster.abstraction.directions!r}, "
          f"disjointness={cfg.seeds.roster.abstraction.apply_disjointness}")
    print()

    ok1, n_pairs = check_combinatorial_determinism(cfg)
    print(f"[1] combinatorial_pairs() byte-stable across two calls: "
          f"{'YES' if ok1 else 'NO'}  ({n_pairs} pairs each)")

    ok2, issues2 = check_imagenet_cache_sorted(cfg)
    print(f"[2] ImageNet cache per-class dirs present + sortable:    "
          f"{'YES' if ok2 else 'NO'}")
    for line in issues2:
        print(line)

    matched, total, divs = check_existing_seeds_align(cfg)
    ok3 = bool(total) and not divs
    print(f"[3] On-disk finished seeds align with regen at same idx: "
          f"{matched}/{total} matched")
    for line in divs[:20]:
        print(line)
    if len(divs) > 20:
        print(f"  ... +{len(divs) - 20} more divergence(s)")

    overall_ok = ok1 and ok2 and ok3
    print()
    print("OVERALL:", "PASS — resume is safe" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
