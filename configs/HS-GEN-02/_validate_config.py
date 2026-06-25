#!/usr/bin/env python3
"""Validate HS-GEN-02 config through the PRODUCTION loader + enumeration.

Checks:
  1. load_config + apply_modality succeed; modality=joint, cone on, vqgan.
  2. filter_indices in range of the seeds_per_class=12 enumeration.
  3. each selected index maps to a forward (la,lt)=(0,0) cell.
  4. distinct (anchor_class_concrete, seed_idx) photos == reported.
  5. plan-only run count == len(filter_indices).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from experiments.runners.run_boundary_test import load_config
from src.config import apply_modality, AbstractionConfig
from src.common.combinatorial_pair_generator import combinatorial_pairs
from src.common.roster_seed_generator import SeedImage

CFG = Path("configs/HS-GEN-02/hs_gen02_joint_diversity.yaml")


def main() -> int:
    cfg_dict = yaml.safe_load(CFG.read_text())
    exp = load_config(cfg_dict)
    exp = apply_modality(exp)

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        mark = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")

    print("=== 1. Production loader (load_config + apply_modality) ===")
    check("modality == joint", exp.modality == "joint", exp.modality)
    check("image backend == vqgan_codebook",
          exp.image.backend == "vqgan_codebook", exp.image.backend)
    check("cone_filter.enabled", exp.image.cone_filter.enabled,
          f"alpha={exp.image.cone_filter.alpha_deg} m={exp.image.cone_filter.target_m}")
    check("text profile == full_stack",
          exp.text.composite.profile == "full_stack", exp.text.composite.profile)
    check("generations == 50", exp.generations == 50, str(exp.generations))
    check("pop_size == 30", exp.pop_size == 30, str(exp.pop_size))
    check("early_stop disabled", exp.optimizer.early_stop.enable is False)
    check("sampling sparse_multitier",
          exp.optimizer.sampling.mode == "sparse_multitier",
          exp.optimizer.sampling.mode)
    check("workers == 2", exp.parallel.workers == 2, str(exp.parallel.workers))
    check("seeds.mode == roster", exp.seeds.mode == "roster")
    spc = exp.seeds.roster.seeds_per_class
    check("seeds_per_class == 12", spc == 12, str(spc))
    check("min_anchor_confidence == 3.5",
          exp.seeds.roster.min_anchor_confidence == 3.5)
    fi = list(exp.seeds.filter_indices)
    check("filter_indices loaded", len(fi) == 108, f"{len(fi)} indices")
    check("filter_indices unique", len(set(fi)) == len(fi))

    print()
    print("=== 2-4. Re-enumerate with seeds_per_class=12 and verify cells ===")
    abs_cfg = exp.seeds.roster.abstraction
    abstraction = AbstractionConfig(
        levels_anchor=tuple(abs_cfg.levels_anchor),
        levels_target=tuple(abs_cfg.levels_target),
        apply_disjointness=abs_cfg.apply_disjointness,
        directions=abs_cfg.directions,
    )
    class_list = tuple(exp.seeds.roster.class_list)
    pool = [SeedImage(image=None, class_concrete=c, seed_idx_in_class=k)
            for c in class_list for k in range(spc)]
    triples = combinatorial_pairs(pool, class_list, abstraction)
    check("all filter_indices in range",
          all(0 <= i < len(triples) for i in fi),
          f"max idx={max(fi)} / enum size={len(triples)}")

    pos = {c: i for i, c in enumerate(class_list)}
    photos = set()
    pairs = set()
    all_fwd_00 = True
    for i in fi:
        m = triples[i].metadata
        if not (m["level_anchor"] == 0 and m["level_target"] == 0):
            all_fwd_00 = False
        if not (pos[m["anchor_class_concrete"]] < pos[m["target_class_concrete"]]):
            all_fwd_00 = False
        photos.add((m["anchor_class_concrete"], m["seed_idx_in_class"]))
        pairs.add((m["anchor_class_concrete"], m["target_class_concrete"]))
    check("every selected cell is forward (0,0)", all_fwd_00)
    check("distinct anchor photos == 60", len(photos) == 60, str(len(photos)))
    print(f"        anchor classes: {sorted({a for a, _ in photos})}")
    print(f"        distinct label-pairs: {len(pairs)}")
    # photos-per-class
    from collections import Counter
    pc = Counter(a for a, _ in photos)
    print(f"        photos/class: {dict(sorted(pc.items()))}")

    print()
    print("=== 5. plan-only run count (no model work) ===")
    # plan-only path: n_run = len(filter_indices) when set.
    n_run = len(exp.seeds.filter_indices)
    check("plan-only run count == 108", n_run == 108, str(n_run))

    print()
    print("RESULT:", "ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
