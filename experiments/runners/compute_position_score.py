#!/usr/bin/env python3
"""Compute per-image-position importance score for score-guided init.

Three score methods (see :mod:`src.optimizer.position_scoring`):

* ``pattern``  — Tian-2021-SparseEA2-style log-frequency-ratio extracted
  from a brief warm-up MOEA run; cheapest (~600 SUT calls).
* ``ablation`` — Breiman-2001 permutation-importance over realistic
  multi-tier-sampled backgrounds (~N × n_image SUT calls).
* ``sobol``    — Saltelli-2010 total-order Sobol indices via A/B-pickfreeze
  (~n_base × (n_image + 2) SUT calls).

The output is a 1-D float64 ``.npy`` of length n_image, where lower =
more important. Consumed by :class:`ScoreGuidedMultiTierSampling` via
``optimizer.sampling.score_path`` in the experiment YAML.

Usage::

    python experiments/runners/compute_position_score.py \\
        configs/Exp-22/score_eval_junco_chickadee.yaml \\
        --method pattern --out runs/Exp-22/scores/pattern_seed83.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dacite
import numpy as np
import torch
import yaml
from PIL import Image

from src.evolutionary.vlm_boundary_tester import pil_to_tensor

from src.common import apply_seed_filter, generate_seeds
from src.config import ExperimentConfig, resolve_categories
from src.data import ImageNetCache
from src.manipulator.image.manipulator import ImageManipulator
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.manipulator.text.composite import CompositeTextManipulator
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextEmbeddingDistance,
)
from src.optimizer.position_scoring import (
    ablation_score,
    pattern_score,
    sobol_score,
)
from src.optimizer.sparse_sampling import MultiTierSparseSampling
from src.sut import VLMSUT

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


_DACITE_CONFIG = dacite.Config(
    cast=[tuple, frozenset],
    type_hooks={
        Path: lambda v: Path(v).expanduser() if isinstance(v, str) else v,
        PatchStrategy: lambda v: PatchStrategy[v] if isinstance(v, str) else v,
        CandidateStrategy: lambda v: CandidateStrategy[v] if isinstance(v, str) else v,
    },
)


def _evaluate_genotypes(
    genotypes: np.ndarray,
    *,
    manipulator: VLMManipulator,
    sut: VLMSUT,
    objectives: CriterionCollection,
    target_classes: tuple[int, int],
    answer_suffix: str,
    categories: tuple[str, ...],
    anchor_text_embedding,
    origin_tensor,
) -> np.ndarray:
    """Evaluate a batch of genotypes through the full pipeline.

    Returns a ``(N,)`` float64 vector of TgtBal fitness (lower = better).
    """
    images, texts = manipulator.manipulate(candidates=None, weights=genotypes)
    logits_list = []
    for img, txt in zip(images, texts):
        logits_list.append(
            sut.process_input(img, text=txt + answer_suffix, categories=categories)
        )
    logits = torch.stack(logits_list)

    perturbed = torch.stack([pil_to_tensor(img) for img in images])
    origin = origin_tensor.unsqueeze(0).expand_as(perturbed)
    text_distances = sut.text_embedder.cosine_distances_to(
        anchor_text_embedding, texts,
    )

    objectives.evaluate_all(
        images=[origin, perturbed],
        logits=logits,
        target_classes=target_classes,
        batch_dim=0,
        text_distances=text_distances,
    )
    results = objectives.results
    fitness = tuple(
        f.cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
        for f in results.values()
    )
    # CriterionCollection result order matches CriterionCollection construction
    # order: (MatrixDistance, TextEmbeddingDistance, TargetedBalance).
    return fitness[2].astype(np.float64)


def _bootstrap(cfg_dict: dict) -> dict:
    """Initialise SUT, manipulator, objectives, seed selection.

    Returns a dict of components ready for evaluation. Keeps the
    bootstrap logic out of the score-method branches so all three
    methods see an identical environment.
    """
    exp = dacite.from_dict(ExperimentConfig, cfg_dict, config=_DACITE_CONFIG)
    data_source = ImageNetCache(dirs=exp.cache_dirs)
    exp = resolve_categories(exp, data_source.labels())

    pool = ThreadPoolExecutor(max_workers=3)
    text_fut: Future = pool.submit(
        CompositeTextManipulator.from_config,
        text_config=exp.text,
        device=exp.device,
        redis_url=exp.sut.redis_url,
    )
    image_fut: Future[ImageManipulator] = pool.submit(
        ImageManipulator.from_preset, device=exp.device, config=exp.image,
    )
    sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp)

    objectives = CriterionCollection(
        MatrixDistance(),
        TextEmbeddingDistance(),
        TargetedBalance(),
    )

    sut = sut_fut.result()
    seeds = list(generate_seeds(sut, exp, data_source))
    indexed_seeds = apply_seed_filter(seeds, exp.seeds.filter_indices)
    if not indexed_seeds:
        raise RuntimeError("Seed filter selected zero seeds.")
    if len(indexed_seeds) > 1:
        raise RuntimeError(
            f"score-computation expects single seed; got {len(indexed_seeds)}"
        )
    seed_idx, seed = indexed_seeds[0]

    image_manip = image_fut.result()
    text_manip = text_fut.result()
    manipulator = VLMManipulator(image_manipulator=image_manip, text_manipulator=text_manip)

    pair = (seed.class_a, seed.class_b)
    answer_suffix = exp.answer_format.format(categories=", ".join(pair))
    if exp.score_full_categories:
        categories = tuple(exp.categories)
        target_classes = (categories.index(seed.class_a), categories.index(seed.class_b))
    else:
        categories = pair
        target_classes = (0, 1)

    manipulator.prepare(seed.image, exp.prompt_template)
    anchor = sut.text_embedder.embed(exp.prompt_template)

    zero_geno = manipulator.zero_genotype().reshape(1, -1)
    baseline_imgs, _ = manipulator.manipulate(candidates=None, weights=zero_geno)
    origin_tensor = pil_to_tensor(baseline_imgs[0])

    return {
        "exp": exp,
        "sut": sut,
        "manipulator": manipulator,
        "objectives": objectives,
        "categories": categories,
        "target_classes": target_classes,
        "answer_suffix": answer_suffix,
        "anchor": anchor,
        "origin_tensor": origin_tensor,
        "seed_idx": seed_idx,
    }


def _make_eval_fn(components: dict):
    def eval_fn(genotypes: np.ndarray) -> np.ndarray:
        return _evaluate_genotypes(
            genotypes,
            manipulator=components["manipulator"],
            sut=components["sut"],
            objectives=components["objectives"],
            target_classes=components["target_classes"],
            answer_suffix=components["answer_suffix"],
            categories=components["categories"],
            anchor_text_embedding=components["anchor"],
            origin_tensor=components["origin_tensor"],
        )
    return eval_fn


# ---------------------------------------------------------------------------
# Method drivers
# ---------------------------------------------------------------------------


def _run_pattern(components: dict, args) -> np.ndarray:
    """Warm-up MOEA + log-ratio score from history."""
    n_var = components["manipulator"].gene_bounds.size
    n_text = components["manipulator"].text_dim
    n_image = n_var - n_text
    image_xu = components["manipulator"].gene_bounds[:n_image] - 1

    rng = np.random.default_rng(args.seed)
    sampler = MultiTierSparseSampling(
        text_dim=n_text,
        tiers=[(0.005, 0.20), (0.030, 0.20), (0.10, 0.25), (0.30, 0.20), (0.50, 0.10)],
        zero_anchor_fraction=0.05,
        seed=args.seed,
    )

    class _Probe:
        def __init__(self, n_var, xu):
            self.n_var = n_var
            self.xu = xu

    eval_fn = _make_eval_fn(components)

    n_pop = args.pattern_pop
    n_gens = args.pattern_gens
    total = n_pop * n_gens

    logger.info("Pattern-mining: %d × %d = %d sample evaluations", n_pop, n_gens, total)
    all_genomes = np.zeros((total, n_var), dtype=np.int64)
    all_tgtbal = np.zeros(total, dtype=np.float64)
    cursor = 0

    # Synthetic warm-up: random sampling + tournament-on-fitness "selection".
    # Not full AGE-MOEA-II — just enough exploration that the top/bot
    # quartile log-ratio is informative.
    for g in range(n_gens):
        if g == 0:
            samples = sampler._do(
                _Probe(n_var, components["manipulator"].gene_bounds - 1),
                n_samples=n_pop,
            )
        else:
            # Refresh by random restart from sampler — simple "memoryless"
            # warm-up emphasising decision-space coverage over local search.
            samples = sampler._do(
                _Probe(n_var, components["manipulator"].gene_bounds - 1),
                n_samples=n_pop,
            )
        fit = eval_fn(samples)
        all_genomes[cursor:cursor + n_pop] = samples
        all_tgtbal[cursor:cursor + n_pop] = fit
        cursor += n_pop
        logger.info(
            "  warmup gen %d/%d: pop_min=%.4f pop_mean=%.4f",
            g + 1, n_gens, float(fit.min()), float(fit.mean()),
        )

    score = pattern_score(all_genomes, all_tgtbal, n_image)
    return score


def _run_ablation(components: dict, args) -> np.ndarray:
    n_var = components["manipulator"].gene_bounds.size
    n_text = components["manipulator"].text_dim
    n_image = n_var - n_text
    image_xu = components["manipulator"].gene_bounds[:n_image] - 1

    rng = np.random.default_rng(args.seed)
    sampler = MultiTierSparseSampling(
        text_dim=n_text,
        tiers=[(0.005, 0.20), (0.030, 0.20), (0.10, 0.25), (0.30, 0.20), (0.50, 0.10)],
        zero_anchor_fraction=0.0,
        seed=args.seed,
    )

    class _Probe:
        def __init__(self, n_var, xu):
            self.n_var = n_var
            self.xu = xu

    backgrounds = sampler._do(
        _Probe(n_var, components["manipulator"].gene_bounds - 1),
        n_samples=args.ablation_backgrounds,
    )
    logger.info(
        "Ablation: %d backgrounds × (1 + %d positions) = %d evaluations",
        args.ablation_backgrounds, n_image,
        args.ablation_backgrounds * (1 + n_image),
    )

    eval_fn = _make_eval_fn(components)
    score = ablation_score(eval_fn, backgrounds, n_image, image_xu, rng)
    return score


def _run_sobol(components: dict, args) -> np.ndarray:
    n_var = components["manipulator"].gene_bounds.size
    n_text = components["manipulator"].text_dim
    n_image = n_var - n_text
    image_xu = components["manipulator"].gene_bounds[:n_image] - 1
    text_xu = components["manipulator"].gene_bounds[n_image:] - 1

    rng = np.random.default_rng(args.seed)

    logger.info(
        "Sobol: n_base=%d × (n_image=%d + 2) = %d evaluations",
        args.sobol_n_base, n_image,
        args.sobol_n_base * (n_image + 2),
    )

    eval_fn = _make_eval_fn(components)
    score = sobol_score(
        eval_fn=eval_fn,
        n_image=n_image,
        n_text=n_text,
        image_xu=image_xu,
        text_xu=text_xu,
        n_base=args.sobol_n_base,
        p_active=args.sobol_p_active,
        rng=rng,
    )
    return score


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


METHODS = {"pattern": _run_pattern, "ablation": _run_ablation, "sobol": _run_sobol}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML config (same schema as run_boundary_test)")
    parser.add_argument("--method", required=True, choices=list(METHODS), help="Score method")
    parser.add_argument("--out", required=True, type=Path, help="Output .npy path for score vector")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--pattern-pop", type=int, default=30, help="[pattern] pop size per warmup gen")
    parser.add_argument("--pattern-gens", type=int, default=20, help="[pattern] number of warmup gens")
    parser.add_argument("--ablation-backgrounds", type=int, default=10, help="[ablation] # backgrounds")
    parser.add_argument("--sobol-n-base", type=int, default=20, help="[sobol] base sample N")
    parser.add_argument("--sobol-p-active", type=float, default=0.10, help="[sobol] Bernoulli p for sampling")
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    with args.config.open() as f:
        cfg_dict = yaml.safe_load(f)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Bootstrap …")
    components = _bootstrap(cfg_dict)
    logger.info(
        "Bootstrap done. seed_idx=%d  pair=%s",
        components["seed_idx"],
        (components["categories"][components["target_classes"][0]],
         components["categories"][components["target_classes"][1]]),
    )

    t0 = time.time()
    score = METHODS[args.method](components, args)
    elapsed = time.time() - t0

    logger.info(
        "Score computed in %.1fs (%.2f min)  shape=%s  range=[%.4f, %.4f]  mean=%.4f",
        elapsed, elapsed / 60.0, score.shape,
        float(score.min()), float(score.max()), float(score.mean()),
    )

    np.save(args.out, score)
    logger.info("Score written to %s", args.out)


if __name__ == "__main__":
    main()
