"""VLM boundary tester — the main orchestrator.

Wires together :class:`VLMManipulator`, :class:`VLMSUT`,
:class:`DiscretePymooOptimizer`, and 4 objectives into a
multi-objective evolutionary search loop.

All 4 objectives (MatrixDistance, TextReplacementDistance,
TargetedBalance, Concentration) are batched and run through a
:class:`CriterionCollection`.

Each seed triple ``(image, class_A, class_B)`` is tested independently:
the manipulator prepares image + text contexts, the optimizer evolves
integer genotypes, and every SUT interaction is logged to a Parquet
trace for offline analysis.

.. note::

   ArchiveSparsity (genotype diversity) was removed — optimal boundary
   genotypes are sparse (mostly zeros), so diversity pressure drives the
   optimizer toward non-sparse genotypes that are not boundary solutions.
"""

from __future__ import annotations

import json
import logging
from time import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from smoo import SMOO
from torch import Tensor
from tqdm import tqdm

from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import CriterionCollection
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer

from src.config import ExperimentConfig, SeedTriple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pil_to_tensor(img: Image.Image) -> Tensor:
    """Convert a PIL image to a ``(C, H, W)`` float tensor in ``[0, 1]``."""
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1)


def _build_trace_rows(
    seed_idx: int,
    gen: int,
    genotypes: NDArray,
    logits: Tensor,
    texts: list[str],
    batched_results: dict[str, Any],
    batched_fitness: tuple,
    target_classes: tuple[int, int],
    categories: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build per-individual trace rows (pure — no side effects)."""
    probs = F.softmax(logits, dim=-1)
    idx_a, idx_b = target_classes
    fitness_names = list(batched_results.keys())
    return [
        {
            "seed_id": seed_idx,
            "generation": gen,
            "individual": i,
            "genotype": genotypes[i].tolist(),
            "logprobs": logits[i].tolist(),
            "decoded_text": texts[i],
            "predicted_class": categories[int(logits[i].argmax())],
            "p_class_a": float(probs[i, idx_a]),
            "p_class_b": float(probs[i, idx_b]),
            **{
                f"fitness_{name}": float(batched_fitness[j][i])
                for j, name in enumerate(fitness_names)
            },
        }
        for i in range(len(genotypes))
    ]


def _build_stats(
    seed_idx: int,
    seed: SeedTriple,
    config: ExperimentConfig,
    manipulator: VLMManipulator,
    n_pareto: int,
    runtime: float,
    categories: tuple[str, ...],
) -> dict[str, Any]:
    """Build stats metadata dict (pure)."""
    return {
        "seed_idx": seed_idx,
        "class_a": seed.class_a,
        "class_b": seed.class_b,
        "prompt_template": config.prompt_template,
        "answer_format": config.answer_format,
        "runtime_seconds": runtime,
        "generations": config.generations,
        "pop_size": config.pop_size,
        "gene_bounds": manipulator.gene_bounds.tolist(),
        "image_dim": manipulator.image_dim,
        "text_dim": manipulator.text_dim,
        "categories": list(categories),
        "n_pareto": n_pareto,
    }


def _build_context_meta(manipulator: VLMManipulator) -> dict[str, Any]:
    """Build context metadata for offline reconstruction (pure).

    Requires ``manipulator.prepare()`` to have been called.
    """
    img_sel = manipulator.image_context.selection
    txt_sel = manipulator.text_context.selection
    return {
        "image_patch_positions": img_sel.positions.tolist(),
        "image_original_codes": img_sel.original_codes.tolist(),
        "image_candidates": [c.tolist() for c in img_sel.candidates],
        "text_word_positions": txt_sel.positions.tolist(),
        "text_original_words": list(txt_sel.original_words),
        "text_candidates": [list(c) for c in txt_sel.candidates],
        "text_candidate_distances": [
            d.tolist() for d in manipulator.text_candidate_distances
        ],
    }


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


class VLMBoundaryTester(SMOO):
    """Multi-objective boundary tester for Vision-Language Models.

    Orchestrates the full optimisation loop per seed::

        seed → prepare → [ask → manipulate → SUT → objectives → fitness
                          → update] × generations → save

    All 4 objectives are batched and passed via the
    :class:`CriterionCollection`.

    :param sut: The VLM system-under-test.
    :param manipulator: Multi-modal manipulator (image + text).
    :param optimizer: Discrete evolutionary optimizer.
    :param objectives: Criterion collection (4 batched objectives).
    :param config: Experiment-level settings.
    """

    _manipulator: VLMManipulator
    _optimizer: DiscretePymooOptimizer

    def __init__(
        self,
        *,
        sut,
        manipulator: VLMManipulator,
        optimizer: DiscretePymooOptimizer,
        objectives: CriterionCollection,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            restrict_classes=None,
            use_wandb=False,
        )
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(self, seeds: Sequence[SeedTriple]) -> None:
        """Run boundary testing for all seeds.

        :param seeds: Sequence of ``(image, class_a, class_b)`` triples.
        """
        categories = self._config.categories
        answer_suffix = self._config.answer_format.format(
            categories=", ".join(categories),
        )

        n_gens = self._config.generations
        pbar = tqdm(total=n_gens, unit="gen")

        for seed_idx, seed in enumerate(seeds):
            pbar.reset()
            pbar.set_description(
                f"[{seed_idx + 1}/{len(seeds)}] "
                f"{seed.class_a} vs {seed.class_b}"
            )
            self._run_seed(
                seed_idx, seed, categories, answer_suffix, pbar,
            )
            self._optimizer.reset()
            self._cleanup()

        pbar.close()

    # ------------------------------------------------------------------
    # Per-seed loop
    # ------------------------------------------------------------------

    def _run_seed(
        self,
        seed_idx: int,
        seed: SeedTriple,
        categories: tuple[str, ...],
        answer_suffix: str,
        pbar: tqdm,
    ) -> None:
        start_time = time()

        # 1. Prepare manipulator with just the question prompt.
        #    Categories are NOT in the mutable text — they are appended
        #    after mutation via answer_suffix.
        self._manipulator.prepare(
            seed.image, self._config.prompt_template,
        )

        # 2. Re-configure optimizer for this seed's genotype dimensions.
        self._optimizer.update_gene_bounds(self._manipulator.gene_bounds)

        # 3. Resolve target class indices.
        idx_a = categories.index(seed.class_a)
        idx_b = categories.index(seed.class_b)
        target_classes = (idx_a, idx_b)

        # 4. Compute VQGAN-reconstructed baseline (fair comparison for
        #    MatrixDistance — both origin and perturbed go through VQGAN).
        zero_geno = self._manipulator.zero_genotype().reshape(1, -1)
        baseline_imgs, _ = self._manipulator.manipulate(
            candidates=None, weights=zero_geno,
        )
        origin_tensor = _pil_to_tensor(baseline_imgs[0])

        # 5. Generation loop.
        trace_rows: list[dict[str, Any]] = []

        for gen in range(self._config.generations):
            gen_start = time()

            gen_traces = self._run_generation(
                seed_idx, gen, target_classes, origin_tensor,
                categories, answer_suffix,
            )
            trace_rows.extend(gen_traces)

            pbar.set_postfix(
                pareto=len(self._optimizer.best_candidates),
                t=f"{time() - gen_start:.0f}s",
            )
            pbar.update(1)

        # 6. Save everything.
        runtime = time() - start_time
        self._save_seed_results(
            seed_idx, seed, trace_rows,
            categories, answer_suffix, runtime,
        )

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------

    def _run_generation(
        self,
        seed_idx: int,
        gen: int,
        target_classes: tuple[int, int],
        origin_tensor: Tensor,
        categories: tuple[str, ...],
        answer_suffix: str,
    ) -> list[dict[str, Any]]:
        """Evaluate one generation and advance the optimizer.

        Returns a list of trace-row dicts (one per individual).
        """
        genotypes = self._optimizer.get_x_current()

        # -- Manipulate ---------------------------------------------------
        images, texts = self._manipulator.manipulate(
            candidates=None, weights=genotypes,
        )

        # -- SUT evaluation -----------------------------------------------
        # Append answer options AFTER text mutation so categories are
        # never exposed to the text optimizer.
        logits = torch.stack([
            self._sut.process_input(
                img, text=txt + answer_suffix, categories=categories,
            )
            for img, txt in zip(images, texts)
        ])

        # -- Tensor conversion for MatrixDistance -------------------------
        perturbed_batch = torch.stack(
            [_pil_to_tensor(img) for img in images]
        )
        origin_batch = origin_tensor.unsqueeze(0).expand_as(perturbed_batch)

        # -- Evaluate 4 batched objectives --------------------------------
        txt_genotypes = genotypes[:, self._manipulator.image_dim :]

        self._objectives.evaluate_all(
            images=[origin_batch, perturbed_batch],
            logits=logits,
            target_classes=target_classes,
            text_genotypes=txt_genotypes,
            text_candidate_distances=(
                self._manipulator.text_candidate_distances
            ),
            batch_dim=0,
        )

        batched_results = self._objectives.results
        fitness = tuple(
            f.cpu().numpy() if isinstance(f, Tensor) else np.asarray(f)
            for f in batched_results.values()
        )

        # -- Assign fitness & update optimizer ----------------------------
        self._optimizer.assign_fitness(fitness)
        self._optimizer.update()

        # -- Build trace rows ---------------------------------------------
        return _build_trace_rows(
            seed_idx, gen, genotypes, logits, texts,
            batched_results, fitness,
            target_classes, categories,
        )

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def _save_seed_results(
        self,
        seed_idx: int,
        seed: SeedTriple,
        trace_rows: list[dict[str, Any]],
        categories: tuple[str, ...],
        answer_suffix: str,
        runtime: float,
    ) -> None:
        run_dir = (
            self._config.save_dir
            / f"{self._config.name}_seed_{seed_idx}_{int(time())}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # -- Trace parquet ------------------------------------------------
        pd.DataFrame(trace_rows).to_parquet(
            run_dir / "trace.parquet", index=False,
        )

        # -- Pareto-optimal candidates ------------------------------------
        for i, cand in enumerate(self._optimizer.best_candidates):
            genotype = cand.solution.astype(np.int64).reshape(1, -1)
            imgs, txts = self._manipulator.manipulate(
                candidates=None, weights=genotype,
            )
            imgs[0].save(run_dir / f"pareto_{i}.png")
            with open(run_dir / f"pareto_{i}.json", "w") as f:
                json.dump(
                    {
                        "genotype": genotype[0].tolist(),
                        "fitness": cand.fitness.tolist()
                        if isinstance(cand.fitness, np.ndarray)
                        else list(cand.fitness),
                        "text": txts[0],
                        "full_prompt": txts[0] + answer_suffix,
                    },
                    f,
                    indent=2,
                )

        # -- Origin image -------------------------------------------------
        seed.image.save(run_dir / "origin.png")

        # -- Stats JSON ---------------------------------------------------
        stats = _build_stats(
            seed_idx, seed, self._config, self._manipulator,
            len(self._optimizer.best_candidates),
            runtime, categories,
        )
        with open(run_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # -- Context metadata (for offline reconstruction) ----------------
        ctx_meta = _build_context_meta(self._manipulator)
        with open(run_dir / "context.json", "w") as f:
            json.dump(ctx_meta, f, indent=2)

        logger.info(f"  Saved results to {run_dir}")
