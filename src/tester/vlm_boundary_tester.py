"""VLM boundary tester — the main orchestrator.

Wires together :class:`VLMManipulator`, :class:`VLMSUT`,
:class:`DiscretePymooOptimizer`, and 5 objectives into a
multi-objective evolutionary search loop.

Four batched objectives run through a :class:`CriterionCollection`;
the fifth (SMOO's :class:`ArchiveSparsity`) is evaluated per-individual
because it is not batched.

Each seed triple ``(image, class_A, class_B)`` is tested independently:
the manipulator prepares image + text contexts, the optimizer evolves
integer genotypes, and every SUT interaction is logged to a Parquet
trace for offline analysis.
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

from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    ArchiveSparsity,
    CriterionCollection,
    NormalizedGenomeDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer

from .config import ExperimentConfig, SeedTriple

logger = logging.getLogger(__name__)

# Number of batched objectives in the CriterionCollection.
# The 5th (ArchiveSparsity) is evaluated separately.
_N_BATCHED = 4
_N_TOTAL = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pil_to_tensor(img: Image.Image) -> Tensor:
    """Convert a PIL image to a ``(C, H, W)`` float tensor in ``[0, 1]``."""
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


class VLMBoundaryTester(SMOO):
    """Multi-objective boundary tester for Vision-Language Models.

    Orchestrates the full optimisation loop per seed::

        seed → prepare → [ask → manipulate → SUT → objectives → fitness
                          → update] × generations → save

    The :class:`CriterionCollection` holds the 4 batched objectives
    (MatrixDistance, TextReplacementDistance, TargetedBalance,
    Concentration).  The 5th objective — :class:`ArchiveSparsity` —
    is evaluated per-individual and appended to the fitness tuple.

    :param sut: The VLM system-under-test.
    :param manipulator: Multi-modal manipulator (image + text).
    :param optimizer: Discrete evolutionary optimizer.
    :param objectives: Criterion collection (4 batched objectives).
    :param config: Experiment-level settings.
    """

    _manipulator: VLMManipulator
    _optimizer: DiscretePymooOptimizer
    _sparsity: ArchiveSparsity | None

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
        self._sparsity = None  # created per-seed (depends on gene_bounds)

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

        logger.info(
            f"Starting experiment '{self._config.name}' "
            f"with {len(seeds)} seed(s), "
            f"{self._config.generations} generations each."
        )

        for seed_idx, seed in enumerate(seeds):
            logger.info(
                f"Seed {seed_idx}: '{seed.class_a}' vs '{seed.class_b}'"
            )
            self._run_seed(
                seed_idx, seed, categories, answer_suffix,
            )
            self._optimizer.reset()
            self._cleanup()

    # ------------------------------------------------------------------
    # Per-seed loop
    # ------------------------------------------------------------------

    def _run_seed(
        self,
        seed_idx: int,
        seed: SeedTriple,
        categories: tuple[str, ...],
        answer_suffix: str,
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

        # 3. Create ArchiveSparsity with gene-bound-aware distance metric.
        self._sparsity = ArchiveSparsity(
            metric=NormalizedGenomeDistance(self._manipulator.gene_bounds),
            regime="min",
            on_genomes=True,
        )

        # 4. Resolve target class indices.
        idx_a = categories.index(seed.class_a)
        idx_b = categories.index(seed.class_b)
        target_classes = (idx_a, idx_b)

        # 5. Compute VQGAN-reconstructed baseline (fair comparison for
        #    MatrixDistance — both origin and perturbed go through VQGAN).
        zero_geno = self._manipulator.zero_genotype().reshape(1, -1)
        baseline_imgs, _ = self._manipulator.manipulate(
            candidates=None, weights=zero_geno,
        )
        origin_tensor = _pil_to_tensor(baseline_imgs[0])

        # 6. Generation loop.
        trace_rows: list[dict[str, Any]] = []
        genome_archive: list[NDArray] = []

        for gen in range(self._config.generations):
            gen_start = time()

            gen_traces = self._run_generation(
                seed_idx, gen, target_classes, origin_tensor,
                genome_archive, categories, answer_suffix,
            )
            trace_rows.extend(gen_traces)

            # Grow archive with current Pareto front.
            for cand in self._optimizer.best_candidates:
                genome_archive.append(cand.solution.copy())

            logger.info(
                f"  Gen {gen + 1}/{self._config.generations} "
                f"({time() - gen_start:.1f}s, "
                f"pareto={len(self._optimizer.best_candidates)}, "
                f"archive={len(genome_archive)})"
            )

        # 7. Save everything.
        runtime = time() - start_time
        self._save_seed_results(
            seed_idx, seed, trace_rows, genome_archive,
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
        genome_archive: list[NDArray],
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
        logits_list: list[Tensor] = []
        for img, txt in zip(images, texts):
            full_prompt = txt + answer_suffix
            logits_list.append(
                self._sut.process_input(
                    img, text=full_prompt, categories=categories,
                )
            )
        logits = torch.stack(logits_list)

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
        batched_fitness = tuple(
            f.cpu().numpy() if isinstance(f, Tensor) else np.asarray(f)
            for f in batched_results.values()
        )

        # -- Evaluate ArchiveSparsity per-individual ----------------------
        sparsity_vals = self._evaluate_sparsity(genotypes, genome_archive)
        sparsity_arr = np.asarray(sparsity_vals)

        # -- Combine into 5-objective fitness tuple -----------------------
        fitness = batched_fitness + (sparsity_arr,)

        # -- Assign fitness & update optimizer ----------------------------
        self._optimizer.assign_fitness(fitness)
        self._optimizer.update()

        # -- Build trace rows ---------------------------------------------
        probs = F.softmax(logits, dim=-1)
        idx_a, idx_b = target_classes

        traces: list[dict[str, Any]] = []
        for i in range(len(genotypes)):
            row: dict[str, Any] = {
                "seed_id": seed_idx,
                "generation": gen,
                "individual": i,
                "genotype": genotypes[i].tolist(),
                "logprobs": logits[i].tolist(),
                "decoded_text": texts[i],
                "predicted_class": categories[int(logits[i].argmax())],
                "p_class_a": float(probs[i, idx_a]),
                "p_class_b": float(probs[i, idx_b]),
            }
            for j, name in enumerate(batched_results.keys()):
                row[f"fitness_{name}"] = float(batched_fitness[j][i])
            row["fitness_ArchiveSparsity"] = float(sparsity_arr[i])
            traces.append(row)

        return traces

    def _evaluate_sparsity(
        self,
        genotypes: NDArray,
        genome_archive: list[NDArray],
    ) -> list[float]:
        """Evaluate ArchiveSparsity per-individual.

        :param genotypes: Population array ``(pop_size, n_var)``.
        :param genome_archive: Archive of genotypes from past generations.
        :returns: List of sparsity values, one per individual.
        """
        if not genome_archive:
            return [0.0] * len(genotypes)

        results: list[float] = []
        for genotype in genotypes:
            val = self._sparsity.evaluate(
                images=[None, None],
                solution_archive=[],
                genome_target=genotype,
                genome_archive=genome_archive,
            )
            results.append(float(val))
        return results

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def _save_seed_results(
        self,
        seed_idx: int,
        seed: SeedTriple,
        trace_rows: list[dict[str, Any]],
        genome_archive: list[NDArray],
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
        stats = {
            "seed_idx": seed_idx,
            "class_a": seed.class_a,
            "class_b": seed.class_b,
            "prompt_template": self._config.prompt_template,
            "answer_format": self._config.answer_format,
            "runtime_seconds": runtime,
            "generations": self._config.generations,
            "pop_size": self._config.pop_size,
            "gene_bounds": self._manipulator.gene_bounds.tolist(),
            "image_dim": self._manipulator.image_dim,
            "text_dim": self._manipulator.text_dim,
            "categories": list(categories),
            "n_pareto": len(self._optimizer.best_candidates),
            "archive_size": len(genome_archive),
        }
        with open(run_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # -- Context metadata (for offline reconstruction) ----------------
        ctx_meta = {
            "image_patch_positions": (
                self._manipulator.image_context.selection.positions.tolist()
            ),
            "image_original_codes": (
                self._manipulator.image_context.selection.original_codes.tolist()
            ),
            "image_candidates": [
                c.tolist()
                for c in self._manipulator.image_context.selection.candidates
            ],
            "text_word_positions": (
                self._manipulator.text_context.selection.positions.tolist()
            ),
            "text_original_words": list(
                self._manipulator.text_context.selection.original_words
            ),
            "text_candidates": [
                list(c)
                for c in self._manipulator.text_context.selection.candidates
            ],
            "text_candidate_distances": [
                d.tolist()
                for d in self._manipulator.text_candidate_distances
            ],
        }
        with open(run_dir / "context.json", "w") as f:
            json.dump(ctx_meta, f, indent=2)

        logger.info(f"  Saved results to {run_dir}")
