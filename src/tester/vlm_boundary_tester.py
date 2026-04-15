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
from pathlib import Path
from time import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from smoo import SMOO
from torch import Tensor
from tqdm import tqdm

# Trace / convergence / stats schema version. Bump whenever the
# on-disk layout of trace.parquet, convergence.parquet, or stats.json
# changes in a way that a naive reader can detect:
#   v1 — logprobs is length-2 (target pair only), no cache_hit column,
#        categories in stats.json are the pair.
#   v2 — logprobs is length-N (full category list), cache_hit per-row,
#        categories in stats.json are the full N-category list, and
#        target_classes records the pair positions in the full list.
SMOO_SCHEMA_VERSION = 2

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
    cache_hits: list[bool],
    batched_results: dict[str, Any],
    batched_fitness: tuple,
    target_classes: tuple[int, int],
    categories: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build per-individual trace rows (pure — no side effects).

    Three representations of the per-individual SUT state are stored:

    * ``logprobs`` — length-N length-normalized log-prob vector as
      returned by the scorer. The raw, canonical source; everything
      else is derivable from it. Full FP32 precision.
    * ``probs`` — length-N softmax of ``logprobs``, the full N-class
      probability distribution. Convenience for post-hoc analyses that
      want to work in probability space (entropy, KL, top-k, trees).
    * ``p_class_a`` / ``p_class_b`` — pair-conditional probabilities
      (2-class softmax over just the target pair). These sum to 1 per
      row and preserve the historical 2-class semantics that existing
      viz code in ``analysis/viz_*`` relies on. Prefer ``probs`` in new
      code.

    All three are redundant in information content but serve different
    ergonomic niches; storage is ~N floats per row, negligible.
    """
    idx_a, idx_b = target_classes

    # Full N-class distribution.
    probs = F.softmax(logits, dim=-1)

    # Pair-conditional 2-class softmax for backward-compatible
    # p_class_a / p_class_b columns. Computed from raw logits (not
    # re-normalised from `probs`) so the pair-conditional semantics is
    # exact regardless of how much N-class mass sits on other classes.
    pair_lp = logits[:, [idx_a, idx_b]]
    pair_probs = F.softmax(pair_lp, dim=-1)

    fitness_names = list(batched_results.keys())
    return [
        {
            "seed_id": seed_idx,
            "generation": gen,
            "individual": i,
            "genotype": genotypes[i].tolist(),
            "logprobs": logits[i].tolist(),
            "probs": probs[i].tolist(),
            "decoded_text": texts[i],
            "cache_hit": cache_hits[i],
            "predicted_class": categories[int(logits[i].argmax())],
            "p_class_a": float(pair_probs[i, 0]),
            "p_class_b": float(pair_probs[i, 1]),
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
    full_categories: tuple[str, ...],
    pair: tuple[str, str],
    target_classes: tuple[int, int],
    cache_stats: dict[str, int],
) -> dict[str, Any]:
    """Build stats metadata dict (pure).

    The ``categories`` field is the FULL N-category list the SUT scored
    against (v2 schema). The ``pair`` / ``target_classes`` fields record
    which two entries in that list are the seed's target pair — readers
    can recover ``class_a`` / ``class_b`` positions in the N-dim
    ``logprobs`` column without any external lookup.

    :param full_categories: Full category list scored by the SUT.
    :param pair: ``(class_a, class_b)`` labels.
    :param target_classes: Positions of ``pair`` inside ``full_categories``.
    :param cache_stats: Aggregate ``{"hits", "misses"}`` counts at save time.
    """
    return {
        "schema_version": SMOO_SCHEMA_VERSION,
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
        # v2: full N-category list the SUT scored against. Readers use
        # this to recover the class-name → logprob-index mapping.
        "categories": list(full_categories),
        "pair": list(pair),
        "target_classes": list(target_classes),
        "n_categories_total": len(full_categories),
        "n_pareto": n_pareto,
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        # Reproducibility metadata
        "model_id": config.sut.model_id,
        "max_logprob_gap": config.seeds.max_logprob_gap,
        "n_per_class": config.seeds.n_per_class,
        "n_categories_seed": config.n_categories,
        "image_preset": config.image.preset,
        "image_patch_ratio": config.image.patch_ratio,
        "image_patch_strategy": config.image.patch_strategy.name,
        "image_candidate_strategy": config.image.candidate_strategy.name,
        "image_n_candidates": config.image.n_candidates,
        "text_n_candidates": config.text.n_candidates,
        "device": config.device,
    }


def _write_parquet_with_metadata(
    df: pd.DataFrame,
    path: Path,
    metadata: dict[bytes, bytes],
) -> None:
    """Write a DataFrame to parquet with file-level key/value metadata.

    Uses pyarrow directly (not ``DataFrame.to_parquet``) because the
    latter does not expose the metadata slot on the top-level schema.
    Any existing pandas metadata the round-trip would otherwise generate
    is preserved by merging with our extras.

    :param df: DataFrame to write.
    :param path: Destination parquet path.
    :param metadata: File-level metadata (bytes keys + bytes values).
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = dict(table.schema.metadata or {})
    existing.update(metadata)
    table = table.replace_schema_metadata(existing)
    pq.write_table(table, path)


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

        Applies the optional ``config.seeds.filter_indices`` post-generation
        filter so that targeted re-runs (e.g. Exp-05 Phase A single-seed
        deep probes) use the same indexing as the unfiltered pool — a
        filtered seed at original position 0 still reports as
        ``seed_idx=0`` in output naming.

        :param seeds: Sequence of ``(image, class_a, class_b)`` triples,
            as emitted by :func:`generate_seeds`.
        """
        # Local import avoids a circular dependency between tester and
        # pdq.runner at module load time.
        from src.pdq.runner import _apply_seed_filter

        indexed_seeds = _apply_seed_filter(
            list(seeds), self._config.seeds.filter_indices,
        )

        if not indexed_seeds:
            logger.warning("No seeds after filter — nothing to test.")
            return

        n_gens = self._config.generations
        pbar = tqdm(total=n_gens, unit="gen")

        for pos, (seed_idx, seed) in enumerate(indexed_seeds):
            pbar.reset()
            pbar.set_description(
                f"[{pos + 1}/{len(indexed_seeds)}] "
                f"seed_{seed_idx} {seed.class_a} vs {seed.class_b}"
            )
            self._run_seed(seed_idx, seed, pbar)
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
        pbar: tqdm,
    ) -> None:
        start_time = time()

        # Build pair-specific prompt suffix (only the two target classes).
        # Categories stay out of the mutable text — appended after mutation.
        # NOTE: The prompt suffix is deliberately pair-constrained even though
        # the SUT now scores against the FULL N-category set. This keeps the
        # decision-boundary semantics intact ("Answer with A or B") while the
        # scorer still produces an N-dim log-prob vector for post-hoc analysis
        # (entropy, concentration, tree metrics, etc.). The non-pair classes
        # will typically receive low probability — that is expected and
        # diagnostically useful.
        pair = (seed.class_a, seed.class_b)
        answer_suffix = self._config.answer_format.format(
            categories=", ".join(pair),
        )

        # Score the full category list and locate the pair indices once per
        # seed. The pair indices become ``target_classes`` downstream so that
        # TargetedBalance / Concentration pick out the correct entries of the
        # N-dim log-prob vector (v2 schema — see trace.parquet docs).
        full_categories = tuple(self._config.categories)
        pair_indices = (
            full_categories.index(seed.class_a),
            full_categories.index(seed.class_b),
        )

        # 1. Prepare manipulator with just the question prompt.
        self._manipulator.prepare(
            seed.image, self._config.prompt_template,
        )

        # 2. Re-configure optimizer for this seed's genotype dimensions.
        self._optimizer.update_gene_bounds(self._manipulator.gene_bounds)

        # 3. Target class indices — actual positions of the pair in the full
        #    category list (replaces the old hard-coded (0, 1), which was only
        #    correct when SMOO scored against just the pair).
        target_classes = pair_indices

        # 4. Compute VQGAN-reconstructed baseline (fair comparison for
        #    MatrixDistance — both origin and perturbed go through VQGAN).
        zero_geno = self._manipulator.zero_genotype().reshape(1, -1)
        baseline_imgs, _ = self._manipulator.manipulate(
            candidates=None, weights=zero_geno,
        )
        origin_tensor = _pil_to_tensor(baseline_imgs[0])

        # 5. Generation loop.
        trace_rows: list[dict[str, Any]] = []
        convergence_rows: list[dict[str, Any]] = []

        for gen in range(self._config.generations):
            gen_start = time()

            gen_traces = self._run_generation(
                seed_idx, gen, target_classes, origin_tensor,
                full_categories, answer_suffix,
            )
            trace_rows.extend(gen_traces)

            # -- Convergence stats from Pareto front ----------------------
            pareto = self._optimizer.best_candidates
            pareto_fitness = np.array([c.fitness for c in pareto])
            obj_names = list(self._objectives.results.keys())

            conv_row: dict[str, Any] = {
                "generation": gen,
                "n_pareto": len(pareto),
                "wall_time": time() - gen_start,
            }
            for j, name in enumerate(obj_names):
                conv_row[f"pareto_min_{name}"] = float(pareto_fitness[:, j].min())
                conv_row[f"pareto_mean_{name}"] = float(pareto_fitness[:, j].mean())

            # Population stats from this generation's traces.
            pop_fitness = {name: [] for name in obj_names}
            for row in gen_traces:
                for name in obj_names:
                    pop_fitness[name].append(row[f"fitness_{name}"])
            for name in obj_names:
                vals = np.array(pop_fitness[name])
                conv_row[f"pop_min_{name}"] = float(vals.min())
                conv_row[f"pop_mean_{name}"] = float(vals.mean())

            convergence_rows.append(conv_row)

            # -- Progress bar with convergence info -----------------------
            best_bal = conv_row.get("pareto_min_TgtBal")
            postfix: dict[str, Any] = {"pareto": len(pareto)}
            if best_bal is not None:
                postfix["bal"] = f"{best_bal:.3f}"
            for name in obj_names:
                if name != "TgtBal":
                    short = name.replace("MatrixDistance_", "img_")
                    postfix[short] = f"{conv_row[f'pareto_min_{name}']:.3f}"
            postfix["t"] = f"{time() - gen_start:.0f}s"
            pbar.set_postfix(postfix)
            pbar.update(1)

        # 6. Save everything.
        runtime = time() - start_time
        self._save_seed_results(
            seed_idx, seed, trace_rows, convergence_rows,
            full_categories, pair, target_classes,
            answer_suffix, runtime,
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
        logits_list: list[Tensor] = []
        cache_hits: list[bool] = []
        for img, txt in zip(images, texts):
            logits_list.append(
                self._sut.process_input(
                    img, text=txt + answer_suffix, categories=categories,
                )
            )
            cache_hits.append(self._sut.last_call_cached)

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
        fitness = tuple(
            f.cpu().numpy() if isinstance(f, Tensor) else np.asarray(f)
            for f in batched_results.values()
        )

        # -- Assign fitness & update optimizer ----------------------------
        self._optimizer.assign_fitness(fitness)
        self._optimizer.update()

        # -- Build trace rows ---------------------------------------------
        return _build_trace_rows(
            seed_idx, gen, genotypes, logits, texts, cache_hits,
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
        convergence_rows: list[dict[str, Any]],
        full_categories: tuple[str, ...],
        pair: tuple[str, str],
        target_classes: tuple[int, int],
        answer_suffix: str,
        runtime: float,
    ) -> None:
        """Persist trace / convergence / pareto / stats / context.

        The signature is v2 — previous (pre-refactor) code passed only
        the target pair as ``categories``; now the FULL N-category list
        (``full_categories``) is written to ``stats.json`` so downstream
        readers can recover the index→class mapping for the N-dim
        ``logprobs`` column. The pair / target_classes args are also
        persisted so the pair positions are recoverable from a single
        file (stats.json) without any external lookup.

        :param seed_idx: 0-based seed index.
        :param seed: The seed triple.
        :param trace_rows: List of per-individual trace dicts.
        :param convergence_rows: List of per-generation convergence dicts.
        :param full_categories: Full N-category list the SUT scored
            against — canonical for the ``logprobs`` column index map.
        :param pair: ``(class_a, class_b)`` labels (used for human-
            readable logging and the pareto full_prompt string).
        :param target_classes: Positions of ``pair`` inside
            ``full_categories``; kept on the signature for symmetry and
            passed through to ``_build_stats``.
        :param answer_suffix: Pair-constrained answer format suffix.
        :param runtime: Wall-clock seconds spent on this seed.
        """
        run_dir = (
            self._config.save_dir
            / f"{self._config.name}_seed_{seed_idx}_{int(time())}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # -- Schema metadata embedded into every parquet ------------------
        # Both trace.parquet and convergence.parquet are stamped with the
        # same file-level metadata so a reader can determine the layout
        # without looking at stats.json. stats.json mirrors the version
        # marker for JSON-only consumers.
        parquet_metadata: dict[bytes, bytes] = {
            b"schema_version": str(SMOO_SCHEMA_VERSION).encode(),
            b"pipeline": b"smoo",
            b"categories": json.dumps(list(full_categories)).encode(),
            b"pair": json.dumps(list(pair)).encode(),
            b"target_classes": json.dumps(list(target_classes)).encode(),
        }

        # -- Trace parquet ------------------------------------------------
        _write_parquet_with_metadata(
            pd.DataFrame(trace_rows), run_dir / "trace.parquet",
            parquet_metadata,
        )

        # -- Convergence parquet ------------------------------------------
        _write_parquet_with_metadata(
            pd.DataFrame(convergence_rows),
            run_dir / "convergence.parquet",
            parquet_metadata,
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
            runtime, full_categories, pair, target_classes,
            cache_stats=self._sut.cache_stats,
        )
        with open(run_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # -- Context metadata (for offline reconstruction) ----------------
        ctx_meta = _build_context_meta(self._manipulator)
        with open(run_dir / "context.json", "w") as f:
            json.dump(ctx_meta, f, indent=2)

        logger.info(f"  Saved results to {run_dir}")
