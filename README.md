# VLM Boundary Testing

Search-based boundary testing for Vision-Language Models. Given an image and two class labels, find the minimal input perturbation (image + text) that pushes the VLM to the decision boundary between those classes.

## Two pipelines

Both pipelines share seed generation, SUT scoring, and the image/text manipulators (`src/common/`, `src/sut/`, `src/manipulator/`). They differ only in the search strategy.

**Evolutionary** — `src/evolutionary/`, AGE-MOEA-II multi-objective search over discrete genotypes.
- Entry: `experiments/runners/run_boundary_test.py`
- Template: `configs/templates/evolutionary_template.yaml`

**PDQ** — `src/pdq/`, AutoBVA-style two-stage directed search (Stage 1 discovers flips, Stage 2 minimises each flip).
- Entry: `experiments/runners/run_pdq_test.py`
- Template: `configs/templates/pdq_template.yaml`

## Core concepts

**Genotype** — `int64[n]` = `[image_genes | text_genes]`. `0` = keep original, `k` = use the `k`-th nearest candidate (sorted by embedding distance, so gene `1` is the minimal perturbation).

**Manipulators** (`src/manipulator/`) apply the genotype:
- Image: VQGAN codebook swaps
- Text: PoS-aware synonym replacement (spaCy + FastText)
- VLMManipulator bridges the two halves of the genotype

**SUT** (`src/sut/`) — teacher-forced log-prob scoring. For each category label, the VLM is forced to decode the label given the perturbed input and the per-token log-probs are length-normalised.

**Objectives** (evolutionary only, `src/objectives/`):
- `MatrixDistance` — Frobenius norm of (origin − perturbed) image
- `TextReplacementDistance` — Σ cosine distance of replaced words
- `TargetedBalance` — `|P(A) − P(B)|`, → 0 at the decision boundary

**Init distribution** (`src/optimizer/sparse_sampling.py`) — `uniform` (PyMoo default) or `sparse` (Bernoulli-gated zero-anchor + geometric depth). Sparse init is required for full-codebook runs (n=16383); without a sparsity prior, uniform init prevents the optimizer from reaching the `(L0, TgtBal)` sparse-near-boundary corner.

## Install and run

```bash
git submodule update --init tools/smoo
pip install -e tools/smoo
pip install -r experiments/requirements.txt

# Evolutionary (example: Exp-10 Phase-1 sparse-init shark)
python experiments/runners/run_boundary_test.py configs/Exp-10/phase1_shark_n16383.yaml

# PDQ
python experiments/runners/run_pdq_test.py configs/templates/pdq_template.yaml
```

Outputs land under `runs/<config.name>_seed_<i>_<ts>/` with `trace.parquet`, `convergence.parquet`, `stats.json`, `context.json`, Pareto snapshots, and the origin image.

## Repository layout

```
src/
├── config.py                 # ExperimentConfig (dacite-loaded from YAML)
├── common/                   # shared between evolutionary and pdq
│   ├── artifacts.py          # ParquetBuffer + SCHEMA_VERSION constants
│   ├── seed_context.py       # apply_seed_filter, build_context_meta
│   ├── seed_generator.py     # generate_seeds (1-vs-all pair scoring)
│   └── seed_matrix.py        # fuzzy/precise sampling-matrix builders
├── data/                     # ImageNet labels + streaming cache
├── evolutionary/
│   └── vlm_boundary_tester.py
├── manipulator/
│   ├── image/                # VQGAN codebook swaps
│   ├── text/                 # PoS-aware synonym replacement
│   └── vlm_manipulator.py    # multi-modal bridge
├── objectives/               # evolutionary-only criteria (+ MatrixDistance re-export from smoo)
├── optimizer/                # DiscretePymooOptimizer, early_stop, sparse_sampling
├── pdq/                      # PDQ directed-search pipeline
├── sut/                      # VLMSUT, VLMScorer, preflight
└── utils/
    └── pair_resolver.py      # CLI helper: "class_a->class_b" → seed-pool index

experiments/
├── runners/                  # run_boundary_test.py, run_pdq_test.py
├── preprocessing/            # generate_*, precompute_taxonomy, sample_pairs_exp11, preview_seed_pool
├── validation/               # validate_saliency_{embedding,rarity}
└── Archive/                  # historical: question_rewrite, imagenet_vlm_probing, run_screening, Phase-0..4

configs/
├── templates/                # evolutionary_template.yaml, pdq_template.yaml, seed_generator.yaml
├── Exp-03/ ... Exp-11/       # per-experiment overrides (names match Obsidian diary)
└── Archive/                  # superseded configs

runs/
├── Exp-02/ ... Exp-10/       # run outputs keyed to the Obsidian Exp-NN numbering
│   └── Exp-03-{mac,workstation}/  # same experiment, separated by machine
├── preprocessing/            # e.g. runs/preprocessing/taxonomy/
└── Archive/                  # historical: Exp-01-SMOO-Pipeline-Validation/{01_5obj,02_4obj,03_cadence}

analysis/
├── core/                     # style, resolve, load_{smoo,pdq}, generate, parquet_utils, metrics
├── viz/                      # boundary, pdq, smoo, topology, g_field, g_surface, comparison,
│                             #   convergence_study, geometry_study, two_weeks
├── outputs/                  # cached analysis artifacts (per-experiment)
└── slides/                   # figure outputs curated for presentations

notebooks/
├── Exp-01-sparsity-analysis.ipynb
├── Exp-01-4obj-analysis.ipynb
└── Exp-01-cadence-analysis.ipynb

tools/
├── smoo/                     # SMOO framework (git submodule)
├── alpamayo/                 # gitignored — separate research thread
└── Archive/                  # parked: VLTest, vlm, scene, parquet_footer_repair

archive_alpamayo_jan2026/     # side research (January '26 VLM grounding / Alpamayo-R1)
infrastructure/               # docker, local (MPS), workstation (CUDA) environment setup
tests/                        # pytest suite
```

## Experiment naming

The `Exp-NN` numbering comes from the Obsidian diary at
`~/Obsidian/Notizen/01 - Active Projects/Master Thesis/Experiments/`
and is the single source of truth for experiment identity. Files and folders
in this repo follow the same scheme:

- Config dir `configs/Exp-NN/` ↔ runs dir `runs/Exp-NN/` ↔ Obsidian note `Exp-NN-Title.md`
- Historical runs that predate the `Exp-NN` scheme live under `runs/Archive/`
- Each domain (`configs/`, `experiments/`, `runs/`, `tools/`) has its own `Archive/` subdir for superseded content

## Design decisions

- **Two-phase manipulator lifecycle** — `prepare(input) → context` (once), then `apply(context, genotype) → output` (many). Context is immutable; multiple genotypes reuse the same prepared context.
- **Candidates sorted by embedding distance** — gene `1` is the minimal perturbation; gene `0` keeps the original. The optimizer is therefore biased toward small integer values when a sparsity prior is in place.
- **Streaming Parquet writers** (`src.common.artifacts.ParquetBuffer`) — row groups flush on interval so a crash loses at most one group rather than the whole seed.
- **Schema versions** (`src.common.artifacts.EVOLUTIONARY_SCHEMA_VERSION`, `PDQ_SCHEMA_VERSION`) — per-pipeline; bumped when on-disk layout changes.
- **ArchiveSparsity / Concentration removed** — earlier objectives that conflicted with the sparse-near-boundary goal. Only the three live objectives above are carried in the tester today.

## SMOO integration

```python
from smoo.objectives import Criterion, CriterionCollection
from smoo.optimizer import Optimizer
from smoo.sut import SUT
```

`tools/smoo/` is installed editable (`pip install -e tools/smoo`); the package's own `pyproject.toml` maps the `smoo` import to its `src/`.

## Supervision

| Role | Name | Affiliation |
|------|------|-------------|
| Supervisor | Prof. Andrea Stocco | TUM / fortiss |
| Co-Supervisor | Oliver Weißl | fortiss |
