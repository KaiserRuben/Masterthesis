# VLM Boundary Testing

Given an image and two class labels, find the minimal input perturbation —
image, text, or both — that drives a Vision-Language Model onto the decision
boundary between them.

Reproduction package for the master's thesis *Multi-Modal Boundary Testing of
Vision–Language Models: A Two-Stage Search Framework for Decision-Space
Geometry* (Technical University of Munich). Contact: **Ruben.Kaiser@tum.de**

| I want to… | Start here |
|---|---|
| Install and verify a clone | [Quickstart](#quickstart) |
| Run a search of my own | [Quickstart](#quickstart), then [Pipelines](#pipelines) |
| Rebuild a thesis figure | [docs/REPRODUCTION.md — figure index](docs/REPRODUCTION.md#figure-index) |
| Find the config behind a claim | [docs/REPRODUCTION.md — experiment index](docs/REPRODUCTION.md#experiment-index) |
| Get the 24 GB raw run archive | [docs/DATA.md — tier 2](docs/DATA.md#tier-2-the-raw-run-archive) |
| Set up the second (OpenVINO) stack | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md#openvino-install-quantized-suts) |
| Change the code | [Where things live](#where-things-live) |

## Quickstart

```bash
git clone --recurse-submodules https://github.com/KaiserRuben/Masterthesis.git
cd Masterthesis
conda create -n uni python=3.13 && conda activate uni

# Install the torch build for your accelerator FIRST. requirements.txt pins
# torch without a platform tag, so pip will otherwise resolve the wrong wheel.
pip install torch==2.8.0 torchvision==0.23.0      # Apple Silicon / CPU
pip install -r experiments/requirements.txt       # also installs tools/smoo editable

pytest tests/                                     # 725 tests
```

Two tests flake on unseeded pymoo fixtures — `test_trace_row_count` and
`test_initial_population_shape`. Both pass in isolation; rerun before assuming
your environment is broken.

**Smoke test that needs nothing else** — no model weights, no ImageNet, no
LaTeX. It reads the human-study records tracked in this repository:

```bash
python -m analysis.hs01.run_all     # 12 tables -> analysis/outputs/hs01/, 12 figures
```

**Run an actual search.** ImageNet is gated on the HuggingFace Hub, so accept
the `ILSVRC/imagenet-1k` licence on your account and export a token; model
weights download on first use.

```bash
export HF_TOKEN=hf_...
python experiments/runners/run_boundary_test.py \
    configs/Exp-10/phase1_shark_n16383.yaml --preflight
```

`--preflight` times 20 representative SUT calls and prints a total-runtime
projection before the search starts — it does not abort, so Ctrl-C if the
number is unacceptable. Campaigns in this repository ran hours to days.
Output lands in `runs/exp10/exp10_phase1_shark_n16383_seed_<i>_<ts>/` as
`trace.parquet`, `convergence.parquet`, `stats.json`, `context.json`, Pareto
snapshots and the origin image.

> `configs/Exp-10/` and `configs/templates/` resolve their caches under `~`, so
> they run unedited. Most other configs do not: **210 of 299 point
> `image.knn_cache_path` and the primary ImageNet cache at
> `/mnt/storage/huggingface/`**, the original workstation. Rewrite those two
> keys before reusing one. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md#configuring-paths).

## Pipelines

Three entry points. All three share seed generation (`src/common/`), SUT
scoring (`src/sut/`) and the manipulators (`src/manipulator/`); they differ in
how they search. `src/evolutionary/` and `src/pdq/` must never import each
other.

| | Evolutionary | PDQ | Boundary-pair |
|---|---|---|---|
| Strategy | AGE-MOEA-II, 3 objectives | AutoBVA-style two-stage directed | evolutionary → PDQ, per seed |
| Code | `src/evolutionary/` | `src/pdq/` | `src/boundary_pair/` |
| Runner | `run_boundary_test.py` | `run_pdq_test.py` | `run_boundary_pair_test.py` |
| Template | `evolutionary_template.yaml` | `pdq_template.yaml` | `boundary_pair_template.yaml` |
| Yields | Pareto front of near-boundary genotypes | flips (stage 1), minimised flips (stage 2) | `(anchor, partner)` pairs at minimum genome distance |

Runners live in `experiments/runners/`, templates in `configs/templates/`.

Boundary-pair is the one to reach for if you are reproducing **Exp-100** — the
most-cited campaign, behind 7 of the 13 results figures. It takes each Pareto
member from the evolutionary stage as a PDQ anchor, which is what yields the
canonical Boundary Value Analysis characterisation.

## Core concepts

**Genotype** — `int64[n]` = `[image_genes | text_genes]`. Gene `0` keeps the
original; gene `k` selects the `k`-th nearest candidate. Candidates are sorted
by embedding distance, so `1` is the *minimal* perturbation and the search is
biased toward small integers whenever a sparsity prior is in place.

**Manipulators** (`src/manipulator/`) turn a genotype into a model input:

| Half | Backends |
|---|---|
| Image | `vqgan_codebook` (default) — codebook swaps; `stylegan_xl` — latent edits, needs a loaded SUT |
| Text | composite stack: MLM-Synonym (ModernBERT-large) → Fragmentation → Character Noise → Saliency, in canonical order |

`VLMManipulator` bridges the two halves. The lifecycle is two-phase:
`prepare(input) → context` once, then `apply(context, genotype)` many times.
Context is immutable and shared across genotypes.

**SUT** (`src/sut/`) — teacher-forced log-prob scoring. Each candidate label is
force-decoded given the perturbed input; per-token log-probs are
length-normalised. Backends: `torch` (MPS/CUDA/CPU) and `openvino` (Intel Arc,
for the INT8/INT4 variants — separate environment, see [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md#openvino-install-quantized-suts)).

**Objectives** (evolutionary only, `src/objectives/`):

| Criterion | Measures |
|---|---|
| `MatrixDistance` | Frobenius norm of (origin − perturbed) image |
| `TextEmbeddingDistance` | cosine distance of prompt vs. anchor in the SUT's own sentence-embedding space |
| `TargetedBalance` | `\|P(A) − P(B)\|` — zero at the decision boundary |

PDQ does not use this module; its metrics are in `src/pdq/metric.py`.

**Init distribution** (`src/optimizer/sparse_sampling.py`) — `uniform` (PyMoo
default) or `sparse` (Bernoulli-gated zero-anchor plus geometric depth). Sparse
init is *required* at full codebook size (n=16383): under uniform sampling zero
is not a privileged gene value, so the optimizer never reaches the
`(L0, TgtBal)` sparse-near-boundary corner.

## Where things live

| To change… | Edit |
|---|---|
| how a genotype becomes an image | `src/manipulator/image/` (VQGAN), `src/manipulator/image_stylegan/` |
| how a genotype becomes a prompt | `src/manipulator/text/composite.py`, `text/operators/` |
| how the VLM is scored | `src/sut/vlm_sut.py`, `src/sut/scorer.py` |
| what the evolutionary search optimises | `src/objectives/` |
| how the initial population is drawn | `src/optimizer/sparse_sampling.py` |
| which (image, class-pair) seeds get tested | `src/common/seed_generator.py` + the `seeds:` config block |
| PDQ's flip rule or distance metrics | `src/pdq/flip_policy.py`, `src/pdq/distances/` |
| any config field | `src/config.py` — the dacite schema every YAML is loaded into |

```
src/          config.py, common/, sut/, manipulator/, objectives/, optimizer/,
              data/, evolutionary/, pdq/, boundary_pair/, utils/
experiments/  runners/ (entry points), preprocessing/, validation/,
              HS-01/ (human oracle study: app, item pool, sessions),
              analysis/ (per-campaign aggregates the figures read)
configs/      templates/, Exp-03…Exp-105/, HS-GEN-01…03/, Archive/
analysis/     core/, viz/ (+ viz/thesis/pgf/ — the 13 results-figure emitters),
              cartography/, hs01/, outputs/ (generated, untracked)
docs/         REPRODUCTION.md, ENVIRONMENT.md, DATA.md
tools/smoo/   SMOO framework (git submodule, patched fork)
runs/         search output — untracked except runs/preprocessing/
```

Experiment identifiers (`Exp-NN`, `HS-GEN-NN`) are shared across `configs/`,
`runs/` and `notebooks/`. Superseded material moves into the `Archive/` subdir
of its own domain rather than being deleted.

Two branches. `main` carries everything, including slides, parked tooling and
two unrelated research threads (`archive_alpamayo_jan2026/`,
`infrastructure/`). `repro/thesis-v1` is pruned to what the thesis cites.

## Reproducing the thesis

A clone rebuilds the HS-01 tables and 8 of the 13 results figures. The other 5
(`boundary_map`, `budget`, `config_effects`, `init_coverage`, `watershed`) read
raw run directories from the [tier-2 archive](docs/DATA.md#tier-2-the-raw-run-archive).

Every results figure additionally needs **`pdflatex` on `PATH`** and **the
thesis tree at `$THESIS_DIR`** — the emitters typeset each figure to measure
its box, and `\input` a style file that lives thesis-side. Without it you get:

```
! LaTeX Error: File `figures/results/results-style.tex' not found.
```

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) maps every experiment to its
config, run directory and figure producer.

## SMOO

```python
from smoo.objectives import Criterion, CriterionCollection
from smoo.optimizer import Optimizer
from smoo.sut import SUT
```

`tools/smoo/` is a submodule tracking
[`KaiserRuben/SMOO@masterarbeit`](https://github.com/KaiserRuben/SMOO/tree/masterarbeit),
three commits over [upstream](https://github.com/oliverweissl/SMOO) adding
packaging metadata, timm 1.0 compatibility for the StyleGAN-XL pickles, and
`inference_mode` on the manipulator entry points. The pinned dependency set
needs all three. Details in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md#why-the-smoo-submodule-points-at-a-fork).

## Licensing

Everything authored for this thesis may be used, modified and redistributed for
any **noncommercial** purpose — academic research, teaching, personal study,
hobby projects. Commercial use needs separate permission
(Ruben.Kaiser@tum.de). These are not OSI-approved open-source licenses; parts
of the dependency stack (StyleGAN-XL, ImageNet imagery) are noncommercial-only
in any case.

| Material | License |
|---|---|
| Source code, configs | [PolyForm Noncommercial 1.0.0](LICENSE) |
| Docs, notebooks, aggregates, figures | [CC BY-NC 4.0](LICENSE-DATA) |
| HS-01 participant records | [Academic research only](experiments/HS-01/results/LICENSE) — narrower, set by the study consent |
| Reference photos, SMOO, model weights | [Third-party terms](THIRD-PARTY-NOTICES.md) |

Two caveats worth reading before you redistribute anything: the `tools/smoo`
submodule has no license upstream, and the ImageNet-derived imagery is included
for research reproduction only. Both in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Supervision

| Role | Name | Affiliation |
|------|------|-------------|
| Supervisor | Prof. Andrea Stocco | TUM / fortiss |
| Co-Supervisor | Oliver Weißl | fortiss |
