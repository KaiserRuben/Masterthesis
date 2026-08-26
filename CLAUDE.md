# Claude project conventions

Instructions for working with this codebase. Point Claude at `README.md` for the
human-oriented description; this file captures invariants and gotchas.

## What this project is

Search-based boundary testing for Vision-Language Models. Two search pipelines
share a common manipulation/SUT/seed layer:

- **Evolutionary** (`src/evolutionary/`) — AGE-MOEA-II over discrete genotypes.
- **PDQ** (`src/pdq/`) — AutoBVA-style two-stage directed search.

Shared infrastructure lives in `src/common/`, `src/sut/`, `src/manipulator/`,
`src/data/`, `src/objectives/`, `src/optimizer/`, `src/utils/`.

## Experiment numbering (authoritative)

`Exp-NN-Title-Case` names come from the Obsidian diary at
`~/Obsidian/Notizen/01 - Active Projects/Master Thesis/Experiments/`. That diary
is the source of truth for experiment identity.

Inside this repo every experiment-tied artifact follows `Exp-NN`:

- Configs: `configs/Exp-NN/...`
- Runs:    `runs/Exp-NN/...` (multi-machine variants use `Exp-NN-mac`, `Exp-NN-workstation`)
- Notebooks: `notebooks/Exp-NN-*.ipynb`

When starting a new experiment, check the diary first for the next free
`Exp-NN` and mirror the title.

## Archive convention

Each top-level domain owns its own `Archive/` subdir. Move superseded work in,
never delete. Do not create a new repo-root `archive/` — that name is taken by
`archive_alpamayo_jan2026/` (an unrelated older research thread).

- `configs/Archive/`
- `experiments/Archive/`
- `runs/Archive/`
- `tools/Archive/`

`runs/` is not tracked in git, so "never delete" buys nothing there — archiving
a run only moves it on disk. `runs/Archive/` no longer exists and Exp-01's data
went with it, which is why that experiment is not reproducible. Copy runs worth
keeping to the external archive (`docs/DATA.md`), not just to `runs/Archive/`.

## Package boundaries (important)

Both pipelines are allowed to depend on `src/common/` and the other shared
packages. They must **not** depend on each other: no `src/evolutionary/` →
`src/pdq/` or `src/pdq/` → `src/evolutionary/` imports.

Private symbols (`_foo`) stay inside the package that defines them. If you find
yourself wanting to import a `_` -prefixed name across packages, promote it
to public and move it to `src/common/` instead.

`src/common/__init__.py` defines `__all__` — treat that as the authoritative
list rather than duplicating it here. Broadly it covers seed generation
(`generate_seeds`, `roster_seeds`, `slot_items_seeds`), seed-pool filtering and
context capture (`apply_seed_filter`, `build_context_meta`), pipeline bootstrap
(`init_shared_components`, `prepare_pipeline_seeds`, `precompute_image_backend`),
resume (`compute_resume_filter`), the Redis byte cache, hex grids, and worker
dispatch. Not re-exported but equally shared:
`seed_matrix.{build_fuzzy_onehot, build_pareto_init, build_precise_scan}` and
`artifacts.{ParquetBuffer, EVOLUTIONARY_SCHEMA_VERSION, PDQ_SCHEMA_VERSION}`.

## Live objectives (`src/objectives/`)

Three criteria drive the evolutionary tester. Do not re-introduce the older
`Concentration` / `ArchiveSparsity` / `NormalizedGenomeDistance` objectives —
they were removed for structural conflict with the sparsity goal.

- `MatrixDistance` (from `smoo.objectives.image_criteria`) — image distance
- `TextEmbeddingDistance` — sentence-level cosine distance in the SUT's
  own embedding space (mean-pooled Qwen text-backbone hidden state)
- `TargetedBalance` — `|P(A) − P(B)|`, boundary proximity

PDQ does not use this module; it has its own distance metrics under
`src/pdq/metric.py`.

## Running

```bash
git submodule update --init tools/smoo   # patched fork, see below
pip install -r experiments/requirements.txt   # installs tools/smoo editable

# Evolutionary
python experiments/runners/run_boundary_test.py configs/Exp-NN/<config>.yaml

# PDQ
python experiments/runners/run_pdq_test.py configs/Exp-NN/<config>.yaml
```

`tools/smoo` tracks `KaiserRuben/SMOO` branch `masterarbeit` — three commits on
top of upstream adding packaging metadata, timm 1.0 compatibility for the
StyleGAN-XL pickles, and `inference_mode` on the manipulator entry points. The
pinned dependency set needs all three, so do not repoint the submodule at
upstream `oliverweissl/SMOO` until they land there.

Configs with `sut.backend: openvino` (the INT8/INT4 SUTs) need a **separate**
environment — `experiments/requirements-openvino.txt` pins an older
`transformers` than the base file, so the two do not co-install.

Reproduction-facing documentation lives in `docs/`:
[REPRODUCTION.md](docs/REPRODUCTION.md) maps experiments to figures,
[ENVIRONMENT.md](docs/ENVIRONMENT.md) covers the two hardware stacks, and
[DATA.md](docs/DATA.md) covers the external run archive. Keep them in sync when
changing configs, requirements, or figure scripts.

Templates in `configs/templates/evolutionary_template.yaml`,
`configs/templates/pdq_template.yaml`, `configs/templates/seed_generator.yaml`.

Run outputs land in `runs/<config.name>_seed_<i>_<ts>/` (tester creates the dir,
writes trace/convergence/stats/context incrementally via `ParquetBuffer`).

## Test baseline

`pytest tests/` is fully green (725 passed) — keep it that way. Any change must
leave the suite passing.

Two tests flake because pymoo's `setup()` is unseeded in their fixtures:
`test_evolutionary.py::TestVLMBoundaryTester::test_trace_row_count` and
`test_discrete_optimizer.py::TestConstruction::test_initial_population_shape`.
Rerun before blaming your change; both pass in isolation.

## Human-subject data (HS-01)

`experiments/HS-01/results/sessions/` holds participant records. The study app
records a browser environment block for quality control whose
`user_agent`/`platform`/`screen`/`device_pixel_ratio` combination is a device
fingerprint — it identified most participants uniquely — and the consent text
promises no personal data is stored.

Any record added or refreshed here must be scrubbed before it is committed:

```bash
python experiments/HS-01/tools/anonymize_sessions.py --check     # verify
python experiments/HS-01/tools/anonymize_sessions.py --in-place  # scrub
```

The analysis reads the derived `device` field, never the raw fields, so this
costs nothing. Keep unscrubbed originals outside the repository.

## External systems referenced by the code

- **Obsidian diary** — `~/Obsidian/Notizen/01 - Active Projects/Master Thesis/`
  - `Experiments/` — `Exp-NN-Title-Case.md` per experiment
  - `Diary/assets/` — default figure output via `analysis/core/style.asset_dir`,
    used only when the vault exists; override with `ANALYSIS_ASSET_ROOT`, and
    point the thesis figure scripts at a checkout with `THESIS_DIR`
- **Redis (inference cache)** — Docker volume on external SanDisk drive (see `memory/infra_redis_volume.md`)

## Gotchas

- Case-insensitive filesystem (APFS). `git mv src/tester src/evolutionary` works
  because the names differ; same-case renames (`EXP-05` → `Exp-05`) require a
  two-step rename via an intermediate name.
- `tools/alpamayo/` is gitignored but present on disk — it is a separate git
  repository for an unrelated research thread, not a submodule.
- Notebooks in `notebooks/` reference run paths as hardcoded strings inside the
  JSON; renaming a run dir means updating the notebooks.
- BSD sed has no `\b`. Use plain substrings or `[^a-z_]` guards for word boundaries.
