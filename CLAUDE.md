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

## Package boundaries (important)

Both pipelines are allowed to depend on `src/common/` and the other shared
packages. They must **not** depend on each other: no `src/evolutionary/` →
`src/pdq/` or `src/pdq/` → `src/evolutionary/` imports.

Private symbols (`_foo`) stay inside the package that defines them. If you find
yourself wanting to import a `_` -prefixed name across packages, promote it
to public and move it to `src/common/` instead.

Shared helpers currently exposed by `src/common/`:

- `generate_seeds` — 1-vs-all pair scoring over ImageNet
- `apply_seed_filter`, `build_context_meta` — seed-pool filtering + context snapshot
- `seed_matrix.{build_fuzzy_onehot, build_pareto_init, build_precise_scan}`
- `artifacts.{ParquetBuffer, EVOLUTIONARY_SCHEMA_VERSION, PDQ_SCHEMA_VERSION}`

## Live objectives (`src/objectives/`)

Three criteria drive the evolutionary tester. Do not re-introduce the older
`Concentration` / `ArchiveSparsity` / `NormalizedGenomeDistance` objectives —
they were removed for structural conflict with the sparsity goal.

- `MatrixDistance` (from `smoo.objectives.image_criteria`) — image distance
- `TextReplacementDistance` — text distance
- `TargetedBalance` — `|P(A) − P(B)|`, boundary proximity

PDQ does not use this module; it has its own distance metrics under
`src/pdq/metric.py`.

## Running

```bash
pip install -e tools/smoo
pip install -r experiments/requirements.txt

# Evolutionary
python experiments/runners/run_boundary_test.py configs/Exp-NN/<config>.yaml

# PDQ
python experiments/runners/run_pdq_test.py configs/Exp-NN/<config>.yaml
```

Templates in `configs/templates/evolutionary_template.yaml`,
`configs/templates/pdq_template.yaml`, `configs/templates/seed_generator.yaml`.

Run outputs land in `runs/<config.name>_seed_<i>_<ts>/` (tester creates the dir,
writes trace/convergence/stats/context incrementally via `ParquetBuffer`).

## Test baseline

`pytest tests/` has **22 pre-existing failures** (test_evolutionary, test_vlm_sut,
test_objectives — fake-object scaffolding issues unrelated to live code). The
contract for any change is "no net regression": 22 → 22, preferably fewer,
never more.

## External systems referenced by the code

- **Obsidian diary** — `~/Obsidian/Notizen/01 - Active Projects/Master Thesis/`
  - `Experiments/` — `Exp-NN-Title-Case.md` per experiment
  - `Diary/assets/` — figure output via `analysis/core/style.asset_dir`
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
