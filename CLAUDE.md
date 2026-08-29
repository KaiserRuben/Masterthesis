# Claude project conventions

Invariants and gotchas only. [README.md](README.md) has the project
description, install, runners, layout and core concepts — do not duplicate it
here.

## Package boundaries

`src/evolutionary/` and `src/pdq/` must **not** import each other. Both may
depend on `src/common/` and the other shared packages.

`src/boundary_pair/` is the one exception: it composes both stages and imports
from each. New code that needs both belongs there, not in a cross-import.

Private symbols (`_foo`) stay inside the package that defines them. Wanting to
import a `_`-prefixed name across packages means promoting it to public and
moving it to `src/common/`.

`src/common/__init__.py` defines `__all__` — that is the authoritative list,
not a copy in this file. Shared but *not* re-exported:
`seed_matrix.{build_fuzzy_onehot, build_pareto_init, build_precise_scan}` and
`artifacts.{ParquetBuffer, EVOLUTIONARY_SCHEMA_VERSION, PDQ_SCHEMA_VERSION}`.

## Objectives that must not come back

`src/objectives/` holds exactly `TargetedBalance` and `TextEmbeddingDistance`
(plus `MatrixDistance` re-exported from smoo). Do not re-introduce
`Concentration`, `ArchiveSparsity` or `NormalizedGenomeDistance` — they were
removed for structural conflict with the sparsity goal.

## Experiment numbering (authoritative)

`Exp-NN-Title-Case` names come from the Obsidian diary at
`~/Obsidian/Notizen/01 - Active Projects/Master Thesis/Experiments/`. That
diary, not this repository, is the source of truth for experiment identity.
Starting a new experiment means checking the diary for the next free `Exp-NN`
and mirroring its title across `configs/Exp-NN/`, `runs/Exp-NN/` and
`notebooks/Exp-NN-*.ipynb`.

## Archive convention

Each top-level domain owns its own `Archive/` subdir — `configs/Archive/`,
`experiments/Archive/`, `tools/Archive/`. Move superseded work in, never
delete. Do not create a repo-root `archive/`; that name is taken by
`archive_alpamayo_jan2026/`, an unrelated research thread.

`runs/` is untracked, so "never delete" buys nothing there — archiving a run
only moves it on disk. `runs/Archive/` no longer exists and Exp-01's data went
with it, which is why that experiment is not reproducible. Copy runs worth
keeping to the external archive ([docs/DATA.md](docs/DATA.md)), not just to
`runs/Archive/`.

## Test baseline

`pytest tests/` must stay green — 725 tests. Two flake on unseeded pymoo
fixtures (`test_trace_row_count`, `test_initial_population_shape`); rerun
before blaming your change, both pass in isolation.

## Human-subject data (HS-01)

`experiments/HS-01/results/sessions/` holds participant records. The study app
records a browser environment block for quality control whose
`user_agent`/`platform`/`screen`/`device_pixel_ratio` combination is a device
fingerprint — it identified most participants uniquely — while the consent text
promises no personal data is stored.

Any record added or refreshed here must be scrubbed before it is committed:

```bash
python experiments/HS-01/tools/anonymize_sessions.py --check     # verify
python experiments/HS-01/tools/anonymize_sessions.py --in-place  # scrub
```

The analysis reads the derived `device` field, never the raw fields, so this
costs nothing. Keep unscrubbed originals outside the repository.

**Never regenerate `consent.en.md` or `study-config.json`.** The config's
`consent.text_sha256` is stale by design: the served consent text was
hand-edited after generation, so regenerating would overwrite the record of
what 49 participants actually saw and break `config_sha256` against every
session record. Background in
[docs/DATA.md](docs/DATA.md#consent-text-the-recorded-hash-is-stale).

`experiments/HS-01/tests/test_make_study_config.py` used to run the generator
against the real repository paths, silently overwriting that served text on
every suite run. It now writes to a temporary directory, and
`test_shipped_files_untouched` fails if that regresses.

## Keeping docs true

`docs/REPRODUCTION.md`, `docs/ENVIRONMENT.md` and `docs/DATA.md` make checkable
claims — file counts, config paths, which figures need raw runs, which tests
pass. Changing configs, requirements, runners or figure scripts means
re-checking them, not assuming.

## External systems referenced by the code

- **Obsidian diary** — `~/Obsidian/Notizen/01 - Active Projects/Master Thesis/`
  - `Experiments/` — `Exp-NN-Title-Case.md` per experiment
  - `Diary/assets/` — default figure output via `analysis/core/style.asset_dir`,
    used only when the vault exists; override with `ANALYSIS_ASSET_ROOT`
- **Redis (inference cache)** — Docker volume on external SanDisk drive (see
  `memory/infra_redis_volume.md`)

## Gotchas

- **Case-insensitive filesystem (APFS).** `git mv src/tester src/evolutionary`
  works because the names differ; same-case renames (`EXP-05` → `Exp-05`) need
  a two-step rename via an intermediate name.
- **`tools/alpamayo/`** is gitignored but present on disk — a separate git
  repository for an unrelated thread, not a submodule.
- **Notebooks hardcode run paths** as strings inside the JSON; renaming a run
  dir means editing the notebooks.
- **BSD sed has no `\b`.** Use plain substrings or `[^a-z_]` guards.
