# Data availability

The evidence behind the thesis is split across three tiers by size and by what
may be published.

## Tier 1 — in this repository

The code, configs, and the derived aggregates that eight of the thirteen
results figures are built from:

| What | Where | Size |
|---|---|---|
| Derived aggregates (per-seed summaries, cartography point clouds, PMI tables) | `experiments/analysis/output/**/*.{csv,parquet}` | 55 MB |
| HS-01 session records (anonymized) and frozen item pool | `experiments/HS-01/results/sessions/`, `experiments/HS-01/pool_frozen/itempool.json` | 1.4 MB |
| HS-01 analysis outputs | regenerated into `analysis/outputs/hs01/` by `python -m analysis.hs01.run_all` — not tracked | — |
| All experiment configs | `configs/` | 1.5 MB |
| Source, tests, analysis and figure code | `src/`, `tests/`, `analysis/`, `tools/` | 4.4 MB |

Eight of the thirteen results figures are emitted from the tier-1 aggregates
alone. The other five — `boundary_map`, `budget`, `config_effects`,
`init_coverage`, `watershed` — read raw run directories and need tier 2 below.
All of them additionally need a LaTeX toolchain and the thesis tree; see
[REPRODUCTION.md](REPRODUCTION.md).

## Tier 2 — raw run archive (TUM-gated)

The full search traces: every generation, every Pareto front, every rendered
image. **24.4 GB across 238,822 files in 29 campaign groups.**

> **Access:** <https://tumde-my.sharepoint.com/:f:/g/personal/ruben_kaiser_tum_de/IgCfNBn7fYP_SqlIuU3LsrRCAf57Jbzh218iMq34cV8KUfE?e=avyzcZ>
>
> Access is gated to members of the Technical University of Munich. If you are
> outside TUM and need the raw traces, request access from
> **Ruben.Kaiser@tum.de**.

Composition:

| Extension | Size | Files | What it is |
|---|---:|---:|---|
| `.png` | 14.8 GB | 115,747 | rendered Pareto-front and origin images |
| `.json` | 8.8 GB | 117,950 | per-run `context.json`, `pareto_*.json`, `stats.json`, `config.json` |
| `.parquet` | 818 MB | 4,727 | `trace.parquet`, `convergence.parquet`, `sut_calls.parquet` |
| other | ~12 MB | ~270 | logs, seed matrices, shell drivers |

The `.parquet` files are the quantitative record; the PNGs are renderings of
genotypes that the traces already encode. If you only need to re-derive
numbers, the parquet subset (818 MB) is sufficient.

Largest groups: `Exp-100` 8.1 GB, `HS-GEN-01` 5.0 GB, `Exp-104` 1.7 GB,
`HS-GEN-02` 1.3 GB, `HS-GEN-03` 1.2 GB, `Exp-105` 1.1 GB, `Exp-102` 1.0 GB.

To use the archive, place it (or a subset) at `runs/` in the repository root,
preserving the group directory names.

## Tier 3 — not redistributable

| What | Why | How to obtain |
|---|---|---|
| ImageNet source images | licence forbids redistribution | accept the licence for `ILSVRC/imagenet-1k` on the HuggingFace Hub; `src/data/imagenet.py` streams and caches it |
| Model weights (Qwen, LLaVA, ModernBERT, VQGAN) | upstream licences | downloaded from the HuggingFace Hub on first use |
| HS-01 stimulus images (`pool_frozen/assets/`, 27 MB) | derived from ImageNet | regenerate with `experiments/HS-01/tools/build_references.py` and the staging scripts |

## Human-subject data

HS-01 collected 49 sessions (35 completed) from a convenience sample. The
archived records in this repository have been stripped of device fingerprints —
user agent, platform, screen geometry and device pixel ratio — which together
identified most participants uniquely. Wall-clock timestamps are truncated to
the hour. What remains is the participant code, the responses, the derived
device class, and quality-control flags.

The transform is implemented in `experiments/HS-01/tools/anonymize_sessions.py`
and is verified lossless: `analysis/hs01/run_all.py` produces byte-identical
tables and figures before and after. Re-check at any time with:

```bash
python experiments/HS-01/tools/anonymize_sessions.py --check
```

Participants consented to use of their responses for academic research in a
master's thesis (`experiments/HS-01/app/config/consent.en.md`). Any use beyond
that scope should be discussed with **Ruben.Kaiser@tum.de** first.

### Consent hash drift

`consent.en.md` and `study-config.json` are both emitted by
`make_study_config.py`, and the config records `consent.text_sha256` so the
exact wording can be tied to each session. They no longer agree:
`sha256(consent.en.md)` is `879205bd…` while the config attests `a97cc9c0…`.

The reason is that `consent.en.md` was hand-edited after generation — a
reworded opening line and the researcher email filled in over a placeholder —
without the config being regenerated. **The edited file is nonetheless the
authoritative record of what participants saw:** it landed at 2026-06-25 09:58
UTC and the first session started at 14:00 UTC the same day, so every one of
the 49 sessions was served the edited text. The recorded `text_sha256` attests
the earlier generated wording that no participant ever saw.

So the file is correct and the hash is stale, not the other way round. Do not
"fix" this by regenerating: that would overwrite the served text with the
pre-edit version and destroy the record, and it would also change
`config_sha256`, breaking its match with all 49 records.
`pool_ref.pool_file_sha256` still matches `itempool.json`, so the stimulus side
is intact.

`experiments/HS-01/tests/test_make_study_config.py` used to run the generator
against the real repository paths, which silently overwrote the served consent
text whenever the suite ran. It now writes to a temporary directory, and
`test_shipped_files_untouched` fails if that regresses.
