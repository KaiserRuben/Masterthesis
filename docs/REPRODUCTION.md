# Reproduction guide

Every experiment the thesis cites, mapped onto the config that ran it, the run
directory it produced, and the script that turns that output into the figure or
number in the text.

Install first: [ENVIRONMENT.md](ENVIRONMENT.md). Raw run data:
[DATA.md](DATA.md).

## Picking the right runner

Three pipelines, three entry points. The config tells you which one it is —
check its top-level blocks:

| Config has | Pipeline | Run with |
|---|---|---|
| `optimizer:` | evolutionary | `python experiments/runners/run_boundary_test.py <config>` |
| `distances:` | PDQ | `python experiments/runners/run_pdq_test.py <config>` |
| `evolutionary:` + `anchor_selection:` + `pdq:` | boundary-pair | `python experiments/runners/run_boundary_pair_test.py <config>` |

Of the 299 configs: 270 evolutionary, 9 PDQ, 4 boundary-pair (the rest are seed
generators and other preprocessing input).

**The boundary-pair pipeline is easy to miss and matters most.** It runs the
evolutionary stage, promotes each Pareto member to a PDQ anchor, and minimises
against it — the canonical Boundary Value Analysis characterisation. Only
`Exp-100` and `Exp-102` use it, but Exp-100 alone is behind 7 of the 13 results
figures. Running its config through `run_boundary_test.py` silently gives you
the evolutionary stage only.

Output lands in `runs/<save_dir>/<config.name>_seed_<i>_<timestamp>/`.
Templates are in `configs/templates/`.

Useful flags on `run_boundary_test.py`:

| Flag | Effect |
|---|---|
| `--preflight` | times 20 SUT calls, prints a runtime projection, then continues anyway |
| `--plan-only` | resolves the resume filter, logs skip/run counts, exits before any model work |
| `--resume` | skips seeds that already have `stats.json` (roster-mode seeds only — see [runtime expectations](#runtime-expectations)) |
| `--seed N` | fixes the master RNG so two runs differ only in what you changed |
| `--device`, `--save-dir` | override the config |

## What it takes to rebuild a figure

A bare clone is enough for the HS-01 tables. It is **not** enough for the
results figures.

```bash
python -m analysis.hs01.run_all      # 12 tables -> analysis/outputs/hs01/, 12 PNG figures
```

The figures go to `ANALYSIS_ASSET_ROOT` if set, otherwise the author's Obsidian
vault if it exists, otherwise `analysis/outputs/assets/`. These are the
diary/analysis PNGs; the ten HS-01 figures typeset in the thesis come from a
thesis-side script that is not part of this repository.

The results figures need two things this repository does not contain. Each
emitter builds its `.tex` and then typesets it to measure the box, so it needs:

1. **A LaTeX toolchain** — `pdflatex` on `PATH`. Not a Python dependency, so
   `requirements.txt` will not warn you.
2. **The thesis tree**, pointed at by `THESIS_DIR`. The emitters `\input`
   `figures/results/results-style.tex`, which lives thesis-side. Without it:

   ```
   ! LaTeX Error: File `figures/results/results-style.tex' not found.
   ```

   The in-repo `THESIS_DIR` default is a writable destination, not a working
   one.

```bash
export THESIS_DIR="/path/to/Master Thesis v0.6.0"
python -m analysis.viz.thesis.pgf.predictor        # fig:res:predictor
python -m analysis.viz.thesis.pgf.dose             # fig:res:dose
python -m analysis.viz.thesis.pgf.walls_heatmap    # fig:res:walls-atlas
```

Those three re-emit byte-identically to the submitted figures.

**Eight of the thirteen results figures need no run data**, reading only the
aggregates tracked here: `dose`, `predictor`, `prior_map`, `region_map`,
`wall_collapse`, `wall_shape`, `walls_bylabel`, `walls_heatmap`.

**Five read raw run directories** and need the tier-2 archive from
[DATA.md](DATA.md) unpacked at `runs/`: `boundary_map`, `budget`,
`config_effects`, `init_coverage`, `watershed`.

Re-running a *campaign* — as opposed to re-deriving its figures — additionally
needs the SUT weights, the dataset cache, and hours to days of compute. See
[runtime expectations](#runtime-expectations).

## Experiment index

Machine: **M** = Apple Silicon / MPS, **A** = Intel Arc / OpenVINO.
Sizes are the run-archive footprint (tier 2 in [DATA.md](DATA.md)).
Configs marked † are boundary-pair; run them with `run_boundary_pair_test.py`.

| ID | M | Claim it supports | Config | Runs | Figure producer |
|---|---|---|---|---|---|
| Exp-01 | M | diversity objective inflates the archive ~7× | `configs/Archive/boundary_test_cadence.yaml` (cadence arm only) | **lost** | `notebooks/Exp-01-*.ipynb` |
| Exp-02 | M | PDQ mapping baseline; 8.9 % flip rate, 52.7 % reduction | `configs/Archive/pdq_overnight.yaml` | `Exp-02/` 175 MB | prose only |
| Exp-03 | M+A | 4.5× scale effect | `configs/Exp-03/pdq_v2_strategies.yaml` | `Exp-03-mac/` 27 MB, `Exp-03-workstation/` 127 MB | prose only |
| Exp-04 | M | 83.8 % reduction (n=1572); 740/740 collapse | `configs/Exp-04/pdq_v2_gap.yaml` | `Exp-04/` 373 MB | prose only |
| Exp-05 | M | FP16 numerical floor, 75 flat generations | `configs/Exp-05/phaseA/` | `Exp-05/` 43 MB | prose only |
| Exp-08 | M | falsified on tier-1 diagnostics | `configs/Archive/Exp-08/` | `Exp-08/` 37 MB | prose only |
| Exp-09 | M | n/(n+1) coverage identity | `configs/Archive/Exp-09/` | `Exp-09/` 273 MB | `pgf/init_coverage.py` |
| Exp-10 | M | sparse quadrant 0 % → 66–87 % | `configs/Exp-10/phase1_*_n16383.yaml` | `Exp-10/` 265 MB | `pgf/init_coverage.py` |
| Exp-12 | M | objective choice moves the floor 0.07 nats | `configs/Archive/Exp-12_legacy_fasttext_singleop/` | `Exp-12/` 183 MB | prose only |
| Exp-20 | M | v1 conservative composite baseline | `configs/Archive/Exp-20_v1_conservative/` | `Exp-20/` 66 MB | prose only |
| Exp-21 | M | floor 1.667 is a subword artefact | `configs/Exp-21/full_stack_junco_chickadee.yaml` | `Exp-21/` 79 MB | prose only |
| Exp-22 / 22b / 22c | M | floor 2.431 → 2.077 → 1.848 | `configs/Exp-22/` (10 files) | `Exp-22/` 588 MB | `pgf/init_coverage.py` (panel B) |
| Exp-22d | — | never ran (1 of 21 configs exists) | `configs/Exp-22d/` | none | — |
| Exp-23 | A | cross-runtime baseline on Arc A770 | `configs/Exp-23/` | `Exp-23/` 149 MB | prose only |
| Exp-24 | M | text 0.005 vs image 1.49 nats (8.46×) | `configs/Exp-24/` | `Exp-24/` 223 MB | `pgf/config_effects.py` (a) |
| Exp-25 | M | FPS lowers the floor 13.9 % | `configs/Exp-25/` | `Exp-25/` 125 MB | prose only |
| Exp-26 | A | only StyleGAN-XL crosses; 14.4× wall time | `configs/Exp-26/` | `Exp-26/` 410 MB | `pgf/config_effects.py` (b) |
| Exp-27 | M | cone worsens the floor at every α | `configs/Exp-27/` (pairA only) | `Exp-27/` 174 MB | `pgf/config_effects.py` (c) |
| Exp-100 | M | boundary cartography; most-cited campaign | `configs/Exp-100/poc_boundary_pair.yaml` † | `Exp-100/` 8.1 GB, 122 seeds | `pgf/{budget,boundary_map,wall_shape,walls_heatmap,walls_bylabel,watershed,region_map}.py` |
| Exp-101 | M | gen-0 margin predictor, ρ = −0.757 | `configs/Exp-101/exp101_margin_predictor.yaml` | `Exp-101/` 770 MB | `pgf/predictor.py` |
| Exp-101q | M | ρ = −0.696; hardness map does not transfer | `configs/Exp-101q/` | `Exp-101q/` 899 MB | prose only |
| Exp-102 | M | snake wall anchor-independent, 208–825× | `configs/Exp-102/exp102_basin_generality.yaml` † | `Exp-102/` 1.0 GB | prose only |
| Exp-103 | M | coordinate-grounding item, p 0.498/0.502 | `configs/Exp-103/exp103_run01.yaml` | `Exp-103/` 300 MB | prose only |
| Exp-104 | M+A | PMI calibration; ρ 0.756/0.923, dose r = 0.92 | `configs/Exp-104/exp104_phaseb_{llava,qwen}_{raw,pmi}.yaml` | `Exp-104/` 1.7 GB | `pgf/{prior_map,wall_collapse,dose}.py` |
| Exp-105 | M | sentence-slot pilot (appendix register only) | `configs/Exp-105/` (steps 2–6 never ran) | `Exp-105/` 1.1 GB | — |
| HS-GEN-01 | M | 78/78 gen-0 gate; 2,139 items ≤ 1e-2 | `configs/HS-GEN-01/` (18 promoted + screen) | `HS-GEN-01/` 5.0 GB, 1024 screen runs | prose only |
| HS-GEN-02/03 | M | referenced in source comments only | `configs/HS-GEN-0{2,3}/` | 1.3 GB / 1.2 GB | — |
| HS-01 | — | human oracle study, 49 sessions | `experiments/HS-01/` (not under `configs/`) | in-repo, not under `runs/` | `analysis/hs01/`, plus a thesis-side script |

Exp-06 and Exp-07 were planned and never started; the thesis says so
explicitly. Exp-11 exists in `configs/` but the thesis never cites it.

The thesis carries two registers this table mirrors: `tab:emp:campaigns` (the 17
campaigns behind the results chapter) and `tab:app:experiments` (all 22 IDs,
including the never-run ones).

## Figure index

Each emitter is `analysis/viz/thesis/pgf/<name>.py` and writes one standalone
`.tex`. Run as `python -m analysis.viz.thesis.pgf.<name>`.

| Figure | Label | Emitter | Reads | Needs raw runs |
|---|---|---|---|:-:|
| exp100-budget | `fig:res:budget` | `budget.py` | `exp100_partial/seed_summary.csv` + convergence traces | ● |
| exp-config-effects | `fig:res:config` | `config_effects.py` | Exp-24 / 26 / 27 convergence traces | ● |
| exp-init-coverage | `fig:res:init` | `init_coverage.py` | Exp-09 / 10 Pareto fronts, Exp-22 convergence | ● |
| boundary-map | `fig:res:boundary-map` | `boundary_map.py` | `cartography/exp100/points.parquet` | ● |
| exp100-attractor-watershed | `fig:res:watershed` | `watershed.py` | aggregate + `sut_calls.parquet` | ● |
| exp101-predictor | `fig:res:predictor` | `predictor.py` | `exp101/exp101_per_seed.csv` | |
| exp100-wall-shape | `fig:res:wall-shape` | `wall_shape.py` | `cartography/exp100/points.parquet` | |
| exp100-walls-heatmap | `fig:res:walls-atlas` | `walls_heatmap.py` | `exp100_poc_aggregate.parquet` | |
| exp100-walls-bylabel | `fig:res:walls-words` | `walls_bylabel.py` | `exp100_poc_aggregate.parquet` | |
| exp100-region-map | `fig:res:region-map` | `region_map.py` | `cartography/exp100/{points,straddle_pairs}.parquet` | |
| exp104-prior-map | `fig:res:prior-map` | `prior_map.py` | `exp104{,_llava}/exp104_pmi.csv` | |
| exp104-wall-collapse | `fig:res:wall-collapse` | `wall_collapse.py` | same | |
| exp104-dose | `fig:res:dose` | `dose.py` | `exp104/phaseb_{qwen,llava}.csv` | |

Every emitter also needs `pdflatex` and `THESIS_DIR`; see above.

Method-chapter graphics come from `tools/render_manipulation_gallery_2axis.py`
(`fig:method:gallery`) and `tools/render_manipulation_distance_3d.py`
(`fig:app:distance-3d`).

## Runtime expectations

Cost is dominated by SUT calls, and `--preflight` underestimates them badly: it
measures the SUT alone (~0.36 s/call), while a cone-active run costs roughly 6×
that (~2 s/call). Concretely, the HS-GEN-01 screen was a 1024-pair chain that
ran about 39 hours — not overnight.

`--resume` only detects roster-mode seeds, which need `seed_metadata`. On a
`gap_filter` config it silently re-runs the whole chain. Restart a stopped
chain by running the missing pair config directly.
