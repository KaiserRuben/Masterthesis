# Reproduction guide

This maps every experiment the thesis cites onto the configuration that ran it,
the run directory it produced, and the script that turns that output into the
figure or number in the text.

Start with [ENVIRONMENT.md](ENVIRONMENT.md) for the install and
[DATA.md](DATA.md) for where the run archive lives.

## What it takes to rebuild a figure

Read this before assuming a clone is enough — it is for the HS-01 tables, but
not for the results figures.

**The HS-01 tables need nothing but a clone:**

```bash
python -m analysis.hs01.run_all      # -> analysis/outputs/hs01/tab_*.csv
```

**The results figures need two things this repository does not contain.** Each
emitter builds its `.tex` and then typesets it to check the box dimensions, so
it needs:

1. **A LaTeX toolchain** (`pdflatex` on `PATH`). Not a Python dependency, so
   `requirements.txt` will not tell you it is missing.
2. **The thesis tree**, pointed at by `THESIS_DIR`. The emitters `\input`
   `figures/results/results-style.tex`, which lives thesis-side; without it the
   build stops with `File 'figures/results/results-style.tex' not found`. The
   in-repo default for `THESIS_DIR` is only a writable destination, not a
   working one.

```bash
export THESIS_DIR="/path/to/Master Thesis v0.6.0"
python -m analysis.viz.thesis.pgf.predictor        # fig:res:predictor
python -m analysis.viz.thesis.pgf.dose             # fig:res:dose
python -m analysis.viz.thesis.pgf.walls_heatmap    # fig:res:walls-atlas
```

Those three were verified to re-emit byte-identically to the submitted figures.

**Eight of the thirteen results figures need no run data**, reading only the
aggregates tracked here: `dose`, `predictor`, `prior_map`, `region_map`,
`wall_collapse`, `wall_shape`, `walls_bylabel`, `walls_heatmap`.

**Five do read raw run directories** and need the tier-2 archive from
[DATA.md](DATA.md) in place at `runs/`: `boundary_map`, `budget`,
`config_effects`, `init_coverage`, `watershed`.

Re-running a *campaign* — as opposed to re-deriving its figures — additionally
needs the SUT weights, the dataset cache, and hours to days of compute. See the
runtime notes at the end.

## Running an experiment

```bash
# Evolutionary (AGE-MOEA-II over discrete genotypes)
python experiments/runners/run_boundary_test.py configs/Exp-NN/<config>.yaml

# PDQ (AutoBVA-style two-stage directed search)
python experiments/runners/run_pdq_test.py configs/Exp-NN/<config>.yaml
```

Output lands in `runs/<config.name>_seed_<i>_<timestamp>/`. Templates live in
`configs/templates/`.

## Experiment index

Machine: **M** = Apple Silicon / MPS, **A** = Intel Arc / OpenVINO.
Sizes are the run-archive footprint (tier 2 in [DATA.md](DATA.md)).

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
| Exp-100 | M | boundary cartography; most-cited campaign | `configs/Exp-100/poc_boundary_pair.yaml` | `Exp-100/` 8.1 GB, 122 seeds | `pgf/{budget,boundary_map,wall_shape,walls_heatmap,walls_bylabel,watershed,region_map}.py` |
| Exp-101 | M | gen-0 margin predictor, ρ = −0.757 | `configs/Exp-101/exp101_margin_predictor.yaml` | `Exp-101/` 770 MB | `pgf/predictor.py` |
| Exp-101q | M | ρ = −0.696; hardness map does not transfer | `configs/Exp-101q/` | `Exp-101q/` 899 MB | prose only |
| Exp-102 | M | snake wall anchor-independent, 208–825× | `configs/Exp-102/exp102_basin_generality.yaml` | `Exp-102/` 1.0 GB | prose only |
| Exp-103 | M | coordinate-grounding item, p 0.498/0.502 | `configs/Exp-103/exp103_run01.yaml` | `Exp-103/` 300 MB | prose only |
| Exp-104 | M+A | PMI calibration; ρ 0.756/0.923, dose r = 0.92 | `configs/Exp-104/exp104_phaseb_{llava,qwen}_{raw,pmi}.yaml` | `Exp-104/` 1.7 GB | `pgf/{prior_map,wall_collapse,dose}.py` |
| Exp-105 | M | sentence-slot pilot (appendix register only) | `configs/Exp-105/` (steps 2–6 never ran) | `Exp-105/` 1.1 GB | — |
| HS-GEN-01 | M | 78/78 gen-0 gate; 2,139 items ≤ 1e-2 | `configs/HS-GEN-01/` (18 promoted + screen) | `HS-GEN-01/` 5.0 GB, 1024 screen runs | prose only |
| HS-GEN-02/03 | M | referenced in source comments only | `configs/HS-GEN-0{2,3}/` | 1.3 GB / 1.2 GB | — |
| HS-01 | — | human oracle study, 49 sessions | `experiments/HS-01/` (not under `configs/`) | in-repo, not under `runs/` | `analysis/hs01/`, plus a thesis-side script (see gaps) |

Exp-06 and Exp-07 were planned and never started; the thesis says so
explicitly. Exp-11 exists in `configs/` but the thesis never cites it.

The thesis carries two registers of its own that this table mirrors:
`tab:emp:campaigns` (the 17 campaigns behind the results chapter) and
`tab:app:experiments` (all 22 IDs including the never-run ones).

## Figure index

The results figures are emitted by `analysis/viz/thesis/pgf/<name>.py`, each
writing one standalone `.tex`:

| Figure | Label | Emitter | Reads |
|---|---|---|---|
| exp100-budget | `fig:res:budget` | `budget.py` | `exp100_partial/seed_summary.csv` + convergence traces |
| exp-config-effects | `fig:res:config` | `config_effects.py` | Exp-24 / 26 / 27 convergence traces |
| exp-init-coverage | `fig:res:init` | `init_coverage.py` | Exp-09 / 10 Pareto fronts, Exp-22 convergence |
| exp101-predictor | `fig:res:predictor` | `predictor.py` | `exp101/exp101_per_seed.csv` |
| boundary-map | `fig:res:boundary-map` | `boundary_map.py` | `cartography/exp100/points.parquet` |
| exp100-wall-shape | `fig:res:wall-shape` | `wall_shape.py` | `cartography/exp100/points.parquet` |
| exp100-walls-heatmap | `fig:res:walls-atlas` | `walls_heatmap.py` | `exp100_poc_aggregate.parquet` |
| exp100-walls-bylabel | `fig:res:walls-words` | `walls_bylabel.py` | `exp100_poc_aggregate.parquet` |
| exp100-attractor-watershed | `fig:res:watershed` | `watershed.py` | aggregate + `sut_calls.parquet` |
| exp100-region-map | `fig:res:region-map` | `region_map.py` | `cartography/exp100/{points,straddle_pairs}.parquet` |
| exp104-prior-map | `fig:res:prior-map` | `prior_map.py` | `exp104{,_llava}/exp104_pmi.csv` |
| exp104-wall-collapse | `fig:res:wall-collapse` | `wall_collapse.py` | same |
| exp104-dose | `fig:res:dose` | `dose.py` | `exp104/phaseb_{qwen,llava}.csv` |

Method-chapter graphics come from `tools/render_manipulation_gallery_2axis.py`
(`fig:method:gallery`) and `tools/render_manipulation_distance_3d.py`
(`fig:app:distance-3d`).

Figures that read raw traces rather than aggregates — `boundary_map.py`,
`budget.py`, `config_effects.py`, `init_coverage.py`, `watershed.py` — need the
corresponding tier-2 run directories in place. Every emitter additionally needs
a LaTeX toolchain and the thesis tree; see the top of this document.

## Known gaps

Recorded here rather than papered over.

**Exp-01's runs are lost.** The `notebooks/Exp-01-*.ipynb` read
`runs/Archive/Exp-01-SMOO-Pipeline-Validation/{02_4obj,03_cadence}`, and
`runs/Archive/` no longer exists. One of the three arms is still configured —
`configs/Archive/boundary_test_cadence.yaml` writes to that exact
`03_cadence` path — so the cadence arm could be re-run, but the 4-objective and
5-objective arms have no surviving config and the original outputs are gone.
The notebooks' stored cell outputs are the only remaining record.

Exp-01 is cited only in the appendix register, as the origin of the
initialization line of enquiry, and its conclusion is superseded by Exp-09 and
Exp-10, which are fully reproducible.

**Exp-02's config is unfiled.** It is `configs/Archive/pdq_overnight.yaml`, not
`configs/Exp-02/`. Confirmed by matching `name`, `model_id` and `n_categories`
against the `config.json` snapshot inside `runs/Exp-02/seed_0000_1775421159/`.

**Every legacy PDQ run carries its own `config.json`.** For `Exp-02`, `Exp-03`
and `Exp-04`, the run directories snapshot the full configuration — device,
categories, prompt template, SUT and all search blocks — so parameters are
recoverable even where the config file is missing or ambiguous.

**Two headline numbers have no in-repo producer.** The results chapter cites
`verify6/agg04.py` for the Exp-04 reduction median (n=1572) and `verify6/cache.py`
for the Exp-100 22.43 % cache-hit rate. No `verify6/` directory exists in this
repository; both numbers are currently unverifiable from the package.

**The HS-01 figure script lives in the thesis tree, not here.**
`figures/hs01/generate_figures.py` emits ten HS-01 figures and is not part of
this repository. It also carries a do-not-rerun warning: one of its functions
emits an abandoned variant that would overwrite a live figure, and several
emitted `.tex` files have since been hand-edited. The HS-01 *tables* rebuild
here via `analysis/hs01/run_all.py`.

**The 27 HS-01 interface screenshots are not reproducible.** They were captured
interactively from the study application on 2026-08-21, not by a checked-in
script. The application itself is in `experiments/HS-01/app/` and can be run to
re-capture them.

**`analysis/viz/thesis/pgf/style.py` does not exist** although
`figures/results/results-style.tex` in the thesis instructs the reader to edit
it. The shared styling actually lives in `pgf/emit.py`.

**Some configs never ran.** `configs/` contains arms that were written and not
executed — all six Exp-27 pairB configs, two Exp-24 arms, one Exp-25 arm, two
Exp-26 arms, Exp-105 steps 2–6, and two Exp-100 variants. Absence of a matching
run directory is the signal.

**Exp-100 holds 122 seed directories; the thesis reports 119 curated runs.**
Three were excluded during curation.

## Runtime expectations

Costs are dominated by SUT calls, and preflight underestimates them badly:
`--preflight` measures the SUT alone (~0.36 s/call), while a cone-active run
costs roughly 6× that (~2 s/call). Concretely, the HS-GEN-01 screen was a
1024-pair chain that ran about 39 hours, not overnight.

`--resume` only detects roster-mode seeds, which need `seed_metadata`. On a
`gap_filter` config it silently re-runs the whole chain; restart a stopped chain
by running the missing pair config directly.
