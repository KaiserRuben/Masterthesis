# HS-01 data_raw — boundary-converged stimulus candidates

Extracted 2026-06-11 by `extract_boundary_samples.py` (this directory; run it
from the repo root); refreshed after the 2026-06-11 workstation sync (see
Update section below). Machine-readable per-run coverage:
`extract_summary.json`.

**Boundary criterion (hard):** `fitness_TgtBal` = |lp_A − lp_B| ≤ **1e-2**.
Subset ≤ **1e-3** flagged per row (`q_le_1e3`). All qualifying *unique
genotypes* per run are kept (not just best-3); duplicates from re-evaluation /
caching are collapsed (`n_occurrences_in_trace` preserves multiplicity, the
row kept is the one with the lowest TgtBal / earliest generation).

## Update — HS-GEN-01 + full Exp-101q/Exp-102 consolidation

`extract_boundary_samples.py` gained an **HS-GEN-01** run-group (promoted full
runs only; `modality=image_only`, `text_dim=0` → every qualifier is strictly
`image_only_drift`) and was re-run after the workstation delivered the complete
Exp-101q (46/46) and Exp-102 (27/27) syncs. New totals:

- **HS-GEN-01 (LLaVA, image_only): 511** unique `image_only_drift` items (66
  ≤ 1e-3) from the 6 promoted pairs → `llava/HS-GEN-01/boundary_individuals.parquet`.
  The first strict image-only-drift items in the pool (joint runs still yield 0).
- **Exp-101q (Qwen): 641** qualifiers (77 ≤ 1e-3) from 25/46 yielding runs →
  `qwen/Exp-101q/boundary_individuals.parquet`. Supersedes the "zero yield"
  reading below — that rested on 5 early-synced high-probe wall cells (a
  sync-order artifact; the 5 stale `_cache/Exp-101q__*.json` were purged before
  this re-run).
- **Exp-102 (LLaVA): 1363** (full 27 runs; was 191 from 10).
- Pool total **15,219** qualifying individuals across 2 SUTs (LLaVA 14,578 incl.
  511 image-only / Qwen 641). `pool_staging` rebuilt accordingly
  (`STAGING_REPORT.md`); both image_heavy strata are now filled from the strict
  image-only items and the weak-proxy file is retired.

The 06-11 sections below are retained as the historical record; their
"image-heavy = 0" / "Qwen arm = 0" / Exp-102 "10/27" statements are **superseded**
by this update. Analysis write-up: Obsidian diary
`2026-06-17-Exp101q-Exp102-HSGen01-Analyse`.

## Update 2026-06-11 (refresh)

Re-ran the extraction after the workstation sync delivered new Exp-102 /
Exp-101q runs. No stale `_cache` entries existed for either experiment
(incomplete runs are never cached), so the new runs were scanned fresh;
Exp-100 / Exp-101 / Exp-27 came from cache and their parquets are
**byte-identical** to the audited originals (sha256-verified before/after).

What changed:

- **NEW** `llava/Exp-102/boundary_individuals.parquet` — **191 rows (14
  ≤1e-3)** from 4 yielding runs of 10 complete (12 dirs synced of 27
  planned), plus per-run provenance copies (`stats.json`,
  `context.json.gz`) for the 4 yielding runs. Details in the Exp-102
  section (no longer "UNAVAILABLE").
- **Exp-101q**: 5 complete runs scanned, **all zero-yield** (per-run min
  TgtBal 0.445–3.41). Status changed from "unavailable" to **scanned, zero
  yield** — these are recorded scans, not missing data. Details in the
  Exp-101q section.
- `extract_summary.json` regenerated with per-run reports for every newly
  scanned run (incl. the zero-yield ones and the footer-less stragglers).

Corrections from the audit (also fixed in place below):

- The `smoke_boundary_pair` run is **Qwen/Qwen3.5-4B (torch, cone
  disabled)**, not LLaVA — its per-run `config.json` says
  `sut.model_id: Qwen/Qwen3.5-4B`, `sut.backend: torch`,
  `cone_filter.enabled: false`. It sits under `runs/Exp-100/` by directory
  layout only; it is now listed with the excluded Qwen runs.
- Exp-100 `seed_0121` **never existed** — numbering gap in the roster
  (dirs jump seed_0120 → seed_0122), not a sync loss.
- `seed_0119_1781075779` lacks `manifest.json` **in the source run dir
  too** — the provenance copy is faithful; nothing was dropped.
- The aggregate-join NULLs on Exp-100 are **148 rows across two runs**
  (seed_0119: 114, seed_0120: 34), not one run.

Noted, **not extracted** (optional stimuli source): the Exp-100 PDQ archives
contain ~9 flipped-variant + 9 min-variant entries across 6 runs with
|lp_A − lp_B| ≤ 1e-2. They are PDQ-stage artifacts outside this extraction's
trace-based TgtBal criterion; extract separately if the study needs them.

Discrepancy worth knowing: `configs/Exp-102/exp102_basin_generality.yaml`
specifies 150 gen × 30 pop, but every completed Exp-102 run reports
`generations: 100, pop_size: 50` in stats.json (5,000-row traces) — the
workstation evidently ran a modified tranche config.

## Layout

```
data_raw/
  llava/Exp-100/boundary_individuals_poc_boundary_pair.parquet   (12,173 rows)
  llava/Exp-100/poc_boundary_pair/<seed_dir>/        per-run provenance:
        config.json, manifest.json, evolutionary/{stats.json, context.json.gz}
  llava/Exp-101/boundary_individuals.parquet                     (   531 rows)
  llava/Exp-101/<run_dir>/{stats.json, context.json.gz}
  llava/Exp-102/boundary_individuals.parquet                     (   191 rows)
  llava/Exp-102/<run_dir>/{stats.json, context.json.gz}
  qwen/Exp-101q/                                     (NOT WRITTEN — 5 runs scanned, 0 qualifying)
  qwen/Exp-27/boundary_individuals.parquet                       (NOT WRITTEN — 0 qualifying)
  qwen/Exp-27/NONQUALIFYING_best3_reference.parquet              (    12 rows, qualifies=False)
  qwen/Exp-27/<run_dir>/{stats.json, context.json}
  configs/Exp-{100,101,101q,102,26,27}/...           experiment YAMLs, verbatim
  extract_summary.json                               per-run scan reports
  _cache/                                            resume cache of the
                                                     extraction script; derived
                                                     data, safe to delete
```

Provenance subfolders exist only for runs that yielded ≥1 qualifying
individual (plus all 4 scanned Qwen cone runs, for the exclusion record).

## Parquet schema (one row = one unique qualifying individual)

Identifiers: `experiment, sut, run_group, run_dir, run_rel, seed_idx,
generation, individual` (`generation/individual = -1` for rows salvaged from
pareto JSONs without a readable trace).
Objectives: `tgtbal` (exact value), `q_le_1e3`, `d_img_matrix`
(= fitness_MatrixDistance_fro), `d_text_embed` (= fitness_TextDist, cosine
distance in the SUT's own text-embedding space; NaN for image_only runs),
`p_class_a, p_class_b, logprobs, predicted_class, cache_hit`.
Text (byte-exact): `decoded_text` (rendered mutated prompt from trace),
`pareto_text` + `full_prompt` (verbatim from `pareto_*.json` when the
individual is in the final front). Verified: `decoded_text == pareto_text` on
all 6,368 matched rows, and `full_prompt == decoded_text + " Answer with
exactly one of these options: {class_a}, {class_b}."` holds exactly
(6,093/6,093 on Exp-100) — so the full prompt of any row is reconstructable.
Genotype: `genotype` (full int vector; image block = first `image_dim` genes,
text block = last `text_dim` genes — same convention as
`experiments/analysis/output/exp100_partial/_pass1_aggregate.py`).
Drift: `n_active_img_genes, n_active_text_genes, frac_active_img,
drift_class` (∈ {mixed, image_only_drift, text_only_drift, no_drift}).
Pareto/render: `in_final_pareto, pareto_idx, pareto_png` (repo-relative path
into the ORIGINAL run dir — PNGs are not copied here).
Cell/pair metadata: `class_a, class_b` (labels in prompt),
`anchor/target_class_concrete, anchor/target_label_in_prompt,
level_anchor/target, common_ancestor_level, seed_idx_in_class,
anchor_position, target_position, prompt_template, answer_format,
modality, sut_model_id, sut_backend, image_backend, cone_enabled,
cone_alpha_deg`. Exp-100 rows additionally carry `bucket_anchor,
bucket_target, cell_kind, cross_subkind, is_diagonal, is_forward` joined from
`experiments/analysis/output/exp100_poc_aggregate.parquet` (148 rows across
two runs not present in that aggregate — seed_0119: 114, seed_0120: 34 —
have NULLs there).

---

## LLaVA (OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov, openvino backend)

### Exp-100 `poc_boundary_pair` (joint modality, 200 gen × 30 pop, cone α=20°)

- Runs found: 122 seed dirs; **121 scanned with complete traces** (6,000 rows
  each); 1 incomplete: `seed_0122_1781090904` (no stats.json, trace without
  parquet footer — still running/syncing from the workstation).
- Qualifying: **12,173 unique individuals (≤1e-2) from 100/121 seeds**;
  **2,396 ≤1e-3 (73 seeds)**. 21 seeds yielded none.
- TgtBal distribution: min 1.43e-06, median 3.15e-03, p90 8.03e-03.
  Per-yielding-seed counts: median 96, max 799; 98/100 seeds have ≥3.
- Pair coverage: junco→{boa constrictor 2,379 / marimba 2,885 / ostrich 2,533
  / green iguana 2,501 / cello 1,841} + ostrich→junco (34, one seed). The PoC
  slice is junco-anchored by design.
- `smoke_boundary_pair`: 1 run (5 gen × 12 pop smoke test, shark/salamander),
  min TgtBal 5.80 → 0 qualifying; excluded by design. **Correction
  (2026-06-11 refresh): this run is Qwen/Qwen3.5-4B (torch, cone disabled),
  not LLaVA** — it lives under `runs/Exp-100/` by directory layout only; see
  the excluded-Qwen-runs list below.
- Each seed dir also contains a `pdq/` stage (archive/flips/trajectories) —
  not part of the TgtBal criterion, left untouched in runs/.

### Exp-101 `exp101_margin_predictor` (joint, 50 gen × 30 pop, cone α=20°)

- Runs found: **46/46 complete** (1,500 rows each). Nothing missing.
- Qualifying: **531 unique individuals from 17/46 runs**; **56 ≤1e-3
  (12 runs)**. Min TgtBal 2.48e-05, median 3.58e-03. 16/17 yielding runs
  have ≥3. 29 runs (the "stuck"/slow cells of the margin-predictor design)
  yielded none within the 50-gen budget — expected, not a data gap.
- Cell coverage of qualifying items: 16 (anchor, target, level) cells across
  8 directed class pairs, incl. non-junco anchors (cello→marimba 228,
  junco→ostrich 143, boa→iguana/marimba, ostrich→iguana, iguana→boa,
  marimba→boa, cello→ostrich/junco).

### Exp-102 `exp102_basin_generality` (joint, 100 gen × 50 pop per stats.json, cone α=20°)

*(refreshed 2026-06-11 after the workstation sync; previously "UNAVAILABLE".
Repo YAML says 150 gen × 30 pop — the executed runs report 100 × 50,
5,000-row traces.)*

- Runs synced: **12 dirs of 27 planned** (per `filter_indices`; 23 cells,
  4 at n=2); **10 complete**, 2 footer-less mid-sync stragglers:
  `seed_196_1781238499`, `seed_198_1781239172`. 15 planned runs not yet
  synced/run.
- Qualifying: **191 unique individuals (≤1e-2) from 4/10 complete runs**;
  **14 ≤1e-3 (2 runs)**. Median TgtBal 2.70e-03. All 4 yielding runs have ≥3.

| run | pair (concrete → concrete, labels in prompt) | ≤1e-2 | ≤1e-3 | min TgtBal |
|---|---|---|---|---|
| seed_106_1781214492 | ostrich→boa (ratite/constrictor) | 138 | 9 | 1.09e-04 |
| seed_26_1781179654 | junco→boa (sparrow/constrictor) | 26 | 5 | 8.94e-04 |
| seed_88_1781203280 | ostrich→green iguana (ratite/iguana) | 16 | 0 | 2.21e-03 |
| seed_80_1781202996 | ostrich→junco (ratite/sparrow) | 11 | 0 | 4.57e-03 |

- Zero-yield complete runs (min TgtBal): seed_0 0.048, seed_28 0.787,
  seed_29 0.883, seed_108 1.068, seed_109 0.735, seed_160 1.030.
- Anchor diversity: 3 of 4 yielding runs are **ostrich-anchored** — the
  first non-junco-anchored boundary items in the LLaVA pool with bucket-level
  prompt labels.
- Text/render availability: `decoded_text` on all 191 rows; 103/191 (53.9%)
  are in the final Pareto front and carry `pareto_text` / `full_prompt` /
  `pareto_png`. The `decoded_text == pareto_text` and full-prompt
  reconstruction identities hold on all 103 matched rows.
- Drift: 190 mixed, 1 text_only_drift, **0 image_only_drift** (strata table
  below). Min `d_text_embed` 0.226 — nothing under the 0.2 relaxed proxy.

---

## Qwen (Qwen/Qwen3.5-4B, torch/MPS) — **zero qualifying individuals**

### Exp-101q `exp101q_margin_predictor_qwen` — scanned, ZERO YIELD (was: unavailable)

Primary Qwen source, same 46-run roster as Exp-101 (`filter_indices`
byte-identical). After the 2026-06-11 sync: **6 dirs of 46 planned; 5
complete (full 1,500-row traces), all zero-yield**; `seed_28_1781242712`
footer-less mid-sync. 40 planned runs not yet synced/run. Config (copied)
confirms cone-enabled VQGAN backend (α=20°, target_m=10), joint modality,
50 gen × 30 pop.

| run | min TgtBal |
|---|---|
| seed_0_1781185212 | 0.445 |
| seed_1_1781191429 | 2.730 |
| seed_2_1781213809 | 2.848 |
| seed_6_1781223228 | 2.629 |
| seed_26_1781232826 | 3.408 |

These are **recorded scans with zero qualifying individuals, not missing
data**: the same seeds under LLaVA (Exp-101) yielded 0 / 9 / 35 / 99 / 23
items respectively; under Qwen four of the five never get closer than
2.6 nats. No provenance subfolders and no parquet were written (the script
only writes them for yielding runs).

### Exp-27 cone fallback (image_only, 100 gen × 20 pop, hammerhead shark vs spotted salamander)

Cone verification per run: run dirs carry no per-run config.json; cone status
is established by the run-name ↔ `configs/Exp-27/qwen_mps_pairA_cone*.yaml`
mapping (each YAML's sole delta vs baseline is the `cone_filter` block) and
recorded per row as `cone_alpha_deg`.

| run | cone α | trace | min TgtBal | qualifying |
|---|---|---|---|---|
| exp27_qwen_mps_pairA_cone05_seed_0_1779717753 | 5°  | ok (1,998 rows) | 2.521 | 0 |
| exp27_qwen_mps_pairA_cone10_seed_0_1779722277 | 10° | ok (2,000 rows) | 2.590 | 0 |
| exp27_qwen_mps_pairA_cone20_seed_0_1779726440 | 20° | footer missing → salvaged from 271 pareto JSONs | 2.661 | 0 |
| exp27_qwen_mps_pairA_cone40_seed_0_1779733211 | 40° | ok (2,000 rows) | 2.541 | 0 |

All four sit ~2.5 nats from the boundary (≈250× the 1e-2 threshold) — these
image_only alpha-sweep runs never approached convergence. The best 3 per run
are preserved in `NONQUALIFYING_best3_reference.parquet` (12 rows,
`qualifies=False`) purely to document the distance; **they are not stimuli**.

Excluded Qwen runs (non-cone image backend, per HS-01 requirement):
`exp27_qwen_mps_pairA_baseline_seed_0_*` (vqgan, cone disabled),
`exp27_qwen_mps_pairA_stylegan_seed_0_*` ×4 (stylegan backend),
`runs/Exp-26/exp26_qwen_mps_vqgan_baseline_seed_1_*` (cone disabled),
`runs/Exp-100/smoke_boundary_pair/seed_0000_*` (Qwen3.5-4B torch smoke test,
cone disabled — mis-filed under the Exp-100 LLaVA tree, see correction in
the Update section; 0 qualifying anyway, min TgtBal 5.80).
Never run (config exists, no run dir): `configs/Exp-26/qwen_mps_vqgan_cone.yaml`,
`configs/Exp-27/qwen_mps_pairB_cone{05,10,20,40}.yaml`.

**Bottom line: the Qwen arm of HS-01 still has no usable boundary items —
now confirmed by 5 completed Exp-101q runs (zero yield, minima 0.445–3.41)
rather than by absence of data; it depends on the remaining 41 Exp-101q
runs (40 unsynced + seed_28).**

---

## Modality strata & drift (study planning)

All 12,895 qualifying individuals come from **joint**-modality runs
(Exp-100 + Exp-101 + Exp-102, LLaVA). No qualifying items exist from
image_only runs (the only ones, Exp-27/Qwen, never converged) and no
text_only runs are in scope.

Drift breakdown (active genes per block; image block = first `image_dim`
genes, text = last 19):

| stratum | Exp-100 | Exp-101 | Exp-102 |
|---|---|---|---|
| mixed (img+text drift) | 12,149 | 522 | 190 |
| text_only_drift (0 image genes) | 24 (10 seeds) | 9 (4 runs) | 1 (1 run) |
| image_only_drift (0 text genes) | **0** | **0** | **0** |

The expected scarcity of image-heavy boundary items is confirmed and is
absolute at the strict definition: **not a single qualifying individual
reaches the boundary on image drift alone** — unchanged after the Exp-102
refresh. Relaxed proxy (semantically near-unchanged text):
`d_text_embed < 0.1` → 65 rows but all from a single seed; `< 0.2` → 800
rows from only 4 seeds (Exp-101: zero below 0.42; Exp-102: zero below
0.226, 7 rows < 0.3). Median active genes among qualifying: ~7 image /
~8 text (Exp-100); 5 image / 7 text (Exp-102, median d_img 0.0073, median
d_text 0.608); Exp-100 median d_img 0.0020, median d_text 0.524. An
image-heavy stratum will need either the remaining Exp-102/Exp-101q runs
(the first 4 yielding Exp-102 runs added none), or a relaxed definition,
or post-hoc re-rendering with text genes zeroed (which changes TgtBal and
would need re-scoring).

## Images & seed provenance (recorded, not copied)

- **Rendered PNGs:** only final-Pareto members have them, in the ORIGINAL run
  dirs (`pareto_png` column; e.g.
  `runs/Exp-100/poc_boundary_pair/<seed>/evolutionary/pareto_<i>.png`).
  Coverage among qualifying rows: 6,093/12,173 (50.1%) Exp-100,
  275/531 (51.8%) Exp-101, 103/191 (53.9%) Exp-102. `origin.png` per run
  dir = VQGAN round-trip of the
  seed image. Non-Pareto individuals have **no** stored render and must be
  re-rendered host-side from `genotype` + `context.json.gz`
  (VQGAN f8-16384; image gene g at patch k: 0 = keep
  `image_original_codes[k]`, else `image_candidates[k][g-1]`-style lookup at
  `image_patch_positions[k]` — see `src/manipulator/`).
- **ImageNet originals:** raw source file paths are NOT recorded in any run
  artifact. The seed image exists only as (a) `origin.png`, (b)
  `image_original_codes` in context.json (exact VQGAN codes), and (c) the
  ImageNet pools in config `cache_dirs`
  (`/mnt/storage/huggingface/imagenet` workstation,
  `~/.cache/imagenet` mac), selected by RosterSeedGenerator.

## Size

133.9 MB total after the 2026-06-11 refresh (was 130.2 MB), of which
32.2 MB is the deletable `_cache/` (net data ≈ 102 MB: llava 74.6 MB —
dominated by 121 gzipped per-run context snapshots needed for re-rendering,
+3.15 MB for the new Exp-102 subtree — qwen 26.9 MB, configs 172 KB,
parquets 0.79 MB total). Budget: ≤2 GB. No trace.parquet was copied
wholesale.

## Gaps (explicit)

1. **Exp-101q: 5/46 runs complete, all zero-yield** → Qwen still has zero
   boundary items (scanned, not missing); 40 runs unsynced + footer-less
   `seed_28_1781242712`.
2. **Exp-102: 10/27 runs complete — 4 yielding, 191 items**; footer-less
   stragglers `seed_196_1781238499`, `seed_198_1781239172` + 15 planned
   runs unsynced.
3. **Exp-100 seed_0122** incomplete (1 seed missing from the PoC slice).
   `seed_0121` never existed — numbering gap, not a sync loss.
4. **Image-heavy stratum still empty** (Exp-102's 191 new rows added none;
   see drift table above).
5. Exp-100 coverage is junco-anchored; anchor-class diversity at the boundary
   comes from Exp-101's 17 yielding runs plus Exp-102's 3 ostrich-anchored
   yielding runs.

Re-run `python experiments/HS-01/data_raw/extract_boundary_samples.py` from
the repo root after the remaining Exp-101q/Exp-102 runs sync; stale `_cache`
entries for the affected runs must be removed first (footer-less runs are
never cached, but if a previously-scanned run's trace *grows*, delete its
`_cache/<Exp>__<run_dir>.json`).
