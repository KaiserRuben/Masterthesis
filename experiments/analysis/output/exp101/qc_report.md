# Exp-101 data-quality audit (gen-0 margin predictor)

Audited: `runs/Exp-101/` (46 dirs), `configs/Exp-101/exp101_margin_predictor.yaml`,
`src/evolutionary/vlm_boundary_tester.py`, `src/common/seed_context.py`,
`src/manipulator/image/{manipulator,selection,cone_candidates}.py`.
Scripts: `experiments/analysis/output/exp101/scripts/qc_exp101.py`,
`qc_psum_detail.py`, `qc_final_aggregates.py`.
Artifacts: `qc_per_run.csv`, `qc_cell_design.csv`, `qc_stdout.txt`,
`qc_psum_detail.txt`, `qc_final_aggregates.txt`.

## Verdict: PASS — dataset is fit for the pre-registered analysis. No blockers.

| # | Check | Result | Severity |
|---|-------|--------|----------|
| 1a | 46 run dirs present | 46/46; seed_idx set == config `filter_indices` exactly; dir-name index == `stats.json:seed_idx` in all runs | OK |
| 1b | Files per dir | trace.parquet, stats.json, convergence.parquet, context.json present in all 46 | OK |
| 1c | trace shape | 1500 rows = gens 0..49 x exactly 30 individuals, every run; 0 duplicate (generation, individual) keys | OK |
| 1d | convergence.parquet | 50 rows in every run | OK |
| 2a | Budget | generations=50, pop_size=30 in all 46 stats.json; every trace reaches gen 49 → early stop never fired | OK |
| 2b | Runtime | 3112–3237 s (median 3167, std 32); zero IQR outliers; extremely uniform (uniform-length design held) | OK |
| 3 | **Backend** | **VQGAN cone_filter actually ran in all 46 runs.** See below. | material (bookkeeping mislabel only) |
| 4 | Cell design | 40 distinct cells; 6 cells n=2 matching the pre-registered set exactly; strata = 18 within-bucket / 12 wall_repl / 10 cross_breadth; all 6 classes appear as anchor; 31/40 cells non-junco-anchored; within-bucket grid (3 pairs x both dirs x {(0,0),(0,1),(1,1)}) exact; stratum-2/3 directed-pair sets exact | OK |
| 5a | NaNs | 0 NaNs in fitness_TgtBal / p_class_a / p_class_b across all 69 000 rows | OK |
| 5b | p_a + p_b | max excess over 1.0 = 8.94e-08 (float32 eps); 0 rows exceed 1+1e-7. Probs are PAIR-RENORMALIZED (`score_full_categories: false`): p_a+p_b == 1, so `crossed` (p_b > p_a) ≡ p_b > 0.5 | OK (note) |
| 5c | TgtBal consistency | 100 sampled rows/run (4600 total): max \|fitness_TgtBal − \|logprobs[0]−logprobs[1]\|\| = 9.5e-07 | OK |
| 5d | Cache | trace `cache_hit` rate per run: 0.5 %–25.7 % (median 4.6 %). stats.json `cache_hits/misses` are PROCESS-CUMULATIVE, not per-run (hits+misses = 1500·k, k=1..23, each value twice → 2 workers x 23 sequential seeds). Use the trace column, never stats.json, for per-run cache rates | minor |

## 3. Backend question (the material item)

**The cone backend ran. The "KNN" in stats.json is a config-default echo, not the runtime path.**

Evidence:

1. `context.json` of **all 46 runs**: `image_backend = vqgan_codebook`,
   `image_candidate_strategy = cone_filter`, `image_target_class` set and equal to
   `seed_metadata.target_class_concrete` in every run. This field is written from the
   live `ManipulationContext` (`src/common/seed_context.py:85/99`), which is set to
   `"cone_filter"` only on the cone path (`src/manipulator/image/manipulator.py:407`).
2. `stats.json:image_candidate_strategy` is written as
   `config.image.candidate_strategy.name` (`src/evolutionary/vlm_boundary_tester.py:213`)
   — the dataclass default `CandidateStrategy.KNN`, which cone mode never overrides.
   Same for `image_n_candidates: 16383` (config echo). All 46 stats.json say "KNN";
   it is meaningless as a runtime record.
3. `gene_bounds` structure: per-image-gene bounds are cone-survivor-set sizes
   (median ≈ 10, typical range 3–93). Per run, only 8–47 positions (median 19)
   have bounds ≥ 1000 (max observed 15730); **no bound equals 16383 in any run**,
   ruling out the unrestricted-codebook KNN genome. The large bounds are genuine
   wide-cone survivor sets (cone geometry retains most codes for some
   origin/target code pairs); `cone_candidates.py` has no full-codebook fallback.
4. Cross-check: per-position candidate-list lengths in `context.json` + 1
   (the "keep origin" slot) equal the image `gene_bounds` exactly in all 46 runs.

**Comparability consequence:** Exp-100 (cone, workstation) vs Exp-101 comparability
is NOT affected — both ran the same cone-alpha=20deg VQGAN backend. The only action
item is bookkeeping: do not cite stats.json's `image_candidate_strategy` anywhere.

### image_dim heterogeneity (expected, handled)

`text_dim = 19` in all 46 runs; text genes are always `genotype[-19:]` with the
identical bounds pattern `26,26,26,(2 x13),3,3,3` (same prompt template everywhere).
`image_dim` varies with the anchor seed image (patch selection is image-dependent):

| image_dim | n runs | seed_idx |
|---|---|---|
| 222 | 14 | 0,2,6,26,28,30,44,50,56,196,198,202,204,212 |
| 223 | 1 | 243 |
| 228 | 8 | 320,322,324,338,354,392,394,398 |
| 229 | 8 | 240,242,244,276,278,282,302,310 |
| 236 | 1 | 393 |
| 253 | 5 | 80,82,86,88,96 |
| 276 | 2 | 1,29 |
| 293 | 5 | 454,462,472,474,478 |
| 338 | 2 | 203,205 |

(222/276 junco groups match the known Exp-100 222-vs-276 heterogeneity.)
`len(genotype) == image_dim + text_dim == len(gene_bounds)` holds in every run.

## Issues list

| Severity | Issue | Detail |
|---|---|---|
| material | stats.json backend fields misleading | `image_candidate_strategy: "KNN"` / `image_n_candidates: 16383` in all 46 stats.json are config-default echoes; runtime truth (`cone_filter`) is in context.json. Affects interpretation only if someone reads stats.json naively; the runs themselves are cone runs, so Exp-100 comparability holds. |
| minor | stats.json cache counters process-cumulative | `cache_hits + cache_misses` = 1500·k per worker process (k = run order 1..23, 2 workers); not per-run. Per-run truth: trace `cache_hit` column (median hit rate 4.6 %). |
| minor (note) | probs are pair-renormalized | p_a + p_b == 1 (float32). Fine for crossed/TgtBal definitions; just don't interpret p_class_a as full-6-class posterior mass. |

## Descriptive snapshot (for cross-agent number agreement, not analysis)

- probe (median gen-0 fitness_TgtBal): min 0.234, median 2.177, max 16.005
- crossed_at_50: 18/46 runs; stuck (min TgtBal@50 > 0.1): 28/46 runs
- min fitness_TgtBal anywhere: 2.48e-05; no negative TgtBal rows
- n_pareto: 16–184 (median 78)
