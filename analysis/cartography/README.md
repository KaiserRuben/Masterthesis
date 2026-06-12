# Boundary cartography store

Unified, projection-agnostic data layer for mapping decision-boundary geometry
from boundary-pair runs (Exp-100 and successors). Built by `build_store.py`,
consumed by visualization/exploration code.

## Why a store

The two pipelines see the boundary differently and neither alone supports all
geometry questions:

- **SMOO (evolutionary)** — a dense, optimizer-biased point cloud with a pair
  decision value g = p_A − p_B under the **2-option abstracted prompt**. Field
  estimates ("the balance tilts here"), no exact localization, no multi-class
  information (pair-only scoring by design).
- **PDQ** — anchor-centered random sampling (stage 1) and gene-by-gene paths
  (stage 2) under the **6-option concrete prompt**, with full 6-class logprob
  vectors. Exact boundary localization (hamming-1 sign changes), region maps,
  basin composition.

## Axis policy (important)

Manipulation strength (rank-sum) is a *chosen abstraction*, not ground truth.
The store therefore carries a broad per-point descriptor set and NO canonical
projection: combinatorial axes (n_active, hamming, rank sums raw/normalized),
semantic axes (MatrixDistance / pixel L2, TextDist / cosine), temporal axes
(generation, step), text-operator-group activity, full logprobs, and the
genotype itself. Choosing axes is an analysis-time decision; projection quality
should be measured (e.g. straddle-midpoint band width per candidate 2D view).

## Tables (parquet, zstd) — `experiments/analysis/output/cartography/<run>/`

### points.parquet — one row per scored evaluation, both pipelines

| group | columns |
|---|---|
| identity | run, seed_dir, seed_idx, source (`smoo` / `pdq_anchor` / `pdq_s1` / `pdq_s2`), row_ref (generation·individual or call_id), candidate_id |
| cell | anchor_class, target_class, level_anchor, level_target, common_ancestor_level, seed_idx_in_class, anchor_word, target_word |
| regime | prompt_regime: `pair2` (SMOO, abstracted labels) / `cat6` (PDQ, concrete 6-option menu). **Boundaries differ between regimes — never mix unflagged.** |
| time | generation (smoo) / step or call order (pdq) |
| combinatorial axes | n_active_img, n_active_txt, rank_sum_img, rank_sum_txt, rank_sum_img_norm, rank_sum_txt_norm (per-seed bounds-sum normalized), hamming_to_anchor (pdq only) |
| semantic axes | d_img_sem (smoo: fitness_MatrixDistance_fro; pdq: image_pixel_L2), d_txt_sem (smoo: fitness_TextDist; pdq: text_cosine_sum) — scales differ between pipelines, comparable within source only |
| text-op groups | txt_active_mlm (genes −19..−17), txt_active_frag (−16..−12), txt_active_charnoise (−11..−4), txt_active_saliency (−3..−1) |
| model output | logprobs (list; 2 entries for pair2, 6 for cat6), pred_label, top_gap (top1−top2), pair_margin (lp_anchorside − lp_targetside; evo pair words for pair2, concrete classes for cat6), g_pair (p_A − p_B pair-softmax) |
| genotype | genotype (list<int16>), image_dim |

Sampling-bias note per source: `smoo` is optimizer-steered (dense near Pareto
front), `pdq_s1` anchor-centered random at mixed densities, `pdq_s2`
path-constrained (shrink toward anchor under the run's flip-preserve criterion
— for Exp-100 archives that criterion was the buggy 6-cat argmax).

### straddle_pairs.parquet — one row per hamming-1 boundary crossing

Adjacent stage-2 evaluations (candidate_before → candidate_after, exactly one
gene changed) where a decision sign flips. These are *surveyed* boundary points
(localized to one gene edit), as opposed to interpolated contours.

Columns: identity + cell + boundary_kind (`pair_margin` = evo-pair lp sign flip;
`argmax` = 6-cat argmax change, with labels_before/after), gene_idx, gene_value
before/after, gene_modality (img/txt), txt_group (for text genes),
margin_before, margin_after (lp pair margin both sides), logprobs_before/after,
midpoint descriptor columns (mean of the two points' combinatorial axes),
hamming_to_anchor_after, flip_id, step.

### transects.parquet — one row per stage-2 step, ordered

The gene-by-gene walks from each flip toward its anchor: flip_id, step,
accepted, still_flipped (run's criterion), gene_idx, old/new value,
hamming_to_anchor, pair_margin, pred_label, logprobs. Supports region-sequence
and crossing-count (folding) analyses. Rejected steps are off-path probes.

## Join keys & identifiers (learned the hard way)

- **`flip_id` is unique only within a seed** — always key walks/straddles by
  `(seed_dir, flip_id)`.
- PDQ point rows join to straddles/transects via `points.row_ref` (string of
  `call_id`) ↔ `call_id_after` / `call_id`, scoped by `seed_dir`.
  `points.candidate_id` is −1 for `pdq_s2`/`pdq_anchor` rows — do not use it
  as a key there.
- `image_dim` is a per-seed-image property (VQGAN patch count varies by seed
  image; 222/253/276 observed) — image `gene_idx` is not comparable across
  seeds. Text genes are always the trailing 19.
- `d_img_sem` / `d_txt_sem` are **NaN for `pdq_s2` and `pdq_anchor` rows**
  (pixel L2 / text cosine were never computed for replayed states). Backfill
  TODO: VQGAN-decode replayed genotypes + embed rendered text. Until then the
  semantic projection — the crispness-benchmark winner — is only available
  for `smoo` and `pdq_s1` sources.
- Pooled cat6 statistics are dominated by `pdq_s2` (95 % of rows,
  path-constrained sampling) — stratify or filter by `source` first.
- Transects cover only the ~30-step stage-2 shells near each flip, never the
  full anchor→flip radial line; radial folding profiles are not measurable
  from this store.
- SUT evaluation is not exactly repeatable: anchor triplicates show nonzero
  pair-margin repeat variance in 44 % of seeds (q90 ≈ 0.38 lp). Margin
  differences below ~0.4 lp should not be over-interpreted; use
  hysteresis-debounced crossing counts.

## Known caveats inherited from Exp-100 data

- All seeds junco-anchored, forward direction (Exp-102 adds anchors).
- PDQ stage-1/2 were steered by the buggy 6-cat flip criterion → stage-2 paths
  minimize the wrong thing; straddle pairs remain valid *point* observations.
- 38.9 % of anchors sit on the target side under the cat6 prompt (prompt-shift
  is material near the boundary).
- image_dim is 222 or 276 per seed — text genes are always the LAST 19.
