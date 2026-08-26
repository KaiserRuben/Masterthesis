# HS-01 Frozen Pool — 2026-06-24T09:21:56Z

Schema validation: **VALID against hs01.itempool.schema.json (Draft 2020-12)**
Sources: 93 · Items: 93 · Assets: 119 PNGs

## Per-stratum selection (target vs selected; closeness-first)

| phase/stratum | target | selected | tgtbal min/med | d_img med | text-genes med | pairs | SUTs | note |
|---|---|---|---|---|---|---|---|---|
| text/clean | 6 | 6 | — | — | — | — | — | clean seed prompt; distinct seeds |
| text/low_drift | 8 | 7 ⚠ | 3.8e-05/2.9e-04 | 0.009 | 6 | 6 | qwen | round-robin photo-disjoint (7/8) |
| text/medium_drift | 8 | 8 | 3.8e-05/5.9e-04 | 0.009 | 8 | 8 | llava | round-robin photo-disjoint (8/8) |
| text/high_drift | 8 | 8 | 4.9e-05/4.0e-04 | 0.012 | 8 | 6 | llava | round-robin photo-disjoint (8/8) |
| image/raw | 6 | 6 | — | — | — | — | — | 6 class-representative originals (NOT exact seed twins) |
| image/roundtrip | 6 | 6 | — | — | — | — | — | round-trip origin.png; distinct seeds |
| image/boundary_joint | 12 | 10 ⚠ | 1.0e-05/6.1e-04 | 0.020 | 10 | 9 | llava,qwen | round-robin photo-disjoint (10/12) |
| image/image_heavy | 6 | 6 | 7.3e-06/6.2e-05 | 0.017 | 0 | 6 | llava | 6 distinct seeds (design 6; full seed-disjoint) |
| pair/baseline | 8 | 8 | — | — | — | — | — | round-trip image + clean prompt; distinct seeds |
| pair/image_heavy | 14 | 8 ⚠ | 4.4e-06/8.6e-05 | 0.023 | 0 | 8 | llava | 8 distinct seeds (design 14 — DATA-CAPPED at distinct promoted anchor images; full seed-disjoint, 1 phase per seed) |
| pair/text_heavy | 14 | 10 ⚠ | 1.9e-05/1.0e-03 | 0.006 | 8 | 6 | llava | round-robin photo-disjoint (10/14) |
| pair/balanced | 14 | 8 ⚠ | 1.4e-06/5.4e-04 | 0.016 | 8 | 5 | llava,qwen | round-robin photo-disjoint (8/14) |

## Attention checks
- `txt-attn-nonsense-01` (scale_leq 2) · `pair-attn-obvious-01` (choice_equals anchor)

## Folder structure
```
pool_frozen/
  itempool.json
  assets/images/<source_id>.png
  POOL_README.md
```

## Selection policy & decisions
- **FULL SEED-DISJOINT (no repeated-exposure confound, even across phases)**: every item comes from a distinct anchor PHOTO (sha of origin.png) used in EXACTLY ONE item/phase — run-dir is NOT unique (gap_filter/seeds_per_class=2 reuse one photo across many runs) — a rater can never see the same anchor image twice, in any phase. `x_seed_key` is carried on every source as a belt-and-suspenders key for the form builder.
- **Closeness first**: primary sort = `tgtbal` asc; quality floors `tgtbal<=1e-3`, `d_img<=0.05`, `text_genes<=7` (relaxed per-stratum only if needed — see notes).
- **Diversity**: round-robin across (SUT × label-pair), one best item per seed.
- **image_heavy is DATA-CAPPED** (design target 14+6=20): there are only as many distinct image-only anchor images as promoted HS-GEN-01 runs. Under full seed-disjointness each anchor serves ONE phase (split: image phase up to 6, pair phase the rest). Expanded by generating more promoted pairs on the workstation (lean batch idx 332/727/586/839/630/58).
- **controls** (clean/roundtrip/baseline): now use 20 DISTINCT seeds (no shared source across phases). image = `origin.png` (round-trip, native res — not 256×256). *Refinement: render 256 round-trips for resolution parity.*
- **image/raw**: class-representative ImageNet originals (exact seed file never logged). *Refinement: host-side exact-seed recovery, or drop the codec-cost micro-control.*
- **SUT balance (trade-off of closeness-first)**: LLaVA has the tighter boundaries, so the pool is LLaVA-heavy; Qwen survives mainly in low_drift / boundary_joint / balanced. Not model-balanced (HS-01 §4). **Lever**: enforce a per-stratum Qwen minimum at a small closeness cost.
