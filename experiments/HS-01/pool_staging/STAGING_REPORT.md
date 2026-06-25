# HS-01 Pool Staging Report — 2026-06-24

Provisional thresholds — see script header. Counts are CANDIDATES, not selected items.

| phase/stratum | target | candidates | runs | <=1e-3 | png-ready |
|---|---|---|---|---|---|
| pair/baseline | 8 | 224 | 224 | — | — |
| pair/image_heavy | 14 | 2139 | 18 | 250 | 115 |
| pair/text_heavy | 14 | 9909 | 166 | 1854 | 5691 |
| pair/balanced | 14 | 2831 | 126 | 421 | 991 |
| image/raw | 6 | **0 — pending** | — | — | — |
| image/roundtrip | 6 | 224 | 224 | — | — |
| image/boundary_joint | 12 | 7727 | 206 | 1775 | 7727 |
| image/image_heavy | 6 | 2139 | 18 | 250 | 115 |
| text/clean | 6 | 224 | 224 | — | — |
| text/low_drift | 8 | 2201 | 40 | 560 | 679 |
| text/medium_drift | 8 | 6550 | 146 | 1070 | 3432 |
| text/high_drift | 8 | 6620 | 177 | 1216 | 3616 |
| checks/synthetic | 2 | **0 — pending** | — | — | — |

Source rows: 17510 qualifying individuals across 224 runs and 2 SUT(s) (llava=16869, qwen=641).
Strict image_only_drift (HS-GEN-01) feeding both image_heavy strata: 2139 rows.
origin.png resolved for 224/224 runs.
