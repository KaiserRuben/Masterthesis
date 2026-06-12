# HS-01 Pool Staging Report — 2026-06-12

Provisional thresholds — see script header. Counts are CANDIDATES, not selected items.

| phase/stratum | target | candidates | runs | <=1e-3 | png-ready |
|---|---|---|---|---|---|
| pair/baseline | 8 | 121 | 121 | — | — |
| pair/image_heavy | 14 | 800 | 4 | 225 | 191 | (weak proxy only)
| pair/text_heavy | 14 | 8773 | 108 | 1600 | 5030 |
| pair/balanced | 14 | 2247 | 74 | 367 | 764 |
| image/raw | 6 | **0 — pending** | — | — | — |
| image/roundtrip | 6 | 121 | 121 | — | — |
| image/boundary_joint | 12 | 6471 | 121 | 1459 | 6471 |
| image/image_heavy | 6 | **0 — pending** | — | — | — |
| text/clean | 6 | 121 | 121 | — | — |
| text/low_drift | 8 | 1709 | 17 | 501 | 481 |
| text/medium_drift | 8 | 5486 | 94 | 947 | 2847 |
| text/high_drift | 8 | 5700 | 118 | 1018 | 3143 |
| checks/synthetic | 2 | **0 — pending** | — | — | — |

Source rows: 12895 qualifying LLaVA individuals, 121 runs. Qwen: pending (see README_QWEN_PENDING.md).
origin.png resolved for 121/121 runs.
