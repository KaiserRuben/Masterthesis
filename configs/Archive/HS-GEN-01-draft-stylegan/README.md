# Archived — HS-GEN-01 draft StyleGAN pair configs

Superseded on the four binding study-owner decisions for HS-GEN-01 (see
`configs/HS-GEN-01/README.md`, "Binding decisions"). Kept per the repo archive
convention (move superseded work in, never delete).

| file | pair | why archived |
|---|---|---|
| `hs_gen01_pairB_hammerhead_salamander_stylegan.yaml` | hammerhead shark → spotted salamander (gap_filter idx 83) | StyleGAN dropped from study-facing generation (decision 2: VQGAN only). This pair is VQGAN-walled in prior evidence; its crossability under the new cone+heavy-mutation regime is now re-tested by `hs_gen01_screen.yaml`, and it is promoted only if it passes. |
| `hs_gen01_pairC_shark_stingray_stylegan.yaml` | great white shark → stingray (gap_filter idx 2) | Same — StyleGAN dropped. Lay-distinguishable (shark vs ray) but never crossed; re-tested by the screen. |

These remain useful as a record of the StyleGAN w-space init tiers and the
budget/patience calibration that crossed idx 83 once (Exp-26). If StyleGAN is
ever reinstated for a non-study-facing diagnostic, start here.
