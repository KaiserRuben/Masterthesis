# HS-GEN-01 — image-only boundary-item generation for HS-01 (LLaVA arm)

Fills the two empty image-heavy strata of the HS-01 human validity study
(`experiments/HS-01/stage_pool_candidates.py`, design targets §3):

| stratum            | target | definition (strict)                                  |
|--------------------|--------|------------------------------------------------------|
| pair/image_heavy   | 14     | `fitness_TgtBal ≤ 1e-2` AND 0 active text genes      |
| image/image_heavy  | 6      | same — disjoint item draw at pool freeze             |

`modality: image_only` removes the text block structurally (text profile
forced to `noop`, 2-objective Pareto MatrixDistance + TgtBal), so every
qualifying individual is a strict image-only-drift item, prompt = clean
template. Across all 12,895 existing joint-run boundary items there is **not
one** `image_only_drift` qualifier — joint search always pays the boundary with
text, which is why this generation campaign exists.

Items must come from multiple anchors/pairs, not one seed — hence a roster-wide
screen plus per-pair full runs.

---

## Binding decisions (override the earlier draft)

The earlier draft optimised for raw yield (cone OFF, StyleGAN escape hatches).
Four study-owner decisions supersede it. Where a decision costs yield, the
trade-off is documented honestly below rather than hidden.

1. **Cone ON for all HS-GEN runs.** The VQGAN cone-filter is a core system
   component that keeps perturbations semantically alive. For human-eval
   stimuli, semantic validity outranks raw yield. *Honest trade-off:* on the
   one crossable pair the cone-OFF baseline floors and yields were strictly
   better (idx 1: floor 2.6e-6/2.5e-5 and 31–42 uniques OFF vs floor 2.3e-4
   and 19 uniques at cone α=20°). We knowingly accept roughly half the yield to
   guarantee the stimuli still read as the intended classes. Cone-α = **20°**
   (see "Cone-α choice").
2. **VQGAN only — no StyleGAN in study-facing generation.** The draft's
   StyleGAN pair B/C configs are archived under
   `configs/Archive/HS-GEN-01-draft-stylegan/` (repo archive convention: move
   superseded work in, never delete). *Honest trade-off:* StyleGAN was the only
   backend that ever crossed idx 83 (hammerhead→salamander); dropping it means
   that pair is re-tested under VQGAN+cone+heavy-mutation by the screen and is
   promoted only if it now crosses.
3. **Pairs must be human-distinguishable.** Participants are laypeople; if they
   cannot tell A from B, confusion measures their ignorance, not stimulus
   ambiguity. Fine-grained pairs may be rescued by label abstraction (presenting
   the choice at a coarser taxonomy level), where that level keeps the two
   labels distinct. Classification below; enforced by `promote_pairs.py` using
   `src/data/taxonomy.py`.
4. **Workflow = screen-then-promote, exploiting the gen-0 margin predictor.**
   Stage 1: a broad full-roster crossability screen (cone ON, **heavy
   mutation**, short budget). Stage 2: full-evolution configs only for pairs
   that pass BOTH the crossability screen AND the distinguishability criterion.
   The old cone-OFF floor table is **advisory only** — the screen re-measures
   crossability under the actual study-facing regime.

---

## Evidence base (every LLaVA image_only run that exists)

Still-valid trace-scan evidence from the draft. "floor" = min fitness_TgtBal;
"yield" = unique genotypes ≤ 1e-2. Exp-26 = 100 gen × 20 pop, Exp-24 = 300×30.

| pair (gap_filter idx)            | backend          | gen-0 min | floor   | yield |
|----------------------------------|------------------|-----------|---------|-------|
| gw shark → hammerhead (1)        | vqgan, no cone   | 0.065     | 2.6e-6  | 31    |
| gw shark → hammerhead (1)        | vqgan, no cone   | 0.008     | 2.5e-5  | 42    |
| gw shark → hammerhead (1)        | **vqgan, cone 20°** | 0.048  | **2.3e-4** | **19** |
| gw shark → hammerhead (1)        | stylegan         | 0.005     | 5.3e-4  | 40    |
| hammerhead → salamander (83)     | stylegan         | 0.644     | 6.6e-3  | 2     |
| hammerhead → salamander (83)     | vqgan, no cone   | 1.78      | 1.380   | 0     |
| hammerhead → salamander (83)     | vqgan, cone 20°  | —         | 1.381   | 0     |
| hammerhead → salamander (83)     | vqgan (Exp-24)   | 1.66      | 1.4907  | 0     |
| gw shark → stingray (2)          | stylegan         | 0.721     | 0.279   | 0     |
| gw shark → stingray (2)          | vqgan ± cone     | 1.10      | 1.000   | 0     |

Readings that still drive the design:

1. **Crossable pairs cross out of init.** Both no-cone idx-1 runs had a
   qualifying individual at gen 0–1. Crossability is a property of the
   (seed, pair, backend) geometry, not of search depth → the gen-0 margin
   predictor (image-space analogue of the validated text-space result).
2. **Gen-0 min TgtBal separates the regimes**: ≤ ~0.3 crossed early; 0.6–0.7
   marginal; ≥ 1.0 walled (300 gens bought 0.0007). This is the screen's
   promotion signal.
3. **Qualifying image-only genotypes are DENSE.** VQGAN idx 1: 9–179 active
   genes of 298 (medians 49/77.5/37). "Init distribution is the lever" applies
   inverted here — *density*, not sparsity, must be injected at init; and the
   default ~1-gene/offspring mutation is too weak to explore that dense band.
4. **Cone vs baseline (the now-overruled comparison):** on idx 1 baseline beat
   cone on floor and yield; on idx 83 cone changed nothing (1.381 vs 1.380).
   The draft concluded cone OFF. Decision 1 overrules this on semantic-validity
   grounds — recorded here for honesty, not as the operative choice.

### Cone-α choice — 20°, from Exp-26

The cone is pair- AND SUT-conditional (`finding_cone_axis_pair_conditional`),
so only **LLaVA** evidence transfers. The single LLaVA cone success is Exp-26
`llava_ov_vqgan_cone` on idx 1 at **α = 20°**, which crossed (floor 2.3e-4,
19 uniques). It is also the only LLaVA half-angle ever run. Exp-27's α sweep
(05/10/20/40) is on **Qwen** and showed cone monotonically *worsening* floors on
that SUT's walled pairs — it does not transfer to LLaVA and is not used to pick
α here. `target_m = 10` matches the Exp-26 LLaVA cone runs and bounds the
roster-wide modal-grid precompute cost.

---

## Pair eligibility (decision 3)

Classified by the smallest taxonomy level at which the two classes share a
cluster (`src/data/taxonomy.common_ancestor_level`):

| pair (idx) | taxonomy | lay-distinguishable? | verdict |
|---|---|---|---|
| great white shark → hammerhead shark (1) | both L0 = `shark` (same_L0) | **No** as-is | **NEEDS ABSTRACTION** |
| great white shark → stingray (2) | `shark` vs `ray`, share L1 `cartilaginous fish` (same_L1) | Yes | lay-distinguishable as-is |
| hammerhead shark → spotted salamander (83) | `fish` vs `amphibian` (cross) | Yes | lay-distinguishable as-is |

- **idx 1 (shark→hammerhead) — NEEDS ABSTRACTION, flagged.** Both classes are
  taxonomy L0 `shark`, so **symmetric coarsening collapses them** (both become
  "shark") — the abstraction machinery cannot rescue this pair into a 2-way
  choice. The only viable framing is the **manual feature contrast**
  `"hammerhead shark"` vs `"shark (other / non-hammerhead)"`, exploiting the
  hammerhead's lay-recognizable cephalofoil. idx 1 is NOT study-eligible and
  gets **no dedicated full run** — this is a human study, and generation
  compute goes only to study-eligible pairs. It participates in the screen
  like every roster entry (useful as the known-crossable sanity row in the
  screen report); its old calibration config is archived under
  `configs/Archive/HS-GEN-01-pairA-calibration/` and may be resurrected only
  by explicit study-owner decision on the abstraction framing (open question
  below).
- **idx 2 (shark→stingray) and idx 83 (shark→salamander)** are lay-eligible as
  fine labels; their L0 cluster labels (`shark`/`ray`, `shark`/`salamander`)
  also give a clean coarse framing if a uniform presentation level is wanted.
  Both are currently **VQGAN-walled** in the evidence table, so their study
  eligibility is gated on the screen re-crossing them under cone+heavy mutation.

`promote_pairs.py` applies this automatically: `common_ancestor_level == 0`
→ `needs_abstraction` (not auto-emitted, listed for manual framing);
`>= 1` or `None` → `distinguishable` (emit, propose L0 labels).

---

## Configs

| file | pair | backend | gens×pop | role |
|---|---|---|---|---|
| `hs_gen01_screen.yaml` | full roster (≤150 entries) | vqgan, **cone 20° ON**, heavy mutation | 4×30 | crossability discovery |
| `hs_gen01_promoted_idx<N>_*.yaml` | screen-promoted | vqgan, cone 20° ON | 150×30 | emitted by `promote_pairs.py` — the ONLY full runs |
| `promote_pairs.py` | — | — | — | screen → promote: gates + YAML emission |
| `validate_configs.py` | — | — | — | every YAML through the production loader |
| `run_hs_gen01_chain.sh` | — | — | — | screen / full-run chain |

---

## Knob table (non-default, with one-line justifications)

| knob | screen | full runs | justification |
|---|---|---|---|
| `modality` | `image_only` | `image_only` | strict stratum def: text_dim=0 ⇒ 0 active text genes by construction |
| `image.backend` | vqgan_codebook | vqgan_codebook | decision 2 — VQGAN only |
| `cone_filter.enabled` | true | true | decision 1 — semantic validity for human-eval stimuli |
| `cone_filter.alpha_deg` | 20.0 | 20.0 | Exp-26 LLaVA cone success setting (only LLaVA α with evidence) |
| `cone_filter.target_m` | 10 | 10 | matches Exp-26 LLaVA; bounds roster modal-grid precompute |
| `seeds` (gap_filter 50/3/3.5) | exact | exact | load-bearing: index→pair identity (1/2/83) only holds under this scan; gap ≤3.5 also starts the seed near the boundary |
| `zero_anchor_fraction` | 0.0 | 0.0 | min active genes among ALL qualifying image-only individuals is 9; exact-zero rows can never qualify |
| `tiers` (p_active 0.03/0.10/0.20/0.35/0.55) | yes | yes | E[n_active] ≈ 9/30/60/104/164 on 298 genes — centered on the observed qualifying band [9,179], med 49–77 |
| `mutation.prob` | **0.2** | **0.1** | screen: ≈60 mutated genes/offspring (max reach over 4 gens); full: ≈30/offspring (explore dense band for yield). Default ~1/298 ≈ 0.003 is far too weak |
| `mutation.eta` | **1.0** | **3.0** | screen: low eta = larger jumps, no front to protect; full: anneal to historical 3.0 so jumps don't overshoot the converged TgtBal≈0 ridge |
| `crossover.prob/eta` | 0.9 / **1.0** | 0.9 / 3.0 (default) | screen lowers SBX eta for broader offspring spread; full keeps the historical operator |
| `pop_size` | 30 | 30 | more init draws per tier per generation; corpus convention |
| `generations` | 4 | 150 | screen: gen-0 margin + 3-gen slope (predictor needs no depth); full: 2.25× Exp-26, buys unique-genotype yield (crossing by gen ≤14) |
| `early_stop.enable` | false | true | screen has a fixed budget; full runs bail on a walled pair |
| `early_stop.hypervolume_reference` | — | `[1.0, 10.0]` | with `null` the plateau trigger is silently dead (Exp-102 header bug); observed maxima d_img ≤0.25, TgtBal ≤3.5 ⇒ (1,10) dominates |
| `plateau_patience` / `no_improvement_warmup` | — | 50 / 40 | idx 1 converges in <15 gens; generous guard for pathological replicates |
| `parallel.workers` | 2 | 1 | screen: many short seeds → fanout pays (OV-GPU/CPU split); full: single seed |

### Heavy-mutation rationale (decision 4, new keys)

PM/SBX parameters are now YAML-tunable (`optimizer.mutation.{prob,eta}`,
`optimizer.crossover.{prob,eta}`; defaults reproduce the historical
PM(eta=3.0)/SBX(0.9, 3.0) exactly). On the 298-gene VQGAN genome the pymoo
default per-gene rate is `min(0.5, 1/n_var) ≈ 0.003` — about one mutated gene
per offspring, which barely moves through the **dense qualifying band** the init
seeds (finding #3). The screen and full runs therefore raise the per-gene
probability:

- **Screen — maximal reach.** `prob=0.2` (≈60 genes/offspring, inside the
  observed band) + `eta=1.0` (larger per-gene jumps). Over a 4-gen budget there
  is no converged front to protect, so we trade front quality for the best chance
  of surfacing a crossing under the aggressive perturbation regime.
- **Full runs — annealed.** `prob=0.1` (≈30 genes/offspring) keeps the search
  inside the dense band for *yield* (diverse qualifying uniques), but
  `eta=3.0` (annealed back to the historical default) keeps individual jumps
  moderate. A full run *does* build a converged near-boundary Pareto front;
  a screen-level low eta would keep overshooting the TgtBal≈0 ridge and degrade
  it. Elevated `prob` (exploration) + default `eta` (convergence) is the
  calibrated middle ground — justified by the dense-init finding and pymoo's
  PM semantics (lower eta = heavier-tailed step).

---

## Screen → promote workflow

```bash
# Stage 1 — broad crossability screen (detached):
HS_GEN_ONLY_SCREEN=1 bash configs/HS-GEN-01/run_hs_gen01_chain.sh
tail -f runs/HS-GEN-01/_logs/hs_gen01_chain.log

# --- between stages, offline ---
conda run -n uni python configs/HS-GEN-01/promote_pairs.py   # classify + emit
conda run -n uni python configs/HS-GEN-01/validate_configs.py
# append emitted YAMLs to PROMOTED=() in run_hs_gen01_chain.sh

# Stage 2 — promoted pairs only:
bash configs/HS-GEN-01/run_hs_gen01_chain.sh
```

`promote_pairs.py` reads `runs/HS-GEN-01/hs_gen01_screen_seed_*`, computes
per-pair `gen0_min` / `best` / 4-gen `rel_slope`, applies both gates, and:

- **emits** `hs_gen01_promoted_idx<N>_<a>_<b>.yaml` (inline standard template) for pairs
  that are crossable (`gen0_min ≤ 0.3`, or marginal `≤ 1.0` with a ≥20% slope)
  **and** lay-distinguishable (`common_ancestor_level ≥ 1`);
- **lists** crossable-but-fine-sibling pairs under "needs manual abstraction"
  (not auto-emitted);
- **skips** walled pairs;
- self-validates every emitted YAML through `load_config` + `apply_modality`.

Prefer distinct anchor *classes* across promotions (the roster spans the first
50 ImageNet classes — fish/birds/reptiles/amphibians); cap at ~4–6 pairs.

---

## Risks / open study-design questions

- **Abstraction presentation (the central open question).** idx 1 cannot be
  symmetrically coarsened (both classes are L0 `shark`). The proposed manual
  framing `"hammerhead shark"` vs `"shark (other)"` needs a study-design
  decision: does presenting an open-ended "other" option bias responses, and is
  the cephalofoil contrast strong enough that the choice measures stimulus
  ambiguity rather than ignorance? For genuinely distinguishable promoted pairs,
  decide whether to present fine labels or a uniform L0 framing.
- **The screen finds no lay-distinguishable crossable pair.** Then `promote_
  pairs.py` emits nothing. Default response: re-screen with a wider roster /
  heavier mutation before declaring the strata image-walled. Crossable
  fine-sibling pairs (e.g. idx 1) enter the study only by explicit study-owner
  decision on an abstraction framing; otherwise the image-heavy cell is
  dropped per HS-01 spec §9/§10.
- **Cone reduces yield (~½).** Accepted per decision 1. The promoted full runs'
  150×30 budget (vs Exp-26's 100×20 cone run that yielded 19) buys back unique
  count.
- **idx 2 / idx 83 stay walled under VQGAN+cone.** Prior VQGAN evidence has them
  at floor ~1.0 / ~1.4; StyleGAN (the only backend that crossed 83) is now out.
  The screen decides; if both wall, the strata are shark-family + idx-1 only.
- **Extraction integration.** `experiments/HS-01/data_raw/
  extract_boundary_samples.py` scans Exp-100/101/102 only — it needs a
  run-group entry for `runs/HS-GEN-01` before these items flow into
  `pool_staging` (image_only rows: `d_text_embed = NaN`,
  `drift_class = image_only_drift`).
- **Stimulus rendering.** Only final-Pareto members get PNGs; non-Pareto
  qualifiers need host-side re-render from `genotype` + `context.json`.

---

## Running (workstation)

```bash
# single config:
python experiments/runners/run_boundary_test.py configs/HS-GEN-01/hs_gen01_screen.yaml
```

Rough cost (workstation OV-GPU, 1.5–3 s/call): screen ≤18k calls + one-time
cone modal-grid precompute ≈ overnight at workers 2; each promoted full run
4.5k calls ≈ 2–4 h per replicate.
