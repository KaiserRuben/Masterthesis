# L2 Taxonomy Migration (Exp-100 prep)

**Date:** 2026-04-24
**Scope:** `src/data/imagenet_class_mapping.py` — stabilize the L2 super-category
layer so that every ImageNet-1k class has a clean `[L0, L1, L2]` 3-level path
and L2 never leaks into L0/L1.

## Canonical L2 set

11 closed super-categories, exported from `imagenet_class_mapping.py` as
`CANONICAL_L2` and re-imported by `tests/test_taxonomy_invariants.py`:

| L2 bucket     | count |
|---------------|------:|
| object        |   533 |
| mammal        |   218 |
| bird          |    59 |
| food          |    47 |
| arthropod     |    47 |
| reptile       |    36 |
| fish          |    16 |
| invertebrate  |    14 |
| plant         |    12 |
| nature        |    10 |
| amphibian     |     8 |
| **total**     |  1000 |

## Rationale

Design goal per Exp-100 (class-geometry): pair samples with `c=None`
(different L2) should correspond to genuinely cross-domain comparisons, not to
noise from the old "same super-category seen at two different levels" bug.

- We keep the **taxonomic classes split** (bird ≠ reptile ≠ mammal ≠ amphibian
  ≠ fish ≠ arthropod ≠ invertebrate) so that an animal-↔-animal comparison is
  expressed as `c=None`, as Exp-100 expects. We did *not* introduce a unified
  `animal` umbrella.
- `plant-like organism` (the 4 non-edible fungi) is merged into `plant`. Edible
  mushrooms (bolete, hen-of-the-woods) remain in `food` as in the source.
- Everything artifactual — vehicle, clothing, structure, building, electronic
  device, container, furniture, tool, sports equipment, textile, machine,
  weapon, watercraft, musical instrument, instrument — collapses into `object`
  (533 classes). This is the bucket that was most leaky in the source; nearly
  all leaks point into the same artifactual domain, so a single `object` L2 is
  the right granularity for Exp-100's cross-domain test. Fine-grained
  differentiation within artifacts is still available at L1 (207 distinct L1
  labels) and L0 (533 distinct L0 labels).
- The 3 `person` classes (baseball player, bridegroom, scuba diver) are kept
  under `object` with L1=`person`. Adding a dedicated `person` L2 for 3 classes
  would have trivialized that bucket.
- `beverage` (2 classes: red wine, espresso) is merged into `food` — they were
  the only sub-food leaks.

## Major merge decisions

Leaky L2 → canonical L2 (count = classes affected):

| Old L2                | → | New L2    | n  |
|-----------------------|---|-----------|---:|
| `dog`                 | → | mammal    | 73 |
| `structure`           | → | object    | 32 |
| `vehicle`             | → | object    | 26 |
| `clothing`            | → | object    | 22 |
| `ungulate`            | → | mammal    |  7 |
| `container`           | → | object    |  5 |
| `snake`               | → | reptile   |  4 |
| `musical instrument`  | → | object    |  4 |
| `textile`             | → | object    |  4 |
| `plant-like organism` | → | plant     |  4 |
| `sports equipment`    | → | object    |  3 |
| `watercraft`          | → | object    |  3 |
| `tool`                | → | object    |  3 |
| `furniture`           | → | object    |  3 |
| `machine`             | → | object    |  3 |
| `venomous snake`      | → | reptile   |  2 |
| `instrument`          | → | object    |  2 |
| `building`            | → | object    |  2 |
| `weapon`              | → | object    |  2 |
| `electronic device`   | → | object    |  2 |
| `beverage`            | → | food      |  2 |
| `arachnid`            | → | arthropod |  1 |

### `sea animal` disambiguation (20 classes)

The old L2 `sea animal` spanned multiple taxonomic kingdoms. Resolved by
inspecting L0/L1:

| resolved L2  | count | examples                                              |
|--------------|------:|-------------------------------------------------------|
| fish         |     8 | great white shark, tiger shark, stingray, clownfish   |
| arthropod    |     7 | Dungeness crab, American lobster, hermit crab         |
| invertebrate |     3 | conch, chambered nautilus, chiton                     |
| reptile      |     1 | sea snake                                             |
| bird         |     1 | king penguin                                          |

## Totals

- **Paths changed:** 351 / 1000
- **Paths extended (1 or 2 levels → 3):** 122
- **Paths unchanged:** 649
- **Bridging L1 labels introduced:** a handful (`cartilaginous fish`,
  `bony fish`, `perching bird`, `tropical bird`, `caudate`, `anuran`,
  `chelonian`, `squamate`, `serpent`, `psittacine`, `cuculiform`, `galliform`,
  `ratite`, `waterbird`, `proboscidean`, `person`, etc.). None invented at L2.
  All introductions are standard zoological or ornithological groupings.

## Depth distribution

| before (source) | after                 |
|-----------------|-----------------------|
| `{3: 878, 2: 116, 1: 6}` | `{3: 1000}` |

## Genuinely ambiguous placements (flagged, not hidden)

- **`eggnog`** — placed at `['alcoholic beverage', 'beverage', 'food']`.
  Arguable between `food` and `object` (as a container). Chose `food` to
  preserve the source `beverage`-lineage decision.
- **`hay`** — placed at `['animal feed', 'plant material', 'object']`.
  Arguable among `plant`, `food`, `object`. Kept at `object` (source
  decision). Note: it would fit `plant` if treated as plant product, but
  ImageNet's label refers to dried fodder.
- **`mushroom` cluster split** — edible mushrooms (bolete, hen-of-the-woods,
  `mushroom` itself) are under `food`; non-edible fungi (agaric, gyromitra,
  stinkhorn mushroom, earth star, coral fungus) are under `plant`. This is
  consistent with the source's partition but note that fungi are a separate
  kingdom. A future taxonomy could introduce a `fungus` L2 — for Exp-100 the
  current split suffices.
- **`king penguin`** — re-bucketed from source `sea animal` to `bird` on
  taxonomic grounds. A habitat-weighted classification would instead place it
  near sea mammals.
- **`sea snake`** — re-bucketed from source `sea animal` to `reptile`. Same
  rationale as penguin.
- **`honeycomb` / `spider web`** — bee/spider *products* rather than
  animals. Kept at `object` (the artifact, not the organism). No inconsistency
  risk since neither bee nor spider has a direct ImageNet class at L0.
- **`crayfish`** — source had `freshwater animal` as L1 (a unique stray
  label). Normalized to `['crayfish', 'crustacean', 'arthropod']` to match
  all other crustaceans (crabs, lobsters, isopod). Also manually corrected
  post-transform.

## Invariants guarded

See `tests/test_taxonomy_invariants.py` (8 tests, all passing):

- `len(path) == 3` for every class.
- `path[2] ∈ CANONICAL_L2`.
- `{path[2]} ∩ {path[1]} == ∅` across the full map.
- `{path[2]} ∩ {path[0]} == ∅` across the full map.
- No path has repeated labels.
- All 11 canonical L2 buckets have ≥1 member.
- Class count is exactly 1000.

## Public API impact

None. `src/data/taxonomy.py` signatures unchanged. `cluster_of`, `path_of`,
`siblings`, `common_ancestor_level`, `pair_bucket`, `members`,
`cluster_labels`, `cluster_sizes`, `pairs_within`, `coverage_report` all
continue to work as before — they just now return meaningful L2 data for every
class instead of `None` for 122 of them.

## Regression check

`pytest tests/` — 28 failed, 277 passed. Baseline was 30 failed, 267 passed
(the extra 8 passes are the new invariant tests). All 28 remaining failures
are pre-existing fake-object scaffolding issues in `test_evolutionary`,
`test_vlm_sut`, `test_objectives`, `test_pair_resolver`. Two baseline
failures in `test_config_override.py` flipped to passing after the change
(unrelated caching effect, not a real improvement). Zero new failures.

---

## Iteration 2: Object split + invertebrate merge

**Date:** 2026-04-24
**Scope:** Refine the L2 layer further for Exp-100 class-geometry comparability
and semantic diversity.

### Motivation

Iteration 1 left two structural problems for Exp-100:

1. **`object` mega-bucket.** 533 classes (53% of the dataset) shared L2=`object`.
   Taxonomic distance via `common_ancestor_level` collapsed at L2 for
   `sports car` ↔ `cello` ↔ `cathedral` ↔ `pretzel`, eliminating the
   monotonicity needed for semantic-distance proxies.
2. **Taxonomic incorrectness: arthropod as a peer of invertebrate.** All
   arthropods are invertebrates. Keeping them as separate L2s inflated the
   cross-L2 pair count with a pair ("spider" vs "snail") that is actually
   same-phylum-level.

### Changes

- **`object` removed from `CANONICAL_L2`**; split into 10 new L2 buckets
  (vehicle, clothing, electronic device, musical instrument, structure,
  container, sports equipment, furniture, kitchenware, tool).
- **`arthropod` merged into `invertebrate`.** All 47 former `L2=arthropod`
  classes now sit under `invertebrate` with `insect / arachnid / crustacean /
  myriapod / parasite / extinct arthropod` L1s. The 14 existing invertebrates
  keep their L1 labels (mollusk, sea animal, cnidarian, worm, etc.).
- **Promoted L1→L2 labels** for all former-object groups: any class that had
  e.g. `L1=vehicle` now has `L2=vehicle` with a new narrower L1 sub-group
  (car, truck, boat, ship, aircraft, rail vehicle, …).
- **Invariant: no L2 label leaks into L0/L1.** Promoting L1=`container` to
  L2=`container` required renaming the single `L0=container` occurrence
  (`chest`) to `L0=chest`; similar small fixes applied to `L0=vehicle`
  (snowmobile), `L0=tool` (hammer), `L0=furniture` (desk, entertainment
  center), `L0=sports equipment` (racket, ski), `L0=structure` (maze),
  `L0=clothing` (apron, bib, diaper, suit).

### Final canonical L2 (19 buckets)

| L2 bucket           | count |
|---------------------|------:|
| mammal              |   218 |
| tool                |   115 |
| clothing            |    80 |
| vehicle             |    75 |
| structure           |    75 |
| invertebrate        |    61 |
| bird                |    59 |
| food                |    47 |
| reptile             |    36 |
| kitchenware         |    33 |
| electronic device   |    32 |
| container           |    31 |
| furniture           |    31 |
| musical instrument  |    30 |
| sports equipment    |    30 |
| fish                |    16 |
| plant               |    13 |
| nature              |    10 |
| amphibian           |     8 |
| **total**           |  1000 |

Balance: the two largest buckets (mammal, tool) still dominate, but the
top-bucket share fell from 53% (object, 533) to 22% (mammal, 218). Seven
buckets now sit in the 30–80 range, which is what Exp-100 needs for
cross-L2 pair sampling.

### Old `L2=object` → new L2 breakdown

| New L2             |  n |
|--------------------|---:|
| tool               | 115 |
| clothing           |  80 |
| vehicle            |  75 |
| structure          |  75 |
| kitchenware        |  33 |
| electronic device  |  32 |
| container          |  31 |
| furniture          |  31 |
| musical instrument |  30 |
| sports equipment   |  30 |
| plant (hay)        |   1 |
| **total**          | 533 |

`hay` (old L1=`plant material`, L0=`animal feed`) was the one class that
didn't fit any artifact bucket — it's dried fodder, a plant product. Moved
to `plant`.

### Arthropod → invertebrate

All 47 former `L2=arthropod` classes now at `L2=invertebrate`. L1 breakdown
inside the merged bucket (61 classes):

| L1 (invertebrate)   |  n |
|---------------------|---:|
| insect              | 27 |
| crustacean          |  9 |
| arachnid            |  8 |
| sea animal          |  7 |
| mollusk             |  5 |
| worm                |  2 |
| extinct arthropod   |  1 |
| parasite            |  1 |
| myriapod            |  1 |

`sea animal` is preserved as a loose L1 on some non-arthropod invertebrates
(starfish, jellyfish, sea urchin, sea anemone, sea cucumber, brain coral,
chiton) since those classes don't neatly share a phylum. An Iteration 3
could tighten these into `echinoderm / cnidarian / mollusk`.

### Genuinely ambiguous placements (flagged, not hidden)

- **`hay`** — moved to `L2=plant` (dried plant matter). Could also argue for
  `food` (animal fodder) or a new `material` L2. Kept with the closest
  existing biological bucket.
- **Person classes** (3: `baseball player`, `bridegroom`, `scuba diver`).
  No artifact bucket fits a human. Placed by role-as-imagery:
  - `baseball player` → `sports equipment` / `sports accessory`
  - `scuba diver` → `sports equipment` / `water sports`
  - `bridegroom` → `clothing` / `apparel` (the image depicts formal wear)
  None of these placements is a clean taxonomic choice; they are judgment
  calls. A future `person` L2 with 3 members would trivialize that bucket.
- **`honeycomb`** — bee product, natural origin. Moved to `structure` /
  `natural_structure` (alongside `spider web`) because it is a built
  structure. Could equally sit under `nature`.
- **`spider web`** — same reasoning as honeycomb.
- **`bubble`** — transient fluid structure. Placed in `tool` / `hand_tool`
  bucket as the closest functional fit (ImageNet class refers to the object,
  not the physical phenomenon). Judgment call.
- **Camera accessory (`lens cap`) → tool / hardware.** Could also be
  `electronic device` (as a camera accessory) — chose `tool` because it's
  non-electronic and mechanical.
- **Clocks/watches** (`analog clock`, `digital clock`, `digital watch`,
  `wall clock`) — placed under `tool` / `measurement_instrument`, grouped
  with stopwatches/sundials/barometers. Could argue for a dedicated
  `timepiece` L2 but the count (4) is too low.
- **Decoration items** (`vase`, `maypole`, `jack-o'-lantern`) — placed under
  `furniture` / `decoration`. Judgment call; `furniture` is the least bad
  among existing artifact buckets for decorative household items.
- **`kitchenware` size (33 classes).** Modest but > 10, above the flag
  threshold. Kept as a first-class L2 since it corresponds to a clean
  semantic domain (cooking/eating artifacts) whose pair samples should not
  collapse into `tool`.

### Small-bucket watch list

`amphibian` (8), `nature` (10), `plant` (13), `fish` (16) are all < 20.
These are inherent properties of ImageNet-1k's class distribution, not
artifacts of the taxonomy. Exp-100 pair sampling from these buckets will be
cardinality-limited; reported for awareness, no structural fix attempted.

### Depth distribution

| before (Iteration 1) | after (Iteration 2) |
|----------------------|---------------------|
| `{3: 1000}`          | `{3: 1000}`         |

### Invariants guarded

See `tests/test_taxonomy_invariants.py` (10 tests, all passing):

- length-3 paths
- `path[2] ∈ CANONICAL_L2`
- L2 ∉ L0 and L2 ∉ L1
- no repeated labels within path
- all 19 canonical L2 buckets populated
- class count == 1000
- `len(CANONICAL_L2) == 19` (new in Iteration 2)
- every bucket ≥ 1 member (new in Iteration 2)

### Regression check

`pytest tests/` — **28 failed, 279 passed**. Baseline after Iteration 1 was
28 failed, 277 passed. The extra 2 passes are the new invariant tests
added in this iteration. Zero new failures. All pre-existing failures
continue to come from fake-object scaffolding in `test_evolutionary`,
`test_vlm_sut`, `test_objectives`.

### Public API impact

None. `src/data/taxonomy.py` unchanged. Downstream code that assumed
`L2=object` as a valid label (if any) will now see 10 more specific labels
instead — this is a deliberate breaking change for the analysis layer:
Exp-100 pair-sampling code should read the new `CANONICAL_L2` at import
time rather than hardcoding the 11-bucket list from Iteration 1.
