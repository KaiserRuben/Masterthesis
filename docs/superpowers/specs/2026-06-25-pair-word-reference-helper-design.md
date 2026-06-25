# Pair-option word reference helper — design

**Experiment:** HS-01 (Human Validity Study)
**Component:** `experiments/HS-01/app` (Next.js 14 study runner)
**Date:** 2026-06-25
**Status:** implemented (2026-06-25)

## Implementation notes (deltas from the original design)

Two small, well-justified refinements emerged during the build:

1. **`image` is nullable; one gloss-only entry.** 20 of the 21 words have an
   example photo. "cock" has no cache folder and the only local rooster image is
   a VQGAN-pipeline `01_original` (a study seed) — off-limits. Since "cock" is a
   *word*-confusion case (everyone knows a rooster), it is a **gloss-only** entry
   (`image: null`, "an adult male chicken; a rooster"). The schema/type allow
   `image: string | null`; the popover degrades to gloss-only (also the image-404
   fallback path).
2. **Schema bump is additive, not a hard cut.** `schema_version` became
   `enum: ["1.0.0", "1.1.0"]` (not `const: "1.1.0"`), so existing 1.0.0 records
   stay valid; the writer now emits `1.1.0`.

## Problem

In the pair phase (the final judgment phase), each trial offers the rater two
class words as options — e.g. *box turtle* or *tench*. Many are fine-grained
ImageNet labels a layperson will not recognise (*tench*, *axolotl*, *indigo
bunting*, *desert grassland whiptail lizard*). A rater who does not know the
word cannot answer meaningfully, which injects noise into a study whose whole
purpose is to measure lay comprehension. They need an on-demand way to learn
what an unfamiliar word refers to, without leaving the page or being primed
more than necessary.

## Decisions (locked during brainstorming)

1. **Content:** an example **photo + a one-line text gloss**.
2. **Source:** a **curated, bundled** reference set committed to the repo —
   seeded from the local ImageNet example cache, with hand-filled gaps. No
   runtime dependency on the (gitignored, 1 GB, out-of-deploy-context) cache;
   no external/web images (those would fire integrity-blur events, be
   unpredictable, and leak the rater's browser to third parties).
3. **Hard invariant:** a reference image must **never be a study stimulus**.
   Enforced by SHA-256 exclusion at curation time *and* a committed test.
4. **Interaction:** a small **ⓘ icon** on each glossed word option opens an
   **in-page popover** (works with mouse and touch, never leaves the page).
5. **Scope of glosses:** the **~21 fine-grained classes** that appear as pair
   options get an entry (gloss + image). The ~10 superordinate words (*snake*,
   *reptile*, *sparrow*, *equine*, *ratite*, *constrictor*, *iguana*,
   *retriever*, *songbird*, *percussion instrument*) get **no helper** — a
   layperson understands them and a single photo cannot represent a category.
6. **Logging:** record which references a rater opened per trial
   (`references_revealed`); this is itself validity signal. Requires an
   additive bump to the session schema.

## Scope

**In scope:** the pair phase only; the ⓘ appears only on the two *word* slots
(`ANCHOR_WORD` / `TARGET_WORD`), never on the tail options ("Something else",
"Nothing recognizable", "I can't tell").

**Out of scope:** class words appearing inside text-phase prompts; any change to
the stimulus images, the AB-randomization, the measurement contract, or the
existing scales.

## The 21 fine-grained pair-option words

These appear as `cell.anchor_word` / `target_word` for pair items across forms
A/B/C and receive a reference entry (gloss + image):

```
American bullfrog, American robin, axolotl, bald eagle, box turtle, cello,
chameleon, cock, desert grassland whiptail lizard, fire salamander, flamingo,
great grey owl, great white shark, indigo bunting, loggerhead sea turtle,
marimba, mud turtle, ostrich, stingray, tench, tiger shark
```

Local example coverage (`.cache/imagenet/category_images/<class>/`): present
for 18 of them (1–5 PNGs each); **3 gaps to source by hand**: *American
bullfrog*, *American robin*, *cock*.

The ~10 superordinate words that get **no** entry: *constrictor, equine,
iguana, percussion instrument, ratite, reptile, retriever, snake, songbird,
sparrow*.

## Data model

New committed file **`config/references.json`** (ships in the Docker image — the
build already `COPY`s `app/config`). Keyed by the exact word string used in
`option_labels`:

```json
{
  "schema_version": "1.0.0",
  "entries": {
    "tench":   { "gloss": "a large European freshwater fish", "image": "ref-tench.png" },
    "axolotl": { "gloss": "a salamander that keeps its larval gills as an adult", "image": "ref-axolotl.png" }
  }
}
```

- Every entry has **both** `gloss` (non-empty string) and `image`
  (a `ref-<class>.png` filename). There is no gloss-only entry.
- Reference images live at **`config/refs/ref-<class>.png`** (inside the
  `config` copy → always present at runtime).
- A new AJV schema `hs01.references.schema.json` validates the file; a loader
  (`loadReferences()`) mirrors `loadConfig()` (read, validate, memoize).

## Serving and data flow

- **New route `GET /api/refs/[name]`** — a near-copy of
  `src/app/api/images/[name]/route.ts`: strict allowlist
  `^ref-[a-z0-9_]+\.png$`, `path.basename` re-check, `404` on miss, immutable
  cache headers. Reads from `refsDir()` (default `config/refs`, overridable via
  `HS01_REFS_DIR`).
- **`CreateResult.references`** — `createSession()` (`src/lib/store.ts`) loads
  `references.json` and forwards the `entries` map as read-only presentation
  config, next to `pair_response`/`scales`. It flows through `useSession` to
  `PairChoice` as a prop. `PairChoice` looks up
  `references[optionLabels.ANCHOR_WORD]` / `[TARGET_WORD]`; a hit renders the ⓘ,
  a miss renders the bare word.

## Components

### `<WordReference word gloss image>` (new)

- Renders an ⓘ **button** plus an in-page popover containing the image
  (`<img src="/api/refs/ref-<class>.png">`) and the gloss.
- Accessible: `aria-haspopup="dialog"`, `aria-expanded`, labelled by the word;
  focus moves into the popover on open and returns to the ⓘ on close; **Esc**
  and outside-click close it.
- Respects `prefers-reduced-motion`. It is chrome only — it never gates or
  perturbs stimulus onset (the animation-free measurement guarantee is
  unaffected).
- Emits an `onReveal(word)` callback the first time it is opened in a trial.

### `PairChoice` integration

- The ⓘ is a **sibling** control next to each word radio, **not nested inside
  the radio button** (nested interactive elements are invalid HTML and would
  break the radio semantics). Row layout becomes
  `[ option radio (flex-1) ] [ ⓘ ]`.
- Clicking the ⓘ **does not select the option and does not increment
  `n_changes`** — the slot-reporting / AB-shuffle / `data-slot` / `data-testid`
  measurement contract is untouched.
- `PairChoice` collects revealed words (a `Set`) and surfaces them so the trial
  answer can record `references_revealed`.

## "Never a study stimulus" guarantee

A study stimulus is any `src-*.png` served to raters: the files under
`experiments/HS-01/pool_frozen/assets/images/` and, equivalently, every
`assets.image.sha256` in `pool_frozen/itempool.json`. `img-raw-*` items show a
clean ImageNet image directly, so those classes need extra care.

**Curation script** `experiments/HS-01/tools/build_references.py` (dev-only;
reads the gitignored cache; output is committed):

1. Build the study-stimulus SHA-256 set from `pool_frozen/assets/images/*.png`.
2. For each fine-grained word, enumerate candidates in
   `.cache/imagenet/category_images/<class>/`, compute each SHA-256, and
   **discard any whose hash is in the study-stimulus set**. Pick the first
   survivor; copy it to `config/refs/ref-<class>.png`.
3. Emit a `references.json` skeleton (`image` filled, `gloss` = `TODO`).

**Manual completion:** author the 21 glosses; source the 3 cache gaps
(*American bullfrog*, *American robin*, *cock*) from a license-clear source;
and **visually compare** the `img-raw-*` classes (e.g. *ostrich*, *cello*,
*marimba*) against their study stimuli, since scene-identity across a re-encode
is not caught by a hash.

**Enforcement test (Vitest):** every file in `config/refs/` has a SHA-256 not
present in the study-stimulus set. This re-runs on every build, so the
invariant cannot silently regress. A second test asserts each fine-grained
pair-option word has an entry and each entry's `image` file exists.

## Reveal logging

Add an optional **`references_revealed`** field to the per-trial `response`
object: an array of the word slots opened during that trial (e.g.
`["ANCHOR_WORD"]`). Recording slots (not raw strings) keeps it robust to the
per-trial AB shuffle and consistent with how `choice` is reported.

Touchpoints:
- `src/lib/types.ts` — add `references_revealed?: SemanticChoice[]` to
  `TrialAnswer`.
- `src/state/useSession.ts` / `src/lib/session-record.ts` — thread it into the
  written `response` object.
- `experiments/HS-01/schemas/hs01.session.schema.json` — add
  `references_revealed` (array of the word-slot enum) to `response.properties`;
  bump `schema_version` const to **`1.1.0`** (additive, backward-compatible)
  and update the writer to emit `1.1.0`.

## Error handling

- No entry for a word → no ⓘ; the word is still fully selectable.
- Image 404 → popover degrades to gloss-only text.
- Malformed/missing `references.json` → build/load-time failure (validated like
  the study config), not a silent runtime fallback.

## Testing

**Vitest**
- SHA-256 exclusion invariant over `config/refs/` (the hard guarantee).
- `references.json` validates; completeness (every fine-grained pair word has an
  entry; every entry image file exists; every gloss non-empty).
- `PairChoice`: ⓘ rendered only on word slots that have an entry; popover
  opens/closes; clicking ⓘ neither selects the option nor increments
  `n_changes`; `references_revealed` accumulates opened slots.

**Playwright e2e**
- Drive to a pair trial → click ⓘ → photo + gloss visible → **Esc** closes →
  option still selectable → submit succeeds and the record validates against the
  bumped schema.

## Files touched (summary)

| File | Change |
| --- | --- |
| `config/references.json` | new — committed entries (21 words) |
| `config/refs/ref-*.png` | new — committed reference images |
| `experiments/HS-01/schemas/hs01.references.schema.json` | new — AJV schema |
| `src/lib/references.ts` | new — `loadReferences()`, `refsDir()` |
| `src/app/api/refs/[name]/route.ts` | new — serve bundled refs |
| `src/components/WordReference.tsx` | new — ⓘ + popover |
| `src/components/PairChoice.tsx` | integrate ⓘ; collect revealed slots |
| `src/lib/store.ts` | add `references` to `CreateResult` |
| `src/lib/types.ts` | `TrialAnswer.references_revealed` |
| `src/state/useSession.ts`, `src/lib/session-record.ts` | thread reveal log |
| `experiments/HS-01/schemas/hs01.session.schema.json` | `references_revealed`; `schema_version` 1.1.0 |
| `experiments/HS-01/tools/build_references.py` | new — dev-only curation |
| tests (vitest + e2e) | new — invariant, component, e2e |

## Out of scope / future

- Glosses on text-phase prompt words.
- Multiple example images per word (one is sufficient and lowest-priming).
- Localisation of glosses (study locale is `en`).
