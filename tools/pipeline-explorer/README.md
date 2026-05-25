# pipeline-explorer

Single-page HTML app for visualising and configuring the two
VLM-boundary-testing pipelines (`evolutionary`, `pdq`). Static, no
build step, no backend.

## Run

```
open tools/pipeline-explorer/index.html
```

Works file:// — the page fetches Tailwind, Google Fonts (Inter +
JetBrains Mono), Motion One, js-yaml, KaTeX, and Alpine.js from
public CDNs, plus its own CSS and JS in `assets/`.

## Rebuild data

```
python3 tools/pipeline-explorer/build_data.py
```

Re-extracts `data/config-schema.json` (109 leaf knobs),
`data/pipeline-data.json` (12 nodes + edges), and
`data/manifest.json` (commit hash) from the live `src/` tree.

## Directory

```
tools/pipeline-explorer/
  index.html              single-page entry, all markup
  assets/
    fonts.css             font-family stack tokens
    styles.css            design tokens (color · type · space · motion)
    components.css        component classes (panel · node · knob · …)
    store.js              shared Alpine store (single source of truth)
    canvas.js             SVG pipeline canvas (nodes, edges, particles)
    widgets.js            detail-panel knob widgets + dependency engine
    vizzes.js             hero micro-vizzes mounted under hero-tier knobs
    yaml-editor.js        live two-way YAML drawer
    step.js               step-mode wizard (5-cluster red thread)
    playground.js         genotype playground (g→ph demo card)
    polish.js             entry choreography, KaTeX, drawer, keyboard, theme
  data/
    config-schema.json    109 leaf knobs from src/config.py
    pipeline-data.json    12 nodes (config..artifacts) + per-pipeline edges
    manifest.json         commit hash, version, default theme/mode
  build_data.py           generator for the three JSON files
  README.md               this file
```

## Design notes

**Accent.** Electric green `#7CFA9C` (dark) / `#14B863` (light). Chosen
over generic blue because the tool is about search and discovery,
not warnings or fintech. Red (`#FF7A59`) is reserved for the
"search-expensive" cost semantic — never as a generic danger colour.
Modality accents (joint = violet, image = cyan, text = amber) sit
hue-adjacent so the legend transfers across themes without re-learning.

**Theme.** Dark is default. Light is warm (cream `#F6F3EC`), not
flipped white — the dribbble-tier feel is preserved by keeping the
green accent + same shadow softness, only with reduced contrast on
borders so the surface still reads as paper rather than glass.

**Type.** Inter Variable for display and body; JetBrains Mono for
config keys, line numbers, and any numeric callout. Display headings
use weight 720 + tracking `-0.02em`; body sits at 15px / 1.6 leading.
Numbers always opt into tabular-nums via the `.tabular` utility so
column-aligned configuration values track at 1px precision.

**Spacing.** 4px grid (`--space-1` = 4px, doubled at every step). Card
radius 14px, button/input radius 10px, chips full-pill. No
`rounded-2xl` everywhere; corner radius signals depth tier.

**Motion.** Three eases — `--ease-standard` for state changes,
`--ease-emphasized` for mode switches, `--ease-bounce` reserved for
affordance (segmented-control thumb). Three durations
(120 / 220 / 400ms). All animations route through Motion One and
respect `prefers-reduced-motion` plus the in-app toggle.

**Layout.** Canvas mode is a three-pane grid (40 / 35 / 25). Step
mode replaces it via Alpine `x-show`, toggled by the Canvas/Step
segmented control in the top bar.

**Wordmark.** `pipeline.` with the dot in accent — chosen over
`boundary.` because the tool's job is to explore the pipeline; the
boundary is the subject, not the product.
