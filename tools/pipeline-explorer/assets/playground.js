/*
 * Genotype playground: a standalone interactive card that renders the
 * deterministic genotype to phenotype mapping for an in-browser demo. No
 * SUT call, no probability prediction, no Pareto outcome — only the
 * rendering pipeline (image patch substitutions + text operator chain).
 *
 * The genotype is a flat int array: image_dim image genes followed by
 * text_dim text genes (text_dim = 8 tokens * 4 operators = 32). The
 * canonical text operator order matches src/manipulator/text/composite:
 *   synonym -> fragmentation -> character_noise -> saliency
 *
 * Two display forms toggle from the card header:
 *   - schematic: flat coloured cells + token chips, fast.
 *   - svg:       16x16 SVG patch grid with per-cell pattern hue rotation,
 *                rendered tokens with character-level overlays.
 *
 * Both forms read the same demoGenotype array on $store.explorer.
 */
(function () {
  "use strict";

  // ── synthetic seed data ────────────────────────────────────────────
  const TOKENS = [
    "What", "is", "the", "main", "subject", "in", "this", "image",
  ];

  // Per-token synonym lookup. Index = synonym gene k (1..4); k=0 leaves
  // the token unchanged. Out-of-range k clamps to length.
  const SYNONYMS = {
    "What":    ["Which",   "How",       "Where",  "Why"],
    "is":      ["be",      "exists",    "stands", "lies"],
    "the":     ["a",       "this",      "that",   "any"],
    "main":    ["primary", "principal", "chief",  "key"],
    "subject": ["theme",   "topic",     "matter", "object"],
    "in":      ["within",  "inside",    "on",     "of"],
    "this":    ["the",     "that",      "a",      "any"],
    "image":   ["picture", "frame",     "photo",  "scene"],
  };

  // Cyrillic homoglyphs — visually identical Latin lookalikes.
  // Order of substitution within a token is deterministic by char index.
  const HOMOGLYPHS = {
    "a": "а", "e": "е", "o": "о",
    "c": "с", "p": "р", "x": "х",
    "y": "у", "i": "і",
  };

  // Image gene bounds (n_candidates) and text gene bounds (k_max).
  // Hard-coded for the playground; not derived from real config.
  const N_CANDIDATES = 25;
  const K_MAX = 5;
  const IMAGE_DIM = 64;    // 8x8 visible patches in schematic mode
  const SVG_GRID = 16;     // 16x16 in SVG mode (16*16=256 -> we view a 8x8 slice)
  const TEXT_OPS = 4;
  const TEXT_DIM = TOKENS.length * TEXT_OPS;
  const TOTAL_DIM = IMAGE_DIM + TEXT_DIM;

  // Operator labels in canonical order (must match application order below).
  const OPERATORS = ["synonym", "fragment", "homoglyph", "saliency"];
  const OP_LABEL = {
    synonym:   "synonym",
    fragment:  "fragmentation",
    homoglyph: "character noise",
    saliency:  "saliency swap",
  };

  // ── small utils ────────────────────────────────────────────────────
  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  // Deterministic hash for cell colour seed (per cell index).
  function cellHash(i) {
    let x = (i + 1) * 2654435761;
    x ^= x >>> 16;
    x = Math.imul(x, 2246822507);
    x ^= x >>> 13;
    return (x >>> 0) / 4294967296;
  }

  // Strategy proxy: in the playground we don't expose KNN/UNIFORM/KFN;
  // we just colour every cell as if KNN (smooth perturbation), which is
  // the default in the real pipeline. Strategy is shown as a label only.
  function cellColour(idx, gene, n) {
    const baseHue = Math.floor(cellHash(idx) * 360);
    const baseSat = 42 + Math.floor(cellHash(idx * 7 + 1) * 18);
    const baseLgt = 22 + Math.floor(cellHash(idx * 13 + 2) * 12);
    if (!gene) {
      return `hsl(${baseHue} ${baseSat}% ${baseLgt}%)`;
    }
    const t = gene / Math.max(1, n);
    // For schematic: rotate hue by t*180 and lift lightness.
    const hue = (baseHue + Math.floor(t * 180)) % 360;
    const sat = clamp(baseSat + Math.floor(t * 30), 0, 90);
    const lgt = clamp(baseLgt + Math.floor(t * 28), 8, 70);
    return `hsl(${hue} ${sat}% ${lgt}%)`;
  }

  // ── transformations (canonical order) ──────────────────────────────
  function applySynonym(token, k) {
    if (!k) return token;
    const opts = SYNONYMS[token];
    if (!opts || opts.length === 0) return token;
    const idx = clamp(k - 1, 0, opts.length - 1);
    return opts[idx];
  }

  // Insert k spaces at deterministic positions inside the token (avoid
  // the leading and trailing characters so the result still parses as a
  // chunk of text). Positions are evenly spaced.
  function applyFragmentation(token, k) {
    if (!k) return token;
    if (token.length < 2) return token;
    const max = Math.max(0, token.length - 1);
    const n = clamp(k, 0, max);
    if (!n) return token;
    const positions = [];
    for (let i = 1; i <= n; i++) {
      positions.push(Math.round((i * token.length) / (n + 1)));
    }
    let out = "";
    for (let i = 0; i < token.length; i++) {
      out += token[i];
      if (positions.includes(i + 1) && i + 1 < token.length) out += " ";
    }
    return out;
  }

  // Replace the first k eligible characters with their Cyrillic homoglyph.
  // We mark substitutions with U+200B-free output so the visual diff stays
  // in the colour overlay (SVG mode), not in extra glyphs.
  function applyCharacterNoise(token, k) {
    if (!k) return token;
    let replaced = 0;
    let out = "";
    for (let i = 0; i < token.length; i++) {
      const c = token[i];
      const lc = c.toLowerCase();
      if (replaced < k && HOMOGLYPHS[lc]) {
        const sub = HOMOGLYPHS[lc];
        out += c === lc ? sub : sub.toUpperCase();
        replaced++;
      } else {
        out += c;
      }
    }
    return out;
  }

  // Swap adjacent character pairs from the left, k pairs total.
  function applySaliency(token, k) {
    if (!k) return token;
    const arr = token.split("");
    const pairs = clamp(k, 0, Math.floor(arr.length / 2));
    for (let i = 0; i < pairs; i++) {
      const j = i * 2;
      if (j + 1 < arr.length) {
        const tmp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = tmp;
      }
    }
    return arr.join("");
  }

  // Apply the full pipeline in canonical order. geneVec = [k_syn, k_frag,
  // k_hom, k_sal] for one token.
  function renderToken(token, geneVec) {
    let s = applySynonym(token, geneVec[0] || 0);
    s = applyFragmentation(s, geneVec[1] || 0);
    s = applyCharacterNoise(s, geneVec[2] || 0);
    s = applySaliency(s, geneVec[3] || 0);
    return s;
  }

  // ── Alpine component ───────────────────────────────────────────────
  document.addEventListener("alpine:init", () => {
    window.Alpine.data("playground", () => ({
      // -----------------------------------------------------------------
      // display state
      // -----------------------------------------------------------------
      view: "schematic",       // 'schematic' | 'svg'
      hoverIdx: null,           // index into the flat genotype | null
      anchorPulse: false,       // briefly true after "All-zero anchor"

      // -----------------------------------------------------------------
      // init: allocate demoGenotype on the shared store
      // -----------------------------------------------------------------
      init() {
        const store = this.$store.explorer;
        if (!Array.isArray(store.demoGenotype) ||
            store.demoGenotype.length !== TOTAL_DIM) {
          store.demoGenotype = new Array(TOTAL_DIM).fill(0);
        }
        // Per-cell DOM refs for the SVG view ─ avoids rebuilding 64 <g>
        // nodes on every gene change. Populated lazily by renderSvgCells.
        this._svgCellRefs = new Map();
        // Per-cell <div> refs for the live phenotype preview grid.
        this._previewCellRefs = new Map();

        // Re-render whenever the genotype, view, or hover state changes.
        this.$watch("$store.explorer.demoGenotype", () => {
          this.renderSvgCells();
          this.renderPreviewCells();
        });
        this.$watch("view", () => {
          // When switching back to svg we may need a full rebuild,
          // because the surrounding wrapper was display:none and the
          // <g> children we keep around are still attached and valid;
          // we just refresh their visual state.
          if (this.view === "svg") this.renderSvgCells();
        });
        this.$watch("hoverIdx", (next, prev) => {
          // Only the hover ring changes; do the fast path.
          if (this.view === "svg") {
            this._refreshHoverRings(prev, next);
          }
        });

        // First paint after Alpine has mounted the DOM children.
        this.$nextTick(() => {
          this.renderSvgCells();
          this.renderPreviewCells();
        });
      },

      // -----------------------------------------------------------------
      // exposed constants (Alpine reads via x-text / loops)
      // -----------------------------------------------------------------
      get IMAGE_DIM() { return IMAGE_DIM; },
      get TEXT_DIM()  { return TEXT_DIM; },
      get N_CANDIDATES() { return N_CANDIDATES; },
      get K_MAX() { return K_MAX; },
      get TOKENS() { return TOKENS; },
      get OPERATORS() { return OPERATORS; },
      get OP_LABEL() { return OP_LABEL; },
      get SVG_GRID() { return SVG_GRID; },

      // -----------------------------------------------------------------
      // read accessors over the shared array
      // -----------------------------------------------------------------
      get genotype() {
        return this.$store.explorer.demoGenotype || [];
      },

      // Image gene at visible cell idx (0..IMAGE_DIM-1).
      imageGene(idx) {
        return this.genotype[idx] | 0;
      },

      // Text genes: tokenIdx in [0, 7], opSlot in [0, 3].
      // Layout: genotype[IMAGE_DIM + tokenIdx*4 + opSlot]
      textGene(tokenIdx, opSlot) {
        return this.genotype[IMAGE_DIM + tokenIdx * TEXT_OPS + opSlot] | 0;
      },

      textGeneVec(tokenIdx) {
        const base = IMAGE_DIM + tokenIdx * TEXT_OPS;
        return [
          this.genotype[base]     | 0,
          this.genotype[base + 1] | 0,
          this.genotype[base + 2] | 0,
          this.genotype[base + 3] | 0,
        ];
      },

      renderedToken(tokenIdx) {
        return renderToken(TOKENS[tokenIdx], this.textGeneVec(tokenIdx));
      },

      // Joined final prompt — read by the live phenotype block.
      get renderedPrompt() {
        return TOKENS.map((_, i) => this.renderedToken(i)).join(" ");
      },

      // True when every gene is zero — drives the "anchor genotype" label.
      get isAnchor() {
        for (let i = 0; i < this.genotype.length; i++) {
          if (this.genotype[i]) return false;
        }
        return true;
      },

      // -----------------------------------------------------------------
      // write actions
      // -----------------------------------------------------------------
      _setImage(idx, value) {
        const next = this.genotype.slice();
        next[idx] = ((value % (N_CANDIDATES + 1)) + (N_CANDIDATES + 1)) %
          (N_CANDIDATES + 1);
        this.$store.explorer.demoGenotype = next;
      },

      _setText(tokenIdx, opSlot, value) {
        const next = this.genotype.slice();
        const flat = IMAGE_DIM + tokenIdx * TEXT_OPS + opSlot;
        next[flat] = clamp(value | 0, 0, K_MAX);
        this.$store.explorer.demoGenotype = next;
      },

      // Click a patch: gene + 1 mod (N+1). Shift-click decrements.
      bumpImage(idx, ev) {
        const cur = this.imageGene(idx);
        const step = ev && ev.shiftKey ? -1 : 1;
        this._setImage(idx, cur + step);
      },

      setTextGene(tokenIdx, opSlot, value) {
        this._setText(tokenIdx, opSlot, value);
      },

      reset() {
        this.$store.explorer.demoGenotype = new Array(TOTAL_DIM).fill(0);
        this.anchorPulse = false;
      },

      randomize() {
        const next = new Array(TOTAL_DIM);
        for (let i = 0; i < IMAGE_DIM; i++) {
          next[i] = Math.floor(Math.random() * (N_CANDIDATES + 1));
        }
        for (let i = 0; i < TEXT_DIM; i++) {
          next[IMAGE_DIM + i] = Math.floor(Math.random() * (K_MAX + 1));
        }
        this.$store.explorer.demoGenotype = next;
        this.anchorPulse = false;
      },

      // Like reset, but triggers a brief animated callout. The label is
      // already visible whenever isAnchor === true; the pulse just draws
      // the eye to it after an explicit user click.
      anchor() {
        this.$store.explorer.demoGenotype = new Array(TOTAL_DIM).fill(0);
        this.anchorPulse = true;
        // Clear the pulse after the CSS animation settles.
        setTimeout(() => { this.anchorPulse = false; }, 1400);
      },

      toggleView() {
        this.view = this.view === "schematic" ? "svg" : "schematic";
      },

      setView(v) {
        if (v === "schematic" || v === "svg") this.view = v;
      },

      // -----------------------------------------------------------------
      // hover cross-link (cell / token <-> dump line)
      // -----------------------------------------------------------------
      setHover(idx) { this.hoverIdx = idx; },
      clearHover() { this.hoverIdx = null; },

      isHovered(idx) { return this.hoverIdx === idx; },

      // Image dump line is highlighted if any of its image indices match
      // the hovered flat index.
      imageDumpHovered() {
        return this.hoverIdx !== null && this.hoverIdx < IMAGE_DIM;
      },
      textDumpHovered() {
        return this.hoverIdx !== null && this.hoverIdx >= IMAGE_DIM;
      },

      // -----------------------------------------------------------------
      // schematic-mode cell styling
      // -----------------------------------------------------------------
      cellStyle(idx) {
        const gene = this.imageGene(idx);
        return {
          background: cellColour(idx, gene, N_CANDIDATES),
          outline: this.isHovered(idx) ? "1px solid var(--accent)" : "none",
        };
      },

      // -----------------------------------------------------------------
      // SVG-mode rendering helpers — keep them pure so the SVG template
      // can read them directly via x-bind.
      // -----------------------------------------------------------------
      svgCellHue(idx) {
        const gene = this.imageGene(idx);
        const base = Math.floor(cellHash(idx) * 360);
        const t = gene / Math.max(1, N_CANDIDATES);
        return (base + Math.floor(t * 360)) % 360;
      },

      svgCellPattern(idx) {
        // Return a small pattern fragment id keyed by gene parity, so
        // SVG mode visibly changes the pattern (not just the hue) when
        // the gene moves through buckets.
        const gene = this.imageGene(idx);
        const bucket = gene === 0 ? 0 : (gene % 4) + 1; // 0..4
        return `pg-pat-${bucket}`;
      },

      // -----------------------------------------------------------------
      // Programmatic SVG cell rendering
      //
      // Alpine <template x-for> does not work inside an <svg> element:
      // the HTML parser puts <template> in the SVG namespace and the
      // resulting node has no HTMLTemplateElement.content, so Alpine
      // throws on Document.importNode. We render the 64 cells via
      // createElementNS instead, keeping per-cell DOM refs in a Map
      // so subsequent updates are cheap.
      // -----------------------------------------------------------------
      _computeSvgCellFill(idx) {
        const gene = this.imageGene(idx);
        const hue = this.svgCellHue(idx);
        const sat = 45 + (gene ? 25 : 0);
        const lgt = 22 + (gene / N_CANDIDATES) * 28;
        return `hsl(${hue} ${sat}% ${lgt}%)`;
      },

      renderSvgCells() {
        const root = document.getElementById("playground-svg-cells");
        if (!root) return;
        const ns = "http://www.w3.org/2000/svg";

        // If the cell skeleton doesn't yet exist, build it. Otherwise
        // just refresh per-cell visuals.
        if (this._svgCellRefs.size !== IMAGE_DIM) {
          while (root.firstChild) root.removeChild(root.firstChild);
          this._svgCellRefs.clear();

          for (let i = 0; i < IMAGE_DIM; i++) {
            const idx = i;
            const tx = (i % 8) * 20;
            const ty = Math.floor(i / 8) * 20;

            const g = document.createElementNS(ns, "g");
            g.setAttribute("transform", `translate(${tx},${ty})`);
            g.setAttribute("style", "cursor: pointer;");
            g.addEventListener("click", (evt) => this.bumpImage(idx, evt));
            g.addEventListener("mouseenter", () => this.setHover(idx));

            const base = document.createElementNS(ns, "rect");
            base.setAttribute("width", "20");
            base.setAttribute("height", "20");

            const pat = document.createElementNS(ns, "rect");
            pat.setAttribute("width", "20");
            pat.setAttribute("height", "20");
            pat.setAttribute("fill-opacity", "0.55");

            const accent = document.createElementNS(ns, "circle");
            accent.setAttribute("cx", "16");
            accent.setAttribute("cy", "4");
            accent.setAttribute("r", "1.6");
            accent.setAttribute("fill", "var(--accent)");

            const ring = document.createElementNS(ns, "rect");
            ring.setAttribute("width", "20");
            ring.setAttribute("height", "20");
            ring.setAttribute("fill", "none");
            ring.setAttribute("stroke", "var(--accent)");
            ring.setAttribute("stroke-width", "1.4");

            g.appendChild(base);
            g.appendChild(pat);
            g.appendChild(accent);
            g.appendChild(ring);
            root.appendChild(g);

            this._svgCellRefs.set(idx, { root: g, base, pat, accent, ring });
          }
        }

        // Refresh visuals for every cell. (Cheap: 64 attribute writes.)
        for (let i = 0; i < IMAGE_DIM; i++) {
          this._refreshSvgCell(i);
        }
      },

      _refreshSvgCell(idx) {
        const refs = this._svgCellRefs.get(idx);
        if (!refs) return;
        const gene = this.imageGene(idx);
        const fill = this._computeSvgCellFill(idx);

        refs.base.setAttribute("fill", fill);
        refs.pat.setAttribute("fill", `url(#${this.svgCellPattern(idx)})`);
        // <rect color="…"> is what gives the pattern's currentColor a value.
        refs.pat.setAttribute("color", fill);
        refs.accent.style.display = gene !== 0 ? "" : "none";
        refs.ring.style.display = this.isHovered(idx) ? "" : "none";
      },

      _refreshHoverRings(prevIdx, nextIdx) {
        // Toggle only the rings that actually changed.
        if (typeof prevIdx === "number" && prevIdx >= 0 && prevIdx < IMAGE_DIM) {
          const refs = this._svgCellRefs.get(prevIdx);
          if (refs) refs.ring.style.display = "none";
        }
        if (typeof nextIdx === "number" && nextIdx >= 0 && nextIdx < IMAGE_DIM) {
          const refs = this._svgCellRefs.get(nextIdx);
          if (refs) refs.ring.style.display = "";
        }
      },

      // -----------------------------------------------------------------
      // Programmatic phenotype-preview rendering
      //
      // The preview grid lives in plain HTML (a <div> grid). We still
      // render it programmatically so the rendering path matches the
      // SVG view and so we don't need any <template x-for> here.
      // -----------------------------------------------------------------
      renderPreviewCells() {
        const root = document.getElementById("playground-preview-cells");
        if (!root) return;

        if (this._previewCellRefs.size !== IMAGE_DIM) {
          while (root.firstChild) root.removeChild(root.firstChild);
          this._previewCellRefs.clear();
          for (let i = 0; i < IMAGE_DIM; i++) {
            const div = document.createElement("div");
            div.className = "playground__preview-cell";
            root.appendChild(div);
            this._previewCellRefs.set(i, div);
          }
        }
        for (let i = 0; i < IMAGE_DIM; i++) {
          const div = this._previewCellRefs.get(i);
          if (!div) continue;
          div.style.background = this.cellStyle(i).background;
        }
      },

      // -----------------------------------------------------------------
      // genotype-dump helpers (compact, with the hovered slot accented)
      // -----------------------------------------------------------------
      imageDumpHTML() {
        const arr = this.genotype.slice(0, IMAGE_DIM);
        return arr.map((v, i) => {
          const hovered = this.hoverIdx === i;
          const mut = v !== 0;
          const cls = [
            "playground__dump-num",
            mut ? "playground__dump-num--mut" : "",
            hovered ? "playground__dump-num--hover" : "",
          ].filter(Boolean).join(" ");
          return `<span class="${cls}">${v}</span>`;
        }).join(", ");
      },

      textDumpHTML() {
        const arr = this.genotype.slice(IMAGE_DIM, IMAGE_DIM + TEXT_DIM);
        return arr.map((v, i) => {
          const flat = IMAGE_DIM + i;
          const hovered = this.hoverIdx === flat;
          const mut = v !== 0;
          const tail = (i + 1) % 4 === 0 && i !== arr.length - 1 ? " &nbsp;" : "";
          const cls = [
            "playground__dump-num",
            mut ? "playground__dump-num--mut" : "",
            hovered ? "playground__dump-num--hover" : "",
          ].filter(Boolean).join(" ");
          return `<span class="${cls}">${v}</span>${tail}`;
        }).join(", ");
      },

      // -----------------------------------------------------------------
      // strategy label (read-only; reflects image.n_candidates knob)
      // -----------------------------------------------------------------
      get strategyLabel() {
        const n = Number(this.$store.explorer.getKnob("image.n_candidates"));
        if (!Number.isFinite(n)) return "KNN · n=25";
        return `KNN · n=${n}`;
      },

      get textProfileLabel() {
        const p = this.$store.explorer.getKnob("text.composite.profile");
        return p ? `profile · ${p}` : "profile · default";
      },

      // Count of mutated genes — driver of the header chip.
      get mutatedCount() {
        let n = 0;
        for (let i = 0; i < this.genotype.length; i++) {
          if (this.genotype[i]) n++;
        }
        return n;
      },
    }));
  });
})();
