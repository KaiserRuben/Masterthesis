/*
 * Shared Alpine store: pipeline-explorer/assets/store.js
 *
 * Defines the single source-of-truth state shape consumed by every later
 * phase. Loads data/{config-schema.json, pipeline-data.json, manifest.json},
 * derives default config from the schema, and exposes the action methods
 * each phase calls.
 *
 * Phase contracts:
 *   - Phase 3 (canvas)         owns:   focusNode, pipeline, mode, theme, particleRate
 *                              reads:  config (for derived particle rate)
 *   - Phase 4 (widgets)        owns:   config, hoverKnob, dependency engine
 *                              reads:  schema, focusNode (to filter knobs)
 *   - Phase 5 (micro-vizzes)   reads:  config + schema
 *   - Phase 6 (YAML editor)    owns:   yamlText, hoverYamlLine
 *                              reads:  config (two-way bound)
 *   - Phase 7 (step mode)      owns:   step
 *   - Phase 8 (genotype play)  owns:   demoGenotype
 *   - Phase 9 (polish)         owns:   transition flags
 *
 * Store registration happens on `alpine:init`. The store name is 'explorer'.
 * Access from any component via `$store.explorer` (Alpine x-data) or
 * `Alpine.store('explorer')` (vanilla JS).
 */

(function () {
  "use strict";

  const DATA_DIR = new URL("../data/", document.currentScript.src).href;

  /** Fetch a JSON file relative to data/. Returns a Promise. */
  async function loadJSON(name) {
    const r = await fetch(DATA_DIR + name);
    if (!r.ok) throw new Error(`Failed to load ${name}: ${r.status}`);
    return r.json();
  }

  /**
   * Walk the schema and emit a nested object of defaults.
   * Schema leaves carry: { path, type, default, item_schema?, ... }
   * Paths are dotted; we expand into nested mutable objects so widgets
   * can $set the field by path and YAML round-trip preserves structure.
   */
  function defaultConfig(schema) {
    const cfg = {};
    for (const leaf of schema.leaves || []) {
      setByPath(cfg, leaf.path, cloneDefault(leaf));
    }
    return cfg;
  }

  function cloneDefault(leaf) {
    if (leaf.default === undefined || leaf.default === null) return null;
    return JSON.parse(JSON.stringify(leaf.default));
  }

  function setByPath(obj, path, value) {
    const parts = path.split(".");
    let cursor = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const k = parts[i];
      if (!(k in cursor) || typeof cursor[k] !== "object" || cursor[k] === null) {
        cursor[k] = {};
      }
      cursor = cursor[k];
    }
    cursor[parts[parts.length - 1]] = value;
  }

  function getByPath(obj, path) {
    const parts = path.split(".");
    let cursor = obj;
    for (const k of parts) {
      if (cursor == null) return undefined;
      cursor = cursor[k];
    }
    return cursor;
  }

  /** Determine which active pipeline a leaf belongs to. */
  function leafActive(leaf, activePipeline) {
    if (!leaf.pipeline || leaf.pipeline === "shared") return true;
    return leaf.pipeline === activePipeline;
  }

  /** Resolve dependsOn: returns true if every required field matches. */
  function dependsSatisfied(leaf, config) {
    const deps = leaf.dependsOn;
    if (!deps || Object.keys(deps).length === 0) return true;
    for (const [depPath, allowed] of Object.entries(deps)) {
      const current = getByPath(config, depPath);
      const wanted = Array.isArray(allowed) ? allowed : [allowed];
      if (!wanted.includes(current)) return false;
    }
    return true;
  }

  document.addEventListener("alpine:init", () => {
    window.Alpine.store("explorer", {
      // -----------------------------------------------------------------
      // raw data — populated by init()
      // -----------------------------------------------------------------
      schema: null,
      graph: null,
      manifest: null,
      ready: false,

      // -----------------------------------------------------------------
      // ui state owned by phase 3 (canvas) / phase 7 (step)
      // -----------------------------------------------------------------
      pipeline: "evolutionary",            // 'evolutionary' | 'pdq'
      mode: "canvas",                      // 'canvas' | 'step'
      theme: "dark",                       // 'dark' | 'light'
      focusNode: "manipulator_image",      // node_id from pipeline-data.json
      hoverNode: null,                     // node_id | null
      step: 1,                             // 1..6 (objectives → output)

      // -----------------------------------------------------------------
      // ui state owned by phase 4 (widgets)
      // -----------------------------------------------------------------
      config: {},                          // mirrors schema defaults
      hoverKnob: null,                     // dotted path | null

      // -----------------------------------------------------------------
      // ui state owned by phase 6 (yaml editor)
      // -----------------------------------------------------------------
      yamlText: "",
      hoverYamlLine: null,                 // 1-based line number | null
      yamlOpen: false,                     // drawer visibility (canvas-mode YAML editor)

      // -----------------------------------------------------------------
      // ui state owned by phase 8 (genotype playground)
      // -----------------------------------------------------------------
      demoGenotype: null,                  // Int array | null

      // -----------------------------------------------------------------
      // initialisation
      // -----------------------------------------------------------------
      async init() {
        const [schema, graph, manifest] = await Promise.all([
          loadJSON("config-schema.json"),
          loadJSON("pipeline-data.json"),
          loadJSON("manifest.json"),
        ]);
        this.schema = schema;
        this.graph = graph;
        this.manifest = manifest;
        this.config = defaultConfig(schema);
        this.theme = manifest.default_theme || "dark";
        this.mode = manifest.default_mode || "canvas";
        document.documentElement.setAttribute("data-theme", this.theme);
        this.ready = true;
        // Notify dependants — Alpine reactivity covers most but explicit
        // event lets non-Alpine code (vanilla viz, canvas SVG renderers)
        // react after data is loaded.
        window.dispatchEvent(new CustomEvent("explorer:ready"));
      },

      // -----------------------------------------------------------------
      // actions — pipeline / mode / theme switching
      // -----------------------------------------------------------------
      switchPipeline(p) {
        if (p !== "evolutionary" && p !== "pdq") return;
        this.pipeline = p;
        window.dispatchEvent(new CustomEvent("explorer:pipeline-changed", { detail: p }));
      },

      switchMode(m) {
        if (m !== "canvas" && m !== "step") return;
        this.mode = m;
      },

      toggleTheme() {
        this.theme = this.theme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", this.theme);
      },

      // -----------------------------------------------------------------
      // actions — YAML drawer (canvas mode)
      // -----------------------------------------------------------------
      openYaml() { this.yamlOpen = true; },
      closeYaml() { this.yamlOpen = false; },
      toggleYaml() { this.yamlOpen = !this.yamlOpen; },

      // -----------------------------------------------------------------
      // actions — focus / hover
      // -----------------------------------------------------------------
      setFocus(nodeId) {
        if (!nodeId || !(nodeId in (this.graph?.nodes || {}))) {
          this.focusNode = null;
          return;
        }
        this.focusNode = nodeId;
        window.dispatchEvent(new CustomEvent("explorer:focus-changed", { detail: nodeId }));
      },

      setHoverNode(nodeId) {
        this.hoverNode = nodeId;
      },

      setHoverKnob(path) {
        this.hoverKnob = path;
      },

      setHoverYamlLine(lineNo) {
        this.hoverYamlLine = lineNo;
      },

      // -----------------------------------------------------------------
      // actions — config mutation
      // -----------------------------------------------------------------
      setKnob(path, value) {
        setByPath(this.config, path, value);
        window.dispatchEvent(
          new CustomEvent("explorer:config-changed", { detail: { path, value } }),
        );
      },

      getKnob(path) {
        return getByPath(this.config, path);
      },

      // -----------------------------------------------------------------
      // derived: knobs visible for the focused node and active pipeline
      // -----------------------------------------------------------------
      activeKnobs(nodeId) {
        if (!this.schema || !this.graph) return [];
        const node = this.graph.nodes?.[nodeId];
        if (!node) return [];
        const paths = new Set(node.knob_paths || []);
        return (this.schema.leaves || []).filter((leaf) => {
          if (!paths.has(leaf.path)) return false;
          if (!leafActive(leaf, this.pipeline)) return false;
          if (!dependsSatisfied(leaf, this.config)) return false;
          return true;
        });
      },

      // -----------------------------------------------------------------
      // derived: active edges for current pipeline
      // -----------------------------------------------------------------
      activeEdges() {
        if (!this.graph) return [];
        return this.graph[this.pipeline === "pdq" ? "edges_pdq" : "edges_evolutionary"] || [];
      },

      // -----------------------------------------------------------------
      // derived: knob count visible for the active pipeline (shared + pipeline-specific)
      // -----------------------------------------------------------------
      activeKnobCount() {
        if (!this.schema) return 0;
        const p = this.pipeline;
        return (this.schema.leaves || []).filter(
          (l) => !l.pipeline || l.pipeline === "shared" || l.pipeline === p,
        ).length;
      },

      // -----------------------------------------------------------------
      // derived: subsystem count = unique nodes touched by active edges
      // -----------------------------------------------------------------
      activeSubsystemCount() {
        const edges = this.activeEdges();
        const ids = new Set();
        for (const e of edges) { ids.add(e.from); ids.add(e.to); }
        return ids.size;
      },

      // -----------------------------------------------------------------
      // derived: particle rate scaled by current budget config
      //   - evolutionary: pop_size * generations (capped)
      //   - pdq:          stage1.budget_sut_calls (capped)
      // returns particles-per-second for canvas animation; phase 3 uses it
      // -----------------------------------------------------------------
      particleRate() {
        if (this.pipeline === "evolutionary") {
          const pop = Number(this.getKnob("pop_size")) || 50;
          const gens = Number(this.getKnob("generations")) || 100;
          // Map [50..50_000] → [4..40] particles/sec, log-scaled
          const total = Math.max(1, pop * gens);
          return clamp(4 + 9 * Math.log10(total / 50), 4, 40);
        }
        const budget = Number(this.getKnob("stage1.budget_sut_calls")) || 1000;
        return clamp(4 + 9 * Math.log10(Math.max(1, budget / 100)), 4, 40);
      },

      // -----------------------------------------------------------------
      // helpers (exposed for phases that need them)
      // -----------------------------------------------------------------
      _setByPath: setByPath,
      _getByPath: getByPath,
      _leafActive: leafActive,
      _dependsSatisfied: dependsSatisfied,
    });
  });

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }
})();
