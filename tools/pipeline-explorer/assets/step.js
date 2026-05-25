/*
 * Step-mode wizard: 5-cluster red-thread walkthrough.
 *
 * Mounts two Alpine components:
 *   stepMode     — owns step navigation, sidebar, derived consequence,
 *                  mini-pipeline chip row. Exposes currentCluster + the
 *                  group descriptor list the panel renders.
 *   clusterPanel — pure renderer of one cluster's groups + knobs. Reads
 *                  groups from its parent stepMode via $data. Duplicates
 *                  phase 4's widget vocabulary (we may not modify
 *                  widgets.js) but routes setVal/val/hover through the
 *                  same store helpers, so the YAML editor and canvas
 *                  cross-link continues to work.
 *
 * The old node-centric STEP_ORDER (per-pipeline, 6 entries each) is gone.
 * The new one is pipeline-agnostic at the top level; clusters 3 and 4
 * carry a `branches: { evolutionary, pdq }` override that swaps the
 * groups when the pipeline changes mid-flow.
 */
(function () {
  "use strict";

  // ─────────────────────────────────────────────────────────────────────
  // 5-cluster red thread. Each cluster:
  //   { id, label, intro, explainer,
  //     groups?: [{ label, paths }],
  //     branches?: { evolutionary: { groups }, pdq: { groups } },
  //     derived?: (store) => string | null }
  //
  // Each group's `paths` is a list of:
  //   - exact dotted path (e.g. "modality")
  //   - glob ending in ".*" (e.g. "image.*", "seeds.roster.*")
  //   - special token "__pipeline__" — renders the pipeline switcher hero
  // The resolver below expands globs against schema.leaves and filters
  // by active pipeline + dependsOn before passing to the panel.
  // ─────────────────────────────────────────────────────────────────────
  const STEP_ORDER = [
    {
      id: 1,
      label: "Problem",
      intro: "What flip am I hunting?",
      explainer:
        "Pick the pipeline, the seeds, and the modality lens. Boundary-near " +
        "(image, label A, label B) triples plus the modality choice fix the " +
        "objective count for the search.",
      groups: [
        { label: "Pipeline",   paths: ["__pipeline__"] },
        { label: "Modality",   paths: ["modality", "score_full_categories"] },
        {
          label: "Seeds",
          paths: [
            "seeds.mode",
            "seeds.filter_indices",
            "seeds.gap_filter.*",
            "seeds.roster.*",
          ],
        },
        {
          label: "Categories",
          paths: ["n_categories", "categories", "prompt_template", "answer_format"],
        },
      ],
      derived: (store) => {
        // PDQ doesn't use the modality lever — its own distance metrics
        // drive the objective count.
        if (store.pipeline !== "evolutionary") {
          return "PDQ uses its own distance metrics (distances.d_i_primary + d_o_primary) — modality is an evolutionary-only lever.";
        }
        const m = store.getKnob("modality");
        if (m === "joint")
          return "Joint modality: optimiser minimises MatrixDistance + TextEmbeddingDistance + TargetedBalance.";
        if (m === "image_only")
          return "Image-only modality: text manipulator forced to noop; 2 objectives (MatrixDistance + TargetedBalance).";
        if (m === "text_only")
          return "Text-only modality: image patch ratio forced to 0; 2 objectives (TextEmbeddingDistance + TargetedBalance).";
        return null;
      },
    },
    {
      id: 2,
      label: "Surface",
      intro: "What can I change?",
      explainer:
        "Image patches and text composite operators are the two perturbation " +
        "axes. Modality may have disabled one — the disabled surface still " +
        "appears here (its values are kept; the runtime forces them at call).",
      groups: [
        { label: "Image manipulator", paths: ["image.*"] },
        { label: "Text manipulator",  paths: ["text.*"] },
      ],
    },
    {
      id: 3,
      label: "Search",
      intro: "How do I look?",
      explainer:
        "Evolutionary runs AGE-MOEA-II with a sampler + early-stop. PDQ " +
        "runs a two-stage directed search with per-stage budgets and its " +
        "own image/output distance metrics.",
      branches: {
        evolutionary: {
          groups: [
            { label: "Population",   paths: ["pop_size", "generations"] },
            { label: "Init sampler", paths: ["optimizer.sampling.*"] },
            { label: "Early stop",   paths: ["optimizer.early_stop.*"] },
          ],
        },
        pdq: {
          groups: [
            { label: "Stage 1 — flip discovery", paths: ["stage1.*"] },
            { label: "Stage 2 — minimisation",   paths: ["stage2.*"] },
            { label: "Distances",                paths: ["distances.*"] },
          ],
        },
      },
    },
    {
      id: 4,
      label: "Evidence",
      intro: "What do I keep?",
      explainer:
        "Evolutionary writes Pareto + traces by default. PDQ has an explicit " +
        "archive filter and finer logging granularity so each stage can be " +
        "analysed independently.",
      branches: {
        evolutionary: {
          groups: [
            { label: "Output", paths: ["save_dir", "name"] },
          ],
        },
        pdq: {
          groups: [
            { label: "Output",  paths: ["save_dir", "name"] },
            { label: "Archive", paths: ["archive.*"] },
            { label: "Logging", paths: ["logging.*"] },
          ],
        },
      },
    },
    {
      id: 5,
      label: "Runtime",
      intro: "Where am I running it?",
      explainer:
        "Hardware, caching, parallelism, reproducibility. Reproduce the " +
        "same scientific claim on a different machine without touching the " +
        "search above.",
      groups: [
        { label: "Device + SUT",    paths: ["device", "sut.*"] },
        { label: "Parallelism",     paths: ["parallel.*", "concurrency.*"] },
        { label: "Reproducibility", paths: ["reproducibility.*"] },
        { label: "Cache",           paths: ["cache_dirs"] },
      ],
    },
  ];

  // Chip-row labels for the mini pipeline preview. Each chip belongs to
  // one or more clusters — chipClusters maps chip → [stepId, …] so a
  // chip can pulse on multiple steps (e.g. SUT chip lights on step 1
  // when categories drive prompts and on step 5 when the model loads).
  // step is the canonical jumpTo target (first cluster the chip belongs
  // to); chipClusters is the full set used for the "pulse" predicate.
  const CHIP_LAYOUT = {
    evolutionary: [
      { node: "seeds",             label: "seeds", step: 1, clusters: [1] },
      { node: "manipulator_image", label: "mImg",  step: 2, clusters: [2] },
      { node: "manipulator_text",  label: "mTxt",  step: 2, clusters: [2] },
      { node: "sut",               label: "sut",   step: 5, clusters: [1, 5] },
      { node: "objectives",        label: "obj",   step: 1, clusters: [1] },
      { node: "optimizer",         label: "opt",   step: 3, clusters: [3] },
      { node: "artifacts",         label: "out",   step: 4, clusters: [4] },
    ],
    pdq: [
      { node: "seeds",             label: "seeds", step: 1, clusters: [1] },
      { node: "manipulator_image", label: "mImg",  step: 2, clusters: [2] },
      { node: "manipulator_text",  label: "mTxt",  step: 2, clusters: [2] },
      { node: "sut",               label: "sut",   step: 5, clusters: [1, 5] },
      { node: "pdq_metric",        label: "metric", step: 3, clusters: [3, 4] },
      { node: "pdq_stage1",        label: "stg1",  step: 3, clusters: [3] },
      { node: "pdq_stage2",        label: "stg2",  step: 3, clusters: [3] },
      { node: "artifacts",         label: "out",   step: 4, clusters: [4] },
    ],
  };

  // For sidebar "X keys in this step" summary.
  function countKeys(groupItems) {
    let n = 0;
    for (const g of groupItems) {
      for (const it of g.items) {
        if (it && !it.__special) n += 1;
      }
    }
    return n;
  }

  // ─────────────────────────────────────────────────────────────────────
  // resolveGroupPaths · glob expand + pipeline/depends filter
  // Reads $store.explorer for schema + active pipeline + config.
  // Returns: { label, items: [leaf | { __special: 'pipeline' }] }
  // ─────────────────────────────────────────────────────────────────────
  function resolveGroupPaths(globs, store) {
    const schema = store.schema;
    const activePipeline = store.pipeline;
    const config = store.config;
    const out = [];
    const seen = new Set();
    if (!schema) return out;
    for (const g of globs) {
      if (g === "__pipeline__") {
        out.push({ __special: "pipeline" });
        continue;
      }
      if (g.endsWith(".*")) {
        const prefix = g.slice(0, -1); // keep trailing "."
        for (const leaf of schema.leaves || []) {
          if (seen.has(leaf.path)) continue;
          if (!leaf.path.startsWith(prefix)) continue;
          if (!store._leafActive(leaf, activePipeline)) continue;
          if (!store._dependsSatisfied(leaf, config)) continue;
          seen.add(leaf.path);
          out.push(leaf);
        }
      } else {
        const leaf = (schema.leaves || []).find((l) => l.path === g);
        if (
          leaf &&
          !seen.has(leaf.path) &&
          store._leafActive(leaf, activePipeline) &&
          store._dependsSatisfied(leaf, config)
        ) {
          seen.add(leaf.path);
          out.push(leaf);
        }
      }
    }
    return out;
  }

  // Resolve the group descriptors for a cluster, applying branch overrides.
  function resolvedGroups(cluster, store) {
    const groups =
      cluster.branches?.[store.pipeline]?.groups || cluster.groups || [];
    return groups.map((g) => ({
      label: g.label,
      items: resolveGroupPaths(g.paths, store),
    }));
  }

  // Pick a canvas focusNode appropriate for the current cluster, used
  // when entering step mode so the canvas-mode mirror behind the scene
  // doesn't sit on a stale subsystem.
  function focusForCluster(clusterId, pipeline) {
    if (clusterId === 1) return "seeds";
    if (clusterId === 2) return "manipulator_image";
    if (clusterId === 3) return pipeline === "pdq" ? "pdq_stage1" : "optimizer";
    if (clusterId === 4) return "artifacts";
    if (clusterId === 5) return "sut";
    return "manipulator_image";
  }

  document.addEventListener("alpine:init", () => {
    // -----------------------------------------------------------------
    // stepMode — owns the cluster navigation + sidebar + chip row
    // -----------------------------------------------------------------
    window.Alpine.data("stepMode", () => ({
      completed: [],

      init() {
        const syncOnReady = () => this._syncFocusToStep();
        if (this.$store.explorer.ready) {
          syncOnReady();
        } else {
          window.addEventListener("explorer:ready", syncOnReady, { once: true });
        }

        this.$watch("$store.explorer.pipeline", () => {
          if (this.$store.explorer.mode === "step") this._syncFocusToStep();
        });

        this.$watch("$store.explorer.mode", (m) => {
          if (m === "step") this._syncFocusToStep();
        });

        this.$watch("$store.explorer.step", () => {
          this._syncFocusToStep();
          this._fadeStepContent();
          window.dispatchEvent(
            new CustomEvent("explorer:step-changed", {
              detail: this.$store.explorer.step,
            }),
          );
        });
      },

      // ─── derived ──────────────────────────────────────────────────
      get stepEntries() {
        return STEP_ORDER;
      },

      get currentCluster() {
        const id = this.$store.explorer.step;
        return STEP_ORDER.find((c) => c.id === id) || STEP_ORDER[0];
      },

      // alias kept for any legacy bindings
      get currentEntry() {
        return this.currentCluster;
      },

      // The groups (with already-resolved knob items) for the current
      // cluster. Reactive to pipeline + config changes.
      get currentGroups() {
        const cl = this.currentCluster;
        if (!cl) return [];
        return resolvedGroups(cl, this.$store.explorer);
      },

      get currentDerived() {
        const cl = this.currentCluster;
        if (!cl || typeof cl.derived !== "function") return null;
        try {
          return cl.derived(this.$store.explorer);
        } catch (_) {
          return null;
        }
      },

      get currentKeyCount() {
        return countKeys(this.currentGroups);
      },

      // ─── mini-pipeline chip row ───────────────────────────────────
      get miniChips() {
        const chips =
          CHIP_LAYOUT[this.$store.explorer.pipeline] || CHIP_LAYOUT.evolutionary;
        const completed = new Set(this.completed);
        const curStep = this.$store.explorer.step;
        return chips.map((c) => ({
          ...c,
          completed: c.clusters.every((id) => completed.has(id)),
          // pulse if any of the chip's clusters is the current step
          touched: c.clusters.includes(curStep),
        }));
      },

      // ─── step state predicates ────────────────────────────────────
      isComplete(id) {
        return this.completed.includes(id);
      },

      // ─── actions ──────────────────────────────────────────────────
      advance() {
        const cur = this.$store.explorer.step;
        if (!this.completed.includes(cur)) this.completed.push(cur);
        this._animateBullet(cur);
        if (cur >= STEP_ORDER.length) {
          this.$store.explorer.switchMode("canvas");
          return;
        }
        this.$store.explorer.step = cur + 1;
      },

      skip() {
        const cur = this.$store.explorer.step;
        if (cur >= STEP_ORDER.length) {
          this.$store.explorer.switchMode("canvas");
          return;
        }
        this.$store.explorer.step = cur + 1;
      },

      back() {
        const cur = this.$store.explorer.step;
        if (cur <= 1) return;
        this.$store.explorer.step = cur - 1;
      },

      jumpTo(id) {
        if (id < 1 || id > STEP_ORDER.length) return;
        if (id === this.$store.explorer.step) return;
        this.$store.explorer.step = id;
      },

      // ─── internal ─────────────────────────────────────────────────
      _syncFocusToStep() {
        const cl = this.currentCluster;
        if (!cl) return;
        const want = focusForCluster(cl.id, this.$store.explorer.pipeline);
        if (this.$store.explorer.focusNode !== want) {
          this.$store.explorer.setFocus(want);
        }
      },

      _fadeStepContent() {
        const motion = window.Motion;
        const el = this.$refs?.stepContent;
        if (!motion || !motion.animate || !el) return;
        motion.animate(
          el,
          { opacity: [0, 1] },
          { duration: 0.2, easing: "ease-out" },
        );
      },

      _animateBullet(stepId) {
        const motion = window.Motion;
        if (!motion || !motion.animate) return;
        const el = document.querySelector(`[data-step-bullet="${stepId}"]`);
        if (!el) return;
        motion.animate(
          el,
          { transform: ["scale(1)", "scale(1.18)", "scale(1)"] },
          { duration: 0.2, easing: "ease-out" },
        );
      },
    }));

    // -----------------------------------------------------------------
    // clusterPanel — renders one cluster's groups + widgets.
    // Reads `currentGroups` from the parent stepMode (Alpine inherits
    // x-data up the scope chain). Widget vocabulary mirrors phase 4.
    // -----------------------------------------------------------------
    window.Alpine.data("clusterPanel", () => ({
      openDropdown: null,
      tupleBuffers: {},

      init() {
        this._closeOnDocClick = (e) => {
          if (!this.openDropdown) return;
          if (!e.target.closest("[data-dropdown-root]")) {
            this.openDropdown = null;
          }
        };
        document.addEventListener("click", this._closeOnDocClick, true);
        this.$watch("$store.explorer.step", () => {
          this.openDropdown = null;
          this.tupleBuffers = {};
        });
      },

      destroy() {
        document.removeEventListener("click", this._closeOnDocClick, true);
      },

      // ─── widget kind dispatcher (re-uses helper from window.PipelineExplorer)
      kind(leaf) {
        const PE = window.PipelineExplorer;
        return PE && PE.widgetKind ? PE.widgetKind(leaf) : "text";
      },

      val(leaf) {
        return this.$store.explorer.getKnob(leaf.path);
      },

      setVal(leaf, v) {
        this.$store.explorer.setKnob(leaf.path, v);
      },

      hoverIn(leaf) {
        this.$store.explorer.setHoverKnob(leaf.path);
        window.dispatchEvent(
          new CustomEvent("explorer:knob-hover", {
            detail: { path: leaf.path, node_id: leaf.node_id },
          }),
        );
      },

      hoverOut() {
        this.$store.explorer.setHoverKnob(null);
      },

      // ─── slider helpers
      fillPct(leaf) {
        const v = this.val(leaf);
        if (v === null || v === undefined) return 0;
        const lo = Number(leaf.min);
        const hi = Number(leaf.max);
        if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi === lo) return 0;
        return Math.max(0, Math.min(100, ((Number(v) - lo) / (hi - lo)) * 100));
      },

      sliderStep(leaf) {
        if (leaf.type === "int") return 1;
        const range = (leaf.max ?? 1) - (leaf.min ?? 0);
        return range < 1 ? 0.001 : 0.01;
      },

      readout(leaf) {
        const PE = window.PipelineExplorer;
        return PE && PE.formatNumber
          ? PE.formatNumber(this.val(leaf), leaf)
          : String(this.val(leaf) ?? "—");
      },

      // ─── dropdown
      options(leaf) {
        const PE = window.PipelineExplorer;
        return PE && PE.enumOptions
          ? PE.enumOptions(leaf, this.$store.explorer.schema)
          : leaf.enum || [];
      },

      isOpen(key) {
        return this.openDropdown === key;
      },

      toggleDropdown(key) {
        this.openDropdown = this.openDropdown === key ? null : key;
      },

      pickOption(leaf, opt) {
        this.setVal(leaf, opt);
        this.openDropdown = null;
      },

      // ─── toggle
      flipToggle(leaf) {
        this.setVal(leaf, !this.val(leaf));
      },

      // ─── number (unbounded)
      onNumberInput(leaf, e) {
        const raw = e.target.value;
        if (raw === "" || raw === "-") return;
        const n =
          leaf.type === "int"
            ? Number.parseInt(raw, 10)
            : Number.parseFloat(raw);
        if (Number.isFinite(n)) this.setVal(leaf, n);
      },

      // ─── tuple buffer
      tupleRaw(leaf) {
        const buf = this.tupleBuffers[leaf.path];
        if (buf !== undefined) return buf.raw;
        const v = this.val(leaf);
        return Array.isArray(v) ? v.join(", ") : "";
      },

      tupleValid(leaf) {
        const buf = this.tupleBuffers[leaf.path];
        return !buf || buf.valid !== false;
      },

      onTupleInput(leaf, e) {
        const raw = e.target.value;
        const PE = window.PipelineExplorer;
        const parsed = PE && PE.parseTuple
          ? PE.parseTuple(raw, leaf.type)
          : { ok: true, value: raw.split(",").map((s) => s.trim()).filter(Boolean) };
        this.tupleBuffers[leaf.path] = { raw, valid: parsed.ok };
        if (parsed.ok) this.setVal(leaf, parsed.value);
      },

      onTupleBlur(leaf) {
        const buf = this.tupleBuffers[leaf.path];
        if (!buf || buf.valid) delete this.tupleBuffers[leaf.path];
      },

      tupleChips(leaf) {
        const v = this.val(leaf);
        return Array.isArray(v) ? v : [];
      },

      // ─── tuple_dataclass items
      items(leaf) {
        const v = this.val(leaf);
        return Array.isArray(v) ? v : [];
      },

      itemKind(field) {
        const PE = window.PipelineExplorer;
        return PE && PE.widgetKind ? PE.widgetKind(field) : "text";
      },

      itemActiveFields(leaf, idx) {
        const item = this.items(leaf)[idx] || {};
        const PE = window.PipelineExplorer;
        return (leaf.item_schema || []).filter((f) =>
          PE && PE.itemDepSatisfied
            ? PE.itemDepSatisfied(f, item, this.$store.explorer.config)
            : true,
        );
      },

      addItem(leaf) {
        const cur = this.items(leaf).slice();
        const PE = window.PipelineExplorer;
        cur.push(PE && PE.itemDefault ? PE.itemDefault(leaf.item_schema) : {});
        this.setVal(leaf, cur);
      },

      removeItem(leaf, idx) {
        const cur = this.items(leaf).slice();
        cur.splice(idx, 1);
        this.setVal(leaf, cur);
      },

      setItemField(leaf, idx, fieldName, value) {
        const cur = this.items(leaf).slice();
        const next = { ...(cur[idx] || {}) };
        next[fieldName] = value;
        cur[idx] = next;
        this.setVal(leaf, cur);
      },

      getItemField(leaf, idx, fieldName) {
        const item = this.items(leaf)[idx] || {};
        return item[fieldName];
      },

      itemFillPct(field, value) {
        const lo = Number(field.min);
        const hi = Number(field.max);
        if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi === lo) return 0;
        return Math.max(0, Math.min(100, ((Number(value) - lo) / (hi - lo)) * 100));
      },

      jsonPreview(leaf) {
        try {
          return JSON.stringify(this.val(leaf), null, 2);
        } catch (_) {
          return "{}";
        }
      },

      sideEffectNote(leaf) {
        if (leaf.path === "modality") {
          const v = this.val(leaf);
          if (v === "text_only")
            return "Runtime forces image.patch_ratio = 0; YAML keeps your value.";
          if (v === "image_only")
            return 'Runtime forces text.composite.profile = "noop"; YAML keeps your value.';
        }
        return null;
      },
    }));
  });

  // Expose for the harness / tests.
  window.PipelineExplorer = window.PipelineExplorer || {};
  Object.assign(window.PipelineExplorer, {
    STEP_ORDER,
    CHIP_LAYOUT,
    resolveGroupPaths,
    resolvedGroups,
  });
})();
