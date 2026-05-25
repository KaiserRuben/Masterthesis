/*
 * widgets.js · phase 4
 *
 * Knob widgets + dependency engine for the centre detail panel.
 *
 * Auto-generates one widget per leaf belonging to the currently-focused
 * pipeline node. Reads schema + config from $store.explorer (populated
 * by store.js), filters by pipeline and dependsOn, and renders Alpine
 * templates that two-way bind through setKnob / getKnob.
 *
 * Visual style follows the design tokens in styles.css. Sliders are
 * native <input type="range"> with CSS skin (no third-party widget);
 * dropdowns are button + popover (not native <select>); toggles are a
 * sliding pill driven by Motion One.
 *
 * Cross-link: hovering a knob populates $store.explorer.hoverKnob and
 * dispatches `explorer:knob-hover` so phase 3 (canvas) and phase 6
 * (YAML editor) can light up their counterpart elements.
 */

(function () {
  "use strict";

  // ───────────────────────────────────────────────────────────────────
  // Widget kind dispatcher · leaf.type → component name
  // ───────────────────────────────────────────────────────────────────
  // We coalesce all enum_* into 'enum'; str + enum routes to enum too.
  function widgetKind(leaf) {
    if (!leaf) return "text";
    const t = leaf.type || "";
    if (t === "bool") return "toggle";
    if (t === "tuple_dataclass") return "items"; // tiers / strategies
    if (t === "dict") return "json"; // text.composite.overrides
    if (
      t === "tuple_int" ||
      t === "tuple_float" ||
      t === "tuple_str" ||
      t === "tuple_path" ||
      t === "tuple_dict" ||
      t === "set_str"
    ) {
      return "tuple";
    }
    if (t.startsWith("enum_") || (Array.isArray(leaf.enum) && leaf.enum.length)) {
      return "enum";
    }
    if (t === "int") {
      return hasBounds(leaf) ? "slider_int" : "number_int";
    }
    if (t === "float") {
      return hasBounds(leaf) ? "slider_float" : "number_float";
    }
    if (t === "path") return "path";
    return "text"; // str fallback
  }

  function hasBounds(leaf) {
    return (
      leaf.min !== null &&
      leaf.min !== undefined &&
      leaf.max !== null &&
      leaf.max !== undefined
    );
  }

  // ───────────────────────────────────────────────────────────────────
  // Numeric step heuristic for float sliders.
  // Range < 1 → 0.001 step (sub-percent control); else → 0.01.
  // ───────────────────────────────────────────────────────────────────
  function floatStep(leaf) {
    const range = (leaf.max ?? 1) - (leaf.min ?? 0);
    return range < 1 ? 0.001 : 0.01;
  }

  // ───────────────────────────────────────────────────────────────────
  // Tuple parse/format · keep raw string in input until valid, then
  // commit parsed array to store. Trim whitespace, drop empties.
  // ───────────────────────────────────────────────────────────────────
  function formatTuple(value) {
    if (!Array.isArray(value)) return "";
    return value.join(", ");
  }

  function parseTuple(raw, kind) {
    const parts = String(raw)
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (kind === "tuple_int") {
      const out = [];
      for (const p of parts) {
        const n = Number.parseInt(p, 10);
        if (!Number.isFinite(n)) return { ok: false };
        out.push(n);
      }
      return { ok: true, value: out };
    }
    if (kind === "tuple_float") {
      const out = [];
      for (const p of parts) {
        const n = Number.parseFloat(p);
        if (!Number.isFinite(n)) return { ok: false };
        out.push(n);
      }
      return { ok: true, value: out };
    }
    // tuple_str / tuple_path / tuple_dict / set_str — accept any text.
    return { ok: true, value: parts };
  }

  // ───────────────────────────────────────────────────────────────────
  // Default factory for a single tuple_dataclass element. Walks the
  // item_schema and returns a defaults object using each field's default
  // (falling back to type-appropriate empties when default is null).
  // ───────────────────────────────────────────────────────────────────
  function itemDefault(itemSchema) {
    const out = {};
    for (const f of itemSchema || []) {
      let v = f.default;
      if (v === null || v === undefined) {
        if (f.type === "int" || f.type === "float") v = f.min ?? 0;
        else if (f.type === "bool") v = false;
        else if (f.type && f.type.startsWith("tuple_")) v = [];
        else if (f.type === "dict") v = {};
        else if (Array.isArray(f.enum) && f.enum.length) v = f.enum[0];
        else v = "";
      } else if (Array.isArray(v) || (typeof v === "object" && v !== null)) {
        v = JSON.parse(JSON.stringify(v));
      }
      out[f.name] = v;
    }
    return out;
  }

  // ───────────────────────────────────────────────────────────────────
  // dependsOn resolver for nested item_schema fields. Checks the local
  // item state first, falls back to the global config (so an item field
  // can depend on a sibling within the item or on a top-level knob).
  // ───────────────────────────────────────────────────────────────────
  function itemDepSatisfied(field, item, globalConfig) {
    const deps = field.dependsOn || {};
    if (!Object.keys(deps).length) return true;
    for (const [depKey, allowed] of Object.entries(deps)) {
      const wanted = Array.isArray(allowed) ? allowed : [allowed];
      const localVal = item ? item[depKey] : undefined;
      if (wanted.includes(localVal)) continue;
      // Fall back to top-level path lookup.
      const parts = depKey.split(".");
      let cur = globalConfig;
      for (const k of parts) {
        if (cur == null) {
          cur = undefined;
          break;
        }
        cur = cur[k];
      }
      if (!wanted.includes(cur)) return false;
    }
    return true;
  }

  // ───────────────────────────────────────────────────────────────────
  // Resolve enum options for a leaf. Falls back to schema.valid_* lists
  // when the leaf doesn't carry its own enum (some schema-generated
  // enum_* types pull their option list from valid_<key>).
  // ───────────────────────────────────────────────────────────────────
  function enumOptions(leaf, schema) {
    if (Array.isArray(leaf.enum) && leaf.enum.length) return leaf.enum;
    // Map common enum_* types → schema.valid_* arrays
    const lookups = {
      enum_strategy: "valid_strategies",
      enum_name: "valid_strategies",
      enum_d_i_primary: "valid_d_i",
      enum_d_o_primary: "valid_d_o",
      enum_dedupe_by: "valid_dedupe_by",
      enum_rank_by: "valid_rank_by",
      enum_flip_policy: "valid_flip_policies",
      enum_flip_preserve_policy: "valid_flip_policies",
      enum_order: "valid_pass_orders",
    };
    const key = lookups[leaf.type];
    if (key && schema && Array.isArray(schema[key])) return schema[key];
    return [];
  }

  // ───────────────────────────────────────────────────────────────────
  // Cross-link hover dispatch. Single-place helper so we keep the
  // event name and detail shape consistent across widgets.
  // ───────────────────────────────────────────────────────────────────
  function emitHover(path, nodeId) {
    window.dispatchEvent(
      new CustomEvent("explorer:knob-hover", {
        detail: { path, node_id: nodeId },
      }),
    );
  }

  // ───────────────────────────────────────────────────────────────────
  // Numeric formatter for slider readouts. Uses leaf.type and step to
  // pick a sensible precision; tabular-nums in CSS handles alignment.
  // ───────────────────────────────────────────────────────────────────
  function formatNumber(value, leaf) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    if (leaf.type === "int") return String(value);
    const step = floatStep(leaf);
    const digits = step < 0.01 ? 3 : 2;
    return Number(value).toFixed(digits);
  }

  // ───────────────────────────────────────────────────────────────────
  // Compute provenance URL for the manual page.
  //
  // Manual pages live in an Obsidian vault (~/Obsidian/Notizen/...),
  // not in git. The old GitHub link (`docs/manual/<page>`) 404s, so by
  // default we don't render a link at all (returns null). The user can
  // opt-in to obsidian:// deep-links via the settings drawer; persisted
  // in localStorage as "pipex.manualTarget" = "none" | "obsidian".
  //
  // Source-file provenance is not handled here — only the manual_page
  // string from the node graph is processed.
  // ───────────────────────────────────────────────────────────────────
  const MANUAL_STORAGE_KEY = "pipex.manualTarget";
  const OBSIDIAN_VAULT = "Notizen";
  const OBSIDIAN_PREFIX = "01 - Active Projects/Master Thesis/Knowledge";

  function manualTargetPref() {
    try { return localStorage.getItem(MANUAL_STORAGE_KEY) || "none"; }
    catch (_) { return "none"; }
  }

  function obsidianUrlFor(page) {
    if (!page) return null;
    const stem = page.replace(/\.md$/i, "");
    const full = `${OBSIDIAN_PREFIX}/${stem}`;
    return `obsidian://open?vault=${encodeURIComponent(OBSIDIAN_VAULT)}` +
           `&file=${encodeURIComponent(full)}`;
  }

  function provenanceUrl(_manifest, manualPage) {
    if (!manualPage) return null;
    const pref = manualTargetPref();
    if (pref === "obsidian") return obsidianUrlFor(manualPage);
    // Default ("none") — render the manual page as plain text. Returning
    // null causes the <a> to be hidden via `x-show="!!provenanceHref"`.
    return null;
  }

  // ───────────────────────────────────────────────────────────────────
  // Alpine component: detailPanel
  // Mounted on the <section id="detail-panel"> root.
  // Exposes the templates' computed lists and runtime helpers.
  // ───────────────────────────────────────────────────────────────────
  function detailPanelComponent() {
    return {
      // Local UI state — open dropdowns, tuple raw buffers.
      openDropdown: null, // dotted path or `${path}#${index}.${field}` for nested
      tupleBuffers: {}, // path → { raw, valid }
      itemEdits: {}, // `${path}#${index}` → ephemeral expand state
      manualTarget: manualTargetPref(), // 'none' | 'obsidian'

      init() {
        // Re-format tuple buffers whenever focus changes so we stay in
        // sync with config defaults for the new node.
        this.$watch("$store.explorer.focusNode", () => {
          this.openDropdown = null;
          this.tupleBuffers = {};
          this.itemEdits = {};
        });
        // Close dropdowns on outside click.
        this._closeOnDocClick = (e) => {
          if (!this.openDropdown) return;
          if (!e.target.closest("[data-dropdown-root]")) {
            this.openDropdown = null;
          }
        };
        document.addEventListener("click", this._closeOnDocClick, true);

        // Listen for settings-drawer manual-target changes. The inline
        // script in index.html persists to localStorage; we mirror that
        // into this reactive field so `provenanceHref` re-evaluates.
        this._onManualChanged = () => { this.manualTarget = manualTargetPref(); };
        window.addEventListener("pipex:manual-target-changed", this._onManualChanged);
      },

      destroy() {
        document.removeEventListener("click", this._closeOnDocClick, true);
        window.removeEventListener("pipex:manual-target-changed", this._onManualChanged);
      },

      // ─── Derived: focused-node metadata ─────────────────────────
      get focusNode() {
        return this.$store.explorer.focusNode;
      },

      get currentNode() {
        const id = this.focusNode;
        return id ? this.$store.explorer.graph?.nodes?.[id] : null;
      },

      get currentLabel() {
        return this.currentNode?.label || "Pipeline";
      },

      get currentSummary() {
        return this.currentNode?.summary || "";
      },

      get currentManualPage() {
        // Returns the bare manual filename (e.g. "05-Subsystem-Manipulator.md")
        // for display. The path no longer prepends `docs/manual/` because
        // manuals live in an Obsidian vault, not the repo's docs tree.
        return this.currentNode?.manual_page || null;
      },

      get provenanceHref() {
        // Read `manualTarget` so this getter participates in Alpine
        // reactivity when the setting changes.
        const _trigger = this.manualTarget;
        return provenanceUrl(this.$store.explorer.manifest, this.currentManualPage);
      },

      // Alias for the named contract some downstream phases expect.
      get currentProvenance() {
        return this.provenanceHref;
      },

      // ─── Derived: active knob lists per tier ────────────────────
      get activeKnobs() {
        if (!this.focusNode) return [];
        return this.$store.explorer.activeKnobs(this.focusNode);
      },

      get heroKnobs() {
        return this.activeKnobs.filter((l) => l.tier === "hero");
      },

      get standardKnobs() {
        return this.activeKnobs.filter((l) => l.tier === "standard");
      },

      get advancedKnobs() {
        return this.activeKnobs.filter((l) => l.tier === "advanced");
      },

      get knobCount() {
        return this.activeKnobs.length;
      },

      // ─── Per-leaf helpers (template-callable) ───────────────────
      kind(leaf) {
        return widgetKind(leaf);
      },

      val(leaf) {
        return this.$store.explorer.getKnob(leaf.path);
      },

      setVal(leaf, v) {
        this.$store.explorer.setKnob(leaf.path, v);
      },

      hoverIn(leaf) {
        this.$store.explorer.setHoverKnob(leaf.path);
        emitHover(leaf.path, leaf.node_id);
      },

      hoverOut() {
        this.$store.explorer.setHoverKnob(null);
      },

      // ─── Slider: percent fill for the visual track ─────────────
      fillPct(leaf) {
        const v = this.val(leaf);
        if (v === null || v === undefined) return 0;
        const lo = Number(leaf.min);
        const hi = Number(leaf.max);
        if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi === lo) return 0;
        return Math.max(0, Math.min(100, ((Number(v) - lo) / (hi - lo)) * 100));
      },

      sliderStep(leaf) {
        return leaf.type === "int" ? 1 : floatStep(leaf);
      },

      readout(leaf) {
        return formatNumber(this.val(leaf), leaf);
      },

      // ─── Dropdown ───────────────────────────────────────────────
      options(leaf) {
        return enumOptions(leaf, this.$store.explorer.schema);
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

      // ─── Toggle ─────────────────────────────────────────────────
      flipToggle(leaf) {
        this.setVal(leaf, !this.val(leaf));
      },

      // ─── Number input (unbounded) ───────────────────────────────
      onNumberInput(leaf, e) {
        const raw = e.target.value;
        if (raw === "" || raw === "-") return;
        const n = leaf.type === "int" ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
        if (Number.isFinite(n)) this.setVal(leaf, n);
      },

      // ─── Tuple buffer (raw text → parsed array) ─────────────────
      tupleRaw(leaf) {
        const buf = this.tupleBuffers[leaf.path];
        if (buf !== undefined) return buf.raw;
        return formatTuple(this.val(leaf));
      },

      tupleValid(leaf) {
        const buf = this.tupleBuffers[leaf.path];
        return !buf || buf.valid !== false;
      },

      onTupleInput(leaf, e) {
        const raw = e.target.value;
        const parsed = parseTuple(raw, leaf.type);
        this.tupleBuffers[leaf.path] = { raw, valid: parsed.ok };
        if (parsed.ok) this.setVal(leaf, parsed.value);
      },

      onTupleBlur(leaf) {
        // Re-format from store on successful parse; keep raw on error.
        const buf = this.tupleBuffers[leaf.path];
        if (!buf || buf.valid) {
          delete this.tupleBuffers[leaf.path];
        }
      },

      tupleChips(leaf) {
        const v = this.val(leaf);
        return Array.isArray(v) ? v : [];
      },

      // ─── tuple_dataclass (tiers, strategies) ────────────────────
      items(leaf) {
        const v = this.val(leaf);
        return Array.isArray(v) ? v : [];
      },

      itemFields(leaf) {
        return leaf.item_schema || [];
      },

      itemKind(field) {
        return widgetKind(field);
      },

      itemActiveFields(leaf, itemIndex) {
        const item = this.items(leaf)[itemIndex] || {};
        return (leaf.item_schema || []).filter((f) =>
          itemDepSatisfied(f, item, this.$store.explorer.config),
        );
      },

      addItem(leaf) {
        const cur = this.items(leaf).slice();
        cur.push(itemDefault(leaf.item_schema));
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

      // ─── JSON / dict preview (read-only fallback) ───────────────
      jsonPreview(leaf) {
        try {
          return JSON.stringify(this.val(leaf), null, 2);
        } catch (_) {
          return "{}";
        }
      },

      // ─── Modality-side-effect note (image.patch_ratio, text profile)
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

      // ─── Copy provenance path to clipboard ──────────────────────
      copyProvenance() {
        const p = this.currentManualPage;
        if (!p || !navigator.clipboard) return;
        navigator.clipboard.writeText(p).catch(() => {});
      },
    };
  }

  // ───────────────────────────────────────────────────────────────────
  // Register Alpine component + global helpers
  // ───────────────────────────────────────────────────────────────────
  document.addEventListener("alpine:init", () => {
    window.Alpine.data("detailPanel", detailPanelComponent);
  });

  window.PipelineExplorer = window.PipelineExplorer || {};
  Object.assign(window.PipelineExplorer, {
    widgetKind,
    formatTuple,
    parseTuple,
    itemDefault,
    itemDepSatisfied,
    enumOptions,
    formatNumber,
    provenanceUrl,
  });
})();
