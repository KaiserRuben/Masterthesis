/*
 * polish.js — phase 9 final polish.
 *
 * Layers on top of phases 1–8 without modifying their files:
 *   Layer 1 — entry choreography (top bar → canvas stagger → panels → footer)
 *   Layer 2 — KaTeX rendering for <span class="kx">…</span> + .kx-block
 *   Layer 3 — provenance footer wiring + settings drawer
 *   Layer 4 — keyboard navigation, jump shortcuts, help overlay, ARIA
 *   Layer 5 — theme crossfade overlay + body transition class
 *
 * Conventions:
 *   - Motion One via window.Motion.animate(el, keyframes, opts).
 *   - LocalStorage keys under "pipex.*".
 *   - Reduced motion: respect prefers-reduced-motion AND the user
 *     toggle in the settings drawer.
 *
 * Kept under ~500 lines. No external dependencies.
 */
(function () {
  "use strict";

  // ── Shared state ────────────────────────────────────────────────
  let _hasEntered = false;
  const _mql = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;
  const _osReducedMotion = !!(_mql && _mql.matches);

  // User-overridable. Settings drawer writes to this and persists.
  const STORAGE = {
    theme: "pipex.theme",            // 'auto' | 'dark' | 'light'
    prov:  "pipex.provLink",         // 'text' | 'github' | 'vscode'
    rm:    "pipex.reducedMotion",    // '1' | '0' | '' (= use OS)
  };

  function reducedMotion() {
    const ovr = localStorage.getItem(STORAGE.rm);
    if (ovr === "1") return true;
    if (ovr === "0") return false;
    return _osReducedMotion;
  }

  /** Safe Motion One wrapper. Falls back to inline style mutation if RM. */
  function animate(el, keyframes, opts) {
    if (!el) return null;
    if (reducedMotion()) {
      // Skip to final state. Pick the last value from each keyframe array.
      Object.entries(keyframes || {}).forEach(([prop, vals]) => {
        const final = Array.isArray(vals) ? vals[vals.length - 1] : vals;
        if (final !== undefined) el.style[prop] = final;
      });
      return null;
    }
    const motion = window.Motion;
    if (!motion || !motion.animate) return null;
    try { return motion.animate(el, keyframes, opts || {}); }
    catch (_) { return null; }
  }

  // ════════════════════════════════════════════════════════════════
  // Layer 1 — entry timeline
  // ════════════════════════════════════════════════════════════════
  function runEntryTimeline() {
    if (_hasEntered) return;
    _hasEntered = true;

    // Reduced motion: snap every entry-target into its final state.
    if (reducedMotion()) {
      document
        .querySelectorAll(
          "#topbar, #footer, #detail-panel, #yaml-editor-panel, .playground, .node",
        )
        .forEach((el) => {
          el.style.opacity = "1";
          el.style.transform = "none";
        });
      return;
    }

    // 1. Top bar fade.
    const topbar = document.getElementById("topbar");
    if (topbar) animate(topbar, { opacity: [0, 1] }, { duration: 0.3, delay: 0 });

    // 2. Canvas nodes stagger-in from centre outward.
    //    Distance is measured in CSS pixels from the canvas centre; the
    //    delay is 8ms per pixel-of-radius / 80 (i.e. ~6px → ~50ms).
    const canvas = document.getElementById("pipeline-canvas");
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const nodes = Array.from(canvas.querySelectorAll(".node, .cluster"));
      nodes.forEach((el) => {
        const r = el.getBoundingClientRect();
        const ex = r.left + r.width / 2 - cx;
        const ey = r.top + r.height / 2 - cy;
        const dist = Math.hypot(ex, ey);
        const delay = Math.min(0.6, (dist * 8) / 1000); // ms→s, cap 600ms
        animate(
          el,
          { opacity: [0, 1], transform: ["scale(0.92)", "scale(1)"] },
          { duration: 0.24, delay, easing: "cubic-bezier(0.2, 0, 0, 1)" },
        );
      });
    }

    // 3. Detail panel — slide-in from right.
    const detail = document.getElementById("detail-panel");
    if (detail) {
      animate(
        detail,
        { opacity: [0, 1], transform: ["translateX(12px)", "translateX(0px)"] },
        { duration: 0.36, delay: 0.2, easing: "cubic-bezier(0.32, 0.72, 0, 1)" },
      );
    }

    // 4. YAML editor — slide-in from right, 80ms after detail.
    const yaml = document.getElementById("yaml-editor-panel");
    if (yaml) {
      animate(
        yaml,
        { opacity: [0, 1], transform: ["translateX(12px)", "translateX(0px)"] },
        { duration: 0.36, delay: 0.28, easing: "cubic-bezier(0.32, 0.72, 0, 1)" },
      );
    }

    // 5. Playground card — fade + scale.
    const playground = document.querySelector(".playground");
    if (playground) {
      animate(
        playground,
        { opacity: [0, 1], transform: ["scale(0.98)", "scale(1)"] },
        { duration: 0.4, delay: 0.6, easing: "cubic-bezier(0.2, 0, 0, 1)" },
      );
    }

    // 6. Footer fade.
    const footer = document.getElementById("footer");
    if (footer) animate(footer, { opacity: [0, 1] }, { duration: 0.2, delay: 0.7 });

    // Mode-swap animation: poll Alpine's store.mode, animate on change.
    let _lastMode = null;
    (function syncMode() {
      const store = window.Alpine && window.Alpine.store("explorer");
      const mode = store ? store.mode : null;
      if (mode && mode !== _lastMode) {
        if (_lastMode) {
          const out = document.getElementById(_lastMode === "canvas" ? "canvas-mode" : "step-mode");
          const into = document.getElementById(mode === "canvas" ? "canvas-mode" : "step-mode");
          if (out) animate(out, { opacity: [1, 0], transform: ["translateY(0px)", "translateY(-8px)"] }, { duration: 0.2 });
          if (into) setTimeout(() => animate(into,
            { opacity: [0, 1], transform: ["translateY(8px)", "translateY(0px)"] },
            { duration: 0.24, easing: "cubic-bezier(0.32, 0.72, 0, 1)" }), 60);
        }
        _lastMode = mode;
      }
      requestAnimationFrame(syncMode);
    })();
  }

  // ════════════════════════════════════════════════════════════════
  // Layer 2 — KaTeX rendering
  // ════════════════════════════════════════════════════════════════
  function renderKaTeX(scope) {
    if (!window.katex || !window.katex.render) return;
    const root = scope || document;
    root.querySelectorAll("span.kx:not(.kx-rendered)").forEach((el) => {
      try {
        window.katex.render(el.textContent.trim(), el, {
          throwOnError: false,
          displayMode: false,
        });
        el.classList.add("kx-rendered");
      } catch (_) { /* leave plain text fallback */ }
    });
    root.querySelectorAll("div.kx-block:not(.kx-rendered)").forEach((el) => {
      try {
        window.katex.render(el.textContent.trim(), el, {
          throwOnError: false,
          displayMode: true,
        });
        el.classList.add("kx-rendered");
      } catch (_) { /* leave plain text fallback */ }
    });
  }

  // ════════════════════════════════════════════════════════════════
  // Layer 3 — provenance footer + settings drawer
  // ════════════════════════════════════════════════════════════════
  function initFooter() {
    // Footer text is already wired via Alpine x-text bindings in index.html.
    // Nothing to do here unless we want to add deep-link click metrics.
    // We DO refine the provenance link target on a settings-drawer change.
    applyProvenanceMode(localStorage.getItem(STORAGE.prov) || "github");
  }

  /** Toggle behaviour of every <a> with class .provenance > a based on
   *  the user setting. text → strip href + role; github → keep current
   *  href; vscode → rewrite to vscode://file/<absolute>. */
  function applyProvenanceMode(mode) {
    document.querySelectorAll(".provenance a").forEach((a) => {
      const orig = a.dataset.provOrigHref || a.getAttribute("href") || "";
      if (!a.dataset.provOrigHref) a.dataset.provOrigHref = orig;
      if (mode === "text") {
        a.removeAttribute("href");
        a.setAttribute("aria-disabled", "true");
        a.style.pointerEvents = "none";
        a.style.opacity = "0.55";
      } else if (mode === "vscode") {
        // Translate the github blob URL into a local vscode:// link.
        // orig looks like: https://github.com/owner/repo/blob/<commit>/<path>
        const m = orig.match(/\/blob\/[^/]+\/(.+)$/);
        const rel = m ? m[1] : "";
        const root = window.Alpine?.store?.("explorer")?.manifest?.repo_local_root || "";
        // vscode:// requires a leading slash on the absolute path; root already
        // starts with one (e.g. "/Users/foo/proj") so concatenation is direct.
        const abs = root && rel ? `${root}/${rel}` : "";
        a.setAttribute(
          "href",
          abs ? `vscode://file${abs}` : orig,
        );
        a.removeAttribute("aria-disabled");
        a.style.pointerEvents = "";
        a.style.opacity = "";
      } else {
        a.setAttribute("href", orig);
        a.removeAttribute("aria-disabled");
        a.style.pointerEvents = "";
        a.style.opacity = "";
      }
    });
  }

  function initSettingsDrawer() {
    const gear = document.getElementById("settings-gear");
    const drawer = document.getElementById("settings-drawer");
    const backdrop = document.getElementById("settings-backdrop");
    const closeBtn = document.getElementById("settings-drawer-close");
    if (!gear || !drawer || !backdrop) return;

    function open() {
      drawer.hidden = false;
      // Force a reflow so the transition picks up.
      void drawer.offsetWidth;
      drawer.classList.add("is-open");
      backdrop.classList.add("is-open");
      gear.setAttribute("aria-expanded", "true");
      setTimeout(() => {
        const first = drawer.querySelector("input, button");
        if (first) first.focus();
      }, 80);
    }
    function close() {
      drawer.classList.remove("is-open");
      backdrop.classList.remove("is-open");
      gear.setAttribute("aria-expanded", "false");
      setTimeout(() => { drawer.hidden = true; }, 320);
    }
    function toggle() {
      if (drawer.classList.contains("is-open")) close(); else open();
    }
    gear.addEventListener("click", toggle);
    closeBtn?.addEventListener("click", close);
    backdrop.addEventListener("click", close);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && drawer.classList.contains("is-open")) close();
    });

    // Initial selection state from storage.
    const themePref = localStorage.getItem(STORAGE.theme) || "auto";
    const provPref = localStorage.getItem(STORAGE.prov) || "github";
    const rmPref = localStorage.getItem(STORAGE.rm);
    drawer.querySelectorAll('input[name="setting-theme"]').forEach((r) => {
      r.checked = r.value === themePref;
      r.addEventListener("change", () => {
        localStorage.setItem(STORAGE.theme, r.value);
        applyThemePref(r.value);
      });
    });
    drawer.querySelectorAll('input[name="setting-prov"]').forEach((r) => {
      r.checked = r.value === provPref;
      r.addEventListener("change", () => {
        localStorage.setItem(STORAGE.prov, r.value);
        applyProvenanceMode(r.value);
      });
    });
    const rmCb = document.getElementById("setting-reduced-motion");
    if (rmCb) {
      rmCb.checked = rmPref === "1" || (rmPref === null && _osReducedMotion);
      rmCb.addEventListener("change", () => {
        localStorage.setItem(STORAGE.rm, rmCb.checked ? "1" : "0");
        document.documentElement.classList.toggle("reduced-motion", rmCb.checked);
      });
      if (rmCb.checked) document.documentElement.classList.add("reduced-motion");
    }

    // Apply theme pref on boot.
    applyThemePref(themePref);
  }

  function applyThemePref(pref) {
    const store = window.Alpine && window.Alpine.store("explorer");
    if (!store) return;
    let want;
    if (pref === "auto") {
      const dark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
      want = dark ? "dark" : "light";
    } else { want = pref; }
    if (store.theme !== want) {
      store.theme = want;
      document.documentElement.setAttribute("data-theme", want);
    }
  }

  // ════════════════════════════════════════════════════════════════
  // Layer 4 — keyboard navigation
  // ════════════════════════════════════════════════════════════════
  // Canonical visit order (top-left → bottom-right). The list is filtered
  // at runtime against the active pipeline's visible-node set.
  const VISIT_ORDER = [
    "config", "sut", "seeds",
    "manipulator_image", "manipulator_text", "manipulator_vlm",
    "pdq_stage1", "pdq_metric", "pdq_stage2",
    "objectives", "optimizer", "artifacts",
  ];
  const JUMP_KEYS = {
    o: "objectives",   p: "pdq_metric", m: "manipulator_image",
    s: "seeds",        u: "sut",        c: "optimizer",
    a: "artifacts",
  };

  function getStore() { return window.Alpine && window.Alpine.store("explorer"); }

  function visibleNodeIds() {
    const store = getStore();
    if (!store || !store.graph) return [];
    const set = store.pipeline === "pdq"
      ? ["config", "seeds", "manipulator_image", "manipulator_text",
         "manipulator_vlm", "sut", "pdq_stage1", "pdq_stage2", "pdq_metric", "artifacts"]
      : ["config", "seeds", "manipulator_image", "manipulator_text",
         "manipulator_vlm", "sut", "objectives", "optimizer", "artifacts"];
    return VISIT_ORDER.filter((id) => set.includes(id));
  }

  function focusNodeEl(nodeId) {
    if (!nodeId) return;
    const store = getStore();
    if (store) store.setFocus(nodeId);
    // Move browser focus to the matching DOM node for visible feedback.
    const canvas = document.getElementById("pipeline-canvas");
    if (!canvas) return;
    const el = canvas.querySelector(`[aria-label*="Focus" i][aria-label$="${labelFor(nodeId)}"]`)
      || canvas.querySelectorAll(".node")[visibleNodeIds().indexOf(nodeId)];
    el?.focus?.();
  }

  function labelFor(id) {
    const map = {
      config: "Config", seeds: "Seeds",
      manipulator_image: "Image manipulator",
      manipulator_text:  "Text manipulator",
      manipulator_vlm:   "VLM bridge",
      sut: "SUT", objectives: "Objectives", optimizer: "Optimizer",
      pdq_stage1: "PDQ Stage 1", pdq_stage2: "PDQ Stage 2",
      pdq_metric: "PDQ metric", artifacts: "Artifacts",
    };
    return map[id] || id;
  }

  /** Pick the spatial neighbour in a given direction by NODE_LAYOUT.
   *  `_depth` guards against runaway recursion in wrap-around fallback. */
  function nearestInDirection(fromId, dir, _depth) {
    const LAYOUT = {
      config:{x:100,y:92}, seeds:{x:130,y:280},
      manipulator_image:{x:400,y:232}, manipulator_text:{x:400,y:330},
      manipulator_vlm:{x:555,y:290}, sut:{x:695,y:154},
      objectives:{x:580,y:408}, optimizer:{x:250,y:422},
      pdq_stage1:{x:615,y:350}, pdq_stage2:{x:615,y:470},
      pdq_metric:{x:440,y:470}, artifacts:{x:100,y:538},
    };
    const visible = new Set(visibleNodeIds());
    const here = LAYOUT[fromId];
    if (!here) return null;
    let best = null, bestScore = Infinity;
    for (const id of Object.keys(LAYOUT)) {
      if (id === fromId || !visible.has(id)) continue;
      const p = LAYOUT[id];
      const dx = p.x - here.x, dy = p.y - here.y;
      if (dir === "left"  && dx >= -10) continue;
      if (dir === "right" && dx <= 10)  continue;
      if (dir === "up"    && dy >= -10) continue;
      if (dir === "down"  && dy <= 10)  continue;
      const along = dir === "left" || dir === "right" ? Math.abs(dx) : Math.abs(dy);
      const cross = dir === "left" || dir === "right" ? Math.abs(dy) : Math.abs(dx);
      const score = along + cross * 1.6;
      if (score < bestScore) { bestScore = score; best = id; }
    }
    if (!best && (_depth || 0) < 1) {
      const opp = { left: "right", right: "left", up: "down", down: "up" }[dir];
      best = nearestInDirection(fromId, opp, (_depth || 0) + 1) || fromId;
    }
    return best;
  }

  function initKeyboard() {
    // ─── Chord state for `g`-leader jumps. ───────────────────────
    let chord = null;          // 'g' | null
    let chordExpiry = 0;
    function leaderActive() { return chord === "g" && performance.now() < chordExpiry; }

    document.addEventListener("keydown", (e) => {
      // Ignore when focus sits in a text input / textarea — except for
      // the very specific shortcuts that target those (e.g. Escape).
      const t = e.target;
      const inField = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      const store = getStore();

      // ── Escape: clear focus / close overlays ──
      if (e.key === "Escape") {
        const overlay = document.getElementById("kbd-overlay");
        if (overlay && overlay.classList.contains("is-open")) {
          closeKbdOverlay();
          return;
        }
        if (store && store.focusNode && !inField) {
          store.setFocus(null);
          return;
        }
      }

      if (inField) return; // bail on remaining shortcuts while typing

      // ── `?` opens help overlay ──
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        openKbdOverlay();
        return;
      }
      // ── `t` toggles theme ──
      if (e.key === "t" && !e.metaKey && !e.ctrlKey && !leaderActive()) {
        e.preventDefault();
        store?.toggleTheme?.();
        return;
      }
      // ── `/` focuses YAML editor ──
      if (e.key === "/") {
        const ta = document.getElementById("yaml-input");
        if (ta) { e.preventDefault(); ta.focus(); return; }
      }
      // ── Step mode digits 1–6 ──
      if (store && store.mode === "step" && /^[1-6]$/.test(e.key)) {
        const n = parseInt(e.key, 10);
        store.step = n;
        window.dispatchEvent(new CustomEvent("explorer:step-changed", { detail: n }));
        return;
      }

      // ── `g`-leader chord ──
      if (e.key === "g" && !leaderActive()) {
        chord = "g";
        chordExpiry = performance.now() + 300;
        return;
      }
      if (leaderActive() && JUMP_KEYS[e.key]) {
        const id = JUMP_KEYS[e.key];
        e.preventDefault();
        chord = null;
        focusNodeEl(id);
        return;
      }
      // Any other key cancels a pending chord.
      if (leaderActive()) chord = null;

      // ── Arrow keys within canvas ──
      const onCanvas = t && t.closest && t.closest(".pipeline-canvas");
      const dir = { ArrowLeft: "left", ArrowRight: "right",
        ArrowUp: "up", ArrowDown: "down" }[e.key];
      if (onCanvas && dir) {
        e.preventDefault();
        // Resolve current node from focused element's aria-label, else store.
        const m = (t.getAttribute("aria-label") || "").match(/Focus (.+)$/i);
        const lookup = {
          Config: "config", Seeds: "seeds",
          "Image manipulator": "manipulator_image",
          "Text manipulator": "manipulator_text",
          "VLM bridge": "manipulator_vlm", SUT: "sut",
          Objectives: "objectives", Optimizer: "optimizer",
          "PDQ Stage 1": "pdq_stage1", "PDQ Stage 2": "pdq_stage2",
          "PDQ metric": "pdq_metric", Artifacts: "artifacts",
        };
        const current = (m && lookup[m[1]]) || store?.focusNode || "manipulator_image";
        const next = nearestInDirection(current, dir);
        if (next) focusNodeEl(next);
      }
    });

    // ── Skip-link helper: when activated, drop focus inside <main>.
    const skip = document.querySelector(".skip-link");
    skip?.addEventListener("click", () => {
      const main = document.getElementById("main");
      if (main) { main.setAttribute("tabindex", "-1"); main.focus(); }
    });

    // ── Help overlay close handlers.
    const overlay = document.getElementById("kbd-overlay");
    overlay?.querySelector("[data-kbd-backdrop]")?.addEventListener("click", closeKbdOverlay);
    document.getElementById("kbd-overlay-close")?.addEventListener("click", closeKbdOverlay);

    // ── Maintain aria-current on the currently focused canvas node.
    window.addEventListener("explorer:focus-changed", () => {
      const canvas = document.getElementById("pipeline-canvas");
      const store = getStore();
      if (!canvas || !store) return;
      canvas.querySelectorAll(".node").forEach((n) => n.removeAttribute("aria-current"));
      const idx = visibleNodeIds().indexOf(store.focusNode);
      const el = canvas.querySelectorAll(".node")[idx];
      if (el) el.setAttribute("aria-current", "true");
    });
  }

  function openKbdOverlay() {
    const o = document.getElementById("kbd-overlay");
    if (!o) return;
    o.hidden = false;
    void o.offsetWidth;
    o.classList.add("is-open");
    document.getElementById("kbd-overlay-close")?.focus();
  }
  function closeKbdOverlay() {
    const o = document.getElementById("kbd-overlay");
    if (!o) return;
    o.classList.remove("is-open");
    setTimeout(() => { o.hidden = true; }, 220);
  }

  // ════════════════════════════════════════════════════════════════
  // Layer 5 — theme crossfade
  // ════════════════════════════════════════════════════════════════
  function initThemeCrossfade() {
    const store = getStore();
    if (!store) return;
    const overlay = document.getElementById("theme-overlay");
    if (!overlay) return;
    const root = document.documentElement;

    // Wrap store.toggleTheme so every invocation triggers the crossfade.
    const original = store.toggleTheme.bind(store);
    store.toggleTheme = function () {
      if (reducedMotion()) { original(); return; }
      const next = store.theme === "dark" ? "light" : "dark";
      // Phase 1: fade overlay in.
      overlay.style.background = next === "dark" ? "#0a0b0d" : "#f6f3ec";
      root.classList.add("theme-transitioning");
      animate(overlay, { opacity: [0, 0.4] }, { duration: 0.16, easing: "ease-out" });
      // Phase 2 (midpoint): flip theme attribute.
      setTimeout(() => {
        original();
        // Phase 3: fade overlay back out.
        animate(overlay, { opacity: [0.4, 0] }, { duration: 0.16, easing: "ease-in" });
        setTimeout(() => {
          root.classList.remove("theme-transitioning");
          overlay.style.opacity = "0";
        }, 180);
      }, 160);
    };
  }

  // ════════════════════════════════════════════════════════════════
  // Boot
  // ════════════════════════════════════════════════════════════════
  function bootPolish() {
    initSettingsDrawer();     // safe before ready (reads localStorage)
    initFooter();
    initKeyboard();
  }

  // Some layers depend on Alpine being live; wait for explorer:ready.
  window.addEventListener("explorer:ready", () => {
    // Now that the store exists, apply the persisted theme preference.
    applyThemePref(localStorage.getItem(STORAGE.theme) || "auto");
    runEntryTimeline();
    renderKaTeX(document);
    initThemeCrossfade();
    // Alpine renders the detail panel just after ready fires. Give it a
    // tick, then apply provenance mode so the footer + detail-panel
    // links honour the persisted setting on first paint.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      applyProvenanceMode(localStorage.getItem(STORAGE.prov) || "github");
    }));
  });

  // Re-render KaTeX whenever the focused node changes (markers in the
  // detail panel toggle visibility, and Alpine may rebuild subtrees).
  window.addEventListener("explorer:focus-changed", () => {
    requestAnimationFrame(() => {
      const dp = document.getElementById("detail-panel");
      renderKaTeX(dp || document);
      // Also re-apply provenance link mode after Alpine rebuilds the line.
      applyProvenanceMode(localStorage.getItem(STORAGE.prov) || "github");
    });
  });

  // Drawer / keyboard / footer init runs after DOMContentLoaded, even
  // before explorer:ready (Alpine wires those independently).
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootPolish);
  } else {
    bootPolish();
  }
})();
