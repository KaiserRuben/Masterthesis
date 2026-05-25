/*
 * Pipeline canvas controller: pipeline-explorer/assets/canvas.js
 *
 * Phase 3b — programmatic SVG node + edge rendering.
 *
 * The canvas is one SVG layer: nodes are <g> groups, edges are
 * <path> elements, particles are <circle> elements that animate along
 * the forward edges via <animateMotion>. There are no absolutely-
 * positioned HTML node cards anymore — every visible node lives in
 * the dynamic SVG and shares one coordinate system with the edges.
 *
 * Node footprint: 148 × 84 px in viewBox coords, rounded rect.
 *
 * ViewBox: 0..1400 × 0..480 (three horizontal tiers at y ≈ 90/240/390).
 *
 * Public DOM contract:
 *   - <g id="dyn-nodes"> receives one <g class="cnode" data-node-id=…>
 *     per visible node. Each cnode contains the rect, eyebrow, title,
 *     and optional bespoke viz illustration.
 *   - <g id="dyn-edges"> receives one <path class="edge-path" …> per
 *     active edge, anchored on the from/to node boundaries.
 *   - .particle--flow circles live directly on the SVG root and are
 *     reused across pipeline switches.
 */

(function () {
  "use strict";

  // ───────────────────────────────────────────────────────────────
  // Layout constants
  // ───────────────────────────────────────────────────────────────

  // SVG viewBox extents; must match index.html canvas SVG viewBox attr.
  const VIEW_W = 1400;
  const VIEW_H = 480;

  // Uniform node footprint in viewBox coords. The illustration
  // variants share width/height so the diagram reads as a flow.
  const NODE_W = 148;
  const NODE_H = 84;
  const NODE_RX = 10;

  // Per-node layout: x/y is the node CENTRE in viewBox coords. The
  // optional `viz` field selects a bespoke illustration drawn inside
  // the box. `pipeline` controls visibility (a node may belong to
  // both pipelines or to just one).
  const NODE_LAYOUT = {
    // Tier 1 — main forward flow (left → right):
    config:            { x:   84, y: 100, pipeline: "both",         eyebrow: "config",          title: "Experiment" },
    seeds:             { x:  254, y: 100, pipeline: "both",         eyebrow: "seeds",           title: "Seed pool",      viz: "dotcluster" },
    manipulator_image: { x:  424, y: 100, pipeline: "both",         eyebrow: "manipulator",     title: "Image patches",  viz: "patchgrid" },
    sut:               { x:  748, y: 100, pipeline: "both",         eyebrow: "sut",             title: "VLM under test", viz: "transformer" },
    objectives:        { x:  918, y: 100, pipeline: "evolutionary", eyebrow: "objectives",      title: "Pareto criteria" },
    optimizer:         { x: 1088, y: 100, pipeline: "evolutionary", eyebrow: "optimizer",       title: "AGE-MOEA-II",    viz: "pareto" },
    pdq_stage1:        { x:  918, y: 100, pipeline: "pdq",          eyebrow: "stage 1",         title: "Flip discovery", viz: "strategy_mix" },
    pdq_metric:        { x:  588, y: 250, pipeline: "pdq",          eyebrow: "metric",          title: "d_o / d_i" },
    pdq_stage2:        { x: 1088, y: 100, pipeline: "pdq",          eyebrow: "stage 2",         title: "Minimise flip" },
    artifacts:         { x: 1290, y: 100, pipeline: "both",         eyebrow: "artifacts",       title: "Parquet write",  viz: "parquet" },

    // Tier 2 — text branch:
    manipulator_text:  { x:  424, y: 250, pipeline: "both",         eyebrow: "manipulator",     title: "Text composite", viz: "tokens" },

    // Tier 3 — composite bridge under the main flow:
    manipulator_vlm:   { x:  588, y: 390, pipeline: "both",         eyebrow: "bridge",          title: "VLM compose" },
  };

  // Particle housekeeping shared across the runtime.
  const PARTICLES = {
    active: 0,            // currently-flying particle count
    maxConcurrent: 80,    // hard cap (perf guardrail)
    rafId: null,          // requestAnimationFrame id for spawn loop
    lastSpawnT: 0,        // last spawn timestamp (ms)
    spawnAccumulator: 0,  // fractional particles owed
  };

  // ───────────────────────────────────────────────────────────────
  // Geometry helpers
  // ───────────────────────────────────────────────────────────────

  /**
   * Intersection of the line from (cx,cy) to (tx,ty) with the rect
   * boundary centred at (cx,cy). Used to anchor edges on the node
   * outline rather than the centre.
   */
  function anchorOnRect(cx, cy, w, h, tx, ty) {
    const dx = tx - cx;
    const dy = ty - cy;
    if (dx === 0 && dy === 0) return { x: cx + w / 2, y: cy };
    const halfW = w / 2;
    const halfH = h / 2;
    const ax = Math.abs(dx);
    const ay = Math.abs(dy);
    // Pick the boundary edge the ray exits through.
    if (ax / halfW > ay / halfH) {
      const sign = dx > 0 ? 1 : -1;
      const x = cx + sign * halfW;
      const y = cy + (dy / ax) * halfW;
      return { x, y };
    }
    const sign = dy > 0 ? 1 : -1;
    const y = cy + sign * halfH;
    const x = cx + (dx / ay) * halfH;
    return { x, y };
  }

  function anchorOnNode(nodeId, tx, ty) {
    const c = NODE_LAYOUT[nodeId];
    return anchorOnRect(c.x, c.y, NODE_W, NODE_H, tx, ty);
  }

  /**
   * Cubic-Bézier path between two nodes. Control points are biased
   * along the dominant axis of the connection (horizontal for same-
   * tier edges, vertical for cross-tier) so the curve enters and
   * exits perpendicular-ish to the node boundary.
   */
  function edgePath(fromId, toId, isFeedback) {
    const A = NODE_LAYOUT[fromId];
    const B = NODE_LAYOUT[toId];
    const a = anchorOnNode(fromId, B.x, B.y);
    const b = anchorOnNode(toId, A.x, A.y);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const horizontal = Math.abs(dx) >= Math.abs(dy);
    let c1x, c1y, c2x, c2y;
    if (horizontal) {
      // Same-tier flow: control points hugging the horizontal.
      const k = Math.max(28, Math.abs(dx) * 0.32);
      c1x = a.x + Math.sign(dx) * k;
      c1y = a.y;
      c2x = b.x - Math.sign(dx) * k;
      c2y = b.y;
    } else {
      // Cross-tier flow: vertical control points so we enter the
      // top/bottom edge of the destination instead of the side.
      const k = Math.max(28, Math.abs(dy) * 0.42);
      c1x = a.x;
      c1y = a.y + Math.sign(dy) * k;
      c2x = b.x;
      c2y = b.y - Math.sign(dy) * k;
    }
    if (isFeedback) {
      // Bow feedback edges further out so they don't trace the same
      // line as the corresponding forward edge underneath.
      const px = -dy;
      const py = dx;
      const len = Math.hypot(px, py) || 1;
      const bow = horizontal ? 26 : 36;
      c1x += (px / len) * bow;
      c1y += (py / len) * bow;
      c2x += (px / len) * bow;
      c2y += (py / len) * bow;
    }
    return `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
  }

  // ───────────────────────────────────────────────────────────────
  // Particle spawning (SVG <animateMotion>)
  // ───────────────────────────────────────────────────────────────

  function spawnParticleOnPath(svgRoot, pathEl, durationMs) {
    if (!svgRoot || !pathEl) return;
    if (PARTICLES.active >= PARTICLES.maxConcurrent) return;
    const ns = "http://www.w3.org/2000/svg";
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("r", String(2.6 + Math.random() * 0.6));
    circle.setAttribute("class", "particle particle--flow");
    circle.setAttribute("fill", "var(--accent)");
    circle.setAttribute("opacity", "0.9");

    const motion = document.createElementNS(ns, "animateMotion");
    motion.setAttribute("dur", `${durationMs}ms`);
    motion.setAttribute("fill", "freeze");
    motion.setAttribute("rotate", "auto");
    motion.setAttribute("begin", "indefinite");
    const mpath = document.createElementNS(ns, "mpath");
    mpath.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", `#${pathEl.id}`);
    mpath.setAttribute("href", `#${pathEl.id}`);
    motion.appendChild(mpath);
    circle.appendChild(motion);

    svgRoot.appendChild(circle);
    PARTICLES.active += 1;
    motion.addEventListener("endEvent", () => {
      circle.remove();
      PARTICLES.active = Math.max(0, PARTICLES.active - 1);
    });
    try { motion.beginElement(); } catch (e) { /* noop */ }
  }

  function startParticleLoop(host) {
    cancelParticleLoop();
    PARTICLES.lastSpawnT = performance.now();
    PARTICLES.spawnAccumulator = 0;

    function tick(now) {
      const store = window.Alpine?.store("explorer");
      if (!store || !store.ready) {
        PARTICLES.rafId = requestAnimationFrame(tick);
        return;
      }
      const rate = store.particleRate(); // particles/sec
      const dt = (now - PARTICLES.lastSpawnT) / 1000;
      PARTICLES.lastSpawnT = now;
      PARTICLES.spawnAccumulator += rate * dt;

      const svgRoot = host.querySelector(".pipeline-canvas__svg");
      const forwardPaths = svgRoot
        ? Array.from(svgRoot.querySelectorAll(".edge-path[data-flow='forward']"))
        : [];

      while (
        PARTICLES.spawnAccumulator >= 1 &&
        PARTICLES.active < PARTICLES.maxConcurrent &&
        forwardPaths.length > 0
      ) {
        const path = forwardPaths[Math.floor(Math.random() * forwardPaths.length)];
        const len = (() => { try { return path.getTotalLength(); } catch (e) { return 200; } })();
        const dur = Math.max(1500, Math.min(2600, 1400 + len * 4));
        spawnParticleOnPath(svgRoot, path, dur);
        PARTICLES.spawnAccumulator -= 1;
      }
      PARTICLES.rafId = requestAnimationFrame(tick);
    }
    PARTICLES.rafId = requestAnimationFrame(tick);
  }

  function cancelParticleLoop() {
    if (PARTICLES.rafId !== null) {
      cancelAnimationFrame(PARTICLES.rafId);
      PARTICLES.rafId = null;
    }
  }

  function clearAllParticles(host) {
    if (!host) return;
    host.querySelectorAll("circle.particle--flow").forEach((c) => c.remove());
    PARTICLES.active = 0;
  }

  // ───────────────────────────────────────────────────────────────
  // Visible-node resolver
  // ───────────────────────────────────────────────────────────────

  function visibleNodesFor(pipeline) {
    return Object.keys(NODE_LAYOUT).filter((id) => {
      const cfg = NODE_LAYOUT[id];
      return cfg.pipeline === "both" || cfg.pipeline === pipeline;
    });
  }

  // ───────────────────────────────────────────────────────────────
  // Bespoke node illustrations (programmatic SVG)
  // ───────────────────────────────────────────────────────────────
  //
  // Each builder returns an SVG element to append to a node group.
  // All are positioned in node-local coordinates: the node group is
  // translated to (centerX - w/2, centerY - h/2) so child coords are
  // relative to the node's top-left.

  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    const n = document.createElementNS(NS, tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null) continue;
        n.setAttribute(k, String(v));
      }
    }
    return n;
  }

  // Patch grid: 4×4 cells, ~6 accented. Sits at bottom-right.
  function vizPatchGrid(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--patchgrid" });
    const size = 36;
    const cells = 4;
    const step = size / cells;
    g.appendChild(el("rect", { x: 0, y: 0, width: size, height: size, rx: 2, class: "viz-bg" }));
    const accents = new Set(["0,2", "1,1", "1,3", "2,0", "2,2", "3,1"]);
    for (let r = 0; r < cells; r++) {
      for (let c = 0; c < cells; c++) {
        const accent = accents.has(`${r},${c}`);
        g.appendChild(el("rect", {
          x: c * step + 1.5, y: r * step + 1.5,
          width: step - 3, height: step - 3,
          rx: 1,
          class: accent ? "viz-cell viz-cell--accent" : "viz-cell",
        }));
      }
    }
    return g;
  }

  // Token chips: 5 small rounded rects, one accent.
  function vizTokens(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--tokens" });
    const widths = [12, 10, 14, 13, 11];
    const accentIdx = 3;
    let px = 0;
    widths.forEach((w, i) => {
      g.appendChild(el("rect", {
        x: px, y: 12, width: w, height: 12, rx: 3,
        class: i === accentIdx ? "viz-chip viz-chip--accent" : "viz-chip",
      }));
      px += w + 3;
    });
    return g;
  }

  // Pareto curve: one cubic curve + scattered dots above + Pareto-front dots.
  function vizPareto(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--pareto" });
    g.appendChild(el("path", {
      d: "M 2 30 C 14 28, 22 18, 32 10 C 38 6, 44 4, 50 3",
      class: "viz-curve",
    }));
    const dots = [
      { x:  8, y: 28, a: false },
      { x: 16, y: 22, a: false },
      { x: 24, y: 16, a: true },
      { x: 32, y: 11, a: true },
      { x: 42, y:  6, a: true },
      { x: 12, y: 32, a: false },
      { x: 28, y: 22, a: false },
    ];
    for (const d of dots) {
      g.appendChild(el("circle", { cx: d.x, cy: d.y, r: 1.5,
        class: d.a ? "viz-dot viz-dot--accent" : "viz-dot" }));
    }
    return g;
  }

  // SUT/transformer: 3 horizontal lines (transformer block) + 2 output bars.
  function vizTransformer(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--transformer" });
    // Block lines.
    for (let i = 0; i < 3; i++) {
      g.appendChild(el("rect", {
        x: 0, y: i * 5, width: 36, height: 2, rx: 1,
        class: "viz-tx-line",
      }));
    }
    // Probabilities A / B as bars.
    g.appendChild(el("rect", { x: 0, y: 22, width: 30, height: 3, rx: 1, class: "viz-bar viz-bar--a" }));
    g.appendChild(el("rect", { x: 0, y: 28, width: 18, height: 3, rx: 1, class: "viz-bar viz-bar--b" }));
    g.appendChild(el("text", { x: 33, y: 25, class: "viz-bar-label" })).textContent = "A";
    g.appendChild(el("text", { x: 33, y: 31, class: "viz-bar-label viz-bar-label--muted" })).textContent = "B";
    return g;
  }

  // Dot cluster: 12 dots, 4 highlighted accent.
  function vizDotCluster(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--dotcluster" });
    const dots = [
      [3, 4, 0], [9, 2, 0], [15, 5, 0], [21, 3, 1],
      [5, 11, 0], [12, 10, 1], [18, 12, 0], [24, 11, 0],
      [3, 18, 1], [10, 19, 0], [17, 18, 0], [23, 19, 1],
    ];
    for (const [cx, cy, accent] of dots) {
      g.appendChild(el("circle", { cx, cy, r: 1.8,
        class: accent ? "viz-dot viz-dot--accent" : "viz-dot" }));
    }
    return g;
  }

  // Parquet: 3 stacked offset rectangles.
  function vizParquet(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--parquet" });
    g.appendChild(el("rect", { x: 0, y: 8, width: 24, height: 16, rx: 2, class: "viz-page viz-page--back" }));
    g.appendChild(el("rect", { x: 3, y: 5, width: 24, height: 16, rx: 2, class: "viz-page viz-page--mid" }));
    g.appendChild(el("rect", { x: 6, y: 2, width: 24, height: 16, rx: 2, class: "viz-page viz-page--front" }));
    return g;
  }

  // Strategy mix: 5 vertical bars of varying height (histogram).
  function vizStrategyMix(x, y) {
    const g = el("g", { transform: `translate(${x}, ${y})`, class: "node-viz node-viz--strategy-mix" });
    const heights = [8, 14, 22, 18, 12];
    const accents = [false, false, true, false, false];
    let px = 0;
    heights.forEach((h, i) => {
      g.appendChild(el("rect", {
        x: px, y: 26 - h, width: 5, height: h, rx: 1,
        class: accents[i] ? "viz-bar viz-bar--accent" : "viz-bar",
      }));
      px += 7;
    });
    return g;
  }

  const VIZ_BUILDERS = {
    patchgrid: vizPatchGrid,
    tokens: vizTokens,
    pareto: vizPareto,
    transformer: vizTransformer,
    dotcluster: vizDotCluster,
    parquet: vizParquet,
    strategy_mix: vizStrategyMix,
  };

  // ───────────────────────────────────────────────────────────────
  // Node rendering
  // ───────────────────────────────────────────────────────────────

  /**
   * Build one SVG <g> for a single node. Returns the group element
   * positioned in viewBox coords (translated to node centre minus
   * half-size). The group is interactive (cursor pointer, focusable,
   * aria-label) and dispatches a setFocus click.
   */
  function buildNodeGroup(nodeId, store) {
    const cfg = NODE_LAYOUT[nodeId];
    const nodeMeta = (store?.graph?.nodes || {})[nodeId] || {};
    const fullLabel = nodeMeta.label || cfg.title;
    const summary = nodeMeta.summary || "";
    const left = cfg.x - NODE_W / 2;
    const top = cfg.y - NODE_H / 2;

    const g = el("g", {
      class: "cnode",
      "data-node-id": nodeId,
      transform: `translate(${left}, ${top})`,
      tabindex: 0,
      role: "button",
      "aria-label": `Focus ${fullLabel}`,
    });

    // Background rect.
    g.appendChild(el("rect", {
      x: 0, y: 0, width: NODE_W, height: NODE_H, rx: NODE_RX,
      class: "cnode__bg",
    }));

    // Eyebrow (mono small caps).
    const eyebrow = el("text", { x: 12, y: 20, class: "cnode__eyebrow" });
    eyebrow.textContent = (cfg.eyebrow || nodeId).toUpperCase();
    g.appendChild(eyebrow);

    // Title (display, bold).
    const title = el("text", { x: 12, y: 40, class: "cnode__title" });
    title.textContent = cfg.title;
    g.appendChild(title);

    // Optional viz, anchored bottom-right inside the box.
    if (cfg.viz && VIZ_BUILDERS[cfg.viz]) {
      const w = NODE_W - 12; // viz right padding 8 + width target ~38
      const vizX = NODE_W - 46;
      const vizY = NODE_H - 42;
      g.appendChild(VIZ_BUILDERS[cfg.viz](vizX, vizY));
    }

    // Cache the summary on the group so the tooltip handler can read
    // it without re-querying the store.
    g.__summary = summary;
    g.__label = fullLabel;

    return g;
  }

  /** Refresh the per-node focus / muted / active class on every cnode. */
  function refreshNodeStates(host, store) {
    const focusId = store.focusNode;
    const hoverId = store.hoverNode;
    host.querySelectorAll(".cnode").forEach((g) => {
      const id = g.getAttribute("data-node-id");
      g.classList.toggle("cnode--focused", id === focusId);
      g.classList.toggle("cnode--hover", id !== focusId && id === hoverId);
      g.classList.toggle("cnode--muted", !!focusId && id !== focusId && id !== hoverId);
      g.classList.toggle("cnode--active", !focusId);
    });
  }

  function renderNodes(host) {
    if (!host) return;
    const store = window.Alpine?.store("explorer");
    if (!store || !store.ready) return;
    const group = host.querySelector("#dyn-nodes");
    if (!group) return;
    while (group.firstChild) group.removeChild(group.firstChild);

    const visible = visibleNodesFor(store.pipeline);
    for (const id of visible) {
      group.appendChild(buildNodeGroup(id, store));
    }
    refreshNodeStates(host, store);
  }

  // ───────────────────────────────────────────────────────────────
  // Edge rendering
  // ───────────────────────────────────────────────────────────────

  function buildDecoratedEdges(store) {
    if (!store) return [];
    const focus = store.focusNode;
    const edges = store.activeEdges() || [];
    const visible = new Set(visibleNodesFor(store.pipeline));
    return edges
      .filter((e) => visible.has(e.from) && visible.has(e.to))
      .map((e, i) => {
        const isFeedback = e.kind === "feedback";
        const isFocused = focus && (focus === e.from || focus === e.to);
        const id = `edge-${store.pipeline}-${i}`;
        return {
          id,
          from: e.from,
          to: e.to,
          kind: e.kind || "forward",
          d: edgePath(e.from, e.to, isFeedback),
          isFeedback,
          isFocused,
          cls:
            "edge edge-path" +
            (isFocused ? " edge--focused" : "") +
            (isFeedback ? " edge--feedback" : ""),
        };
      });
  }

  function renderEdges(host) {
    if (!host) return;
    const store = window.Alpine?.store("explorer");
    if (!store || !store.ready) return;
    const group = host.querySelector("#dyn-edges");
    if (!group) return;
    while (group.firstChild) group.removeChild(group.firstChild);

    const decorated = buildDecoratedEdges(store);
    for (const edge of decorated) {
      const path = el("path", {
        id: edge.id,
        class: edge.cls,
        d: edge.d,
        "data-flow": edge.isFeedback ? "feedback" : "forward",
        fill: "none",
      });
      if (edge.isFeedback) path.setAttribute("marker-end", "url(#arrow-feedback)");
      group.appendChild(path);
    }
  }

  // ───────────────────────────────────────────────────────────────
  // Alpine.data — canvas component
  // ───────────────────────────────────────────────────────────────

  document.addEventListener("alpine:init", () => {
    window.Alpine.data("canvasApp", () => ({
      // Tooltip position cache (HTML-space relative to host).
      tip: { visible: false, x: 0, y: 0, label: "", summary: "" },
      _hostEl: null,
      _bound: false,

      init() {
        this._hostEl = this.$el;
        this._bindNodeHandlers();
        window.addEventListener("explorer:pipeline-changed", () => {
          this.onPipelineChange();
        });
        window.addEventListener("explorer:ready", () => {
          this.$nextTick(() => this._renderAll());
        });
        window.addEventListener("explorer:focus-changed", () => {
          // Only state classes + edge re-render needed; node DOM is stable.
          const store = this.$store.explorer;
          refreshNodeStates(this._hostEl, store);
          renderEdges(this._hostEl);
        });
        const store = this.$store.explorer;
        if (store && store.ready) {
          this.$nextTick(() => this._renderAll());
        }
      },

      _renderAll() {
        renderNodes(this._hostEl);
        renderEdges(this._hostEl);
        startParticleLoop(this._hostEl);
      },

      // Delegated pointer / keyboard handlers on the #dyn-nodes group.
      _bindNodeHandlers() {
        if (this._bound) return;
        const host = this._hostEl;
        const findNode = (target) => {
          if (!target) return null;
          const g = target.closest && target.closest(".cnode");
          return g || null;
        };

        host.addEventListener("click", (ev) => {
          const g = findNode(ev.target);
          if (!g) return;
          const id = g.getAttribute("data-node-id");
          this.$store.explorer.setFocus(id);
        });

        host.addEventListener("keydown", (ev) => {
          if (ev.key !== "Enter" && ev.key !== " ") return;
          const g = findNode(ev.target);
          if (!g) return;
          ev.preventDefault();
          const id = g.getAttribute("data-node-id");
          this.$store.explorer.setFocus(id);
        });

        host.addEventListener("mouseover", (ev) => {
          const g = findNode(ev.target);
          if (!g) return;
          const id = g.getAttribute("data-node-id");
          if (this.$store.explorer.hoverNode === id) return;
          this.$store.explorer.setHoverNode(id);
          refreshNodeStates(this._hostEl, this.$store.explorer);
          if (this.$store.explorer.focusNode !== id) {
            this._showTooltipFor(g);
          }
        });

        host.addEventListener("mouseout", (ev) => {
          const g = findNode(ev.target);
          if (!g) return;
          // Only hide if we're leaving the node entirely (not moving
          // between child SVG elements).
          const next = ev.relatedTarget;
          if (next && g.contains(next)) return;
          this.$store.explorer.setHoverNode(null);
          refreshNodeStates(this._hostEl, this.$store.explorer);
          this.tip = { ...this.tip, visible: false };
        });

        this._bound = true;
      },

      _showTooltipFor(g) {
        const id = g.getAttribute("data-node-id");
        const host = this._hostEl;
        const rect = host.getBoundingClientRect();
        const cfg = NODE_LAYOUT[id];
        // Convert viewBox coords → viewport coords (tooltip is position:fixed
        // to escape ancestor overflow:hidden clips on .panel and .pipeline-canvas).
        const scale = Math.min(rect.width / VIEW_W, rect.height / VIEW_H);
        const drawnW = VIEW_W * scale;
        const drawnH = VIEW_H * scale;
        const offX = (rect.width - drawnW) / 2;
        const offY = (rect.height - drawnH) / 2;
        const cx = rect.left + offX + cfg.x * scale;
        const cy = rect.top + offY + cfg.y * scale;
        this.tip = {
          visible: true,
          x: cx,
          y: cy - (NODE_H / 2) * scale - 12,
          label: g.__label || cfg.title,
          summary: g.__summary || "",
        };
      },

      // Pipeline switch: clear, rebuild, restart particles.
      onPipelineChange() {
        const host = this._hostEl;
        if (!host) return;
        clearAllParticles(host);

        const store = this.$store.explorer;
        const visible = new Set(visibleNodesFor(store.pipeline));
        if (store.focusNode && !visible.has(store.focusNode)) {
          store.setFocus(store.pipeline === "pdq" ? "pdq_metric" : "manipulator_image");
        }

        this.$nextTick(() => {
          renderNodes(host);
          renderEdges(host);
          const motion = window.Motion;
          if (motion && motion.animate) {
            host.querySelectorAll(".cnode").forEach((el2) => {
              motion.animate(el2, { opacity: [0.2, 1] }, {
                duration: 0.3,
                easing: "cubic-bezier(0.32, 0.72, 0, 1)",
              });
            });
          }
          startParticleLoop(host);
        });
      },

      // Legacy hooks retained for any external Alpine binding (unused
      // in the SVG-only flow but harmless to expose).
      visibleNodes() { return visibleNodesFor(this.$store.explorer.pipeline); },
      decoratedEdges() { return buildDecoratedEdges(this.$store.explorer); },
    }));

    // ─────────────────────────────────────────────────────────────
    // Alpine.data — top-bar segmented controls
    // ─────────────────────────────────────────────────────────────
    window.Alpine.data("topBar", () => ({
      togglePipeline(p) {
        this.$store.explorer.switchPipeline(p);
      },
      toggleMode(m) {
        this.$store.explorer.switchMode(m);
      },
      toggleTheme() {
        const motion = window.Motion;
        const body = document.body;
        if (motion && motion.animate) {
          motion.animate(body, { opacity: [1, 0.82, 1] }, { duration: 0.22 });
        }
        this.$store.explorer.toggleTheme();
      },
    }));
  });
})();
