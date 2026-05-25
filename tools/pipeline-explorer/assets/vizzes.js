/*
 * vizzes.js · phase 5
 *
 * Hero micro-vizzes mounted under hero-tier knob widgets.
 * Each viz reads the active config from $store.explorer.config and shows
 * the algorithmic shape that config implies (not predicted outcomes — we
 * cannot run a VLM in-browser).
 *
 * Contract with phase 4 (widgets.js):
 *   - Hero leaves render <div data-viz-slot="true" data-path="<leaf.path>">.
 *   - Alpine's x-for destroys and recreates slot DOM on focus change, so we
 *     remount on `explorer:focus-changed`. We update without remount on
 *     `explorer:config-changed` by attaching a tiny `_update(store)` closure
 *     to each slot during initial render.
 *
 * Performance:
 *   - Slot updates are coalesced into a single rAF tick.
 *   - Renderers build SVG once, keep refs to mutable nodes in the closure,
 *     and only swap attributes during updates.
 *
 * Design language:
 *   - Electric green `--accent`, muted `--fg-muted`, tabular monospace.
 *   - ~60–80px tall; minimal chrome; no chartjunk; asymmetric where it reads.
 */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  // ── tiny utilities ───────────────────────────────────────────────
  function svg(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] === false || attrs[k] === null || attrs[k] === undefined) continue;
        el.setAttribute(k, attrs[k]);
      }
    }
    return el;
  }

  function h(tag, attrs, children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "style" && typeof attrs[k] === "object") {
          Object.assign(el.style, attrs[k]);
        } else if (k === "class") {
          el.className = attrs[k];
        } else {
          el.setAttribute(k, attrs[k]);
        }
      }
    }
    if (children) {
      const arr = Array.isArray(children) ? children : [children];
      for (const c of arr) {
        if (c == null) continue;
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      }
    }
    return el;
  }

  function clearSlot(slot) {
    while (slot.firstChild) slot.removeChild(slot.firstChild);
    slot._update = null;
  }

  // Deterministic seeded RNG (Mulberry32) — used for any "random but stable"
  // visual decoration so the viz doesn't flicker between renders.
  function deterministicRng(seed) {
    let s = (seed | 0) || 1;
    return function () {
      s |= 0;
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashFloat(x) {
    // Map a float in [0, 1] (or [0, ∞)) to a stable 32-bit seed.
    const v = Math.round(Number(x) * 1e6);
    return ((v ^ 0x9e3779b9) * 2654435761) >>> 0;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function fmtInt(n) {
    if (n == null || !Number.isFinite(n)) return "—";
    return Math.round(n).toLocaleString("en-US");
  }

  function fmtPct(p, digits) {
    if (p == null || !Number.isFinite(p)) return "—";
    return (p * 100).toFixed(digits ?? 1) + "%";
  }

  // Common wrapper styles — keeps viz heights consistent across renderers.
  function wrap(className) {
    const div = document.createElement("div");
    div.className = "viz" + (className ? " viz--" + className : "");
    div.style.display = "flex";
    div.style.flexDirection = "column";
    div.style.gap = "6px";
    div.style.alignItems = "stretch";
    div.style.width = "100%";
    return div;
  }

  function captionEl(text) {
    const el = document.createElement("div");
    el.className = "viz__caption";
    el.style.fontFamily = "var(--font-mono)";
    el.style.fontSize = "var(--text-mono-sm)";
    el.style.color = "var(--fg-muted)";
    el.style.letterSpacing = "0.01em";
    el.textContent = text;
    return el;
  }

  // ─────────────────────────────────────────────────────────────────
  // RENDERERS
  // Each renderer:
  //   1. Builds DOM inside `slot` once.
  //   2. Stashes references to mutable nodes in closure scope.
  //   3. Attaches `slot._update = (store) => { ... }` for live updates.
  // ─────────────────────────────────────────────────────────────────

  // ── modality ─────────────────────────────────────────────────────
  function renderModalityBadge(slot, store) {
    clearSlot(slot);
    const root = wrap("modality");
    const row = h("div", { style: { display: "flex", gap: "6px" } });
    const pills = {};
    const labels = [
      ["joint", "joint = 3 objectives"],
      ["image_only", "image_only = 2 objectives (no text distance)"],
      ["text_only", "text_only = 2 objectives (no image distance)"],
    ];
    for (const [val] of labels) {
      const pill = h(
        "span",
        {
          style: {
            padding: "3px 10px",
            borderRadius: "999px",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            border: "1px solid var(--border-default)",
            color: "var(--fg-muted)",
            transition: "all 140ms ease",
          },
        },
        val,
      );
      pills[val] = pill;
      row.appendChild(pill);
    }
    const cap = captionEl("");
    root.appendChild(row);
    root.appendChild(cap);
    slot.appendChild(root);

    function update(s) {
      const v = s.getKnob("modality") || "joint";
      for (const [val] of labels) {
        const active = val === v;
        const pill = pills[val];
        pill.style.background = active ? "var(--accent)" : "transparent";
        pill.style.color = active ? "var(--fg-on-accent)" : "var(--fg-muted)";
        pill.style.borderColor = active ? "var(--accent)" : "var(--border-default)";
      }
      const found = labels.find((l) => l[0] === v);
      cap.textContent = found ? found[1] : "";
    }
    update(store);
    slot._update = update;
  }

  // ── image.patch_ratio: 16×16 grid ─────────────────────────────────
  function renderPatchGrid(slot, store) {
    clearSlot(slot);
    const root = wrap("patch-grid");
    const N = 16;
    const CELL = 8;
    const PAD = 1;
    const sz = N * (CELL + PAD) - PAD;
    const s = svg("svg", {
      viewBox: `0 0 ${sz} ${sz}`,
      width: sz,
      height: sz,
      "aria-hidden": "true",
    });
    s.style.display = "block";
    const cells = [];
    // pre-shuffle a stable permutation; reorder per-ratio via deterministic RNG.
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const rect = svg("rect", {
          x: c * (CELL + PAD),
          y: r * (CELL + PAD),
          width: CELL,
          height: CELL,
          rx: 1,
          fill: "var(--border-subtle)",
        });
        s.appendChild(rect);
        cells.push(rect);
      }
    }
    root.appendChild(s);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const ratio = clamp(Number(st.getKnob("image.patch_ratio")) || 0, 0, 1);
      const total = N * N;
      const lit = Math.round(total * ratio);
      // Stable shuffle by ratio bucket so visualisation reads as monotone.
      const rng = deterministicRng(hashFloat(Math.round(ratio * 10000)));
      const order = cells.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
      // Reset all to muted, light the first `lit`.
      for (let i = 0; i < cells.length; i++) {
        cells[i].setAttribute("fill", "var(--border-subtle)");
      }
      for (let i = 0; i < lit; i++) {
        const idx = order[i];
        cells[idx].setAttribute("fill", "var(--accent)");
        cells[idx].setAttribute("opacity", String(0.55 + 0.45 * (1 - i / Math.max(1, lit))));
      }
      cap.textContent = `~${lit} patches mutable (${(ratio * 100).toFixed(1)}% of ${total})`;
    }
    update(store);
    slot._update = update;
  }

  // ── image.n_candidates ────────────────────────────────────────────
  function renderDepthSparkline(slot, store) {
    clearSlot(slot);
    const root = wrap("depth-spark");
    const W = 240;
    const H = 28;
    const s = svg("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: H,
      preserveAspectRatio: "none",
      "aria-hidden": "true",
    });
    s.style.display = "block";
    s.appendChild(svg("rect", { x: 0, y: H / 2 - 1, width: W, height: 2, fill: "var(--border-subtle)" }));
    const group = svg("g", {});
    s.appendChild(group);
    // Identity tick (gene = 0) is always present; emphasised.
    const idTick = svg("rect", { x: 0, y: 2, width: 2.5, height: H - 4, fill: "var(--accent)", rx: 1 });
    s.appendChild(idTick);
    root.appendChild(s);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const n = clamp(Number(st.getKnob("image.n_candidates")) || 0, 0, 100000);
      // Cap drawn notches at 50 to keep readable.
      const drawn = Math.min(n, 50);
      while (group.firstChild) group.removeChild(group.firstChild);
      const leftPad = 8;
      const rightPad = 4;
      const span = W - leftPad - rightPad;
      for (let i = 1; i <= drawn; i++) {
        const x = leftPad + (i / Math.max(1, drawn)) * span;
        const tick = svg("rect", {
          x: x - 0.5,
          y: 6,
          width: 1,
          height: H - 12,
          fill: "var(--fg-muted)",
          opacity: 0.7,
        });
        group.appendChild(tick);
      }
      cap.textContent =
        n <= 50
          ? `identity (gene=0) + ${n} candidate${n === 1 ? "" : "s"}`
          : `identity + ${fmtInt(n)} candidates (showing first 50 of axis)`;
    }
    update(store);
    slot._update = update;
  }

  // ── image.patch_strategy ─────────────────────────────────────────
  function renderPatchStrategy(slot, store) {
    clearSlot(slot);
    const root = wrap("patch-strategy");
    const row = h("div", {
      style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" },
    });
    function miniGrid(kind) {
      const cardEl = h("div", {
        style: {
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          padding: "6px",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle)",
          transition: "border-color 140ms ease",
        },
      });
      const N = 10;
      const CELL = 6;
      const PAD = 1;
      const sz = N * (CELL + PAD) - PAD;
      const s = svg("svg", { viewBox: `0 0 ${sz} ${sz}`, width: sz, height: sz });
      const rng = deterministicRng(kind === "FREQUENCY" ? 11 : 47);
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          let lit;
          if (kind === "FREQUENCY") {
            // Bias selection toward top-left quadrant (low-freq tokens).
            const bias = (1 - r / N) * (1 - c / N);
            lit = rng() < 0.18 + 0.55 * bias;
          } else {
            lit = rng() < 0.25;
          }
          s.appendChild(
            svg("rect", {
              x: c * (CELL + PAD),
              y: r * (CELL + PAD),
              width: CELL,
              height: CELL,
              rx: 1,
              fill: lit ? "var(--accent)" : "var(--border-subtle)",
              opacity: lit ? 0.85 : 1,
            }),
          );
        }
      }
      const label = h(
        "div",
        {
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            color: "var(--fg-muted)",
            textAlign: "center",
          },
        },
        kind,
      );
      cardEl.appendChild(s);
      cardEl.appendChild(label);
      return cardEl;
    }
    const cards = { FREQUENCY: miniGrid("FREQUENCY"), ALL: miniGrid("ALL") };
    row.appendChild(cards.FREQUENCY);
    row.appendChild(cards.ALL);
    root.appendChild(row);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const v = st.getKnob("image.patch_strategy") || "FREQUENCY";
      for (const k of Object.keys(cards)) {
        const active = k === v;
        cards[k].style.borderColor = active ? "var(--accent)" : "var(--border-subtle)";
        cards[k].style.background = active ? "var(--accent-soft)" : "transparent";
      }
      cap.textContent =
        v === "FREQUENCY"
          ? "FREQUENCY = patches biased to high-information tokens"
          : "ALL = uniform patch selection across image";
    }
    update(store);
    slot._update = update;
  }

  // ── image.candidate_strategy ─────────────────────────────────────
  function renderCandidateStrategy(slot, store) {
    clearSlot(slot);
    const root = wrap("candidate-strategy");
    const row = h("div", {
      style: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px" },
    });
    function dist(kind) {
      const card = h("div", {
        style: {
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          padding: "6px",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle)",
          transition: "border-color 140ms ease",
        },
      });
      const W = 64;
      const H = 26;
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
      const bars = 16;
      const bw = W / bars;
      for (let i = 0; i < bars; i++) {
        const x = i / (bars - 1);
        let y;
        if (kind === "KNN") {
          y = Math.exp(-Math.pow((x - 0.05) * 4, 2));
        } else if (kind === "UNIFORM") {
          y = 0.65 + 0.05 * Math.sin(i * 1.2);
        } else {
          // KFN — peak near max distance
          y = Math.exp(-Math.pow((x - 0.95) * 4, 2));
        }
        const bh = clamp(y, 0.05, 1) * (H - 4);
        s.appendChild(
          svg("rect", {
            x: i * bw + 1,
            y: H - 2 - bh,
            width: bw - 1.5,
            height: bh,
            rx: 1,
            fill: "var(--fg-muted)",
            opacity: 0.6,
          }),
        );
      }
      const label = h(
        "div",
        {
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            color: "var(--fg-muted)",
            textAlign: "center",
          },
        },
        kind,
      );
      card.appendChild(s);
      card.appendChild(label);
      card._svg = s;
      return card;
    }
    const cards = { KNN: dist("KNN"), UNIFORM: dist("UNIFORM"), KFN: dist("KFN") };
    Object.values(cards).forEach((c) => row.appendChild(c));
    root.appendChild(row);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const v = st.getKnob("image.candidate_strategy") || "KNN";
      for (const k of Object.keys(cards)) {
        const active = k === v;
        cards[k].style.borderColor = active ? "var(--accent)" : "var(--border-subtle)";
        cards[k].style.background = active ? "var(--accent-soft)" : "transparent";
        const bars = cards[k]._svg.querySelectorAll("rect");
        bars.forEach((b) => {
          b.setAttribute("fill", active ? "var(--accent)" : "var(--fg-muted)");
          b.setAttribute("opacity", active ? "0.9" : "0.5");
        });
      }
      const captions = {
        KNN: "KNN = neighbours close to seed in embedding space",
        UNIFORM: "UNIFORM = flat distance distribution",
        KFN: "KFN = neighbours far from seed (more diverse)",
      };
      cap.textContent = captions[v] || "";
    }
    update(store);
    slot._update = update;
  }

  // ── text.composite.profile ───────────────────────────────────────
  function renderTextProfile(slot, store) {
    clearSlot(slot);
    const PROFILES = {
      noop: { synonym: 0, fragmentation: 0, character_noise: 0, saliency: 0 },
      light: { synonym: 0.2, fragmentation: 0.1, character_noise: 0.1, saliency: 0.1 },
      medium: { synonym: 0.35, fragmentation: 0.2, character_noise: 0.2, saliency: 0.2 },
      full_stack: { synonym: 0.5, fragmentation: 0.3, character_noise: 0.3, saliency: 0.3 },
      synonym_only: { synonym: 0.5, fragmentation: 0, character_noise: 0, saliency: 0 },
      saliency_only: { synonym: 0, fragmentation: 0, character_noise: 0, saliency: 0.5 },
    };
    const root = wrap("text-profile");
    const ops = ["synonym", "fragmentation", "character_noise", "saliency"];
    const grid = h("div", {
      style: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "6px" },
    });
    const refs = {};
    ops.forEach((op) => {
      const card = h("div", {
        style: {
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          padding: "4px 6px",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle)",
          background: "var(--bg-elevated)",
        },
      });
      const name = h(
        "div",
        {
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            color: "var(--fg-secondary)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          },
        },
        op,
      );
      const barWrap = h("div", {
        style: {
          height: "4px",
          background: "var(--border-subtle)",
          borderRadius: "2px",
          overflow: "hidden",
        },
      });
      const fill = h("div", {
        style: {
          height: "100%",
          width: "0%",
          background: "var(--accent)",
          transition: "width 180ms ease",
        },
      });
      barWrap.appendChild(fill);
      const num = h(
        "div",
        {
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            color: "var(--fg-muted)",
            fontVariantNumeric: "tabular-nums",
          },
        },
        "0.00",
      );
      card.appendChild(name);
      card.appendChild(barWrap);
      card.appendChild(num);
      grid.appendChild(card);
      refs[op] = { fill, num };
    });
    root.appendChild(grid);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const raw = st.getKnob("text.composite.profile");
      const key = raw && PROFILES[raw] ? raw : "full_stack";
      const sev = PROFILES[key];
      ops.forEach((op) => {
        const v = sev[op] || 0;
        refs[op].fill.style.width = (v * 100).toFixed(0) + "%";
        refs[op].num.textContent = v.toFixed(2);
      });
      cap.textContent = raw
        ? `profile = ${raw}`
        : "profile unset · showing canonical full_stack severities";
    }
    update(store);
    slot._update = update;
  }

  // ── seeds.mode ───────────────────────────────────────────────────
  function renderSeedsMode(slot, store) {
    clearSlot(slot);
    const root = wrap("seeds-mode");
    const row = h("div", {
      style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" },
    });
    function card(kind) {
      const c = h("div", {
        style: {
          padding: "6px",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle)",
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          transition: "border-color 140ms ease",
        },
      });
      if (kind === "gap_filter") {
        const W = 110;
        const H = 36;
        const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
        const bars = 18;
        for (let i = 0; i < bars; i++) {
          const x = i / (bars - 1);
          const y = Math.exp(-Math.pow((x - 0.35) * 3, 2));
          const bh = clamp(y, 0.05, 1) * (H - 6);
          s.appendChild(
            svg("rect", {
              x: i * (W / bars) + 1,
              y: H - 2 - bh,
              width: W / bars - 2,
              height: bh,
              rx: 1,
              fill: "var(--fg-muted)",
              opacity: 0.45,
            }),
          );
        }
        s.appendChild(
          svg("line", {
            x1: W * 0.55,
            x2: W * 0.55,
            y1: 0,
            y2: H,
            stroke: "var(--accent)",
            "stroke-width": 1.5,
            "stroke-dasharray": "3 2",
          }),
        );
        c.appendChild(s);
        c.appendChild(
          h(
            "div",
            {
              style: {
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-mono-sm)",
                color: "var(--fg-muted)",
                textAlign: "center",
              },
            },
            "gap_filter",
          ),
        );
      } else {
        // roster — schematic table
        const tbl = h("div", {
          style: {
            display: "grid",
            gridTemplateColumns: "1fr 12px 1fr",
            gap: "2px 4px",
            padding: "4px",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            color: "var(--fg-muted)",
          },
        });
        const pairs = [
          ["shark", "ray"],
          ["finch", "junco"],
          ["dog", "wolf"],
        ];
        for (const [a, b] of pairs) {
          tbl.appendChild(h("span", {}, a));
          tbl.appendChild(h("span", { style: { color: "var(--accent)" } }, "↔"));
          tbl.appendChild(h("span", {}, b));
        }
        c.appendChild(tbl);
        c.appendChild(
          h(
            "div",
            {
              style: {
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-mono-sm)",
                color: "var(--fg-muted)",
                textAlign: "center",
              },
            },
            "roster",
          ),
        );
      }
      return c;
    }
    const cards = { gap_filter: card("gap_filter"), roster: card("roster") };
    row.appendChild(cards.gap_filter);
    row.appendChild(cards.roster);
    root.appendChild(row);
    slot.appendChild(root);

    function update(st) {
      const v = st.getKnob("seeds.mode") || "gap_filter";
      for (const k of Object.keys(cards)) {
        const active = k === v;
        cards[k].style.borderColor = active ? "var(--accent)" : "var(--border-subtle)";
        cards[k].style.background = active ? "var(--accent-soft)" : "transparent";
      }
    }
    update(store);
    slot._update = update;
  }

  // ── seeds.gap_filter.max_logprob_gap ─────────────────────────────
  function renderGapCurve(slot, store) {
    clearSlot(slot);
    const root = wrap("gap-curve");
    const W = 240;
    const H = 60;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    s.style.display = "block";
    // Build a gaussian-ish density over [0, 5]; sample 80 points.
    const N = 80;
    const xs = new Array(N).fill(0).map((_, i) => (i / (N - 1)) * 5);
    const ys = xs.map((x) => Math.exp(-Math.pow((x - 1.2) / 0.95, 2)));
    const maxY = Math.max(...ys);
    const xpx = (x) => (x / 5) * (W - 4) + 2;
    const ypx = (y) => H - 4 - (y / maxY) * (H - 8);
    // Filled area (under-cutoff) — will be re-pointed on update.
    const fill = svg("path", { fill: "var(--accent-soft)", stroke: "none" });
    s.appendChild(fill);
    // Line over the full curve.
    let line = "M ";
    for (let i = 0; i < N; i++) {
      line += `${xpx(xs[i]).toFixed(2)},${ypx(ys[i]).toFixed(2)} `;
      if (i < N - 1) line += "L ";
    }
    s.appendChild(svg("path", { d: line, fill: "none", stroke: "var(--fg-muted)", "stroke-width": 1 }));
    // Cutoff line.
    const cutoff = svg("line", {
      x1: 0,
      x2: 0,
      y1: 0,
      y2: H,
      stroke: "var(--accent)",
      "stroke-width": 1.5,
    });
    s.appendChild(cutoff);
    // Axis baseline
    s.appendChild(svg("line", { x1: 2, x2: W - 2, y1: H - 4, y2: H - 4, stroke: "var(--border-subtle)" }));
    root.appendChild(s);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const g = clamp(Number(st.getKnob("seeds.gap_filter.max_logprob_gap")) || 0, 0, 5);
      const cx = xpx(g);
      cutoff.setAttribute("x1", cx);
      cutoff.setAttribute("x2", cx);
      // Build filled path: x ∈ [0, g]
      let d = `M ${xpx(0).toFixed(2)},${(H - 4).toFixed(2)} `;
      for (let i = 0; i < N; i++) {
        if (xs[i] > g) break;
        d += `L ${xpx(xs[i]).toFixed(2)},${ypx(ys[i]).toFixed(2)} `;
      }
      d += `L ${cx.toFixed(2)},${(H - 4).toFixed(2)} Z`;
      fill.setAttribute("d", d);
      cap.textContent = `cutoff = ${g.toFixed(2)} · seeds with gap < ${g.toFixed(2)} kept`;
    }
    update(store);
    slot._update = update;
  }

  // ── seeds.gap_filter.n_per_class ─────────────────────────────────
  function renderCounter(slot, store) {
    clearSlot(slot);
    const root = wrap("counter");
    const big = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--accent)",
        fontVariantNumeric: "tabular-nums",
        letterSpacing: "0.02em",
      },
    });
    const sub = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-sm)",
        color: "var(--fg-muted)",
      },
    });
    root.appendChild(big);
    root.appendChild(sub);
    slot.appendChild(root);

    function update(st) {
      const n = Number(st.getKnob("seeds.gap_filter.n_per_class")) || 0;
      const cats = Number(st.getKnob("n_categories"));
      if (Number.isFinite(cats) && cats > 0) {
        big.textContent = `${n} × ${cats} = ${fmtInt(n * cats)}`;
        sub.textContent = "images scored (n_per_class × n_categories)";
      } else {
        big.textContent = `${n} × …`;
        sub.textContent = "images scored per class (× n_categories)";
      }
    }
    update(store);
    slot._update = update;
  }

  // ── optimizer.sampling.mode ──────────────────────────────────────
  function renderSamplerMode(slot, store) {
    clearSlot(slot);
    const modes = ["uniform", "sparse", "sparse_multitier", "sparse_multitier_fps", "sparse_score_guided"];
    const root = wrap("sampler-mode");
    const grid = h("div", {
      style: { display: "grid", gridTemplateColumns: `repeat(${modes.length}, 1fr)`, gap: "4px" },
    });
    const refs = {};
    modes.forEach((m) => {
      const card = h("div", {
        style: {
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          padding: "4px",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle)",
          transition: "border-color 140ms ease",
        },
      });
      const W = 38;
      const H = 22;
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
      const bars = 10;
      // Shape per mode: histogram of expected per-individual activation count.
      // Uniform mass; sparse mass on tiny; multitier multimodal; fps wider; score_guided right-skew.
      const heights = (() => {
        const rng = deterministicRng(modes.indexOf(m) * 17 + 3);
        const out = new Array(bars).fill(0);
        for (let i = 0; i < bars; i++) {
          const x = i / (bars - 1);
          if (m === "uniform") out[i] = 0.55 + 0.1 * rng();
          else if (m === "sparse") out[i] = Math.exp(-Math.pow((x - 0.05) * 6, 2));
          else if (m === "sparse_multitier")
            out[i] = 0.3 * Math.exp(-Math.pow((x - 0.1) * 6, 2)) + 0.7 * Math.exp(-Math.pow((x - 0.6) * 4, 2));
          else if (m === "sparse_multitier_fps")
            out[i] =
              0.4 * Math.exp(-Math.pow((x - 0.15) * 5, 2)) + 0.6 * Math.exp(-Math.pow((x - 0.5) * 3, 2));
          else if (m === "sparse_score_guided") out[i] = Math.exp(-Math.pow((x - 0.7) * 4, 2));
        }
        return out;
      })();
      const bw = W / bars;
      heights.forEach((y, i) => {
        const bh = clamp(y, 0.05, 1) * (H - 4);
        s.appendChild(
          svg("rect", {
            x: i * bw + 0.5,
            y: H - 2 - bh,
            width: bw - 1,
            height: bh,
            rx: 0.5,
            fill: "var(--fg-muted)",
            opacity: 0.5,
          }),
        );
      });
      const label = h(
        "div",
        {
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "10px",
            color: "var(--fg-muted)",
            textAlign: "center",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          },
        },
        m,
      );
      card.appendChild(s);
      card.appendChild(label);
      grid.appendChild(card);
      refs[m] = { card, svg: s, label };
    });
    root.appendChild(grid);
    slot.appendChild(root);

    function update(st) {
      const v = st.getKnob("optimizer.sampling.mode") || "sparse_multitier";
      for (const m of modes) {
        const active = m === v;
        refs[m].card.style.borderColor = active ? "var(--accent)" : "var(--border-subtle)";
        refs[m].card.style.background = active ? "var(--accent-soft)" : "transparent";
        refs[m].label.style.color = active ? "var(--accent)" : "var(--fg-muted)";
        refs[m].svg.querySelectorAll("rect").forEach((r) => {
          r.setAttribute("fill", active ? "var(--accent)" : "var(--fg-muted)");
          r.setAttribute("opacity", active ? "0.9" : "0.4");
        });
      }
    }
    update(store);
    slot._update = update;
  }

  // ── optimizer.sampling.tiers ─────────────────────────────────────
  function renderTierHistogram(slot, store) {
    clearSlot(slot);
    const root = wrap("tier-hist");
    const W = 240;
    const H = 60;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    s.style.display = "block";
    const group = svg("g", {});
    s.appendChild(group);
    s.appendChild(svg("line", { x1: 4, x2: W - 4, y1: H - 8, y2: H - 8, stroke: "var(--border-subtle)" }));
    // Axis tick labels
    const axisLabels = svg("g", {});
    s.appendChild(axisLabels);
    root.appendChild(s);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const tiers = st.getKnob("optimizer.sampling.tiers") || [];
      while (group.firstChild) group.removeChild(group.firstChild);
      while (axisLabels.firstChild) axisLabels.removeChild(axisLabels.firstChild);
      const safe = Array.isArray(tiers) ? tiers : [];
      const padL = 4;
      const padR = 8;
      const usable = W - padL - padR;
      // x-axis log10(p_active); domain log10(1e-3) .. log10(1.0)
      const xMin = -3;
      const xMax = 0;
      const xpx = (p) => {
        const lp = Math.log10(Math.max(1e-4, p));
        const t = clamp((lp - xMin) / (xMax - xMin), 0, 1);
        return padL + t * usable;
      };
      let totalFrac = 0;
      for (const t of safe) {
        const frac = Number(t?.fraction) || 0;
        totalFrac += frac;
        const pa = Number(t?.p_active) || 0;
        const x = xpx(pa);
        const bh = clamp(frac, 0, 1) * (H - 16);
        group.appendChild(
          svg("rect", {
            x: x - 3,
            y: H - 8 - bh,
            width: 6,
            height: bh,
            rx: 1,
            fill: "var(--accent)",
            opacity: 0.85,
          }),
        );
      }
      // Axis ticks at p = 1e-3, 1e-2, 1e-1, 1
      [-3, -2, -1, 0].forEach((lp) => {
        const x = xpx(Math.pow(10, lp));
        axisLabels.appendChild(
          svg("text", {
            x,
            y: H - 1,
            "text-anchor": "middle",
            "font-family": "var(--font-mono)",
            "font-size": 8,
            fill: "var(--fg-muted)",
          }),
        ).textContent = lp === 0 ? "1" : `1e${lp}`;
      });
      cap.textContent = `Σ fraction = ${totalFrac.toFixed(2)} · ${safe.length} tier${safe.length === 1 ? "" : "s"} (x = log10 p_active)`;
    }
    update(store);
    slot._update = update;
  }

  // ── optimizer.sampling.zero_anchor_fraction ──────────────────────
  function renderZeroAnchorPie(slot, store) {
    clearSlot(slot);
    const root = wrap("zero-anchor");
    const row = h("div", {
      style: { display: "flex", alignItems: "center", gap: "10px" },
    });
    const W = 64;
    const cx = W / 2;
    const cy = W / 2;
    const r = 28;
    const s = svg("svg", { viewBox: `0 0 ${W} ${W}`, width: W, height: W });
    s.appendChild(svg("circle", { cx, cy, r, fill: "var(--border-subtle)" }));
    const slice = svg("path", { fill: "var(--accent)" });
    s.appendChild(slice);
    s.appendChild(svg("circle", { cx, cy, r: r - 10, fill: "var(--bg-elevated)" }));
    const label = svg("text", {
      x: cx,
      y: cy + 3,
      "text-anchor": "middle",
      "font-family": "var(--font-mono)",
      "font-size": 10,
      fill: "var(--fg-primary)",
    });
    s.appendChild(label);
    row.appendChild(s);
    const sub = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-sm)",
        color: "var(--fg-muted)",
        flex: "1",
        lineHeight: 1.4,
      },
    });
    row.appendChild(sub);
    root.appendChild(row);
    slot.appendChild(root);

    function update(st) {
      const f = clamp(Number(st.getKnob("optimizer.sampling.zero_anchor_fraction")) || 0, 0, 1);
      const angle = f * Math.PI * 2;
      if (f <= 0) {
        slice.setAttribute("d", "");
      } else if (f >= 0.999) {
        slice.setAttribute("d", `M ${cx} ${cy - r} A ${r} ${r} 0 1 1 ${cx - 0.01} ${cy - r} Z`);
      } else {
        const x2 = cx + r * Math.sin(angle);
        const y2 = cy - r * Math.cos(angle);
        const large = angle > Math.PI ? 1 : 0;
        slice.setAttribute("d", `M ${cx} ${cy} L ${cx} ${cy - r} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)} Z`);
      }
      label.textContent = (f * 100).toFixed(0) + "%";
      sub.textContent = `${(f * 100).toFixed(1)}% of population pinned to zero (anchor)`;
    }
    update(store);
    slot._update = update;
  }

  // ── optimizer.sampling.p_active — 10×10 Bernoulli grid ───────────
  function renderBernoulli(slot, store) {
    clearSlot(slot);
    const root = wrap("bernoulli");
    const N = 10;
    const CELL = 10;
    const PAD = 2;
    const sz = N * (CELL + PAD) - PAD;
    const s = svg("svg", { viewBox: `0 0 ${sz} ${sz}`, width: sz, height: sz });
    const cells = [];
    for (let i = 0; i < N * N; i++) {
      const r = Math.floor(i / N);
      const c = i % N;
      const rect = svg("rect", {
        x: c * (CELL + PAD),
        y: r * (CELL + PAD),
        width: CELL,
        height: CELL,
        rx: 1,
        fill: "var(--border-subtle)",
      });
      s.appendChild(rect);
      cells.push(rect);
    }
    root.appendChild(s);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const p = clamp(Number(st.getKnob("optimizer.sampling.p_active")) || 0, 0, 1);
      const lit = Math.round(p * 100);
      // Use a deterministic permutation so the grid grows monotonically.
      const rng = deterministicRng(hashFloat(Math.round(p * 10000)));
      const order = cells.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
      cells.forEach((c) => c.setAttribute("fill", "var(--border-subtle)"));
      for (let i = 0; i < lit; i++) {
        cells[order[i]].setAttribute("fill", "var(--accent)");
      }
      cap.textContent = `p_active = ${(p * 100).toFixed(1)}% · avg ${lit}/100 genes active per individual`;
    }
    update(store);
    slot._update = update;
  }

  // ── pop_size ─────────────────────────────────────────────────────
  function renderPopulationGrid(slot, store) {
    clearSlot(slot);
    const root = wrap("pop-grid");
    const svgWrap = h("div", { style: { minHeight: "44px" } });
    const callout = h("div", {
      style: {
        display: "none",
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--accent)",
        fontVariantNumeric: "tabular-nums",
      },
    });
    root.appendChild(svgWrap);
    root.appendChild(callout);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const pop = clamp(Math.round(Number(st.getKnob("pop_size")) || 0), 0, 100000);
      svgWrap.replaceChildren();
      if (pop > 200) {
        svgWrap.style.display = "none";
        callout.style.display = "block";
        callout.textContent = `${fmtInt(pop)} individuals`;
        cap.textContent = "(too many dots to render; population callout instead)";
      } else {
        svgWrap.style.display = "block";
        callout.style.display = "none";
        // Wrap into a ~25-cols grid.
        const cols = Math.min(25, Math.max(8, Math.ceil(Math.sqrt(pop * 2.5))));
        const rows = Math.ceil(pop / cols);
        const CELL = 6;
        const PAD = 2;
        const W = cols * (CELL + PAD) - PAD;
        const H = rows * (CELL + PAD) - PAD;
        const s = svg("svg", { viewBox: `0 0 ${W} ${Math.max(H, 1)}`, width: "100%", height: H });
        for (let i = 0; i < pop; i++) {
          const r = Math.floor(i / cols);
          const c = i % cols;
          s.appendChild(
            svg("circle", {
              cx: c * (CELL + PAD) + CELL / 2,
              cy: r * (CELL + PAD) + CELL / 2,
              r: CELL / 2,
              fill: "var(--accent)",
              opacity: 0.85,
            }),
          );
        }
        svgWrap.appendChild(s);
        cap.textContent = `${pop} individuals per generation`;
      }
    }
    update(store);
    slot._update = update;
  }

  // ── generations ──────────────────────────────────────────────────
  function renderGenerationsAxis(slot, store) {
    clearSlot(slot);
    const root = wrap("gens-axis");
    const W = 240;
    const H = 28;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    s.style.display = "block";
    // Define a linear gradient from accent → transparent for the fill.
    const defs = svg("defs", {});
    const lg = svg("linearGradient", { id: "gens-grad", x1: 0, y1: 0, x2: 1, y2: 0 });
    lg.appendChild(svg("stop", { offset: "0", "stop-color": "var(--accent)", "stop-opacity": 0.0 }));
    lg.appendChild(svg("stop", { offset: "1", "stop-color": "var(--accent)", "stop-opacity": 0.55 }));
    defs.appendChild(lg);
    s.appendChild(defs);
    const fill = svg("rect", { x: 4, y: H / 2 - 4, height: 8, fill: "url(#gens-grad)", rx: 2 });
    s.appendChild(fill);
    s.appendChild(svg("line", { x1: 4, x2: W - 4, y1: H / 2, y2: H / 2, stroke: "var(--border-subtle)" }));
    const ticks = svg("g", {});
    s.appendChild(ticks);
    root.appendChild(s);
    const cap = captionEl("");
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const g = clamp(Math.round(Number(st.getKnob("generations")) || 0), 0, 100000);
      while (ticks.firstChild) ticks.removeChild(ticks.firstChild);
      const tickStep = g <= 50 ? 10 : g <= 200 ? 25 : g <= 1000 ? 100 : 500;
      const padL = 4;
      const padR = 4;
      const usable = W - padL - padR;
      for (let v = 0; v <= g; v += tickStep) {
        const x = padL + (v / Math.max(1, g)) * usable;
        ticks.appendChild(
          svg("line", { x1: x, x2: x, y1: H / 2 - 4, y2: H / 2 + 4, stroke: "var(--fg-muted)", "stroke-width": 0.7 }),
        );
      }
      // Last tick = current value (accent).
      const lastX = padL + usable;
      ticks.appendChild(
        svg("line", { x1: lastX, x2: lastX, y1: H / 2 - 7, y2: H / 2 + 7, stroke: "var(--accent)", "stroke-width": 2 }),
      );
      ticks.appendChild(
        svg("text", {
          x: lastX,
          y: H / 2 + 18,
          "text-anchor": "end",
          "font-family": "var(--font-mono)",
          "font-size": 9,
          fill: "var(--accent)",
        }),
      ).textContent = String(g);
      fill.setAttribute("width", Math.max(0, lastX - 4));
      cap.textContent = `AGE-MOEA-II runs for ${g} generation${g === 1 ? "" : "s"} per seed.`;
    }
    update(store);
    slot._update = update;
  }

  // ── score_full_categories ────────────────────────────────────────
  function renderCategoryStrip(slot, store) {
    clearSlot(slot);
    const root = wrap("cat-strip");
    function strip(labelText) {
      const wrapEl = h("div", {
        style: {
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          padding: "5px",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle)",
          transition: "border-color 140ms ease",
        },
      });
      const inner = h("div", {
        style: { display: "flex", gap: "2px", height: "8px", overflow: "hidden" },
      });
      const label = h(
        "div",
        {
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-mono-sm)",
            color: "var(--fg-muted)",
          },
        },
        labelText,
      );
      wrapEl.appendChild(inner);
      wrapEl.appendChild(label);
      return { root: wrapEl, inner, label };
    }
    const off = strip("off · 2 categories (anchor, target)");
    const on = strip("on · all N categories scored");
    // off — always 2 cells
    for (let i = 0; i < 2; i++) {
      off.inner.appendChild(
        h("div", {
          style: { flex: "1", background: i === 0 ? "var(--accent)" : "var(--accent-muted)", borderRadius: "2px" },
        }),
      );
    }
    root.appendChild(off.root);
    root.appendChild(on.root);
    slot.appendChild(root);

    function update(st) {
      const v = !!st.getKnob("score_full_categories");
      const cats = Math.max(2, Number(st.getKnob("n_categories")) || 50);
      // Rebuild on-strip cells if count changed.
      const drawn = Math.min(cats, 60);
      if (on.inner.childElementCount !== drawn) {
        on.inner.replaceChildren();
        for (let i = 0; i < drawn; i++) {
          on.inner.appendChild(
            h("div", {
              style: {
                flex: "1",
                background: "var(--accent)",
                opacity: 0.4 + 0.55 * (i / drawn),
                borderRadius: "1px",
              },
            }),
          );
        }
      }
      off.root.style.borderColor = !v ? "var(--accent)" : "var(--border-subtle)";
      off.root.style.background = !v ? "var(--accent-soft)" : "transparent";
      on.root.style.borderColor = v ? "var(--accent)" : "var(--border-subtle)";
      on.root.style.background = v ? "var(--accent-soft)" : "transparent";
      on.label.textContent = `on · ${cats} categor${cats === 1 ? "y" : "ies"} scored`;
    }
    update(store);
    slot._update = update;
  }

  // ── stage1.budget_sut_calls ──────────────────────────────────────
  function renderBudgetCallout(slot, store) {
    clearSlot(slot);
    const root = wrap("budget");
    const big = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--accent)",
        fontVariantNumeric: "tabular-nums",
        letterSpacing: "0.02em",
      },
    });
    const sub = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-sm)",
        color: "var(--fg-muted)",
      },
    });
    root.appendChild(big);
    root.appendChild(sub);
    slot.appendChild(root);

    function update(st) {
      const b = Math.max(0, Number(st.getKnob("stage1.budget_sut_calls")) || 0);
      big.textContent = `${fmtInt(b)} SUT calls / seed`;
      // Compute "~N per strategy" if strategies are visible.
      const strategies = st.getKnob("stage1.strategies") || [];
      if (Array.isArray(strategies) && strategies.length > 0) {
        const totalW = strategies.reduce((a, s) => a + (Number(s?.weight) || 0), 0);
        if (totalW > 0) {
          const active = strategies.filter((s) => (Number(s?.weight) || 0) > 0);
          const perStrategy = Math.round(b / Math.max(1, active.length));
          sub.textContent = `÷ ${active.length} active strateg${active.length === 1 ? "y" : "ies"} ≈ ${fmtInt(perStrategy)} / strategy`;
          return;
        }
      }
      sub.textContent = "stage 1 total budget";
    }
    update(store);
    slot._update = update;
  }

  // ── stage1.max_flips_per_seed ────────────────────────────────────
  function renderFlipCounter(slot, store) {
    clearSlot(slot);
    const root = wrap("flip-counter");
    const row = h("div", {
      style: { display: "flex", alignItems: "center", gap: "8px" },
    });
    const glyph = svg("svg", { viewBox: "0 0 18 18", width: 18, height: 18 });
    glyph.appendChild(svg("path", { d: "M3 9 L9 3 L9 6 L15 6 L15 12 L9 12 L9 15 Z", fill: "var(--accent)" }));
    const big = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--accent)",
        fontVariantNumeric: "tabular-nums",
      },
    });
    const sub = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-sm)",
        color: "var(--fg-muted)",
      },
    });
    row.appendChild(glyph);
    row.appendChild(big);
    root.appendChild(row);
    root.appendChild(sub);
    slot.appendChild(root);

    function update(st) {
      const n = Math.max(0, Number(st.getKnob("stage1.max_flips_per_seed")) || 0);
      big.textContent = `${fmtInt(n)} flips / seed`;
      sub.textContent = "cap on distinct label flips recorded per seed";
    }
    update(store);
    slot._update = update;
  }

  // ── stage1.max_distinct_targets ──────────────────────────────────
  function renderTargetCounter(slot, store) {
    clearSlot(slot);
    const root = wrap("target-counter");
    const row = h("div", {
      style: { display: "flex", alignItems: "center", gap: "8px" },
    });
    const glyph = svg("svg", { viewBox: "0 0 18 18", width: 18, height: 18 });
    glyph.appendChild(svg("circle", { cx: 9, cy: 9, r: 7, fill: "none", stroke: "var(--accent)", "stroke-width": 1.5 }));
    glyph.appendChild(svg("circle", { cx: 9, cy: 9, r: 3.5, fill: "none", stroke: "var(--accent)", "stroke-width": 1.5 }));
    glyph.appendChild(svg("circle", { cx: 9, cy: 9, r: 1.2, fill: "var(--accent)" }));
    const big = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--accent)",
        fontVariantNumeric: "tabular-nums",
      },
    });
    const sub = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-sm)",
        color: "var(--fg-muted)",
      },
    });
    row.appendChild(glyph);
    row.appendChild(big);
    root.appendChild(row);
    root.appendChild(sub);
    slot.appendChild(root);

    function update(st) {
      const n = Math.max(0, Number(st.getKnob("stage1.max_distinct_targets")) || 0);
      big.textContent = `${fmtInt(n)} distinct targets`;
      sub.textContent = "cap on flip-target diversity per seed";
    }
    update(store);
    slot._update = update;
  }

  // ── stage1.strategies — pie chart ────────────────────────────────
  function renderStrategyPie(slot, store) {
    const PALETTE = [
      "#7cfa9c", // accent (highest)
      "#5a8c70",
      "#6e7888",
      "#8a6e72",
      "#7a8076",
      "#5c6678",
      "#88837e",
      "#646a6c",
    ];
    clearSlot(slot);
    const root = wrap("strategy-pie");
    const row = h("div", { style: { display: "flex", gap: "10px", alignItems: "center" } });
    const W = 80;
    const cx = W / 2;
    const cy = W / 2;
    const r = 32;
    const s = svg("svg", { viewBox: `0 0 ${W} ${W}`, width: W, height: W });
    const sliceGroup = svg("g", {});
    s.appendChild(sliceGroup);
    s.appendChild(svg("circle", { cx, cy, r: r - 10, fill: "var(--bg-elevated)" }));
    row.appendChild(s);
    const legend = h("div", {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "2px",
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-sm)",
        color: "var(--fg-secondary)",
        flex: "1",
        minWidth: 0,
      },
    });
    row.appendChild(legend);
    root.appendChild(row);
    slot.appendChild(root);

    function arcPath(a0, a1, large) {
      const x0 = cx + r * Math.sin(a0);
      const y0 = cy - r * Math.cos(a0);
      const x1 = cx + r * Math.sin(a1);
      const y1 = cy - r * Math.cos(a1);
      return `M ${cx} ${cy} L ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} Z`;
    }

    function update(st) {
      const strategies = st.getKnob("stage1.strategies") || [];
      const arr = Array.isArray(strategies) ? strategies : [];
      while (sliceGroup.firstChild) sliceGroup.removeChild(sliceGroup.firstChild);
      legend.replaceChildren();
      const totalW = arr.reduce((a, x) => a + (Number(x?.weight) || 0), 0);
      if (totalW <= 0) {
        sliceGroup.appendChild(svg("circle", { cx, cy, r, fill: "var(--border-subtle)" }));
        legend.appendChild(h("span", { style: { color: "var(--fg-muted)" } }, "(all weights 0)"));
        return;
      }
      // Sort descending by weight; highest gets accent.
      const sorted = arr
        .map((x, i) => ({ ...x, _i: i, w: Number(x.weight) || 0 }))
        .sort((a, b) => b.w - a.w);
      let acc = 0;
      sorted.forEach((s2, i) => {
        if (s2.w <= 0) return;
        const a0 = (acc / totalW) * Math.PI * 2;
        acc += s2.w;
        const a1 = (acc / totalW) * Math.PI * 2;
        const large = a1 - a0 > Math.PI ? 1 : 0;
        const colour = PALETTE[Math.min(i, PALETTE.length - 1)];
        sliceGroup.appendChild(svg("path", { d: arcPath(a0, a1, large), fill: colour, opacity: 0.92 }));
        const item = h("div", {
          style: { display: "flex", gap: "6px", alignItems: "center", whiteSpace: "nowrap", overflow: "hidden" },
        });
        item.appendChild(
          h("span", {
            style: { width: "8px", height: "8px", borderRadius: "2px", background: colour, flexShrink: "0" },
          }),
        );
        item.appendChild(
          h(
            "span",
            { style: { overflow: "hidden", textOverflow: "ellipsis", flex: "1" } },
            `${s2.name ?? "?"} ${(s2.w * 100 / totalW).toFixed(0)}%`,
          ),
        );
        legend.appendChild(item);
      });
    }
    update(store);
    slot._update = update;
  }

  // ── stage2.budget_sut_calls_per_flip ─────────────────────────────
  function renderStage2Budget(slot, store) {
    clearSlot(slot);
    const root = wrap("stage2-budget");
    const big = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--accent)",
        fontVariantNumeric: "tabular-nums",
      },
    });
    const sub = h("div", {
      style: { fontFamily: "var(--font-mono)", fontSize: "var(--text-mono-sm)", color: "var(--fg-muted)" },
    });
    root.appendChild(big);
    root.appendChild(sub);
    slot.appendChild(root);

    function update(st) {
      const b = Math.max(0, Number(st.getKnob("stage2.budget_sut_calls_per_flip")) || 0);
      big.textContent = `${fmtInt(b)} SUT calls / flip`;
      sub.textContent = "stage 2 refinement budget per recorded flip";
    }
    update(store);
    slot._update = update;
  }

  // ── distances.d_i_primary / d_o_primary formula cards ────────────
  function renderFormulaCard(slot, store, knobPath, formulas, captions) {
    clearSlot(slot);
    const root = wrap("formula");
    root.style.padding = "8px";
    root.style.border = "1px solid var(--border-subtle)";
    root.style.borderRadius = "6px";
    root.style.background = "var(--bg-elevated)";
    const math = h("div", {
      style: {
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-mono-lg)",
        color: "var(--fg-primary)",
        textAlign: "center",
        padding: "4px 0",
      },
    });
    const cap = captionEl("");
    root.appendChild(math);
    root.appendChild(cap);
    slot.appendChild(root);

    function update(st) {
      const v = st.getKnob(knobPath);
      const fall = Object.keys(formulas)[0];
      const key = v && formulas[v] ? v : fall;
      math.replaceChildren();
      const expr = formulas[key];
      if (window.katex && expr.tex) {
        try {
          window.katex.render(expr.tex, math, { throwOnError: false, displayMode: false });
        } catch (_) {
          math.textContent = expr.mono;
        }
      } else {
        math.textContent = expr.mono;
      }
      cap.textContent = captions[key] || key;
    }
    update(store);
    slot._update = update;
  }

  function renderDIFormula(slot, store) {
    const formulas = {
      rank_sum: { mono: "∑ᵢ gᵢ", tex: "\\sum_i g_i" },
      sparsity: { mono: "(1/n) ∑ᵢ 𝟙[gᵢ > 0]", tex: "\\frac{1}{n}\\sum_i \\mathbf{1}[g_i > 0]" },
      hamming: { mono: "∑ᵢ 𝟙[gᵢ ≠ aᵢ]", tex: "\\sum_i \\mathbf{1}[g_i \\neq a_i]" },
      weighted_content: { mono: "∑ᵢ wᵢ · gᵢ", tex: "\\sum_i w_i \\cdot g_i" },
      image_pixel_L2: { mono: "‖x(g) − x(a)‖₂", tex: "\\lVert x(g) - x(a) \\rVert_2" },
    };
    const captions = {
      rank_sum: "sum of gene ranks · favours large total perturbation",
      sparsity: "fraction of active genes · favours sparse perturbations",
      hamming: "edits relative to anchor · raw bit-diff count",
      weighted_content: "rank-weighted by per-gene saliency / content",
      image_pixel_L2: "L2 in pixel space (post-manipulation image)",
    };
    renderFormulaCard(slot, store, "distances.d_i_primary", formulas, captions);
  }

  function renderDOFormula(slot, store) {
    const formulas = {
      label_mismatch: { mono: "𝟙[L_anchor ≠ L_target]", tex: "\\mathbf{1}[L_{anchor} \\neq L_{target}]" },
      label_edit: { mono: "Levenshtein(L_anchor, L_target)", tex: "\\mathrm{Levenshtein}(L_{anchor}, L_{target})" },
    };
    const captions = {
      label_mismatch: "binary: did the predicted class change at all?",
      label_edit: "string-edit distance between label sequences",
    };
    renderFormulaCard(slot, store, "distances.d_o_primary", formulas, captions);
  }

  // ─────────────────────────────────────────────────────────────────
  // REGISTRY · path → renderer
  // ─────────────────────────────────────────────────────────────────
  const VIZ_REGISTRY = {
    modality: renderModalityBadge,
    "image.patch_ratio": renderPatchGrid,
    "image.n_candidates": renderDepthSparkline,
    "image.patch_strategy": renderPatchStrategy,
    "image.candidate_strategy": renderCandidateStrategy,
    "text.composite.profile": renderTextProfile,
    "seeds.mode": renderSeedsMode,
    "seeds.gap_filter.max_logprob_gap": renderGapCurve,
    "seeds.gap_filter.n_per_class": renderCounter,
    "optimizer.sampling.mode": renderSamplerMode,
    "optimizer.sampling.tiers": renderTierHistogram,
    "optimizer.sampling.zero_anchor_fraction": renderZeroAnchorPie,
    "optimizer.sampling.p_active": renderBernoulli,
    pop_size: renderPopulationGrid,
    generations: renderGenerationsAxis,
    score_full_categories: renderCategoryStrip,
    "stage1.budget_sut_calls": renderBudgetCallout,
    "stage1.max_flips_per_seed": renderFlipCounter,
    "stage1.max_distinct_targets": renderTargetCounter,
    "stage1.strategies": renderStrategyPie,
    "stage2.budget_sut_calls_per_flip": renderStage2Budget,
    "distances.d_i_primary": renderDIFormula,
    "distances.d_o_primary": renderDOFormula,
  };

  // ─────────────────────────────────────────────────────────────────
  // MOUNT / UPDATE LIFECYCLE
  // ─────────────────────────────────────────────────────────────────
  function mountAll() {
    const store = window.Alpine && window.Alpine.store && window.Alpine.store("explorer");
    if (!store || !store.ready) return;
    const slots = document.querySelectorAll("[data-viz-slot]");
    slots.forEach((slot) => {
      const path = slot.getAttribute("data-path");
      const renderer = VIZ_REGISTRY[path];
      if (!renderer) {
        // No viz for this hero path — leave the slot empty.
        return;
      }
      if (store.getKnob(path) === undefined) {
        // Knob hasn't been initialised yet — skip; we'll be re-invoked.
        return;
      }
      // Skip if already mounted with this path. The DOM element identity is
      // sufficient because Alpine recreates the slot when focus changes.
      if (slot._mountedPath === path && slot._update) return;
      try {
        renderer(slot, store);
        slot._mountedPath = path;
      } catch (err) {
        // Swallow to avoid breaking the page on a single viz bug.
        if (window.console) console.warn("[vizzes] mount failed for", path, err);
      }
    });
  }

  function updateAll(store) {
    if (!store || !store.ready) return;
    const slots = document.querySelectorAll("[data-viz-slot]");
    slots.forEach((slot) => {
      if (typeof slot._update !== "function") return;
      try {
        slot._update(store);
      } catch (err) {
        if (window.console) console.warn("[vizzes] update failed", err);
      }
    });
  }

  // rAF-coalesced update scheduler.
  let _rafPending = false;
  let _pendingStore = null;
  function rafSchedule(fn, store) {
    _pendingStore = store;
    if (_rafPending) return;
    _rafPending = true;
    requestAnimationFrame(() => {
      _rafPending = false;
      const s = _pendingStore;
      _pendingStore = null;
      fn(s);
    });
  }

  // ── event wiring ─────────────────────────────────────────────────
  window.addEventListener("explorer:ready", () => {
    requestAnimationFrame(mountAll);
  });
  window.addEventListener("explorer:focus-changed", () => {
    // Alpine destroys + recreates the slot DOM on focus change. Wait a tick.
    requestAnimationFrame(mountAll);
  });
  window.addEventListener("explorer:pipeline-changed", () => {
    requestAnimationFrame(mountAll);
  });
  window.addEventListener("explorer:config-changed", () => {
    const store = window.Alpine && window.Alpine.store && window.Alpine.store("explorer");
    if (!store) return;
    rafSchedule(updateAll, store);
  });

  // Some knobs render below the fold or in advanced disclosures that open
  // later. We watch for new slot nodes appearing under the detail panel and
  // remount them as they show up.
  document.addEventListener(
    "DOMContentLoaded",
    () => {
      const target = document.getElementById("detail-panel") || document.body;
      const mo = new MutationObserver(() => {
        rafSchedule(() => {
          mountAll();
        }, window.Alpine && window.Alpine.store && window.Alpine.store("explorer"));
      });
      mo.observe(target, { childList: true, subtree: true });
    },
    { once: true },
  );

  // Expose registry for debugging.
  window.PipelineExplorer = window.PipelineExplorer || {};
  window.PipelineExplorer.vizRegistry = VIZ_REGISTRY;
})();
