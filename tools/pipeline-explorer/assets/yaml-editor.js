/*
 * YAML editor + cross-link bridge.
 * Round-trips $store.explorer.config to a textarea, supports edit-back,
 * line↔path mapping, and hover sync with the detail panel + canvas.
 */
(function () {
  "use strict";

  // ── module state ───────────────────────────────────────────────────
  let pathToLine = new Map();
  let lineToPath = new Map();
  let lineToNode = new Map();          // line → node_id (canvas pulse)
  let leafByPath = new Map();          // dotted path → schema leaf
  let prevYaml = "";
  let inEditBack = false;              // true while we apply textarea edits
  let debounceTimer = null;
  let highlightRafId = null;

  // DOM refs (resolved in init)
  let elInput, elHighlight, elGutter;
  let elLineCount, elStatusPill, elStatusLabel, elErrorBar;
  let elDownloadBtn, elValidateBtn, elFilterToggle;

  const DUMP_OPTS = {
    lineWidth: 100,
    noRefs: true,
    sortKeys: false,
    quotingType: '"',
    forceQuotes: false,
  };

  // ── utilities ──────────────────────────────────────────────────────

  /** Map dotted scalar-key path ↔ line number across the dumped YAML. */
  function buildLineIndex(text) {
    const pToL = new Map(), lToP = new Map(), lToN = new Map();
    const stack = []; // [{indent, name}]
    const keyRe = /^(\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:/;
    const listKeyRe = /^(\s*)-\s+([A-Za-z_][A-Za-z0-9_-]*)\s*:/;
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const listMatch = line.match(listKeyRe);
      const m = listMatch || line.match(keyRe);
      if (!m) continue;
      const indent = m[1].length + (listMatch ? 2 : 0);
      while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
      stack.push({ indent, name: m[2] });
      const path = stack.map((s) => s.name).join(".");
      const lineNo = i + 1;
      pToL.set(path, lineNo);
      lToP.set(lineNo, path);
      const leaf = leafByPath.get(path);
      if (leaf && leaf.node_id) lToN.set(lineNo, leaf.node_id);
    }
    return { pathToLine: pToL, lineToPath: lToP, lineToNode: lToN };
  }

  const escHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /** Wrap every source line in a `.yaml-hl-line` block for paint targeting. */
  function highlight(text) {
    const lines = text.split("\n");
    const out = new Array(lines.length);
    for (let i = 0; i < lines.length; i++) {
      out[i] =
        '<span class="yaml-hl-line" data-line="' + (i + 1) + '">' +
        highlightLineHtml(lines[i]) + "</span>";
    }
    return out.join("\n") + "\n";
  }

  function highlightLineHtml(line) {
    if (line.length === 0) return "​"; // zero-width keeps line height
    const hashIdx = findCommentIdx(line);
    let preComment = line, commentPart = "";
    if (hashIdx >= 0) {
      preComment = line.slice(0, hashIdx);
      commentPart = '<span class="tok-comment">' + escHtml(line.slice(hashIdx)) + "</span>";
    }
    let prefix = "", body = preComment;
    const dashM = body.match(/^(\s*)-\s+/);
    if (dashM) {
      prefix = escHtml(dashM[1]) + '<span class="tok-dash">- </span>';
      body = body.slice(dashM[0].length);
    }
    const keyM = body.match(/^(\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*):(.*)$/);
    if (keyM) {
      const [, lead, key, gap, rest] = keyM;
      return (
        prefix + escHtml(lead) +
        '<span class="tok-key">' + escHtml(key) + "</span>" +
        escHtml(gap) + '<span class="tok-colon">:</span>' +
        highlightValue(rest) + commentPart
      );
    }
    return prefix + highlightValue(body) + commentPart;
  }

  /** First `#` not inside a string, with whitespace boundary. */
  function findCommentIdx(line) {
    let inStr = false, quote = "";
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (inStr) {
        if (c === "\\") { i++; continue; }
        if (c === quote) inStr = false;
        continue;
      }
      if (c === '"' || c === "'") { inStr = true; quote = c; continue; }
      if (c === "#" && (i === 0 || /\s/.test(line[i - 1]))) return i;
    }
    return -1;
  }

  function highlightValue(raw) {
    if (!raw) return "";
    let out = "", i = 0;
    while (i < raw.length) {
      const c = raw[i];
      if (c === '"' || c === "'") {
        const quote = c;
        let j = i + 1;
        while (j < raw.length) {
          if (raw[j] === "\\") { j += 2; continue; }
          if (raw[j] === quote) { j++; break; }
          j++;
        }
        out += '<span class="tok-string">' + escHtml(raw.slice(i, j)) + "</span>";
        i = j;
        continue;
      }
      if ("[]{},".indexOf(c) !== -1) {
        out += '<span class="tok-bracket">' + escHtml(c) + "</span>";
        i++;
        continue;
      }
      let j = i;
      while (j < raw.length && "\"'[]{},".indexOf(raw[j]) === -1) j++;
      out += atomiseValue(raw.slice(i, j));
      i = j;
    }
    return out;
  }

  function atomiseValue(chunk) {
    return chunk.replace(
      /(\s+)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(true|false|yes|no|on|off)\b|(null|~)\b|([^\s,{}\[\]]+)/g,
      (_, ws, num, bool, nul, word) => {
        if (ws) return escHtml(ws);
        if (num) return '<span class="tok-number">' + escHtml(num) + "</span>";
        if (bool) return '<span class="tok-bool">' + escHtml(bool) + "</span>";
        if (nul) return '<span class="tok-null">' + escHtml(nul) + "</span>";
        if (word) return '<span class="tok-value">' + escHtml(word) + "</span>";
        return "";
      },
    );
  }

  /** 1-based indices of lines that differ between two YAML strings. */
  function diffLines(a, b) {
    const A = a ? a.split("\n") : [], B = b ? b.split("\n") : [];
    const max = Math.max(A.length, B.length);
    const out = [];
    for (let i = 0; i < max; i++) if (A[i] !== B[i]) out.push(i + 1);
    return out;
  }

  /** Flatten any nested object → [dottedPath, leafValue] iterator. */
  function* walkLeaves(obj, prefix = "") {
    if (obj === null || obj === undefined || typeof obj !== "object" || Array.isArray(obj)) {
      yield [prefix, obj];
      return;
    }
    for (const key of Object.keys(obj)) {
      const sub = prefix ? prefix + "." + key : key;
      yield* walkLeaves(obj[key], sub);
    }
  }

  function leafEqual(a, b) {
    if (a === b) return true;
    if (a === null || b === null) return false;
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) if (!leafEqual(a[i], b[i])) return false;
      return true;
    }
    if (typeof a === "object" && typeof b === "object") {
      try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; }
    }
    return false;
  }

  /** Coerce strings into the schema-declared numeric type when applicable. */
  function coerce(value, path) {
    const leaf = leafByPath.get(path);
    if (!leaf || value === null || value === undefined) return value;
    if (typeof value !== "string") return value;
    if (leaf.type === "int") {
      const n = parseInt(value, 10);
      if (!Number.isNaN(n)) return n;
    }
    if (leaf.type === "float") {
      const n = parseFloat(value);
      if (!Number.isNaN(n)) return n;
    }
    return value;
  }

  // ── rendering ──────────────────────────────────────────────────────

  function scheduleRender() {
    if (highlightRafId !== null) return;
    highlightRafId = requestAnimationFrame(() => {
      highlightRafId = null;
      renderHighlight();
      renderGutter();
      syncScroll();
    });
  }

  function renderHighlight() {
    elHighlight.innerHTML = highlight(elInput.value);
    const hoverPath = Alpine.store("explorer").hoverKnob;
    if (hoverPath) {
      const ln = pathToLine.get(hoverPath);
      if (ln) applyHoverClass(ln, false);
    }
  }

  function renderGutter() {
    const total = elInput.value.split("\n").length;
    const digits = Math.max(2, String(total).length);
    elGutter.style.minWidth = digits + 1 + "ch";
    const existing = elGutter.children.length;
    if (existing < total) {
      const frag = document.createDocumentFragment();
      for (let i = existing; i < total; i++) {
        const el = document.createElement("span");
        el.className = "yaml-gutter__line";
        el.dataset.line = String(i + 1);
        el.textContent = String(i + 1);
        frag.appendChild(el);
      }
      elGutter.appendChild(frag);
    } else if (existing > total) {
      while (elGutter.children.length > total) elGutter.removeChild(elGutter.lastChild);
    }
    if (elLineCount) elLineCount.textContent = total + " line" + (total === 1 ? "" : "s");
  }

  function syncScroll() {
    elHighlight.scrollTop = elInput.scrollTop;
    elHighlight.scrollLeft = elInput.scrollLeft;
    elGutter.scrollTop = elInput.scrollTop;
  }

  // ── hover paint ────────────────────────────────────────────────────

  function clearHoverPaint() {
    elHighlight.querySelectorAll(".yaml-hl-line--hover").forEach((el) =>
      el.classList.remove("yaml-hl-line--hover"));
    elGutter.querySelectorAll(".yaml-gutter__line--hover").forEach((el) =>
      el.classList.remove("yaml-gutter__line--hover"));
  }

  function applyHoverClass(lineNo, scrollIntoView) {
    clearHoverPaint();
    const hl = elHighlight.querySelector('.yaml-hl-line[data-line="' + lineNo + '"]');
    const g = elGutter.querySelector('.yaml-gutter__line[data-line="' + lineNo + '"]');
    if (hl) {
      hl.classList.add("yaml-hl-line--hover");
      if (scrollIntoView) {
        const lineH = hl.getBoundingClientRect().height || 20;
        const targetTop = (lineNo - 1) * lineH - elInput.clientHeight / 3;
        elInput.scrollTop = Math.max(0, targetTop);
        syncScroll();
      }
    }
    if (g) g.classList.add("yaml-gutter__line--hover");
  }

  // ── diff bar animation ─────────────────────────────────────────────

  function flashDiffLines(changedLines) {
    if (!changedLines.length) return;
    const motion = window.Motion;
    for (const ln of changedLines) {
      const g = elGutter.querySelector('.yaml-gutter__line[data-line="' + ln + '"]');
      const hl = elHighlight.querySelector('.yaml-hl-line[data-line="' + ln + '"]');
      if (g) {
        g.classList.add("yaml-gutter__line--diff");
        if (motion && motion.animate) {
          motion.animate(g, { opacity: [1, 0.85, 1] }, { duration: 0.8 });
        }
        setTimeout(() => g.classList.remove("yaml-gutter__line--diff"), 800);
      }
      if (hl) {
        hl.classList.add("yaml-hl-line--diff");
        if (motion && motion.animate) {
          motion.animate(hl, { opacity: [0.55, 1] },
            { duration: 0.8, easing: "ease-out" });
        }
        setTimeout(() => hl.classList.remove("yaml-hl-line--diff"), 800);
      }
    }
  }

  // ── status pill ────────────────────────────────────────────────────

  function setStatus(state, message) {
    if (!elStatusPill) return;
    elStatusPill.classList.toggle("pill-valid--invalid", state === "invalid");
    if (elStatusLabel) {
      elStatusLabel.textContent =
        state === "invalid" ? "invalid ✗" :
        state === "flash" ? "valid ✓" : "valid";
    }
    if (state === "flash") {
      elStatusPill.classList.add("pill-valid--flash");
      setTimeout(() => elStatusPill.classList.remove("pill-valid--flash"), 200);
    }
    if (elErrorBar) {
      if (state === "invalid" && message) {
        elErrorBar.hidden = false;
        elErrorBar.textContent = message;
      } else {
        elErrorBar.hidden = true;
        elErrorBar.textContent = "";
      }
    }
  }

  function paintErrorLine(lineNo) {
    elHighlight.querySelectorAll(".yaml-hl-line--error").forEach((el) =>
      el.classList.remove("yaml-hl-line--error"));
    if (!lineNo) return;
    const hl = elHighlight.querySelector('.yaml-hl-line[data-line="' + lineNo + '"]');
    if (hl) hl.classList.add("yaml-hl-line--error");
  }

  // ── store → yaml (serialise) ───────────────────────────────────────

  function serialiseFromStore() {
    if (inEditBack) return; // suppress while parseAndPushBack is applying
    const store = Alpine.store("explorer");
    if (!store || !store.ready) return;
    let text = "";
    try { text = window.jsyaml.dump(store.config, DUMP_OPTS); }
    catch (err) { console.warn("YAML dump failed:", err); return; }
    const changed = diffLines(prevYaml, text);
    prevYaml = text;
    store.yamlText = text;
    if (elInput.value !== text) {
      const s0 = elInput.selectionStart, s1 = elInput.selectionEnd;
      elInput.value = text;
      try { elInput.setSelectionRange(s0, s1); } catch (_) { /* range shrunk */ }
    }
    const idx = buildLineIndex(text);
    pathToLine = idx.pathToLine;
    lineToPath = idx.lineToPath;
    lineToNode = idx.lineToNode;
    scheduleRender();
    requestAnimationFrame(() => flashDiffLines(changed));
    setStatus("valid");
  }

  // ── yaml → store (parse + edit-back) ───────────────────────────────

  function parseAndPushBack() {
    const store = Alpine.store("explorer");
    if (!store || !store.ready) return;
    const text = elInput.value;
    let parsed;
    try { parsed = window.jsyaml.load(text); }
    catch (err) {
      const mark = err && err.mark ? err.mark : null;
      setStatus("invalid", err && err.message ? err.message : String(err));
      paintErrorLine(mark ? mark.line + 1 : null);
      return;
    }
    setStatus("valid");
    paintErrorLine(null);
    if (parsed === null || parsed === undefined || typeof parsed !== "object") {
      prevYaml = text;
      store.yamlText = text;
      return;
    }
    const oldLeaves = new Map();
    for (const [p, v] of walkLeaves(store.config)) oldLeaves.set(p, v);
    const newLeaves = new Map();
    for (const [p, v] of walkLeaves(parsed)) newLeaves.set(p, v);

    inEditBack = true;
    try {
      for (const [p, nv] of newLeaves) {
        const cv = coerce(nv, p);
        if (!oldLeaves.has(p)) { store.setKnob(p, cv); continue; }
        if (!leafEqual(oldLeaves.get(p), cv)) store.setKnob(p, cv);
      }
    } finally {
      inEditBack = false;
    }
    prevYaml = text;
    store.yamlText = text;
    const idx = buildLineIndex(text);
    pathToLine = idx.pathToLine;
    lineToPath = idx.lineToPath;
    lineToNode = idx.lineToNode;
    scheduleRender();
  }

  // ── DOM event handlers ─────────────────────────────────────────────

  function onInput() {
    scheduleRender();
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(parseAndPushBack, 200);
  }

  function onScroll() { syncScroll(); }

  function onKnobHover(e) {
    const path = e && e.detail ? e.detail.path : null;
    if (!path) return;
    const lineNo = pathToLine.get(path);
    if (lineNo) applyHoverClass(lineNo, true);
  }

  function handleYamlLineEnter(lineNo) {
    const store = Alpine.store("explorer");
    store.setHoverYamlLine(lineNo);
    const path = lineToPath.get(lineNo) || null;
    store.setHoverKnob(path);
    applyHoverClass(lineNo, false);
    const nodeId = lineToNode.get(lineNo) || null;
    if (path) {
      window.dispatchEvent(new CustomEvent("explorer:knob-hover",
        { detail: { path, node_id: nodeId } }));
    }
    window.dispatchEvent(new CustomEvent("explorer:yaml-line-hover",
      { detail: { lineNo, path, node_id: nodeId } }));
  }

  function handleYamlLineLeave() {
    const store = Alpine.store("explorer");
    store.setHoverYamlLine(null);
    store.setHoverKnob(null);
    clearHoverPaint();
  }

  // ── toolbar ────────────────────────────────────────────────────────

  function downloadYaml() {
    const store = Alpine.store("explorer");
    let text = store.yamlText || elInput.value;
    if (elFilterToggle && elFilterToggle.checked) text = filterToActivePipeline(text);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").replace(/Z$/, "");
    const filename = (store.pipeline || "config") + "_" + stamp + ".yaml";
    const blob = new Blob([text], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function filterToActivePipeline(text) {
    const store = Alpine.store("explorer");
    let parsed;
    try { parsed = window.jsyaml.load(text); } catch { return text; }
    if (!parsed || typeof parsed !== "object") return text;
    const active = store.pipeline;
    const keep = new Set();
    for (const leaf of store.schema?.leaves || []) {
      if (!leaf.pipeline || leaf.pipeline === "shared" || leaf.pipeline === active) {
        const parts = leaf.path.split(".");
        for (let i = 1; i <= parts.length; i++) keep.add(parts.slice(0, i).join("."));
      }
    }
    function prune(obj, prefix) {
      if (!obj || typeof obj !== "object" || Array.isArray(obj)) return obj;
      const out = {};
      for (const k of Object.keys(obj)) {
        const sub = prefix ? prefix + "." + k : k;
        if (!keep.has(sub)) continue;
        out[k] = prune(obj[k], sub);
      }
      return out;
    }
    try { return window.jsyaml.dump(prune(parsed, ""), DUMP_OPTS); }
    catch { return text; }
  }

  function validateNow() {
    try {
      window.jsyaml.load(elInput.value);
      setStatus("flash");
      paintErrorLine(null);
    } catch (err) {
      const mark = err && err.mark ? err.mark : null;
      setStatus("invalid", err && err.message ? err.message : String(err));
      paintErrorLine(mark ? mark.line + 1 : null);
    }
  }

  // ── lifecycle ──────────────────────────────────────────────────────

  function init() {
    elInput = document.getElementById("yaml-input");
    elHighlight = document.getElementById("yaml-highlight");
    elGutter = document.getElementById("yaml-gutter");
    elLineCount = document.getElementById("yaml-line-count");
    elStatusPill = document.getElementById("yaml-status-pill");
    elStatusLabel = document.getElementById("yaml-status-label");
    elErrorBar = document.getElementById("yaml-error-bar");
    elDownloadBtn = document.getElementById("yaml-download-btn");
    elValidateBtn = document.getElementById("yaml-validate-btn");
    elFilterToggle = document.getElementById("yaml-filter-toggle");
    if (!elInput || !elHighlight || !elGutter) return;

    const store = Alpine.store("explorer");
    if (!store) return;

    leafByPath = new Map();
    for (const leaf of store.schema?.leaves || []) leafByPath.set(leaf.path, leaf);

    elInput.addEventListener("input", onInput);
    elInput.addEventListener("scroll", onScroll, { passive: true });

    // Hover-tracking lives on the gutter (pointer-events: none on the
    // highlight layer would block this; the textarea owns clicks/drags).
    elGutter.addEventListener("mouseover", (e) => {
      const t = e.target.closest(".yaml-gutter__line");
      if (!t) return;
      const lineNo = Number(t.dataset.line);
      if (lineNo) handleYamlLineEnter(lineNo);
    });
    elGutter.addEventListener("mouseleave", handleYamlLineLeave);

    if (elDownloadBtn) elDownloadBtn.addEventListener("click", downloadYaml);
    if (elValidateBtn) elValidateBtn.addEventListener("click", validateNow);

    window.addEventListener("explorer:config-changed", serialiseFromStore);
    window.addEventListener("explorer:pipeline-changed", serialiseFromStore);
    window.addEventListener("explorer:knob-hover", onKnobHover);

    // Phase 4's hoverOut sets hoverKnob=null on the store; observe it
    // so we clear the YAML paint when the detail-panel hover ends.
    if (window.Alpine && Alpine.effect) {
      Alpine.effect(() => {
        const hk = Alpine.store("explorer").hoverKnob;
        if (hk === null && Alpine.store("explorer").hoverYamlLine === null) {
          clearHoverPaint();
        }
      });
    }

    serialiseFromStore();
  }

  if (window.Alpine && Alpine.store && Alpine.store("explorer")?.ready) {
    init();
  } else {
    window.addEventListener("explorer:ready", init, { once: true });
  }
})();
