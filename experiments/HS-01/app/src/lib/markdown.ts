/**
 * markdown.ts — a tiny, dependency-free renderer for the consent file.
 *
 * Supports ONLY the syntax the consent markdown uses: a single `#` H1, `**bold**`
 * inline emphasis, `-` bullet lists, and paragraphs. Anything else is rendered
 * as literal text. Inline content is split into React nodes (never HTML), so an
 * angle-bracket literal like `<researcher email>` is shown verbatim and cannot
 * inject markup.
 *
 * This is deliberately minimal: pulling in a full markdown engine for one
 * static file is not worth the surface area.
 */

import type { ReactNode } from "react";
import { createElement } from "react";

export type ConsentBlock =
  | { type: "h1"; children: ReactNode[] }
  | { type: "p"; children: ReactNode[] }
  | { type: "ul"; items: ReactNode[][] };

/** Split a line into text / <strong> nodes on **…** spans. */
export function inlineToNodes(line: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /\*\*([^*]+)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) nodes.push(line.slice(last, m.index));
    nodes.push(createElement("strong", { key: key++ }, m[1]));
    last = re.lastIndex;
  }
  if (last < line.length) nodes.push(line.slice(last));
  return nodes.length > 0 ? nodes : [""];
}

export function renderConsentMarkdown(md: string): ConsentBlock[] {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const blocks: ConsentBlock[] = [];

  let para: string[] = [];
  let list: ReactNode[][] | null = null;

  const flushPara = () => {
    if (para.length > 0) {
      blocks.push({ type: "p", children: inlineToNodes(para.join(" ")) });
      para = [];
    }
  };
  const flushList = () => {
    if (list && list.length > 0) {
      blocks.push({ type: "ul", items: list });
    }
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.trim() === "") {
      flushPara();
      flushList();
      continue;
    }
    if (line.startsWith("# ")) {
      flushPara();
      flushList();
      blocks.push({ type: "h1", children: inlineToNodes(line.slice(2).trim()) });
      continue;
    }
    if (line.startsWith("- ")) {
      flushPara();
      if (!list) list = [];
      list.push(inlineToNodes(line.slice(2).trim()));
      continue;
    }
    // Indented continuation of the current bullet (a wrapped list line). Append
    // it to the last item rather than flushing the list into a spurious
    // paragraph — which would split one list into two and break the rhythm
    // (the consent file wraps its "anonymous" bullet across two lines).
    if (list && list.length > 0 && /^\s/.test(raw)) {
      const i = list.length - 1;
      list[i] = [...list[i], " ", ...inlineToNodes(line.trim())];
      continue;
    }
    // paragraph line
    flushList();
    para.push(line.trim());
  }
  flushPara();
  flushList();

  return blocks;
}
