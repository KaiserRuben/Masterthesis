/**
 * markdown tests — the consent renderer. Confirms the limited syntax maps to
 * the right block types and that inline content is split into nodes (so an
 * angle-bracket literal like "<researcher email>" stays verbatim text and is
 * never interpreted as markup).
 */

import { describe, it, expect } from "vitest";
import { renderConsentMarkdown, inlineToNodes } from "../src/lib/markdown";

describe("renderConsentMarkdown", () => {
  it("parses headings, paragraphs, and bullet lists", () => {
    const md = ["# Title", "", "A paragraph.", "", "- one", "- two", "", "End."].join("\n");
    const blocks = renderConsentMarkdown(md);
    expect(blocks.map((b) => b.type)).toEqual(["h1", "p", "ul", "p"]);
    const ul = blocks.find((b) => b.type === "ul");
    expect(ul && ul.type === "ul" && ul.items.length).toBe(2);
  });

  it("keeps an angle-bracket literal verbatim (no markup interpretation)", () => {
    const nodes = inlineToNodes("Questions: <researcher email>");
    // Plain string node, unchanged — no HTML, no entity folding.
    expect(nodes).toContain("Questions: <researcher email>");
  });

  it("splits **bold** into a strong element plus surrounding text", () => {
    const nodes = inlineToNodes("you can stop at **any time** now");
    expect(nodes.length).toBe(3);
    // The middle node is a React element (object), the outer two are strings.
    expect(typeof nodes[0]).toBe("string");
    expect(typeof nodes[1]).toBe("object");
    expect(typeof nodes[2]).toBe("string");
  });
});
