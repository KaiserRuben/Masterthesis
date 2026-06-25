/**
 * prompt-font tests — guard the bundled stimulus-prompt webfont.
 *
 * WHY THIS EXISTS
 * ---------------
 * Stimulus prompts are rendered verbatim and deliberately contain Cyrillic
 * homoglyph codepoints (the adversarial perturbation), e.g. the letter "i" in
 * "What is the main text in this area" may actually be U+0456 (Cyrillic і).
 *
 * If the prompt font does NOT itself cover those Cyrillic codepoints, the
 * browser falls back per-glyph to a DIFFERENT physical font for just those
 * letters. That fallback font is typically proportional / heavier / taller, so
 * the homoglyphs render visibly bold and larger — a glaring tell that defeats
 * the subtlety the attack relies on, and a confound for the human-validity
 * measurement.
 *
 * The fix bundles a single self-hosted monospace webfont (JetBrains Mono) whose
 * Latin AND Cyrillic glyphs share matching metrics and that covers all 7
 * homoglyph codepoints present in the pool. This test fails loudly if a future
 * change ships a prompt font that silently lacks that Cyrillic coverage (or if
 * the vendored asset goes missing), which would re-open the bug.
 *
 * It reads the ACTUAL vendored font binaries under public/fonts/ — not a
 * config string — so it cannot be fooled by a stale CSS variable.
 */

import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "fs";
import path from "path";
import * as fontkit from "fontkit";

const FONTS_DIR = path.resolve(__dirname, "../public/fonts");
const CYRILLIC_FONT = path.join(
  FONTS_DIR,
  "jetbrains-mono-cyrillic-400-normal.woff2"
);
const LATIN_FONT = path.join(
  FONTS_DIR,
  "jetbrains-mono-latin-400-normal.woff2"
);

/**
 * The 7 Cyrillic homoglyph codepoints present in the HS-01 item pool, paired
 * with the ASCII Latin twin each one impersonates. The prompt font MUST cover
 * the Cyrillic side; the ASCII side is used for the metric-match assertion.
 */
const HOMOGLYPHS: ReadonlyArray<{ cyr: number; ascii: number; label: string }> =
  [
    { cyr: 0x0430, ascii: 0x0061, label: "а→a" },
    { cyr: 0x0435, ascii: 0x0065, label: "е→e" },
    { cyr: 0x0441, ascii: 0x0063, label: "с→c" },
    { cyr: 0x043e, ascii: 0x006f, label: "о→o" },
    { cyr: 0x0440, ascii: 0x0070, label: "р→p" },
    { cyr: 0x0455, ascii: 0x0073, label: "ѕ→s" },
    { cyr: 0x0456, ascii: 0x0069, label: "і→i" },
  ];

function loadFont(file: string) {
  // fontkit.create parses the in-memory buffer (handles woff2 brotli inside).
  return fontkit.create(readFileSync(file)) as ReturnType<
    typeof fontkit.create
  > & {
    hasGlyphForCodePoint(cp: number): boolean;
    glyphForCodePoint(cp: number): { advanceWidth: number };
    unitsPerEm: number;
  };
}

describe("bundled prompt font (JetBrains Mono)", () => {
  it("the vendored, self-hosted woff2 assets exist", () => {
    expect(existsSync(CYRILLIC_FONT)).toBe(true);
    expect(existsSync(LATIN_FONT)).toBe(true);
  });

  it("covers all 7 Cyrillic homoglyph codepoints in the pool", () => {
    const font = loadFont(CYRILLIC_FONT);
    const missing = HOMOGLYPHS.filter(
      (h) => !font.hasGlyphForCodePoint(h.cyr)
    ).map((h) => `U+${h.cyr.toString(16).toUpperCase().padStart(4, "0")} (${h.label})`);
    expect(missing, `homoglyph codepoints missing from prompt font: ${missing.join(", ")}`).toEqual([]);
  });

  it("renders each homoglyph with the SAME advance width as its ASCII twin (no width tell)", () => {
    const cyr = loadFont(CYRILLIC_FONT);
    const lat = loadFont(LATIN_FONT);

    // Subsets of the same monospace face: same design grid is required for the
    // per-glyph advances to be comparable across the two unicode-range subsets.
    expect(cyr.unitsPerEm).toBe(lat.unitsPerEm);

    for (const h of HOMOGLYPHS) {
      const cyrAdv = cyr.glyphForCodePoint(h.cyr).advanceWidth;
      const asciiAdv = lat.glyphForCodePoint(h.ascii).advanceWidth;
      expect(
        cyrAdv,
        `${h.label}: Cyrillic advance ${cyrAdv} != ASCII advance ${asciiAdv}`
      ).toBe(asciiAdv);
    }
  });
});
