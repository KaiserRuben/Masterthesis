/**
 * Reference-asset invariants — run in Node environment (vitest).
 *
 * These guard the committed curated reference set (config/references.json +
 * config/refs/*.png) against two failure modes:
 *
 *  1. HARD INVARIANT: no reference image may be a study stimulus. Every committed
 *     ref PNG's SHA-256 must be absent from the set of study-stimulus SHA-256s.
 *     Refs are byte-exact copies of cache images, so this byte-level check is
 *     meaningful (it would catch a study stimulus dropped into config/refs).
 *  2. INTEGRITY: every entry is well-formed (non-empty gloss; image is null or a
 *     file that exists), every key is a real pair-option word, and the 21 curated
 *     fine-grained words are all present.
 */

import { describe, it, expect } from "vitest";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import { loadReferences } from "../src/lib/references";

const APP_DIR = path.resolve(__dirname, "..");
const POOL_PATH = path.resolve(APP_DIR, "../pool_frozen/itempool.json");
const CONFIG_PATH = path.resolve(APP_DIR, "config/study-config.json");
const REFS_DIR = path.resolve(APP_DIR, "config/refs");
const REFERENCES_PATH = path.resolve(APP_DIR, "config/references.json");
const STUDY_IMAGES = path.resolve(APP_DIR, "../pool_frozen/assets/images");

function sha256(p: string): string {
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

function studyStimulusHashes(): Set<string> {
  const hashes = new Set<string>();
  for (const f of fs.readdirSync(STUDY_IMAGES)) {
    if (f.endsWith(".png")) hashes.add(sha256(path.join(STUDY_IMAGES, f)));
  }
  return hashes;
}

/** Words that appear as ANCHOR_WORD / TARGET_WORD of a pair item in any form. */
function pairOptionWords(): Set<string> {
  const pool = JSON.parse(fs.readFileSync(POOL_PATH, "utf-8"));
  const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));
  const srcById = new Map<string, any>(pool.sources.map((s: any) => [s.source_id, s]));
  const itemToSrc = new Map<string, string>(
    pool.items.map((it: any) => [it.item_id, it.source_id])
  );
  const words = new Set<string>();
  for (const form of cfg.forms) {
    for (const pid of form.pair_items as string[]) {
      if (pid.startsWith("pair-attn")) continue;
      const src = srcById.get(itemToSrc.get(pid)!);
      words.add(src.cell.anchor_word);
      words.add(src.cell.target_word);
    }
  }
  return words;
}

// The 21 fine-grained pair-option words the study glosses (curation frozen set).
const EXPECTED_WORDS = [
  "American bullfrog", "American robin", "axolotl", "bald eagle", "box turtle",
  "cello", "chameleon", "cock", "desert grassland whiptail lizard",
  "fire salamander", "flamingo", "great grey owl", "great white shark",
  "indigo bunting", "loggerhead sea turtle", "marimba", "mud turtle", "ostrich",
  "stingray", "tench", "tiger shark",
];

describe("reference assets", () => {
  it("references.json loads and validates", () => {
    const data = loadReferences(REFERENCES_PATH);
    expect(Object.keys(data.entries).length).toBeGreaterThan(0);
  });

  it("contains exactly the 21 curated fine-grained words", () => {
    const data = loadReferences(REFERENCES_PATH);
    expect(Object.keys(data.entries).sort()).toEqual([...EXPECTED_WORDS].sort());
  });

  it("every entry key is a real pair-option word in the pool", () => {
    const data = loadReferences(REFERENCES_PATH);
    const valid = pairOptionWords();
    for (const word of Object.keys(data.entries)) {
      expect(valid.has(word), `"${word}" is not a pair-option word`).toBe(true);
    }
  });

  it("every entry has a non-empty gloss and an existing image (or null)", () => {
    const data = loadReferences(REFERENCES_PATH);
    for (const [word, entry] of Object.entries(data.entries)) {
      expect(entry.gloss.trim().length, `${word} gloss`).toBeGreaterThan(0);
      if (entry.image !== null) {
        expect(
          fs.existsSync(path.join(REFS_DIR, entry.image)),
          `${word} image ${entry.image} missing`
        ).toBe(true);
      }
    }
  });

  it("NO reference image is a study stimulus (SHA-256 exclusion)", () => {
    const study = studyStimulusHashes();
    expect(study.size).toBeGreaterThan(0);
    const refs = fs.readdirSync(REFS_DIR).filter((f) => f.endsWith(".png"));
    expect(refs.length).toBeGreaterThan(0);
    for (const f of refs) {
      const h = sha256(path.join(REFS_DIR, f));
      expect(study.has(h), `${f} is a study stimulus!`).toBe(false);
    }
  });

  it("every bundled ref file is referenced by an entry (no orphans)", () => {
    const data = loadReferences(REFERENCES_PATH);
    const used = new Set(
      Object.values(data.entries).map((e) => e.image).filter(Boolean)
    );
    for (const f of fs.readdirSync(REFS_DIR).filter((f) => f.endsWith(".png"))) {
      expect(used.has(f), `${f} is orphaned (no entry references it)`).toBe(true);
    }
  });
});
