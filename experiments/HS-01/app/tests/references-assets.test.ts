/**
 * Reference-asset invariants — run in Node environment (vitest).
 *
 * These guard the committed curated reference set (config/references.json +
 * config/refs/*.png) against two failure modes:
 *
 *  1. HARD INVARIANT: no reference image may be the same PHOTO as a study
 *     stimulus. The naive byte-SHA check is INSUFFICIENT: the pipeline
 *     re-encodes a clean seed photo (origin.png) into the committed stimulus —
 *     pixel-identical but byte-different — so a SHA guard waves it through. We
 *     therefore compare DECODED PIXELS: each ref must be perceptually disjoint
 *     (pixel-MAE and dHash-Hamming both clear of a small threshold) from every
 *     committed stimulus AND every clean seed origin the pool derives from.
 *  2. COVERAGE / INTEGRITY: every pair-option word that the UI can show (all of
 *     them, resolved from the pool exactly as PairChoice does) has a well-formed
 *     entry (non-empty gloss; an existing ref image), no key is a non-pair word,
 *     and no ref file is orphaned.
 */

import { describe, it, expect } from "vitest";
import path from "path";
import fs from "fs";
import { loadReferences } from "../src/lib/references";
import { pixelMae, dhash, hamming } from "./perceptual-png";

const APP_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(APP_DIR, "../../.."); // app -> HS-01 -> experiments -> repo
const POOL_PATH = path.resolve(APP_DIR, "../pool_frozen/itempool.json");
const CONFIG_PATH = path.resolve(APP_DIR, "config/study-config.json");
const REFS_DIR = path.resolve(APP_DIR, "config/refs");
const REFERENCES_PATH = path.resolve(APP_DIR, "config/references.json");
const STUDY_IMAGES = path.resolve(APP_DIR, "../pool_frozen/assets/images");
const RUNS_DIR = path.resolve(REPO_ROOT, "runs");

// Perceptual disjointness thresholds. The 8 known seed-photo leaks sit at exactly
// MAE 0 / Ham 0; the closest legitimate reference is at MAE ~22 / Ham ~90. These
// thresholds sit comfortably in that gap.
const MAE_THRESHOLD = 3.0; // mean abs grayscale pixel diff (0..255)
const HAM_THRESHOLD = 8; // /256-bit dHash

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
      const src = srcById.get(itemToSrc.get(pid)!);
      words.add(src.cell.anchor_word);
      words.add(src.cell.target_word);
    }
  }
  return words;
}

/** Recursively collect every origin.png beneath a directory. */
function collectOrigins(dir: string, out: string[]): void {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) collectOrigins(p, out);
    else if (e.name === "origin.png") out.push(p);
  }
}

/**
 * The full stimulus-exclusion set: every committed stimulus PNG PLUS every clean
 * seed origin.png the pool sources derive from (resolved via experiment_ref.run_id
 * -> runs/**, mirroring freeze_pool.py). The seed origins are the load-bearing
 * additions — they are what the pipeline re-encodes into stimuli.
 */
function stimulusExclusionSet(): string[] {
  const excl: string[] = [];
  for (const f of fs.readdirSync(STUDY_IMAGES)) {
    if (f.endsWith(".png")) excl.push(path.join(STUDY_IMAGES, f));
  }
  const pool = JSON.parse(fs.readFileSync(POOL_PATH, "utf-8"));
  const runIds = new Set<string>();
  for (const s of pool.sources) {
    const rid = s.experiment_ref?.run_id;
    if (rid) runIds.add(rid);
  }
  // Index run-dir name -> origin paths, scanning runs/ once.
  const allOrigins: string[] = [];
  collectOrigins(RUNS_DIR, allOrigins);
  const seen = new Set<string>();
  for (const o of allOrigins) {
    const parts = o.split(path.sep);
    if (parts.some((p) => runIds.has(p)) && !seen.has(o)) {
      seen.add(o);
      excl.push(o);
    }
  }
  return excl;
}

describe("reference assets", () => {
  it("references.json loads and validates", () => {
    const data = loadReferences(REFERENCES_PATH);
    expect(Object.keys(data.entries).length).toBeGreaterThan(0);
  });

  it("covers EVERY pair-option word the UI can show", () => {
    const data = loadReferences(REFERENCES_PATH);
    const words = [...pairOptionWords()].sort();
    const keys = Object.keys(data.entries).sort();
    expect(keys).toEqual(words);
  });

  it("every entry key is a real pair-option word in the pool", () => {
    const data = loadReferences(REFERENCES_PATH);
    const valid = pairOptionWords();
    for (const word of Object.keys(data.entries)) {
      expect(valid.has(word), `"${word}" is not a pair-option word`).toBe(true);
    }
  });

  it("every entry has a non-empty gloss and an existing image", () => {
    const data = loadReferences(REFERENCES_PATH);
    for (const [word, entry] of Object.entries(data.entries)) {
      expect(entry.gloss.trim().length, `${word} gloss`).toBeGreaterThan(0);
      expect(entry.image, `${word} must have a reference image`).not.toBeNull();
      expect(
        fs.existsSync(path.join(REFS_DIR, entry.image as string)),
        `${word} image ${entry.image} missing`
      ).toBe(true);
    }
  });

  it("NO reference image is the same photo as a study stimulus (perceptual)", () => {
    const exclusion = stimulusExclusionSet();
    expect(exclusion.length).toBeGreaterThan(0);
    const exclusionDhash = exclusion.map((p) => ({ p, d: dhash(p) }));

    const refs = fs.readdirSync(REFS_DIR).filter((f) => f.endsWith(".png"));
    expect(refs.length).toBeGreaterThan(0);

    const collisions: string[] = [];
    for (const f of refs) {
      const refPath = path.join(REFS_DIR, f);
      const refDhash = dhash(refPath);
      let minMae = Infinity;
      let minHam = Infinity;
      let nearest = "";
      for (const { p, d } of exclusionDhash) {
        const mae = pixelMae(refPath, p);
        const ham = hamming(refDhash, d);
        if (mae < minMae) {
          minMae = mae;
          nearest = p;
        }
        if (ham < minHam) minHam = ham;
      }
      if (minMae <= MAE_THRESHOLD || minHam <= HAM_THRESHOLD) {
        collisions.push(
          `${f} collides (MAE=${minMae.toFixed(2)}, Ham=${minHam}) with ${path.basename(
            nearest
          )}`
        );
      }
    }
    expect(collisions, `reference images that are study photos:\n${collisions.join("\n")}`).toEqual(
      []
    );
  });

  it("every bundled ref file is referenced by an entry (no orphans)", () => {
    const data = loadReferences(REFERENCES_PATH);
    const used = new Set(
      Object.values(data.entries)
        .map((e) => e.image)
        .filter(Boolean)
    );
    for (const f of fs.readdirSync(REFS_DIR).filter((f) => f.endsWith(".png"))) {
      expect(used.has(f), `${f} is orphaned (no entry references it)`).toBe(true);
    }
  });
});
