/**
 * References loader tests — run in Node environment (vitest).
 *
 * loadReferences() reads, validates (AJV), and memoizes config/references.json
 * — the curated word → {gloss, image} map shown in the pair phase. refsDir()
 * resolves the directory the bundled reference PNGs are served from.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import path from "path";
import fs from "fs";
import os from "os";

const APP_DIR = path.resolve(__dirname, "..");

function writeTempJson(obj: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "hs01-refs-test-"));
  const p = path.join(dir, "references.json");
  fs.writeFileSync(p, JSON.stringify(obj));
  return p;
}

const VALID = {
  schema_version: "1.0.0",
  entries: {
    tench: { gloss: "a large European freshwater fish", image: "ref-tench.png" },
    axolotl: {
      gloss: "a salamander that keeps its larval gills as an adult",
      image: "ref-axolotl.png",
    },
  },
};

describe("loadReferences()", () => {
  afterEach(async () => {
    const { _resetReferencesCache } = await import("../src/lib/references");
    _resetReferencesCache();
    delete process.env.HS01_REFERENCES;
  });

  it("reads and returns the curated entries from a valid file", async () => {
    const p = writeTempJson(VALID);
    const { loadReferences } = await import("../src/lib/references");
    const data = loadReferences(p);
    expect(data.entries.tench.gloss).toMatch(/freshwater fish/);
    expect(data.entries.tench.image).toBe("ref-tench.png");
    expect(Object.keys(data.entries)).toHaveLength(2);
  });

  it("is memoized on the default (env) path — same object reference", async () => {
    process.env.HS01_REFERENCES = writeTempJson(VALID);
    const { loadReferences } = await import("../src/lib/references");
    const a = loadReferences();
    const b = loadReferences();
    expect(a).toBe(b);
  });

  it("accepts a gloss-only entry (image: null)", async () => {
    const p = writeTempJson({
      schema_version: "1.0.0",
      entries: { cock: { gloss: "an adult male chicken; a rooster", image: null } },
    });
    const { loadReferences } = await import("../src/lib/references");
    const data = loadReferences(p);
    expect(data.entries.cock.gloss).toMatch(/rooster/);
    expect(data.entries.cock.image).toBeNull();
  });

  it("throws when an entry is missing its gloss", async () => {
    const p = writeTempJson({
      schema_version: "1.0.0",
      entries: { tench: { image: "ref-tench.png" } },
    });
    const { loadReferences } = await import("../src/lib/references");
    expect(() => loadReferences(p)).toThrow(/schema validation/i);
  });

  it("throws when an image name is not a ref-*.png", async () => {
    const p = writeTempJson({
      schema_version: "1.0.0",
      entries: { tench: { gloss: "a fish", image: "../../etc/passwd" } },
    });
    const { loadReferences } = await import("../src/lib/references");
    expect(() => loadReferences(p)).toThrow(/schema validation/i);
  });
});

describe("refsDir()", () => {
  afterEach(() => {
    delete process.env.HS01_REFS_DIR;
  });

  it("honors HS01_REFS_DIR when set", async () => {
    process.env.HS01_REFS_DIR = "/tmp/some/refs";
    const { refsDir } = await import("../src/lib/references");
    expect(refsDir()).toBe("/tmp/some/refs");
  });

  it("defaults to <app>/config/refs", async () => {
    delete process.env.HS01_REFS_DIR;
    const { refsDir } = await import("../src/lib/references");
    expect(refsDir()).toBe(path.resolve(APP_DIR, "config/refs"));
  });
});
