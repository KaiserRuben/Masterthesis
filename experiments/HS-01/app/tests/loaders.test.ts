/**
 * Loader tests — run in Node environment (vitest).
 *
 * Exercises:
 *  - loadPool()      → 93 sources & 93 items
 *  - loadConfig()    → 3 forms
 *  - assertPoolBinding  passes for the real pair, throws on tampered hash
 *  - ensureInputs()  throws on a tampered config hash
 */

import { describe, it, expect, beforeEach } from "vitest";
import path from "path";
import crypto from "crypto";
import fs from "fs";

// Resolve paths relative to this file so tests work from any CWD
const APP_DIR = path.resolve(__dirname, "..");
const POOL_PATH = path.resolve(APP_DIR, "../pool_frozen/itempool.json");
const CONFIG_PATH = path.resolve(APP_DIR, "./config/study-config.json");

// Set env vars before importing loader modules (they read process.env at load
// time via their defaults, but we override here to be explicit)
process.env.HS01_POOL = POOL_PATH;
process.env.HS01_CONFIG = CONFIG_PATH;

describe("loadPool()", () => {
  it("returns a pool with 93 sources and 93 items", async () => {
    // Dynamic import so env vars are set first
    const { loadPool } = await import("../src/lib/pool");
    const pool = loadPool();
    expect(pool.sources).toHaveLength(93);
    expect(pool.items).toHaveLength(93);
  });

  it("is memoized — same object reference on second call", async () => {
    const { loadPool } = await import("../src/lib/pool");
    const a = loadPool();
    const b = loadPool();
    expect(a).toBe(b);
  });
});

describe("loadConfig()", () => {
  it("returns a config with 3 forms", async () => {
    const { loadConfig } = await import("../src/lib/config");
    const config = loadConfig();
    expect(config.forms).toHaveLength(3);
  });

  it("is memoized — same object reference on second call", async () => {
    const { loadConfig } = await import("../src/lib/config");
    const a = loadConfig();
    const b = loadConfig();
    expect(a).toBe(b);
  });
});

describe("assertPoolBinding()", () => {
  it("passes silently for the real pool + config pair", async () => {
    const { loadPool } = await import("../src/lib/pool");
    const { loadConfig, assertPoolBinding } = await import("../src/lib/config");
    const pool = loadPool();
    const config = loadConfig();
    // Should not throw
    expect(() => assertPoolBinding(pool, config)).not.toThrow();
  });

  it("throws when the config pool_file_sha256 is mutated", async () => {
    const { loadPool } = await import("../src/lib/pool");
    const { loadConfig, assertPoolBinding } = await import("../src/lib/config");
    const pool = loadPool();
    const realConfig = loadConfig();

    // Deep-clone and tamper the hash
    const tamperedConfig = JSON.parse(JSON.stringify(realConfig));
    tamperedConfig.pool_ref.pool_file_sha256 =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    expect(() => assertPoolBinding(pool, tamperedConfig)).toThrow(
      /pool_file_sha256 mismatch/i
    );
  });
});

describe("ensureInputs()", () => {
  it("runs without throwing for the real inputs", async () => {
    const { ensureInputs } = await import("../src/lib/boot");
    expect(() => ensureInputs()).not.toThrow();
  });

  it("throws when config contains a tampered pool hash", async () => {
    const { ensureInputs } = await import("../src/lib/boot");

    // Write a tampered config to a temp file, point env at it, then reset
    const realConfigBytes = fs.readFileSync(CONFIG_PATH, "utf-8");
    const tamperedConfig = JSON.parse(realConfigBytes);
    tamperedConfig.pool_ref.pool_file_sha256 =
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    const tmpPath = path.resolve(APP_DIR, "tests/.tmp_tampered_config.json");
    fs.writeFileSync(tmpPath, JSON.stringify(tamperedConfig));

    const originalConfigEnv = process.env.HS01_CONFIG;
    process.env.HS01_CONFIG = tmpPath;

    try {
      // Import a fresh module reference by clearing the module cache
      // vitest doesn't support jest.resetModules, but we can call ensureInputs
      // with the tampered env. Since ensureInputs reads env at call time (not
      // at module import), this will exercise the tampered path.
      expect(() => ensureInputs({ configPath: tmpPath })).toThrow(
        /pool_file_sha256 mismatch/i
      );
    } finally {
      process.env.HS01_CONFIG = originalConfigEnv;
      fs.unlinkSync(tmpPath);
    }
  });
});
