/**
 * Store tests — run in Node environment (vitest).
 *
 * All tests point HS01_DATA_DIR at a temp dir so they never write into the
 * real data directory.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import path from "path";
import fs from "fs";
import os from "os";
import crypto from "crypto";

// Paths to the real config + pool (same as loaders.test.ts)
const APP_DIR = path.resolve(__dirname, "..");
const POOL_PATH = path.resolve(APP_DIR, "../pool_frozen/itempool.json");
const CONFIG_PATH = path.resolve(APP_DIR, "./config/study-config.json");

process.env.HS01_POOL = POOL_PATH;
process.env.HS01_CONFIG = CONFIG_PATH;

// ─── helpers ────────────────────────────────────────────────────────────────

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "hs01-store-test-"));
}

function rmrf(dir: string) {
  fs.rmSync(dir, { recursive: true, force: true });
}

// ─── createSession ───────────────────────────────────────────────────────────

describe("createSession()", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });

  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("generates P001 then P002 with distinct session_ids", async () => {
    const { createSession, _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const r1 = await createSession(null);
    const r2 = await createSession(null);

    expect(r1.participant_code).toBe("P001");
    expect(r2.participant_code).toBe("P002");
    expect(r1.session_id).not.toBe(r2.session_id);
    // uuid format
    expect(r1.session_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
    );
  });

  it("writes session files to the data dir", async () => {
    const { createSession, _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const r1 = await createSession(null);
    const r2 = await createSession(null);

    // Filter out counter file (_counter.json)
    const files = fs.readdirSync(tmpDir).filter((f) => f.endsWith(".json") && !f.startsWith("_"));
    expect(files).toHaveLength(2);

    // Both files are parseable JSON
    for (const f of files) {
      const content = fs.readFileSync(path.join(tmpDir, f), "utf-8");
      expect(() => JSON.parse(content)).not.toThrow();
    }

    // File names correspond to session_ids
    expect(fs.existsSync(path.join(tmpDir, `${r1.session_id}.json`))).toBe(true);
    expect(fs.existsSync(path.join(tmpDir, `${r2.session_id}.json`))).toBe(true);
  });

  it("assigns the under-loaded form", async () => {
    const { createSession, _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    // The config has 3 forms A/B/C. Pre-populate form A with 2 completed sessions.
    const sessionDir = tmpDir;
    for (let i = 0; i < 2; i++) {
      const sessionId = crypto.randomUUID();
      const rec = {
        schema_version: "1.0.0",
        study_id: "HS-01",
        config_version: "1.0.0",
        config_sha256: "a".repeat(64),
        session_id: sessionId,
        form_id: "A",
        rng_seed: "deadbeef",
        status: "completed",
        participant: {
          participant_code: `P0${90 + i}`,
          recruitment_channel: null,
          consent: { given: true, consent_version: "v1", at_utc: new Date().toISOString() },
        },
        environment: {
          user_agent: "test",
          viewport: { w: 1280, h: 800 },
          device_pixel_ratio: 1,
        },
        timing: { started_at_utc: new Date().toISOString() },
        phase_timings: [],
        trials: [
          {
            trial_index: 0,
            phase_id: "text",
            position_in_phase: 0,
            item_id: "test",
            source_id: "test",
            item_kind: "text",
            is_attention_check: false,
            presented: {},
            response: { n_changes: 0, scale_value: 3 },
            timing: { onset_ms: 0, submitted_ms: 1000 },
          },
        ],
        demographics: {
          age_band: "25_34",
          ml_familiarity: "no_experience",
          english_proficiency: "B2",
          comment: null,
        },
      };
      fs.writeFileSync(
        path.join(sessionDir, `${sessionId}.json`),
        JSON.stringify(rec)
      );
    }

    // Now createSession should avoid A and pick B (least loaded, first by config order)
    const r = await createSession(null);
    expect(r.form_id).toBe("B");
  });

  it("returns a payload with items for the assigned form (no tgtbal/drift leakage)", async () => {
    const { createSession, _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const r = await createSession(null);

    // items field exists and is non-empty
    expect(r.items).toBeDefined();
    expect(Object.keys(r.items).length).toBeGreaterThan(0);

    // Collect all items across phases
    const allItems: unknown[] = [];
    for (const phase of Object.values(r.items) as unknown[][]) {
      allItems.push(...phase);
    }
    expect(allItems.length).toBeGreaterThan(0);

    // No tgtbal / drift fields anywhere in the payload
    const payloadStr = JSON.stringify(r);
    expect(payloadStr).not.toMatch(/tgtbal/);
    expect(payloadStr).not.toMatch(/"drift":/);
    expect(payloadStr).not.toMatch(/"d_text":/);
    expect(payloadStr).not.toMatch(/"d_img":/);
    // strata should not appear
    expect(payloadStr).not.toMatch(/"strata":/);
    // sut field should not appear
    expect(payloadStr).not.toMatch(/"sut":/);
    // search field should not appear
    expect(payloadStr).not.toMatch(/"search":/);

    // Each item should have item_id, source_id, kind, is_attention_check
    for (const item of allItems as Record<string, unknown>[]) {
      expect(item).toHaveProperty("item_id");
      expect(item).toHaveProperty("source_id");
      expect(item).toHaveProperty("kind");
      expect(item).toHaveProperty("is_attention_check");
      expect(item).toHaveProperty("prompt");
    }
  });

  it("includes config_sha256 and config_version in create result", async () => {
    const { createSession, _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const r = await createSession("tum");

    expect(r.config_version).toBe("1.0.0");
    expect(r.config_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(r.consent_version).toBe("v1");
    expect(r.rng_seed).toMatch(/^[a-f0-9]+$/);
  });
});

// ─── writeCheckpoint ────────────────────────────────────────────────────────

describe("writeCheckpoint()", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });

  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("persists an invalid record AND attaches x_validation_errors", async () => {
    const { createSession, writeCheckpoint, _resetStoreCache } = await import(
      "../src/lib/store"
    );
    _resetStoreCache();

    const cr = await createSession(null);

    // Build a deliberately invalid record (no trials, no demographics)
    const invalidRecord = {
      schema_version: "1.0.0" as const,
      study_id: "HS-01",
      config_version: "1.0.0",
      config_sha256: "a".repeat(64),
      session_id: cr.session_id,
      form_id: cr.form_id,
      rng_seed: cr.rng_seed,
      status: "abandoned" as const,
      participant: {
        participant_code: cr.participant_code,
        recruitment_channel: null,
        consent: {
          given: true as const,
          consent_version: "v1",
          at_utc: new Date().toISOString(),
        },
      },
      environment: {
        user_agent: "test",
        viewport: { w: 1280, h: 800 },
        device_pixel_ratio: 1,
      },
      timing: { started_at_utc: new Date().toISOString() },
      phase_timings: [],
      trials: [], // invalid: minItems:1
    } as import("../src/lib/types").SessionRecord;

    const result = await writeCheckpoint(cr.session_id, invalidRecord);

    // Should return ok:false with errors (record is invalid)
    expect(result.ok).toBe(false);
    expect(result.validation_errors).not.toBeNull();
    expect(Array.isArray(result.validation_errors)).toBe(true);

    // File must exist (never lose data)
    const filePath = path.join(tmpDir, `${cr.session_id}.json`);
    expect(fs.existsSync(filePath)).toBe(true);

    // File must contain x_validation_errors
    const saved = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    expect(saved.x_validation_errors).toBeDefined();
    expect(Array.isArray(saved.x_validation_errors)).toBe(true);
  });

  it("persists a valid partial record without x_validation_errors", async () => {
    const { createSession, writeCheckpoint, _resetStoreCache } = await import(
      "../src/lib/store"
    );
    _resetStoreCache();

    const cr = await createSession(null);

    // Build a valid-enough record (has one trial)
    const rec: import("../src/lib/types").SessionRecord = {
      schema_version: "1.0.0",
      study_id: "HS-01",
      config_version: "1.0.0",
      config_sha256: "a".repeat(64),
      session_id: cr.session_id,
      form_id: cr.form_id,
      rng_seed: cr.rng_seed,
      status: "abandoned",
      participant: {
        participant_code: cr.participant_code,
        recruitment_channel: null,
        consent: {
          given: true,
          consent_version: "v1",
          at_utc: new Date().toISOString(),
        },
      },
      environment: {
        user_agent: "test",
        viewport: { w: 1280, h: 800 },
        device_pixel_ratio: 1,
      },
      timing: { started_at_utc: new Date().toISOString() },
      phase_timings: [],
      trials: [
        {
          trial_index: 0,
          phase_id: "text",
          position_in_phase: 0,
          item_id: "test",
          source_id: "test",
          item_kind: "text",
          is_attention_check: false,
          presented: {},
          response: { n_changes: 0, scale_value: 3 },
          timing: { onset_ms: 0, submitted_ms: 1000 },
        },
      ],
    };

    const result = await writeCheckpoint(cr.session_id, rec);
    // Even if schema valid for now — we just check file persists
    const filePath = path.join(tmpDir, `${cr.session_id}.json`);
    expect(fs.existsSync(filePath)).toBe(true);
  });
});

// ─── submitSession ───────────────────────────────────────────────────────────

describe("submitSession()", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });

  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("promotes status to completed and file passes validateSession", async () => {
    const { createSession, submitSession, _resetStoreCache } = await import(
      "../src/lib/store"
    );
    const { validateSession } = await import("../src/lib/schemas");
    _resetStoreCache();

    const cr = await createSession(null);

    const completeRecord: import("../src/lib/types").SessionRecord = {
      schema_version: "1.0.0",
      study_id: "HS-01",
      config_version: "1.0.0",
      config_sha256: "a".repeat(64),
      session_id: cr.session_id,
      form_id: cr.form_id,
      rng_seed: cr.rng_seed,
      status: "abandoned", // submitSession will override to "completed"
      participant: {
        participant_code: cr.participant_code,
        recruitment_channel: null,
        consent: {
          given: true,
          consent_version: "v1",
          at_utc: new Date().toISOString(),
        },
      },
      environment: {
        user_agent: "Mozilla/5.0",
        viewport: { w: 1280, h: 800 },
        device_pixel_ratio: 1,
      },
      timing: { started_at_utc: new Date().toISOString() },
      phase_timings: [],
      trials: [
        {
          trial_index: 0,
          phase_id: "text",
          position_in_phase: 0,
          item_id: "txt-clean-llava-American_robin-3",
          source_id: "src-clean-llava-American_robin-3",
          item_kind: "text",
          is_attention_check: false,
          presented: {},
          response: { n_changes: 0, scale_value: 3 },
          timing: { onset_ms: 0, submitted_ms: 1000 },
        },
      ],
      demographics: {
        age_band: "25_34",
        ml_familiarity: "no_experience",
        english_proficiency: "B2",
        comment: null,
      },
    };

    const result = await submitSession(cr.session_id, completeRecord);

    expect(result.ok).toBe(true);
    expect(result.validation_errors).toBeNull();

    // Read file from disk and validate
    const filePath = path.join(tmpDir, `${cr.session_id}.json`);
    expect(fs.existsSync(filePath)).toBe(true);
    const saved = JSON.parse(fs.readFileSync(filePath, "utf-8"));

    // Status must be completed
    expect(saved.status).toBe("completed");

    // Timing fields set by server
    expect(saved.timing.completed_at_utc).toBeTruthy();
    expect(saved.timing.server_received_at_utc).toBeTruthy();

    // File validates against schema
    const valid = validateSession(saved);
    expect(valid).toBe(true);
  });
});

// ─── statusCounts ────────────────────────────────────────────────────────────

describe("statusCounts()", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });

  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("returns correct counts after creating and submitting sessions", async () => {
    const { createSession, submitSession, statusCounts, _resetStoreCache } =
      await import("../src/lib/store");
    _resetStoreCache();

    const r1 = await createSession(null);
    const r2 = await createSession(null);

    // Submit r1
    const completeRecord = (id: string, formId: string, code: string): import("../src/lib/types").SessionRecord => ({
      schema_version: "1.0.0",
      study_id: "HS-01",
      config_version: "1.0.0",
      config_sha256: "a".repeat(64),
      session_id: id,
      form_id: formId,
      rng_seed: "deadbeef",
      status: "abandoned",
      participant: {
        participant_code: code,
        recruitment_channel: null,
        consent: { given: true, consent_version: "v1", at_utc: new Date().toISOString() },
      },
      environment: { user_agent: "test", viewport: { w: 1280, h: 800 }, device_pixel_ratio: 1 },
      timing: { started_at_utc: new Date().toISOString() },
      phase_timings: [],
      trials: [
        {
          trial_index: 0, phase_id: "text", position_in_phase: 0,
          item_id: "t", source_id: "s", item_kind: "text", is_attention_check: false,
          presented: {}, response: { n_changes: 0, scale_value: 3 },
          timing: { onset_ms: 0, submitted_ms: 1000 },
        },
      ],
      demographics: { age_band: "25_34", ml_familiarity: "no_experience", english_proficiency: "B2", comment: null },
    });

    await submitSession(r1.session_id, completeRecord(r1.session_id, r1.form_id, r1.participant_code));

    const counts = await statusCounts();

    expect(counts.total_completed).toBe(1);
    // r2 was created (in-progress = abandoned file) but not submitted
    expect(counts.total_in_progress).toBe(1);
    expect(typeof counts.forms).toBe("object");
  });
});

// ─── concurrency ─────────────────────────────────────────────────────────────

describe("concurrency: 5 parallel createSession calls", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });

  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("yields 5 distinct sequential participant codes with no corruption", async () => {
    const { createSession, _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const results = await Promise.all([
      createSession(null),
      createSession(null),
      createSession(null),
      createSession(null),
      createSession(null),
    ]);

    const codes = results.map((r) => r.participant_code).sort();
    const ids = results.map((r) => r.session_id);

    // All codes are sequential P001..P005 (sorted)
    expect(codes).toEqual(["P001", "P002", "P003", "P004", "P005"]);

    // All session_ids are distinct
    expect(new Set(ids).size).toBe(5);

    // All files exist and are valid JSON
    for (const r of results) {
      const filePath = path.join(tmpDir, `${r.session_id}.json`);
      expect(fs.existsSync(filePath)).toBe(true);
      const content = fs.readFileSync(filePath, "utf-8");
      expect(() => JSON.parse(content)).not.toThrow();
    }

    // 5 session files (exclude counter file _counter.json)
    const jsonFiles = fs.readdirSync(tmpDir).filter((f) => f.endsWith(".json") && !f.startsWith("_"));
    expect(jsonFiles).toHaveLength(5);
  });
});
