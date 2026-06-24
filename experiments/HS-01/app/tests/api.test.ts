/**
 * API route tests — run in Node environment (vitest).
 *
 * Imports the App Router route handlers directly and invokes them with
 * constructed Web `Request` objects (no running server). Persistence points at
 * a temp HS01_DATA_DIR so tests never touch the real data directory.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import path from "path";
import fs from "fs";
import os from "os";

// Real config + pool + image dir (same resolution as store.test.ts)
const APP_DIR = path.resolve(__dirname, "..");
const POOL_PATH = path.resolve(APP_DIR, "../pool_frozen/itempool.json");
const CONFIG_PATH = path.resolve(APP_DIR, "./config/study-config.json");
const IMAGE_DIR = path.resolve(APP_DIR, "../pool_frozen/assets/images");

process.env.HS01_POOL = POOL_PATH;
process.env.HS01_CONFIG = CONFIG_PATH;
process.env.HS01_IMAGE_DIR = IMAGE_DIR;

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "hs01-api-test-"));
}
function rmrf(dir: string) {
  fs.rmSync(dir, { recursive: true, force: true });
}

// Build a complete, schema-valid record for submit given a create result.
function completeRecord(cr: {
  session_id: string;
  form_id: string;
  rng_seed: string;
  participant_code: string;
}): import("../src/lib/types").SessionRecord {
  return {
    schema_version: "1.0.0",
    study_id: "HS-01",
    config_version: "1.0.0",
    config_sha256: "a".repeat(64),
    session_id: cr.session_id,
    form_id: cr.form_id,
    rng_seed: cr.rng_seed,
    status: "abandoned", // submit overrides to completed
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
}

// ─── POST /api/sessions ──────────────────────────────────────────────────────

describe("POST /api/sessions", () => {
  let tmpDir: string;
  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });
  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("returns 201 with a payload whose items match the form and leak no analysis fields", async () => {
    const { POST } = await import("../src/app/api/sessions/route");
    const { _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const req = new Request("http://localhost/api/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ recruitment_channel: "tum" }),
    });
    const res = await POST(req);
    expect(res.status).toBe(201);

    const body = await res.json();
    expect(body.session_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
    );
    expect(body.participant_code).toBe("P001");
    expect(["A", "B", "C"]).toContain(body.form_id);
    expect(body.config_version).toBe("1.0.0");
    expect(body.config_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(body.consent_version).toBe("v1");

    // items non-empty across phases
    expect(body.items).toBeDefined();
    const allItems: Record<string, unknown>[] = [];
    for (const phase of Object.values(body.items) as Record<string, unknown>[][]) {
      allItems.push(...phase);
    }
    expect(allItems.length).toBeGreaterThan(0);

    // No-leak guarantee re-asserted at the HTTP boundary
    const str = JSON.stringify(body);
    expect(str).not.toMatch(/tgtbal/);
    expect(str).not.toMatch(/"drift":/);
    expect(str).not.toMatch(/"d_text":/);
    expect(str).not.toMatch(/"d_img":/);
    expect(str).not.toMatch(/"strata":/);
    expect(str).not.toMatch(/"sut":/);
    expect(str).not.toMatch(/"search":/);

    for (const item of allItems) {
      expect(item).toHaveProperty("item_id");
      expect(item).toHaveProperty("source_id");
      expect(item).toHaveProperty("kind");
      expect(item).toHaveProperty("is_attention_check");
    }
  });

  it("rejects an invalid recruitment_channel with 400", async () => {
    const { POST } = await import("../src/app/api/sessions/route");
    const { _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const req = new Request("http://localhost/api/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ recruitment_channel: "linkedin" }),
    });
    const res = await POST(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBeTruthy();
  });

  it("accepts a missing/null channel (201)", async () => {
    const { POST } = await import("../src/app/api/sessions/route");
    const { _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const req = new Request("http://localhost/api/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    const res = await POST(req);
    expect(res.status).toBe(201);
  });
});

// ─── PUT /api/sessions/[id]/checkpoint ───────────────────────────────────────

describe("PUT /api/sessions/[id]/checkpoint", () => {
  let tmpDir: string;
  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });
  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("persists a partial record and returns {ok, validation_errors}", async () => {
    const { POST } = await import("../src/app/api/sessions/route");
    const { PUT } = await import("../src/app/api/sessions/[id]/checkpoint/route");
    const { _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const createRes = await POST(
      new Request("http://localhost/api/sessions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      })
    );
    const cr = await createRes.json();

    // Partial record (valid-enough: one trial)
    const partial = completeRecord(cr);
    delete (partial as Record<string, unknown>).demographics;

    const req = new Request(
      `http://localhost/api/sessions/${cr.session_id}/checkpoint`,
      {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(partial),
      }
    );
    const res = await PUT(req, { params: { id: cr.session_id } });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("ok");
    expect(body).toHaveProperty("validation_errors");

    // File persisted
    const filePath = path.join(tmpDir, `${cr.session_id}.json`);
    expect(fs.existsSync(filePath)).toBe(true);
  });
});

// ─── POST /api/sessions/[id]/submit ──────────────────────────────────────────

describe("POST /api/sessions/[id]/submit", () => {
  let tmpDir: string;
  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });
  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("submits a complete record → {ok:true, validation_errors:null}, on-disk file validates", async () => {
    const { POST: CREATE } = await import("../src/app/api/sessions/route");
    const { POST: SUBMIT } = await import(
      "../src/app/api/sessions/[id]/submit/route"
    );
    const { _resetStoreCache } = await import("../src/lib/store");
    const { validateSession } = await import("../src/lib/schemas");
    _resetStoreCache();

    const createRes = await CREATE(
      new Request("http://localhost/api/sessions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      })
    );
    const cr = await createRes.json();

    const rec = completeRecord(cr);
    const res = await SUBMIT(
      new Request(`http://localhost/api/sessions/${cr.session_id}/submit`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(rec),
      }),
      { params: { id: cr.session_id } }
    );
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.validation_errors).toBeNull();

    const filePath = path.join(tmpDir, `${cr.session_id}.json`);
    const saved = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    expect(saved.status).toBe("completed");
    expect(saved.timing.completed_at_utc).toBeTruthy();
    expect(validateSession(saved)).toBe(true);
  });
});

// ─── GET /api/status ─────────────────────────────────────────────────────────

describe("GET /api/status", () => {
  let tmpDir: string;
  beforeEach(() => {
    tmpDir = makeTempDir();
    process.env.HS01_DATA_DIR = tmpDir;
  });
  afterEach(() => {
    rmrf(tmpDir);
    delete process.env.HS01_DATA_DIR;
  });

  it("returns counts {forms, total_completed, total_in_progress}", async () => {
    const { POST: CREATE } = await import("../src/app/api/sessions/route");
    const { POST: SUBMIT } = await import(
      "../src/app/api/sessions/[id]/submit/route"
    );
    const { GET } = await import("../src/app/api/status/route");
    const { _resetStoreCache } = await import("../src/lib/store");
    _resetStoreCache();

    const c1 = await (
      await CREATE(
        new Request("http://localhost/api/sessions", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({}),
        })
      )
    ).json();
    await (
      await CREATE(
        new Request("http://localhost/api/sessions", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({}),
        })
      )
    ).json();

    await SUBMIT(
      new Request(`http://localhost/api/sessions/${c1.session_id}/submit`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(completeRecord(c1)),
      }),
      { params: { id: c1.session_id } }
    );

    const res = await GET();
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("forms");
    expect(body.total_completed).toBe(1);
    expect(body.total_in_progress).toBe(1);
  });
});

// ─── GET /api/images/[name] ──────────────────────────────────────────────────

describe("GET /api/images/[name]", () => {
  it("serves a real src-*.png with image/png + immutable cache header", async () => {
    const { GET } = await import("../src/app/api/images/[name]/route");
    const name = "src-clean-llava-American_robin-3.png";
    const res = await GET(
      new Request(`http://localhost/api/images/${name}`),
      { params: { name } }
    );
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("image/png");
    expect(res.headers.get("cache-control")).toBe(
      "public, max-age=31536000, immutable"
    );
    const buf = Buffer.from(await res.arrayBuffer());
    expect(buf.length).toBeGreaterThan(0);
    // PNG magic bytes
    expect(buf[0]).toBe(0x89);
    expect(buf[1]).toBe(0x50);
  });

  it("rejects path traversal (../../etc/passwd) with 400", async () => {
    const { GET } = await import("../src/app/api/images/[name]/route");
    const name = "../../etc/passwd";
    const res = await GET(new Request(`http://localhost/api/images/x`), {
      params: { name },
    });
    expect(res.status).toBe(400);
  });

  it("rejects a non-src name (foo.png) with 400", async () => {
    const { GET } = await import("../src/app/api/images/[name]/route");
    const name = "foo.png";
    const res = await GET(new Request(`http://localhost/api/images/${name}`), {
      params: { name },
    });
    expect(res.status).toBe(400);
  });

  it("returns 404 for a well-formed but missing src-*.png", async () => {
    const { GET } = await import("../src/app/api/images/[name]/route");
    const name = "src-does-not-exist-zzz.png";
    const res = await GET(new Request(`http://localhost/api/images/${name}`), {
      params: { name },
    });
    expect(res.status).toBe(404);
  });
});
