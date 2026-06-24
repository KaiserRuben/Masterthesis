/**
 * store.ts — server-side session persistence layer.
 *
 * Provides:
 *  - createSession()    — form balancing, participant counter, atomic write
 *  - writeCheckpoint()  — idempotent atomic overwrite with best-effort validation
 *  - submitSession()    — promote to completed, stamp server timestamps
 *  - statusCounts()     — per-form + total counts
 *
 * All file writes use the write-tmp → fsync → rename atomic pattern.
 * Concurrent createSession calls are serialized via a tiny in-process mutex.
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";

import { loadConfig } from "./config";
import { loadPool } from "./pool";
import { validateSession } from "./schemas";
import type { SessionRecord, Source, Item } from "./types";

// ─── RecruitmentChannel type ─────────────────────────────────────────────────

export type RecruitmentChannel = "tum" | "fortiss" | "personal" | "other";

// ─── Return type for createSession ───────────────────────────────────────────

export interface ItemPayload {
  item_id: string;
  source_id: string;
  kind: "text" | "image" | "pair";
  is_attention_check: boolean;
  prompt: string | null;
  image: { uri_name: string; natural_w: number; natural_h: number } | null;
  option_labels: { ANCHOR_WORD: string; TARGET_WORD: string } | null;
}

export interface CreateResult {
  session_id: string;
  participant_code: string;
  form_id: string;
  rng_seed: string;
  config_version: string;
  config_sha256: string;
  consent_version: string;
  /** Per-phase item payloads. Keys are phase_ids. */
  items: Record<string, ItemPayload[]>;
}

// ─── Path helpers ─────────────────────────────────────────────────────────────

function dataDir(): string {
  return process.env.HS01_DATA_DIR ?? path.resolve(__dirname, "../../data/sessions");
}

function sessionPath(id: string): string {
  return path.join(dataDir(), `${id}.json`);
}

function counterPath(): string {
  return path.join(dataDir(), "_counter.json");
}

function ensureDataDir(): void {
  fs.mkdirSync(dataDir(), { recursive: true });
}

// ─── Config sha256 (memoized) ─────────────────────────────────────────────────

let _configSha256: string | null = null;
let _configPath: string | null = null;

function getConfigSha256(): string {
  const cfgPath = process.env.HS01_CONFIG ?? "";
  // Re-compute if config path changed (e.g. between tests)
  if (_configSha256 !== null && _configPath === cfgPath) return _configSha256;

  let bytes: Buffer;
  if (cfgPath) {
    bytes = fs.readFileSync(cfgPath);
  } else {
    // Fall back to default path used by loadConfig
    const APP_DIR = path.resolve(__dirname, "../..");
    bytes = fs.readFileSync(path.resolve(APP_DIR, "config/study-config.json"));
  }
  _configSha256 = crypto.createHash("sha256").update(bytes).digest("hex");
  _configPath = cfgPath;
  return _configSha256;
}

// ─── Atomic file write ────────────────────────────────────────────────────────

async function atomicWriteJson(filePath: string, data: unknown): Promise<void> {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });

  const tmp = `${filePath}.tmp`;
  const content = JSON.stringify(data, null, 2);

  await new Promise<void>((resolve, reject) => {
    fs.open(tmp, "w", (err, fd) => {
      if (err) return reject(err);
      const buf = Buffer.from(content, "utf-8");
      fs.write(fd, buf, 0, buf.length, 0, (writeErr) => {
        if (writeErr) {
          fs.close(fd, () => reject(writeErr));
          return;
        }
        fs.fsync(fd, (fsyncErr) => {
          if (fsyncErr) {
            fs.close(fd, () => reject(fsyncErr));
            return;
          }
          fs.close(fd, (closeErr) => {
            if (closeErr) return reject(closeErr);
            fs.rename(tmp, filePath, (renameErr) => {
              if (renameErr) reject(renameErr);
              else resolve();
            });
          });
        });
      });
    });
  });
}

// ─── In-process mutex ─────────────────────────────────────────────────────────

let _mutexTail: Promise<void> = Promise.resolve();

function withMutex<T>(fn: () => Promise<T>): Promise<T> {
  // Chain onto the tail of any queued operations
  const result = _mutexTail.then(() => fn());
  // Update tail (suppress rejection so the queue doesn't stall)
  _mutexTail = result.then(
    () => {},
    () => {}
  );
  return result;
}

// ─── Participant counter ───────────────────────────────────────────────────────

/** Read the counter, increment, write back, return the NEW value. Must run inside mutex. */
async function incrementCounter(): Promise<number> {
  const p = counterPath();
  let current = 0;
  if (fs.existsSync(p)) {
    try {
      const raw = fs.readFileSync(p, "utf-8");
      const parsed = JSON.parse(raw) as { count: number };
      current = parsed.count ?? 0;
    } catch {
      current = 0;
    }
  }
  const next = current + 1;
  await atomicWriteJson(p, { count: next });
  return next;
}

function formatParticipantCode(n: number): string {
  const digits = String(n);
  // Zero-pad to at least 3 digits
  return `P${digits.padStart(3, "0")}`;
}

// ─── Form balancing ───────────────────────────────────────────────────────────

/**
 * Count active sessions per form.
 * A session counts as "active" if status==="completed" OR its file mtime is within the last 24h.
 */
function countActiveSessions(): Record<string, number> {
  const dir = dataDir();
  if (!fs.existsSync(dir)) return {};

  const counts: Record<string, number> = {};
  const now = Date.now();
  const windowMs = 24 * 60 * 60 * 1000;

  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json") && !f.startsWith("_"));
  for (const file of files) {
    const fullPath = path.join(dir, file);
    try {
      const stat = fs.statSync(fullPath);
      const raw = fs.readFileSync(fullPath, "utf-8");
      const rec = JSON.parse(raw) as Partial<SessionRecord>;
      const formId = rec.form_id;
      if (!formId) continue;

      const isCompleted = rec.status === "completed";
      const isRecent = now - stat.mtimeMs < windowMs;

      if (isCompleted || isRecent) {
        counts[formId] = (counts[formId] ?? 0) + 1;
      }
    } catch {
      // Skip unparseable files
    }
  }
  return counts;
}

function chooseLeastLoadedForm(counts: Record<string, number>): string {
  const config = loadConfig();
  let best: string | null = null;
  let bestCount = Infinity;

  for (const form of config.forms) {
    const c = counts[form.form_id] ?? 0;
    if (c < bestCount) {
      bestCount = c;
      best = form.form_id;
    }
  }
  return best ?? config.forms[0].form_id;
}

// ─── Item payload resolution ───────────────────────────────────────────────────

function resolveItems(formId: string): Record<string, ItemPayload[]> {
  const config = loadConfig();
  const pool = loadPool();

  const form = config.forms.find((f) => f.form_id === formId);
  if (!form) throw new Error(`store: unknown form_id "${formId}"`);

  // Build source lookup
  const sourceMap = new Map<string, Source>(pool.sources.map((s) => [s.source_id, s]));
  // Build item lookup
  const itemMap = new Map<Item["item_id"], Item>(pool.items.map((i) => [i.item_id, i]));

  function resolvePhase(itemIds: string[]): ItemPayload[] {
    return itemIds.map((itemId) => {
      const item = itemMap.get(itemId);
      if (!item) throw new Error(`store: item_id "${itemId}" not found in pool`);

      const source = sourceMap.get(item.source_id);
      if (!source) throw new Error(`store: source_id "${item.source_id}" not found in pool`);

      // Prompt text (verbatim)
      const prompt = source.assets.prompt?.text ?? null;

      // Image (uri basename only, dimensions)
      let image: ItemPayload["image"] = null;
      if (source.assets.image) {
        const img = source.assets.image;
        image = {
          uri_name: path.basename(img.uri),
          natural_w: img.width,
          natural_h: img.height,
        };
      }

      // Option labels for pair items
      let option_labels: ItemPayload["option_labels"] = null;
      if (item.kind === "pair" && source.cell) {
        option_labels = {
          ANCHOR_WORD: source.cell.anchor_word,
          TARGET_WORD: source.cell.target_word,
        };
      }

      return {
        item_id: item.item_id,
        source_id: item.source_id,
        kind: item.kind,
        is_attention_check: item.is_attention_check ?? false,
        prompt,
        image,
        option_labels,
      };
    });
  }

  return {
    text: resolvePhase(form.text_items),
    image: resolvePhase(form.image_items),
    pair: resolvePhase(form.pair_items),
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────

/** Reset memoized state (used in tests when HS01_CONFIG/DATA_DIR changes). */
export function _resetStoreCache(): void {
  _configSha256 = null;
  _configPath = null;
  _mutexTail = Promise.resolve();
}

export async function createSession(
  channel?: RecruitmentChannel | null
): Promise<CreateResult> {
  return withMutex(async () => {
    ensureDataDir();

    const config = loadConfig();
    const configSha256 = getConfigSha256();

    // Form balancing (outside mutex-sensitive IO, but inside mutex for consistency)
    const counts = countActiveSessions();
    const formId = chooseLeastLoadedForm(counts);

    // Atomic counter increment
    const participantN = await incrementCounter();
    const participantCode = formatParticipantCode(participantN);

    // Generate session identity
    const sessionId = crypto.randomUUID();
    const rngSeed = crypto.randomBytes(16).toString("hex");

    // Build initial session record (status: abandoned per spec)
    const now = new Date().toISOString();
    const record: SessionRecord = {
      schema_version: "1.0.0",
      study_id: config.study_id,
      config_version: config.config_version,
      config_sha256: configSha256,
      session_id: sessionId,
      form_id: formId,
      rng_seed: rngSeed,
      status: "abandoned",
      participant: {
        participant_code: participantCode,
        recruitment_channel: channel ?? null,
        consent: {
          given: true,
          consent_version: config.consent.consent_version,
          at_utc: now,
        },
      },
      environment: {
        user_agent: "",
        viewport: { w: 0, h: 0 },
        device_pixel_ratio: 1,
      },
      timing: {
        started_at_utc: now,
      },
      phase_timings: [],
      trials: [],
    };

    await atomicWriteJson(sessionPath(sessionId), record);

    // Resolve form items (no analysis-internal fields)
    const items = resolveItems(formId);

    return {
      session_id: sessionId,
      participant_code: participantCode,
      form_id: formId,
      rng_seed: rngSeed,
      config_version: config.config_version,
      config_sha256: configSha256,
      consent_version: config.consent.consent_version,
      items,
    };
  });
}

export async function writeCheckpoint(
  id: string,
  record: SessionRecord
): Promise<{ ok: boolean; validation_errors: object[] | null }> {
  ensureDataDir();

  const valid = validateSession(record);
  let validationErrors: object[] | null = null;

  if (!valid && validateSession.errors) {
    validationErrors = validateSession.errors.map((e) => ({ ...e }));
    // Attach errors to record (x_ prefix allowed by schema patternProperties)
    (record as SessionRecord & { x_validation_errors: object[] }).x_validation_errors =
      validationErrors;
  } else if (valid) {
    // Clean up any previous errors if now valid
    delete (record as Record<string, unknown>).x_validation_errors;
  }

  await atomicWriteJson(sessionPath(id), record);

  return { ok: valid, validation_errors: validationErrors };
}

export async function submitSession(
  id: string,
  record: SessionRecord
): Promise<{ ok: boolean; validation_errors: object[] | null }> {
  ensureDataDir();

  const now = new Date().toISOString();

  // Promote to completed and stamp server timestamps
  const submitted: SessionRecord = {
    ...record,
    status: "completed",
    timing: {
      ...record.timing,
      completed_at_utc: now,
      server_received_at_utc: now,
    },
  };

  const valid = validateSession(submitted);
  let validationErrors: object[] | null = null;

  if (!valid && validateSession.errors) {
    validationErrors = validateSession.errors.map((e) => ({ ...e }));
    (submitted as SessionRecord & { x_validation_errors: object[] }).x_validation_errors =
      validationErrors;
  }

  await atomicWriteJson(sessionPath(id), submitted);

  return { ok: valid, validation_errors: validationErrors };
}

export async function statusCounts(): Promise<{
  forms: Record<string, number>;
  total_completed: number;
  total_in_progress: number;
}> {
  const dir = dataDir();
  if (!fs.existsSync(dir)) {
    return { forms: {}, total_completed: 0, total_in_progress: 0 };
  }

  const now = Date.now();
  const windowMs = 24 * 60 * 60 * 1000;

  const formsCompleted: Record<string, number> = {};
  let totalCompleted = 0;
  let totalInProgress = 0;

  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".json") && !f.startsWith("_"));

  for (const file of files) {
    const fullPath = path.join(dir, file);
    try {
      const stat = fs.statSync(fullPath);
      const raw = fs.readFileSync(fullPath, "utf-8");
      const rec = JSON.parse(raw) as Partial<SessionRecord>;
      const formId = rec.form_id;
      if (!formId) continue;

      if (rec.status === "completed") {
        totalCompleted++;
        formsCompleted[formId] = (formsCompleted[formId] ?? 0) + 1;
      } else {
        // In-progress: status=abandoned + mtime within 24h
        const isRecent = now - stat.mtimeMs < windowMs;
        if (isRecent) {
          totalInProgress++;
        }
      }
    } catch {
      // Skip
    }
  }

  return {
    forms: formsCompleted,
    total_completed: totalCompleted,
    total_in_progress: totalInProgress,
  };
}
