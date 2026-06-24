/**
 * loadConfig()    — reads, validates (AJV), and memoizes the study config.
 * assertPoolBinding() — verifies that the config's pool_file_sha256 matches
 *                       the actual sha256 of the itempool.json bytes.
 *
 * Path resolution order:
 *   1. process.env.HS01_CONFIG (absolute path, set by next.config.js or test)
 *   2. Default: <app-root>/config/study-config.json
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";
import { validateConfig } from "./schemas";
import type { ItemPool, StudyConfig } from "./types";

const APP_DIR = path.resolve(__dirname, "../..");

function resolveConfigPath(): string {
  if (process.env.HS01_CONFIG) {
    return process.env.HS01_CONFIG;
  }
  return path.resolve(APP_DIR, "config/study-config.json");
}

let _config: StudyConfig | null = null;

export function loadConfig(configPath?: string): StudyConfig {
  if (_config !== null && !configPath) return _config;

  const cfgPath = configPath ?? resolveConfigPath();
  let raw: string;
  try {
    raw = fs.readFileSync(cfgPath, "utf-8");
  } catch (err) {
    throw new Error(
      `loadConfig: cannot read config file at "${cfgPath}": ${(err as Error).message}`
    );
  }

  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `loadConfig: failed to parse JSON at "${cfgPath}": ${(err as Error).message}`
    );
  }

  const valid = validateConfig(data);
  if (!valid) {
    const errors = validateConfig.errors
      ?.map((e) => `  ${e.instancePath || "/"} ${e.message}`)
      .join("\n");
    throw new Error(
      `loadConfig: study config at "${cfgPath}" failed schema validation:\n${errors}`
    );
  }

  const config = data as StudyConfig;
  if (!configPath) {
    _config = config;
  }
  return config;
}

/**
 * Recompute sha256 of the raw itempool file bytes and compare to
 * config.pool_ref.pool_file_sha256. Throws if they differ.
 *
 * Note: the pool_path is derived from HS01_POOL env or default — the same
 * path used by loadPool(). This guarantees we hash the file that was actually
 * loaded.
 */
export function assertPoolBinding(
  _pool: ItemPool,
  config: StudyConfig,
  poolPath?: string
): void {
  const resolvedPoolPath =
    poolPath ??
    process.env.HS01_POOL ??
    path.resolve(APP_DIR, "../pool_frozen/itempool.json");

  let bytes: Buffer;
  try {
    bytes = fs.readFileSync(resolvedPoolPath);
  } catch (err) {
    throw new Error(
      `assertPoolBinding: cannot read pool file at "${resolvedPoolPath}": ${(err as Error).message}`
    );
  }

  const actual = crypto.createHash("sha256").update(bytes).digest("hex");
  const expected = config.pool_ref.pool_file_sha256;

  if (actual !== expected) {
    throw new Error(
      `assertPoolBinding: pool_file_sha256 mismatch. ` +
        `config expects "${expected}" but file "${resolvedPoolPath}" hashes to "${actual}".`
    );
  }
}

/** Reset the memoized config (used in tests). */
export function _resetConfigCache(): void {
  _config = null;
}
