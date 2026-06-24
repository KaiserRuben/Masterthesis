/**
 * Boot guard — server-only.
 *
 * ensureInputs() runs loadPool / loadConfig / assertPoolBinding once and
 * throws a fatal error on any mismatch. Call it from a server-side
 * route/layout module so the app refuses to start with invalid inputs.
 */

import { loadPool } from "./pool";
import { loadConfig, assertPoolBinding } from "./config";
import path from "path";

const APP_DIR = path.resolve(__dirname, "../..");

export interface EnsureInputsOptions {
  /** Override config file path (used in tests to supply a tampered config). */
  configPath?: string;
  /** Override pool file path. */
  poolPath?: string;
}

let _verified = false;

export function ensureInputs(opts: EnsureInputsOptions = {}): void {
  // When called without options we can short-circuit after the first success
  if (_verified && !opts.configPath && !opts.poolPath) return;

  const poolPath =
    opts.poolPath ??
    process.env.HS01_POOL ??
    path.resolve(APP_DIR, "../pool_frozen/itempool.json");

  const pool = loadPool();

  // If a custom configPath is supplied, load without memoization
  const config = opts.configPath
    ? loadConfig(opts.configPath)
    : loadConfig();

  assertPoolBinding(pool, config, poolPath);

  if (!opts.configPath && !opts.poolPath) {
    _verified = true;
  }
}

/** Reset boot guard state (tests only). */
export function _resetBootGuard(): void {
  _verified = false;
}
