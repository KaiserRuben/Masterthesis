/**
 * loadPool() — reads, validates (AJV), and memoizes the frozen item pool.
 *
 * Path resolution order:
 *   1. process.env.HS01_POOL (absolute path, set by next.config.js or test)
 *   2. Default: <app-root>/../pool_frozen/itempool.json
 */

import fs from "fs";
import path from "path";
import { validatePool } from "./schemas";
import type { ItemPool } from "./types";

const APP_DIR = path.resolve(__dirname, "../..");

function resolvePoolPath(): string {
  if (process.env.HS01_POOL) {
    return process.env.HS01_POOL;
  }
  return path.resolve(APP_DIR, "../pool_frozen/itempool.json");
}

let _pool: ItemPool | null = null;

export function loadPool(): ItemPool {
  if (_pool !== null) return _pool;

  const poolPath = resolvePoolPath();
  let raw: string;
  try {
    raw = fs.readFileSync(poolPath, "utf-8");
  } catch (err) {
    throw new Error(
      `loadPool: cannot read pool file at "${poolPath}": ${(err as Error).message}`
    );
  }

  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `loadPool: failed to parse JSON at "${poolPath}": ${(err as Error).message}`
    );
  }

  const valid = validatePool(data);
  if (!valid) {
    const errors = validatePool.errors
      ?.map((e) => `  ${e.instancePath || "/"} ${e.message}`)
      .join("\n");
    throw new Error(
      `loadPool: item pool at "${poolPath}" failed schema validation:\n${errors}`
    );
  }

  _pool = data as ItemPool;
  return _pool;
}

/** Reset the memoized pool (used in tests). */
export function _resetPoolCache(): void {
  _pool = null;
}
