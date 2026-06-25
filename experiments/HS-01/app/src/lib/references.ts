/**
 * loadReferences() — reads, validates (AJV), and memoizes the curated
 * pair-option word reference map (config/references.json).
 *
 * The map provides, for the fine-grained classes that appear as pair options,
 * a one-line gloss and a bundled example image. It is read-only presentation
 * data forwarded to the rater UI; it never carries analysis fields.
 *
 * Path resolution mirrors loadConfig():
 *   1. process.env.HS01_REFERENCES (absolute path, set by next.config.js / Docker)
 *   2. Default: <app-root>/config/references.json
 *
 * refsDir() resolves the directory the bundled reference PNGs live in, served
 * by /api/refs/[name]:
 *   1. process.env.HS01_REFS_DIR
 *   2. Default: <app-root>/config/refs
 */

import fs from "fs";
import path from "path";
import { validateReferences } from "./schemas";
import type { ReferenceData } from "./types";

const APP_DIR = path.resolve(__dirname, "../..");

export function referencesPath(): string {
  return (
    process.env.HS01_REFERENCES ??
    path.resolve(APP_DIR, "config/references.json")
  );
}

export function refsDir(): string {
  return process.env.HS01_REFS_DIR ?? path.resolve(APP_DIR, "config/refs");
}

let _references: ReferenceData | null = null;

export function loadReferences(refPath?: string): ReferenceData {
  if (_references !== null && !refPath) return _references;

  const p = refPath ?? referencesPath();
  let raw: string;
  try {
    raw = fs.readFileSync(p, "utf-8");
  } catch (err) {
    throw new Error(
      `loadReferences: cannot read references file at "${p}": ${(err as Error).message}`
    );
  }

  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `loadReferences: failed to parse JSON at "${p}": ${(err as Error).message}`
    );
  }

  const valid = validateReferences(data);
  if (!valid) {
    const errors = validateReferences.errors
      ?.map((e) => `  ${e.instancePath || "/"} ${e.message}`)
      .join("\n");
    throw new Error(
      `loadReferences: references file at "${p}" failed schema validation:\n${errors}`
    );
  }

  const references = data as ReferenceData;
  if (!refPath) {
    _references = references;
  }
  return references;
}

/** Reset the memoized references (used in tests). */
export function _resetReferencesCache(): void {
  _references = null;
}
