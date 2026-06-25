/**
 * AJV instance configured for JSON Schema draft 2020-12 + ajv-formats.
 *
 * Compiles and exports validators for the three HS-01 schemas:
 *   - validatePool    (itempool)
 *   - validateConfig  (study-config)
 *   - validateSession (session record)
 *
 * The schema files live at ../../schemas/ relative to the app root.
 */

import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";
import path from "path";
import fs from "fs";
import type { ValidateFunction } from "ajv";
import type {
  ItemPool,
  StudyConfig,
  SessionRecord,
  ReferenceData,
} from "./types";

/**
 * Locate the schemas directory. Resolution order:
 *   1. process.env.HS01_SCHEMAS_DIR (explicit override)
 *   2. <repo>/experiments/HS-01/schemas relative to this module's __dirname
 *      (correct under ts-node / vitest where __dirname === src/lib)
 *   3. <cwd>/../schemas and <cwd>/schemas (correct when the Next server runs
 *      bundled: __dirname points into .next/, but cwd is the app dir)
 *
 * The first candidate that actually contains the itempool schema wins. This
 * keeps the loader working in dev, test, and the bundled `next build` server
 * where __dirname is unreliable.
 */
function resolveSchemasDir(): string {
  const probe = "hs01.itempool.schema.json";
  const candidates = [
    process.env.HS01_SCHEMAS_DIR,
    path.resolve(__dirname, "../../../schemas"),
    path.resolve(process.cwd(), "../schemas"),
    path.resolve(process.cwd(), "schemas"),
  ].filter((c): c is string => typeof c === "string" && c.length > 0);

  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, probe))) return dir;
  }
  // Fall back to the original relative path so the error message is the
  // familiar ENOENT pointing at the expected location.
  return path.resolve(__dirname, "../../../schemas");
}

const SCHEMAS_DIR = resolveSchemasDir();

function readSchema(filename: string): object {
  const fullPath = path.join(SCHEMAS_DIR, filename);
  const raw = fs.readFileSync(fullPath, "utf-8");
  return JSON.parse(raw) as object;
}

// Build a single AJV instance with all three schemas loaded
const ajv = new Ajv2020({ strict: false, allErrors: true });
addFormats(ajv);

const itempoolSchema = readSchema("hs01.itempool.schema.json");
const studyConfigSchema = readSchema("hs01.study-config.schema.json");
const sessionSchema = readSchema("hs01.session.schema.json");
const referencesSchema = readSchema("hs01.references.schema.json");

// Add all schemas first so cross-references can resolve
ajv.addSchema(itempoolSchema);
ajv.addSchema(studyConfigSchema);
ajv.addSchema(sessionSchema);
ajv.addSchema(referencesSchema);

export const validatePool = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/itempool-1.0.0.schema.json"
) as ValidateFunction<ItemPool>;

export const validateConfig = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/study-config-1.0.0.schema.json"
) as ValidateFunction<StudyConfig>;

export const validateSession = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/session-1.0.0.schema.json"
) as ValidateFunction<SessionRecord>;

export const validateReferences = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/references-1.0.0.schema.json"
) as ValidateFunction<ReferenceData>;

if (!validatePool) throw new Error("Failed to compile validatePool");
if (!validateConfig) throw new Error("Failed to compile validateConfig");
if (!validateSession) throw new Error("Failed to compile validateSession");
if (!validateReferences) throw new Error("Failed to compile validateReferences");
