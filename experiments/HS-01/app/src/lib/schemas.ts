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
import type { ItemPool, StudyConfig, SessionRecord } from "./types";

const SCHEMAS_DIR = path.resolve(__dirname, "../../../schemas");

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

// Add all schemas first so cross-references can resolve
ajv.addSchema(itempoolSchema);
ajv.addSchema(studyConfigSchema);
ajv.addSchema(sessionSchema);

export const validatePool = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/itempool-1.0.0.schema.json"
) as ValidateFunction<ItemPool>;

export const validateConfig = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/study-config-1.0.0.schema.json"
) as ValidateFunction<StudyConfig>;

export const validateSession = ajv.getSchema(
  "https://masterthesis.local/schemas/hs01/session-1.0.0.schema.json"
) as ValidateFunction<SessionRecord>;

if (!validatePool) throw new Error("Failed to compile validatePool");
if (!validateConfig) throw new Error("Failed to compile validateConfig");
if (!validateSession) throw new Error("Failed to compile validateSession");
