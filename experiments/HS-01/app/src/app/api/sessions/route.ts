/**
 * POST /api/sessions
 *
 * Creates a new participant session. Validates the optional recruitment_channel
 * against the allowed enum and delegates to the store layer's createSession,
 * whose payload already strips all analysis-only provenance.
 *
 * 201 → { session_id, participant_code, form_id, rng_seed, config_version,
 *         config_sha256, consent_version, items }
 * 400 → invalid recruitment_channel / malformed body
 * 500 → boot guard / config error (pool-hash mismatch surfaces here)
 */

import { NextResponse } from "next/server";

import { ensureInputs } from "@/lib/boot";
import { createSession } from "@/lib/store";
import type { RecruitmentChannel } from "@/lib/store";

export const dynamic = "force-dynamic";

const ALLOWED_CHANNELS: ReadonlyArray<RecruitmentChannel> = [
  "tum",
  "fortiss",
  "personal",
  "other",
];

export async function POST(req: Request): Promise<Response> {
  // Boot guard — fail fast (and loud) on pool-hash mismatch / bad inputs.
  ensureInputs();

  let body: unknown = {};
  const raw = await req.text();
  if (raw.trim().length > 0) {
    try {
      body = JSON.parse(raw);
    } catch {
      return NextResponse.json({ error: "Malformed JSON body" }, { status: 400 });
    }
  }

  if (typeof body !== "object" || body === null) {
    return NextResponse.json(
      { error: "Body must be a JSON object" },
      { status: 400 }
    );
  }

  const channel = (body as Record<string, unknown>).recruitment_channel;

  // null / undefined are both acceptable (anonymous channel).
  let resolved: RecruitmentChannel | null = null;
  if (channel !== undefined && channel !== null) {
    if (
      typeof channel !== "string" ||
      !ALLOWED_CHANNELS.includes(channel as RecruitmentChannel)
    ) {
      return NextResponse.json(
        {
          error: `Invalid recruitment_channel. Allowed: ${ALLOWED_CHANNELS.join(
            ", "
          )}, null`,
        },
        { status: 400 }
      );
    }
    resolved = channel as RecruitmentChannel;
  }

  const result = await createSession(resolved);
  return NextResponse.json(result, { status: 201 });
}
