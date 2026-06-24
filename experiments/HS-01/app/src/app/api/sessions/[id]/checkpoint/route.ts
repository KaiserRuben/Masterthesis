/**
 * PUT /api/sessions/[id]/checkpoint
 *
 * Idempotent partial save. Body is the record-as-known so far; the store
 * persists it atomically (best-effort validation, never loses data) and
 * reports any schema errors back to the client.
 *
 * 200 → { ok, validation_errors }
 * 400 → malformed body
 * 500 → boot guard / config error
 */

import { NextResponse } from "next/server";

import { ensureInputs } from "@/lib/boot";
import { writeCheckpoint } from "@/lib/store";
import type { SessionRecord } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function PUT(
  req: Request,
  { params }: { params: { id: string } }
): Promise<Response> {
  ensureInputs();

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Malformed JSON body" }, { status: 400 });
  }

  if (typeof body !== "object" || body === null) {
    return NextResponse.json(
      { error: "Body must be a JSON object" },
      { status: 400 }
    );
  }

  const result = await writeCheckpoint(params.id, body as SessionRecord);
  return NextResponse.json(result, { status: 200 });
}
