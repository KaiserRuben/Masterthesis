/**
 * POST /api/sessions/[id]/submit
 *
 * Final submission. Body is the full record; the store promotes it to
 * "completed", stamps server timestamps, validates against the session schema,
 * and persists atomically.
 *
 * 200 → { ok, validation_errors }
 * 400 → malformed body
 * 500 → boot guard / config error
 */

import { NextResponse } from "next/server";

import { ensureInputs } from "@/lib/boot";
import { submitSession } from "@/lib/store";
import type { SessionRecord } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function POST(
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

  const result = await submitSession(params.id, body as SessionRecord);
  return NextResponse.json(result, { status: 200 });
}
