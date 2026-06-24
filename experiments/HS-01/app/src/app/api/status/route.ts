/**
 * GET /api/status
 *
 * Recruitment dashboard counts. Delegates to the store layer.
 *
 * 200 → { forms, total_completed, total_in_progress }
 * 500 → boot guard / config error
 */

import { NextResponse } from "next/server";

import { ensureInputs } from "@/lib/boot";
import { statusCounts } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  ensureInputs();
  const counts = await statusCounts();
  return NextResponse.json(counts, { status: 200 });
}
