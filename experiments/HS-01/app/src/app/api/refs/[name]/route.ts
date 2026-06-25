/**
 * GET /api/refs/[name]
 *
 * Streams a bundled reference PNG (an example image for a fine-grained pair
 * option word) from refsDir(). These are committed, curated assets — never
 * study stimuli (enforced by tests/references-assets).
 *
 * Path-traversal hardening mirrors /api/images/[name]: the name must match
 * ^ref-[A-Za-z0-9_]+\.png$. Anything else (`..`, slashes, non-ref, wrong
 * extension) is rejected with 400 BEFORE the name is joined into a path. A
 * defensive path.basename equality check keeps the resolved file inside the
 * refs directory.
 *
 * 200 → image/png, Cache-Control: public, max-age=31536000, immutable
 * 400 → invalid / unsafe name
 * 404 → well-formed name but no such file
 */

import fs from "fs";
import path from "path";

import { NextResponse } from "next/server";

import { refsDir } from "@/lib/references";

export const dynamic = "force-dynamic";

// Strict allowlist: ref- prefix, [A-Za-z0-9_] body, .png suffix. No dots,
// slashes, or path separators can pass.
const NAME_RE = /^ref-[A-Za-z0-9_]+\.png$/;

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
): Promise<Response> {
  const name = params.name;

  if (typeof name !== "string" || !NAME_RE.test(name)) {
    return NextResponse.json({ error: "Invalid image name" }, { status: 400 });
  }

  const dir = refsDir();
  const filePath = path.join(dir, name);

  // Defensive: the joined path's basename must equal the sanitized name, and
  // the file must remain directly inside the refs dir.
  if (path.basename(filePath) !== name || path.dirname(filePath) !== dir) {
    return NextResponse.json({ error: "Invalid image name" }, { status: 400 });
  }

  let data: Buffer;
  try {
    data = fs.readFileSync(filePath);
  } catch {
    return NextResponse.json({ error: "Image not found" }, { status: 404 });
  }

  // Copy into a Uint8Array backed by a plain ArrayBuffer, then wrap in a Blob —
  // an unambiguous Web BodyInit (a raw Node Buffer's `.buffer` is ArrayBufferLike
  // and does not satisfy BodyInit under lib-dom).
  const bytes = Uint8Array.from(data);
  const body = new Blob([bytes], { type: "image/png" });

  return new NextResponse(body, {
    status: 200,
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}
