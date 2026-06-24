/**
 * GET /api/images/[name]
 *
 * Streams a frozen PNG asset from HS01_IMAGE_DIR.
 *
 * Path-traversal hardening: the name must match ^src-[A-Za-z0-9_-]+\.png$.
 * Anything else (`..`, slashes, non-src, wrong extension) is rejected with 400
 * BEFORE the name is ever joined into a filesystem path. A defensive
 * path.basename equality check guarantees the resolved file stays inside the
 * image directory.
 *
 * 200 → image/png, Cache-Control: public, max-age=31536000, immutable
 * 400 → invalid / unsafe name
 * 404 → well-formed name but no such file
 */

import fs from "fs";
import path from "path";

import { NextResponse } from "next/server";

import { imageDir } from "@/lib/config";

export const dynamic = "force-dynamic";

// Strict allowlist: src- prefix, [A-Za-z0-9_-] body, .png suffix. No dots,
// slashes, or path separators can pass.
const NAME_RE = /^src-[A-Za-z0-9_-]+\.png$/;

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
): Promise<Response> {
  const name = params.name;

  if (typeof name !== "string" || !NAME_RE.test(name)) {
    return NextResponse.json(
      { error: "Invalid image name" },
      { status: 400 }
    );
  }

  const dir = imageDir();
  const filePath = path.join(dir, name);

  // Defensive: the joined path's basename must equal the sanitized name, and
  // the file must remain directly inside the image dir. (The regex already
  // forbids separators; this is belt-and-suspenders.)
  if (path.basename(filePath) !== name || path.dirname(filePath) !== dir) {
    return NextResponse.json(
      { error: "Invalid image name" },
      { status: 400 }
    );
  }

  let data: Buffer;
  try {
    data = fs.readFileSync(filePath);
  } catch {
    return NextResponse.json({ error: "Image not found" }, { status: 404 });
  }

  // Copy into a Uint8Array backed by a plain ArrayBuffer, then wrap in a Blob —
  // an unambiguous Web BodyInit. A raw Node Buffer's `.buffer` is typed
  // ArrayBufferLike (possibly SharedArrayBuffer), which does not satisfy
  // BlobPart/BodyInit under lib-dom.
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
