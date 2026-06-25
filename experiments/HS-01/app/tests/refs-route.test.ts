/**
 * GET /api/refs/[name] route tests — run in Node environment (vitest).
 *
 * Mirrors the /api/images/[name] hardening: a strict ^ref-[a-z0-9_]+\.png$
 * allowlist, path-traversal rejection, 404 for well-formed-but-missing names,
 * and immutable cache headers for a real bundled reference image.
 */

import { describe, it, expect } from "vitest";
import path from "path";

const APP_DIR = path.resolve(__dirname, "..");
process.env.HS01_REFS_DIR = path.resolve(APP_DIR, "config/refs");

describe("GET /api/refs/[name]", () => {
  it("serves a real ref-*.png with image/png + immutable cache header", async () => {
    const { GET } = await import("../src/app/api/refs/[name]/route");
    const name = "ref-tench.png";
    const res = await GET(new Request(`http://localhost/api/refs/${name}`), {
      params: { name },
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("image/png");
    expect(res.headers.get("cache-control")).toBe(
      "public, max-age=31536000, immutable"
    );
    const buf = Buffer.from(await res.arrayBuffer());
    expect(buf.length).toBeGreaterThan(0);
    expect(buf[0]).toBe(0x89); // PNG magic
    expect(buf[1]).toBe(0x50);
  });

  it("rejects path traversal with 400", async () => {
    const { GET } = await import("../src/app/api/refs/[name]/route");
    const name = "../../etc/passwd";
    const res = await GET(new Request(`http://localhost/api/refs/x`), {
      params: { name },
    });
    expect(res.status).toBe(400);
  });

  it("rejects a non-ref name (src-foo.png) with 400", async () => {
    const { GET } = await import("../src/app/api/refs/[name]/route");
    const name = "src-clean-llava-American_robin-3.png";
    const res = await GET(new Request(`http://localhost/api/refs/${name}`), {
      params: { name },
    });
    expect(res.status).toBe(400);
  });

  it("returns 404 for a well-formed but missing ref-*.png", async () => {
    const { GET } = await import("../src/app/api/refs/[name]/route");
    const name = "ref-does_not_exist.png";
    const res = await GET(new Request(`http://localhost/api/refs/${name}`), {
      params: { name },
    });
    expect(res.status).toBe(404);
  });
});
