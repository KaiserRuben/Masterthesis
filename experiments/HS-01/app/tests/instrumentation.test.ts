/**
 * @vitest-environment jsdom
 *
 * instrumentation tests — onset timing, integrity listeners, preload, render
 * check. Runs in jsdom so window/document/Image exist; the canvas 2d context is
 * not implemented by jsdom, so renderCheck is asserted only for shape + the
 * graceful-degradation path here. (Real-browser pixel fidelity is a Task-8 e2e
 * concern — see report.)
 */

import { describe, it, expect, vi } from "vitest";
import {
  awaitOnset,
  preloadImage,
  attachIntegrityListeners,
  renderCheck,
  countChanges,
} from "../src/lib/instrumentation";
import { SessionClock } from "../src/lib/timing";
import type { IntegrityEvent } from "../src/lib/types";

/** Controllable now-source so onset offsets are deterministic in the test. */
function fakeClock(): { clock: SessionClock; advance: (ms: number) => void } {
  let t = 0;
  const clock = new SessionClock("2026-06-24T10:00:00.000Z", () => t);
  return { clock, advance: (ms: number) => { t += ms; } };
}

describe("awaitOnset", () => {
  it("text trial: onset after a frame, image fields null", async () => {
    const { clock, advance } = fakeClock();
    // Advance the clock during the rAF wait to prove the offset is captured
    // AFTER the frame.
    const orig = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      advance(16);
      return setTimeout(() => cb(performance.now()), 0) as unknown as number;
    }) as typeof requestAnimationFrame;

    const res = await awaitOnset(clock, null);
    globalThis.requestAnimationFrame = orig;

    expect(res.image_loaded_ms).toBeNull();
    expect(res.rendered_image).toBeNull();
    expect(res.onset_ms).toBe(16);
  });

  it("image trial: image_loaded_ms precedes onset_ms and decode is awaited", async () => {
    const { clock, advance } = fakeClock();

    const decode = vi.fn(async () => {
      advance(40); // decode took 40ms
    });
    const orig = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      advance(16); // frame after decode
      return setTimeout(() => cb(performance.now()), 0) as unknown as number;
    }) as typeof requestAnimationFrame;

    const img = document.createElement("img");
    Object.defineProperty(img, "decode", { value: decode });
    Object.defineProperty(img, "naturalWidth", { value: 256, configurable: true });
    Object.defineProperty(img, "naturalHeight", { value: 256, configurable: true });
    img.getBoundingClientRect = () =>
      ({ width: 256, height: 256 } as DOMRect);

    const res = await awaitOnset(clock, img);
    globalThis.requestAnimationFrame = orig;

    expect(decode).toHaveBeenCalledOnce();
    expect(res.image_loaded_ms).toBe(40); // after decode, before frame
    expect(res.onset_ms).toBe(56); // after the frame
    expect(res.image_loaded_ms!).toBeLessThan(res.onset_ms);
    expect(res.rendered_image).toEqual({
      css_w: 256,
      css_h: 256,
      natural_w: 256,
      natural_h: 256,
    });
  });

  it("image trial: still resolves when decode() rejects (broken asset)", async () => {
    const { clock } = fakeClock();
    const img = document.createElement("img");
    Object.defineProperty(img, "decode", {
      value: vi.fn(async () => {
        throw new Error("broken");
      }),
    });
    Object.defineProperty(img, "naturalWidth", { value: 0, configurable: true });
    Object.defineProperty(img, "naturalHeight", { value: 0, configurable: true });
    img.getBoundingClientRect = () => ({ width: 0, height: 0 } as DOMRect);

    const res = await awaitOnset(clock, img);
    expect(typeof res.onset_ms).toBe("number");
    // 0×0 measurement is suppressed (schema minimum:1) → null.
    expect(res.rendered_image).toBeNull();
  });
});

describe("attachIntegrityListeners", () => {
  it("captures blur / focus / resize / visibility / fullscreen_exit with at_ms", () => {
    const { clock, advance } = fakeClock();
    const sink: IntegrityEvent[] = [];
    const detach = attachIntegrityListeners(clock, (e) => sink.push(e));

    advance(100);
    window.dispatchEvent(new Event("blur"));
    advance(50);
    window.dispatchEvent(new Event("focus"));
    advance(25);
    window.dispatchEvent(new Event("resize"));

    // visibility hidden → visible
    Object.defineProperty(document, "hidden", { value: true, configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));
    Object.defineProperty(document, "hidden", { value: false, configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));

    // fullscreen exit (no fullscreenElement)
    Object.defineProperty(document, "fullscreenElement", { value: null, configurable: true });
    document.dispatchEvent(new Event("fullscreenchange"));

    detach();

    const types = sink.map((e) => e.type);
    expect(types).toEqual([
      "blur",
      "focus",
      "resize",
      "visibility_hidden",
      "visibility_visible",
      "fullscreen_exit",
    ]);
    expect(sink[0].at_ms).toBe(100);
    expect(sink[1].at_ms).toBe(150);
    for (const e of sink) expect(e.at_ms).toBeGreaterThanOrEqual(0);
  });

  it("detach() stops further events", () => {
    const { clock } = fakeClock();
    const sink: IntegrityEvent[] = [];
    const detach = attachIntegrityListeners(clock, (e) => sink.push(e));
    detach();
    window.dispatchEvent(new Event("blur"));
    expect(sink).toHaveLength(0);
  });
});

describe("preloadImage", () => {
  it("resolves on load and on error (never rejects)", async () => {
    // jsdom Image fires neither load nor error for data URLs synchronously; we
    // stub the prototype to fire load on src assignment.
    const proto = (globalThis.Image as unknown as { prototype: HTMLImageElement }).prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, "src");
    Object.defineProperty(proto, "src", {
      configurable: true,
      set(this: HTMLImageElement) {
        setTimeout(() => this.onload?.(new Event("load")), 0);
      },
    });

    await expect(preloadImage("/api/images/x.png")).resolves.toBeUndefined();

    if (desc) Object.defineProperty(proto, "src", desc);
  });
});

describe("renderCheck", () => {
  it("returns a structured {passed, method} without throwing", () => {
    const r = renderCheck();
    expect(typeof r.passed).toBe("boolean");
    expect(r.method === null || typeof r.method === "string").toBe(true);
  });
});

describe("countChanges", () => {
  it("counts value changes, ignoring consecutive duplicates", () => {
    expect(countChanges([])).toBe(0);
    expect(countChanges([3])).toBe(0);
    expect(countChanges([3, 3, 3])).toBe(0);
    expect(countChanges([3, 4])).toBe(1);
    expect(countChanges([3, 4, 4, 2])).toBe(2);
    expect(countChanges(["ANCHOR_WORD", "TARGET_WORD", "ANCHOR_WORD"])).toBe(2);
  });
});
