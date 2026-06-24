/**
 * instrumentation.ts — onset timing, integrity events, and the render-fidelity
 * check. Every browser-API access is guarded so importing this module under SSR
 * or in a jsdom test never crashes.
 *
 * Onset model (binding, per hs01.session.schema.json): a trial's onset_ms is the
 * instant the stimulus is FULLY RENDERED, not navigation:
 *   - image / pair: await img.decode() (→ image_loaded_ms), then ONE
 *     requestAnimationFrame (→ onset_ms). image_loaded_ms therefore always
 *     precedes onset_ms.
 *   - text: one requestAnimationFrame; image fields are null.
 *
 * Presentation is animation-free (framer-motion only on chrome) precisely so
 * this onset is deterministic.
 */

import type { SessionClock } from "./timing";
import type { IntegrityEvent } from "./types";

export interface RenderedImage {
  css_w: number;
  css_h: number;
  natural_w: number;
  natural_h: number;
}

export interface OnsetResult {
  onset_ms: number;
  image_loaded_ms: number | null;
  rendered_image: RenderedImage | null;
}

const isBrowser = (): boolean => typeof window !== "undefined";

/**
 * Await one animation frame. Falls back to a microtask/timeout when rAF is
 * unavailable (SSR / jsdom) so callers always resolve.
 */
function nextFrame(): Promise<void> {
  if (typeof requestAnimationFrame === "function") {
    return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
  return new Promise<void>((resolve) => setTimeout(resolve, 0));
}

/**
 * Measure the realized on-screen vs natural size of an image element. Returns
 * null if dimensions are not yet known (avoids emitting an invalid 0×0 record,
 * which the schema forbids — minimum:1).
 */
function measure(imgEl: HTMLImageElement): RenderedImage | null {
  const rect = imgEl.getBoundingClientRect();
  const cssW = Math.round(rect.width);
  const cssH = Math.round(rect.height);
  const natW = imgEl.naturalWidth;
  const natH = imgEl.naturalHeight;
  if (cssW < 1 || cssH < 1 || natW < 1 || natH < 1) return null;
  return { css_w: cssW, css_h: cssH, natural_w: natW, natural_h: natH };
}

/**
 * awaitOnset — resolve when the stimulus is fully rendered, stamping monotonic
 * offsets from the clock.
 *
 * With an image element: await decode() (stamp image_loaded_ms), then one rAF
 * (stamp onset_ms), then measure rendered_image. Without an element (text
 * trials): one rAF, image fields null. image_loaded_ms is guaranteed ≤ onset_ms
 * because it is stamped strictly before the awaited frame.
 */
export async function awaitOnset(
  clock: SessionClock,
  imgEl?: HTMLImageElement | null
): Promise<OnsetResult> {
  if (imgEl) {
    // decode() resolves once the bitmap is ready to paint without jank.
    if (typeof imgEl.decode === "function") {
      try {
        await imgEl.decode();
      } catch {
        // decode can reject if the element is detached or the src is broken;
        // fall through — onset still fires so the trial is never wedged.
      }
    }
    const imageLoadedMs = clock.nowMs();
    await nextFrame();
    const onsetMs = clock.nowMs();
    return {
      onset_ms: onsetMs,
      image_loaded_ms: imageLoadedMs,
      rendered_image: measure(imgEl),
    };
  }

  // Text trial: a single frame after the DOM update.
  await nextFrame();
  return { onset_ms: clock.nowMs(), image_loaded_ms: null, rendered_image: null };
}

/**
 * preloadImage — warm the browser cache + decode for the NEXT trial's asset so
 * its awaitOnset resolves quickly. Resolves on load or error (never rejects);
 * no-op under SSR.
 */
export function preloadImage(url: string): Promise<void> {
  if (!isBrowser() || typeof Image === "undefined") return Promise.resolve();
  return new Promise<void>((resolve) => {
    const img = new Image();
    const done = (): void => resolve();
    img.onload = () => {
      if (typeof img.decode === "function") {
        img.decode().then(done, done);
      } else {
        done();
      }
    };
    img.onerror = done;
    img.src = url;
  });
}

/** A sink for integrity events (e.g. an array push or a store dispatch). */
export type IntegritySink = (ev: IntegrityEvent) => void;

/**
 * attachIntegrityListeners — wire up blur/focus/visibility/resize/fullscreen
 * listeners that push {at_ms, type} events into the sink. Returns a detach
 * function. No-op (returns a no-op detach) under SSR.
 */
export function attachIntegrityListeners(
  clock: SessionClock,
  sink: IntegritySink
): () => void {
  if (!isBrowser()) return () => {};

  const emit = (type: IntegrityEvent["type"], detail: string | null = null): void => {
    sink({ at_ms: clock.nowMs(), type, detail });
  };

  const onBlur = (): void => emit("blur");
  const onFocus = (): void => emit("focus");
  const onResize = (): void => emit("resize");
  const onVisibility = (): void => {
    if (typeof document === "undefined") return;
    emit(document.hidden ? "visibility_hidden" : "visibility_visible");
  };
  const onFullscreen = (): void => {
    if (typeof document === "undefined") return;
    // Only the EXIT transition is a quality signal per schema.
    if (!document.fullscreenElement) emit("fullscreen_exit");
  };

  window.addEventListener("blur", onBlur);
  window.addEventListener("focus", onFocus);
  window.addEventListener("resize", onResize);
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", onVisibility);
    document.addEventListener("fullscreenchange", onFullscreen);
  }

  return () => {
    window.removeEventListener("blur", onBlur);
    window.removeEventListener("focus", onFocus);
    window.removeEventListener("resize", onResize);
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", onVisibility);
      document.removeEventListener("fullscreenchange", onFullscreen);
    }
  };
}

export interface RenderCheckResult {
  passed: boolean;
  method: string | null;
}

/**
 * renderCheck — confirm the browser renders a known homoglyph string WITHOUT
 * normalizing it (a stimulus-fidelity guard: some environments silently fold
 * Cyrillic/Greek look-alikes to ASCII, which would corrupt the very thing the
 * study measures).
 *
 * Method: canvas-render a homoglyph variant and its pure-ASCII twin. The
 * homoglyph codepoints have different glyph metrics/pixels from ASCII, so if the
 * two renders DIFFER, no normalization occurred → pass. If they render
 * identically, the environment folded the homoglyphs → fail.
 *
 * Degrades gracefully: returns {passed:false, method:"unsupported"} where no
 * canvas exists (SSR), since fidelity cannot be asserted.
 */
export function renderCheck(): RenderCheckResult {
  if (!isBrowser() || typeof document === "undefined") {
    return { passed: false, method: "unsupported" };
  }
  try {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return { passed: false, method: "unsupported" };

    canvas.width = 256;
    canvas.height = 64;
    ctx.font = "32px monospace";
    ctx.textBaseline = "top";

    // "apple" with Cyrillic а (U+0430) and е (U+0435) swapped in for ASCII a/e.
    const homoglyph = "аppлe"; // а p p l(Cyr) e — visually "apple"
    const ascii = "apple";

    const render = (s: string): string => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillText(s, 0, 0);
      return canvas.toDataURL();
    };

    const homoData = render(homoglyph);
    const asciiData = render(ascii);

    // Different pixels ⇒ homoglyph codepoints were preserved (not normalized).
    const passed = homoData !== asciiData;
    return { passed, method: "canvas_homoglyph_v1" };
  } catch {
    return { passed: false, method: "unsupported" };
  }
}

/**
 * countChanges — derive n_changes from a sequence of selections: the number of
 * times the answer actually changed value before submit. Consecutive duplicates
 * do not count.
 */
export function countChanges(selections: Array<string | number | null>): number {
  let changes = 0;
  let prev: string | number | null | undefined = undefined;
  for (const s of selections) {
    if (prev !== undefined && s !== prev) changes++;
    prev = s;
  }
  return changes;
}
