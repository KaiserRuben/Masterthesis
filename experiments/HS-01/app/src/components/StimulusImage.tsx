"use client";

/**
 * StimulusImage — frozen-asset image renderer with onset instrumentation.
 *
 * Measurement contract:
 *  - NO UPSCALING (inline): the <img> is capped at its NATURAL pixels (width set
 *    to natural_w, never exceeded) so it is never enlarged past source. On a
 *    viewport narrower than the image it DOWNSCALES to fit (max-width:100%,
 *    height:auto, aspect preserved) instead of overflowing — many raters are on
 *    phones and a ~500px stimulus would otherwise spill past a ~390px screen.
 *    css <= natural is the invariant (the schema's "no upscaling" rule); the
 *    realized css size is measured + logged per-trial via awaitOnset's
 *    rendered_image, so analysis sees exactly which trials were shown downscaled.
 *    Tap-to-zoom (below) restores full detail on demand.
 *  - Onset: once the bitmap is decoded (image_loaded_ms) and one rAF has passed
 *    (onset_ms), `onReady` fires with the full OnsetResult. The element ref is
 *    threaded through awaitOnset so decode()/getBoundingClientRect run on the
 *    real painted element.
 *  - Animation-free: framer-motion never wraps this; onset must be deterministic.
 *
 * Zoom/lightbox: clicking, tapping, or keyboard-activating the stimulus opens a
 * full-screen overlay that scales the image to fill the viewport (object-contain,
 * no overflow); clicking again or pressing Escape closes it. The overlay is a
 * SEPARATE <img> (same cached src) — the inline, measured stimulus is never
 * remounted or resized, so onset timing and the recorded rendered_image are
 * unaffected. The min-size GATE (viewport too small for min_rendered_image_css_px)
 * is enforced in the page, not here.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { SessionClock } from "@/lib/timing";
import { awaitOnset, type OnsetResult } from "@/lib/instrumentation";

export interface StimulusImageProps {
  /** Basename of the frozen asset; served via /api/images/<uri_name>. */
  uriName: string;
  naturalW: number;
  naturalH: number;
  clock: SessionClock;
  /** Fires once the stimulus is fully rendered (decode + one frame). */
  onReady: (result: OnsetResult) => void;
  alt?: string;
}

export function StimulusImage({
  uriName,
  naturalW,
  naturalH,
  clock,
  onReady,
  alt,
}: StimulusImageProps) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const [zoomed, setZoomed] = useState(false);

  const src = `/api/images/${uriName}`;
  const altText = alt ?? "";

  // Onset fires once per uriName. We rely on the `cancelled` flag (NOT a
  // "fired once" ref): under React StrictMode the effect runs setup→cleanup→setup
  // on mount, so a one-shot ref would let the first (cancelled) run win and the
  // second run skip — leaving onReady un-called and the stimulus stuck on
  // "Preparing…". The cancelled-flag pattern is StrictMode-safe: the first run is
  // cancelled, the second run completes and fires onReady exactly once.
  useEffect(() => {
    let cancelled = false;
    const el = imgRef.current;
    if (!el) return;

    (async () => {
      const result = await awaitOnset(clock, el);
      if (!cancelled) onReady(result);
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uriName]);

  const close = useCallback(() => {
    setZoomed(false);
    // Return focus to the trigger so keyboard users land where they left off.
    triggerRef.current?.focus();
  }, []);

  // While open: focus the close control and let Escape dismiss the overlay.
  useEffect(() => {
    if (!zoomed) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [zoomed, close]);

  return (
    <div className="flex justify-center">
      {/* Trigger: a focusable wrapper so Tab + Enter/Space opens the lightbox.
          It wraps — and never resizes — the native-pixel stimulus below. */}
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setZoomed(true)}
        aria-label="Enlarge image to full screen"
        aria-haspopup="dialog"
        className="block cursor-zoom-in rounded-control"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={src}
          alt={altText}
          width={naturalW}
          height={naturalH}
          // No upscaling (width capped at natural), but downscale to fit a
          // narrow viewport instead of overflowing it. height:auto keeps the
          // aspect ratio; the width/height attrs reserve correct space.
          style={{
            width: `${naturalW}px`,
            height: "auto",
            maxWidth: "100%",
          }}
          decoding="async"
          draggable={false}
        />
      </button>

      {zoomed && (
        // Full-screen lightbox. Clicking anywhere (backdrop or image) closes it.
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Enlarged image"
          onClick={close}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/90 p-4"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={altText}
            // Fill the viewport, preserving aspect ratio, with no overflow.
            className="h-full w-full select-none object-contain"
            draggable={false}
          />
          <button
            ref={closeRef}
            type="button"
            onClick={close}
            aria-label="Close enlarged image"
            className="absolute right-4 top-4 rounded-full bg-white/10 px-3 py-1 text-2xl leading-none text-white hover:bg-white/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
