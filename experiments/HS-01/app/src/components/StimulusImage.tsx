"use client";

/**
 * StimulusImage — frozen-asset image renderer with onset instrumentation.
 *
 * Measurement contract:
 *  - NO UPSCALING: the <img> is sized to its NATURAL pixels (width/height attrs
 *    = natural_w/natural_h, and an explicit max-width:none so a container can
 *    never stretch it). css == natural is the invariant; it is measured and
 *    logged via awaitOnset's rendered_image.
 *  - Onset: once the bitmap is decoded (image_loaded_ms) and one rAF has passed
 *    (onset_ms), `onReady` fires with the full OnsetResult. The element ref is
 *    threaded through awaitOnset so decode()/getBoundingClientRect run on the
 *    real painted element.
 *  - Animation-free: framer-motion never wraps this; onset must be deterministic.
 *
 * The min-size GATE (viewport too small for min_rendered_image_css_px) is NOT
 * enforced here — it is a phase-entry decision made in the page, since refusing
 * is a flow concern, not a per-image one.
 */

import { useEffect, useRef } from "react";
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
  // Guard so onset fires exactly once per mounted stimulus.
  const firedFor = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const el = imgRef.current;
    if (!el) return;
    if (firedFor.current === uriName) return;
    firedFor.current = uriName;

    (async () => {
      const result = await awaitOnset(clock, el);
      if (!cancelled) onReady(result);
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uriName]);

  return (
    <div className="flex justify-center">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={`/api/images/${uriName}`}
        alt={alt ?? ""}
        width={naturalW}
        height={naturalH}
        // No upscaling: natural pixels, never stretched by the container.
        style={{
          width: `${naturalW}px`,
          height: `${naturalH}px`,
          maxWidth: "none",
        }}
        decoding="async"
        draggable={false}
      />
    </div>
  );
}
