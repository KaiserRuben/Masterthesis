"use client";

/**
 * LikertScale — a 5-point, verbally-anchored agreement scale.
 *
 * Measurement contract:
 *  - Emits the chosen value 1–5 (1 = first point_label … 5 = last).
 *  - Reports n_changes alongside each emission: the number of times the answer
 *    has CHANGED value since first selection. The first pick reports 0;
 *    re-selecting the same value does not increment. The count is tracked here
 *    (not derived in the parent) so it survives re-renders within a trial.
 *  - Animation-free: no framer-motion. Onset of the response widget must not
 *    depend on a running transition.
 *
 * Controlled `value` (1–5 or null) so a resumed trial can repaint the prior
 * selection; the internal change-counter is seeded from `value` on mount.
 */

import { useRef } from "react";
import type { Scale } from "@/lib/types";

export interface LikertScaleProps {
  scale: Scale;
  value: number | null;
  onChange: (value: number, nChanges: number) => void;
  disabled?: boolean;
}

export function LikertScale({ scale, value, onChange, disabled }: LikertScaleProps) {
  // Change-tracking state lives in a ref: it is measurement metadata, not
  // render state, and must not be reset by a parent re-render mid-trial.
  const lastValue = useRef<number | null>(value);
  const nChanges = useRef<number>(0);
  const hasSelected = useRef<boolean>(value != null);

  const select = (point: number) => {
    if (disabled) return;
    if (!hasSelected.current) {
      hasSelected.current = true;
    } else if (point !== lastValue.current) {
      nChanges.current += 1;
    }
    lastValue.current = point;
    onChange(point, nChanges.current);
  };

  return (
    <fieldset
      className="border-0 p-0 m-0"
      role="radiogroup"
      aria-label={scale.statement}
    >
      <legend className="text-base font-medium text-ink mb-4">
        {scale.statement}
      </legend>
      <div className="flex flex-col gap-2 sm:flex-row sm:gap-2.5">
        {scale.point_labels.map((label, idx) => {
          const point = idx + 1;
          const selected = value === point;
          return (
            <button
              key={point}
              type="button"
              role="radio"
              aria-checked={selected}
              aria-label={label}
              data-testid={`likert-point-${point}`}
              disabled={disabled}
              onClick={() => select(point)}
              className={[
                // Mobile: compact single-line rows (number badge + anchor),
                // ample tap height. Desktop (sm+): equal columns, stacked.
                "flex min-h-[44px] w-full items-center gap-3 rounded-control border px-3.5 py-2.5 text-left text-sm transition-colors",
                "sm:min-h-[88px] sm:flex-1 sm:flex-col sm:justify-center sm:gap-1.5 sm:px-3 sm:py-3 sm:text-center",
                selected
                  ? "border-tum-500 bg-tum-50 text-tum-900 font-medium"
                  : "border-line bg-white text-body hover:border-tum-300",
                disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
              ].join(" ")}
            >
              <span
                aria-hidden="true"
                className={[
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs tabular-nums sm:h-auto sm:w-auto sm:rounded-none",
                  selected ? "bg-tum-100 text-tum-700 sm:bg-transparent" : "bg-surface text-muted sm:bg-transparent",
                ].join(" ")}
              >
                {point}
              </span>
              <span className="leading-snug">{label}</span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
