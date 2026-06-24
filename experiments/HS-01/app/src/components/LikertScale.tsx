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
      <legend className="text-base font-medium text-neutral-900 mb-4">
        {scale.statement}
      </legend>
      <div className="flex flex-col gap-2 sm:flex-row sm:gap-3">
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
                "flex-1 rounded-lg border px-3 py-3 text-sm text-center transition-colors",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
                selected
                  ? "border-blue-600 bg-blue-50 text-blue-900 font-medium"
                  : "border-neutral-300 bg-white text-neutral-700 hover:border-neutral-400",
                disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
              ].join(" ")}
            >
              <span aria-hidden="true" className="block text-xs text-neutral-400 mb-1">
                {point}
              </span>
              {label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
