"use client";

/**
 * PhaseProgress — a minimal, non-distracting progress indicator for the current
 * phase ("question 3 of 11" + a bar). Position is 1-based for display. It must
 * NOT reveal anything about item identity or expose attention-check status.
 */

export interface PhaseProgressProps {
  /** 1-based position within the phase. */
  position: number;
  total: number;
  label?: string;
}

export function PhaseProgress({ position, total, label }: PhaseProgressProps) {
  const pct = total > 0 ? Math.min(100, Math.round((position / total) * 100)) : 0;
  return (
    <div className="mb-6">
      <div className="flex justify-between text-xs text-neutral-500 mb-1">
        <span>{label ?? "Progress"}</span>
        <span>
          {position} of {total}
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-neutral-200">
        <div
          className="h-1.5 rounded-full bg-blue-600 transition-[width] duration-200"
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={position}
          aria-valuemin={0}
          aria-valuemax={total}
        />
      </div>
    </div>
  );
}
