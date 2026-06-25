"use client";

/**
 * PhaseProgress — a calm, non-distracting progress footer: the current section
 * name, a "Part N of 3" indicator, and a single bar showing OVERALL progress
 * across the whole session. Showing the section + part (instead of a bare
 * "1 of 34") avoids the misread that the current section has 34 items. It sits
 * at the FOOT of the trial column (not as a leading element) so the stimulus is
 * what the rater meets first.
 *
 * It must NOT reveal anything about item identity or expose attention-check
 * status. `position`/`total` are the overall (whole-session) counts, used only
 * to fill the bar + drive the aria values; the raw "x of y" number is
 * intentionally not shown.
 */

export interface PhaseProgressProps {
  /** 1-based current part. */
  part: number;
  /** Number of parts (sections) in the study. */
  partCount: number;
  /** Human-readable name of the current section. */
  sectionLabel: string;
  /** Overall 1-based trial position across the whole session (bar + aria). */
  position: number;
  /** Overall trial count across the whole session. */
  total: number;
}

export function PhaseProgress({
  part,
  partCount,
  sectionLabel,
  position,
  total,
}: PhaseProgressProps) {
  const pct = total > 0 ? Math.min(100, Math.round((position / total) * 100)) : 0;
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-ink">{sectionLabel}</span>
        <span className="text-xs font-medium uppercase tracking-wider text-muted">
          Part {part} of {partCount}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full bg-tum-500 transition-[width] duration-300 ease-out"
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={position}
          aria-valuemin={0}
          aria-valuemax={total}
          aria-label={`${sectionLabel}, part ${part} of ${partCount}`}
        />
      </div>
    </div>
  );
}
