"use client";

/**
 * MotionProvider — global framer-motion configuration.
 *
 * `reducedMotion="user"` makes every motion element honor the OS-level
 * "prefers-reduced-motion" setting: transform and layout animations are
 * skipped for users who asked for less motion (an accessibility requirement),
 * while opacity fades are kept. This also stabilizes automated testing — with
 * reduced motion the chrome no longer animates its position, so transient
 * translate animations can't make a control read as "not stable".
 *
 * It does NOT touch stimulus timing: stimuli are never wrapped in motion
 * elements (onset must stay deterministic), so onset measurement is unaffected.
 */

import { MotionConfig } from "framer-motion";

export function MotionProvider({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
