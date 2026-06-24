/**
 * timing.ts — monotonic session clock.
 *
 * TIMING MODEL (binding, per hs01.session.schema.json): every *_ms field is a
 * monotonic millisecond offset from session start, derived from
 * performance.now() so it is immune to wall-clock adjustment. Reaction times are
 * NEVER stored — they are differences of these offsets computed in analysis.
 *
 * Across reloads the perf timeline resets to 0, so we CHAIN epochs using the
 * page-load wall-clock instant:
 *
 *     offset = (load_epoch_utc − started_at_utc) + (perf.now() − perf_at_load)
 *
 * The first term advances by the real elapsed time across the reload; the second
 * is the fresh perf delta. This keeps nowMs() strictly increasing across a
 * reload even though performance.now() restarted.
 */

import type { PhaseTiming } from "./types";

type NowSource = () => number;

/** SSR/jsdom-safe default now-source. performance exists in Node ≥16 and in
 *  jsdom, but we never assume it. */
function defaultNow(): NowSource {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return () => performance.now();
  }
  // Last-resort monotonic-ish fallback (used only if performance is absent).
  if (typeof process !== "undefined" && typeof process.hrtime === "function") {
    const start = process.hrtime.bigint();
    return () => Number(process.hrtime.bigint() - start) / 1e6;
  }
  // Pure SSR with no high-res clock: a frozen 0 source. Offsets stay valid
  // (non-negative, non-decreasing) but flat; the browser clock takes over on
  // hydration. This path must never throw at import/construct time.
  return () => 0;
}

interface OpenPhase {
  phase_id: PhaseTiming["phase_id"];
  entered_ms: number;
}

export class SessionClock {
  private readonly startedAtMs: number;
  private readonly now: NowSource;

  /** Wall-clock anchor for the current perf timeline, as ms offset from session
   *  start. 0 for the original page-load; positive after rehydrate(). */
  private epochOffsetMs: number;
  /** perf.now() value captured when the current epoch was anchored. */
  private perfAtEpoch: number;

  private phaseTimings: PhaseTiming[] = [];
  private openPhases = new Map<PhaseTiming["phase_id"], OpenPhase>();

  /** Highest offset returned so far — guards strict monotonicity. */
  private lastOffset = 0;

  constructor(startedAtUtc: string, nowSource?: NowSource) {
    this.startedAtMs = Date.parse(startedAtUtc);
    this.now = nowSource ?? defaultNow();
    this.epochOffsetMs = 0;
    this.perfAtEpoch = this.now();
  }

  /**
   * Monotonic offset (ms) from session start. Clamped to be non-negative and
   * non-decreasing so that even a misbehaving clock source can never emit a
   * value below schema minimum:0 or below a previously reported offset.
   */
  nowMs(): number {
    const perfDelta = this.now() - this.perfAtEpoch;
    let offset = this.epochOffsetMs + perfDelta;
    if (offset < this.lastOffset) offset = this.lastOffset;
    this.lastOffset = offset;
    return Math.round(offset);
  }

  /**
   * Re-anchor the clock after a reload. load_epoch_utc is the wall-clock instant
   * of the new page load; its distance from started_at_utc becomes the new epoch
   * offset, and the current perf reading becomes the new perf anchor — so the
   * very next nowMs() ≈ (load_epoch_utc − started_at_utc).
   */
  rehydrate(loadEpochUtc: string): void {
    const elapsedSinceStart = Date.parse(loadEpochUtc) - this.startedAtMs;
    // Never go backwards: a reload can only push the floor forward.
    this.epochOffsetMs = Math.max(elapsedSinceStart, this.lastOffset);
    this.perfAtEpoch = this.now();
    this.lastOffset = Math.max(this.lastOffset, this.epochOffsetMs);
  }

  /**
   * Record entering / exiting a phase. An 'exit' closes the most recent matching
   * 'enter' and appends a {phase_id, entered_ms, exited_ms} row to phase_timings.
   */
  markPhase(phaseId: PhaseTiming["phase_id"], kind: "enter" | "exit"): void {
    const t = this.nowMs();
    if (kind === "enter") {
      this.openPhases.set(phaseId, { phase_id: phaseId, entered_ms: t });
      return;
    }
    const open = this.openPhases.get(phaseId);
    if (!open) {
      // Exit without a matching enter: record a zero-length marker at t so the
      // event is not silently lost.
      this.phaseTimings.push({ phase_id: phaseId, entered_ms: t, exited_ms: t });
      return;
    }
    this.openPhases.delete(phaseId);
    this.phaseTimings.push({
      phase_id: phaseId,
      entered_ms: open.entered_ms,
      exited_ms: t,
    });
  }

  /** Accessor — a defensive copy of the accumulated, closed phase timings. */
  getPhaseTimings(): PhaseTiming[] {
    return this.phaseTimings.map((p) => ({ ...p }));
  }
}
