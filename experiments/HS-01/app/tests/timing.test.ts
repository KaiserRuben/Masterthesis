/**
 * timing tests — monotonic SessionClock (Node environment; injected now-source).
 *
 * Contract:
 *  - nowMs() returns a monotonic offset from started_at_utc.
 *  - rehydrate(load_epoch_utc) chains epochs across reloads so offsets keep
 *    increasing: offset = (load_epoch_utc − started_at_utc) + perf_delta.
 *  - markPhase(phase_id, 'enter'|'exit') accumulates phase_timings, exposed
 *    via an accessor.
 */

import { describe, it, expect } from "vitest";
import { SessionClock } from "../src/lib/timing";

/** A controllable monotonic clock source standing in for performance.now(). */
function fakeNow(): { now: () => number; advance: (ms: number) => void } {
  let t = 0;
  return {
    now: () => t,
    advance: (ms: number) => {
      t += ms;
    },
  };
}

describe("SessionClock", () => {
  it("nowMs() is monotonic and starts at ~0", () => {
    const src = fakeNow();
    const started = "2026-06-24T10:00:00.000Z";
    const clock = new SessionClock(started, src.now);

    const t0 = clock.nowMs();
    expect(t0).toBe(0);

    src.advance(120);
    const t1 = clock.nowMs();
    src.advance(80);
    const t2 = clock.nowMs();

    expect(t1).toBe(120);
    expect(t2).toBe(200);
    expect(t2).toBeGreaterThan(t1);
    expect(t1).toBeGreaterThan(t0);
  });

  it("rehydrate() keeps offsets increasing across a simulated reload", () => {
    const src = fakeNow();
    const started = "2026-06-24T10:00:00.000Z";
    const clock = new SessionClock(started, src.now);

    src.advance(5000); // 5s of activity pre-reload
    const beforeReload = clock.nowMs();
    expect(beforeReload).toBe(5000);

    // Simulate a reload: a NEW perf timeline (resets to 0) but 3s of wall-clock
    // elapsed since session start. New page-load epoch = started + 8s.
    const src2 = fakeNow();
    const reloadEpoch = "2026-06-24T10:00:08.000Z"; // 8s after start
    const clock2 = new SessionClock(started, src2.now);
    clock2.rehydrate(reloadEpoch);

    const afterReload = clock2.nowMs();
    // offset = (reloadEpoch − started) + perf_delta(0) = 8000
    expect(afterReload).toBe(8000);
    expect(afterReload).toBeGreaterThan(beforeReload);

    src2.advance(250);
    const later = clock2.nowMs();
    expect(later).toBe(8250);
    expect(later).toBeGreaterThan(afterReload);
  });

  it("markPhase() records enter/exit into phase_timings", () => {
    const src = fakeNow();
    const clock = new SessionClock("2026-06-24T10:00:00.000Z", src.now);

    src.advance(100);
    clock.markPhase("text", "enter");
    src.advance(4000);
    clock.markPhase("text", "exit");

    src.advance(50);
    clock.markPhase("image", "enter");
    src.advance(2000);
    clock.markPhase("image", "exit");

    const phases = clock.getPhaseTimings();
    expect(phases).toEqual([
      { phase_id: "text", entered_ms: 100, exited_ms: 4100 },
      { phase_id: "image", entered_ms: 4150, exited_ms: 6150 },
    ]);
  });

  it("falls back to a default now-source without crashing when none injected", () => {
    // performance.now exists in Node ≥16, but the ctor must not throw under SSR.
    const clock = new SessionClock("2026-06-24T10:00:00.000Z");
    const a = clock.nowMs();
    const b = clock.nowMs();
    expect(typeof a).toBe("number");
    expect(b).toBeGreaterThanOrEqual(a);
  });
});
