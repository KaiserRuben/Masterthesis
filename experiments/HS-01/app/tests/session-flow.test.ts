/**
 * session-flow tests — the pure logic behind useSession's resume + phase-timing
 * merge. These guard the measurement-critical resume behaviour without a DOM:
 *
 *  - resumePhase lands on the FIRST unanswered trial across phases, in order.
 *  - countByPhase tallies recorded trials per judgment phase.
 *  - closePhase MERGES a freshly-closed phase row into prior phase_timings
 *    (a reload must not drop timings of earlier phases) and dedupes by
 *    entered_ms.
 */

import { describe, it, expect, vi } from "vitest";

import {
  countByPhase,
  resumePhase,
  closePhase,
  durableSubmit,
  getMinPx,
} from "../src/state/useSession";
import { SessionClock } from "../src/lib/timing";
import type { CreateResult } from "../src/lib/store";
import type { SessionRecord, Trial } from "../src/lib/types";

function item(id: string, kind: "text" | "image" | "pair") {
  return {
    item_id: id,
    source_id: `src-${id}`,
    kind,
    is_attention_check: false,
    check_rule: null,
    prompt: kind === "image" ? null : "p",
    image: kind === "text" ? null : { uri_name: `src-${id}.png`, natural_w: 300, natural_h: 300 },
    option_labels: kind === "pair" ? { ANCHOR_WORD: "a", TARGET_WORD: "b" } : null,
  };
}

function makeCreate(): CreateResult {
  return {
    session_id: "s",
    participant_code: "P001",
    form_id: "A",
    rng_seed: "seed",
    config_version: "1.0.0",
    config_sha256: "a".repeat(64),
    consent_version: "v1",
    items: {
      text: [item("t1", "text"), item("t2", "text")],
      image: [item("i1", "image")],
      pair: [item("p1", "pair"), item("p2", "pair")],
    },
    scales: [],
    pair_response: {
      semantic_options: [
        "ANCHOR_WORD",
        "TARGET_WORD",
        "OTHER_CLASS",
        "NOTHING_RECOGNIZABLE",
        "CANT_TELL",
      ],
      display_labels: {
        OTHER_CLASS: "Something else",
        NOTHING_RECOGNIZABLE: "Nothing recognizable",
        CANT_TELL: "I can't tell",
      },
      ab_order: "randomized_per_trial",
      other_class_free_text: true,
    },
    demographics_fields: [],
    phases: [],
  };
}

function trial(phase: "text" | "image" | "pair", idx: number): Trial {
  return {
    trial_index: idx,
    phase_id: phase,
    position_in_phase: 0,
    item_id: `${phase}-${idx}`,
    source_id: "s",
    item_kind: phase,
    is_attention_check: false,
    presented: {},
    response: { n_changes: 0 },
    timing: { onset_ms: 0, submitted_ms: 1 },
  };
}

function baseRecord(trials: Trial[]): SessionRecord {
  return {
    schema_version: "1.0.0",
    study_id: "HS-01",
    config_version: "1.0.0",
    config_sha256: "a".repeat(64),
    session_id: "s",
    form_id: "A",
    rng_seed: "seed",
    status: "abandoned",
    participant: {
      participant_code: "P001",
      recruitment_channel: null,
      consent: { given: true, consent_version: "v1", at_utc: "2026-01-01T00:00:00.000Z" },
    },
    environment: { user_agent: "x", viewport: { w: 1280, h: 800 }, device_pixel_ratio: 1 },
    timing: { started_at_utc: "2026-01-01T00:00:00.000Z" },
    phase_timings: [],
    trials,
  };
}

describe("countByPhase", () => {
  it("tallies recorded trials per judgment phase", () => {
    const rec = baseRecord([trial("text", 0), trial("text", 1), trial("image", 2)]);
    expect(countByPhase(rec)).toEqual({ text: 2, image: 1, pair: 0 });
  });
});

describe("resumePhase", () => {
  const create = makeCreate();

  it("resumes at the first text item when nothing is done", () => {
    expect(resumePhase(create, { text: 0, image: 0, pair: 0 })).toEqual({
      phase: "text",
      atPosition: 0,
    });
  });

  it("skips a finished phase and resumes mid next phase", () => {
    // text complete (2/2), image at 0/1
    expect(resumePhase(create, { text: 2, image: 0, pair: 0 })).toEqual({
      phase: "image",
      atPosition: 0,
    });
  });

  it("resumes mid-pair at the unanswered position", () => {
    expect(resumePhase(create, { text: 2, image: 1, pair: 1 })).toEqual({
      phase: "pair",
      atPosition: 1,
    });
  });

  it("routes to demographics when all judgment phases are complete", () => {
    expect(resumePhase(create, { text: 2, image: 1, pair: 2 })).toBe("demographics");
  });
});

describe("closePhase", () => {
  it("merges the freshly-closed phase row into existing phase_timings", () => {
    const clock = new SessionClock("2026-01-01T00:00:00.000Z");
    clock.markPhase("text", "enter");
    // Pretend a prior load already recorded the text phase.
    const rec = {
      ...baseRecord([]),
      phase_timings: [{ phase_id: "text" as const, entered_ms: 0, exited_ms: 100 }],
    };
    clock.markPhase("image", "enter");
    const out = closePhase(rec, clock, "image");
    const ids = out.phase_timings.map((p) => p.phase_id);
    expect(ids).toContain("text"); // prior row preserved
    expect(ids).toContain("image"); // newly closed row added
    expect(out.phase_timings.filter((p) => p.phase_id === "image")).toHaveLength(1);
  });

  it("does not duplicate a phase row already present at the same entered_ms", () => {
    const clock = new SessionClock("2026-01-01T00:00:00.000Z");
    clock.markPhase("text", "enter");
    clock.markPhase("text", "exit");
    const closed = clock.getPhaseTimings().filter((p) => p.phase_id === "text");
    const rec = { ...baseRecord([]), phase_timings: closed };
    // Re-closing with the same clock would re-emit nothing new (already exited),
    // so the record is unchanged in count.
    const out = closePhase(rec, clock, "text");
    expect(out.phase_timings.filter((p) => p.phase_id === "text").length).toBe(
      closed.length
    );
  });
});

// ─── durableSubmit — checkpoint-before-submit + failure safety ────────────────

/**
 * A complete final record as submitDemographics would build it: trials,
 * demographics, quality_summary and total timing all present. Status stays
 * "abandoned" until the submit confirms (the server promotes it to completed).
 */
function completeFinalRecord(): SessionRecord {
  return {
    ...baseRecord([trial("text", 0), trial("image", 1), trial("pair", 2)]),
    phase_timings: [
      { phase_id: "text" as const, entered_ms: 0, exited_ms: 100 },
      { phase_id: "demographics" as const, entered_ms: 100, exited_ms: 200 },
    ],
    demographics: {
      age_band: "25_34",
      ml_familiarity: "no_experience",
      english_proficiency: "B2",
      comment: null,
    },
    quality_summary: {
      attention_total: 0,
      attention_failed: 0,
      focus_loss_count: 0,
    },
    timing: {
      started_at_utc: "2026-01-01T00:00:00.000Z",
      completed_at_utc: "2026-01-01T00:10:00.000Z",
      total_duration_ms: 600000,
    },
  };
}

describe("durableSubmit", () => {
  it("checkpoints the COMPLETE record (carrying demographics) BEFORE attempting submit", async () => {
    const order: string[] = [];
    const checkpoint = vi.fn(async (rec: SessionRecord) => {
      order.push("checkpoint");
      // The checkpoint must carry the full record so the server holds it even
      // if the fallible submit never lands.
      expect(rec.demographics).toBeTruthy();
      expect(rec.quality_summary).toBeTruthy();
      expect(rec.timing.total_duration_ms).toBe(600000);
      expect(rec.phase_timings.some((p) => p.phase_id === "demographics")).toBe(true);
    });
    const submit = vi.fn(async () => {
      order.push("submit");
      return { ok: true };
    });

    const out = await durableSubmit(completeFinalRecord(), { checkpoint, submit });

    expect(checkpoint).toHaveBeenCalledTimes(1);
    expect(submit).toHaveBeenCalledTimes(1);
    expect(order).toEqual(["checkpoint", "submit"]); // checkpoint strictly first
    expect(out.submitted).toBe(true);
  });

  it("returns submitted:false when submit REJECTS (record already safe via checkpoint)", async () => {
    const checkpoint = vi.fn(async () => {});
    const submit = vi.fn(async () => {
      throw new Error("network down");
    });

    const out = await durableSubmit(completeFinalRecord(), { checkpoint, submit });

    expect(checkpoint).toHaveBeenCalledTimes(1); // record was preserved server-side
    expect(out.submitted).toBe(false); // caller must NOT mark completed
  });

  it("returns submitted:false when submit responds !ok", async () => {
    const checkpoint = vi.fn(async () => {});
    const submit = vi.fn(async () => ({ ok: false }));

    const out = await durableSubmit(completeFinalRecord(), { checkpoint, submit });

    expect(checkpoint).toHaveBeenCalledTimes(1);
    expect(out.submitted).toBe(false);
  });

  it("still attempts submit even if the checkpoint PUT fails (best-effort)", async () => {
    const checkpoint = vi.fn(async () => {
      throw new Error("checkpoint flaked");
    });
    const submit = vi.fn(async () => ({ ok: true }));

    const out = await durableSubmit(completeFinalRecord(), { checkpoint, submit });

    expect(submit).toHaveBeenCalledTimes(1);
    expect(out.submitted).toBe(true);
  });
});

// ─── getMinPx — viewport gate tracks the config quality block ─────────────────

describe("getMinPx", () => {
  it("uses the config quality.min_rendered_image_css_px when present", () => {
    const create = { ...makeCreate(), quality: { log_integrity_events: true, render_check: true, min_rendered_image_css_px: 384 } };
    expect(getMinPx(create)).toBe(384);
  });

  it("falls back to 256 when quality is absent", () => {
    expect(getMinPx(makeCreate())).toBe(256);
  });

  it("falls back to 256 when min_rendered_image_css_px is null", () => {
    const create = { ...makeCreate(), quality: { log_integrity_events: true, render_check: true, min_rendered_image_css_px: null } };
    expect(getMinPx(create)).toBe(256);
  });
});
