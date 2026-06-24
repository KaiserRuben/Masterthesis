/**
 * session-record tests — pure record assembly (Node environment).
 *
 * Focus: finalizeQualitySummary semantics.
 *  - attention_total  = trials whose item is_attention_check.
 *  - attention_failed = attention items whose recorded response violates the
 *    item's check_rule:
 *      scale_leq N  -> PASS iff scale_value <= N
 *      scale_geq N  -> PASS iff scale_value >= N
 *      choice_equals WORD -> PASS iff the rater's semantic `choice` maps (via the
 *        trial's option_labels) to the slot whose word == WORD.
 *  - focus_loss_count = number of blur + visibility_hidden integrity events.
 *  - exclusion is NOT applied here.
 */

import { describe, it, expect } from "vitest";
import {
  initRecord,
  appendTrial,
  setDemographics,
  finalizeQualitySummary,
} from "../src/lib/session-record";
import type { CreateResult } from "../src/lib/store";
import type { SessionRecord, Trial, CheckRule } from "../src/lib/types";

// ─── fixtures ──────────────────────────────────────────────────────────────

const ENV: SessionRecord["environment"] = {
  user_agent: "vitest",
  viewport: { w: 1280, h: 800 },
  device_pixel_ratio: 2,
};

/**
 * Minimal CreateResult. finalizeQualitySummary needs item_id ->
 * { is_attention_check, check_rule } resolvable from create.items.
 */
function makeCreate(): CreateResult {
  return {
    session_id: "11111111-1111-4111-8111-111111111111",
    participant_code: "P001",
    form_id: "A",
    rng_seed: "seed-fixed",
    study_id: "HS-01",
    config_version: "1.0.0",
    config_sha256: "a".repeat(64),
    consent_version: "v1",
    items: {
      text: [
        {
          item_id: "txt-attn-nonsense-01",
          source_id: "src-attn-nonsense-01",
          kind: "text",
          is_attention_check: true,
          check_rule: { metric: "scale_leq", value: 2 } as CheckRule,
          prompt: "asdf qwpoeiruz mmm nonsense",
          image: null,
          option_labels: null,
        },
        {
          item_id: "txt-clean-01",
          source_id: "src-clean-01",
          kind: "text",
          is_attention_check: false,
          check_rule: null,
          prompt: "What bird is this?",
          image: null,
          option_labels: null,
        },
      ],
      image: [],
      pair: [
        {
          item_id: "pair-attn-obvious-01",
          source_id: "src-attn-obvious-01",
          kind: "pair",
          is_attention_check: true,
          check_rule: { metric: "choice_equals", value: "green iguana" } as CheckRule,
          prompt: null,
          image: { uri_name: "iguana.png", natural_w: 256, natural_h: 256 },
          option_labels: { ANCHOR_WORD: "green iguana", TARGET_WORD: "boa constrictor" },
        },
      ],
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

function textTrial(
  item_id: string,
  is_attention_check: boolean,
  scale_value: number,
  trial_index: number
): Trial {
  return {
    trial_index,
    phase_id: "text",
    position_in_phase: trial_index,
    item_id,
    source_id: `src-${item_id}`,
    item_kind: "text",
    is_attention_check,
    presented: { scale_id: "text-comprehensibility-v1" },
    response: { n_changes: 0, scale_value },
    timing: { onset_ms: 10, submitted_ms: 1000 + trial_index },
  };
}

function pairTrial(
  item_id: string,
  choice: Trial["response"]["choice"],
  option_labels: { ANCHOR_WORD: string; TARGET_WORD: string },
  trial_index: number
): Trial {
  return {
    trial_index,
    phase_id: "pair",
    position_in_phase: trial_index,
    item_id,
    source_id: `src-${item_id}`,
    item_kind: "pair",
    is_attention_check: true,
    presented: {
      option_display_order: ["TARGET_WORD", "ANCHOR_WORD", "OTHER_CLASS", "NOTHING_RECOGNIZABLE", "CANT_TELL"],
      option_labels,
      rendered_image: { css_w: 256, css_h: 256, natural_w: 256, natural_h: 256 },
    },
    response: { n_changes: 1, choice },
    timing: { onset_ms: 10, image_loaded_ms: 5, submitted_ms: 2000 + trial_index },
  };
}

// ─── initRecord / appendTrial / setDemographics ──────────────────────────────

describe("initRecord / appendTrial / setDemographics", () => {
  it("initRecord copies identity from CreateResult and seeds an abandoned record", () => {
    const create = makeCreate();
    const rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");

    expect(rec.schema_version).toBe("1.0.0");
    expect(rec.study_id).toBe(create.study_id); // sourced from config, not hardcoded
    expect(rec.session_id).toBe(create.session_id);
    expect(rec.form_id).toBe(create.form_id);
    expect(rec.rng_seed).toBe(create.rng_seed);
    expect(rec.config_sha256).toBe(create.config_sha256);
    expect(rec.status).toBe("abandoned");
    expect(rec.participant.participant_code).toBe("P001");
    expect(rec.environment).toEqual(ENV);
    expect(rec.timing.started_at_utc).toBe("2026-06-24T10:00:00.000Z");
    expect(rec.trials).toEqual([]);
    expect(rec.phase_timings).toEqual([]);
  });

  it("initRecord sources study_id from the create payload (not a hardcoded constant)", () => {
    const create = { ...makeCreate(), study_id: "HS-XX-other" };
    const rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    expect(rec.study_id).toBe("HS-XX-other");
  });

  it("appendTrial returns a new record with the trial added (pure)", () => {
    const create = makeCreate();
    const rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    const t = textTrial("txt-clean-01", false, 4, 0);
    const rec2 = appendTrial(rec, t);

    expect(rec.trials).toHaveLength(0); // original untouched
    expect(rec2.trials).toHaveLength(1);
    expect(rec2.trials[0]).toEqual(t);
  });

  it("setDemographics attaches demographics (pure)", () => {
    const create = makeCreate();
    const rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    const demo = {
      age_band: "25_34" as const,
      ml_familiarity: "some_exposure" as const,
      english_proficiency: "C1" as const,
      comment: null,
    };
    const rec2 = setDemographics(rec, demo);
    expect(rec.demographics).toBeUndefined();
    expect(rec2.demographics).toEqual(demo);
  });
});

// ─── finalizeQualitySummary ──────────────────────────────────────────────────

describe("finalizeQualitySummary", () => {
  it("text scale_leq 2: nonsense rated 5 FAILS, rated 1 PASSES", () => {
    const create = makeCreate();

    // FAIL case: rater agreed (scale 5) the nonsense was comprehensible.
    let rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    rec = appendTrial(rec, textTrial("txt-attn-nonsense-01", true, 5, 0));
    rec = appendTrial(rec, textTrial("txt-clean-01", false, 4, 1));
    let out = finalizeQualitySummary(rec, create);
    expect(out.quality_summary!.attention_total).toBe(1);
    expect(out.quality_summary!.attention_failed).toBe(1);

    // PASS case: rater disagreed (scale 1).
    let rec2 = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    rec2 = appendTrial(rec2, textTrial("txt-attn-nonsense-01", true, 1, 0));
    let out2 = finalizeQualitySummary(rec2, create);
    expect(out2.quality_summary!.attention_total).toBe(1);
    expect(out2.quality_summary!.attention_failed).toBe(0);
  });

  it("pair choice_equals 'green iguana': picking the iguana slot PASSES, else FAILS", () => {
    const create = makeCreate();
    const labels = { ANCHOR_WORD: "green iguana", TARGET_WORD: "boa constrictor" };

    // PASS: rater picked ANCHOR_WORD, whose word == "green iguana".
    let recPass = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    recPass = appendTrial(recPass, pairTrial("pair-attn-obvious-01", "ANCHOR_WORD", labels, 0));
    const passOut = finalizeQualitySummary(recPass, create);
    expect(passOut.quality_summary!.attention_total).toBe(1);
    expect(passOut.quality_summary!.attention_failed).toBe(0);

    // FAIL: rater picked TARGET_WORD (boa constrictor).
    let recFail = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    recFail = appendTrial(recFail, pairTrial("pair-attn-obvious-01", "TARGET_WORD", labels, 0));
    const failOut = finalizeQualitySummary(recFail, create);
    expect(failOut.quality_summary!.attention_failed).toBe(1);
  });

  it("attention_total counts BOTH attention checks across phases", () => {
    const create = makeCreate();
    const labels = { ANCHOR_WORD: "green iguana", TARGET_WORD: "boa constrictor" };

    let rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    rec = appendTrial(rec, textTrial("txt-attn-nonsense-01", true, 1, 0)); // PASS
    rec = appendTrial(rec, textTrial("txt-clean-01", false, 3, 1));
    rec = appendTrial(rec, pairTrial("pair-attn-obvious-01", "TARGET_WORD", labels, 2)); // FAIL

    const out = finalizeQualitySummary(rec, create);
    expect(out.quality_summary!.attention_total).toBe(2);
    expect(out.quality_summary!.attention_failed).toBe(1);
  });

  it("focus_loss_count counts blur + visibility_hidden (not focus/resize/visible)", () => {
    const create = makeCreate();
    let rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    rec = appendTrial(rec, textTrial("txt-clean-01", false, 4, 0));
    rec = {
      ...rec,
      integrity_events: [
        { at_ms: 100, type: "blur", detail: null },
        { at_ms: 150, type: "focus", detail: null },
        { at_ms: 200, type: "visibility_hidden", detail: null },
        { at_ms: 250, type: "visibility_visible", detail: null },
        { at_ms: 300, type: "resize", detail: null },
        { at_ms: 350, type: "blur", detail: null },
      ],
    };
    const out = finalizeQualitySummary(rec, create);
    expect(out.quality_summary!.focus_loss_count).toBe(3); // 2 blur + 1 visibility_hidden
  });

  it("does not apply exclusion (status unchanged) and is pure", () => {
    const create = makeCreate();
    let rec = initRecord(create, ENV, "2026-06-24T10:00:00.000Z");
    rec = appendTrial(rec, textTrial("txt-attn-nonsense-01", true, 5, 0)); // would exclude

    const out = finalizeQualitySummary(rec, create);
    // status is NOT flipped to anything exclusion-related here
    expect(out.status).toBe(rec.status);
    // original record left without a quality_summary (pure)
    expect(rec.quality_summary).toBeUndefined();
    expect(out.quality_summary).toBeDefined();
  });
});
