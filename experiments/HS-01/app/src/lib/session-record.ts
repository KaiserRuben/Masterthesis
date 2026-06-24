/**
 * session-record.ts — pure assembly of the SessionRecord output.
 *
 * No browser APIs, no I/O — these are pure functions over the typed record so
 * they are trivially testable and reproducible. The server (store.ts) persists
 * what these produce; exclusion decisions are NOT made here (per schema,
 * exclusion is an analysis-stage decision — we only record raw outcomes in
 * quality_summary).
 */

import type {
  SessionRecord,
  Trial,
  CheckRule,
  IntegrityEvent,
} from "./types";
import type { CreateResult, ItemPayload } from "./store";

type Demographics = NonNullable<SessionRecord["demographics"]>;
type QualitySummary = NonNullable<SessionRecord["quality_summary"]>;

/**
 * initRecord — seed a fresh record from the server CreateResult. status starts
 * as "abandoned" and is promoted to "completed" by the server on submit
 * (submitSession). environment is captured client-side; started_at_utc is the
 * monotonic-clock anchor (session start instant).
 */
export function initRecord(
  create: CreateResult,
  env: SessionRecord["environment"],
  startedAtUtc: string
): SessionRecord {
  return {
    schema_version: "1.0.0",
    study_id: create.study_id,
    config_version: create.config_version,
    config_sha256: create.config_sha256,
    session_id: create.session_id,
    form_id: create.form_id,
    rng_seed: create.rng_seed,
    status: "abandoned",
    participant: {
      participant_code: create.participant_code,
      recruitment_channel: null,
      consent: {
        given: true,
        consent_version: create.consent_version,
        at_utc: startedAtUtc,
      },
    },
    environment: env,
    timing: { started_at_utc: startedAtUtc },
    phase_timings: [],
    trials: [],
  };
}

/** appendTrial — pure: returns a new record with the trial appended. */
export function appendTrial(record: SessionRecord, trial: Trial): SessionRecord {
  return { ...record, trials: [...record.trials, trial] };
}

/** setDemographics — pure: returns a new record with demographics attached. */
export function setDemographics(
  record: SessionRecord,
  demographics: Demographics
): SessionRecord {
  return { ...record, demographics };
}

// ─── attention-check evaluation ───────────────────────────────────────────────

/**
 * Build item_id → ItemPayload lookup from the per-phase create payload. This is
 * where is_attention_check and check_rule live (denormalized from the pool).
 */
function indexItems(create: CreateResult): Map<string, ItemPayload> {
  const map = new Map<string, ItemPayload>();
  for (const phase of Object.values(create.items)) {
    for (const item of phase) map.set(item.item_id, item);
  }
  return map;
}

/**
 * Evaluate one attention trial against its check_rule. Returns true on PASS,
 * false on FAIL. A missing/unknown rule or a missing response is treated as a
 * FAIL — an attention item with no usable answer did not pass.
 */
function attentionPassed(trial: Trial, rule: CheckRule | null): boolean {
  if (!rule) return false;
  const resp = trial.response;

  switch (rule.metric) {
    case "scale_leq": {
      if (typeof resp.scale_value !== "number") return false;
      return resp.scale_value <= Number(rule.value);
    }
    case "scale_geq": {
      if (typeof resp.scale_value !== "number") return false;
      return resp.scale_value >= Number(rule.value);
    }
    case "choice_equals": {
      // rule.value is a WORD ("green iguana"). The rater's choice is a SEMANTIC
      // slot (ANCHOR_WORD / TARGET_WORD / …). Map the word → the slot it was
      // shown in via this trial's option_labels, then PASS iff the rater picked
      // that slot.
      const labels = trial.presented.option_labels;
      const choice = resp.choice;
      if (!labels || !choice) return false;
      const word = String(rule.value);
      let expectedSlot: "ANCHOR_WORD" | "TARGET_WORD" | null = null;
      if (labels.ANCHOR_WORD === word) expectedSlot = "ANCHOR_WORD";
      else if (labels.TARGET_WORD === word) expectedSlot = "TARGET_WORD";
      if (!expectedSlot) return false; // word not among the shown options
      return choice === expectedSlot;
    }
    default:
      return false;
  }
}

/** Count integrity events that represent focus loss: blur + visibility_hidden. */
function focusLossCount(events: IntegrityEvent[] | undefined): number {
  if (!events) return 0;
  return events.filter(
    (e) => e.type === "blur" || e.type === "visibility_hidden"
  ).length;
}

/**
 * finalizeQualitySummary — compute {attention_total, attention_failed,
 * focus_loss_count} from the recorded trials and integrity events. Pure: returns
 * a new record with quality_summary attached; the input is left untouched.
 *
 *   attention_total  = trials whose item is_attention_check (resolved from the
 *                      create payload, with the trial's own flag as a fallback).
 *   attention_failed = those whose response violates the item's check_rule.
 *   focus_loss_count = blur + visibility_hidden integrity events.
 *
 * NO exclusion is applied — status is not changed here.
 */
export function finalizeQualitySummary(
  record: SessionRecord,
  create: CreateResult
): SessionRecord {
  const items = indexItems(create);

  let attentionTotal = 0;
  let attentionFailed = 0;

  for (const trial of record.trials) {
    const item = items.get(trial.item_id);
    const isAttention = item?.is_attention_check ?? trial.is_attention_check;
    if (!isAttention) continue;

    attentionTotal++;
    const rule = item?.check_rule ?? null;
    if (!attentionPassed(trial, rule)) attentionFailed++;
  }

  const quality_summary: QualitySummary = {
    attention_total: attentionTotal,
    attention_failed: attentionFailed,
    focus_loss_count: focusLossCount(record.integrity_events),
  };

  return { ...record, quality_summary };
}
