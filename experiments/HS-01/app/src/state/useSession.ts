"use client";

/**
 * useSession — the client state machine + localStorage resume for the rater flow.
 *
 * Responsibilities:
 *  - Hold the create-response (item payloads + presentation config) and the
 *    in-progress SessionRecord; persist both to localStorage after EVERY trial
 *    so a reload loses nothing.
 *  - Derive the DETERMINISTIC presentation order from rng_seed (per phase) and
 *    the per-pair option order (per item) — both reproducible on resume.
 *  - Walk consent → text → image → pair → demographics → done. Resume lands on
 *    the first UNANSWERED trial (count of recorded trials per phase).
 *  - Own the SessionClock (rehydrated from started_at_utc on load) and the
 *    integrity-event sink (attached during judgment phases by the page).
 *  - Checkpoint to the server at each phase exit; submit at the end.
 *
 * Everything that touches presentation order routes through src/lib/rng.ts —
 * the ONLY randomness source — so the realized order is reproducible in analysis.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { makeRng, shuffle, optionOrder } from "@/lib/rng";
import { SessionClock } from "@/lib/timing";
import {
  initRecord,
  appendTrial,
  setDemographics,
  finalizeQualitySummary,
} from "@/lib/session-record";
import type { CreateResult, ItemPayload } from "@/lib/store";
import type {
  SessionRecord,
  Trial,
  Scale,
  IntegrityEvent,
  SemanticChoice,
  PhaseTiming,
} from "@/lib/types";
import { preloadImage } from "@/lib/instrumentation";
import type { OnsetResult } from "@/lib/instrumentation";
import type { DemographicsValues } from "@/components/Demographics";

// ─── localStorage keys ──────────────────────────────────────────────────────

export const LS_COMPLETED = "hs01:completed";
export const LS_RECORD = "hs01:record";
export const LS_CREATE = "hs01:create";

// ─── public flow types ──────────────────────────────────────────────────────

export type JudgmentPhase = "text" | "image" | "pair";
export type FlowPhase =
  | "loading"
  | "instructions"
  | JudgmentPhase
  | "demographics"
  | "submitting"
  | "submit_failed"
  | "done"
  | "too_small"
  | "missing";

export interface CurrentTrial {
  phase: JudgmentPhase;
  position: number; // 1-based within phase
  total: number; // items in phase
  item: ItemPayload;
  /** For pair items: the realized per-trial option order. */
  optionDisplayOrder?: SemanticChoice[];
  /** The scale to show for text/image phases. */
  scale?: Scale;
}

const JUDGMENT_ORDER: JudgmentPhase[] = ["text", "image", "pair"];

// ─── deterministic ordering ─────────────────────────────────────────────────

/** Stable per-phase order: a fresh rng namespaced by phase ⇒ reproducible. */
function phaseOrder(items: ItemPayload[], seed: string | number, phase: JudgmentPhase): ItemPayload[] {
  const rng = makeRng(`${seed}:order:${phase}`);
  return shuffle(items, rng);
}

/** Stable per-pair option order: namespaced by item_id ⇒ reproducible on resume. */
function pairOptionOrder(seed: string | number, itemId: string): SemanticChoice[] {
  return optionOrder(makeRng(`${seed}:pairopt:${itemId}`));
}

function scaleFor(create: CreateResult, appliesTo: "text" | "image"): Scale | undefined {
  return create.scales.find((s) => s.applies_to === appliesTo);
}

// ─── localStorage helpers (SSR-safe) ────────────────────────────────────────

function lsGet(key: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function lsSet(key: string, value: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* quota / disabled — non-fatal */
  }
}

// ─── trial assembly ──────────────────────────────────────────────────────────

interface PendingAnswer {
  scale_value?: number;
  choice?: SemanticChoice;
  other_class_text?: string | null;
  n_changes: number;
  onset_ms: number;
  image_loaded_ms: number | null;
  rendered_image: Trial["presented"]["rendered_image"];
  first_interaction_ms: number | null;
  response_selected_ms: number | null;
}

export interface UseSession {
  phase: FlowPhase;
  /** Instructions are shown before each judgment phase. */
  instructionsFor: JudgmentPhase | null;
  current: CurrentTrial | null;
  create: CreateResult | null;
  clock: SessionClock | null;
  /** Count of judgment trials done across all phases (for a global progress hint). */
  totalTrials: number;
  completedTrials: number;
  /** Advance past the current phase's instructions screen. */
  beginPhase: () => void;
  /** Submit the current trial's answer and advance. */
  submitTrial: (answer: TrialAnswer) => void;
  /** Submit demographics and finalize/POST the session. */
  submitDemographics: (values: DemographicsValues) => Promise<void>;
  /**
   * Re-attempt the final submit after a failure. The complete record is already
   * safe on the server (checkpointed before the first attempt); this confirms
   * the `completed` status and routes to /done on success.
   */
  retrySubmit: () => Promise<void>;
  /** Record an integrity event (wired to attachIntegrityListeners' sink). */
  pushIntegrity: (ev: IntegrityEvent) => void;
  /**
   * Schema validation errors reported by the server on a failed submit
   * (HTTP 200 with body.ok === false). Surfaced for the submit-failed retry
   * panel; null when the last submit had no schema errors.
   */
  submitValidationErrors: object[] | null;
}

export interface TrialAnswer {
  scale_value?: number;
  choice?: SemanticChoice;
  other_class_text?: string | null;
  n_changes: number;
  onset: OnsetResult;
  /** ms offset of the first interaction with the widget (optional). */
  first_interaction_ms?: number | null;
  /** ms offset when the response was selected (optional). */
  response_selected_ms?: number | null;
}

export function useSession(): UseSession {
  const [create, setCreate] = useState<CreateResult | null>(null);
  const [record, setRecord] = useState<SessionRecord | null>(null);
  const [phase, setPhase] = useState<FlowPhase>("loading");
  // When entering a judgment phase we first show instructions; this holds which.
  const [instructionsFor, setInstructionsFor] = useState<JudgmentPhase | null>(null);
  // Schema validation errors from a failed submit (HTTP 200 + body.ok === false).
  const [submitValidationErrors, setSubmitValidationErrors] = useState<
    object[] | null
  >(null);

  const clockRef = useRef<SessionClock | null>(null);
  // Mirror the latest record in a ref so async callbacks don't capture stale state.
  const recordRef = useRef<SessionRecord | null>(null);
  recordRef.current = record;

  // ── load + resume on mount ──────────────────────────────────────────────
  useEffect(() => {
    const rawCreate = lsGet(LS_CREATE);
    const rawRecord = lsGet(LS_RECORD);
    if (!rawCreate || !rawRecord) {
      setPhase("missing");
      return;
    }
    let parsedCreate: CreateResult;
    let parsedRecord: SessionRecord;
    try {
      parsedCreate = JSON.parse(rawCreate) as CreateResult;
      parsedRecord = JSON.parse(rawRecord) as SessionRecord;
    } catch {
      setPhase("missing");
      return;
    }

    const clock = new SessionClock(parsedRecord.timing.started_at_utc);
    clock.rehydrate(new Date().toISOString());
    clockRef.current = clock;

    setCreate(parsedCreate);
    setRecord(parsedRecord);

    // Determine where to resume: which judgment phase, and how many of its
    // trials are already recorded.
    const phaseDone = countByPhase(parsedRecord);
    const resume = resumePhase(parsedCreate, phaseDone);
    if (resume === "demographics") {
      setPhase("demographics");
    } else if (resume.atPosition === 0) {
      // Phase not started → show its instructions first (the too-small gate
      // fires when the rater clicks through into the image/pair phase).
      setInstructionsFor(resume.phase);
      setPhase("instructions");
    } else if (
      (resume.phase === "image" || resume.phase === "pair") &&
      !viewportFits(parsedCreate)
    ) {
      // Resuming mid image/pair phase on a now-too-small viewport: refuse so we
      // never present an image at non-native size.
      setPhase("too_small");
    } else {
      setInstructionsFor(null);
      setPhase(resume.phase);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── persistence ──────────────────────────────────────────────────────────
  const persist = useCallback((rec: SessionRecord) => {
    recordRef.current = rec;
    setRecord(rec);
    lsSet(LS_RECORD, JSON.stringify(rec));
  }, []);

  // ── derived: per-phase ordered item lists ─────────────────────────────────
  const orders = useMemo(() => {
    if (!create) return null;
    const seed = create.rng_seed;
    return {
      text: phaseOrder(create.items.text ?? [], seed, "text"),
      image: phaseOrder(create.items.image ?? [], seed, "image"),
      pair: phaseOrder(create.items.pair ?? [], seed, "pair"),
    } as Record<JudgmentPhase, ItemPayload[]>;
  }, [create]);

  // ── current position within the active judgment phase ──────────────────────
  const completedByPhase = useMemo(
    () => (record ? countByPhase(record) : { text: 0, image: 0, pair: 0 }),
    [record]
  );

  const current: CurrentTrial | null = useMemo(() => {
    if (!create || !orders) return null;
    if (phase !== "text" && phase !== "image" && phase !== "pair") return null;
    const list = orders[phase];
    const pos = completedByPhase[phase]; // 0-based index of next unanswered
    if (pos >= list.length) return null;
    const item = list[pos];
    const trial: CurrentTrial = {
      phase,
      position: pos + 1,
      total: list.length,
      item,
    };
    if (phase === "pair") {
      trial.optionDisplayOrder = pairOptionOrder(create.rng_seed, item.item_id);
    } else {
      trial.scale = scaleFor(create, phase);
    }
    return trial;
  }, [create, orders, phase, completedByPhase]);

  // ── preload the NEXT trial's image during the current trial ────────────────
  useEffect(() => {
    if (!orders) return;
    const next = nextImageUrl(orders, phase, completedByPhase);
    // Fire-and-forget; preloadImage is SSR-safe and never blocks/rejects.
    if (next) void preloadImage(next);
  }, [orders, phase, completedByPhase]);

  // ── flow actions ───────────────────────────────────────────────────────────

  const enterPhase = useCallback(
    (p: JudgmentPhase) => {
      // Image / pair phases need a viewport wide enough for native px. If the
      // viewport can't fit min_rendered_image_css_px, refuse at entry.
      if ((p === "image" || p === "pair") && create && !viewportFits(create)) {
        setPhase("too_small");
        return;
      }
      // The phase-enter clock mark is owned by the [phase] effect below, so it
      // is stamped exactly once whether we arrive via instructions or resume.
      setInstructionsFor(null);
      setPhase(p);
    },
    [create]
  );

  const beginPhase = useCallback(() => {
    if (instructionsFor) enterPhase(instructionsFor);
  }, [instructionsFor, enterPhase]);

  // Single owner of phase-enter marking: fires once per landing in a judgment
  // phase, whether reached by clicking through instructions or by resume. Guard
  // against StrictMode double-invoke / re-renders with a per-phase latch.
  const enteredPhases = useRef<Set<JudgmentPhase>>(new Set());
  useEffect(() => {
    if (phase === "text" || phase === "image" || phase === "pair") {
      if (!enteredPhases.current.has(phase)) {
        enteredPhases.current.add(phase);
        clockRef.current?.markPhase(phase, "enter");
      }
    }
  }, [phase]);

  const checkpoint = useCallback(async (rec: SessionRecord) => {
    try {
      await fetch(`/api/sessions/${rec.session_id}/checkpoint`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rec),
      });
    } catch {
      /* offline / transient — localStorage already holds the record */
    }
  }, []);

  const advancePhase = useCallback(
    (fromPhase: JudgmentPhase, rec: SessionRecord) => {
      const clock = clockRef.current;
      const withTimings = closePhase(rec, clock, fromPhase);
      persist(withTimings);
      void checkpoint(withTimings); // checkpoint at phase EXIT

      const idx = JUDGMENT_ORDER.indexOf(fromPhase);
      const next = JUDGMENT_ORDER[idx + 1];
      if (next && (orders?.[next]?.length ?? 0) > 0) {
        setInstructionsFor(next);
        setPhase("instructions");
      } else if (next) {
        // Empty next phase — skip straight to it / past it recursively.
        advancePhase(next, withTimings);
      } else {
        setPhase("demographics");
      }
    },
    [persist, checkpoint, orders]
  );

  const submitTrial = useCallback(
    (answer: TrialAnswer) => {
      const rec = recordRef.current;
      const clock = clockRef.current;
      if (!rec || !create || !current || !orders) return;
      if (current.phase !== phase) return;

      const item = current.item;
      const trial: Trial = {
        trial_index: rec.trials.length,
        phase_id: current.phase,
        position_in_phase: current.position - 1,
        item_id: item.item_id,
        source_id: item.source_id,
        item_kind: item.kind,
        is_attention_check: item.is_attention_check,
        presented: {
          scale_id: current.scale?.scale_id ?? null,
          option_display_order: current.optionDisplayOrder ?? null,
          option_labels: current.phase === "pair" ? item.option_labels : null,
          rendered_image: answer.onset.rendered_image,
        },
        response: {
          scale_value: answer.scale_value,
          choice: answer.choice,
          other_class_text: answer.other_class_text ?? null,
          n_changes: answer.n_changes,
        },
        timing: {
          image_loaded_ms: answer.onset.image_loaded_ms,
          onset_ms: answer.onset.onset_ms,
          first_interaction_ms: answer.first_interaction_ms ?? null,
          response_selected_ms: answer.response_selected_ms ?? null,
          submitted_ms: clock ? clock.nowMs() : answer.onset.onset_ms,
        },
      };

      const nextRec = appendTrial(rec, trial);
      const phaseList = orders[current.phase];
      const isLastInPhase = current.position >= phaseList.length;

      if (isLastInPhase) {
        advancePhase(current.phase, nextRec);
      } else {
        persist(nextRec); // persist after every trial
      }
    },
    [create, current, orders, phase, persist, advancePhase]
  );

  const pushIntegrity = useCallback((ev: IntegrityEvent) => {
    const rec = recordRef.current;
    if (!rec) return;
    const events = [...(rec.integrity_events ?? []), ev];
    const next: SessionRecord = { ...rec, integrity_events: events };
    recordRef.current = next;
    setRecord(next);
    lsSet(LS_RECORD, JSON.stringify(next));
  }, []);

  // Holds the COMPLETE final record awaiting a confirmed submit, so a retry can
  // re-attempt without rebuilding (and without re-reading demographics).
  const pendingSubmitRef = useRef<SessionRecord | null>(null);

  // Run the durable submit for an already-built complete record: checkpoint the
  // full record first (so the server holds it even if submit fails), then POST
  // submit. Only a confirmed success flips us to `completed` + /done.
  const runSubmit = useCallback(
    async (complete: SessionRecord) => {
      setPhase("submitting");
      // Captured across the durableSubmit boundary so the failure branch can
      // surface the server's schema validation_errors for the retry panel.
      let lastValidationErrors: object[] | null = null;
      const { submitted } = await durableSubmit(complete, {
        checkpoint, // PUT — persists verbatim regardless of validity
        submit: async (rec) => {
          const res = await fetch(`/api/sessions/${rec.session_id}/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(rec),
          });
          // HTTP 200 is NOT success: the route returns 200 even for a
          // schema-invalid record (body.ok === false). interpretSubmitResponse
          // reads the body and only reports ok when body.ok === true, so an
          // invalid final record never gets recorded as `completed`.
          const { ok, validation_errors } = await interpretSubmitResponse(res);
          lastValidationErrors = validation_errors;
          return { ok };
        },
      });

      if (!submitted) {
        // Either the HTTP request failed OR the server reported the record as
        // schema-invalid (body.ok === false). The record is already safe on the
        // server (checkpoint step) and verbatim in LS_RECORD, so retry is safe.
        // Do NOT mark completed, do NOT clear localStorage, do NOT navigate —
        // surface any validation errors and offer a retry.
        setSubmitValidationErrors(lastValidationErrors);
        setPhase("submit_failed");
        return;
      }

      // Confirmed valid: stamp the completed status locally, persist, set the
      // flag, and route to /done (which then clears the in-progress caches).
      setSubmitValidationErrors(null);
      const done: SessionRecord = { ...complete, status: "completed" };
      persist(done);
      pendingSubmitRef.current = null;
      lsSet(LS_COMPLETED, new Date().toISOString());
      setPhase("done");
    },
    [checkpoint, persist]
  );

  const submitDemographics = useCallback(
    async (values: DemographicsValues) => {
      const rec = recordRef.current;
      const clock = clockRef.current;
      if (!rec || !create) return;

      setPhase("submitting");

      const demographics = mapDemographics(values);
      let withDemo = setDemographics(rec, demographics);
      // Close the demographics phase (merges into prior loads' phase_timings).
      withDemo = closePhase(withDemo, clock, "demographics");
      withDemo = finalizeQualitySummary(withDemo, create);
      // Final timing is stamped now, but status stays "abandoned" until the
      // submit confirms — the checkpoint write below carries this complete
      // record so the server never holds a demographics-less copy.
      withDemo = {
        ...withDemo,
        timing: {
          ...withDemo.timing,
          completed_at_utc: new Date().toISOString(),
          total_duration_ms: clock ? clock.nowMs() : null,
        },
      };

      // Persist the complete record locally and stash it for retry.
      persist(withDemo);
      pendingSubmitRef.current = withDemo;

      await runSubmit(withDemo);
    },
    [create, persist, runSubmit]
  );

  const retrySubmit = useCallback(async () => {
    const complete = pendingSubmitRef.current ?? recordRef.current;
    if (!complete) return;
    await runSubmit(complete);
  }, [runSubmit]);

  // Mark demographics-phase entry once (latched against re-renders).
  const demoEntered = useRef(false);
  useEffect(() => {
    if (phase === "demographics" && !demoEntered.current) {
      demoEntered.current = true;
      clockRef.current?.markPhase("demographics", "enter");
    }
  }, [phase]);

  const totals = useMemo(() => {
    if (!orders) return { total: 0, completed: 0 };
    const total =
      orders.text.length + orders.image.length + orders.pair.length;
    const completed =
      completedByPhase.text + completedByPhase.image + completedByPhase.pair;
    return { total, completed };
  }, [orders, completedByPhase]);

  return {
    phase,
    instructionsFor,
    current,
    create,
    clock: clockRef.current,
    totalTrials: totals.total,
    completedTrials: totals.completed,
    beginPhase,
    submitTrial,
    submitDemographics,
    retrySubmit,
    pushIntegrity,
    submitValidationErrors,
  };
}

// ─── pure helpers (exported for tests) ───────────────────────────────────────

export function countByPhase(record: SessionRecord): Record<JudgmentPhase, number> {
  const out: Record<JudgmentPhase, number> = { text: 0, image: 0, pair: 0 };
  for (const t of record.trials) {
    if (t.phase_id === "text" || t.phase_id === "image" || t.phase_id === "pair") {
      out[t.phase_id] += 1;
    }
  }
  return out;
}

/**
 * closePhase — mark `phaseId` as exited on the clock and MERGE the newly closed
 * timing row into the record's existing phase_timings.
 *
 * The clock resets to a fresh timeline on every reload, so clock.getPhaseTimings
 * only holds phases closed in THIS load. We therefore append the row(s) the
 * clock has just produced for `phaseId` to whatever the record already carries
 * (entries from prior loads), instead of overwriting — otherwise a reload would
 * drop the timings of phases completed before it. Pure w.r.t. the record.
 */
export function closePhase(
  record: SessionRecord,
  clock: SessionClock | null,
  phaseId: PhaseTiming["phase_id"]
): SessionRecord {
  if (!clock) return record;
  clock.markPhase(phaseId, "exit");
  // Rows the current clock has closed for this phase id (usually exactly one).
  const closedNow = clock
    .getPhaseTimings()
    .filter((p) => p.phase_id === phaseId);
  // Keep prior-load rows for OTHER phases, plus any already-recorded rows for
  // this phase, then add the fresh ones not already present (by entered_ms).
  const existing = record.phase_timings ?? [];
  const haveEntered = new Set(
    existing.filter((p) => p.phase_id === phaseId).map((p) => p.entered_ms)
  );
  const additions = closedNow.filter((p) => !haveEntered.has(p.entered_ms));
  return { ...record, phase_timings: [...existing, ...additions] };
}

/**
 * SubmitResponseLike — the minimal shape of the fetch Response the submit route
 * returns. Exposed so the interpreter below is unit-testable without a network.
 */
export interface SubmitResponseLike {
  ok: boolean;
  json: () => Promise<unknown>;
}

/**
 * interpretSubmitResponse — decide whether a submit POST truly succeeded.
 *
 * CRITICAL: the submit route returns HTTP 200 EVEN for a schema-INVALID record
 * (store.submitSession persists verbatim and returns `body.ok === false`). A
 * passing HTTP status (`res.ok`) is therefore NOT a successful submit. We must
 * also read the JSON body and require `body.ok === true`; otherwise a
 * schema-invalid final record would be recorded as `completed`, LS_COMPLETED set
 * and LS_RECORD cleared — locking the participant out with only invalid data.
 *
 * Returns `{ ok, validation_errors }` where `ok` is true only when BOTH the HTTP
 * status is ok AND the body reports `ok === true`. validation_errors carries the
 * server's schema errors (for the retry panel) when present.
 */
export async function interpretSubmitResponse(
  res: SubmitResponseLike
): Promise<{ ok: boolean; validation_errors: object[] | null }> {
  let body: { ok?: boolean; validation_errors?: object[] | null } | null = null;
  try {
    body = (await res.json()) as {
      ok?: boolean;
      validation_errors?: object[] | null;
    };
  } catch {
    /* non-JSON / empty body → treated as not-ok below */
  }
  return {
    ok: res.ok && body?.ok === true,
    validation_errors: body?.validation_errors ?? null,
  };
}

/**
 * durableSubmit — make the final submit survive a flaky network.
 *
 * The LAST write to reach the server before this point is the pair-phase EXIT
 * checkpoint: it has status "abandoned" and NO demographics / quality_summary /
 * final timing. If the submit POST then fails, the participant's completed work
 * would be lost. To prevent that we FIRST checkpoint the COMPLETE record
 * (demographics + quality_summary + all phase_timings, status still
 * "abandoned") — writeCheckpoint persists verbatim regardless of validity, so
 * the server now holds the full record — and only THEN attempt the fallible
 * submit. A failed checkpoint is non-fatal (we still try to submit); a failed
 * submit returns {submitted:false} so the caller withholds the `completed`
 * flag and offers a retry.
 *
 * Pure w.r.t. I/O: both side effects are injected, so this is unit-testable.
 */
export async function durableSubmit(
  record: SessionRecord,
  io: {
    checkpoint: (rec: SessionRecord) => Promise<void>;
    submit: (rec: SessionRecord) => Promise<{ ok: boolean }>;
  }
): Promise<{ submitted: boolean }> {
  // 1) Durably persist the complete record server-side BEFORE the fallible
  //    submit. Best-effort: a checkpoint failure must not block the submit.
  try {
    await io.checkpoint(record);
  } catch {
    /* transient — the submit attempt below may still land the record */
  }
  // 2) Attempt the confirming submit. Only a truthy `ok` counts as completed.
  try {
    const res = await io.submit(record);
    return { submitted: res.ok };
  } catch {
    return { submitted: false };
  }
}

type ResumeTarget =
  | "demographics"
  | { phase: JudgmentPhase; atPosition: number };

export function resumePhase(
  create: CreateResult,
  done: Record<JudgmentPhase, number>
): ResumeTarget {
  for (const p of JUDGMENT_ORDER) {
    const total = create.items[p]?.length ?? 0;
    if (done[p] < total) {
      return { phase: p, atPosition: done[p] };
    }
  }
  return "demographics";
}

function nextImageUrl(
  orders: Record<JudgmentPhase, ItemPayload[]>,
  phase: FlowPhase,
  done: Record<JudgmentPhase, number>
): string | null {
  if (phase !== "image" && phase !== "pair") return null;
  const list = orders[phase];
  const next = list[done[phase] + 1]; // the trial AFTER the current one
  if (next?.image) return `/api/images/${next.image.uri_name}`;
  return null;
}

/** Default minimum rendered CSS px when the create payload omits `quality`. */
const DEFAULT_MIN_RENDERED_PX = 256;

export function getMinPx(create: CreateResult): number {
  // quality.min_rendered_image_css_px is forwarded on the create payload (see
  // store.ts). Fall back to the known study default only when absent/null.
  return create.quality?.min_rendered_image_css_px ?? DEFAULT_MIN_RENDERED_PX;
}

/**
 * Whether the current viewport can fit an image at min_rendered_image_css_px in
 * both dimensions. SSR-safe: returns true when window is unavailable (the gate
 * re-checks client-side). NOTE: this is a coarse viewport gate, not a per-image
 * one — a stimulus larger than the viewport is still possible and is the reason
 * StimulusImage never upscales. Task-8 e2e should exercise an oversized image.
 */
function viewportFits(create: CreateResult): boolean {
  if (typeof window === "undefined") return true;
  const min = getMinPx(create);
  return window.innerWidth >= min && window.innerHeight >= min;
}

function mapDemographics(values: DemographicsValues): NonNullable<SessionRecord["demographics"]> {
  return {
    age_band: values.age_band as NonNullable<SessionRecord["demographics"]>["age_band"],
    ml_familiarity: values.ml_familiarity as NonNullable<
      SessionRecord["demographics"]
    >["ml_familiarity"],
    english_proficiency: values.english_proficiency as NonNullable<
      SessionRecord["demographics"]
    >["english_proficiency"],
    comment: values.comment ? values.comment : null,
  };
}

// Re-export so consent page can seed localStorage with a fresh record.
export { initRecord };
