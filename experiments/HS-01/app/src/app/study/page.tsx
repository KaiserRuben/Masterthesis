"use client";

/**
 * /study — the phased runner. Drives consent→text→image→pair→demographics→done
 * via useSession, mounts the right phase component, and owns per-trial onset
 * timing + integrity instrumentation.
 *
 * Animation-free stimulus: framer-motion wraps only chrome (AnimatePresence on
 * the phase shell), never the stimulus/response, so onset stays deterministic.
 */

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { useSession } from "@/state/useSession";
import type { CurrentTrial, TrialAnswer } from "@/state/useSession";
import { attachIntegrityListeners } from "@/lib/instrumentation";
import { awaitOnset, type OnsetResult } from "@/lib/instrumentation";
import type { SessionClock } from "@/lib/timing";
import type { ReferenceEntry, SemanticChoice } from "@/lib/types";

import { Instructions } from "@/components/Instructions";
import { PhaseProgress } from "@/components/PhaseProgress";
import { PromptText } from "@/components/PromptText";
import { StimulusImage } from "@/components/StimulusImage";
import { LikertScale } from "@/components/LikertScale";
import { PairChoice } from "@/components/PairChoice";
import { Demographics } from "@/components/Demographics";

export default function StudyPage() {
  const router = useRouter();
  const s = useSession();

  // Send the rater back to consent if there is no resumable session.
  useEffect(() => {
    if (s.phase === "missing") router.replace("/");
  }, [s.phase, router]);

  // ── integrity listeners during judgment phases ──────────────────────────
  useEffect(() => {
    const judging = s.phase === "text" || s.phase === "image" || s.phase === "pair";
    if (!judging || !s.clock) return;
    const detach = attachIntegrityListeners(s.clock, s.pushIntegrity);
    return detach;
  }, [s.phase, s.clock, s.pushIntegrity]);

  if (s.phase === "loading" || s.phase === "missing") {
    return <Centered>Loading…</Centered>;
  }

  if (s.phase === "too_small") {
    return (
      <Centered>
        <div className="max-w-md text-center">
          <h2 className="text-xl font-semibold mb-3">A larger screen is needed</h2>
          <p className="text-neutral-600">
            The images in this study are shown at their exact size and your
            window is too small to display them faithfully. Please open this
            study on a larger screen or maximize your browser window, then
            reload.
          </p>
        </div>
      </Centered>
    );
  }

  if (s.phase === "submitting") {
    return <Centered>Submitting your responses…</Centered>;
  }

  if (s.phase === "submit_failed") {
    const errCount = s.submitValidationErrors?.length ?? 0;
    return (
      <Centered>
        <div
          className="max-w-md text-center"
          role="alert"
          aria-live="assertive"
          data-testid="submit-failed"
        >
          <h2 className="mb-3 text-xl font-semibold text-neutral-900">
            We couldn&apos;t confirm your submission
          </h2>
          <p className="mb-6 leading-relaxed text-neutral-600">
            Your responses are saved and have not been lost. We just need to
            confirm them with the server. Please check your connection and try
            again.
          </p>
          {errCount > 0 && (
            <p
              className="mb-6 text-sm text-neutral-500"
              data-testid="submit-validation-errors"
            >
              The server reported {errCount} issue{errCount === 1 ? "" : "s"} with
              the recorded data. Your responses are still saved.
            </p>
          )}
          <button
            type="button"
            data-testid="retry-submit"
            onClick={() => void s.retrySubmit()}
            className="rounded-control bg-tum-600 px-6 py-3 font-medium text-white hover:bg-tum-700"
          >
            Retry submit
          </button>
        </div>
      </Centered>
    );
  }

  if (s.phase === "done") {
    // Route to the dedicated done page so a reload won't re-enter the runner.
    router.replace("/done");
    return <Centered>Thank you!</Centered>;
  }

  if (s.phase === "instructions" && s.instructionsFor) {
    return (
      <Shell phaseKey={`instructions:${s.instructionsFor}`}>
        <Instructions
          phase={s.instructionsFor}
          part={s.order.indexOf(s.instructionsFor) + 1}
          onContinue={s.beginPhase}
        />
      </Shell>
    );
  }

  if (s.phase === "demographics" && s.create) {
    return (
      <Shell phaseKey="demographics">
        <Demographics
          fields={s.create.demographics_fields}
          onSubmit={(v) => void s.submitDemographics(v)}
        />
      </Shell>
    );
  }

  if (s.current && s.clock) {
    return (
      <Shell phaseKey={`trial:${s.current.item.item_id}`}>
        {/* Viewport-height column: the stimulus leads (content first); the
            footer (progress strip + Next control, inside TrialView) pins to the
            bottom as one unit. Only the stimulus + response region above it
            scrolls when an image is taller than the viewport, so "the last part"
            is never pushed below the fold. */}
        <div className="mx-auto flex h-screen max-w-2xl flex-col px-6 py-6">
          <TrialView
            // key forces a fresh mount per trial → answer state + onset reset.
            key={s.current.item.item_id}
            trial={s.current}
            clock={s.clock}
            onSubmit={s.submitTrial}
            pairDisplayLabels={s.create?.pair_response.display_labels}
            otherClassFreeText={s.create?.pair_response.other_class_free_text ?? true}
            references={s.create?.references}
            position={Math.min(s.completedTrials + 1, s.totalTrials)}
            total={s.totalTrials}
          />
        </div>
      </Shell>
    );
  }

  return <Centered>Loading…</Centered>;
}

// ─── per-trial view ───────────────────────────────────────────────────────────

// Brief highlight window after a pick before the trial auto-advances. Long
// enough to confirm the selection landed (and to allow a quick re-pick to
// correct a misclick), short enough not to feel like a wait. The e2e walker
// mirrors this with its own post-pick wait.
const AUTO_ADVANCE_MS = 250;

interface PairDisplayLabels {
  OTHER_CLASS: string;
  NOTHING_RECOGNIZABLE: string;
  CANT_TELL: string;
}

interface TrialViewProps {
  trial: CurrentTrial;
  clock: SessionClock;
  onSubmit: (answer: TrialAnswer) => void;
  pairDisplayLabels?: PairDisplayLabels;
  otherClassFreeText: boolean;
  references?: Record<string, ReferenceEntry>;
  /** Overall (whole-session) progress, rendered in the pinned footer. */
  position: number;
  total: number;
}

function TrialView({
  trial,
  clock,
  onSubmit,
  pairDisplayLabels,
  otherClassFreeText,
  references,
  position,
  total,
}: TrialViewProps) {
  // Onset for image/pair comes from StimulusImage's onReady; for text we await
  // a single frame here on mount.
  const [onset, setOnset] = useState<OnsetResult | null>(null);

  // Answer state (local to this mounted trial).
  const [scaleValue, setScaleValue] = useState<number | null>(null);
  const [choice, setChoice] = useState<SemanticChoice | null>(null);
  const [otherText, setOtherText] = useState("");
  const [nChanges, setNChanges] = useState(0);

  const firstInteractionMs = useRef<number | null>(null);
  const responseSelectedMs = useRef<number | null>(null);
  // Word-option ⓘ helpers opened during this trial (insertion-ordered set).
  const revealedRef = useRef<Set<"ANCHOR_WORD" | "TARGET_WORD">>(new Set());
  // Pending auto-advance timer (the highlight window before the trial commits).
  const advanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isText = trial.phase === "text";
  const isImage = trial.phase === "image";
  const isPair = trial.phase === "pair";

  // Text-trial onset: one frame after mount (no image element).
  useEffect(() => {
    if (!isText) return;
    let cancelled = false;
    awaitOnset(clock).then((r) => {
      if (!cancelled) setOnset(r);
    });
    return () => {
      cancelled = true;
    };
  }, [isText, clock]);

  // Never let a scheduled advance fire after this trial has unmounted.
  useEffect(() => {
    return () => {
      if (advanceTimer.current !== null) clearTimeout(advanceTimer.current);
    };
  }, []);

  // Onset must have fired before an answer can be committed (text waits for the
  // frame; image/pair wait for decode+frame via StimulusImage). The response
  // widget is disabled until then, so a pick cannot beat its own onset.
  const ready = onset !== null;

  const markInteraction = () => {
    if (firstInteractionMs.current === null) firstInteractionMs.current = clock.nowMs();
  };

  // Build + emit the trial answer. The answer values are passed in explicitly
  // (not read from state) so a delayed auto-advance commits the chosen answer,
  // not a stale render's snapshot. onset / timing refs are live, so they read
  // correctly at fire time.
  const commit = (payload: {
    scaleValue?: number;
    choice?: SemanticChoice;
    otherText?: string;
    nChanges: number;
  }) => {
    if (!onset) return;
    const answer: TrialAnswer = {
      n_changes: payload.nChanges,
      onset,
      first_interaction_ms: firstInteractionMs.current,
      response_selected_ms: responseSelectedMs.current,
    };
    if (isPair) {
      answer.choice = payload.choice ?? undefined;
      answer.other_class_text =
        payload.choice === "OTHER_CLASS" ? payload.otherText ?? "" : null;
      if (revealedRef.current.size > 0) {
        answer.references_revealed = [...revealedRef.current];
      }
    } else {
      answer.scale_value = payload.scaleValue;
    }
    onSubmit(answer);
  };

  // No Next button: a pick auto-advances after AUTO_ADVANCE_MS. A re-pick within
  // the window resets the timer (and the widget counts it as a change), so a
  // quick correction is still possible before the trial commits.
  const scheduleAdvance = (run: () => void) => {
    if (advanceTimer.current !== null) clearTimeout(advanceTimer.current);
    advanceTimer.current = setTimeout(() => {
      advanceTimer.current = null;
      run();
    }, AUTO_ADVANCE_MS);
  };

  const handleScale = (value: number, changes: number) => {
    markInteraction();
    setScaleValue(value);
    setNChanges(changes);
    responseSelectedMs.current = clock.nowMs();
    scheduleAdvance(() => commit({ scaleValue: value, nChanges: changes }));
  };

  const handleChoice = (c: SemanticChoice, changes: number) => {
    markInteraction();
    setChoice(c);
    setNChanges(changes);
    responseSelectedMs.current = clock.nowMs();
    // "Something else" + its free-text field must NOT auto-advance: the rater
    // needs a beat to (optionally) type, then commits via Confirm. Every other
    // option auto-advances like the rating phases.
    if (c === "OTHER_CLASS" && otherClassFreeText) {
      if (advanceTimer.current !== null) clearTimeout(advanceTimer.current);
      advanceTimer.current = null;
      return;
    }
    setOtherText("");
    scheduleAdvance(() => commit({ choice: c, otherText: "", nChanges: changes }));
  };

  return (
    // Fill the height the parent column hands us, so the Next footer can pin to
    // the bottom while the stimulus/response region above it scrolls if needed.
    <div className="flex min-h-0 flex-1 flex-col">
      {/* ── scrollable column: stimulus (top) + response (bottom) ──
          min-h-0 lets this flex child shrink below its content. The stimulus
          pins to the top; the response block pins to the BOTTOM (mt-auto) so its
          click targets hold a stable position across trials — see the response
          comment below. Both collapse to flush + scroll when a tall image fills
          the space. */}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        {/* ── stimulus: top of the scroll area ── */}
        <div className="pt-1">
          {/* Text phase is intentionally image-free. Without a cue, the empty
              zone where the image normally sits reads as a failed/missing image,
              so raters report "the image is gone". A quiet, neutral chrome pill
              (UI font, not the stimulus font) occupies that zone and states the
              absence is by design. Neutral wording only — it must not hint that
              the question text is altered, which would bias the rating. */}
          {isText && (
            <div className="mb-6 flex justify-center">
              <span className="rounded-full bg-surface px-3 py-1 text-xs font-medium text-muted">
                Text only — no image in this part
              </span>
            </div>
          )}

          {(isImage || isPair) && trial.item.image && (
            <div className="mb-5">
              <StimulusImage
                uriName={trial.item.image.uri_name}
                naturalW={trial.item.image.natural_w}
                naturalH={trial.item.image.natural_h}
                clock={clock}
                onReady={setOnset}
              />
            </div>
          )}

          {(isText || isPair) && trial.item.prompt != null && (
            <div>
              <PromptText text={trial.item.prompt} />
            </div>
          )}
        </div>

        {/* ── response: pinned to the FOOT of the scroll area (mt-auto) ──
            With auto-advance the option you click is ALSO what advances the
            trial — so the options must hold a stable vertical position, or they
            jump under the cursor as each trial's stimulus height changes.
            Anchoring the whole response block to the bottom keeps every option
            in the same place trial-to-trial; the variable gap lands harmlessly
            in the middle. The hairline + TUM eyebrow still fence it off from the
            stimulus above. Until onset fires the widget is disabled (greyed): a
            pick commits the trial, so we must not accept one before its onset is
            recorded. */}
        <div className="mt-auto border-t border-line pt-6">
          <p className="mb-4 text-xs font-semibold uppercase tracking-[0.12em] text-tum-600">
            {responseKicker(trial.phase)}
          </p>
          {(isText || isImage) && trial.scale && (
            <LikertScale
              scale={trial.scale}
              value={scaleValue}
              onChange={handleScale}
              disabled={!ready}
            />
          )}

          {isPair &&
            trial.optionDisplayOrder &&
            trial.item.option_labels &&
            pairDisplayLabels && (
              <PairChoice
                optionLabels={trial.item.option_labels}
                displayLabels={pairDisplayLabels}
                displayOrder={trial.optionDisplayOrder}
                otherClassFreeText={otherClassFreeText}
                value={choice}
                otherClassText={otherText}
                onChange={handleChoice}
                onOtherClassText={setOtherText}
                references={references}
                onReveal={(slot) => revealedRef.current.add(slot)}
                disabled={!ready}
              />
            )}

          {/* The ONE commit button left in the flow: "Something else" + its
              optional free text doesn't auto-advance, so the rater confirms it
              explicitly once they've (optionally) typed. */}
          {isPair && choice === "OTHER_CLASS" && otherClassFreeText && (
            <div className="mt-5 flex justify-end">
              <button
                type="button"
                data-testid="trial-confirm"
                onClick={() => commit({ choice: "OTHER_CLASS", otherText, nChanges })}
                disabled={!ready}
                className={[
                  "rounded-control px-6 py-3 font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-tum-500",
                  !ready
                    ? "bg-neutral-300 cursor-not-allowed"
                    : "bg-tum-600 hover:bg-tum-700",
                ].join(" ")}
              >
                Confirm
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── pinned footer: just the progress strip ──
          There is no Next button in the judgment phases — a pick auto-advances
          (see scheduleAdvance) — so the only thing pinned here is overall
          progress, divided from the scrolling content by a single hairline. */}
      <div className="mt-4 shrink-0 border-t border-line pt-4">
        <PhaseProgress
          part={trial.partIndex}
          partCount={3}
          sectionLabel={phaseLabel(trial.phase)}
          position={position}
          total={total}
        />
      </div>
    </div>
  );
}

// ─── small chrome helpers ─────────────────────────────────────────────────────

function phaseLabel(phase: "text" | "image" | "pair"): string {
  if (phase === "text") return "Questions";
  if (phase === "image") return "Images";
  return "Image and question";
}

// Eyebrow that labels the response zone as the rater's task. Neutral wording
// (it must not hint at the answer): the rating phases ask for a rating, the
// pair phase asks for a choice.
function responseKicker(phase: "text" | "image" | "pair"): string {
  return phase === "pair" ? "Your answer" : "Your rating";
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex min-h-screen items-center justify-center px-6 text-neutral-600">
      {children}
    </main>
  );
}

function Shell({ children, phaseKey }: { children: React.ReactNode; phaseKey: string }) {
  // Under prefers-reduced-motion the cross-fade is instant. This is an
  // accessibility requirement, but it also removes the overlap window that the
  // exit fade otherwise creates: with mode="wait" + a non-zero exit duration,
  // the OUTGOING trial (its still-interactive Next button included) lingers in
  // the DOM while fading, so a fast actor can interact with a trial that is on
  // its way out. Zeroing the duration makes each trial unmount the instant the
  // next one is committed.
  const reduce = useReducedMotion();
  return (
    <main className="min-h-screen">
      <AnimatePresence mode="wait">
        <motion.div
          key={phaseKey}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduce ? 0 : 0.15 }}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </main>
  );
}
