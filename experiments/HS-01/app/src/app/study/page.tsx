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
import { AnimatePresence, motion } from "framer-motion";

import { useSession } from "@/state/useSession";
import type { CurrentTrial, TrialAnswer } from "@/state/useSession";
import { attachIntegrityListeners } from "@/lib/instrumentation";
import { awaitOnset, type OnsetResult } from "@/lib/instrumentation";
import type { SessionClock } from "@/lib/timing";
import type { SemanticChoice } from "@/lib/types";

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

  if (s.phase === "done") {
    // Route to the dedicated done page so a reload won't re-enter the runner.
    router.replace("/done");
    return <Centered>Thank you!</Centered>;
  }

  if (s.phase === "instructions" && s.instructionsFor) {
    return (
      <Shell phaseKey={`instructions:${s.instructionsFor}`}>
        <Instructions phase={s.instructionsFor} onContinue={s.beginPhase} />
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
        <div className="mx-auto max-w-2xl px-6 py-10">
          <PhaseProgress
            position={s.current.position}
            total={s.current.total}
            label={phaseLabel(s.current.phase)}
          />
          <TrialView
            // key forces a fresh mount per trial → answer state + onset reset.
            key={s.current.item.item_id}
            trial={s.current}
            clock={s.clock}
            onSubmit={s.submitTrial}
            pairDisplayLabels={s.create?.pair_response.display_labels}
            otherClassFreeText={s.create?.pair_response.other_class_free_text ?? true}
          />
        </div>
      </Shell>
    );
  }

  return <Centered>Loading…</Centered>;
}

// ─── per-trial view ───────────────────────────────────────────────────────────

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
}

function TrialView({
  trial,
  clock,
  onSubmit,
  pairDisplayLabels,
  otherClassFreeText,
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

  const markInteraction = () => {
    if (firstInteractionMs.current === null) firstInteractionMs.current = clock.nowMs();
  };

  const handleScale = (value: number, changes: number) => {
    markInteraction();
    setScaleValue(value);
    setNChanges(changes);
    responseSelectedMs.current = clock.nowMs();
  };

  const handleChoice = (c: SemanticChoice, changes: number) => {
    markInteraction();
    setChoice(c);
    setNChanges(changes);
    responseSelectedMs.current = clock.nowMs();
    if (c !== "OTHER_CLASS") setOtherText("");
  };

  const answered = isPair ? choice !== null : scaleValue !== null;
  // Onset must have fired before we let the answer be committed (text waits for
  // the frame; image/pair wait for decode+frame via StimulusImage).
  const ready = onset !== null;

  const submit = () => {
    if (!answered || !onset) return;
    const answer: TrialAnswer = {
      n_changes: nChanges,
      onset,
      first_interaction_ms: firstInteractionMs.current,
      response_selected_ms: responseSelectedMs.current,
    };
    if (isPair) {
      answer.choice = choice ?? undefined;
      answer.other_class_text = choice === "OTHER_CLASS" ? otherText : null;
    } else {
      answer.scale_value = scaleValue ?? undefined;
    }
    onSubmit(answer);
  };

  return (
    <div>
      {/* ── stimulus ── */}
      {(isImage || isPair) && trial.item.image && (
        <div className="mb-6">
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
        <div className="mb-6">
          <PromptText text={trial.item.prompt} />
        </div>
      )}

      {/* ── response ── */}
      <div className="mt-6">
        {(isText || isImage) && trial.scale && (
          <LikertScale scale={trial.scale} value={scaleValue} onChange={handleScale} />
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
            />
          )}
      </div>

      <div className="mt-8 flex items-center justify-end gap-3">
        {!ready && <span className="text-xs text-neutral-400">Preparing…</span>}
        <button
          type="button"
          onClick={submit}
          disabled={!answered || !ready}
          className={[
            "rounded-lg px-6 py-3 font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
            !answered || !ready
              ? "bg-neutral-300 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700",
          ].join(" ")}
        >
          Next
        </button>
      </div>
    </div>
  );
}

// ─── small chrome helpers ─────────────────────────────────────────────────────

function phaseLabel(phase: "text" | "image" | "pair"): string {
  if (phase === "text") return "Question";
  if (phase === "image") return "Image";
  return "Image and question";
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex min-h-screen items-center justify-center px-6 text-neutral-600">
      {children}
    </main>
  );
}

function Shell({ children, phaseKey }: { children: React.ReactNode; phaseKey: string }) {
  return (
    <main className="min-h-screen">
      <AnimatePresence mode="wait">
        <motion.div
          key={phaseKey}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </main>
  );
}
