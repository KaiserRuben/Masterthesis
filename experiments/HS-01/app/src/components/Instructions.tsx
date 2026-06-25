"use client";

/**
 * Instructions — a brief, NEUTRAL per-phase intro screen.
 *
 * Framing must not prime the rater toward any class or judgment (no "find the
 * snake", no "is this hard"). framer-motion is allowed here (chrome, not
 * stimulus). One continue button.
 *
 * The text phase carries an extra neutral note: it has no images, only the
 * wording of questions. This is the reported "the image is missing" confusion —
 * stated here AND reinforced on the trial itself. Neutral only: it describes the
 * medium, never hints the question text is altered (which would bias the rating).
 */

import { motion } from "framer-motion";

export type InstructionPhase = "text" | "image" | "pair";

const COPY: Record<
  InstructionPhase,
  { title: string; body: string[]; note?: string }
> = {
  text: {
    title: "Questions",
    body: [
      "You will see a series of short questions, one at a time.",
      "For each, tell us how much you agree that you can tell what the question is asking. There are no right or wrong answers — we want your honest first impression.",
    ],
    note: "This part has no images — you'll just see the wording of each question.",
  },
  image: {
    title: "Images",
    body: [
      "You will see a series of images, one at a time.",
      "For each, tell us how much you agree that you can tell what the image is displaying.",
    ],
  },
  pair: {
    title: "Image and question",
    body: [
      "You will see an image together with a short question. Please read both carefully; they belong together, and each one matters.",
      "Base your answer on the image and the question combined. If none of the named options fit, you can say it is something else, that nothing is recognizable, or that you can't tell.",
    ],
  },
};

export interface InstructionsProps {
  phase: InstructionPhase;
  /**
   * 1-based position of this phase in the session's counterbalanced order. The
   * copy is phase-keyed but the part index is NOT — image can lead (Part 1) and
   * text can follow it (Part 2). Computed by the parent from the session order.
   */
  part: number;
  onContinue: () => void;
}

export function Instructions({ phase, part, onContinue }: InstructionsProps) {
  const copy = COPY[phase];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="mx-auto max-w-xl px-6 py-14"
    >
      <div
        className="mb-6 flex items-center gap-2"
        aria-label={`Part ${part} of 3`}
      >
        {[1, 2, 3].map((p) => (
          <span
            key={p}
            className={[
              "h-1.5 rounded-full transition-colors",
              p === part
                ? "w-8 bg-tum-500"
                : p < part
                  ? "w-4 bg-tum-300"
                  : "w-4 bg-line",
            ].join(" ")}
          />
        ))}
        <span className="ml-2 text-xs font-medium uppercase tracking-wider text-muted">
          Part {part} of 3
        </span>
      </div>

      <h2 className="mb-5 text-3xl font-semibold tracking-tight">{copy.title}</h2>

      <div className="mb-8 space-y-4 leading-relaxed text-body">
        {copy.body.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>

      {copy.note && (
        <div className="mb-9 rounded-control border border-line bg-surface px-4 py-3 text-sm text-body">
          {copy.note}
        </div>
      )}

      <button
        type="button"
        data-testid="instructions-continue"
        onClick={onContinue}
        className="rounded-control bg-tum-600 px-6 py-3 font-medium text-white transition-colors hover:bg-tum-700"
      >
        Begin
      </button>
    </motion.div>
  );
}
