"use client";

/**
 * Instructions — a brief, NEUTRAL per-phase intro screen.
 *
 * Framing must not prime the rater toward any class or judgment (no "find the
 * snake", no "is this hard"). framer-motion is allowed here (chrome, not
 * stimulus). One continue button.
 */

import { motion } from "framer-motion";

export type InstructionPhase = "text" | "image" | "pair";

const COPY: Record<
  InstructionPhase,
  { title: string; body: string[] }
> = {
  text: {
    title: "Part 1 of 3 — Questions",
    body: [
      "You will see a series of short questions, one at a time.",
      "For each, tell us how much you agree that you can tell what the question is asking. There are no right or wrong answers — we want your honest first impression.",
    ],
  },
  image: {
    title: "Part 2 of 3 — Images",
    body: [
      "You will see a series of images, one at a time.",
      "For each, tell us how much you agree that you can tell what the image is displaying.",
    ],
  },
  pair: {
    title: "Part 3 of 3 — Image and question",
    body: [
      "You will see an image together with a short question.",
      "Choose the answer that best matches what you see. If none of the named options fit, you can say it is something else, that nothing is recognizable, or that you can't tell.",
    ],
  },
};

export interface InstructionsProps {
  phase: InstructionPhase;
  onContinue: () => void;
}

export function Instructions({ phase, onContinue }: InstructionsProps) {
  const copy = COPY[phase];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="mx-auto max-w-xl px-6 py-12"
    >
      <h2 className="text-2xl font-semibold text-neutral-900 mb-6">{copy.title}</h2>
      <div className="space-y-4 text-neutral-700 leading-relaxed mb-10">
        {copy.body.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>
      <button
        type="button"
        data-testid="instructions-continue"
        onClick={onContinue}
        className="rounded-lg bg-blue-600 px-6 py-3 text-white font-medium hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      >
        Begin
      </button>
    </motion.div>
  );
}
