"use client";

/**
 * PromptText — verbatim stimulus-text renderer.
 *
 * FIDELITY RULE (binding): the prompt is inserted as a plain TEXT NODE inside a
 * pinned monospace element (.font-prompt, defined in globals.css). React renders
 * a string child as a text node, so NO HTML interpretation, NO Unicode
 * normalization, NO ligature folding occurs here — the bytes the rater sees are
 * the bytes in the pool. Do not run .normalize(), .trim(), or any transform on
 * `text` before display.
 *
 * Animation-free: presentation must not depend on a transition for onset.
 */

export interface PromptTextProps {
  text: string;
}

export function PromptText({ text }: PromptTextProps) {
  return (
    <div
      className="font-prompt rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-4 text-lg leading-relaxed text-neutral-900"
      // No dangerouslySetInnerHTML — {text} is rendered as a text node verbatim.
    >
      {text}
    </div>
  );
}
