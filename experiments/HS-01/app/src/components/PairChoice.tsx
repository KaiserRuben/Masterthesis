"use client";

/**
 * PairChoice — the pair-phase response widget.
 *
 * Measurement contract (the load-bearing part of the whole study):
 *  - Options are rendered in the REALIZED `displayOrder` (an array of semantic
 *    slots produced by optionOrder(rng) for this trial). The two word slots are
 *    AB-randomized; the three tail slots are fixed and always last.
 *  - The button for a word slot shows optionLabels[slot]; the tail buttons show
 *    displayLabels[slot]. Clicking a button reports the SEMANTIC slot
 *    (ANCHOR_WORD / TARGET_WORD / OTHER_CLASS / …), never a positional index —
 *    so analysis is robust to the per-trial shuffle.
 *  - n_changes is tracked here (ref-based) exactly like LikertScale.
 *  - Selecting OTHER_CLASS reveals an optional free-text field (when
 *    otherClassFreeText is on); the text is reported separately.
 *  - Animation-free.
 *
 * Controlled by `value` / `otherClassText` so a resumed trial repaints.
 */

import { useRef } from "react";
import type { SemanticChoice } from "@/lib/types";

export interface PairChoiceProps {
  /** Words for the two AB slots. */
  optionLabels: { ANCHOR_WORD: string; TARGET_WORD: string };
  /** Display strings for the three fixed tail slots. */
  displayLabels: {
    OTHER_CLASS: string;
    NOTHING_RECOGNIZABLE: string;
    CANT_TELL: string;
  };
  /** Realized per-trial order (from optionOrder(rng)). */
  displayOrder: SemanticChoice[];
  /** Whether OTHER_CLASS reveals a free-text field. */
  otherClassFreeText: boolean;
  value: SemanticChoice | null;
  otherClassText: string;
  onChange: (choice: SemanticChoice, nChanges: number) => void;
  onOtherClassText: (text: string) => void;
  disabled?: boolean;
}

const WORD_SLOTS: ReadonlySet<SemanticChoice> = new Set([
  "ANCHOR_WORD",
  "TARGET_WORD",
]);

export function PairChoice({
  optionLabels,
  displayLabels,
  displayOrder,
  otherClassFreeText,
  value,
  otherClassText,
  onChange,
  onOtherClassText,
  disabled,
}: PairChoiceProps) {
  const lastValue = useRef<SemanticChoice | null>(value);
  const nChanges = useRef<number>(0);
  const hasSelected = useRef<boolean>(value != null);

  const select = (choice: SemanticChoice) => {
    if (disabled) return;
    if (!hasSelected.current) {
      hasSelected.current = true;
    } else if (choice !== lastValue.current) {
      nChanges.current += 1;
    }
    lastValue.current = choice;
    onChange(choice, nChanges.current);
  };

  const labelFor = (slot: SemanticChoice): string => {
    if (slot === "ANCHOR_WORD") return optionLabels.ANCHOR_WORD;
    if (slot === "TARGET_WORD") return optionLabels.TARGET_WORD;
    return displayLabels[slot as keyof typeof displayLabels];
  };

  return (
    <div>
      <div role="radiogroup" aria-label="What does this image show?" className="flex flex-col gap-2">
        {displayOrder.map((slot) => {
          const selected = value === slot;
          const isWord = WORD_SLOTS.has(slot);
          return (
            <button
              key={slot}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              data-slot={slot}
              onClick={() => select(slot)}
              className={[
                "rounded-lg border px-4 py-3 text-left text-base transition-colors",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
                selected
                  ? "border-blue-600 bg-blue-50 text-blue-900 font-medium"
                  : "border-neutral-300 bg-white text-neutral-800 hover:border-neutral-400",
                isWord ? "" : "italic text-neutral-600",
                disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
              ].join(" ")}
            >
              {labelFor(slot)}
            </button>
          );
        })}
      </div>

      {otherClassFreeText && value === "OTHER_CLASS" && (
        <div className="mt-3">
          <label
            htmlFor="other-class-text"
            className="block text-sm text-neutral-600 mb-1"
          >
            Optional: what did you see instead?
          </label>
          <input
            id="other-class-text"
            type="text"
            value={otherClassText}
            disabled={disabled}
            onChange={(e) => onOtherClassText(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            placeholder="(optional)"
          />
        </div>
      )}
    </div>
  );
}
