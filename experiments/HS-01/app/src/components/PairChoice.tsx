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
import type { ReferenceEntry, SemanticChoice } from "@/lib/types";
import { WordReference } from "./WordReference";

type WordSlot = "ANCHOR_WORD" | "TARGET_WORD";

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
  /**
   * Curated word → {gloss, image} map. A word slot whose label has an entry
   * gets an ⓘ helper next to it; absence is silent (no helper, word still
   * selectable).
   */
  references?: Record<string, ReferenceEntry>;
  /** Fired once per word slot, the first time its helper popover is opened. */
  onReveal?: (slot: WordSlot) => void;
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
  references,
  onReveal,
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
          const label = labelFor(slot);
          const ref = isWord ? references?.[label] : undefined;
          return (
            <div key={slot} className="relative">
              <button
                type="button"
                role="radio"
                aria-checked={selected}
                disabled={disabled}
                data-slot={slot}
                data-testid={`pair-option-${slot}`}
                onClick={() => select(slot)}
                className={[
                  "block w-full rounded-control border px-4 py-3 text-left text-base transition-colors",
                  selected
                    ? "border-tum-500 bg-tum-50 text-tum-900 font-medium"
                    : isWord
                      ? "border-line bg-white text-body hover:border-tum-300"
                      : // Fixed non-word tail: same shape and affordance, but a
                        // quieter surface so it reads as secondary without
                        // priming any class answer.
                        "border-line bg-surface text-muted hover:border-tum-300",
                  ref ? "pr-12" : "",
                  disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
                ].join(" ")}
              >
                {label}
              </button>
              {ref && (
                // inset-y-0 + flex centres the ⓘ vertically WITHOUT a transform.
                // (A `transform` here would create a stacking context that traps
                // the popover's z-index behind the later option rows.)
                <span className="absolute inset-y-0 right-2 flex items-center">
                  <WordReference
                    word={label}
                    gloss={ref.gloss}
                    image={ref.image}
                    disabled={disabled}
                    onReveal={() => onReveal?.(slot as WordSlot)}
                  />
                </span>
              )}
            </div>
          );
        })}
      </div>

      {otherClassFreeText && value === "OTHER_CLASS" && (
        <div className="mt-3">
          <label
            htmlFor="other-class-text"
            className="block text-sm text-muted mb-1"
          >
            Optional: what did you see instead?
          </label>
          <input
            id="other-class-text"
            type="text"
            data-testid="other-class-text"
            value={otherClassText}
            disabled={disabled}
            onChange={(e) => onOtherClassText(e.target.value)}
            className="w-full rounded-control border border-line bg-white px-3 py-2 text-base text-body placeholder:text-muted"
            placeholder="(optional)"
          />
        </div>
      )}
    </div>
  );
}
