"use client";

/**
 * WordReference — the on-demand ⓘ helper for an unfamiliar pair-option word.
 *
 * A small info button that opens an in-page popover with a one-line gloss and
 * (when available) a bundled example photo served from /api/refs. It stays in
 * the page — it never navigates away — so it does not perturb the study's
 * integrity-blur instrumentation.
 *
 * Measurement contract: this sits NEXT TO a radio option but is a separate
 * control. Clicking it must never select that option, so every click is
 * stopPropagation'd at the wrapper and never reaches the radio's handler.
 *
 * onReveal fires once, on first open — the per-trial reveal signal PairChoice
 * records as references_revealed.
 *
 * Animation-free: the popover appears instantly (it is chrome, but keeping it
 * still matches the stimulus-area discipline and avoids motion for reduced-
 * motion users).
 */

import { useEffect, useId, useRef, useState } from "react";
import { Info } from "lucide-react";

export interface WordReferenceProps {
  /** The class word this explains (e.g. "tench"). */
  word: string;
  /** One-line, neutral definition. */
  gloss: string;
  /** Bundled ref filename (ref-<class>.png) or null for a gloss-only entry. */
  image: string | null;
  /** Called once, the first time the popover is opened. */
  onReveal?: () => void;
  disabled?: boolean;
}

export function WordReference({
  word,
  gloss,
  image,
  onReveal,
  disabled,
}: WordReferenceProps) {
  const [open, setOpen] = useState(false);
  const revealed = useRef(false);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const dialogId = useId();

  const toggle = () => {
    setOpen((prev) => {
      const next = !prev;
      if (next && !revealed.current) {
        revealed.current = true;
        onReveal?.();
      }
      return next;
    });
  };

  const close = (returnFocus: boolean) => {
    setOpen(false);
    if (returnFocus) buttonRef.current?.focus();
  };

  // Esc to close (return focus to the button); outside-click to close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close(true);
      }
    };
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        close(false);
      }
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  // Move focus into the popover when it opens.
  useEffect(() => {
    if (open) dialogRef.current?.focus();
  }, [open]);

  return (
    <span
      ref={wrapRef}
      className="relative inline-flex"
      // Belt-and-suspenders: nothing inside this helper should reach the radio.
      onClick={(e) => e.stopPropagation()}
    >
      <button
        ref={buttonRef}
        type="button"
        disabled={disabled}
        aria-label={`What is a “${word}”?`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? dialogId : undefined}
        data-testid={`word-ref-${word}`}
        onClick={(e) => {
          e.stopPropagation();
          toggle();
        }}
        className={[
          "inline-flex h-7 w-7 items-center justify-center rounded-full text-neutral-400",
          "hover:text-neutral-700 hover:bg-neutral-100",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
          disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
        ].join(" ")}
      >
        <Info className="h-4 w-4" aria-hidden="true" />
      </button>

      {open && (
        <div
          ref={dialogRef}
          id={dialogId}
          role="dialog"
          aria-label={`About “${word}”`}
          tabIndex={-1}
          className={[
            "absolute right-0 top-full z-50 mt-2 w-64 rounded-lg border border-neutral-200",
            "bg-white p-3 text-left shadow-lg focus:outline-none",
          ].join(" ")}
        >
          {image && (
            <img
              src={`/api/refs/${image}`}
              alt={`Example: ${word}`}
              className="mb-2 max-h-40 w-full rounded object-cover"
            />
          )}
          <p className="text-sm font-medium text-neutral-900">{word}</p>
          <p className="mt-0.5 text-sm leading-snug text-neutral-600">{gloss}</p>
        </div>
      )}
    </span>
  );
}
