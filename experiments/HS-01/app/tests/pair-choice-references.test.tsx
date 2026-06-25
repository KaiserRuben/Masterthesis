/**
 * PairChoice × WordReference integration (jsdom).
 *
 * The ⓘ helper is wired onto the two WORD slots only, and only when a reference
 * entry exists for that word. The load-bearing guarantee: opening the helper is
 * NOT a response — it must not select the option and must not bump n_changes.
 */

import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { PairChoice } from "@/components/PairChoice";
import type { ReferenceEntry, SemanticChoice } from "@/lib/types";

const optionLabels = { ANCHOR_WORD: "tench", TARGET_WORD: "box turtle" };
const displayLabels = {
  OTHER_CLASS: "Something else",
  NOTHING_RECOGNIZABLE: "Nothing recognizable",
  CANT_TELL: "I can't tell",
};
const order: SemanticChoice[] = [
  "ANCHOR_WORD",
  "TARGET_WORD",
  "OTHER_CLASS",
  "NOTHING_RECOGNIZABLE",
  "CANT_TELL",
];
const references: Record<string, ReferenceEntry> = {
  tench: { gloss: "a freshwater fish", image: "ref-tench.png" },
  // intentionally NO entry for "box turtle"
};

function renderPair(overrides: Partial<React.ComponentProps<typeof PairChoice>> = {}) {
  const props: React.ComponentProps<typeof PairChoice> = {
    optionLabels,
    displayLabels,
    displayOrder: order,
    otherClassFreeText: true,
    value: null,
    otherClassText: "",
    onChange: vi.fn(),
    onOtherClassText: vi.fn(),
    references,
    onReveal: vi.fn(),
    ...overrides,
  };
  return { ...render(<PairChoice {...props} />), props };
}

describe("PairChoice references", () => {
  it("shows an ⓘ only for word slots that have a reference entry", () => {
    renderPair();
    expect(screen.queryByRole("button", { name: /what is a .*tench/i })).toBeTruthy();
    // box turtle has no entry -> no helper
    expect(screen.queryByRole("button", { name: /what is a .*box turtle/i })).toBeNull();
  });

  it("shows no ⓘ on the tail options", () => {
    renderPair();
    expect(
      screen.queryByRole("button", { name: /what is a .*something else/i })
    ).toBeNull();
  });

  it("opening the helper does NOT select the option or change n_changes", () => {
    const onChange = vi.fn();
    renderPair({ onChange });
    fireEvent.click(screen.getByRole("button", { name: /what is a .*tench/i }));
    // popover opened…
    expect(screen.getByText("a freshwater fish")).toBeTruthy();
    // …but no selection happened
    expect(onChange).not.toHaveBeenCalled();
  });

  it("fires onReveal with the word slot when the helper is opened", () => {
    const onReveal = vi.fn();
    renderPair({ onReveal });
    fireEvent.click(screen.getByRole("button", { name: /what is a .*tench/i }));
    expect(onReveal).toHaveBeenCalledWith("ANCHOR_WORD");
  });

  it("still selects the option when the WORD itself is clicked", () => {
    const onChange = vi.fn();
    renderPair({ onChange });
    fireEvent.click(screen.getByRole("radio", { name: "tench" }));
    const [choice] = onChange.mock.calls.at(-1)!;
    expect(choice).toBe("ANCHOR_WORD");
  });
});
