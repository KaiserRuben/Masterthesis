/**
 * Widget tests (jsdom) — measurement-correctness of the two response widgets.
 *
 * LikertScale:
 *  - emits the chosen 1–5 value on click
 *  - reports n_changes that increments on every *change* of answer (re-clicking
 *    the same value does not increment).
 *
 * PairChoice (the load-bearing semantic-vs-positional guarantee):
 *  - given a FORCED option order where TARGET_WORD renders FIRST, clicking the
 *    ANCHOR word still returns choice:"ANCHOR_WORD" (semantic, not positional).
 *  - the three tail options (OTHER_CLASS / NOTHING_RECOGNIZABLE / CANT_TELL)
 *    always render LAST, after the two word buttons.
 *  - selecting OTHER_CLASS reveals the optional free-text field.
 */

import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { LikertScale } from "@/components/LikertScale";
import { PairChoice } from "@/components/PairChoice";
import type { Scale, SemanticChoice } from "@/lib/types";

const SCALE: Scale = {
  scale_id: "text-comprehensibility-v1",
  applies_to: "text",
  statement: "I can tell what this question is asking",
  points: 5,
  point_labels: [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
  ],
};

describe("LikertScale", () => {
  it("emits the chosen 1–5 value on click", () => {
    const onChange = vi.fn();
    render(<LikertScale scale={SCALE} value={null} onChange={onChange} />);

    fireEvent.click(screen.getByRole("radio", { name: /^Agree$/i }));

    expect(onChange).toHaveBeenCalled();
    const [value] = onChange.mock.calls[onChange.mock.calls.length - 1];
    expect(value).toBe(4);
  });

  it("increments n_changes on every change of answer, not on re-selecting the same", () => {
    const onChange = vi.fn();

    function Harness() {
      const [v, setV] = React.useState<number | null>(null);
      return (
        <LikertScale
          scale={SCALE}
          value={v}
          onChange={(value: number, nChanges: number) => {
            setV(value);
            onChange(value, nChanges);
          }}
        />
      );
    }
    render(<Harness />);

    const get = (n: number) =>
      screen.getByRole("radio", {
        name: new RegExp(`^${SCALE.point_labels[n - 1]}$`, "i"),
      });

    fireEvent.click(get(1)); // first selection: n_changes stays 0
    fireEvent.click(get(3)); // change -> 1
    fireEvent.click(get(3)); // same -> still 1
    fireEvent.click(get(5)); // change -> 2

    const calls = onChange.mock.calls;
    expect(calls[0]).toEqual([1, 0]);
    expect(calls[1]).toEqual([3, 1]);
    expect(calls[2]).toEqual([3, 1]);
    expect(calls[3]).toEqual([5, 2]);
  });
});

describe("PairChoice", () => {
  const optionLabels = { ANCHOR_WORD: "green iguana", TARGET_WORD: "boa constrictor" };
  const displayLabels = {
    OTHER_CLASS: "Something else",
    NOTHING_RECOGNIZABLE: "Nothing recognizable",
    CANT_TELL: "I can't tell",
  };
  // FORCED order: TARGET_WORD first, then ANCHOR_WORD, then the fixed tail.
  const forcedOrder: SemanticChoice[] = [
    "TARGET_WORD",
    "ANCHOR_WORD",
    "OTHER_CLASS",
    "NOTHING_RECOGNIZABLE",
    "CANT_TELL",
  ];

  function renderPair(
    overrides: Partial<React.ComponentProps<typeof PairChoice>> = {}
  ) {
    const props: React.ComponentProps<typeof PairChoice> = {
      optionLabels,
      displayLabels,
      displayOrder: forcedOrder,
      otherClassFreeText: true,
      value: null,
      otherClassText: "",
      onChange: vi.fn(),
      onOtherClassText: vi.fn(),
      ...overrides,
    };
    return { ...render(<PairChoice {...props} />), props };
  }

  it("returns choice ANCHOR_WORD when the anchor WORD is clicked, even though TARGET renders first", () => {
    const { props } = renderPair();
    // The anchor word is rendered second on screen, but selecting it must yield
    // the SEMANTIC slot, never a positional index.
    fireEvent.click(screen.getByRole("radio", { name: "green iguana" }));
    expect(props.onChange).toHaveBeenCalled();
    const [choice] = (props.onChange as ReturnType<typeof vi.fn>).mock.calls.at(-1)!;
    expect(choice).toBe("ANCHOR_WORD");
  });

  it("renders the three tail options last, after the two word buttons", () => {
    renderPair();
    const radios = screen.getAllByRole("radio");
    const labels = radios.map((r) => r.textContent?.trim());
    expect(labels).toEqual([
      "boa constrictor", // TARGET (first per forced order)
      "green iguana", // ANCHOR
      "Something else",
      "Nothing recognizable",
      "I can't tell",
    ]);
  });

  it("reveals the optional free-text field only when OTHER_CLASS is selected", () => {
    const { props, rerender } = renderPair();
    expect(screen.queryByRole("textbox")).toBeNull();

    fireEvent.click(screen.getByRole("radio", { name: "Something else" }));
    const [choice] = (props.onChange as ReturnType<typeof vi.fn>).mock.calls.at(-1)!;
    expect(choice).toBe("OTHER_CLASS");

    // PairChoice is controlled: re-render with value=OTHER_CLASS reveals it.
    rerender(
      <PairChoice
        optionLabels={optionLabels}
        displayLabels={displayLabels}
        displayOrder={forcedOrder}
        otherClassFreeText={true}
        value="OTHER_CLASS"
        otherClassText=""
        onChange={vi.fn()}
        onOtherClassText={vi.fn()}
      />
    );
    expect(screen.getByRole("textbox")).toBeTruthy();
  });
});
