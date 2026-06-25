/**
 * Instructions tests (jsdom) — the per-phase intro must show the part number it
 * is GIVEN, not a hardcoded text=1/image=2/pair=3 map. Phase order is
 * counterbalanced (image can lead), so the image-phase intro must be able to
 * read "Part 1 of 3" and the text-phase intro "Part 2 of 3". The copy stays
 * phase-keyed; only the part index is input-driven.
 */

import * as React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { Instructions } from "@/components/Instructions";

describe("Instructions", () => {
  it("labels the image-phase intro as Part 1 when image leads (image-first arm)", () => {
    render(<Instructions phase="image" part={1} onContinue={() => {}} />);
    expect(screen.getByText("Part 1 of 3")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Images" })).toBeInTheDocument();
  });

  it("labels the text-phase intro as Part 2 when text follows image (image-first arm)", () => {
    render(<Instructions phase="text" part={2} onContinue={() => {}} />);
    expect(screen.getByText("Part 2 of 3")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Questions" })).toBeInTheDocument();
  });

  it("keeps the canonical text=1/image=2 labelling for the text-first arm", () => {
    render(<Instructions phase="text" part={1} onContinue={() => {}} />);
    expect(screen.getByText("Part 1 of 3")).toBeInTheDocument();
  });
});
