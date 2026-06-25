/**
 * WordReference tests (jsdom) — the on-demand ⓘ helper for an unfamiliar
 * pair-option word.
 *
 *  - renders an ⓘ button named after the word; popover hidden initially
 *  - clicking ⓘ opens a popover with the gloss; an image entry also shows the
 *    bundled photo (/api/refs/<image>)
 *  - a gloss-only entry (image: null) shows no photo
 *  - onReveal fires once, on first open (the per-trial reveal log)
 *  - Esc closes the popover
 *  - clicking ⓘ does NOT bubble to a parent handler (so it never selects the
 *    underlying radio option)
 */

import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { WordReference } from "@/components/WordReference";

describe("WordReference", () => {
  it("renders an ⓘ button named after the word; popover hidden initially", () => {
    render(<WordReference word="tench" gloss="a freshwater fish" image="ref-tench.png" />);
    const btn = screen.getByRole("button", { name: /tench/i });
    expect(btn).toBeTruthy();
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("a freshwater fish")).toBeNull();
  });

  it("opens a popover with the gloss and the bundled photo when clicked", () => {
    render(<WordReference word="tench" gloss="a freshwater fish" image="ref-tench.png" />);
    fireEvent.click(screen.getByRole("button", { name: /tench/i }));
    expect(screen.getByText("a freshwater fish")).toBeTruthy();
    const img = screen.getByRole("img") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/api/refs/ref-tench.png");
  });

  it("shows no photo for a gloss-only entry", () => {
    render(<WordReference word="cock" gloss="an adult male chicken; a rooster" image={null} />);
    fireEvent.click(screen.getByRole("button", { name: /cock/i }));
    expect(screen.getByText(/rooster/)).toBeTruthy();
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("calls onReveal once, on first open only", () => {
    const onReveal = vi.fn();
    render(
      <WordReference word="tench" gloss="a fish" image="ref-tench.png" onReveal={onReveal} />
    );
    const btn = screen.getByRole("button", { name: /tench/i });
    fireEvent.click(btn); // open -> reveal
    fireEvent.click(btn); // close
    fireEvent.click(btn); // open again -> no second reveal
    expect(onReveal).toHaveBeenCalledTimes(1);
  });

  it("closes the popover on Escape", () => {
    render(<WordReference word="tench" gloss="a fish" image="ref-tench.png" />);
    fireEvent.click(screen.getByRole("button", { name: /tench/i }));
    expect(screen.getByText("a fish")).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByText("a fish")).toBeNull();
  });

  it("does not bubble the click to a parent handler", () => {
    const parentClick = vi.fn();
    render(
      <div onClick={parentClick}>
        <WordReference word="tench" gloss="a fish" image="ref-tench.png" />
      </div>
    );
    fireEvent.click(screen.getByRole("button", { name: /tench/i }));
    expect(parentClick).not.toHaveBeenCalled();
  });
});
