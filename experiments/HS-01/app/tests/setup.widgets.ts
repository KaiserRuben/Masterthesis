/**
 * Vitest setup — shared across suites. The jest-dom matchers and RTL cleanup
 * are only meaningful under jsdom, so we guard on the presence of a DOM. Node
 * suites import nothing extra and pay no cost.
 */
import { afterEach } from "vitest";

if (typeof document !== "undefined") {
  // jest-dom custom matchers (toBeInTheDocument, toHaveTextContent, …)
  await import("@testing-library/jest-dom/vitest");
  const { cleanup } = await import("@testing-library/react");
  afterEach(() => cleanup());
}
