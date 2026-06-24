/**
 * rng.ts — deterministic seeded randomization for trial / option ordering.
 *
 * This is the ONLY randomness source for presentation order. No Math.random()
 * / Date.now() may touch trial or option ordering — given a seed (logged in the
 * session record as rng_seed) the entire order must be reproducible in analysis.
 *
 * - makeRng(seed)   mulberry32 PRNG, string seeds hashed via xmur3.
 * - shuffle(arr,rng) pure Fisher–Yates (returns a new array).
 * - optionOrder(rng) the two word options in randomized order, then the FIXED
 *   TAIL [OTHER_CLASS, NOTHING_RECOGNIZABLE, CANT_TELL] — always last.
 */

import type { SemanticChoice } from "./types";

/** The two semantic word slots whose on-screen order is AB-randomized. */
export type SemanticOption = SemanticChoice;

/** Tail options that are NEVER shuffled and always appear last, in this order. */
const FIXED_TAIL: readonly SemanticOption[] = [
  "OTHER_CLASS",
  "NOTHING_RECOGNIZABLE",
  "CANT_TELL",
] as const;

/**
 * xmur3 string-hash → seed generator. Produces a 32-bit unsigned integer to
 * seed mulberry32. Deterministic per string.
 */
function xmur3(str: string): number {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^= h >>> 16) >>> 0;
}

/**
 * makeRng — deterministic [0,1) generator (mulberry32). Same seed ⇒ identical
 * sequence. Numeric seeds are normalized to a 32-bit unsigned integer; string
 * seeds are hashed with xmur3.
 */
export function makeRng(seed: string | number): () => number {
  let a: number;
  if (typeof seed === "number") {
    // Normalize to uint32; mix so small/zero integers don't degenerate.
    a = (seed >>> 0) || xmur3(String(seed));
  } else {
    a = xmur3(seed);
  }
  return function mulberry32(): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * shuffle — pure Fisher–Yates. Returns a NEW array; the input is not mutated.
 * Consumes one rng() draw per swap, so the permutation is reproducible for a
 * given seed.
 */
export function shuffle<T>(arr: readonly T[], rng: () => number): T[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/**
 * optionOrder — realized pair-option order for one trial.
 *
 * The two word options (ANCHOR_WORD / TARGET_WORD) are shuffled by the rng, then
 * the fixed tail [OTHER_CLASS, NOTHING_RECOGNIZABLE, CANT_TELL] is appended
 * unchanged. Logged per trial as presented.option_display_order.
 */
export function optionOrder(rng: () => number): SemanticOption[] {
  const head = shuffle<SemanticOption>(["ANCHOR_WORD", "TARGET_WORD"], rng);
  return [...head, ...FIXED_TAIL];
}
