/**
 * rng tests — deterministic seeded randomization (Node environment).
 *
 * Contract:
 *  - makeRng(seed): same seed => identical [0,1) sequence.
 *  - shuffle: pure (returns new array, leaves input intact), Fisher–Yates,
 *    reproducible under same seed.
 *  - optionOrder: the two word options (ANCHOR_WORD/TARGET_WORD) in
 *    rng-randomized order, then the FIXED TAIL
 *    [OTHER_CLASS, NOTHING_RECOGNIZABLE, CANT_TELL] — always last, always
 *    in that order.
 */

import { describe, it, expect } from "vitest";
import { makeRng, shuffle, optionOrder } from "../src/lib/rng";

describe("makeRng()", () => {
  it("produces a deterministic [0,1) sequence for the same seed", () => {
    const a = makeRng("session-abc");
    const b = makeRng("session-abc");
    const seqA = Array.from({ length: 10 }, () => a());
    const seqB = Array.from({ length: 10 }, () => b());
    expect(seqA).toEqual(seqB);
    // all values in [0,1)
    for (const v of seqA) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("produces different sequences for different seeds", () => {
    const a = makeRng("seed-A");
    const b = makeRng("seed-B");
    const seqA = Array.from({ length: 10 }, () => a());
    const seqB = Array.from({ length: 10 }, () => b());
    expect(seqA).not.toEqual(seqB);
  });

  it("accepts numeric seeds and is deterministic", () => {
    const a = makeRng(12345);
    const b = makeRng(12345);
    expect(Array.from({ length: 5 }, () => a())).toEqual(
      Array.from({ length: 5 }, () => b())
    );
  });
});

describe("shuffle()", () => {
  it("is reproducible for the same seed and pure (does not mutate input)", () => {
    const input = [1, 2, 3, 4, 5, 6, 7, 8];
    const r1 = makeRng("shuffle-seed");
    const r2 = makeRng("shuffle-seed");
    const out1 = shuffle(input, r1);
    const out2 = shuffle(input, r2);

    expect(out1).toEqual(out2);
    // input untouched
    expect(input).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    // new array, same multiset
    expect(out1).not.toBe(input);
    expect([...out1].sort((a, b) => a - b)).toEqual(input);
  });

  it("produces a different permutation under a different seed", () => {
    const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const a = shuffle(input, makeRng("X"));
    const b = shuffle(input, makeRng("Y"));
    expect(a).not.toEqual(b);
    // still a permutation
    expect([...a].sort((x, y) => x - y)).toEqual(input);
    expect([...b].sort((x, y) => x - y)).toEqual(input);
  });
});

describe("optionOrder()", () => {
  const TAIL = ["OTHER_CLASS", "NOTHING_RECOGNIZABLE", "CANT_TELL"] as const;

  it("returns exactly 5 options: two words first, fixed tail last", () => {
    const order = optionOrder(makeRng("pair-1"));
    expect(order).toHaveLength(5);

    // first two are the word options (some order)
    const head = order.slice(0, 2);
    expect([...head].sort()).toEqual(["ANCHOR_WORD", "TARGET_WORD"]);

    // tail is fixed and in fixed order
    expect(order.slice(2)).toEqual([...TAIL]);
  });

  it("is deterministic for the same seed", () => {
    const o1 = optionOrder(makeRng("pair-seed-77"));
    const o2 = optionOrder(makeRng("pair-seed-77"));
    expect(o1).toEqual(o2);
  });

  it("randomizes the head order across seeds (both AB orders reachable)", () => {
    const heads = new Set<string>();
    for (let i = 0; i < 50; i++) {
      const o = optionOrder(makeRng(`s-${i}`));
      heads.add(o.slice(0, 2).join(","));
      // tail invariant holds for every seed
      expect(o.slice(2)).toEqual([...TAIL]);
    }
    expect(heads.has("ANCHOR_WORD,TARGET_WORD")).toBe(true);
    expect(heads.has("TARGET_WORD,ANCHOR_WORD")).toBe(true);
  });
});
