/**
 * full-session.spec.ts — the LIVE end-to-end validation harness for HS-01.
 *
 * Drives the entire rater flow in a real browser, then re-reads the persisted
 * session record off disk and validates it against the FROZEN session schema.
 * This is the integration check the vitest unit suite cannot perform: real
 * timing (performance.now onsets), real image decode (css == natural, no
 * upscaling), the per-trial AB shuffle, the attention-check tallies, and the
 * atomic fs write all have to line up for the schema to pass.
 *
 * Assumes the dev server is already running on http://localhost:3939 (the
 * controller starts it; playwright.config has no webServer block).
 */

import { test, expect, type Page } from "@playwright/test";
import fs from "fs";
import path from "path";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";

// ─── on-disk locations ────────────────────────────────────────────────────────

const APP_DIR = path.resolve(__dirname, "..");
const DATA_DIR =
  process.env.HS01_DATA_DIR ?? path.join(APP_DIR, "data", "sessions");
const SCHEMA_PATH = path.resolve(
  APP_DIR,
  "../schemas/hs01.session.schema.json"
);

// Compile the session validator exactly like the app (draft 2020-12 + formats).
function buildValidator() {
  const ajv = new Ajv2020({ strict: false, allErrors: true });
  addFormats(ajv);
  const schema = JSON.parse(fs.readFileSync(SCHEMA_PATH, "utf-8"));
  return ajv.compile(schema);
}

/** Newest *.json session record under DATA_DIR (skips the _counter file). */
function newestSessionFile(): string {
  const files = fs
    .readdirSync(DATA_DIR)
    .filter((f) => f.endsWith(".json") && !f.startsWith("_"))
    .map((f) => {
      const full = path.join(DATA_DIR, f);
      return { full, mtime: fs.statSync(full).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  if (files.length === 0) throw new Error(`no session files in ${DATA_DIR}`);
  return files[0].full;
}

function readNewestRecord(): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(newestSessionFile(), "utf-8"));
}

/**
 * Count of trials in the IN-PROGRESS record held in the browser's localStorage.
 *
 * This is the per-trial durability layer: the app persists each answered trial
 * to localStorage immediately, but only checkpoints to the server (disk) at
 * phase boundaries and on final submit. So a mid-phase trial count must be read
 * from localStorage — the on-disk record still reflects the last phase exit
 * (or the empty create-time record). Resume itself rehydrates from this store.
 */
async function lsTrialCount(page: Page): Promise<number> {
  return page.evaluate(() => {
    const raw = window.localStorage.getItem("hs01:record");
    if (!raw) return 0;
    try {
      return (JSON.parse(raw).trials ?? []).length;
    } catch {
      return 0;
    }
  });
}

// ─── flow helpers ───────────────────────────────────────────────────────────

/** Click the consent CTA and land in the study runner. */
async function consentAndBegin(page: Page) {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: /I consent and want to begin/i })
  ).toBeVisible();
  await page.getByTestId("consent-begin").click();
  await page.waitForURL(/\/study/);
}

/**
 * Wait for the study runner to leave its "Loading…" state and settle on a real
 * screen: an instructions screen, a judgment trial, or the demographics form.
 *
 * This matters because `waitForURL(/study/)` resolves on navigation, but the
 * client then spends a beat in the "loading" phase rehydrating localStorage
 * before it decides what to show. A decision made during that window (e.g.
 * "are instructions present?") races the loading screen and answers "no" for a
 * screen that is about to appear.
 */
async function waitForStudyReady(page: Page) {
  await page
    .locator(
      [
        '[data-testid="instructions-continue"]',
        '[data-testid="likert-point-3"]',
        '[data-testid="pair-option-ANCHOR_WORD"]',
        '[data-testid="demographics-submit"]',
      ].join(", ")
    )
    .first()
    .waitFor({ state: "visible", timeout: 15_000 })
    .catch(() => {});
}

/**
 * If a phase-instructions screen is up, click through it.
 *
 * Waits for the runner to settle first (so the instructions-vs-trial decision
 * is never made on the loading screen). The continue button is server-rendered,
 * so a click dispatched before React has hydrated its onClick handler is a
 * no-op (the button stays put). Rather than guess at a hydration delay, we
 * retry the click until the button actually detaches — the only reliable signal
 * that `beginPhase` ran and the screen advanced. Returns true if an
 * instructions screen was passed.
 */
async function passInstructionsIfPresent(page: Page): Promise<boolean> {
  await waitForStudyReady(page);
  const begin = page.getByTestId("instructions-continue");
  if (!(await begin.isVisible().catch(() => false))) return false;
  for (let i = 0; i < 20; i++) {
    await begin.click().catch(() => {});
    const gone = await begin
      .waitFor({ state: "detached", timeout: 500 })
      .then(() => true)
      .catch(() => false);
    if (gone) return true;
  }
  return false;
}

/** True if a judgment trial is currently presented (a Likert point or pair option is on screen). */
async function trialPresent(page: Page): Promise<boolean> {
  const pairAnchor = page.getByTestId("pair-option-ANCHOR_WORD");
  const likert = page.getByTestId("likert-point-3");
  return (
    (await pairAnchor.isVisible().catch(() => false)) ||
    (await likert.isVisible().catch(() => false))
  );
}

/** Covers the auto-advance highlight window (AUTO_ADVANCE_MS in the app) plus
 *  the next-trial mount + cross-fade, so the walker never acts on the outgoing
 *  trial after committing one. */
const ADVANCE_WAIT_MS = 450;

/**
 * Answer one judgment trial. Likert phases pick a scale point; pair phases pick
 * a semantic option. `pairSlot` lets the caller force a specific slot (used to
 * trigger OTHER_CLASS + its free-text field at least once). Returns the phase
 * kind that was answered, or null if no trial was present.
 *
 * There is NO Next button: response options are disabled until onset fires, and
 * a pick AUTO-ADVANCES after a brief highlight. So we wait for the option to be
 * enabled (this is where onset/decode is awaited), pick it, then wait out the
 * highlight window. The lone exception is OTHER_CLASS + free text, which reveals
 * a Confirm button instead of auto-advancing.
 */
async function answerOneTrial(
  page: Page,
  opts: { likertPoint?: number; pairSlot?: string; otherText?: string } = {}
): Promise<"likert" | "pair" | null> {
  // A pair trial exposes the ANCHOR_WORD option; a likert trial exposes points.
  const pairAnchor = page.getByTestId("pair-option-ANCHOR_WORD");
  const isPair = await pairAnchor.isVisible().catch(() => false);

  if (isPair) {
    const slot = opts.pairSlot ?? "ANCHOR_WORD";
    const opt = page.getByTestId(`pair-option-${slot}`);
    // Options are disabled until onset/decode fires; wait for it, then pick.
    await expect(opt).toBeEnabled({ timeout: 15_000 });
    await opt.click();
    if (slot === "OTHER_CLASS") {
      const free = page.getByTestId("other-class-text");
      await expect(free).toBeVisible();
      await free.fill(opts.otherText ?? "a different animal");
      // "Something else" does NOT auto-advance — confirm explicitly.
      const confirm = page.getByTestId("trial-confirm");
      await expect(confirm).toBeEnabled({ timeout: 15_000 });
      await confirm.click();
    }
    await page.waitForTimeout(ADVANCE_WAIT_MS);
    return "pair";
  }

  const point = opts.likertPoint ?? 3;
  const likert = page.getByTestId(`likert-point-${point}`);
  if (await likert.isVisible().catch(() => false)) {
    await expect(likert).toBeEnabled({ timeout: 15_000 });
    await likert.click();
    await page.waitForTimeout(ADVANCE_WAIT_MS);
    return "likert";
  }
  return null;
}

/**
 * Walk the ENTIRE judgment flow to demographics. Triggers OTHER_CLASS on the
 * first pair trial so the free-text branch is exercised at least once. Stops
 * when the demographics form appears. Bounded loop so a stuck flow fails fast.
 */
async function walkToDemographics(page: Page) {
  let firstPairDone = false;
  for (let i = 0; i < 500; i++) {
    // Demographics form? done walking.
    if (await page.getByTestId("demographics-submit").isVisible().catch(() => false)) {
      return;
    }
    await passInstructionsIfPresent(page);
    if (!(await trialPresent(page))) {
      // Could be an instructions screen that just appeared, or demographics.
      if (await page.getByTestId("demographics-submit").isVisible().catch(() => false)) {
        return;
      }
      await passInstructionsIfPresent(page);
      // Give the next trial a beat to mount.
      await page.waitForTimeout(50);
      continue;
    }

    // Decide the pair answer: force OTHER_CLASS once to cover that branch.
    const isPair = await page
      .getByTestId("pair-option-ANCHOR_WORD")
      .isVisible()
      .catch(() => false);
    const pairSlot = isPair && !firstPairDone ? "OTHER_CLASS" : "ANCHOR_WORD";
    const answered = await answerOneTrial(page, { likertPoint: 3, pairSlot });
    if (answered === "pair" && pairSlot === "OTHER_CLASS") firstPairDone = true;
  }
  throw new Error("walkToDemographics: exceeded trial bound without reaching demographics");
}

/** Fill the required demographics selects + an optional comment, then submit. */
async function fillDemographicsAndSubmit(page: Page) {
  await expect(page.getByTestId("demographics-submit")).toBeVisible();
  await page.locator("#demo-age_band").selectOption("25_34");
  await page.locator("#demo-ml_familiarity").selectOption("some_exposure");
  await page.locator("#demo-english_proficiency").selectOption("C1");
  // Optional free-text comment field (field_id "comment").
  const comment = page.locator("#demo-comment");
  if (await comment.isVisible().catch(() => false)) {
    await comment.fill("e2e run");
  }
  await expect(page.getByTestId("demographics-submit")).toBeEnabled();
  await page.getByTestId("demographics-submit").click();
}

// ─── Test 1: full flow → schema-valid completed record ────────────────────────

test("walks the entire flow and persists a schema-valid completed record", async ({
  page,
}) => {
  await consentAndBegin(page);
  await walkToDemographics(page);
  await fillDemographicsAndSubmit(page);

  // Reaches /done on a confirmed valid submit.
  await page.waitForURL(/\/done/, { timeout: 20_000 });

  const rec = readNewestRecord();
  const validate = buildValidator();
  const ok = validate(rec);
  if (!ok) {
    // Surface schema errors directly in the failure for fast triage.
    throw new Error(
      "session record failed schema validation:\n" +
        JSON.stringify(validate.errors, null, 2)
    );
  }
  expect(ok).toBe(true);

  // ── status + identity ──
  expect(rec.status).toBe("completed");

  // ── timing monotonicity per trial + non-decreasing offsets across trials ──
  const trials = rec.trials as Array<Record<string, any>>;
  expect(trials.length).toBeGreaterThan(0);
  let prevOffset = -1;
  for (const t of trials) {
    const onset = t.timing.onset_ms as number;
    const submitted = t.timing.submitted_ms as number;
    expect(onset).toBeLessThan(submitted); // onset strictly before submit
    expect(onset).toBeGreaterThanOrEqual(prevOffset); // offsets non-decreasing
    prevOffset = onset;
  }

  // ── pair trials: display order (5 entries, 3 non-word slots last) +
  //    a semantic choice + native-size rendered image ──
  const NON_WORD = new Set(["OTHER_CLASS", "NOTHING_RECOGNIZABLE", "CANT_TELL"]);
  const SEMANTIC = new Set([
    "ANCHOR_WORD",
    "TARGET_WORD",
    "OTHER_CLASS",
    "NOTHING_RECOGNIZABLE",
    "CANT_TELL",
  ]);
  const pairTrials = trials.filter((t) => t.phase_id === "pair");
  expect(pairTrials.length).toBeGreaterThan(0);
  for (const t of pairTrials) {
    const order = t.presented.option_display_order as string[];
    expect(order).toHaveLength(5);
    // The three non-word options are always the last three.
    const tail = order.slice(2);
    expect(tail.every((s) => NON_WORD.has(s))).toBe(true);
    expect(order.slice(0, 2).every((s) => !NON_WORD.has(s))).toBe(true);
    // Response choice is a semantic slot, not a positional index.
    expect(SEMANTIC.has(t.response.choice)).toBe(true);
  }

  // ── image & pair trials: rendered at native size (no upscaling) ──
  const stimulusTrials = trials.filter(
    (t) => t.phase_id === "image" || t.phase_id === "pair"
  );
  expect(stimulusTrials.length).toBeGreaterThan(0);
  for (const t of stimulusTrials) {
    const ri = t.presented.rendered_image;
    expect(ri).toBeTruthy();
    // No UPSCALING: css never exceeds natural. On a viewport narrower than the
    // image the stimulus DOWNSCALES to fit (rather than overflow), so css may be
    // < natural with the aspect ratio preserved.
    expect(ri.css_w).toBeLessThanOrEqual(ri.natural_w);
    expect(ri.css_h).toBeLessThanOrEqual(ri.natural_h);
    expect(
      Math.abs(ri.css_w / ri.css_h - ri.natural_w / ri.natural_h)
    ).toBeLessThan(0.02);
  }

  // ── attention checks: exactly 2 (one text, one pair) ──
  expect((rec.quality_summary as any).attention_total).toBe(2);

  // ── demographics populated ──
  const demo = rec.demographics as Record<string, unknown>;
  expect(demo).toBeTruthy();
  expect(demo.age_band).toBe("25_34");
  expect(demo.ml_familiarity).toBe("some_exposure");
  expect(demo.english_proficiency).toBe("C1");

  // ── render_check passed (captured at consent) ──
  expect((rec.environment as any).render_check.passed).toBe(true);

  // ── all phases present in phase_timings (text, image, pair, demographics) ──
  const phaseIds = new Set(
    (rec.phase_timings as Array<Record<string, unknown>>).map((p) => p.phase_id)
  );
  for (const id of ["text", "image", "pair", "demographics"]) {
    expect(phaseIds.has(id)).toBe(true);
  }
});

// ─── Test 2: reload mid-study resumes on an unanswered trial ──────────────────

test("reload mid-study resumes on an unanswered trial; final record validates", async ({
  page,
}) => {
  await consentAndBegin(page);

  // Answer a couple of text trials, then reload mid-study.
  await passInstructionsIfPresent(page);
  await expect(page.getByTestId("likert-point-4")).toBeVisible({ timeout: 10_000 });
  await answerOneTrial(page, { likertPoint: 4 });
  await expect(page.getByTestId("likert-point-2")).toBeVisible({ timeout: 10_000 });
  await answerOneTrial(page, { likertPoint: 2 });

  // Record the count of answered trials before reload (from localStorage, the
  // per-trial durability layer — disk only updates at phase exit / submit).
  const before = await lsTrialCount(page);
  expect(before).toBeGreaterThanOrEqual(2);

  await page.reload();
  await page.waitForURL(/\/study/);

  // Resume must land on a live, unanswered trial (a response widget), NOT replay
  // an already-answered one — and not jump to /done.
  await passInstructionsIfPresent(page);
  expect(await trialPresent(page)).toBe(true);
  // No new trial was appended just by reloading.
  expect(await lsTrialCount(page)).toBe(before);

  // Finish the run; the submitted record must still validate with all phases.
  await walkToDemographics(page);
  await fillDemographicsAndSubmit(page);
  await page.waitForURL(/\/done/, { timeout: 20_000 });

  const rec = readNewestRecord();
  const validate = buildValidator();
  expect(validate(rec)).toBe(true);
  const phaseIds = new Set(
    (rec.phase_timings as Array<Record<string, unknown>>).map((p) => p.phase_id)
  );
  for (const id of ["text", "image", "pair", "demographics"]) {
    expect(phaseIds.has(id)).toBe(true);
  }
});

// ─── Test 3: FIX 1 — HTTP 200 + body.ok:false must NOT complete ───────────────

test("a schema-invalid submit (200 + ok:false) shows submit-failed, never /done", async ({
  page,
}) => {
  // Intercept the submit POST and force the server's "schema-invalid" reply:
  // HTTP 200 with { ok:false, validation_errors:[...] }. The app must NOT treat
  // this as success — no /done, no LS_COMPLETED, retry panel shown.
  await page.route("**/api/sessions/*/submit", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: false, validation_errors: [{ msg: "forced" }] }),
    });
  });

  await consentAndBegin(page);
  await walkToDemographics(page);
  await fillDemographicsAndSubmit(page);

  // The submit-failed retry panel must appear; /done must NOT be reached.
  await expect(page.getByTestId("submit-failed")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("retry-submit")).toBeVisible();
  expect(page.url()).not.toMatch(/\/done/);

  // LS_COMPLETED must NOT be set (participant is not locked out).
  const completed = await page.evaluate(() =>
    window.localStorage.getItem("hs01:completed")
  );
  expect(completed).toBeNull();
  // LS_RECORD must be retained so retry can re-submit the same record.
  const recordKept = await page.evaluate(() =>
    window.localStorage.getItem("hs01:record")
  );
  expect(recordKept).not.toBeNull();
});

// ─── Test 4: word-reference ⓘ helper — popover + reveal logging ───────────────

test("opens a pair-option ⓘ helper; the reveal is logged and the record validates", async ({
  page,
}) => {
  await consentAndBegin(page);

  let openedTestId: string | null = null;

  // Walk to demographics; on the first pair trial that offers an ⓘ helper, open
  // it, verify the in-page popover (and that /api/refs serves a real image when
  // present), close it with Escape, then answer the trial normally.
  for (let i = 0; i < 500; i++) {
    if (await page.getByTestId("demographics-submit").isVisible().catch(() => false)) {
      break;
    }
    await passInstructionsIfPresent(page);
    if (!(await trialPresent(page))) {
      await page.waitForTimeout(50);
      continue;
    }

    const isPair = await page
      .getByTestId("pair-option-ANCHOR_WORD")
      .isVisible()
      .catch(() => false);

    if (isPair && !openedTestId) {
      const info = page.locator('[data-testid^="word-ref-"]').first();
      if (await info.isVisible().catch(() => false)) {
        openedTestId = await info.getAttribute("data-testid");
        await info.click();
        const dialog = page.getByRole("dialog");
        await expect(dialog).toBeVisible();
        // If this entry has a photo, it must actually load via /api/refs.
        const img = dialog.locator("img");
        if (await img.count()) {
          await expect
            .poll(async () =>
              img.first().evaluate((el) => (el as HTMLImageElement).naturalWidth)
            )
            .toBeGreaterThan(0);
        }
        await page.keyboard.press("Escape");
        await expect(dialog).toBeHidden();
        // Opening the helper must NOT have selected the option.
        await expect(
          page.getByTestId("pair-option-ANCHOR_WORD")
        ).toHaveAttribute("aria-checked", "false");
      }
    }

    await answerOneTrial(page, { likertPoint: 3, pairSlot: "ANCHOR_WORD" });
  }

  await fillDemographicsAndSubmit(page);
  await page.waitForURL(/\/done/, { timeout: 20_000 });

  // We must have found at least one glossed pair option.
  expect(openedTestId).not.toBeNull();

  // The reveal is persisted on the pair trial whose word we opened, and the
  // whole record still validates against the (1.1.0) session schema.
  const rec = readNewestRecord();
  const trials = (rec.trials ?? []) as Array<Record<string, any>>;
  const withReveals = trials.filter(
    (t) =>
      t.phase_id === "pair" &&
      Array.isArray(t.response?.references_revealed) &&
      t.response.references_revealed.length > 0
  );
  expect(withReveals.length).toBeGreaterThan(0);
  for (const t of withReveals) {
    for (const slot of t.response.references_revealed) {
      expect(["ANCHOR_WORD", "TARGET_WORD"]).toContain(slot);
    }
  }

  const validate = buildValidator();
  const ok = validate(rec);
  if (!ok) console.error(validate.errors);
  expect(ok).toBe(true);
  expect(rec.schema_version).toBe("1.1.0");
});
