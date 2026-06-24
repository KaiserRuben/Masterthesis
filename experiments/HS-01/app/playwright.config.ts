import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config — the LIVE validation harness for the HS-01 study app.
 *
 * No auto-webServer: the controller starts the dev server (`npm run dev`, port
 * 3939) before invoking `npm run e2e`. The spec drives the entire rater flow in
 * a real browser and then validates the persisted session record on disk
 * against the frozen JSON schema, so it catches integration breakage the
 * vitest unit suite cannot (real timing, real image decode, real fs writes).
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3939",
    trace: "on-first-retry",
    // The viewport must comfortably exceed the study's
    // min_rendered_image_css_px (256) so the image/pair phases are not gated by
    // the too-small refusal screen.
    viewport: { width: 1440, height: 1000 },
    // Honor reduced motion in the browser so framer-motion (configured with
    // reducedMotion="user") skips transform animations. This removes the
    // position-animation that otherwise makes controls read as "not stable"
    // and lets clicks land on the first attempt.
    contextOptions: { reducedMotion: "reduce" },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
