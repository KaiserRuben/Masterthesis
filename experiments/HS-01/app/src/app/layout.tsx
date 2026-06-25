import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { MotionProvider } from "@/components/MotionProvider";

/**
 * Stimulus-prompt font — JetBrains Mono, SELF-HOSTED via next/font/local.
 *
 * next/font/local copies these woff2 files into the build's static output and
 * emits a local @font-face (no runtime fetch to any external host), so it works
 * offline and under the deployment's strict CSP. The vendored subsets live in
 * public/fonts/ (source: the @fontsource/jetbrains-mono npm package).
 *
 * WHY this font: it is a true monospace whose Latin AND Cyrillic glyphs share
 * the same advance metrics and that covers all 7 Cyrillic homoglyph codepoints
 * in the pool (U+0430 U+0435 U+0441 U+043E U+0440 U+0455 U+0456). With it
 * bundled, those homoglyphs render as their subtle Latin twins instead of
 * triggering per-glyph script fallback to a heavier/taller proportional font
 * (the bold/larger "tell" raters on phones/Linux/minimal-Docker would see).
 * See tests/prompt-font.test.ts for the coverage + metric guard.
 *
 * The CSS variable is consumed by --font-prompt / .font-prompt in globals.css.
 */
const promptFont = localFont({
  src: [
    {
      path: "../../public/fonts/jetbrains-mono-latin-400-normal.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "../../public/fonts/jetbrains-mono-cyrillic-400-normal.woff2",
      weight: "400",
      style: "normal",
    },
  ],
  display: "swap",
  variable: "--font-jetbrains-mono",
  // Keep verbatim fidelity: no glyph-altering transforms. The local fallback
  // metric-overrides next/font would otherwise synthesize are unnecessary here
  // because the bundled face itself covers the homoglyphs.
  fallback: ["ui-monospace", "Menlo", "Consolas", "Courier New", "monospace"],
});

export const metadata: Metadata = {
  title: "HS-01 Human Validity Study",
  description: "VLM boundary item human validity study",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={promptFont.variable}>
      <body>
        <MotionProvider>{children}</MotionProvider>
      </body>
    </html>
  );
}
