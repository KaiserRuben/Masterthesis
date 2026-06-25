import type { Config } from "tailwindcss";

/**
 * HS-01 design tokens (calm clinical instrument, TUM-derived palette).
 * Values mirror the CSS variables in src/app/globals.css so components can use
 * semantic classes (bg-tum-600, text-ink, border-line, rounded-card, shadow-card,
 * ring-tum) instead of ad-hoc default-Tailwind blue/neutral. Keep the two in sync.
 */
const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        tum: {
          50: "var(--tum-50)",
          100: "var(--tum-100)",
          200: "var(--tum-200)",
          300: "var(--tum-300)",
          400: "var(--tum-400)",
          500: "var(--tum-500)",
          600: "var(--tum-600)",
          700: "var(--tum-700)",
          900: "var(--tum-900)",
        },
        ink: "var(--ink)",
        body: "var(--body)",
        muted: "var(--muted)",
        line: "var(--line)",
        surface: "var(--surface)",
      },
      borderRadius: {
        card: "var(--radius-card)",
        control: "var(--radius-control)",
      },
      boxShadow: {
        card: "var(--shadow-card)",
        pop: "var(--shadow-pop)",
      },
      ringColor: {
        tum: "var(--tum-500)",
      },
    },
  },
  plugins: [],
};

export default config;
