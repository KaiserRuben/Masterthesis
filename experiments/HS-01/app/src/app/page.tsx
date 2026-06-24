/**
 * Landing / consent page — SERVER component.
 *
 * Reads the frozen consent markdown (config/consent.en.md) at request time and
 * renders it with a tiny, dependency-free markdown renderer (headings, bold,
 * bullet lists, paragraphs — the only syntax the consent file uses). The
 * consent ACTION (button + already-completed gate) is a client island,
 * <ConsentGate/>, since it touches localStorage and the network.
 */

import fs from "fs";
import path from "path";

import { ConsentGate } from "@/components/ConsentGate";
import { renderConsentMarkdown } from "@/lib/markdown";

export const dynamic = "force-dynamic";

function readConsent(): string {
  const p = path.resolve(process.cwd(), "config/consent.en.md");
  try {
    return fs.readFileSync(p, "utf-8");
  } catch {
    return "# Study consent\n\nConsent text is unavailable.";
  }
}

export default function Home() {
  const md = readConsent();
  const blocks = renderConsentMarkdown(md);

  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <article className="prose-neutral">
        {blocks.map((block, i) => {
          if (block.type === "h1") {
            return (
              <h1 key={i} className="text-3xl font-semibold text-neutral-900 mb-6">
                {block.children}
              </h1>
            );
          }
          if (block.type === "ul") {
            return (
              <ul key={i} className="my-4 list-disc space-y-2 pl-6 text-neutral-700">
                {block.items.map((item, j) => (
                  <li key={j}>{item}</li>
                ))}
              </ul>
            );
          }
          return (
            <p key={i} className="my-4 leading-relaxed text-neutral-700">
              {block.children}
            </p>
          );
        })}
      </article>
      <ConsentGate />
    </main>
  );
}
