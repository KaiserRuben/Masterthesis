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
    <main className="mx-auto max-w-xl px-6 py-12 sm:py-16">
      <div className="rounded-card border border-line bg-white p-6 shadow-card sm:p-8">
        <article>
          {blocks.map((block, i) => {
            if (block.type === "h1") {
              return (
                <h1 key={i} className="mb-6 text-3xl font-semibold tracking-tight text-ink">
                  {block.children}
                </h1>
              );
            }
            if (block.type === "ul") {
              return (
                <ul
                  key={i}
                  className="my-5 list-disc space-y-2.5 pl-5 leading-relaxed text-body marker:text-tum-500"
                >
                  {block.items.map((item, j) => (
                    <li key={j} className="pl-1">
                      {item}
                    </li>
                  ))}
                </ul>
              );
            }
            return (
              <p key={i} className="my-4 leading-relaxed text-body">
                {block.children}
              </p>
            );
          })}
        </article>
        <div className="mt-8 border-t border-line pt-6">
          <ConsentGate />
        </div>
      </div>
    </main>
  );
}
