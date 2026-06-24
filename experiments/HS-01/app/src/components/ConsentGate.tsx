"use client";

/**
 * ConsentGate — the single consent button + the "already completed" block.
 *
 * On click:
 *  1. If localStorage holds the `completed` flag → BLOCK re-entry (one response
 *     per participant). Show a thank-you and do nothing else.
 *  2. Otherwise: capture the environment (UA, viewport, screen, dpr, is_touch),
 *     run renderCheck() (stimulus-fidelity guard), POST /api/sessions, build the
 *     initial SessionRecord (initRecord), persist BOTH the create-response and
 *     the record to localStorage, and route to /study.
 *
 * The consent TEXT is rendered by the server page; this island is only the
 * action + the completed-gate.
 */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { renderCheck } from "@/lib/instrumentation";
import { initRecord } from "@/lib/session-record";
import type { CreateResult } from "@/lib/store";
import type { SessionRecord } from "@/lib/types";
import { LS_COMPLETED, LS_CREATE, LS_RECORD } from "@/state/useSession";

function captureEnvironment(): SessionRecord["environment"] {
  const rc = renderCheck();
  return {
    user_agent: navigator.userAgent,
    platform: (navigator as Navigator & { platform?: string }).platform ?? null,
    viewport: { w: window.innerWidth, h: window.innerHeight },
    screen: { w: window.screen.width, h: window.screen.height },
    device_pixel_ratio: window.devicePixelRatio || 1,
    is_touch: "ontouchstart" in window || navigator.maxTouchPoints > 0,
    render_check: { passed: rc.passed, method: rc.method },
  };
}

export function ConsentGate() {
  const router = useRouter();
  const [alreadyDone, setAlreadyDone] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      if (window.localStorage.getItem(LS_COMPLETED)) setAlreadyDone(true);
    } catch {
      /* localStorage disabled — treat as not-done */
    }
  }, []);

  const begin = async () => {
    setError(null);
    // Re-check at click time in case it changed since mount.
    try {
      if (window.localStorage.getItem(LS_COMPLETED)) {
        setAlreadyDone(true);
        return;
      }
    } catch {
      /* ignore */
    }

    setStarting(true);
    try {
      const env = captureEnvironment();
      const startedAtUtc = new Date().toISOString();

      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        throw new Error(`Could not start session (HTTP ${res.status}).`);
      }
      const create = (await res.json()) as CreateResult;
      const record = initRecord(create, env, startedAtUtc);

      window.localStorage.setItem(LS_CREATE, JSON.stringify(create));
      window.localStorage.setItem(LS_RECORD, JSON.stringify(record));

      router.push("/study");
    } catch (e) {
      setStarting(false);
      setError(
        e instanceof Error
          ? e.message
          : "Something went wrong starting the study. Please try again."
      );
    }
  };

  if (alreadyDone) {
    return (
      <div className="mt-8 rounded-lg border border-green-200 bg-green-50 px-6 py-5 text-green-900">
        <p className="font-medium">You&rsquo;ve already completed this study.</p>
        <p className="mt-1 text-sm">Thank you for taking part.</p>
      </div>
    );
  }

  return (
    <div className="mt-8">
      <button
        type="button"
        onClick={begin}
        disabled={starting}
        className={[
          "rounded-lg px-6 py-3 font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
          starting ? "bg-neutral-400 cursor-wait" : "bg-blue-600 hover:bg-blue-700",
        ].join(" ")}
      >
        {starting ? "Starting…" : "I consent and want to begin"}
      </button>
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
    </div>
  );
}
