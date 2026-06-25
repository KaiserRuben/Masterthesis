"use client";

/**
 * /done — terminal thank-you screen. Reaching here means the session was
 * submitted and the `completed` flag is set, so re-entry to the runner is
 * blocked by ConsentGate. We clear the in-progress record/create caches (the
 * server holds the authoritative copy) but keep the `completed` flag.
 */

import { useEffect } from "react";
import { CheckCircle2 } from "lucide-react";

import { LS_COMPLETED, LS_CREATE, LS_RECORD } from "@/state/useSession";

export default function DonePage() {
  useEffect(() => {
    try {
      // Only clear the in-progress caches once the session is CONFIRMED
      // completed (the submit succeeded). Guarding on LS_COMPLETED ensures an
      // erroneous arrival here can never wipe an unsubmitted record.
      if (window.localStorage.getItem(LS_COMPLETED)) {
        window.localStorage.removeItem(LS_RECORD);
        window.localStorage.removeItem(LS_CREATE);
      }
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-12">
      <div className="w-full max-w-md rounded-card border border-line bg-white px-8 py-10 text-center shadow-card">
        <span className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full bg-tum-50">
          <CheckCircle2 className="h-7 w-7 text-tum-600" aria-hidden="true" />
        </span>
        <h1 className="text-2xl font-semibold text-ink mb-3">
          Thank you for taking part
        </h1>
        <p className="text-body leading-relaxed">
          Your responses have been recorded. You can close this tab now. Your
          help with this research is much appreciated.
        </p>
      </div>
    </main>
  );
}
