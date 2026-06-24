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
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="max-w-md text-center">
        <CheckCircle2 className="mx-auto mb-4 h-12 w-12 text-green-600" aria-hidden="true" />
        <h1 className="text-2xl font-semibold text-neutral-900 mb-3">
          Thank you for taking part
        </h1>
        <p className="text-neutral-600 leading-relaxed">
          Your responses have been recorded. You can close this tab now. Your
          help with this research is much appreciated.
        </p>
      </div>
    </main>
  );
}
