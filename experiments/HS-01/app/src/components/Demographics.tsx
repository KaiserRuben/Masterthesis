"use client";

/**
 * Demographics — the final form. Renders the config-driven demographics_fields
 * (selects for enumerations, a free-text area for the optional comment).
 * Required fields gate submit: the button is disabled until every required
 * field has a value.
 *
 * Values are kept as a flat field_id -> string map (controlled) and reported up
 * on submit; the caller maps them into the typed `demographics` object.
 */

import { useState } from "react";
import type { DemographicsField } from "@/lib/types";

export type DemographicsValues = Record<string, string>;

export interface DemographicsProps {
  fields: DemographicsField[];
  initial?: DemographicsValues;
  onSubmit: (values: DemographicsValues) => void;
  submitting?: boolean;
}

/** Human-readable labels for the enumerated option codes. */
const OPTION_LABELS: Record<string, string> = {
  // age_band
  "18_24": "18–24",
  "25_34": "25–34",
  "35_44": "35–44",
  "45_54": "45–54",
  "55_plus": "55 or older",
  prefer_not_to_say: "Prefer not to say",
  // ml_familiarity
  no_experience: "No experience",
  some_exposure: "Some exposure",
  regular_practice: "Regular practice",
  // english_proficiency
  A1: "A1 (beginner)",
  A2: "A2 (elementary)",
  B1: "B1 (intermediate)",
  B2: "B2 (upper intermediate)",
  C1: "C1 (advanced)",
  C2: "C2 (proficient)",
  native: "Native speaker",
};

function optionLabel(code: string): string {
  return OPTION_LABELS[code] ?? code;
}

export function Demographics({
  fields,
  initial,
  onSubmit,
  submitting,
}: DemographicsProps) {
  const [values, setValues] = useState<DemographicsValues>(initial ?? {});

  const set = (fieldId: string, v: string) =>
    setValues((prev) => ({ ...prev, [fieldId]: v }));

  const requiredComplete = fields
    .filter((f) => f.required)
    .every((f) => (values[f.field_id] ?? "").length > 0);

  return (
    <form
      className="mx-auto max-w-xl px-6 py-12"
      onSubmit={(e) => {
        e.preventDefault();
        if (requiredComplete && !submitting) onSubmit(values);
      }}
    >
      <h2 className="text-2xl font-semibold text-neutral-900 mb-2">
        A few questions about you
      </h2>
      <p className="text-neutral-600 mb-8">
        This helps us describe who took part. It stays anonymous.
      </p>

      <div className="space-y-6">
        {fields.map((field) => {
          const id = `demo-${field.field_id}`;
          return (
            <div key={field.field_id}>
              <label htmlFor={id} className="block text-sm font-medium text-neutral-800 mb-1">
                {field.label}
                {field.required && <span className="text-red-600"> *</span>}
              </label>

              {field.type === "select" && field.options ? (
                <select
                  id={id}
                  value={values[field.field_id] ?? ""}
                  onChange={(e) => set(field.field_id, e.target.value)}
                  className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                >
                  <option value="" disabled>
                    Select…
                  </option>
                  {field.options.map((opt) => (
                    <option key={opt} value={opt}>
                      {optionLabel(opt)}
                    </option>
                  ))}
                </select>
              ) : (
                <textarea
                  id={id}
                  value={values[field.field_id] ?? ""}
                  onChange={(e) => set(field.field_id, e.target.value)}
                  rows={3}
                  className="w-full rounded-lg border border-neutral-300 px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  placeholder="(optional)"
                />
              )}
            </div>
          );
        })}
      </div>

      <button
        type="submit"
        disabled={!requiredComplete || submitting}
        className={[
          "mt-10 rounded-lg px-6 py-3 font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
          !requiredComplete || submitting
            ? "bg-neutral-300 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700",
        ].join(" ")}
      >
        {submitting ? "Submitting…" : "Finish and submit"}
      </button>
    </form>
  );
}
