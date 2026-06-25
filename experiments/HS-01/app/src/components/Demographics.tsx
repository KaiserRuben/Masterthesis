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
      <h2 className="text-2xl font-semibold text-ink mb-2">
        A few questions about you
      </h2>
      <p className="text-body leading-relaxed mb-8">
        This helps us describe who took part. It stays anonymous.
      </p>

      <div className="rounded-card border border-line bg-white p-6 shadow-card sm:p-8">
        <div className="space-y-6">
          {fields.map((field) => {
            const id = `demo-${field.field_id}`;
            return (
              <div key={field.field_id}>
                <label htmlFor={id} className="block text-sm font-medium text-ink mb-1.5">
                  {field.label}
                  {field.required && <span className="text-tum-600"> *</span>}
                </label>

                {field.type === "select" && field.options ? (
                  <select
                    id={id}
                    value={values[field.field_id] ?? ""}
                    onChange={(e) => set(field.field_id, e.target.value)}
                    className="w-full rounded-control border border-line bg-white px-3 py-2.5 text-base text-ink"
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
                    className="w-full rounded-control border border-line bg-white px-3 py-2.5 text-base text-ink placeholder:text-muted"
                    placeholder="(optional)"
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <button
        type="submit"
        data-testid="demographics-submit"
        disabled={!requiredComplete || submitting}
        className={[
          "mt-8 w-full rounded-control px-6 py-3 font-medium text-white transition-colors sm:w-auto",
          !requiredComplete || submitting
            ? "bg-tum-300 cursor-not-allowed"
            : "bg-tum-600 hover:bg-tum-700",
        ].join(" ")}
      >
        {submitting ? "Submitting…" : "Finish and submit"}
      </button>
    </form>
  );
}
