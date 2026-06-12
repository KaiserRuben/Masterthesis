# HS-01 Survey Tool — Architecture & Design Decisions

Decided 2026-06-11 (design session). Components: **Design** (vault: `HS-01-Human-Validity-Study.md`) · **Datastructure** (`schemas/`) · **Data** (`data_raw/`) · **Tool** (this doc → implementation).

## Stack

- **Next.js** (App Router; API routes = backend), **Tailwind**, **framer-motion**, **lucide** icons. No separate backend service.
- **Docker** single container (next standalone build), session data on a volume mount.
- Deployment: co-hosted on user's VPS behind own domain (TLS/reverse proxy per host setup). Build is host-agnostic: one process + one data dir.

## Storage model

- **One JSON file per session**: `data/sessions/{session_id}.json`, conforming to `schemas/hs01.session.schema.json`. No database — ~30 sessions; files are diffable and the final aggregation is a one-time manual pandas/jq job (explicitly out of scope for the tool).
- **Atomic writes**: write to `{session_id}.json.tmp`, fsync, rename.
- Server-side validation with **ajv** (draft 2020-12) on every write; invalid payloads are still persisted verbatim with a `x_validation_errors` extension field — never lose raw data.
- Item pool + study config are read-only inputs baked into the image (or volume-mounted), validated against their schemas at startup; refuse to boot on mismatch (`pool_ref.pool_file_sha256`).

## Session lifecycle

1. **Open link** (single public URL). Landing + consent.
2. On consent: `POST /api/sessions` → server generates `session_id` (uuid) + `participant_code` (sequential `P###`, counter file), assigns **form** by least-loaded balancing (count over sessions with status `completed` or in-progress younger than 24 h), creates the session file (status field via checkpoint), returns `{session_id, participant_code, form payload}`.
3. Client keeps the full in-progress record in **localStorage** after every trial → break/pickup on the same device; on revisit, resume at the next unanswered trial. A `completed` flag in localStorage blocks re-entry from the same device (open-link duplicate risk across devices is accepted — convenience sample, documented limitation).
4. **Phase checkpoints** (decision: yes): at each phase end, `PUT /api/sessions/{id}/checkpoint` with the full record-as-known (idempotent overwrite, status in-progress). Abandoned sessions keep their completed phases.
5. **Final write-back**: `POST /api/sessions/{id}/submit` with the complete record → validate, atomic write, `status: completed`.

Endpoints: `POST /api/sessions` · `PUT /api/sessions/{id}/checkpoint` · `POST /api/sessions/{id}/submit` · static frontend + `/assets/images/*`.

## Timing implementation (schema semantics)

- All `*_ms` are offsets from session start. Within a page load they come from `performance.now()` (monotonic). Across reloads (break/pickup), the client chains epochs: `offset = (load_epoch_utc − started_at_utc) + perf_delta`.
- Consequence: **within-trial differences (RTs) are strictly monotonic** (a trial never spans a reload); session-level offsets are wall-clock-approximate across breaks. Acceptable; documented here, no schema change.
- `onset_ms` fires at render-complete: image decode finished (`img.decode()`) + next animation frame. The **next trial's image is preloaded** during the current trial so onset is deterministic.
- framer-motion: keep stimulus presentation animation-free (fade-ins distort onset); motion is for chrome/transitions only.

## Fidelity & quality hooks (from study config)

- Verbatim prompt rendering: prompts inserted as text nodes (never through any normalizing pipeline), font stack pinned; **render check** at session start (canvas-render a known homoglyph string, compare) → `environment.render_check`.
- Per-trial `rendered_image` (css vs natural px) asserts the no-upscaling rule; `min_rendered_image_css_px` gate at entry.
- Integrity events (blur/focus/visibility) logged during judgment phases.
- Per-trial randomization (item order within phase, A/B option order) seeded from `rng_seed`, logged → fully reproducible.

## Privacy

Anonymous `participant_code` only; no IPs, no cookies (localStorage only), no personal data. Consent text versioned (`consent_version` + sha256).

## Out of scope (decided)

- Aggregation tool: submissions → analysis table is a **one-time manual job** at collection end.
- Cross-device resume, accounts, admin UI: not needed at n≈30. A `GET /api/status` (session counts per form) is the only operational nicety.

## Open prerequisites (data side, see data_raw/DATA_MANIFEST.md)

- Qwen items: Exp-101q/Exp-102 not yet synced; Exp-27 cone runs never reached the boundary (min TgtBal ≈ 2.5) → Qwen stimuli pending new data.
- Image-heavy stratum: **0 qualifying individuals** in current archives → dedicated `modality: image_only` (and balanced) generation runs required before pool freeze.
- Image assets: ~50 % of qualifying rows have rendered PNGs (final-Pareto); the rest need host-side VQGAN re-render from genotype + context (see manifest).
