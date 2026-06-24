# HS-01 study app

A small Next.js 14 (App Router, `output: "standalone"`) web app that runs the
HS-01 human-validity rating study: raters look at the prompts / images / pairs
that the VLMs were shown and report how clearly *they* can understand them.

The app reads three **read-only** inputs at runtime and writes one **append-only**
output (one JSON file per session). It does no analysis itself.

```
experiments/HS-01/
├── app/                 ← this app
│   ├── config/          ← study-config.json + consent.en.md (generated)
│   └── data/sessions/   ← session JSON output (writable; bind-mounted / VOLUME)
├── pool_frozen/         ← frozen item pool + image assets (read-only input)
│   ├── itempool.json
│   └── assets/images/
└── schemas/             ← three JSON Schemas (read-only input)
```

## Inputs & environment variables

Every input path is resolved from an env var with a sensible default. Defaults
are relative to the app dir and are correct for local `npm run dev`; the Docker
image overrides all of them to baked-in absolute paths (see below).

| Env var            | Default (relative to `app/`)        | Purpose                                    | Mode  |
|--------------------|-------------------------------------|--------------------------------------------|-------|
| `HS01_POOL`        | `../pool_frozen/itempool.json`      | Frozen item pool                           | read  |
| `HS01_IMAGE_DIR`   | `../pool_frozen/assets/images`      | PNG stimulus assets                        | read  |
| `HS01_SCHEMAS_DIR` | *(multi-candidate resolver)*        | The three JSON Schemas (`../../schemas`)   | read  |
| `HS01_CONFIG`      | `./config/study-config.json`        | Study config (forms, scales, policy)       | read  |
| `HS01_DATA_DIR`    | `./data/sessions`                   | Per-session JSON output                    | write |

`HS01_SCHEMAS_DIR` has no fixed default: `src/lib/schemas.ts` tries the env var
first, then a `__dirname`-relative path (dev / vitest), then `cwd`-relative
fallbacks (bundled server). The schema JSONs are **not** traced into the Next
standalone bundle, so in the Docker image `HS01_SCHEMAS_DIR` is set explicitly.

> **Build-time inlining (Docker gotcha).** `next.config.js` declares an `env`
> block for `HS01_POOL`, `HS01_CONFIG`, `HS01_DATA_DIR`, and `HS01_IMAGE_DIR`.
> Next **inlines** those four into the bundle at *build* time, so they cannot be
> changed by a `docker run -e ...` at runtime — they are frozen to whatever was
> set when `npm run build` ran. The Dockerfile therefore pins them (to the
> baked-in paths) in the **builder** stage, before the build. To relocate the
> session output you would rebuild with a different `HS01_DATA_DIR`, or simply
> bind-mount your host dir onto the in-image default `/app/data/sessions`
> (what the run command and Compose file do). `HS01_SCHEMAS_DIR` is *not*
> inlined and is honoured at runtime.

### Boot guard (refuses to start on a stale pool)

Every API route calls `ensureInputs()`, which loads + AJV-validates the pool and
config and then asserts that `config.pool_ref.pool_file_sha256` equals the
sha256 of the actual `itempool.json` bytes. **If the pool and the config drift
apart, the app refuses to serve and returns a 500 with an explicit mismatch
message.** This is the guarantee that the rendered study always matches the
config it was built against. Regenerate the config (below) whenever the pool
changes.

## Develop

```bash
cd experiments/HS-01/app
npm install
npm run dev          # → http://localhost:3939  (binds 0.0.0.0)
```

Sessions are written to `app/data/sessions/` (gitignored). Images are served by
`/api/images/[name]` straight from `HS01_IMAGE_DIR`.

## Test

```bash
npm test             # vitest run — 80 unit/integration tests (loaders, store,
                     # session flow, widgets, instrumentation, …)
```

End-to-end is scaffolded only (`npm run e2e` → Playwright). There is no
committed Playwright config / spec yet; the unit + integration suite is the
authoritative gate.

## Regenerate the study config

The config and consent text are **generated deterministically from the frozen
pool** by `make_study_config.py` (same pool → identical config). Run it whenever
`pool_frozen/itempool.json` changes — it recomputes `pool_file_sha256`, so it is
also the way you clear a boot-guard mismatch. Run from the **HS-01 dir**, not
the app dir:

```bash
cd experiments/HS-01
conda run -n uni python make_study_config.py
# → wrote app/config/study-config.json (… items, forms A:…/B:…/C:…)
```

This writes `app/config/study-config.json` and `app/config/consent.en.md`.

## Docker

### Why the build context is `experiments/HS-01/`, not the app dir

The pool and schemas live **outside** the app dir (`../pool_frozen`,
`../schemas`). A context rooted at the app dir cannot `COPY` them. So the build
context is the parent `experiments/HS-01/`, and the Dockerfile is passed with
`-f`. The runner stage bakes `pool_frozen/` + `schemas/` in at fixed paths and
points the `HS01_*` env vars at them, so runtime path resolution never depends
on the (unreliable in standalone) `__dirname` guesses.

### Build & run

From the **repo root**:

```bash
docker build \
  -f experiments/HS-01/app/Dockerfile \
  -t hs01-study \
  experiments/HS-01

# Session output goes to a host dir you control. Create it writable for the
# in-image "node" user (uid/gid 1000) and bind-mount it:
mkdir -p ./hs01-sessions
docker run --rm -p 3939:3939 \
  -v "$(pwd)/hs01-sessions:/app/data/sessions" \
  hs01-study
# → http://localhost:3939
```

Or with Compose (run from the app dir — its `context: ..` points at
`experiments/HS-01/`, and session output lands in `app/data/sessions/`):

```bash
cd experiments/HS-01/app
docker compose up --build
```

The image runs as the unprivileged `node` user and exposes `3939`. Session JSON
is written under the `/app/data/sessions` volume; back this with a bind mount or
named volume so sessions survive container removal.

## Accepted limitation — duplicate entry across devices

This is a **convenience sample** run on machines we control. Within one device a
participant counter and per-device safeguards apply, but the tool does **not**
deduplicate the same person re-entering from a *different* device/browser — each
visit gets a fresh `participant_code` and a fresh session file. We accept this:
for a small in-person convenience sample the risk is low and the cost of a
hardened identity scheme is not warranted. Treat session files as the unit of
analysis and sanity-check counts against the recruitment dashboard
(`/api/status`).

## One-time aggregation (out of scope for this tool)

The app only **emits** one JSON file per session under `HS01_DATA_DIR`. Turning
that pile of session records into a tidy analysis table (one row per trial /
per rater, joined to the item pool) is a **manual one-time job** done with
pandas / jq when data collection is complete — it is deliberately not part of
this tool. Sketch:

```bash
# collect the completed sessions off the host volume, then e.g.
jq -s '[.[] | select(.status=="completed")]' data/sessions/*.json > sessions.json
# … load sessions.json in a notebook, explode .trials, join on item_id → pool.
```

Keep the raw session JSON; it is the canonical record. Aggregation is derived
and re-runnable.
