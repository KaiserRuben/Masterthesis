# syntax=docker/dockerfile:1
#
# HS-01 study app — multi-stage build.
#
# IMPORTANT — build context:
#   The three READ-ONLY runtime inputs the app needs (the frozen item pool, its
#   image assets, and the JSON Schemas) live OUTSIDE the app dir, at
#   ../pool_frozen and ../schemas. A build context rooted at the app dir cannot
#   COPY them. So this Dockerfile is written against a context rooted at
#   experiments/HS-01/ and is referenced with -f:
#
#       docker build \
#         -f experiments/HS-01/app/Dockerfile \
#         -t hs01-study \
#         experiments/HS-01
#
#   All COPY paths below are therefore relative to experiments/HS-01/ (i.e.
#   "app/...", "pool_frozen/...", "schemas/...").
#
# Runtime inputs are baked in at fixed in-image paths and the HS01_* env vars
# point at them, so the app's __dirname-relative path guesses (unreliable in the
# Next standalone bundle) are never consulted.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app

# Install deps against the lockfile first (better layer caching).
COPY app/package.json app/package-lock.json ./
RUN npm ci

# App sources. The .dockerignore (relative to the experiments/HS-01/ context)
# keeps node_modules/.next/data/tests/e2e out of the context.
COPY app/ ./

# The schemas must also be present AT BUILD TIME: `next build` collects page
# data by evaluating the API route modules, which import src/lib/schemas.ts —
# and that module reads the three schema JSONs at import time. Bake them in and
# point the resolver at them so the build step resolves identically to runtime.
COPY schemas/hs01.itempool.schema.json      /inputs/schemas/hs01.itempool.schema.json
COPY schemas/hs01.study-config.schema.json  /inputs/schemas/hs01.study-config.schema.json
COPY schemas/hs01.session.schema.json       /inputs/schemas/hs01.session.schema.json
COPY schemas/hs01.references.schema.json    /inputs/schemas/hs01.references.schema.json

# CRITICAL: next.config.js declares an `env` block for HS01_POOL / HS01_CONFIG /
# HS01_DATA_DIR / HS01_IMAGE_DIR / HS01_REFERENCES / HS01_REFS_DIR. Next *inlines*
# those `process.env.*` reads into
# the bundle at BUILD TIME (`process.env.X || <default>`). If they are unset
# here, the build bakes in the app-dir-relative DEFAULTS (e.g.
# /pool_frozen/itempool.json), and any runtime `ENV` for them is ignored. So the
# in-image runtime paths must be pinned here, before `npm run build`, identical
# to the runner-stage values. (HS01_SCHEMAS_DIR is NOT in the env block, so it is
# read at runtime normally — but pinning it here also satisfies the build-time
# schema read above.)
ENV HS01_POOL=/inputs/pool_frozen/itempool.json
ENV HS01_IMAGE_DIR=/inputs/pool_frozen/assets/images
ENV HS01_SCHEMAS_DIR=/inputs/schemas
ENV HS01_CONFIG=/app/config/study-config.json
ENV HS01_DATA_DIR=/app/data/sessions
ENV HS01_REFERENCES=/app/config/references.json
ENV HS01_REFS_DIR=/app/config/refs

# Produce the standalone server bundle (next.config.js sets output: "standalone").
RUN npm run build

# ── Stage 2: runner ───────────────────────────────────────────────────────────
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3939
ENV HOSTNAME=0.0.0.0

# Standalone server + its static assets. server.js sits at the standalone root;
# the static dir is not bundled into standalone and must be copied alongside.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Study config (also bundled into standalone, but copied explicitly so the path
# the env var points at is unambiguous and version-pinned to the build).
COPY --from=builder /app/config ./config

# Read-only inputs baked at fixed paths. These are NOT traced into the Next
# standalone output, so they must be present in the image and addressed by env.
COPY pool_frozen/itempool.json        /inputs/pool_frozen/itempool.json
COPY pool_frozen/assets/images        /inputs/pool_frozen/assets/images
COPY schemas/hs01.itempool.schema.json      /inputs/schemas/hs01.itempool.schema.json
COPY schemas/hs01.study-config.schema.json  /inputs/schemas/hs01.study-config.schema.json
COPY schemas/hs01.session.schema.json       /inputs/schemas/hs01.session.schema.json
COPY schemas/hs01.references.schema.json    /inputs/schemas/hs01.references.schema.json

# COPY preserves source file modes. The schema files on disk are 0600
# (owner-only), so without this the non-root "node" user gets EACCES reading
# them. Make every baked input world-readable (dirs traversable, files read).
RUN chmod -R a+rX /inputs

# Point every input resolver at its baked-in location. Must match the builder
# stage exactly: the four next.config `env` vars were inlined at build time, so
# these mainly cover the runtime-read path (HS01_SCHEMAS_DIR) and keep the
# container's environment self-documenting / overridable for the data dir.
ENV HS01_POOL=/inputs/pool_frozen/itempool.json
ENV HS01_IMAGE_DIR=/inputs/pool_frozen/assets/images
ENV HS01_SCHEMAS_DIR=/inputs/schemas
ENV HS01_CONFIG=/app/config/study-config.json
ENV HS01_DATA_DIR=/app/data/sessions
ENV HS01_REFERENCES=/app/config/references.json
ENV HS01_REFS_DIR=/app/config/refs

# Writable session output. node:alpine ships the unprivileged "node" user
# (uid/gid 1000). Create the data dir and hand it to that user, then drop root.
RUN mkdir -p /app/data/sessions && chown -R node:node /app/data
VOLUME ["/app/data/sessions"]

USER node

EXPOSE 3939

CMD ["node", "server.js"]
