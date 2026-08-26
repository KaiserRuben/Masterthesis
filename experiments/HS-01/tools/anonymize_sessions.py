#!/usr/bin/env python3
"""Strip device fingerprints from HS-01 session records.

The study application records a browser environment block for quality control:
user agent, platform, screen geometry and device pixel ratio. Together those
fields form a device fingerprint. Across the 50 collected sessions the
combination is unique for most participants, which makes the records
re-identifiable to anyone who knows the participant pool — and the consent text
(``app/config/consent.en.md``) promises that no personal data is stored.

The analysis never uses the raw fields. ``analysis/hs01/load.py`` consumes the
user agent only through a four-way device classifier (mobile / tablet / desktop
/ unknown) and reads ``viewport.w`` into a column that nothing downstream
touches. So the fingerprint can be replaced by the derived class with no effect
on any published number.

This rewrites each record to keep the derived ``device`` class, the touch flag
and the render check, and drops the identifying fields. Wall-clock timestamps
are truncated to the hour, which preserves the day-level recruitment curve the
figures plot while removing millisecond-precision correlation handles.

Keep the unscrubbed originals outside the public repository — they are the raw
research data, and only the scrubbed form is safe to publish.

    python experiments/HS-01/tools/anonymize_sessions.py --check
    python experiments/HS-01/tools/anonymize_sessions.py --dst /tmp/scrubbed
    python experiments/HS-01/tools/anonymize_sessions.py --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "experiments" / "HS-01" / "results" / "sessions"

# Fields dropped from the environment block.
FINGERPRINT_FIELDS = ("user_agent", "platform", "viewport", "screen", "device_pixel_ratio")
# Environment fields that survive: a coarse device class plus quality flags.
KEPT_ENV_FIELDS = ("device", "is_touch", "render_check")
# Wall-clock fields truncated to the hour.
TIMESTAMP_PATHS = (
    ("participant", "consent", "at_utc"),
    ("timing", "started_at_utc"),
    ("timing", "completed_at_utc"),
    ("timing", "server_received_at_utc"),
)


def device_class(user_agent: str | None, is_touch: bool | None) -> str:
    """Collapse a user agent to a four-way device class.

    Mirrors ``_device_class`` in ``analysis/hs01/load.py``; once a record is
    scrubbed this is the only place the classification is ever made, so the
    stored value becomes the single source of truth.
    """
    ua = user_agent or ""
    if any(k in ua for k in ("iPhone", "Android", "Mobile")):
        return "mobile"
    if "iPad" in ua or (is_touch and "Macintosh" in ua):
        return "tablet"
    if ua:
        return "desktop"
    return "unknown"


def _truncate_to_hour(value: str) -> str:
    """``2026-06-30T16:49:31.104Z`` -> ``2026-06-30T16:00:00Z``."""
    if not isinstance(value, str) or "T" not in value:
        return value
    date, _, time_part = value.partition("T")
    hour = time_part[:2]
    if not hour.isdigit():
        return value
    return f"{date}T{hour}:00:00Z"


def scrub_session(record: dict) -> tuple[dict, bool]:
    """Return (scrubbed record, changed?). Idempotent."""
    changed = False
    env = record.get("environment")
    if isinstance(env, dict):
        if any(f in env for f in FINGERPRINT_FIELDS):
            changed = True
        # Derive before dropping; a record scrubbed earlier keeps its class.
        device = env.get("device") or device_class(env.get("user_agent"), env.get("is_touch"))
        new_env = {"device": device}
        for field in KEPT_ENV_FIELDS:
            if field == "device":
                continue
            if field in env:
                new_env[field] = env[field]
        record["environment"] = new_env

    for path in TIMESTAMP_PATHS:
        node = record
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and isinstance(node.get(path[-1]), str):
            original = node[path[-1]]
            truncated = _truncate_to_hour(original)
            if truncated != original:
                node[path[-1]] = truncated
                changed = True

    return record, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help=f"session directory to read (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", type=Path,
                        help="write scrubbed copies here instead of modifying --src")
    parser.add_argument("--in-place", action="store_true",
                        help="rewrite the files in --src")
    parser.add_argument("--check", action="store_true",
                        help="report which records still carry fingerprints; change nothing")
    args = parser.parse_args(argv)

    if not args.check and not args.in_place and args.dst is None:
        parser.error("pass one of --check, --dst DIR, or --in-place")

    files = sorted(p for p in args.src.glob("*.json") if p.name != "_counter.json")
    if not files:
        print(f"no session records under {args.src}", file=sys.stderr)
        return 1

    if args.dst:
        args.dst.mkdir(parents=True, exist_ok=True)

    dirty = 0
    for path in files:
        record = json.loads(path.read_text())
        record, changed = scrub_session(record)
        dirty += changed

        if args.check:
            if changed:
                print(f"  fingerprint present: {path.name}")
            continue

        target = (args.dst / path.name) if args.dst else path
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    counter = args.src / "_counter.json"
    if args.dst and counter.exists():
        shutil.copy2(counter, args.dst / counter.name)

    verb = "carry fingerprints" if args.check else "scrubbed"
    print(f"{dirty}/{len(files)} records {verb}")
    if args.check and dirty:
        print("run with --in-place (or --dst) to scrub", file=sys.stderr)
        return 1
    if not args.check:
        where = args.dst if args.dst else args.src
        print(f"wrote {len(files)} records to {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
