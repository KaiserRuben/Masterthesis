#!/usr/bin/env python3
"""Build the curated pair-option word reference set for the HS-01 app.

For each fine-grained class that appears as a pair option, this picks one
example image from the local ImageNet example cache, copies it into the app's
bundled refs dir, and writes ``config/references.json`` (word -> {gloss, image}).

HARD INVARIANT: a reference image must never be a study stimulus. Every
candidate's SHA-256 is checked against the set of study-stimulus SHA-256s
(``pool_frozen/assets/images/*.png``) and any collision is skipped. The
committed result is re-checked on every build by ``tests/references-assets``.

This script is a DEV-ONLY curation tool: it reads the gitignored ImageNet cache
at the repo root. Its *output* (the refs + references.json) is committed; the
cache is not. Re-running it is idempotent given the same cache + glosses.

Usage:
    python experiments/HS-01/tools/build_references.py            # write
    python experiments/HS-01/tools/build_references.py --check    # dry run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Repo layout anchors (this file: experiments/HS-01/tools/build_references.py).
HS01_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HS01_DIR.parents[1]
CACHE_DIR = REPO_ROOT / ".cache" / "imagenet" / "category_images"
STUDY_IMAGES = HS01_DIR / "pool_frozen" / "assets" / "images"
APP_CONFIG = HS01_DIR / "app" / "config"
REFS_DIR = APP_CONFIG / "refs"
REFERENCES_JSON = APP_CONFIG / "references.json"

SCHEMA_VERSION = "1.0.0"

# One-line, neutral glosses. Deliberately factual and non-leading: they name the
# class without hinting at any perturbation or at which option is "correct".
GLOSSES: dict[str, str] = {
    "American bullfrog": "a large frog native to North America",
    "American robin": "a North American songbird with a reddish-orange breast",
    "axolotl": "a salamander that keeps its feathery external gills as an adult",
    "bald eagle": "a large bird of prey with a white head and tail",
    "box turtle": "a small land turtle with a high, domed shell that can close tightly",
    "cello": "a large stringed instrument played upright with a bow",
    "chameleon": "a lizard known for changing colour and having a long tongue",
    "cock": "an adult male chicken; a rooster",
    "desert grassland whiptail lizard": "a slender, long-tailed lizard of North American grasslands",
    "fire salamander": "a black salamander with bright yellow markings",
    "flamingo": "a tall wading bird with pink plumage and long legs",
    "great grey owl": "a very large grey owl with a round, ringed facial disc",
    "great white shark": "a large shark with a grey back and white underside",
    "indigo bunting": "a small songbird; the breeding male is bright blue",
    "loggerhead sea turtle": "a large sea turtle with a big head and a reddish-brown shell",
    "marimba": "a wooden xylophone-like instrument played with mallets",
    "mud turtle": "a small freshwater turtle with a dull, smooth shell",
    "ostrich": "the largest living bird; flightless, with a long neck and legs",
    "stingray": "a flat-bodied fish with a long, whip-like tail",
    "tench": "a thick-bodied European freshwater fish",
    "tiger shark": "a large shark with dark vertical stripes on its sides",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study_stimulus_hashes() -> set[str]:
    return {sha256(p) for p in STUDY_IMAGES.glob("*.png")}


def folder_for(word: str) -> Path:
    return CACHE_DIR / word.lower().replace(" ", "_")


def image_name_for(word: str) -> str:
    return "ref-" + word.lower().replace(" ", "_") + ".png"


def pick_candidate(word: str, study: set[str]) -> tuple[Path | None, list[str]]:
    """Return (chosen source path or None, notes).

    Skips any candidate whose bytes are a study stimulus, then picks the
    SMALLEST-filesize survivor (keeps the committed bundle lean while staying a
    byte-exact copy, so the SHA-256 exclusion test stays meaningful).
    """
    folder = folder_for(word)
    notes: list[str] = []
    if not folder.is_dir():
        notes.append("no cache folder -> gloss-only")
        return None, notes
    pngs = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")
    survivors = []
    for p in pngs:
        if sha256(p) in study:
            notes.append(f"skip {p.name} (matches a study stimulus)")
            continue
        survivors.append(p)
    if not survivors:
        notes.append("no non-stimulus candidate -> gloss-only")
        return None, notes
    return min(survivors, key=lambda p: p.stat().st_size), notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry run; write nothing")
    args = ap.parse_args()

    if not CACHE_DIR.is_dir():
        print(f"ERROR: ImageNet cache not found at {CACHE_DIR}", file=sys.stderr)
        return 2

    study = study_stimulus_hashes()
    print(f"study stimuli: {len(study)} hashes from {STUDY_IMAGES}")

    if not args.check:
        REFS_DIR.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict[str, str | None]] = {}
    gloss_only: list[str] = []
    for word in sorted(GLOSSES):
        src, notes = pick_candidate(word, study)
        for n in notes:
            print(f"  {word}: {n}")
        if src is None:
            entries[word] = {"gloss": GLOSSES[word], "image": None}
            gloss_only.append(word)
            continue
        name = image_name_for(word)
        if not args.check:
            shutil.copyfile(src, REFS_DIR / name)
        entries[word] = {"gloss": GLOSSES[word], "image": name}
        print(f"  {word}: {src.relative_to(REPO_ROOT)} -> {name}")

    doc = {"schema_version": SCHEMA_VERSION, "entries": entries}
    if args.check:
        print("\n--check: would write", REFERENCES_JSON)
    else:
        REFERENCES_JSON.write_text(json.dumps(doc, indent=2) + "\n")
        print("\nwrote", REFERENCES_JSON)

    print(f"\n{len(entries)} entries; {len(gloss_only)} gloss-only: {gloss_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
