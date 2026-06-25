#!/usr/bin/env python3
"""Build the curated pair-option word reference set for the HS-01 app.

Every word that can appear as a pair option in the study (resolved from the
frozen pool exactly as the app's PairChoice does) gets a one-line neutral gloss
AND an example reference photo. Some option words are abstraction lexemes
("songbird", "ratite", "constrictor", ...) rather than concrete ImageNet
classes; for those we resolve to the representative concrete class carried on
the pool item's ``cell`` (cross-checked against the taxonomy mapping in
``src.common.abstraction`` / ``src.data.taxonomy``) and use a photo of that
class.

HARD INVARIANT: a reference image must NEVER be the same PHOTO as a study
stimulus. The byte-SHA check that an earlier version used is INSUFFICIENT — the
manipulation pipeline re-encodes a clean seed photo (``origin.png``) into the
committed stimulus: pixel-identical, byte-DIFFERENT, so a SHA guard waves it
through. We therefore exclude PERCEPTUALLY: each candidate is rejected if its
pixel-MAE OR dHash-Hamming distance to ANY member of the exclusion set (every
committed stimulus + every clean seed origin the pool derives from) is below a
small threshold. If every cached candidate for a class collides, we fetch FRESH
(not-yet-cached) photos for that class from the ImageNet provider and pick the
first disjoint one.

This script is a DEV-ONLY curation tool: it reads the gitignored ImageNet cache
and the run dirs at the repo root, and may stream a few fresh images from
HuggingFace. Its *output* (the refs + references.json) is committed; the cache
is not. Re-running it is idempotent given the same cache + glosses.

Usage (run via the repo's conda env):
    conda run -n uni python experiments/HS-01/tools/build_references.py
    conda run -n uni python experiments/HS-01/tools/build_references.py --check
    conda run -n uni python experiments/HS-01/tools/build_references.py --no-fetch
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Repo layout anchors (this file: experiments/HS-01/tools/build_references.py).
HS01_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HS01_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.abstraction import resolve_label  # noqa: E402
from src.data.imagenet import ImageNetCache  # noqa: E402
from src.data.taxonomy import path_of  # noqa: E402

CACHE_DIR = REPO_ROOT / ".cache" / "imagenet"
CACHE_IMAGES = CACHE_DIR / "category_images"
RUNS_DIR = REPO_ROOT / "runs"
STUDY_IMAGES = HS01_DIR / "pool_frozen" / "assets" / "images"
POOL_JSON = HS01_DIR / "pool_frozen" / "itempool.json"
STUDY_CONFIG = HS01_DIR / "app" / "config" / "study-config.json"
APP_CONFIG = HS01_DIR / "app" / "config"
REFS_DIR = APP_CONFIG / "refs"
REFERENCES_JSON = APP_CONFIG / "references.json"
REFERENCES_SCHEMA = HS01_DIR / "schemas" / "hs01.references.schema.json"

SCHEMA_VERSION = "1.1.0"

# Perceptual disjointness thresholds (mirrored in tests/references-assets.test.ts).
# Known seed-photo leaks sit at MAE 0 / Ham 0; the closest legitimate reference is
# at MAE ~22 / Ham ~90, so these thresholds sit comfortably in the gap.
CMP_SIZE = 64
MAE_THRESHOLD = 3.0
DHASH_SIZE = 16
HAM_THRESHOLD = 8
# How many fresh photos to stream for a class whose whole cache collides.
FRESH_FETCH_N = 8

# One-line, neutral glosses. Deliberately factual and non-leading: they name the
# class/category without hinting at any perturbation or at which option is
# "correct". Concrete-species glosses reuse the design spec's wording; the
# abstraction lexemes get plain category definitions. The displayed photo is of
# the representative concrete class behind each abstraction word.
GLOSSES: dict[str, str] = {
    # --- concrete species / objects (spec wording) ---
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
    # --- attention-check concrete classes (the obvious-pair options) ---
    "boa constrictor": "a large, heavy-bodied snake that kills prey by squeezing",
    "green iguana": "a large green, tree-dwelling lizard with a spiny crest and a throat fan",
    # --- abstraction / superordinate lexemes (category definitions) ---
    "constrictor": "a large snake that kills prey by squeezing, such as a boa",
    "equine": "a horse or a close relative such as a zebra or donkey",
    "iguana": "a large plant-eating lizard with a row of spines along its back",
    "percussion instrument": "an instrument played by striking, such as a drum or xylophone",
    "ratite": "a large flightless bird such as an ostrich or emu",
    "reptile": "a cold-blooded, scaly animal such as a snake, lizard or turtle",
    "retriever": "a breed of dog originally bred to fetch game",
    "snake": "a long, legless reptile",
    "songbird": "a small perching bird known for its song",
    "sparrow": "a small, plump brownish songbird",
}


# ── pool word enumeration + word -> concrete-class resolution ────────────────
def pair_words_and_classes() -> dict[str, str]:
    """Map every pair-option word the UI can show to a representative concrete
    ImageNet class.

    Words are enumerated from the frozen pool exactly as the app does: for each
    pair item referenced by a form (including the attention-check pair, so its
    options are indistinguishable from real trials), take the
    source's ``cell.anchor_word`` / ``target_word``. The concrete class behind a
    slot is the matching ``cell.anchor_class`` / ``target_class`` (already the
    concrete ImageNet label, whether the displayed word is concrete or an
    abstraction). We cross-check the word against the taxonomy mapping so a stale
    pool can't silently mis-resolve.
    """
    pool = json.loads(POOL_JSON.read_text())
    cfg = json.loads(STUDY_CONFIG.read_text())
    src_by_id = {s["source_id"]: s for s in pool["sources"]}
    item_to_src = {it["item_id"]: it["source_id"] for it in pool["items"]}

    word_to_concretes: dict[str, set[str]] = defaultdict(set)
    for form in cfg["forms"]:
        for pid in form["pair_items"]:
            cell = src_by_id[item_to_src[pid]]["cell"]
            word_to_concretes[cell["anchor_word"]].add(cell["anchor_class"])
            word_to_concretes[cell["target_word"]].add(cell["target_class"])

    reps: dict[str, str] = {}
    for word, concretes in word_to_concretes.items():
        concrete = sorted(concretes)[0]  # deterministic representative
        # Cross-check: the displayed word must be the concrete class itself or
        # one of its taxonomy cluster labels (level 0/1/2).
        ok = word == concrete
        if not ok:
            for lvl in range(3):
                try:
                    if resolve_label(concrete, lvl) == word:
                        ok = True
                        break
                except (KeyError, ValueError):
                    pass
        if not ok:
            try:
                p = path_of(concrete)
            except KeyError:
                p = "UNKNOWN"
            print(
                f"  WARNING: word {word!r} not a taxonomy label of {concrete!r} "
                f"(path={p}); using the pool cell verbatim."
            )
        reps[word] = concrete
    return reps


# ── perceptual primitives (PIL + numpy only) ────────────────────────────────
def _gray(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float64)


def pixel_mae_arr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def dhash(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize((DHASH_SIZE + 1, DHASH_SIZE), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def dhash_image(im: Image.Image) -> np.ndarray:
    g = im.convert("L").resize((DHASH_SIZE + 1, DHASH_SIZE), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def gray_image(im: Image.Image, size: int) -> np.ndarray:
    return np.asarray(
        im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64
    )


class ExclusionSet:
    """The stimulus-photo exclusion set: every committed stimulus + every clean
    seed origin the pool derives from. Precomputes grayscale-resize + dHash."""

    def __init__(self) -> None:
        paths = list(STUDY_IMAGES.glob("*.png"))
        n_stim = len(paths)
        paths += self._seed_origins()
        self._gray = [_gray(p, CMP_SIZE) for p in paths]
        self._dhash = [dhash(p) for p in paths]
        self._paths = paths
        print(
            f"exclusion set: {len(paths)} images "
            f"({n_stim} committed stimuli + {len(paths) - n_stim} seed origins)"
        )

    @staticmethod
    def _seed_origins() -> list[Path]:
        pool = json.loads(POOL_JSON.read_text())
        run_ids = {
            s["experiment_ref"]["run_id"]
            for s in pool["sources"]
            if s.get("experiment_ref") and s["experiment_ref"].get("run_id")
        }
        origins: list[Path] = []
        seen: set[str] = set()
        for o in RUNS_DIR.glob("**/origin.png"):
            if run_ids.intersection(o.parts) and str(o) not in seen:
                seen.add(str(o))
                origins.append(o)
        return origins

    def disjoint_arr(self, g: np.ndarray, dh: np.ndarray) -> tuple[bool, float, int]:
        """(is_disjoint, min_MAE, min_Ham) for a candidate's gray+dhash arrays."""
        min_mae = min(pixel_mae_arr(g, eg) for eg in self._gray)
        min_ham = min(hamming(dh, ed) for ed in self._dhash)
        return (min_mae > MAE_THRESHOLD and min_ham > HAM_THRESHOLD, min_mae, min_ham)

    def disjoint_path(self, path: Path) -> tuple[bool, float, int]:
        return self.disjoint_arr(_gray(path, CMP_SIZE), dhash(path))

    def disjoint_image(self, im: Image.Image) -> tuple[bool, float, int]:
        return self.disjoint_arr(gray_image(im, CMP_SIZE), dhash_image(im))


# ── candidate selection ──────────────────────────────────────────────────────
def slug(class_name: str) -> str:
    return class_name.replace(" ", "_").lower()


def image_name_for(word: str) -> str:
    return "ref-" + slug(word) + ".png"


def cached_pngs(class_name: str) -> list[Path]:
    folder = CACHE_IMAGES / slug(class_name)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.png") if not p.name.startswith("._"))


def pick_for_class(
    class_name: str, excl: ExclusionSet, cache: ImageNetCache | None, fetch: bool
) -> tuple[Image.Image | None, list[str]]:
    """Pick a perceptually-disjoint example image for *class_name*.

    Tries cached candidates first; if all collide (or none exist) and *fetch* is
    on, streams FRESH_FETCH_N fresh photos from the provider (which appends to
    the cache) and re-scans the newly fetched ones. Returns (PIL image | None,
    notes)."""
    notes: list[str] = []
    for p in cached_pngs(class_name):
        ok, mae, ham = excl.disjoint_path(p)
        if ok:
            notes.append(f"cache {p.name} disjoint (MAE={mae:.1f}, Ham={ham})")
            return Image.open(p).convert("RGB"), notes
        notes.append(f"skip cache {p.name} (MAE={mae:.1f}, Ham={ham})")

    if not fetch or cache is None:
        notes.append("no disjoint cached candidate; fetch disabled")
        return None, notes

    before = {p.name for p in cached_pngs(class_name)}
    notes.append(f"fetching {FRESH_FETCH_N} fresh photos from provider...")
    try:
        cache.load_samples([class_name], n_per_class=len(before) + FRESH_FETCH_N)
    except Exception as e:  # network / HF gate / unknown class
        notes.append(f"FETCH FAILED: {type(e).__name__}: {e}")
        return None, notes
    fresh = [p for p in cached_pngs(class_name) if p.name not in before]
    notes.append(f"fetched {len(fresh)} new photos")
    for p in fresh:
        ok, mae, ham = excl.disjoint_path(p)
        if ok:
            notes.append(f"fresh {p.name} disjoint (MAE={mae:.1f}, Ham={ham})")
            return Image.open(p).convert("RGB"), notes
        notes.append(f"skip fresh {p.name} (MAE={mae:.1f}, Ham={ham})")
    notes.append("no disjoint candidate even after fetch")
    return None, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry run; write nothing")
    ap.add_argument(
        "--no-fetch",
        action="store_true",
        help="never stream fresh photos (offline); report classes still needing one",
    )
    args = ap.parse_args()

    if not CACHE_IMAGES.is_dir():
        print(f"ERROR: ImageNet cache not found at {CACHE_IMAGES}", file=sys.stderr)
        return 2

    word_to_class = pair_words_and_classes()
    words = sorted(word_to_class)
    missing_gloss = [w for w in words if w not in GLOSSES]
    if missing_gloss:
        print(f"ERROR: no gloss for words: {missing_gloss}", file=sys.stderr)
        return 2
    print(f"{len(words)} pair-option words to cover.\n")

    excl = ExclusionSet()
    cache = None if args.no_fetch else ImageNetCache(dirs=[CACHE_DIR])

    if not args.check:
        REFS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve one image per DISTINCT concrete class (several words can share one,
    # e.g. ratite/ostrich), then bind each word to its class's image filename.
    classes = sorted(set(word_to_class.values()))
    class_image: dict[str, Image.Image] = {}
    blocked: list[str] = []
    for cls in classes:
        img, notes = pick_for_class(cls, excl, cache, fetch=not args.no_fetch)
        print(f"{cls!r}:")
        for n in notes:
            print(f"    {n}")
        if img is None:
            blocked.append(cls)
        else:
            class_image[cls] = img

    entries: dict[str, dict[str, str]] = {}
    for word in words:
        cls = word_to_class[word]
        if cls not in class_image:
            continue  # blocked class; reported below
        name = image_name_for(word)
        if not args.check:
            class_image[cls].save(REFS_DIR / name, "PNG")
        entries[word] = {"gloss": GLOSSES[word], "image": name}

    doc = {"schema_version": SCHEMA_VERSION, "entries": entries}

    if not args.check:
        # Remove any orphaned refs from a previous run (e.g. the old null-image case).
        used = {e["image"] for e in entries.values()}
        for f in REFS_DIR.glob("ref-*.png"):
            if f.name not in used:
                f.unlink()
                print(f"removed orphan {f.name}")
        REFERENCES_JSON.write_text(json.dumps(doc, indent=2) + "\n")
        print("\nwrote", REFERENCES_JSON)
    else:
        print("\n--check: would write", REFERENCES_JSON)

    # Validate against the references schema.
    valid_msg = "skipped (jsonschema not available)"
    try:
        import jsonschema

        schema = json.loads(REFERENCES_SCHEMA.read_text())
        jsonschema.Draft202012Validator(schema).validate(doc)
        valid_msg = "VALID against hs01.references.schema.json"
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        valid_msg = f"INVALID: {e}"
    print(f"schema: {valid_msg}")

    covered = len(entries)
    print(
        f"\n{covered}/{len(words)} words covered; "
        f"{len(set(word_to_class.values()))} distinct classes."
    )
    if blocked:
        print(
            "\nBLOCKED — these classes still need a perceptually-disjoint photo "
            "(no disjoint cached candidate; fetch unavailable/failed):"
        )
        for cls in blocked:
            words_for = [w for w in words if word_to_class[w] == cls]
            print(f"  - {slug(cls)}  (class {cls!r}; words {words_for})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
