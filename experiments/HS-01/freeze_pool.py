#!/usr/bin/env python3
"""HS-01 — Pool freeze: final item selection + assets + schema-valid item pool.

Layer 3 of the data process:
    data_raw/  ->  pool_staging/  ->  pool_frozen/  (THIS)

Selects EXACTLY the design-target item count per (phase, stratum) from the
staged candidates, copies the showable assets into a clean structure, and emits
`itempool.json` validating against `schemas/hs01.itempool.schema.json`.

Selection policy (study-owner directive, 2026-06-17):
  TOP PRIORITY = boundary closeness  -> primary sort by tgtbal ascending.
  Quality FLOORS (as far as the data allows):
    * image clarity   : d_img_matrix <= D_IMG_CAP  (subject still clear)
    * prompt readable : n_active_text_genes <= TG_CAP  (avoid the >=7 collapse
                        zone; selects "semantic gutting" over char-corruption)
    * closeness floor : tgtbal <= TGT_TIGHT (very-close boundary)
  Floors relax per stratum only if it cannot otherwise fill; relaxation is logged.
  Diversity: round-robin across (SUT x label-pair) so no stratum is one-pair /
  one-SUT dominated (pool must be balanced over models, HS-01 §4).

Output:
    pool_frozen/
        itempool.json                  (schema 1.0.0)
        assets/images/<source_id>.png  (256x256 SUT-input PNGs / origins)
        POOL_README.md                 (per-stratum achieved quality + decisions)

Re-runnable / idempotent (full rebuild of pool_frozen/).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
STAGE = HERE / "pool_staging"
OUT = HERE / "pool_frozen"
ASSETS = OUT / "assets" / "images"
REPO = HERE.parent.parent
SCHEMA = HERE / "schemas" / "hs01.itempool.schema.json"
IMAGENET = Path.home() / ".cache" / "imagenet" / "category_images"

# ---- selection floors (study-owner directive: closeness first) -------------
TGT_TIGHT = 1e-3      # very-close boundary
D_IMG_CAP = 0.05      # image subject still clear
TG_CAP = 7            # prompt readable (Exp-100: >=7 text genes -> collapse)

TARGETS = {
    "text":  {"clean": 6, "low_drift": 8, "medium_drift": 8, "high_drift": 8},
    "image": {"raw": 6, "roundtrip": 6, "boundary_joint": 12, "image_heavy": 6},
    "pair":  {"baseline": 8, "image_heavy": 14, "text_heavy": 14, "balanced": 14},
}
ROSTER_CLASSES = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]

_slug = lambda s: re.sub(r"[^A-Za-z0-9_-]", "", str(s).replace(" ", "_").replace("→", "-"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load(phase, stratum, name="candidates.parquet"):
    p = STAGE / phase / stratum / name
    return pd.read_parquet(p) if p.exists() else None


_PHOTO_CACHE: dict = {}
def photo_of(run_rel) -> str:
    """Anchor-PHOTO identity = sha of the run's origin.png (the seed image). Two
    runs that reused the same gap_filter/roster photo hash identically. This is
    the true 'seed' for the no-repeat rule — run_dir is NOT sufficient
    (seeds_per_class=2 and gap_filter reuse one photo across many run_dirs)."""
    if run_rel in _PHOTO_CACHE:
        return _PHOTO_CACHE[run_rel]
    g = list((REPO / str(run_rel)).glob("**/origin.png"))
    h = sha256_bytes(g[0].read_bytes())[:16] if g else f"NOPNG:{run_rel}"
    _PHOTO_CACHE[run_rel] = h
    return h


def add_helpers(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["pid"] = d["class_a"].astype(str) + "→" + d["class_b"].astype(str)
    if "sut" not in d:
        d["sut"] = d["sut_model_id"].map(lambda m: "qwen" if "Qwen" in str(m) else "llava")
    d["grp"] = d["sut"] + "|" + d["pid"]
    d["photo"] = d["run_rel"].map(photo_of)          # anchor-photo identity
    return d


USED_PHOTOS: set = set()  # anchor-photo hashes consumed (pool-wide PHOTO-disjoint)


def select_diverse(d: pd.DataFrame, n: int, sort="tgtbal") -> pd.DataFrame:
    """Round-robin across `grp` (sut x pair), lowest-tgtbal-first within group."""
    d = d.sort_values(sort)
    groups = {k: list(v.index) for k, v in d.groupby("grp", sort=False)}
    keys = sorted(groups, key=lambda k: d.loc[groups[k][0], sort])  # best group first
    picked = []
    while len(picked) < n and any(groups.values()):
        for k in keys:
            if groups[k]:
                picked.append(groups[k].pop(0))
                if len(picked) >= n:
                    break
    return d.loc[picked]


def _seed_unique(x):
    """One best-tgtbal row per anchor PHOTO, excluding photos already used anywhere
    in the pool (pool-wide photo-disjoint: no anchor image appears twice)."""
    x = x.sort_values("tgtbal").drop_duplicates("photo", keep="first")
    return x[~x.photo.isin(USED_PHOTOS)]


def pick_boundary(d, n, kind):
    """Closeness-first floors + SEED-DISJOINT (>=1 anchor image per item, pool-wide).
    Relaxes floors per-stratum only if the distinct-seed pool can't otherwise fill."""
    base = d[d.png_ready] if ("png_ready" in d and kind in ("image", "both")) else d
    steps = [
        ("≤1e-3 · d_img≤0.05" + (" · tg≤7" if kind in ("prompt", "both") else "") + " · 1/seed",
            lambda x: x[(x.tgtbal <= TGT_TIGHT) & (x.d_img_matrix <= D_IMG_CAP)
                        & ((x.n_active_text_genes <= TG_CAP) if kind in ("prompt", "both") else True)]),
        ("relax tg≤10 · 1/seed", lambda x: x[(x.tgtbal <= TGT_TIGHT) & (x.d_img_matrix <= D_IMG_CAP)
                        & ((x.n_active_text_genes <= 10) if kind in ("prompt", "both") else True)]),
        ("relax tgtbal≤1e-2 · 1/seed", lambda x: x[(x.tgtbal <= 1e-2) & (x.d_img_matrix <= 0.1)]),
    ]
    for note, f in steps:
        cand = _seed_unique(f(base))
        if len(cand) >= n:
            return select_diverse(cand, n), note
    cand = _seed_unique(base).sort_values("tgtbal").head(n)
    return cand, f"DATA-LIMITED: {len(_seed_unique(base))} distinct-seed candidates"


# ───────────────────────────────────────────────────────────────────────────
SOURCES: list[dict] = []
ITEMS: list[dict] = []
_seen_src: dict[str, dict] = {}
REPORT = {}


def copy_image(src_rel, source_id: str):
    """Copy a repo-relative image into assets/; return (uri, sha256, w, h) or None."""
    if src_rel is None or (not isinstance(src_rel, str) and pd.isna(src_rel)):
        return None
    src = REPO / src_rel
    if not src.exists():
        return None
    ASSETS.mkdir(parents=True, exist_ok=True)
    dst = ASSETS / f"{source_id}.png"
    shutil.copy2(src, dst)
    b = dst.read_bytes()
    with Image.open(dst) as im:
        w, h = im.size
    return f"assets/images/{source_id}.png", sha256_bytes(b), w, h


def boundary_source(r, source_id):
    txt = r.get("decoded_text") or r.get("prompt_template") or ""
    orig = r.get("prompt_template")
    manipulated = bool(r.get("n_active_text_genes", 0)) and r.get("modality") != "image_only"
    img = copy_image(r["pareto_png"], source_id)
    gen = int(r["generation"]) if pd.notna(r.get("generation")) and r["generation"] >= 0 else None
    pidx = r.get("pareto_idx")
    indiv = f"pareto-{int(pidx)}" if pd.notna(pidx) else (
        f"gen{gen}-ind{int(r['individual'])}" if gen is not None else None)
    crossed = (bool(r["p_class_b"] > r["p_class_a"])
               if pd.notna(r.get("p_class_a")) and pd.notna(r.get("p_class_b")) else None)
    ac = r.get("anchor_class_concrete") or r.get("class_a")
    tc = r.get("target_class_concrete") or r.get("class_b")
    aw = r.get("anchor_label_in_prompt") or r.get("class_a")
    tw = r.get("target_label_in_prompt") or r.get("class_b")
    src = {
        "source_id": source_id,
        "x_seed_key": r["photo"],   # anchor-PHOTO hash; the no-repeat key (run-dir is not unique)
        "origin": "boundary_individual",
        "experiment_ref": {
            "experiment_id": str(r["experiment"]),
            "run_id": str(r["run_dir"]),
            "seed_index": int(r["seed_idx"]) if pd.notna(r.get("seed_idx")) else None,
            "generation": gen,
            "individual_id": indiv,
        },
        "sut": {
            "model_id": str(r["sut_model_id"]),
            "backend": "torch-mps" if "Qwen" in str(r["sut_model_id"]) else "openvino",
            "scoring": "teacher-forced-logprob",
        },
        "cell": {
            "anchor_class": str(ac), "target_class": str(tc),
            "anchor_word": str(aw), "target_word": str(tw),
            "level_anchor": int(r["level_anchor"]) if pd.notna(r.get("level_anchor")) else None,
            "level_target": int(r["level_target"]) if pd.notna(r.get("level_target")) else None,
            "direction": None, "bucket_relation": None,
        },
        "search": {
            "modality": r.get("modality"),
            "tgtbal": float(r["tgtbal"]),
            "crossed": crossed,
            "gen_first_cross": None,
        },
        "drift": {
            "d_text": (None if pd.isna(r.get("d_text_embed")) else float(r["d_text_embed"])),
            "d_img": float(r["d_img_matrix"]),
            "active_text_genes": int(r.get("n_active_text_genes", 0)),
            "hamming_to_anchor_norm": None,
        },
        "strata": {"text": None, "image": None, "pair": None},  # set by caller
        "assets": {
            "prompt": {
                "text": txt, "sha256": sha256_text(txt), "char_count": len(txt),
                "contains_homoglyphs": any(ord(c) > 127 for c in txt),
                "original_text": (orig if manipulated else None),
            },
            "image": ({"uri": img[0], "sha256": img[1], "width": img[2],
                       "height": img[3], "format": "png"} if img else None),
        },
    }
    return src


def register(src):
    SOURCES.append(src)
    _seen_src[src["source_id"]] = src
    return src


def add_item(prefix, source_id, kind, extra=None):
    it = {"item_id": f"{prefix}-{source_id[4:]}", "kind": kind, "source_id": source_id}
    if extra:
        it.update(extra)
    ITEMS.append(it)


def stratum_report(name, rows, note):
    if rows is None or len(rows) == 0:
        REPORT[name] = {"selected": 0, "note": note}
        return
    REPORT[name] = {
        "selected": len(rows), "note": note,
        "tgtbal_min": float(rows.tgtbal.min()), "tgtbal_med": float(rows.tgtbal.median()),
        "tgtbal_max": float(rows.tgtbal.max()),
        "d_img_med": float(rows.d_img_matrix.median()),
        "tg_med": float(rows.n_active_text_genes.median()) if "n_active_text_genes" in rows else None,
        "n_pairs": int(rows.pid.nunique()), "suts": sorted(set(rows.sut)),
    }


# ── BOUNDARY STRATA ─────────────────────────────────────────────────────────
def do_boundary(phase, stratum, kind):
    d = add_helpers(load(phase, stratum))
    n = TARGETS[phase][stratum]
    rows, note = pick_boundary(d, n, kind)
    USED_PHOTOS.update(rows.photo.tolist())    # pool-wide PHOTO-disjoint
    stratum_report(f"{phase}/{stratum}", rows, note)
    out = []
    for _, r in rows.iterrows():
        sid = _slug(f"src-{r['experiment']}-{r['sut']}-s{r['seed_idx']}-g"
                    f"{int(r['generation']) if r['generation']>=0 else 'x'}-p{r.get('pareto_idx')}")
        if sid in _seen_src:
            src = _seen_src[sid]
        else:
            src = register(boundary_source(r, sid))
        out.append(src)
    return out


# ── JOINT BOUNDARY STRATA — ROUND-ROBIN photo-disjoint allocation ────────────
# All 6 joint strata draw from the SAME ~38 distinct yielding photos (one run
# yields individuals at several drift levels, but a photo can serve only ONE
# stratum under the no-repeat rule). Fill-one-then-next starves later strata, so
# allocate round-robin: each cycle every unfilled stratum claims its next-best
# UNUSED photo. The 38 photos spread ~evenly (~6/stratum) instead of 8/14/0/0.
ITEMPREFIX = {"text": "txt", "image": "img", "pair": "pair"}
JOINT = [("text", "low_drift", "prompt"), ("pair", "balanced", "both"),
         ("pair", "text_heavy", "both"), ("text", "medium_drift", "prompt"),
         ("text", "high_drift", "prompt"), ("image", "boundary_joint", "image")]
jstate = [{"phase": p, "stratum": s, "kind": k, "d": add_helpers(load(p, s)),
           "need": TARGETS[p][s], "rows": []} for p, s, k in JOINT]
progress = True
while progress:
    progress = False
    for st in jstate:
        if len(st["rows"]) >= st["need"]:
            continue
        got, _ = pick_boundary(st["d"], 1, st["kind"])   # next-best UNUSED photo
        if len(got) == 0:
            continue
        r = got.iloc[0]; USED_PHOTOS.add(r["photo"]); st["rows"].append(r); progress = True
for st in jstate:
    phase, stratum = st["phase"], st["stratum"]
    rdf = pd.DataFrame(st["rows"]) if st["rows"] else st["d"].iloc[0:0]
    stratum_report(f"{phase}/{stratum}", rdf,
                   f"round-robin photo-disjoint ({len(st['rows'])}/{st['need']})")
    for r in st["rows"]:
        sid = _slug(f"src-{r['experiment']}-{r['sut']}-s{r['seed_idx']}-g"
                    f"{int(r['generation']) if r['generation']>=0 else 'x'}-p{r.get('pareto_idx')}")
        src = register(boundary_source(r, sid))
        src["strata"][phase] = stratum
        add_item(ITEMPREFIX[phase], src["source_id"], phase)

# ── image_heavy — DATA-CAPPED at the distinct promoted anchor images ──────────
# Only 6 HS-GEN-01 promoted runs exist; every image-only item is a perturbation of
# one of 6 photos. Seed-disjoint => ONE representative (closest tgtbal) per run.
# Those 6 seeds serve BOTH phases — the single allowed cross-phase reuse; the form
# builder keeps them apart per rater via x_seed_key (only 6 anchor images exist).
d_ih = add_helpers(load("pair", "image_heavy"))
ih = _seed_unique(d_ih[d_ih.png_ready]).sort_values("tgtbal").reset_index(drop=True)
# FULL seed-disjoint (study-owner directive: no seed twice, even across phases):
# each anchor image serves EXACTLY ONE phase. Reserve up to its target (6) for the
# image phase via odd-rank interleave; the closest (rank 0) + the rest go to the
# pair phase (the H-HS3 modality-asymmetry core, design target 14).
img_quota = min(TARGETS["image"]["image_heavy"], len(ih) // 2)
image_idx = list(ih.index[1::2])[:img_quota]
image_rows = ih.loc[image_idx]
pair_rows = ih.drop(image_idx)
USED_PHOTOS.update(ih.photo.tolist())
stratum_report("pair/image_heavy", pair_rows,
               f"{len(pair_rows)} distinct seeds (design 14 — DATA-CAPPED at distinct "
               f"promoted anchor images; full seed-disjoint, 1 phase per seed)")
stratum_report("image/image_heavy", image_rows,
               f"{len(image_rows)} distinct seeds (design 6; full seed-disjoint)")
for tag, rows, prefix, kind in [("pair", pair_rows, "pair", "pair"),
                                ("image", image_rows, "img", "image")]:
    for _, r in rows.iterrows():
        sid = _slug(f"src-{r['experiment']}-{r['sut']}-s{r['seed_idx']}-g"
                    f"{int(r['generation']) if r['generation']>=0 else 'x'}-p{r.get('pareto_idx')}")
        src = register(boundary_source(r, sid))
        src["strata"][tag] = "image_heavy"
        add_item(prefix, sid, kind)

# ── CONTROLS — DISTINCT photos sourced from NON-boundary runs ────────────────
# After joint+image_heavy consume the yielding photos, controls must use photos
# NOT already in the pool. The richest source is the NON-yielding HS-GEN-02 runs
# (roster photos that didn't cross) + any unused Exp/roster origins. Each control
# is a distinct anchor photo, disjoint from boundary items and from each other.
def control_source(r, sid):
    txt = r.get("prompt_template") or "What is the main subject in this image?"
    img = copy_image(r["origin_png"], sid)
    ac = r.get("anchor_class_concrete"); tc = r.get("target_class_concrete")
    return register({
        "source_id": sid, "x_seed_key": r["photo"],
        "origin": "vqgan_roundtrip", "experiment_ref": {
            "experiment_id": str(r["experiment"]), "run_id": str(r["run_dir"]),
            "seed_index": int(r["seed_idx"]) if pd.notna(r.get("seed_idx")) else None,
            "generation": None, "individual_id": None},
        "sut": {"model_id": str(r["sut_model_id"]),
                "backend": "torch-mps" if r["sut"] == "qwen" else "openvino",
                "scoring": "teacher-forced-logprob"},
        "cell": ({"anchor_class": str(ac), "target_class": str(tc),
                  "anchor_word": str(r.get("anchor_label_in_prompt") or ac),
                  "target_word": str(r.get("target_label_in_prompt") or tc),
                  "level_anchor": int(r["level_anchor"]) if pd.notna(r.get("level_anchor")) else None,
                  "level_target": int(r["level_target"]) if pd.notna(r.get("level_target")) else None,
                  "direction": None, "bucket_relation": None} if pd.notna(ac) else None),
        "search": None, "drift": None,
        "strata": {"text": None, "image": None, "pair": None},
        "assets": {
            "prompt": {"text": txt, "sha256": sha256_text(txt), "char_count": len(txt),
                       "contains_homoglyphs": False, "original_text": None},
            "image": ({"uri": img[0], "sha256": img[1], "width": img[2],
                       "height": img[3], "format": "png"} if img else None)},
    })


def scan_control_candidates() -> pd.DataFrame:
    """One row per DISTINCT non-used anchor photo from the run dirs (origins)."""
    rows, seen = [], set()
    # incl. the HS-GEN-01 SCREEN (~150 fresh distinct photos across 50 classes,
    # mostly non-boundary) — the richest source of unused control anchor images.
    for exp, base, sub in [("HS-GEN-02", "runs/HS-GEN-02", None),
                           ("HS-GEN-01-screen", "runs/HS-GEN-01", "screen"),
                           ("Exp-101", "runs/Exp-101", None), ("Exp-102", "runs/Exp-102", None),
                           ("Exp-101q", "runs/Exp-101q", None),
                           ("Exp-100", "runs/Exp-100/poc_boundary_pair", None)]:
        bp = REPO / base
        if not bp.is_dir():
            continue
        for d in sorted(p for p in bp.iterdir() if p.is_dir() and (sub is None or sub in p.name)):
            ops = list(d.glob("**/origin.png")); sps = list(d.glob("**/stats.json"))
            if not ops or not sps:
                continue
            photo = sha256_bytes(ops[0].read_bytes())[:16]
            if photo in USED_PHOTOS or photo in seen:
                continue
            s = json.loads(sps[0].read_text()); sm = s.get("seed_metadata") or {}
            seen.add(photo)
            rows.append(dict(
                experiment=exp, run_dir=d.name, run_rel=str(d.relative_to(REPO)), photo=photo,
                origin_png=str(ops[0].relative_to(REPO)), sut_model_id=s.get("model_id"),
                sut="qwen" if "Qwen" in str(s.get("model_id")) else "llava",
                anchor_class_concrete=sm.get("anchor_class_concrete") or s.get("class_a"),
                target_class_concrete=sm.get("target_class_concrete") or s.get("class_b"),
                anchor_label_in_prompt=sm.get("anchor_label_in_prompt") or s.get("class_a"),
                target_label_in_prompt=sm.get("target_label_in_prompt") or s.get("class_b"),
                level_anchor=sm.get("level_anchor"), level_target=sm.get("level_target"),
                prompt_template=s.get("prompt_template"), seed_idx=s.get("seed_idx")))
    return pd.DataFrame(rows)

def take_distinct(df, n):
    df = df[~df.photo.isin(USED_PHOTOS)]
    by = {k: list(v.index) for k, v in df.groupby("anchor_class_concrete", sort=False)}
    keys = list(by); out = []
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k]:
                out.append(by[k].pop(0))
                if len(out) >= n: break
    sel = df.loc[out]; USED_PHOTOS.update(sel.photo.tolist()); return sel

ctrl_pool = scan_control_candidates()
_rost = ctrl_pool[ctrl_pool.anchor_class_concrete.isin(ROSTER_CLASSES)] if len(ctrl_pool) else ctrl_pool
rt = take_distinct(_rost if len(_rost) >= 6 else ctrl_pool, 6)   # roundtrip (image)
clean = take_distinct(ctrl_pool, 6)                             # clean (text)
basex = take_distinct(ctrl_pool, 8)                             # baseline (pair)

CONTROL_CLASSES = []
for i, (_, r) in enumerate(rt.iterrows()):
    sid = _slug(f"src-rt-{r['sut']}-{r['anchor_class_concrete']}-{i}")
    control_source(r, sid)["strata"].update(image="roundtrip")
    add_item("img", sid, "image"); CONTROL_CLASSES.append(r["anchor_class_concrete"])
for i, (_, r) in enumerate(clean.iterrows()):
    sid = _slug(f"src-clean-{r['sut']}-{r['anchor_class_concrete']}-{i}")
    control_source(r, sid)["strata"].update(text="clean")
    add_item("txt", sid, "text")
for i, (_, r) in enumerate(basex.iterrows()):
    sid = _slug(f"src-base-{r['sut']}-{r['anchor_class_concrete']}-{i}")
    control_source(r, sid)["strata"].update(pair="baseline")
    add_item("pair", sid, "pair")
REPORT["text/clean"] = {"selected": len(clean), "note": "clean seed prompt; distinct seeds"}
REPORT["image/roundtrip"] = {"selected": len(rt), "note": "round-trip origin.png; distinct seeds"}
REPORT["pair/baseline"] = {"selected": len(basex), "note": "round-trip image + clean prompt; distinct seeds"}

# ── image/raw (class-representative ImageNet originals; exact seed not logged)─
raw_map = {"junco": "junco", "ostrich": "ostrich", "green iguana": "green_iguana",
           "boa constrictor": "boa_constrictor", "cello": "cello", "marimba": "marimba"}
raw_ok = 0
# distinct raw classes (CONTROL_CLASSES can repeat a class), padded from the roster
raw_classes = list(dict.fromkeys(CONTROL_CLASSES))
raw_classes += [c for c in ROSTER_CLASSES if c not in raw_classes]
for cls in raw_classes[:TARGETS["image"]["raw"]]:
    folder = raw_map.get(cls) or _slug(cls).lower()
    cand = IMAGENET / folder
    pngs = sorted(cand.glob("*.png")) if cand.is_dir() else []
    sid = _slug(f"src-raw-{cls}")
    if not pngs:
        # placeholder: structure present, asset pending host-side recovery
        register({"source_id": sid, "x_seed_key": sid, "origin": "raw_original",
                  "experiment_ref": None, "sut": None, "cell": None, "search": None, "drift": None,
                  "strata": {"text": None, "image": "raw", "pair": None},
                  "assets": {"prompt": None, "image": None},
                  "x_asset_status": "PENDING: no cache image for class"})
        add_item("img", sid, "image")
        continue
    dst = ASSETS / f"{sid}.png"
    ASSETS.mkdir(parents=True, exist_ok=True)
    with Image.open(pngs[0]) as im:
        im = im.convert("RGB"); im.save(dst, "PNG"); w, h = im.size
    b = dst.read_bytes()
    register({"source_id": sid, "x_seed_key": sid, "origin": "raw_original",
              "experiment_ref": None, "sut": None, "cell": None, "search": None, "drift": None,
              "strata": {"text": None, "image": "raw", "pair": None},
              "assets": {"prompt": None,
                         "image": {"uri": f"assets/images/{sid}.png", "sha256": sha256_bytes(b),
                                   "width": w, "height": h, "format": "png"}},
              "x_asset_status": "class-representative ImageNet original (exact seed not logged)"})
    add_item("img", sid, "image"); raw_ok += 1
REPORT["image/raw"] = {"selected": len(CONTROL_CLASSES),
                       "note": f"{raw_ok} class-representative originals (NOT exact seed twins)"}

# ── attention checks (2) ────────────────────────────────────────────────────
# 1) nonsense prompt (text); expected low comprehension
nons = "Gr fnx ple wzzt kqj morb dlee?"
register({"source_id": "src-attn-nonsense-01", "x_seed_key": "src-attn-nonsense-01",
          "origin": "attention_synthetic",
          "experiment_ref": None, "sut": None, "cell": None, "search": None, "drift": None,
          "strata": {"text": "high_drift", "image": None, "pair": None},
          "assets": {"prompt": {"text": nons, "sha256": sha256_text(nons), "char_count": len(nons),
                                "contains_homoglyphs": False, "original_text": None}, "image": None}})
add_item("txt", "src-attn-nonsense-01", "text",
         {"is_attention_check": True, "check_rule": {"metric": "scale_leq", "value": 2}})
# 2) obvious-class clean pair: its OWN clear cache image (no control reuse)
oc = CONTROL_CLASSES[0] if CONTROL_CLASSES else "junco"
other = next((c for c in CONTROL_CLASSES if c != oc), "marimba")
opngs = sorted((IMAGENET / (raw_map.get(oc) or _slug(oc).lower())).glob("*.png"))
if opngs:
    sid = "src-attn-obvious-01"; dst = ASSETS / f"{sid}.png"
    with Image.open(opngs[-1]) as im:           # different index than raw's [0]
        im = im.convert("RGB"); im.save(dst, "PNG"); w, h = im.size
    txt = "What is the main subject in this image?"
    register({"source_id": sid, "x_seed_key": sid, "origin": "attention_synthetic",
              "experiment_ref": None, "sut": None,
              "cell": {"anchor_class": oc, "target_class": other, "anchor_word": oc,
                       "target_word": other, "level_anchor": None, "level_target": None,
                       "direction": None, "bucket_relation": None},
              "search": None, "drift": None,
              "strata": {"text": None, "image": None, "pair": "baseline"},
              "assets": {"prompt": {"text": txt, "sha256": sha256_text(txt), "char_count": len(txt),
                                    "contains_homoglyphs": False, "original_text": None},
                         "image": {"uri": f"assets/images/{sid}.png",
                                   "sha256": sha256_bytes(dst.read_bytes()),
                                   "width": w, "height": h, "format": "png"}}})
    add_item("pair", sid, "pair",
             {"is_attention_check": True, "check_rule": {"metric": "choice_equals", "value": oc}})

# ── assemble + write ────────────────────────────────────────────────────────
pool = {
    "schema_version": "1.0.0",
    "pool_id": "hs01-pool-v1",
    "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "frozen": True,
    "generator": {
        "pipeline_commit": None,
        "selection_script": "experiments/HS-01/freeze_pool.py",
        "notes": "Closeness-first selection (tgtbal<=1e-3) with clarity (d_img<=0.05) "
                 "and readability (text_genes<=7) floors; SUT+pair round-robin diversity. "
                 "Sources: LLaVA (Exp-100/101/102 + HS-GEN-01 image_only) + Qwen (Exp-101q).",
    },
    "composition_targets": TARGETS,
    "sources": SOURCES,
    "items": ITEMS,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "itempool.json").write_text(json.dumps(pool, indent=2, ensure_ascii=False))

# ── validate against the schema ─────────────────────────────────────────────
valid_msg = "skipped (jsonschema not available)"
try:
    import jsonschema
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator(schema).validate(pool)
    valid_msg = "VALID against hs01.itempool.schema.json (Draft 2020-12)"
except ImportError:
    pass
except Exception as e:
    valid_msg = f"INVALID: {e}"

# ── report ──────────────────────────────────────────────────────────────────
n_by = {}
for it in ITEMS:
    src = _seen_src[it["source_id"]]
    st = src["strata"].get(it["kind"])
    n_by[(it["kind"], st)] = n_by.get((it["kind"], st), 0) + 1

lines = [f"# HS-01 Frozen Pool — {pool['created']}", "",
         f"Schema validation: **{valid_msg}**",
         f"Sources: {len(SOURCES)} · Items: {len(ITEMS)} · Assets: {len(list(ASSETS.glob('*.png')))} PNGs",
         "", "## Per-stratum selection (target vs selected; closeness-first)", "",
         "| phase/stratum | target | selected | tgtbal min/med | d_img med | text-genes med | pairs | SUTs | note |",
         "|---|---|---|---|---|---|---|---|---|"]
for phase in ["text", "image", "pair"]:
    for stratum, tgt in TARGETS[phase].items():
        key = f"{phase}/{stratum}"
        r = REPORT.get(key, {})
        sel = r.get("selected", 0)
        tb = (f"{r['tgtbal_min']:.1e}/{r['tgtbal_med']:.1e}" if "tgtbal_min" in r else "—")
        di = (f"{r['d_img_med']:.3f}" if r.get("d_img_med") is not None else "—")
        tg = (f"{r['tg_med']:.0f}" if r.get("tg_med") is not None else "—")
        np_ = r.get("n_pairs", "—"); su = ",".join(r.get("suts", [])) or "—"
        flag = "" if sel == tgt else " ⚠"
        lines.append(f"| {key} | {tgt} | {sel}{flag} | {tb} | {di} | {tg} | {np_} | {su} | {r.get('note','')} |")
lines += ["", "## Attention checks", "- `txt-attn-nonsense-01` (scale_leq 2) · `pair-attn-obvious-01` (choice_equals anchor)",
          "", "## Folder structure", "```", "pool_frozen/", "  itempool.json",
          "  assets/images/<source_id>.png", "  POOL_README.md", "```",
          "", "## Selection policy & decisions",
          "- **FULL SEED-DISJOINT (no repeated-exposure confound, even across phases)**: every item comes from a distinct anchor PHOTO (sha of origin.png) used in EXACTLY ONE item/phase — run-dir is NOT unique (gap_filter/seeds_per_class=2 reuse one photo across many runs) — a rater can never see the same anchor image twice, in any phase. `x_seed_key` is carried on every source as a belt-and-suspenders key for the form builder.",
          "- **Closeness first**: primary sort = `tgtbal` asc; quality floors `tgtbal<=1e-3`, `d_img<=0.05`, `text_genes<=7` (relaxed per-stratum only if needed — see notes).",
          "- **Diversity**: round-robin across (SUT × label-pair), one best item per seed.",
          "- **image_heavy is DATA-CAPPED** (design target 14+6=20): there are only as many distinct image-only anchor images as promoted HS-GEN-01 runs. Under full seed-disjointness each anchor serves ONE phase (split: image phase up to 6, pair phase the rest). Expanded by generating more promoted pairs on the workstation (lean batch idx 332/727/586/839/630/58).",
          "- **controls** (clean/roundtrip/baseline): now use 20 DISTINCT seeds (no shared source across phases). image = `origin.png` (round-trip, native res — not 256×256). *Refinement: render 256 round-trips for resolution parity.*",
          "- **image/raw**: class-representative ImageNet originals (exact seed file never logged). *Refinement: host-side exact-seed recovery, or drop the codec-cost micro-control.*",
          "- **SUT balance (trade-off of closeness-first)**: LLaVA has the tighter boundaries, so the pool is LLaVA-heavy; Qwen survives mainly in low_drift / boundary_joint / balanced. Not model-balanced (HS-01 §4). **Lever**: enforce a per-stratum Qwen minimum at a small closeness cost."]
(OUT / "POOL_README.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nitems by (kind,stratum): {n_by}")
