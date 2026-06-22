"""RefCOCO+ two-referent seed source for grounding-modality runs.

Yields SeedTriples whose class_a/class_b are the two referents' boxes as
coordinate strings in the SUT's output space, so TargetedBalance scores
|lp(box_A) - lp(box_B)| via the existing teacher-forced path.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
from PIL import Image
from src.config import SeedTriple, ExperimentConfig

logger = logging.getLogger(__name__)


def normalize_box(bbox_px: tuple[int, int, int, int], img_w: int, img_h: int,
                  space: str) -> str:
    """Format a pixel bbox as a candidate string in the SUT's coordinate space."""
    x1, y1, x2, y2 = bbox_px
    if space == "norm_1000":
        x1 = round(x1 / img_w * 1000); x2 = round(x2 / img_w * 1000)
        y1 = round(y1 / img_h * 1000); y2 = round(y2 / img_h * 1000)
    elif space != "abs_pixels":
        raise ValueError(f"unknown coordinate_space {space!r}")
    return f"[{x1}, {y1}, {x2}, {y2}]"


def build_seed_triples(items: list[dict[str, Any]], coordinate_space: str) -> list[SeedTriple]:
    """Pure transform: list of two-referent item dicts -> list[SeedTriple]."""
    seeds: list[SeedTriple] = []
    for it in items:
        a, b = it["ref_a"], it["ref_b"]
        meta = {
            "prompt_template": f"Locate the {it['referent']}.",
            "referent": it["referent"],
            "image_id": it["image_id"],
            "ref_id_a": a["ref_id"], "ref_id_b": b["ref_id"],
            "bbox_a_px": list(a["bbox_px"]), "bbox_b_px": list(b["bbox_px"]),
            "coordinate_space": coordinate_space,
        }
        seeds.append(SeedTriple(
            image=it["image"],
            class_a=normalize_box(a["bbox_px"], it["image_w"], it["image_h"], coordinate_space),
            class_b=normalize_box(b["bbox_px"], it["image_w"], it["image_h"], coordinate_space),
            metadata=meta,
        ))
    return seeds


def _load_refcocoplus_items(cfg, n_items: int) -> list[dict[str, Any]]:
    """Load up to n_items two-same-category-referent items via the refer API.

    Lazy import so the dependency is only needed for grounding runs.
    """
    import sys
    sys.path.insert(0, str(Path("tools/refer").expanduser()))
    from refer import REFER  # lichengunc/refer

    root = str(Path(cfg.data_root).expanduser())
    refer = REFER(root, dataset="refcoco+", splitBy=cfg.splitBy)
    ref_ids = refer.getRefIds(split=cfg.split)
    # group refs by (image_id, category_id); keep images with >=2 refs of one category
    by_img_cat: dict[tuple[int, int], list[int]] = {}
    for rid in ref_ids:
        ref = refer.Refs[rid]
        by_img_cat.setdefault((ref["image_id"], ref["category_id"]), []).append(rid)

    items: list[dict[str, Any]] = []
    for (image_id, cat_id), rids in by_img_cat.items():
        if cfg.same_category and len(rids) < 2:
            continue
        img_info = refer.Imgs[image_id]
        img_path = Path(root) / "images" / "mscoco" / "images" / "train2014" / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        ra, rb = rids[0], rids[1]
        def _box(rid):  # refer.getRefBox returns [x, y, w, h]
            x, y, w, h = refer.getRefBox(rid)
            return (int(x), int(y), int(x + w), int(y + h))
        items.append({
            "image": image, "image_id": image_id,
            "image_w": img_info["width"], "image_h": img_info["height"],
            "referent": refer.Cats[cat_id],
            "ref_a": {"ref_id": ra, "bbox_px": _box(ra)},
            "ref_b": {"ref_id": rb, "bbox_px": _box(rb)},
        })
        if len(items) >= n_items:
            break
    logger.info("RefCOCO+ %s: %d two-referent items", cfg.split, len(items))
    return items


def refcocoplus_seeds(sut: Any, exp_cfg: ExperimentConfig, data_source: Any) -> list[SeedTriple]:
    """Entry point matching the other seed generators' (sut, cfg, data_source) signature."""
    cfg = exp_cfg.seeds.refcocoplus
    items = _load_refcocoplus_items(cfg, cfg.n_items)
    return build_seed_triples(items, exp_cfg.grounding.coordinate_space)
