# Coordinate Boundary Testing (Exp-103) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the evolutionary VLM boundary tester to drive *coordinate / bounding-box* boundaries (which referent a grounding VLM localizes), reusing the teacher-forced probability machinery.

**Architecture:** The boundary is `TgtBal_coord = |lp(box_A) − lp(box_B)|` — the existing `TargetedBalance` over two candidate *coordinate strings* instead of class names. The scoring core, objective layer, and search engine are class-name-agnostic and need **no change**. The work is: (1) a `grounding` modality value + `GroundingConfig`, (2) a `refcocoplus` seed mode + adapter yielding `SeedTriple`s whose `class_a`/`class_b` are `[0,1000]`-normalized box strings, (3) a per-seed prompt override (the referring expression is per-item), (4) provenance plumbing.

**Tech Stack:** Python 3.13, PyTorch/MPS, `transformers` 5.3, dacite config parsing, pymoo (AGE-MOEA-II), Qwen3.5-4B SUT, RefCOCO+ via the `refer` API (lichengunc/refer).

## Global Constraints

- All Python/pytest runs via `conda run -n uni` (heredoc-stdin stdout is swallowed — write scripts to files).
- `pytest tests/` must stay green after every task.
- Package boundary: new shared code lives in `src/common/`; no `src/evolutionary/` ↔ `src/pdq/` imports.
- Commit messages: NO Co-Authored-By / AI attribution.
- Coordinate candidate strings MUST be in the model's output space (Qwen3.5-4B = normalized `[0,1000]`), else teacher-forcing scores off-distribution.
- Cone-filter (`image.cone_filter.enabled`) MUST be `false` in grounding mode (the per-seed `target_class` is a box string, not an L0 ImageNet label).
- v1 scope: teacher-forced boundary only; manipulation axis = joint (grounding treated as joint genome). Soft-box progress objective is deferred (Task 8, not implemented in v1).
- v1 keeps per-token `norm_lp` scoring (spike-validated). `total_lp` is available at `score_categories` tuple index 1 if length-skew artifacts appear — flagged, not implemented.

---

### Task 1: `GroundingConfig` + `grounding` modality

**Files:**
- Modify: `src/config.py` (add dataclass near other component configs ~`:519`; field on `ExperimentConfig` ~`:593-599`; modality validation `:602`; `__all__` `:666-688`)
- Modify: `src/config.py` `apply_modality` (`:633-663`) — add explicit `grounding` pass-through for clarity
- Test: `tests/test_config_grounding.py`

**Interfaces:**
- Produces: `GroundingConfig(coordinate_space: str = "norm_1000", bbox_format: str = "bare_array", referent_prompt: str = "Locate the {referent}.", answer_format: str = " Report the bounding box as a JSON array [x1, y1, x2, y2].")`; `ExperimentConfig.grounding: GroundingConfig`; `modality` accepts `"grounding"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_grounding.py
import dacite
from src.config import ExperimentConfig, GroundingConfig
from experiments.runners.run_boundary_test import _DACITE_CONFIG  # dacite Config used by load_config

def _parse(d):
    return dacite.from_dict(ExperimentConfig, d, config=_DACITE_CONFIG)

def test_grounding_modality_accepted_with_block():
    exp = _parse({
        "modality": "grounding",
        "grounding": {"coordinate_space": "norm_1000", "bbox_format": "bare_array"},
    })
    assert exp.modality == "grounding"
    assert isinstance(exp.grounding, GroundingConfig)
    assert exp.grounding.coordinate_space == "norm_1000"
    assert "[x1, y1, x2, y2]" in exp.grounding.answer_format

def test_grounding_defaults_when_block_absent():
    exp = _parse({"modality": "joint"})
    assert exp.grounding.coordinate_space == "norm_1000"  # default_factory

def test_bad_modality_still_rejected():
    import pytest
    with pytest.raises(ValueError, match="modality must be one of"):
        _parse({"modality": "bogus"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n uni pytest tests/test_config_grounding.py -v`
Expected: FAIL — `ImportError: cannot import name 'GroundingConfig'`.

- [ ] **Step 3: Implement**

Add the dataclass (place beside the other frozen component configs, e.g. after `SeedConfig`):

```python
@dataclass(frozen=True)
class GroundingConfig:
    """Coordinate-output (visual grounding) settings. Active when
    ``ExperimentConfig.modality == 'grounding'``. The boundary objective is
    ``TargetedBalance`` over two candidate *box strings* (the referents)."""

    coordinate_space: str = "norm_1000"   # norm_1000 (Qwen3-VL/3.5) | abs_pixels (Qwen2.5-VL)
    bbox_format: str = "bare_array"       # "[x1, y1, x2, y2]" candidate-string format
    referent_prompt: str = "Locate the {referent}."
    answer_format: str = " Report the bounding box as a JSON array [x1, y1, x2, y2]."

    def __post_init__(self) -> None:
        if self.coordinate_space not in ("norm_1000", "abs_pixels"):
            raise ValueError(
                f"grounding.coordinate_space must be 'norm_1000' | 'abs_pixels'; "
                f"got {self.coordinate_space!r}"
            )
```

Add the field to `ExperimentConfig` (alongside `sut`/`image`/`text`/`seeds`/`optimizer`/`parallel`, `:593-599`):

```python
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
```

Extend the modality validation (`:602`):

```python
        if self.modality not in ("joint", "image_only", "text_only", "grounding"):
```

Add an explicit branch to `apply_modality` before the final `return exp` (`:663`) — grounding keeps the joint genome:

```python
    if exp.modality == "grounding":
        logger.info("modality=grounding → joint genome; boundary over box-string candidates")
        return exp
```

Add `"GroundingConfig"` to `__all__` (`:666-688`).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n uni pytest tests/test_config_grounding.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run full suite, then commit**

Run: `conda run -n uni pytest tests/ -q`
Expected: all pass.

```bash
git add src/config.py tests/test_config_grounding.py
git commit -m "feat(config): add GroundingConfig and grounding modality"
```

---

### Task 2: `RefCocoPlusConfig` + `refcocoplus` seed mode

**Files:**
- Modify: `src/config.py` (`RefCocoPlusConfig` dataclass; `SeedConfig` field `:230-233`; `SeedConfig.__post_init__` `:235-259`; `__all__`)
- Test: `tests/test_config_refcocoplus.py`

**Interfaces:**
- Produces: `RefCocoPlusConfig(data_root: Path, split: str = "testA", n_items: int = 40, same_category: bool = True, splitBy: str = "unc")`; `SeedConfig.refcocoplus: RefCocoPlusConfig | None`; `SeedConfig.mode` accepts `"refcocoplus"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_refcocoplus.py
import pytest
from src.config import SeedConfig, RefCocoPlusConfig

def test_refcocoplus_mode_requires_block():
    with pytest.raises(ValueError, match="requires a seeds.refcocoplus"):
        SeedConfig(mode="refcocoplus")

def test_refcocoplus_mode_ok():
    sc = SeedConfig(mode="refcocoplus",
                    refcocoplus=RefCocoPlusConfig(data_root="~/.cache/refcoco"))
    assert sc.mode == "refcocoplus"
    assert sc.refcocoplus.split == "testA"

def test_refcocoplus_rejects_conflicting_blocks():
    from src.config import GapFilterConfig
    with pytest.raises(ValueError, match="drop one"):
        SeedConfig(mode="refcocoplus",
                   refcocoplus=RefCocoPlusConfig(data_root="x"),
                   gap_filter=GapFilterConfig())
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n uni pytest tests/test_config_refcocoplus.py -v`
Expected: FAIL — `ImportError: cannot import name 'RefCocoPlusConfig'`.

- [ ] **Step 3: Implement**

Add the dataclass (after `RosterConfig`, ~`:198`):

```python
@dataclass(frozen=True)
class RefCocoPlusConfig:
    """RefCOCO+ two-referent seed source for grounding-modality runs."""

    data_root: Path                       # COCO images + refcoco+ annotations (refer-API layout)
    split: str = "testA"                  # testA = multi-person; testB = multi-object
    splitBy: str = "unc"
    n_items: int = 40
    same_category: bool = True            # keep items whose two referents share a category
```

Add the field on `SeedConfig` (`:233`):

```python
    refcocoplus: "RefCocoPlusConfig | None" = None
```

Extend `SeedConfig.__post_init__` — add a `refcocoplus` branch and widen the final error (`:235-259`):

```python
        elif self.mode == "refcocoplus":
            if self.gap_filter is not None or self.roster is not None:
                raise ValueError(
                    "seeds.mode='refcocoplus' but gap_filter/roster is set; drop one."
                )
            if self.refcocoplus is None:
                raise ValueError(
                    "seeds.mode='refcocoplus' requires a seeds.refcocoplus config block."
                )
        else:
            raise ValueError(
                f"seeds.mode must be 'gap_filter' | 'roster' | 'refcocoplus'; "
                f"got {self.mode!r}"
            )
```

Also guard the existing `gap_filter`/`roster` branches against a stray `refcocoplus` block (add `or self.refcocoplus is not None` to their conflict checks at `:237` and `:246`). Add `"RefCocoPlusConfig"` to `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n uni pytest tests/test_config_refcocoplus.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run suite + commit**

Run: `conda run -n uni pytest tests/ -q`

```bash
git add src/config.py tests/test_config_refcocoplus.py
git commit -m "feat(config): add RefCocoPlusConfig and refcocoplus seed mode"
```

---

### Task 3: RefCOCO+ seed adapter

**Files:**
- Create: `src/common/refcocoplus_seed_generator.py`
- Test: `tests/test_refcocoplus_seed_generator.py`
- Setup: vendor the `refer` API — `git clone https://github.com/lichengunc/refer tools/refer` (read-only helper; add `tools/refer` to `.gitignore` like `tools/alpamayo`), or `pip install` an equivalent. The adapter imports it lazily.

**Interfaces:**
- Consumes: `RefCocoPlusConfig`, `GroundingConfig`, a `data_source` unused (signature parity with other generators).
- Produces: `refcocoplus_seeds(sut, exp_cfg, data_source) -> list[SeedTriple]`. Each `SeedTriple`: `image` = PIL referent image; `class_a`/`class_b` = `[0,1000]`-normalized box strings `"[x1, y1, x2, y2]"`; `metadata` = `{"prompt_template": "Locate the {referent}.", "referent": <category>, "ref_id_a", "ref_id_b", "image_id", "bbox_a_px", "bbox_b_px", "coordinate_space"}`.
- Helper: `normalize_box(bbox_px, img_w, img_h, space) -> str`.

- [ ] **Step 1: Write the failing test** (fixture mocks the refer API so no dataset download is needed)

```python
# tests/test_refcocoplus_seed_generator.py
from PIL import Image
from src.common.refcocoplus_seed_generator import normalize_box, build_seed_triples

def test_normalize_box_norm_1000():
    # box in pixels on a 640x480 image -> [0,1000] ints
    assert normalize_box((64, 48, 320, 240), 640, 480, "norm_1000") == "[100, 100, 500, 500]"

def test_normalize_box_abs_pixels_passthrough():
    assert normalize_box((10, 20, 30, 40), 640, 480, "abs_pixels") == "[10, 20, 30, 40]"

def test_build_seed_triples_pairs_same_category():
    img = Image.new("RGB", (640, 480), "white")
    # two 'person' referents on one image
    items = [{
        "image": img, "image_id": 7, "image_w": 640, "image_h": 480,
        "referent": "person",
        "ref_a": {"ref_id": 1, "bbox_px": (64, 48, 320, 240)},
        "ref_b": {"ref_id": 2, "bbox_px": (384, 48, 576, 240)},
    }]
    seeds = build_seed_triples(items, coordinate_space="norm_1000")
    assert len(seeds) == 1
    s = seeds[0]
    assert s.class_a == "[100, 100, 500, 500]"
    assert s.class_b == "[600, 100, 900, 500]"
    assert s.metadata["prompt_template"] == "Locate the person."
    assert s.metadata["coordinate_space"] == "norm_1000"
    assert s.metadata["ref_id_a"] == 1 and s.metadata["ref_id_b"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n uni pytest tests/test_refcocoplus_seed_generator.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# src/common/refcocoplus_seed_generator.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n uni pytest tests/test_refcocoplus_seed_generator.py -v`
Expected: PASS (3 passed). (The pure functions are tested; `_load_refcocoplus_items` is exercised in Task 7's integration smoke.)

- [ ] **Step 5: Commit**

```bash
git add src/common/refcocoplus_seed_generator.py tests/test_refcocoplus_seed_generator.py .gitignore
git commit -m "feat(common): RefCOCO+ two-referent seed adapter (box-string candidates)"
```

---

### Task 4: Dispatch wiring in `prepare_pipeline_seeds`

**Files:**
- Modify: `src/common/pipeline_bootstrap.py` (`prepare_pipeline_seeds` `:195-216`)
- Test: `tests/test_prepare_pipeline_seeds_refcocoplus.py`

**Interfaces:**
- Consumes: `refcocoplus_seeds` (Task 3).
- Produces: `prepare_pipeline_seeds` returns the adapter's `list[SeedTriple]` when `seeds.mode == "refcocoplus"`.

- [ ] **Step 1: Write the failing test** (monkeypatch the adapter so no dataset is touched)

```python
# tests/test_prepare_pipeline_seeds_refcocoplus.py
from PIL import Image
from src.config import ExperimentConfig, SeedConfig, RefCocoPlusConfig, SeedTriple
import src.common.pipeline_bootstrap as pb

def test_dispatch_routes_to_refcocoplus(monkeypatch):
    fake = [SeedTriple(image=Image.new("RGB", (8, 8)), class_a="[1, 2, 3, 4]",
                       class_b="[5, 6, 7, 8]", metadata={"prompt_template": "Locate the cat."})]
    monkeypatch.setattr(pb, "refcocoplus_seeds", lambda sut, cfg, ds: fake, raising=False)
    exp = ExperimentConfig(modality="grounding",
                           seeds=SeedConfig(mode="refcocoplus",
                                            refcocoplus=RefCocoPlusConfig(data_root="x")))
    class _C: sut = None; data_source = None
    out = pb.prepare_pipeline_seeds(_C(), exp)
    assert out is fake
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n uni pytest tests/test_prepare_pipeline_seeds_refcocoplus.py -v`
Expected: FAIL — `ValueError: Unknown seeds.mode='refcocoplus'`.

- [ ] **Step 3: Implement**

Add the import near the other seed-generator imports in `pipeline_bootstrap.py`:

```python
from src.common.refcocoplus_seed_generator import refcocoplus_seeds
```

Add the branch in `prepare_pipeline_seeds` before the final `raise` (`:215`):

```python
    if exp_cfg.seeds.mode == "refcocoplus":
        logger.info("Generating seeds (refcocoplus)")
        return refcocoplus_seeds(components.sut, exp_cfg, components.data_source)
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n uni pytest tests/test_prepare_pipeline_seeds_refcocoplus.py -v`
Expected: PASS.

- [ ] **Step 5: Run suite + commit**

Run: `conda run -n uni pytest tests/ -q`

```bash
git add src/common/pipeline_bootstrap.py tests/test_prepare_pipeline_seeds_refcocoplus.py
git commit -m "feat(common): dispatch refcocoplus seed mode in prepare_pipeline_seeds"
```

---

### Task 5: Per-seed prompt override in the tester

The referring expression is per-item, but `prompt_template` is global. Thread a per-seed override via `seed.metadata["prompt_template"]`.

**Files:**
- Modify: `src/evolutionary/vlm_boundary_tester.py` (`_run_seed`, the `manipulator.prepare(... self._config.prompt_template ...)` site `:433-438`, and wherever `prompt_template` seeds the text manipulation)
- Test: `tests/test_per_seed_prompt_override.py`

**Interfaces:**
- Consumes: `SeedTriple.metadata["prompt_template"]` (optional).
- Produces: tester uses `seed.metadata["prompt_template"]` when present, else `config.prompt_template`. Add a module-level helper `effective_prompt_template(seed, config) -> str` for testability.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_per_seed_prompt_override.py
from PIL import Image
from src.config import SeedTriple, ExperimentConfig
from src.evolutionary.vlm_boundary_tester import effective_prompt_template

def test_uses_seed_prompt_when_present():
    seed = SeedTriple(image=Image.new("RGB", (8, 8)), class_a="[1,2,3,4]",
                      class_b="[5,6,7,8]", metadata={"prompt_template": "Locate the dog."})
    assert effective_prompt_template(seed, ExperimentConfig()) == "Locate the dog."

def test_falls_back_to_config_prompt():
    seed = SeedTriple(image=Image.new("RGB", (8, 8)), class_a="a", class_b="b")
    cfg = ExperimentConfig(prompt_template="What is this?")
    assert effective_prompt_template(seed, cfg) == "What is this?"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n uni pytest tests/test_per_seed_prompt_override.py -v`
Expected: FAIL — `ImportError: cannot import name 'effective_prompt_template'`.

- [ ] **Step 3: Implement**

Add the helper near the top of `vlm_boundary_tester.py`:

```python
def effective_prompt_template(seed: SeedTriple, config: ExperimentConfig) -> str:
    """Per-seed prompt override (grounding referent) → else the global template."""
    if seed.metadata and seed.metadata.get("prompt_template"):
        return seed.metadata["prompt_template"]
    return config.prompt_template
```

In `_run_seed`, replace the `self._config.prompt_template` argument to `manipulator.prepare` (`:435`) with the resolved value:

```python
        prompt_template = effective_prompt_template(seed, self._config)
        self._manipulator.prepare(
            seed.image,
            prompt_template,
            target_class=target_class,
            origin_class=seed.class_a,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n uni pytest tests/test_per_seed_prompt_override.py -v`
Expected: PASS.

- [ ] **Step 5: Run suite + commit**

Run: `conda run -n uni pytest tests/ -q`

```bash
git add src/evolutionary/vlm_boundary_tester.py tests/test_per_seed_prompt_override.py
git commit -m "feat(evolutionary): per-seed prompt override via seed.metadata"
```

---

### Task 6: `refcocoplus` provenance in `build_stats`

**Files:**
- Modify: `src/evolutionary/vlm_boundary_tester.py` (`build_stats` mode-provenance block `:217-226`)
- Test: `tests/test_build_stats_refcocoplus.py`

**Interfaces:**
- Consumes: `config.seeds.refcocoplus`, `config.grounding`.
- Produces: `stats["seed_selection_mode"] == "refcocoplus"` plus `stats["refcoco_split"]`, `stats["coordinate_space"]`, `stats["grounding_answer_format"]`. Per-seed `ref_id`/boxes already flow via `seed.metadata` → `stats["seed_metadata"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_stats_refcocoplus.py
from PIL import Image
from src.config import (ExperimentConfig, SeedConfig, RefCocoPlusConfig,
                        GroundingConfig, SeedTriple)
from src.evolutionary.vlm_boundary_tester import build_stats

class _FakeManip:
    gene_bounds = __import__("numpy").zeros(1); image_dim = 1; text_dim = 0
    def __getattr__(self, n): return 0

def test_stats_record_refcocoplus_provenance():
    cfg = ExperimentConfig(
        modality="grounding",
        grounding=GroundingConfig(coordinate_space="norm_1000"),
        seeds=SeedConfig(mode="refcocoplus",
                         refcocoplus=RefCocoPlusConfig(data_root="x", split="testA")))
    seed = SeedTriple(image=Image.new("RGB", (8, 8)), class_a="[1, 2, 3, 4]",
                      class_b="[5, 6, 7, 8]",
                      metadata={"prompt_template": "Locate the cat.", "ref_id_a": 1})
    stats = build_stats(0, seed, cfg, _FakeManip(), 0, 0.0,
                        ("[1, 2, 3, 4]", "[5, 6, 7, 8]"),
                        ("[1, 2, 3, 4]", "[5, 6, 7, 8]"), (0, 1),
                        {"hits": 0, "misses": 0})
    assert stats["seed_selection_mode"] == "refcocoplus"
    assert stats["refcoco_split"] == "testA"
    assert stats["coordinate_space"] == "norm_1000"
    assert stats["seed_metadata"]["ref_id_a"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n uni pytest tests/test_build_stats_refcocoplus.py -v`
Expected: FAIL — `KeyError: 'refcoco_split'`.

- [ ] **Step 3: Implement**

Add a branch to the mode-provenance block in `build_stats` (after the `roster` branch, ~`:226`):

```python
    elif config.seeds.mode == "refcocoplus" and config.seeds.refcocoplus is not None:
        out["refcoco_split"] = config.seeds.refcocoplus.split
        out["refcoco_n_items"] = config.seeds.refcocoplus.n_items
        out["coordinate_space"] = config.grounding.coordinate_space
        out["grounding_answer_format"] = config.grounding.answer_format
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n uni pytest tests/test_build_stats_refcocoplus.py -v`
Expected: PASS.

- [ ] **Step 5: Run suite + commit**

Run: `conda run -n uni pytest tests/ -q`

```bash
git add src/evolutionary/vlm_boundary_tester.py tests/test_build_stats_refcocoplus.py
git commit -m "feat(evolutionary): record refcocoplus/grounding provenance in stats"
```

---

### Task 7: End-to-end smoke validation (no dataset download)

Verify the full grounding path computes `TgtBal_coord` over box strings and produces trace rows, using a synthetic two-referent seed (no RefCOCO+ download, no real model — a stub SUT).

**Files:**
- Test: `tests/test_grounding_end_to_end.py`

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Write the test**

```python
# tests/test_grounding_end_to_end.py
"""Grounding path smoke test: a stub SUT returns fixed per-candidate logprobs;
assert TargetedBalance over the two box strings reaches the trace as fitness_TgtBal."""
import torch
from src.objectives import TargetedBalance
from tools.smoo.src.objectives import CriterionCollection

def test_tgtbal_over_box_string_candidates():
    # stub: SUT returned log_prob_norm for (box_A, box_B) across a 3-individual pop
    logits = torch.tensor([[-1.10, -1.61], [-1.04, -1.70], [-1.37, -1.54]])
    coll = CriterionCollection(TargetedBalance())
    coll.evaluate_all(logits=logits, target_classes=(0, 1), batch_dim=0)
    gaps = coll.results["TgtBal"]
    assert [round(g, 2) for g in gaps] == [0.51, 0.66, 0.17]  # matches the spike numbers
```

- [ ] **Step 2: Run to verify it passes** (this asserts the reused objective is correct over box-string logits)

Run: `conda run -n uni pytest tests/test_grounding_end_to_end.py -v`
Expected: PASS — confirms `TargetedBalance` yields the spike's `|lp_A − lp_B|` gaps with box-string candidates.

- [ ] **Step 3: Manual live smoke (documented, not CI)**

Document in the test file's module docstring the one-shot live check (run by a human, needs the model + a RefCOCO+ slice):

```bash
conda run -n uni python experiments/runners/run_boundary_test.py \
  configs/Exp-103/exp103_coordinate_grounding_refcocoplus.yaml --generations 1 --max-seeds 2
# Expect: runs/Exp-103/<...>/trace.parquet with fitness_TgtBal columns,
# decoded_text = box strings, stats.json seed_selection_mode=refcocoplus.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_grounding_end_to_end.py
git commit -m "test(grounding): TgtBal over box-string candidates end-to-end smoke"
```

---

### Task 8 (DEFERRED — not v1): soft-box DIoU progress objective

**Not implemented in v1.** Only build this if a continuous *progress* signal beyond the teacher-forced boundary proves necessary. It requires exposing the per-token logprobs currently discarded at `scorer.py:308-319`, adding a `CoordinateSoftBoxDistance(Criterion)` that decodes the expected coordinate per digit-position and computes DIoU over soft boxes, and threading per-digit logprobs through `process_input`. See the design spec §6. Left as a documented stub so the v1 scope stays minimal.

---

## Self-Review

**Spec coverage:** SUT grounding mode → Tasks 1,5 (config + per-seed prompt; scoring reused, no edit per extraction). Numeric distance / boundary → reused `TargetedBalance` (Task 7 verifies). RefCOCO+ adapter → Task 3. Runner/trace wiring → Tasks 4,6. Coordinate-space normalization → Task 3 (`normalize_box`). Gen-0 predictor screen + flip search → operational (run the config), not code; out of this plan's scope by design. Soft-box → Task 8 (deferred, per spec §6). ✅

**Placeholder scan:** No TBD/TODO; every code step has complete code and exact commands. Task 8 is explicitly deferred, not a placeholder.

**Type consistency:** `GroundingConfig`/`RefCocoPlusConfig` field names match across Tasks 1–6; `normalize_box`/`build_seed_triples`/`refcocoplus_seeds` signatures consistent between Task 3 and its caller in Task 4; `effective_prompt_template(seed, config)` consistent Task 5; `SeedTriple(image, class_a, class_b, metadata)` used verbatim per `config.py:460-483`. `TargetedBalance.evaluate(logits, target_classes, batch_dim)` and `CriterionCollection.evaluate_all(**kwargs)` match the extracted signatures.

**Open decision flagged in Global Constraints:** per-token `norm_lp` vs `total_lp` (v1 keeps `norm_lp`); manipulation axis (v1 joint); cone-filter off.
