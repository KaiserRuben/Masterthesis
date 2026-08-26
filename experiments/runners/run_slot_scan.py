#!/usr/bin/env python3
"""Exhaustive slot-scan runner — scoring only, no search (Exp-105).

The evolutionary / PDQ runners *search*.  This one enumerates: it takes a
carrier sentence with one or more slots, expands every slot filler (and
optionally every grid point of a prompt template), and scores the full
cartesian product with teacher-forced log-probs.  That is the readout the
sentence-slot pilot needs for its scan steps (0 / 2 / 3 / 5 / 6) — the
contrast field itself, not a search trajectory through it.

Config contract (``configs/Exp-105/*_scan_*.yaml``).  Everything outside
the ``scan:`` block is the canonical :class:`~src.config.ExperimentConfig`
(``sut``, ``pmi``, ``device``, ``name``, ``save_dir``); the ``scan:``
block is parsed here and deliberately *not* part of that dataclass — it
is runner-local, and ExperimentConfig is shared with two other pipelines.

``scan:`` keys
--------------

Inputs (pick one):
  ``image: null`` / omitted   the PMI null image (gray 448 by default)
  ``image: <path>``           one image
  ``images: [<path>, ...]``   several images

Prompts (pick one):
  ``prompt: <str>``                        one fixed prompt
  ``carrier_template`` + ``grid: {k: [v]}``  cartesian product, one prompt
                                             per grid point via ``str.format``

Candidates (pick one; mutually exclusive with the chain keys):
  ``candidates: [<sentence>, ...]``            one unnamed set (``default``)
  ``candidate_sets: {name: [<sentence>]}``     several named sets, one
                                               scoring call per set
  ``hex_grid: {template: "... {hex} ...", …}``  a generated hex-code set
                                               (token-count validated)

Chains (steps 5/6):
  ``chain_templates: {name: "... {A} ... {B} ..."}``
  ``slots: {A: [...], B: [...]}``
  ``per_slot_scoring: true``     → one teacher-forced pass per chain scores
                                   every slot under its realised prefix
  ``perturbations: {...}``       → accepted but NOT implemented yet

Readout:
  ``report: [raw, pmi_baseline]``   persist the measured length-normalised
      log-probs (``lp_norm``) *and* the null-image baseline per (prompt,
      candidate tuple) (``lp_norm_null``), so the PMI-corrected margin is
      reconstructible post-hoc without a second run.  The baseline is
      scored under the *scan* prompt — Δ∅ up to the carrier — which is why
      ``baseline_prompt`` is persisted next to it.  Note that under
      ``pmi.enabled`` the SUT corrects ``lp_norm`` against its own
      baseline at the *canonical* prompt (see ``PMIConfig``), so on a PMI
      arm the two columns are not two views of one subtraction; the scan
      configs run raw for exactly that reason.
  ``free_generation_probe: true``   one free ``generate()`` per image,
      transcript written to the run dir (refusal check).

Outputs land in ``<save_dir>/<name>_<timestamp>/``:
``scan.parquet`` (one row per input × candidate, or per chain × slot),
``stats.json``, ``config.yaml``, and ``free_generation.json`` when probed.

Usage::

    python experiments/runners/run_slot_scan.py \\
        configs/Exp-105/exp105_step0_dinner_scan_qwen.yaml --dry-run
    python experiments/runners/run_slot_scan.py \\
        configs/Exp-105/exp105_step0_dinner_scan_qwen.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

# Single source of truth for the dacite hooks (Path / enum coercion).
from experiments.runners.run_boundary_test import load_config
from src.common.hex_grid import (
    build_hex_grid,
    filter_equal_token_count,
    validate_token_counts,
)
from src.config import ExperimentConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Stand-in for "no image file" — the PMI null image.  Written to the
#: ``image_path`` column so null-image rows are greppable.
NULL_IMAGE_ID = "<pmi_null_image>"

_SCAN_KEYS = frozenset({
    "image", "images", "prompt", "carrier_template", "grid",
    "candidates", "candidate_sets", "hex_grid", "report",
    "free_generation_probe",
    "chain_templates", "slots", "per_slot_scoring", "perturbations",
})

_HEX_KEYS = frozenset({
    "name", "template", "hue_start", "hue_end", "steps", "s", "v",
    "require_equal_token_count",
})

_REPORT_KEYS = frozenset({"raw", "pmi_baseline"})

#: Column names the runner owns.  A ``grid:`` key colliding with one of
#: these would silently overwrite a measurement, so it is rejected.
_RESERVED_COLUMNS = frozenset({
    "input_index", "image_path", "image_is_null", "prompt", "pmi_enabled",
    "candidate_set", "candidate_index", "candidate", "candidate_label",
    "lp_norm", "lp_norm_null", "baseline_prompt",
    "template_name", "template", "chain_text", "slot_name", "slot_filler",
    "fillers_json", "total_lp", "n_tokens",
    "total_lp_null", "n_tokens_null",
})

_CHAIN_API = "score_chain_slots"


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanInput:
    """One (image, prompt) pair — the unit every candidate set is scored on."""

    index: int
    image_path: str | None          # None → PMI null image
    prompt: str
    grid_params: dict[str, Any] = field(default_factory=dict)

    @property
    def image_id(self) -> str:
        """Stable identifier used for memoisation and the parquet column."""
        return self.image_path if self.image_path is not None else NULL_IMAGE_ID


@dataclass(frozen=True)
class CandidateSet:
    """One category tuple = one scoring call per input."""

    name: str
    candidates: tuple[str, ...]
    labels: tuple[str, ...]         # short label per candidate (hex code, …)
    source: str = "literal"         # "literal" | "hex_grid"


@dataclass(frozen=True)
class HexGridSpec:
    """Generated hex-code candidate set (token-count validated at run time)."""

    name: str
    template: str
    hue_start: float
    hue_end: float
    steps: int
    s: float
    v: float
    require_equal_token_count: bool


@dataclass(frozen=True)
class ChainSpec:
    """Multi-slot templates + per-slot filler sets (steps 5/6)."""

    templates: tuple[tuple[str, str], ...]        # (name, template)
    slots: tuple[tuple[str, tuple[str, ...]], ...]  # (slot, fillers)

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.slots)

    def filler_combinations(self) -> list[dict[str, str]]:
        """Every assignment of one filler per slot, in declaration order."""
        names = self.slot_names
        return [
            dict(zip(names, combo))
            for combo in product(*(fillers for _, fillers in self.slots))
        ]


@dataclass(frozen=True)
class ScanPlan:
    """Everything the runner will do, computed without touching a model."""

    name: str
    mode: str                                   # "score" | "chain"
    inputs: tuple[ScanInput, ...]
    candidate_sets: tuple[CandidateSet, ...]
    grid_keys: tuple[str, ...]
    report: tuple[str, ...]
    free_generation_probe: bool
    hex_grid: HexGridSpec | None = None
    chain: ChainSpec | None = None
    perturbations: dict[str, Any] | None = None

    # -- derived -----------------------------------------------------------

    @property
    def wants_baseline(self) -> bool:
        return "pmi_baseline" in self.report

    @property
    def image_ids(self) -> tuple[str, ...]:
        """Distinct images in first-appearance order."""
        seen: list[str] = []
        for inp in self.inputs:
            if inp.image_id not in seen:
                seen.append(inp.image_id)
        return tuple(seen)

    @property
    def n_chains(self) -> int:
        if self.chain is None:
            return 0
        return len(self.chain.templates) * len(self.chain.filler_combinations())

    def n_rows(self) -> int:
        """Parquet rows the run will produce."""
        if self.mode == "chain":
            assert self.chain is not None
            return len(self.inputs) * self.n_chains * len(self.chain.slots)
        per_input = sum(len(cs.candidates) for cs in self.candidate_sets)
        return len(self.inputs) * per_input

    def call_counts(self, pmi_enabled: bool = False) -> dict[str, int]:
        """Projected SUT calls, de-duplicated the way the run does it.

        Baseline calls that coincide with a main call (i.e. the scan already
        runs on the null image, as in step 0) are counted once — but not on
        a PMI arm, where the main call returns the corrected score while the
        baseline is forced raw, so the two are distinct forward passes.
        Chain scoring goes straight to the scorer and is therefore raw
        regardless of ``pmi.enabled``; there the collapse always applies.

        :param pmi_enabled: Value of ``config.pmi.enabled`` for the run.
        """
        main: set[tuple] = set()
        baseline: set[tuple] = set()
        if self.mode == "chain":
            assert self.chain is not None
            chains = [
                (tname, tuple(sorted(fillers.items())))
                for tname, _ in self.chain.templates
                for fillers in self.chain.filler_combinations()
            ]
            for inp in self.inputs:
                for chain in chains:
                    main.add((inp.image_id, inp.prompt, chain))
                    if self.wants_baseline:
                        baseline.add((NULL_IMAGE_ID, inp.prompt, chain))
        else:
            for inp in self.inputs:
                for cs in self.candidate_sets:
                    main.add((inp.image_id, inp.prompt, cs.name))
                    if self.wants_baseline:
                        baseline.add((NULL_IMAGE_ID, inp.prompt, cs.name))
        collapses = self.mode == "chain" or not pmi_enabled
        extra_baseline = baseline - main if collapses else baseline
        return {
            "main": len(main),
            "baseline": len(extra_baseline),
            "generate": len(self.image_ids) if self.free_generation_probe else 0,
            "total": len(main) + len(extra_baseline)
            + (len(self.image_ids) if self.free_generation_probe else 0),
        }


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _placeholders(template: str) -> tuple[str, ...]:
    """Field names in a ``str.format`` template, in order of appearance."""
    return tuple(
        fname for _, fname, _, _ in Formatter().parse(template)
        if fname
    )


def _as_str_tuple(values: Iterable[Any], where: str) -> tuple[str, ...]:
    out = tuple(str(v) for v in values)
    if not out:
        raise ValueError(f"scan.{where} is empty")
    return out


def _build_prompts(
    scan: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any]]], tuple[str, ...]]:
    """Expand ``prompt`` or ``carrier_template`` × ``grid`` into prompts.

    :returns: ``([(prompt, grid_params), ...], grid_keys)``.
    """
    has_carrier = bool(scan.get("carrier_template"))
    has_prompt = bool(scan.get("prompt"))
    if has_carrier and has_prompt:
        raise ValueError(
            "scan defines both `prompt` and `carrier_template` — pick one "
            "(carrier_template is the grid form, prompt is the fixed form)."
        )
    if not has_carrier and not has_prompt:
        raise ValueError(
            "scan needs either `prompt` (fixed) or `carrier_template` + "
            "`grid` (one prompt per grid point)."
        )

    if has_prompt:
        if scan.get("grid"):
            raise ValueError(
                "scan.grid is only meaningful with `carrier_template`; a "
                "fixed `prompt` has nothing to substitute into."
            )
        return [(str(scan["prompt"]), {})], ()

    carrier = str(scan["carrier_template"])
    grid = scan.get("grid") or {}
    if not grid:
        raise ValueError("scan.carrier_template requires a non-empty scan.grid")

    grid_keys = tuple(grid.keys())
    collisions = sorted(set(grid_keys) & _RESERVED_COLUMNS)
    if collisions:
        raise ValueError(
            f"scan.grid keys collide with runner-owned parquet columns: "
            f"{collisions}. Rename them."
        )
    fields = set(_placeholders(carrier))
    if fields != set(grid_keys):
        raise ValueError(
            f"scan.carrier_template placeholders {sorted(fields)} must match "
            f"scan.grid keys {sorted(grid_keys)} exactly."
        )

    axes = [list(grid[k]) for k in grid_keys]
    for k, vals in zip(grid_keys, axes):
        if not vals:
            raise ValueError(f"scan.grid['{k}'] is empty")

    prompts: list[tuple[str, dict[str, Any]]] = []
    for combo in product(*axes):
        params = dict(zip(grid_keys, combo))
        prompts.append((carrier.format(**params), params))
    return prompts, grid_keys


def _build_images(scan: dict[str, Any]) -> list[str | None]:
    """Resolve the image list.  ``[None]`` means "the PMI null image"."""
    if scan.get("images"):
        if scan.get("image") is not None:
            raise ValueError("scan defines both `image` and `images` — pick one.")
        return [str(p) for p in scan["images"]]
    if scan.get("image") is not None:
        return [str(scan["image"])]
    return [None]


def _build_hex_spec(raw: dict[str, Any]) -> HexGridSpec:
    unknown = sorted(set(raw) - _HEX_KEYS)
    if unknown:
        raise ValueError(f"unknown scan.hex_grid keys: {unknown}")
    template = str(raw.get("template", "{hex}"))
    if "hex" not in _placeholders(template):
        raise ValueError(
            "scan.hex_grid.template must contain a {hex} placeholder; got "
            f"{template!r}"
        )
    return HexGridSpec(
        name=str(raw.get("name", "hex_grid")),
        template=template,
        hue_start=float(raw.get("hue_start", 120.0)),
        hue_end=float(raw.get("hue_end", 240.0)),
        steps=int(raw.get("steps", 17)),
        s=float(raw.get("s", 1.0)),
        v=float(raw.get("v", 1.0)),
        require_equal_token_count=bool(
            raw.get("require_equal_token_count", True)
        ),
    )


def _hex_candidate_set(spec: HexGridSpec, codes: Sequence[str]) -> CandidateSet:
    return CandidateSet(
        name=spec.name,
        candidates=tuple(spec.template.format(hex=c) for c in codes),
        labels=tuple(codes),
        source="hex_grid",
    )


def _build_candidate_sets(
    scan: dict[str, Any],
    hex_spec: HexGridSpec | None,
) -> list[CandidateSet]:
    sets: list[CandidateSet] = []
    if scan.get("candidates"):
        if scan.get("candidate_sets"):
            raise ValueError(
                "scan defines both `candidates` and `candidate_sets` — pick "
                "one (`candidates` is the unnamed single-tuple form)."
            )
        cands = _as_str_tuple(scan["candidates"], "candidates")
        sets.append(CandidateSet("default", cands, cands))
    elif scan.get("candidate_sets"):
        for name, values in scan["candidate_sets"].items():
            cands = _as_str_tuple(values, f"candidate_sets['{name}']")
            sets.append(CandidateSet(str(name), cands, cands))
    if hex_spec is not None:
        # Provisional: the token-count filter needs the SUT tokenizer and
        # runs in materialise_candidate_sets() just before scoring.
        sets.append(
            _hex_candidate_set(
                hex_spec,
                build_hex_grid(
                    hue_start=hex_spec.hue_start,
                    hue_end=hex_spec.hue_end,
                    steps=hex_spec.steps,
                    s=hex_spec.s,
                    v=hex_spec.v,
                ),
            )
        )
    if not sets:
        raise ValueError(
            "scan needs `candidates`, `candidate_sets`, or `hex_grid` "
            "(or the chain keys for a step-5/6 scan)."
        )
    return sets


def _build_chain(scan: dict[str, Any]) -> ChainSpec:
    if not scan.get("slots"):
        raise ValueError("scan.chain_templates requires a scan.slots block.")
    if not scan.get("per_slot_scoring"):
        raise ValueError(
            "scan.chain_templates requires `per_slot_scoring: true` — a chain "
            "scan without per-slot conditionals has no readout."
        )
    for key in ("candidates", "candidate_sets", "hex_grid"):
        if scan.get(key):
            raise ValueError(
                f"scan.{key} cannot be combined with chain_templates; chain "
                "candidates are generated from the templates × slots."
            )
    slots = tuple(
        (str(name), _as_str_tuple(vals, f"slots['{name}']"))
        for name, vals in scan["slots"].items()
    )
    slot_names = {name for name, _ in slots}
    templates: list[tuple[str, str]] = []
    for tname, template in scan["chain_templates"].items():
        fields = set(_placeholders(str(template)))
        if fields != slot_names:
            raise ValueError(
                f"chain template '{tname}' has placeholders {sorted(fields)} "
                f"but scan.slots declares {sorted(slot_names)} — every "
                "template must realise every slot (the orders differ, the "
                "slot set does not)."
            )
        templates.append((str(tname), str(template)))
    return ChainSpec(templates=tuple(templates), slots=slots)


def build_plan(scan: dict[str, Any], name: str = "slot_scan") -> ScanPlan:
    """Turn a raw ``scan:`` block into a fully enumerated :class:`ScanPlan`.

    Pure: touches no model, no tokenizer, and no filesystem.

    :param scan: The raw ``scan:`` mapping from the YAML config.
    :param name: Experiment name (used for logging / the run dir).
    :returns: The plan.
    :raises ValueError: On any malformed / contradictory scan block.
    """
    if not scan:
        raise ValueError("config has no `scan:` block — nothing to scan.")
    unknown = sorted(set(scan) - _SCAN_KEYS)
    if unknown:
        raise ValueError(
            f"unknown scan keys: {unknown}. Known keys: {sorted(_SCAN_KEYS)}"
        )

    report = tuple(str(r) for r in (scan.get("report") or ("raw",)))
    bad_report = sorted(set(report) - _REPORT_KEYS)
    if bad_report:
        raise ValueError(
            f"unknown scan.report entries: {bad_report}; known: "
            f"{sorted(_REPORT_KEYS)}"
        )
    if "raw" not in report:
        raise ValueError(
            "scan.report must include `raw` — the runner always persists the "
            "measured log-probs; `pmi_baseline` only adds the null-image "
            "baseline alongside them."
        )

    prompts, grid_keys = _build_prompts(scan)
    images = _build_images(scan)
    inputs = tuple(
        ScanInput(index=i, image_path=img, prompt=prompt, grid_params=params)
        for i, (img, (prompt, params)) in enumerate(
            (img, pp) for img in images for pp in prompts
        )
    )

    is_chain = bool(scan.get("chain_templates"))
    if is_chain:
        chain = _build_chain(scan)
        return ScanPlan(
            name=name,
            mode="chain",
            inputs=inputs,
            candidate_sets=(),
            grid_keys=grid_keys,
            report=report,
            free_generation_probe=bool(scan.get("free_generation_probe")),
            chain=chain,
            perturbations=scan.get("perturbations") or None,
        )

    for key in ("slots", "per_slot_scoring", "perturbations"):
        if scan.get(key):
            raise ValueError(
                f"scan.{key} only applies to a chain scan; add "
                "`chain_templates` or drop the key."
            )

    hex_spec = (
        _build_hex_spec(scan["hex_grid"]) if scan.get("hex_grid") else None
    )
    return ScanPlan(
        name=name,
        mode="score",
        inputs=inputs,
        candidate_sets=tuple(_build_candidate_sets(scan, hex_spec)),
        grid_keys=grid_keys,
        report=report,
        free_generation_probe=bool(scan.get("free_generation_probe")),
        hex_grid=hex_spec,
    )


# ---------------------------------------------------------------------------
# Plan reporting (--dry-run)
# ---------------------------------------------------------------------------


def _truncate(text: str, width: int = 62) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def format_plan(plan: ScanPlan, exp: ExperimentConfig) -> str:
    """Render the human-readable plan summary printed by ``--dry-run``."""
    calls = plan.call_counts(exp.pmi.enabled)
    n_missing = sum(
        1 for p in plan.image_ids
        if p != NULL_IMAGE_ID and not _resolve_image_path(p).exists()
    )
    lines = [
        "=" * 72,
        f"Exp-105 slot scan plan — {plan.name}   [mode={plan.mode}]",
        "=" * 72,
        f"  model            : {exp.sut.model_id} "
        f"(backend={exp.sut.backend}, device={exp.device})",
        f"  pmi.enabled      : {exp.pmi.enabled}"
        + ("  → `lp_norm` holds PMI-CORRECTED scores" if exp.pmi.enabled
           else "  → `lp_norm` holds raw length-normalised log-probs"),
        f"  null image       : {exp.pmi.null_image} @ {exp.pmi.null_image_size}px",
        f"  report           : {', '.join(plan.report)}",
        f"  images           : {len(plan.image_ids)}"
        + (f"  ({n_missing} MISSING on disk)" if n_missing else ""),
    ]
    for img in plan.image_ids:
        mark = ""
        if img != NULL_IMAGE_ID and not _resolve_image_path(img).exists():
            mark = "   [MISSING]"
        lines.append(f"      - {_truncate(img)}{mark}")
    n_prompts = len({inp.prompt for inp in plan.inputs})
    lines.append(f"  prompts          : {n_prompts}")
    if plan.grid_keys:
        lines.append(
            "  grid             : "
            + " × ".join(
                f"{k}({len({inp.grid_params[k] for inp in plan.inputs})})"
                for k in plan.grid_keys
            )
        )
    lines.append(f"  inputs           : {len(plan.inputs)}  (image × prompt)")

    if plan.mode == "chain":
        assert plan.chain is not None
        lines.append(
            f"  chain templates  : {len(plan.chain.templates)}  "
            + ", ".join(n for n, _ in plan.chain.templates)
        )
        for sname, fillers in plan.chain.slots:
            lines.append(
                f"      slot {sname}: {len(fillers)}  "
                f"[{', '.join(_truncate(f, 24) for f in fillers)}]"
            )
        lines.append(
            f"  chains per input : {plan.n_chains}  "
            f"(templates × filler combinations)"
        )
        lines.append(
            f"  chain API        : VLMScorer.{_CHAIN_API} "
            + ("available" if _chain_api_available() else "MISSING (run aborts)")
        )
    else:
        lines.append(f"  candidate sets   : {len(plan.candidate_sets)}")
        for cs in plan.candidate_sets:
            lines.append(
                f"      - {cs.name} ({len(cs.candidates)}"
                + (f", {cs.source}" if cs.source != "literal" else "")
                + "):"
            )
            for cand in cs.candidates[:4]:
                lines.append(f"          · {_truncate(cand)}")
            if len(cs.candidates) > 4:
                lines.append(f"          · … {len(cs.candidates) - 4} more")
        if plan.hex_grid is not None:
            lines.append(
                "  hex token check  : "
                + ("MANDATORY filter on (off-modal codes dropped at run time)"
                   if plan.hex_grid.require_equal_token_count
                   else "REPORT ONLY — codes with unequal token counts are "
                        "kept; the prefix-cancellation argument does not hold")
            )

    lines += [
        f"  SUT calls        : {calls['main']} main "
        f"+ {calls['baseline']} null-baseline "
        f"+ {calls['generate']} generate = {calls['total']}",
        f"  parquet rows     : {plan.n_rows()}",
        f"  output dir       : {exp.save_dir}/{plan.name}_<timestamp>/",
    ]
    if plan.perturbations:
        lines.append(
            "  perturbations    : DECLARED BUT NOT IMPLEMENTED — a real run "
            "aborts (see NotImplementedError in run_scan)"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _resolve_image_path(raw: str) -> Path:
    """Resolve a config image path against CWD, then the repo root."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    cwd_rel = Path.cwd() / p
    if cwd_rel.exists():
        return cwd_rel
    return REPO_ROOT / p


def _chain_api_available() -> bool:
    """Whether :class:`VLMScorer` exposes the pinned chain-scoring method."""
    from src.sut.scorer import VLMScorer

    return hasattr(VLMScorer, _CHAIN_API)


def _require_chain_api() -> None:
    if not _chain_api_available():
        raise SystemExit(
            f"This config needs per-slot chain scoring, but "
            f"src.sut.scorer.VLMScorer has no `{_CHAIN_API}` method.\n"
            f"Pinned signature:\n"
            f"    {_CHAIN_API}(image, prompt, "
            f"parts: list[str | tuple[str, str]])\n"
            f"        -> dict[str, tuple[float, float, int]]   "
            f"# {{slot: (total_lp, lp_norm, n_tokens)}}\n"
            f"It is being added under a separate task — re-run once it lands."
        )


def template_parts(
    template: str,
    fillers: dict[str, str],
) -> list[str | tuple[str, str]]:
    """Split a chain template into literal / (slot, filler) segments.

    ``"Because the applicant was {G}, the application was {D}."`` with
    ``{"G": "a man", "D": "accepted"}`` becomes::

        ["Because the applicant was ", ("G", "a man"),
         ", the application was ", ("D", "accepted"), "."]

    Empty literals are dropped so the scorer sees no zero-length segments.
    """
    parts: list[str | tuple[str, str]] = []
    for literal, fname, _, _ in Formatter().parse(template):
        if literal:
            parts.append(literal)
        if fname:
            if fname not in fillers:
                raise KeyError(
                    f"chain template references slot {fname!r} with no filler"
                )
            parts.append((fname, fillers[fname]))
    return parts


def chain_text(parts: Sequence[str | tuple[str, str]]) -> str:
    """Reassemble the realised sentence from :func:`template_parts` output."""
    return "".join(p if isinstance(p, str) else p[1] for p in parts)


def materialise_candidate_sets(
    plan: ScanPlan,
    tokenizer: Any,
) -> tuple[CandidateSet, ...]:
    """Apply the mandatory hex token-count check, returning the final sets.

    Literal sets pass through untouched.  A hex set is re-derived from its
    spec and, when ``require_equal_token_count`` (the default), reduced to
    the modal-token-count codes; the dropped complement is logged.

    :param plan: The plan whose ``hex_grid`` spec (if any) is applied.
    :param tokenizer: SUT tokenizer (``encode(text, add_special_tokens=)``).
        May be ``None`` when the plan declares no hex grid.
    :returns: Candidate sets in plan order.
    """
    if plan.hex_grid is None:
        return plan.candidate_sets
    if tokenizer is None:
        raise ValueError(
            "a hex-grid scan needs the SUT tokenizer for the mandatory "
            "token-count check"
        )

    spec = plan.hex_grid
    codes = build_hex_grid(
        hue_start=spec.hue_start, hue_end=spec.hue_end,
        steps=spec.steps, s=spec.s, v=spec.v,
    )
    counts = validate_token_counts(codes, tokenizer)
    distinct = sorted(set(counts.values()))
    logger.info(
        "Hex grid '%s': %d codes, token counts %s",
        spec.name, len(codes), distinct,
    )
    if len(distinct) == 1:
        kept = codes
    elif spec.require_equal_token_count:
        kept, modal = filter_equal_token_count(codes, tokenizer)
        dropped = [c for c in codes if c not in set(kept)]
        logger.warning(
            "Hex grid '%s': keeping %d/%d codes at the modal token count "
            "(%d tokens); dropped %s — constant string length does not imply "
            "constant BPE length, and the carrier prefix only cancels at "
            "equal token count.",
            spec.name, len(kept), len(codes), modal,
            {c: counts[c] for c in dropped},
        )
    else:
        kept = codes
        logger.warning(
            "Hex grid '%s': token counts differ %s but "
            "require_equal_token_count is off — the length-normalised scores "
            "are NOT comparable across codes.",
            spec.name, {c: counts[c] for c in codes if counts[c] != distinct[0]},
        )

    final = _hex_candidate_set(spec, kept)
    return tuple(
        final if cs.name == spec.name and cs.source == "hex_grid" else cs
        for cs in plan.candidate_sets
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class _ScanScorer:
    """Memoising wrapper around the SUT for one scan run.

    The SUT's Redis cache already de-duplicates across runs; this in-process
    memo additionally collapses the null-image baseline, which is shared by
    every input that uses the same prompt (and, under a raw arm, coincides
    with the main call when the scan itself runs on the null image).

    The raw/corrected distinction only exists when ``pmi.enabled``: with PMI
    off, ``force_raw()`` is a no-op and both readings are the same forward
    pass, so the memo key ignores the flag.
    """

    def __init__(self, sut, null_image, pmi_enabled: bool) -> None:
        self._sut = sut
        self._null = null_image
        self._pmi_enabled = pmi_enabled
        self._memo: dict[tuple, list[float]] = {}
        self.n_calls = 0

    def scores(
        self,
        image,
        image_id: str,
        prompt: str,
        candidates: tuple[str, ...],
        *,
        raw: bool,
    ) -> list[float]:
        """Length-normalised log-probs for *candidates*, in input order."""
        force_raw = raw and self._pmi_enabled
        key = (image_id, prompt, candidates, force_raw)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.n_calls += 1
        if force_raw:
            # With PMI on, the recorded baseline must stay the *raw*
            # surface-form prior (Δ∅) — that is what makes the corrected
            # margin reconstructible from the persisted columns.
            with self._sut.force_raw():
                out = self._sut.process_input(
                    image, text=prompt, categories=candidates,
                )
        else:
            out = self._sut.process_input(
                image, text=prompt, categories=candidates,
            )
        vals = [float(v) for v in out.tolist()]
        self._memo[key] = vals
        return vals

    def baseline(
        self, prompt: str, candidates: tuple[str, ...],
    ) -> list[float]:
        """Raw null-image scores for the same prompt + candidate tuple."""
        return self.scores(
            self._null, NULL_IMAGE_ID, prompt, candidates, raw=True,
        )


def _load_images(plan: ScanPlan, null_image) -> dict[str, Any]:
    """Open every distinct image once.  ``NULL_IMAGE_ID`` maps to *null_image*."""
    from PIL import Image

    out: dict[str, Any] = {NULL_IMAGE_ID: null_image}
    for image_id in plan.image_ids:
        if image_id == NULL_IMAGE_ID:
            continue
        path = _resolve_image_path(image_id)
        if not path.exists():
            raise FileNotFoundError(
                f"scan image not found: {image_id} (resolved to {path})"
            )
        out[image_id] = Image.open(path).convert("RGB")
    return out


def _base_row(plan: ScanPlan, inp: ScanInput, pmi_enabled: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "input_index": inp.index,
        "image_path": inp.image_id,
        "image_is_null": inp.image_path is None,
        "prompt": inp.prompt,
        "pmi_enabled": pmi_enabled,
    }
    for key in plan.grid_keys:
        row[key] = inp.grid_params[key]
    return row


def _run_score_mode(
    plan: ScanPlan,
    exp: ExperimentConfig,
    scorer: _ScanScorer,
    images: dict[str, Any],
    candidate_sets: Sequence[CandidateSet],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(plan.inputs) * len(candidate_sets)
    step = max(1, total // 20)
    done = 0
    for inp in plan.inputs:
        image = images[inp.image_id]
        for cs in candidate_sets:
            lp = scorer.scores(
                image, inp.image_id, inp.prompt, cs.candidates, raw=False,
            )
            base = (
                scorer.baseline(inp.prompt, cs.candidates)
                if plan.wants_baseline else [None] * len(cs.candidates)
            )
            for i, cand in enumerate(cs.candidates):
                row = _base_row(plan, inp, exp.pmi.enabled)
                row.update({
                    "candidate_set": cs.name,
                    "candidate_index": i,
                    "candidate": cand,
                    "candidate_label": cs.labels[i],
                    "lp_norm": lp[i],
                    "lp_norm_null": base[i],
                    "baseline_prompt": (
                        inp.prompt if plan.wants_baseline else None
                    ),
                })
                rows.append(row)
            done += 1
            if done % step == 0 or done == total:
                logger.info(
                    "  scored %d/%d input×set (%d SUT calls so far)",
                    done, total, scorer.n_calls,
                )
    return rows


def _run_chain_mode(
    plan: ScanPlan,
    exp: ExperimentConfig,
    sut,
    images: dict[str, Any],
    null_image,
) -> tuple[list[dict[str, Any]], int]:
    """Per-slot conditional scoring for every template × filler combination.

    Chain scoring goes straight to ``sut.scorer`` (the pinned
    ``score_chain_slots`` API lives there, not on the SUT wrapper), so these
    calls bypass the Redis inference cache; the in-process memo below still
    collapses the shared null-image baselines.
    """
    assert plan.chain is not None
    chain_api = getattr(sut.scorer, _CHAIN_API, None)
    if chain_api is None:
        _require_chain_api()  # raises with the full signature
    combos = plan.chain.filler_combinations()
    memo: dict[tuple, dict[str, tuple[float, float, int]]] = {}
    n_calls = 0

    def _score(image, image_id: str, prompt: str, parts) -> dict:
        nonlocal n_calls
        key = (image_id, prompt, tuple(parts))
        hit = memo.get(key)
        if hit is not None:
            return hit
        n_calls += 1
        out = chain_api(image, prompt, list(parts))
        memo[key] = out
        return out

    rows: list[dict[str, Any]] = []
    total = len(plan.inputs) * plan.n_chains
    step = max(1, total // 20)
    done = 0
    for inp in plan.inputs:
        image = images[inp.image_id]
        for tname, template in plan.chain.templates:
            for fillers in combos:
                parts = template_parts(template, fillers)
                text = chain_text(parts)
                scored = _score(image, inp.image_id, inp.prompt, parts)
                base = (
                    _score(null_image, NULL_IMAGE_ID, inp.prompt, parts)
                    if plan.wants_baseline else {}
                )
                for slot_name, _ in plan.chain.slots:
                    if slot_name not in scored:
                        raise KeyError(
                            f"{_CHAIN_API} returned slots {sorted(scored)} "
                            f"but the template declares {slot_name!r}"
                        )
                    total_lp, lp_norm, n_tok = scored[slot_name]
                    row = _base_row(plan, inp, exp.pmi.enabled)
                    row.update({
                        "template_name": tname,
                        "template": template,
                        "chain_text": text,
                        "slot_name": slot_name,
                        "slot_filler": fillers[slot_name],
                        "fillers_json": json.dumps(fillers, sort_keys=True),
                        "total_lp": float(total_lp),
                        "lp_norm": float(lp_norm),
                        "n_tokens": int(n_tok),
                        "baseline_prompt": (
                            inp.prompt if plan.wants_baseline else None
                        ),
                    })
                    if base:
                        b_total, b_norm, b_tok = base[slot_name]
                        row.update({
                            "total_lp_null": float(b_total),
                            "lp_norm_null": float(b_norm),
                            "n_tokens_null": int(b_tok),
                        })
                    else:
                        row.update({
                            "total_lp_null": None,
                            "lp_norm_null": None,
                            "n_tokens_null": None,
                        })
                    rows.append(row)
                done += 1
                if done % step == 0 or done == total:
                    logger.info(
                        "  scored %d/%d chains (%d scorer calls so far)",
                        done, total, n_calls,
                    )
    return rows, n_calls


def _run_free_generation(
    plan: ScanPlan,
    sut,
    images: dict[str, Any],
    run_dir: Path,
) -> int:
    """One free ``generate()`` per distinct image — the refusal check."""
    prompt_for: dict[str, str] = {}
    for inp in plan.inputs:
        prompt_for.setdefault(inp.image_id, inp.prompt)

    transcripts = []
    for image_id in plan.image_ids:
        prompt = prompt_for[image_id]
        answer, thinking, _ = sut.scorer.generate(images[image_id], prompt)
        logger.info(
            "  free generation [%s] → %s", image_id, _truncate(answer, 100),
        )
        transcripts.append({
            "image_path": image_id,
            "prompt": prompt,
            "answer": answer,
            "thinking": thinking,
        })
    with open(run_dir / "free_generation.json", "w") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
    return len(transcripts)


def execute_scan(
    plan: ScanPlan,
    exp: ExperimentConfig,
    sut,
    run_dir: Path,
    raw_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run *plan* against a live *sut* and write all artifacts to *run_dir*.

    :param plan: Fully enumerated scan plan.
    :param exp: Canonical experiment config (SUT / PMI provenance).
    :param sut: A :class:`~src.sut.vlm_sut.VLMSUT` (or an equivalent double).
    :param run_dir: Output directory; created if missing.
    :param raw_cfg: Raw post-override YAML dict, copied to ``config.yaml``.
    :returns: The stats dict also written to ``stats.json``.
    """
    import pandas as pd

    start = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    if raw_cfg is not None:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(raw_cfg, f, sort_keys=False, allow_unicode=True)

    # The baseline image is taken from the SUT itself so the recorded Δ∅ is
    # byte-identical to the one `pmi.enabled` would subtract internally.
    null_image = sut._null_image()
    images = _load_images(plan, null_image)

    if exp.pmi.enabled:
        logger.warning(
            "pmi.enabled → the `lp_norm` column holds PMI-CORRECTED scores "
            "(baseline at the canonical prompt, see PMIConfig), while "
            "`lp_norm_null` is the raw null-image score at the scan prompt. "
            "The two are NOT two views of one subtraction on this arm."
        )

    candidate_sets: tuple[CandidateSet, ...] = ()
    n_scorer_calls = 0
    if plan.mode == "chain":
        rows, n_scorer_calls = _run_chain_mode(
            plan, exp, sut, images, null_image,
        )
        n_sut_calls = 0
    else:
        # Tokenizer only needed for the hex token-count check — resolved
        # lazily so a scan with literal candidates never touches it.
        tokenizer = sut.scorer.tokenizer if plan.hex_grid is not None else None
        candidate_sets = materialise_candidate_sets(plan, tokenizer)
        scorer = _ScanScorer(sut, null_image, exp.pmi.enabled)
        rows = _run_score_mode(plan, exp, scorer, images, candidate_sets)
        n_sut_calls = scorer.n_calls

    df = pd.DataFrame(rows)
    parquet_path = run_dir / "scan.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Wrote %d rows → %s", len(df), parquet_path)

    n_generate = 0
    if plan.free_generation_probe:
        n_generate = _run_free_generation(plan, sut, images, run_dir)

    stats: dict[str, Any] = {
        "name": plan.name,
        "mode": plan.mode,
        "model_id": exp.sut.model_id,
        "backend": exp.sut.backend,
        "device": exp.device,
        "pmi_enabled": exp.pmi.enabled,
        "pmi_null_image": exp.pmi.null_image,
        "pmi_null_image_size": exp.pmi.null_image_size,
        "report": list(plan.report),
        "n_inputs": len(plan.inputs),
        "n_images": len(plan.image_ids),
        "n_prompts": len({inp.prompt for inp in plan.inputs}),
        "n_candidate_sets": len(candidate_sets),
        "candidate_sets": {
            cs.name: list(cs.candidates) for cs in candidate_sets
        },
        "n_chains": plan.n_chains,
        "n_rows": len(df),
        "n_sut_calls": n_sut_calls,
        "n_scorer_calls": n_scorer_calls,
        "n_generate_calls": n_generate,
        "cache_stats": dict(sut.cache_stats),
        "wall_time_sec": round(time.time() - start, 3),
    }
    with open(run_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        "Scan complete in %.1fs — %d rows, %d SUT calls, cache %s",
        stats["wall_time_sec"], stats["n_rows"],
        stats["n_sut_calls"], stats["cache_stats"],
    )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_scan(cfg: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any] | None:
    """Load config, build the plan, and (unless *dry_run*) execute it."""
    exp = load_config(cfg)
    plan = build_plan(cfg.get("scan") or {}, name=exp.name)

    if dry_run:
        print(format_plan(plan, exp))
        return None

    if plan.perturbations:
        raise NotImplementedError(
            "scan.perturbations is accepted by the schema but not implemented "
            f"yet (declared: {json.dumps(plan.perturbations, sort_keys=True)}). "
            "Perturbation sampling for the step-5/6 neighbourhood is a "
            "follow-up task; remove the block to run the un-perturbed chain "
            "scan in the meantime."
        )
    if plan.mode == "chain":
        _require_chain_api()

    logger.info("%s", format_plan(plan, exp))

    # Mirror of the SUT half of src.common.pipeline_bootstrap: this runner
    # never manipulates, so the text/image manipulators are not built.
    from src.sut import VLMSUT

    sut_device = (
        exp.sut.ov_device if exp.sut.backend == "openvino" else exp.device
    )
    logger.info("SUT starting...  %s on %s", exp.sut.model_id, sut_device)
    sut = VLMSUT(exp)
    logger.info("SUT loaded")

    run_dir = exp.save_dir / f"{plan.name}_{int(time.time())}"
    return execute_scan(plan, exp, sut, run_dir, raw_cfg=cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustive slot scan (no search): score every input × candidate "
            "tuple and persist raw log-probs plus the null-image baseline."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to a scan YAML config")
    parser.add_argument(
        "--device", help="Override device (e.g. cuda, mps, cpu)",
    )
    parser.add_argument(
        "--save-dir", type=str, help="Override output directory for results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Build the full input/candidate plan, print the summary, and exit "
            "without loading a model."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device
    if args.save_dir:
        cfg["save_dir"] = args.save_dir

    run_scan(cfg, dry_run=args.dry_run)

    if not args.dry_run:
        # HF streaming leaves daemon threads — force exit.
        os._exit(0)


if __name__ == "__main__":
    main()
