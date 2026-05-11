"""Profile library + operator-spec resolution + builder.

The composite manipulator (:mod:`.composite`) consumes resolved
:class:`OperatorSpec` lists. This module loads them from the YAML
library, applies overrides, and instantiates the concrete operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorSpec:
    """One row of a profile: operator name, severity, and extra kwargs."""

    name: str
    severity: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextProfile:
    """A named bundle of :class:`OperatorSpec` entries."""

    name: str
    operators: tuple[OperatorSpec, ...]


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


def load_profile_library(path: Path | str) -> dict[str, TextProfile]:
    """Load a profile-library YAML into ``{name: TextProfile}``.

    File format::

        profiles:
          casual_typing:
            operators:
              - {name: saliency, severity: 0.25}
              - {name: synonym, severity: 0.10}
          ...
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Profile library not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    library: dict[str, TextProfile] = {}
    for prof_name, prof_data in (raw.get("profiles") or {}).items():
        ops_data = prof_data.get("operators", [])
        ops = tuple(_spec_from_dict(item) for item in ops_data)
        library[prof_name] = TextProfile(name=prof_name, operators=ops)
    return library


def _spec_from_dict(item: dict[str, Any]) -> OperatorSpec:
    if "name" not in item or "severity" not in item:
        raise ValueError(
            f"operator entry missing 'name' or 'severity': {item!r}"
        )
    extras = {k: v for k, v in item.items() if k not in ("name", "severity")}
    return OperatorSpec(
        name=str(item["name"]),
        severity=float(item["severity"]),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------


def resolve_profile(
    library: dict[str, TextProfile],
    profile_name: str | None,
    operators: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[OperatorSpec, ...]:
    """Resolve a config block into the final list of operator specs.

    Priority: explicit *operators* > *profile_name + overrides* > *profile_name*.

    :raises KeyError: if *profile_name* is not in *library*.
    :raises ValueError: if neither *profile_name* nor *operators* is provided.
    """
    if operators:
        return tuple(_spec_from_dict(item) for item in operators)
    if profile_name is None:
        raise ValueError(
            "TextCompositeConfig requires either a profile name or an "
            "explicit operators list"
        )
    if profile_name not in library:
        raise KeyError(
            f"Profile {profile_name!r} not in library; available: "
            f"{sorted(library)}"
        )
    base = library[profile_name].operators
    if not overrides:
        return base
    return tuple(
        OperatorSpec(
            name=op.name,
            severity=float(overrides.get(op.name, {}).get("severity", op.severity)),
            extras={
                **op.extras,
                **{
                    k: v
                    for k, v in overrides.get(op.name, {}).items()
                    if k != "severity"
                },
            },
        )
        for op in base
    )


# ---------------------------------------------------------------------------
# Builder (lazy operator imports)
# ---------------------------------------------------------------------------


def build_operators_from_specs(
    specs: tuple[OperatorSpec, ...],
    content_pos: frozenset[str] | None = None,
    device: str | None = None,
    redis_url: str | None = None,
) -> list[Any]:
    """Instantiate the concrete operators for a given spec list.

    Operators with ``severity == 0`` are skipped.

    Universal optional ``extras["k_max"]`` overrides the
    severity → K_max formula (useful for Synonym pool depth).

    :param device: Default device for model-bearing operators (currently
        Synonym). Per-operator ``extras["device"]`` overrides this.
    :param redis_url: Optional Redis URL for the Synonym MLM candidate
        cache. ``None`` disables caching.

    :raises ValueError: on unknown operator names.
    """
    from .operators.character_noise import CharacterNoiseOperator
    from .operators.fragmentation import FragmentationOperator
    from .operators.saliency import SaliencyOperator
    from .operators.synonym import SynonymOperator
    from .types import CONTENT_POS_TAGS

    cp = content_pos or CONTENT_POS_TAGS
    out: list[Any] = []
    for spec in specs:
        if spec.severity == 0.0:
            continue
        k_max_override = spec.extras.get("k_max")
        if k_max_override is not None:
            k_max_override = int(k_max_override)
        if spec.name == "synonym":
            op_device = spec.extras.get("device", device if device is not None else "cpu")
            out.append(
                SynonymOperator(
                    severity=spec.severity,
                    model_name=spec.extras.get(
                        "model_name", "answerdotai/ModernBERT-large",
                    ),
                    topk_pre_filter=int(spec.extras.get("topk_pre_filter", 50)),
                    device=op_device,
                    spacy_model=spec.extras.get("spacy_model", "en_core_web_sm"),
                    content_pos=cp,
                    k_max_override=k_max_override,
                    redis_url=redis_url,
                )
            )
        elif spec.name == "fragmentation":
            out.append(
                FragmentationOperator(
                    severity=spec.severity,
                    k_max_override=k_max_override,
                )
            )
        elif spec.name == "character_noise":
            modes_raw = spec.extras.get("modes", ("homoglyph",))
            modes = tuple(modes_raw) if not isinstance(modes_raw, str) else (modes_raw,)
            out.append(
                CharacterNoiseOperator(
                    severity=spec.severity,
                    modes=modes,
                    k_max_override=k_max_override,
                )
            )
        elif spec.name == "saliency":
            out.append(
                SaliencyOperator(
                    severity=spec.severity,
                    k_max_override=k_max_override,
                )
            )
        else:
            raise ValueError(
                f"unknown text operator {spec.name!r}; supported: "
                f"synonym, fragmentation, character_noise, saliency"
            )
    return out
