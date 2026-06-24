"""Tests for make_study_config.py — HS-01 study-config generator.

Run via: conda run -n uni python -m pytest experiments/HS-01/tests/test_make_study_config.py -v
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent  # experiments/HS-01/tests -> repo root
HS01 = HERE.parent  # experiments/HS-01/

sys.path.insert(0, str(HS01))

import make_study_config  # noqa: E402 — must be after sys.path insert

POOL_FILE = HS01 / "pool_frozen" / "itempool.json"
SCHEMA_FILE = HS01 / "schemas" / "hs01.study-config.schema.json"
CONFIG_OUT = HS01 / "app" / "config" / "study-config.json"
CONSENT_OUT = HS01 / "app" / "config" / "consent.en.md"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated():
    """Run the generator once and return the parsed config + pool."""
    make_study_config.main()
    config = json.loads(CONFIG_OUT.read_text())
    pool = json.loads(POOL_FILE.read_text())
    return config, pool


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_schema_valid(generated):
    config, _ = generated
    try:
        import jsonschema
        from jsonschema import validate
        schema = json.loads(SCHEMA_FILE.read_text())
        validator_cls = jsonschema.Draft202012Validator
        validator_cls.check_schema(schema)
        errors = list(validator_cls(schema).iter_errors(config))
        assert errors == [], "\n".join(str(e) for e in errors)
    except ImportError:
        # Structural fallback — cover top-level required keys + key enums
        required_keys = [
            "schema_version", "study_id", "config_version", "pool_ref", "locale",
            "phases", "scales", "pair_response", "forms", "randomization",
            "attention_policy", "demographics_fields", "consent",
        ]
        for k in required_keys:
            assert k in config, f"missing top-level key: {k}"
        assert config["schema_version"] == "1.0.0"
        assert config["locale"] == "en"
        assert isinstance(config["forms"], list) and len(config["forms"]) >= 2
        assert config["randomization"]["within_phase_shuffle"] is True
        assert config["randomization"]["log_seed"] is True
        assert config["pair_response"]["ab_order"] == "randomized_per_trial"


# ---------------------------------------------------------------------------
# Form structure
# ---------------------------------------------------------------------------

def test_three_forms(generated):
    config, _ = generated
    assert len(config["forms"]) == 3
    form_ids = [f["form_id"] for f in config["forms"]]
    assert form_ids == ["A", "B", "C"]


def test_attention_items_in_all_forms(generated):
    config, pool = generated
    attn_ids = {it["item_id"] for it in pool["items"] if it.get("is_attention_check")}
    assert attn_ids == {"txt-attn-nonsense-01", "pair-attn-obvious-01"}
    for form in config["forms"]:
        all_ids = set(form["text_items"]) | set(form["image_items"]) | set(form["pair_items"])
        for aid in attn_ids:
            assert aid in all_ids, f"attention item {aid} missing from form {form['form_id']}"


def test_regular_items_in_exactly_one_form(generated):
    config, pool = generated
    attn_ids = {it["item_id"] for it in pool["items"] if it.get("is_attention_check")}
    regular_ids = {it["item_id"] for it in pool["items"] if not it.get("is_attention_check")}

    cover: dict[str, int] = {}
    for form in config["forms"]:
        for iid in form["text_items"] + form["image_items"] + form["pair_items"]:
            cover[iid] = cover.get(iid, 0) + 1

    for iid in regular_ids:
        assert cover.get(iid, 0) == 1, f"regular item {iid} appears {cover.get(iid, 0)}x (expected 1)"

    for iid in attn_ids:
        assert cover.get(iid, 0) == 3, f"attention item {iid} appears {cover.get(iid, 0)}x (expected 3)"


# ---------------------------------------------------------------------------
# Source disjointness within each form
# ---------------------------------------------------------------------------

def test_source_disjoint_within_form(generated):
    config, pool = generated
    items_by_id = {it["item_id"]: it for it in pool["items"]}

    for form in config["forms"]:
        all_items_in_form = form["text_items"] + form["image_items"] + form["pair_items"]
        sources = [items_by_id[iid]["source_id"] for iid in all_items_in_form]
        assert len(sources) == len(set(sources)), (
            f"form {form['form_id']}: duplicate source_id across phases"
        )


# ---------------------------------------------------------------------------
# Union coverage per phase
# ---------------------------------------------------------------------------

def test_phase_union_coverage(generated):
    config, pool = generated
    pool_text = {it["item_id"] for it in pool["items"] if it["kind"] == "text"}
    pool_image = {it["item_id"] for it in pool["items"] if it["kind"] == "image"}
    pool_pair = {it["item_id"] for it in pool["items"] if it["kind"] == "pair"}

    union_text: set[str] = set()
    union_image: set[str] = set()
    union_pair: set[str] = set()
    for form in config["forms"]:
        union_text |= set(form["text_items"])
        union_image |= set(form["image_items"])
        union_pair |= set(form["pair_items"])

    assert union_text == pool_text, f"text union mismatch: missing {pool_text - union_text}, extra {union_text - pool_text}"
    assert union_image == pool_image, f"image union mismatch: missing {pool_image - union_image}, extra {union_image - pool_image}"
    assert union_pair == pool_pair, f"pair union mismatch: missing {pool_pair - union_pair}, extra {union_pair - pool_pair}"


# ---------------------------------------------------------------------------
# SHA256 cross-checks
# ---------------------------------------------------------------------------

def test_pool_ref_sha256(generated):
    config, _ = generated
    expected = sha256_file(POOL_FILE)
    actual = config["pool_ref"]["pool_file_sha256"]
    assert actual == expected, f"pool_file_sha256 mismatch: got {actual}, expected {expected}"


def test_consent_sha256(generated):
    config, _ = generated
    expected = sha256_file(CONSENT_OUT)
    actual = config["consent"]["text_sha256"]
    assert actual == expected, f"consent.text_sha256 mismatch: got {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic(tmp_path, monkeypatch):
    """Running generator twice produces byte-identical study-config.json."""
    import make_study_config as msc

    # Run once
    msc.main()
    first_bytes = CONFIG_OUT.read_bytes()

    # Run again
    msc.main()
    second_bytes = CONFIG_OUT.read_bytes()

    assert first_bytes == second_bytes, "Generator is not deterministic"
