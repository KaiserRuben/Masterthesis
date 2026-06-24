#!/usr/bin/env python3
"""Generate the HS-01 study-config.json (forms + scales + policy) from the frozen pool.
Deterministic: same pool -> identical config. Run via `conda run -n uni python`."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = HERE / "pool_frozen" / "itempool.json"
OUT_DIR = HERE / "app" / "config"
CONSENT_MD = OUT_DIR / "consent.en.md"
CONFIG_JSON = OUT_DIR / "study-config.json"
N_FORMS = 3
FORM_IDS = ["A", "B", "C"]

CONSENT_TEXT = """\
# Study consent

Thank you for helping with this research.

In this short task you will look at some questions and images that an AI system
was shown, and tell us how clearly **you** can understand them. There are no
right or wrong answers — we are interested in your honest impression.

- It takes about **10 minutes**.
- Participation is **voluntary**; you can stop at any time by closing the tab.
- It is **anonymous**: we store only a random participant code and your answers.
  We do not collect your name, email, or any personal data.
- Your answers are used for academic research in a master's thesis.

Questions: <researcher email>

By choosing **"I consent and want to begin"** you confirm you are at least 18
years old and agree to take part on these terms.
"""

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def main() -> None:
    pool = json.loads(POOL.read_text())
    pool_sha = sha256_bytes(POOL.read_bytes())
    items = pool["items"]
    src_by_id = {s["source_id"]: s for s in pool["sources"]}

    def stratum(it):
        s = src_by_id[it["source_id"]]["strata"]
        return s[it["kind"]] or "_none"

    forms = {f: {"form_id": f, "text_items": [], "image_items": [], "pair_items": []} for f in FORM_IDS}
    key = {"text": "text_items", "image": "image_items", "pair": "pair_items"}
    attention = {it["item_id"]: it for it in items if it.get("is_attention_check")}

    for kind in ("text", "image", "pair"):
        regular = sorted(
            [it for it in items if it["kind"] == kind and not it.get("is_attention_check")],
            key=lambda it: (stratum(it), it["item_id"]),
        )
        # round-robin within stratum: re-sort groups so dealing is stratum-balanced
        for i, it in enumerate(regular):
            forms[FORM_IDS[i % N_FORMS]][key[kind]].append(it["item_id"])
        # attention items -> every form
        for it in items:
            if it["kind"] == kind and it.get("is_attention_check"):
                for f in FORM_IDS:
                    forms[f][key[kind]].append(it["item_id"])

    # invariants
    for f in FORM_IDS:
        srcs = [src_for(items, iid) for iid in forms[f]["text_items"] + forms[f]["image_items"] + forms[f]["pair_items"]]
        assert len(srcs) == len(set(srcs)), f"form {f} reuses a source across phases"
    cover = {}
    for f in FORM_IDS:
        for iid in forms[f]["text_items"] + forms[f]["image_items"] + forms[f]["pair_items"]:
            cover[iid] = cover.get(iid, 0) + 1
    for iid, n in cover.items():
        exp = N_FORMS if iid in attention else 1
        assert n == exp, f"item {iid} appears {n}x, expected {exp}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONSENT_MD.write_text(CONSENT_TEXT)
    consent_sha = sha256_bytes(CONSENT_MD.read_bytes())

    def per_form_max(k):
        return max(len(forms[f][key[k]]) for f in FORM_IDS)

    config = {
        "schema_version": "1.0.0", "study_id": "HS-01", "config_version": "1.0.0",
        "preregistration": {"hypotheses_frozen": False,
            "doc_ref": "01 - Active Projects/Master Thesis/Experiments/HS-01-Human-Validity-Study.md"},
        "pool_ref": {"pool_id": pool["pool_id"], "pool_file_sha256": pool_sha},
        "locale": "en",
        "phases": [
            {"phase_id": "consent", "target_duration_s": 60, "trials_per_rater": None},
            {"phase_id": "text", "target_duration_s": 120, "trials_per_rater": per_form_max("text")},
            {"phase_id": "image", "target_duration_s": 120, "trials_per_rater": per_form_max("image")},
            {"phase_id": "pair", "target_duration_s": 240, "trials_per_rater": per_form_max("pair")},
            {"phase_id": "demographics", "target_duration_s": 60, "trials_per_rater": None},
        ],
        "scales": [
            {"scale_id": "text-comprehensibility-v1", "applies_to": "text",
             "statement": "I can tell what this question is asking", "points": 5,
             "point_labels": ["Strongly disagree","Disagree","Neither agree nor disagree","Agree","Strongly agree"]},
            {"scale_id": "image-clarity-v1", "applies_to": "image",
             "statement": "I can tell what this image is displaying", "points": 5,
             "point_labels": ["Strongly disagree","Disagree","Neither agree nor disagree","Agree","Strongly agree"]},
        ],
        "pair_response": {
            "semantic_options": ["ANCHOR_WORD","TARGET_WORD","OTHER_CLASS","NOTHING_RECOGNIZABLE","CANT_TELL"],
            "display_labels": {"OTHER_CLASS":"Something else","NOTHING_RECOGNIZABLE":"Nothing recognizable","CANT_TELL":"I can't tell"},
            "ab_order": "randomized_per_trial", "fixed_tail": True, "other_class_free_text": True,
        },
        "forms": [forms[f] for f in FORM_IDS],
        "randomization": {"within_phase_shuffle": True, "session_seed_source": "server_generated", "log_seed": True},
        "attention_policy": {"item_ids": sorted(attention.keys()), "placement": "interleaved", "exclusion_fail_threshold": 2},
        "demographics_fields": [
            {"field_id":"age_band","label":"Your age group","type":"select","options":["18_24","25_34","35_44","45_54","55_plus","prefer_not_to_say"],"required":True},
            {"field_id":"ml_familiarity","label":"How familiar are you with machine learning?","type":"select","options":["no_experience","some_exposure","regular_practice","prefer_not_to_say"],"required":True},
            {"field_id":"english_proficiency","label":"Your English proficiency","type":"select","options":["A1","A2","B1","B2","C1","C2","native","prefer_not_to_say"],"required":True},
            {"field_id":"comment","label":"Anything you want to tell us?","type":"free_text","options":None,"required":False},
        ],
        "consent": {"consent_version": "v1", "text_sha256": consent_sha, "required": True},
        "quality": {"log_integrity_events": True, "render_check": True, "min_rendered_image_css_px": 256},
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {CONFIG_JSON} ({len(items)} items, forms "
          + ", ".join(f'{f}:{len(forms[f]["text_items"])}t/{len(forms[f]["image_items"])}i/{len(forms[f]["pair_items"])}p' for f in FORM_IDS) + ")")

def src_for(items, item_id):
    for it in items:
        if it["item_id"] == item_id:
            return it["source_id"]
    raise KeyError(item_id)

if __name__ == "__main__":
    main()
