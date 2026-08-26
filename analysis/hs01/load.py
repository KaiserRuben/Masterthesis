"""Load HS-01 session records + frozen pool into tidy DataFrames."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HS = REPO / "experiments" / "HS-01"
SESSIONS_DIR = HS / "results" / "sessions"
POOL_FILE = HS / "pool_frozen" / "itempool.json"

# canonical orderings (increasing manipulation / drift)
PAIR_ORDER = ["baseline", "image_heavy", "balanced", "text_heavy"]
TEXT_ORDER = ["clean", "low_drift", "medium_drift", "high_drift"]
IMG_ORDER = ["raw", "roundtrip", "boundary_joint", "image_heavy"]
CHOICES = ["ANCHOR_WORD", "TARGET_WORD", "OTHER_CLASS", "NOTHING_RECOGNIZABLE", "CANT_TELL"]
CHOICE_LABELS = {
    "ANCHOR_WORD": "A (anchor)",
    "TARGET_WORD": "B (target)",
    "OTHER_CLASS": "another class",
    "NOTHING_RECOGNIZABLE": "nothing recognizable",
    "CANT_TELL": "can't tell",
}
STRATUM_LABELS = {
    "baseline": "baseline",
    "image_heavy": "image-heavy",
    "balanced": "balanced",
    "text_heavy": "text-heavy",
    "clean": "clean",
    "low_drift": "low drift",
    "medium_drift": "medium drift",
    "high_drift": "high drift",
    "raw": "raw",
    "roundtrip": "round-trip",
    "boundary_joint": "boundary (joint)",
}
# sessions completed before the counterbalancing redeploy (fixed text-first)
FIXED_ORDER_REGIME = {"P001", "P003", "P004", "P005"}

SUT_SHORT = {
    "OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov": "LLaVA",
    "Qwen/Qwen3.5-4B": "Qwen",
}

META_COLS = [
    "kind", "stratum", "sut", "anchor_class", "target_class", "anchor_word",
    "target_word", "modality", "tgtbal", "d_text", "d_img",
    "active_text_genes", "contains_homoglyphs", "experiment_id", "seed_key",
]


def load_pool() -> dict[str, dict]:
    """item_id -> flat metadata dict (stratum, sut, cell, drift, ...)."""
    pool = json.loads(POOL_FILE.read_text())
    src = {s["source_id"]: s for s in pool["sources"]}
    meta: dict[str, dict] = {}
    for it in pool["items"]:
        s = src.get(it["source_id"], {})
        kind = it["kind"]
        cell = s.get("cell") or {}
        search = s.get("search") or {}
        drift = s.get("drift") or {}
        sut = (s.get("sut") or {}).get("model_id")
        meta[it["item_id"]] = {
            "kind": kind,
            "stratum": (s.get("strata") or {}).get(kind),
            "origin": s.get("origin"),
            "sut": SUT_SHORT.get(sut, sut),
            "anchor_class": cell.get("anchor_class"),
            "target_class": cell.get("target_class"),
            "anchor_word": cell.get("anchor_word"),
            "target_word": cell.get("target_word"),
            "modality": search.get("modality"),
            "tgtbal": search.get("tgtbal"),
            "d_text": drift.get("d_text"),
            "d_img": drift.get("d_img"),
            "active_text_genes": drift.get("active_text_genes"),
            "contains_homoglyphs": ((s.get("assets") or {}).get("prompt") or {}).get("contains_homoglyphs"),
            "experiment_id": (s.get("experiment_ref") or {}).get("experiment_id"),
            "seed_key": s.get("x_seed_key"),
            "is_attention": bool(it.get("is_attention_check")),
        }
    return meta


def _device_class(user_agent: str, is_touch: bool | None) -> str:
    """Fallback for unscrubbed records; published sessions store ``device``.

    See experiments/HS-01/tools/anonymize_sessions.py — the archived records
    carry the derived class instead of the user agent it was derived from.
    """
    ua = user_agent or ""
    if any(k in ua for k in ("iPhone", "Android", "Mobile")):
        return "mobile"
    if "iPad" in ua or (is_touch and "Macintosh" in ua):
        return "tablet"
    if ua:
        return "desktop"
    return "unknown"


def load_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sessions, trials) tidy DataFrames over ALL session records."""
    item_meta = load_pool()
    sess_rows, trial_rows = [], []
    for f in sorted(SESSIONS_DIR.glob("*.json")):
        if f.name == "_counter.json":
            continue
        s = json.loads(f.read_text())
        code = s["participant"]["participant_code"]
        env = s.get("environment") or {}
        demo = s.get("demographics") or {}
        qs = s.get("quality_summary") or {}
        phases = {p["phase_id"]: (p["exited_ms"] - p["entered_ms"]) / 60000
                  for p in s.get("phase_timings", [])}
        unimodal = [p["phase_id"] for p in s.get("phase_timings", [])
                    if p["phase_id"] in ("text", "image")]
        sess_rows.append({
            "participant": code,
            "status": s.get("status"),
            "form": s.get("form_id"),
            "started_utc": s["timing"].get("started_at_utc"),
            "completed_utc": s["timing"].get("completed_at_utc"),
            "duration_min": (s["timing"].get("total_duration_ms") or float("nan")) / 60000,
            "n_trials": len(s.get("trials", [])),
            "first_unimodal": unimodal[0] if unimodal else None,
            "fixed_order_regime": code in FIXED_ORDER_REGIME,
            "min_text": phases.get("text"),
            "min_image": phases.get("image"),
            "min_pair": phases.get("pair"),
            "min_demographics": phases.get("demographics"),
            "age_band": demo.get("age_band"),
            "ml_familiarity": demo.get("ml_familiarity"),
            "english": demo.get("english_proficiency"),
            "comment": demo.get("comment"),
            "device": env.get("device") or _device_class(env.get("user_agent"), env.get("is_touch")),
            "attention_failed": qs.get("attention_failed"),
            "focus_loss": qs.get("focus_loss_count"),
            "n_integrity": len(s.get("integrity_events", [])),
        })
        for t in s.get("trials", []):
            m = item_meta.get(t["item_id"], {})
            r = t.get("response") or {}
            tm = t.get("timing") or {}
            onset, sel = tm.get("onset_ms"), tm.get("response_selected_ms")
            order = (t.get("presented") or {}).get("option_display_order")
            trial_rows.append({
                "participant": code,
                "status": s.get("status"),
                "form": s.get("form_id"),
                "first_unimodal": unimodal[0] if unimodal else None,
                "phase": t["phase_id"],
                "pos": t.get("position_in_phase"),
                "item_id": t["item_id"],
                "is_attention": t.get("is_attention_check", False),
                "scale_value": r.get("scale_value"),
                "choice": r.get("choice"),
                "other_text": r.get("other_class_text"),
                "n_changes": r.get("n_changes"),
                "n_refs_revealed": len(r.get("references_revealed") or []),
                "rt_s": (sel - onset) / 1000 if (sel is not None and onset is not None) else None,
                "anchor_displayed_first": (order[0] == "ANCHOR_WORD") if order else None,
                **{k: m.get(k) for k in META_COLS},
            })
    return pd.DataFrame(sess_rows), pd.DataFrame(trial_rows)


def analysis_frames() -> dict[str, pd.DataFrame]:
    """Primary analysis frames: completed sessions, attention trials split off."""
    sessions, trials = load_sessions()
    completed = sessions[sessions.status == "completed"].copy()
    t = trials[(trials.status == "completed")].copy()
    attention = t[t.is_attention].copy()
    t = t[~t.is_attention].copy()
    pair = t[t.phase == "pair"].copy()
    pair["is_valid"] = pair.choice.isin(["ANCHOR_WORD", "TARGET_WORD"]).astype(float)
    for c in CHOICES:
        pair[f"is_{c}"] = (pair.choice == c).astype(float)
    return {
        "sessions_all": sessions,
        "sessions": completed,
        "trials_all": trials,
        "trials": t,
        "attention": attention,
        "text": t[t.phase == "text"].copy(),
        "image": t[t.phase == "image"].copy(),
        "pair": pair,
    }
