"""All HS-01 result tables. Each builder returns a DataFrame; `write_tables`
persists them as CSV (repo) plus a human-readable markdown bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as st
from .load import CHOICES, IMG_ORDER, PAIR_ORDER, TEXT_ORDER

OUT = Path(__file__).resolve().parents[1] / "outputs" / "hs01"

BOUNDARY = ["image_heavy", "balanced", "text_heavy"]


def _ci_str(p: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def pair_stratum(pair: pd.DataFrame) -> pd.DataFrame:
    """Headline MIMICRY-mirrored validity table, per stratum + pooled boundary."""
    rows = []
    groups = [(s, pair[pair.stratum == s]) for s in PAIR_ORDER]
    groups.append(("boundary (pooled)", pair[pair.stratum.isin(BOUNDARY)]))
    for name, g in groups:
        row = {"stratum": name, "n_items": g.item_id.nunique(), "n_ratings": len(g)}
        for col, label in [("is_valid", "valid (A or B)"), ("is_ANCHOR_WORD", "A (anchor)"),
                           ("is_TARGET_WORD", "B (target)"), ("is_OTHER_CLASS", "another class"),
                           ("is_NOTHING_RECOGNIZABLE", "nothing recognizable"),
                           ("is_CANT_TELL", "can't tell")]:
            row[label] = _ci_str(*st.boot_ci(g, col))
        rows.append(row)
    return pd.DataFrame(rows)


def unimodal_stratum(text: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, df, order in [("text", text, TEXT_ORDER), ("image", image, IMG_ORDER)]:
        for s in order:
            g = df[df.stratum == s].assign(v=lambda x: x.scale_value.astype(float))
            p, lo, hi = st.boot_ci(g, "v")
            dist = g.scale_value.value_counts(normalize=True).reindex(range(1, 6)).fillna(0)
            rows.append({
                "phase": phase, "stratum": s,
                "n_items": g.item_id.nunique(), "n_ratings": len(g),
                "mean score": _ci_str(p, lo, hi, 2),
                **{f"share {k}": round(dist[k], 3) for k in range(1, 6)},
            })
    return pd.DataFrame(rows)


def comparisons(pair: pd.DataFrame, text: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    """Item-level Mann-Whitney + Cliff's delta vs the phase reference stratum."""
    rows = []
    item_pair = pair.groupby(["stratum", "item_id"]).agg(
        valid=("is_valid", "mean"), A=("is_ANCHOR_WORD", "mean"),
        cant=("is_CANT_TELL", "mean")).reset_index()
    ref = item_pair[item_pair.stratum == "baseline"]
    for s in BOUNDARY:
        g = item_pair[item_pair.stratum == s]
        for m in ["valid", "A", "cant"]:
            d, p = st.cliffs_delta(g[m], ref[m])
            rows.append({"phase": "pair", "stratum": s, "reference": "baseline",
                         "metric": m, "median": g[m].median(),
                         "ref median": ref[m].median(),
                         "cliffs delta": round(d, 3), "MW p": round(p, 4)})
    for phase, df, order, refname in [("text", text, TEXT_ORDER, "clean"),
                                      ("image", image, IMG_ORDER, "raw")]:
        item = df.groupby(["stratum", "item_id"]).scale_value.mean().reset_index()
        ref = item[item.stratum == refname].scale_value
        for s in order[1:]:
            g = item[item.stratum == s].scale_value
            d, p = st.cliffs_delta(g, ref)
            rows.append({"phase": phase, "stratum": s, "reference": refname,
                         "metric": "mean score", "median": g.median(),
                         "ref median": ref.median(),
                         "cliffs delta": round(d, 3), "MW p": round(p, 4)})
    return pd.DataFrame(rows)


def agreement(pair: pd.DataFrame, text: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    """Krippendorff alpha next to prevalence-robust measures (AC1, split-half)."""
    pair = pair.copy()
    pair["choice_code"] = pair.choice.map({c: i for i, c in enumerate(CHOICES)})
    rows = []

    def block(name, df, value_col, level, cats=None):
        alpha = st.kripp_alpha(df, value_col, level)
        if cats is not None:
            ac1, pa = st.gwet_ac1(df, value_col, cats)
        else:
            ac1, pa = np.nan, st.pairwise_agreement(df, value_col)
        rho, sb = st.split_half(df, value_col)
        rows.append({"scope": name, "krippendorff alpha": round(alpha, 3),
                     "pairwise agreement": round(pa, 3),
                     "gwet AC1": round(ac1, 3) if not np.isnan(ac1) else None,
                     "split-half rho": round(rho, 3),
                     "spearman-brown": round(sb, 3)})

    block("text (ordinal)", text, "scale_value", "ordinal")
    block("image (ordinal)", image, "scale_value", "ordinal")
    block("pair (nominal)", pair, "choice_code", "nominal", list(range(5)))
    for s in PAIR_ORDER:
        g = pair[pair.stratum == s]
        alpha = st.kripp_alpha(g, "choice_code", "nominal")
        ac1, pa = st.gwet_ac1(g, "choice_code", list(range(5)))
        rows.append({"scope": f"pair / {s}", "krippendorff alpha": round(alpha, 3),
                     "pairwise agreement": round(pa, 3), "gwet AC1": round(ac1, 3),
                     "split-half rho": None, "spearman-brown": None})
    return pd.DataFrame(rows)


def dose_response(pair: pd.DataFrame, text: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    """Item-level Spearman correlations: human judgment vs search/drift measures."""
    bitem = pair[pair.stratum.isin(BOUNDARY)].groupby("item_id").agg(
        valid=("is_valid", "mean"), d_text=("d_text", "first"),
        d_img=("d_img", "first"), atg=("active_text_genes", "first"),
        tgtbal=("tgtbal", "first")).reset_index()
    bitem["log_tgtbal"] = np.log10(bitem.tgtbal.astype(float))
    titem = text.groupby("item_id").agg(
        score=("scale_value", "mean"), d_text=("d_text", "first"),
        atg=("active_text_genes", "first"), tgtbal=("tgtbal", "first")).reset_index()
    iitem = image[image.stratum.isin(["boundary_joint", "image_heavy"])].groupby("item_id").agg(
        score=("scale_value", "mean"), d_img=("d_img", "first"),
        tgtbal=("tgtbal", "first")).reset_index()
    rows = []
    specs = [
        ("pair validity", bitem.valid, "active_text_genes", bitem.atg),
        ("pair validity", bitem.valid, "d_text", bitem.d_text),
        ("pair validity", bitem.valid, "d_img", bitem.d_img),
        ("pair validity", bitem.valid, "log10 tgtbal", bitem.log_tgtbal),
        ("text score", titem.score, "d_text", titem.d_text),
        ("text score", titem.score, "active_text_genes", titem.atg),
        ("image score", iitem.score, "d_img", iitem.d_img),
    ]
    for outcome, y, pred, x in specs:
        rho, p, n = st.spearman(x, y)
        rows.append({"outcome (item level)": outcome, "predictor": pred,
                     "spearman rho": round(rho, 3), "p": round(p, 4), "n items": n})
    return pd.DataFrame(rows)


def items_pair(pair: pd.DataFrame) -> pd.DataFrame:
    t = pair.groupby(["stratum", "item_id"]).agg(
        sut=("sut", "first"), anchor=("anchor_word", "first"),
        target=("target_word", "first"),
        A=("is_ANCHOR_WORD", "mean"), B=("is_TARGET_WORD", "mean"),
        other=("is_OTHER_CLASS", "mean"), nothing=("is_NOTHING_RECOGNIZABLE", "mean"),
        cant=("is_CANT_TELL", "mean"), valid=("is_valid", "mean"),
        n=("choice", "size"), tgtbal=("tgtbal", "first"),
        d_text=("d_text", "first"), d_img=("d_img", "first"),
        active_text_genes=("active_text_genes", "first"),
        median_rt_s=("rt_s", "median"),
    ).reset_index()
    t["stratum"] = pd.Categorical(t.stratum, PAIR_ORDER, ordered=True)
    return t.sort_values(["stratum", "valid"]).round(3)


def items_unimodal(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    t = df.groupby(["stratum", "item_id"]).agg(
        sut=("sut", "first"), mean_score=("scale_value", "mean"),
        sd=("scale_value", "std"), n=("scale_value", "size"),
        d_text=("d_text", "first"), d_img=("d_img", "first"),
        active_text_genes=("active_text_genes", "first"),
        tgtbal=("tgtbal", "first"), median_rt_s=("rt_s", "median"),
    ).reset_index()
    t["stratum"] = pd.Categorical(t.stratum, order, ordered=True)
    return t.sort_values(["stratum", "mean_score"]).round(3)


def participants(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["age_band", "ml_familiarity", "english", "device",
                "first_unimodal", "form"]:
        counts = sessions[var].value_counts(dropna=False)
        for level, n in counts.items():
            rows.append({"variable": var, "level": level, "n": int(n),
                         "share": round(n / len(sessions), 3)})
    return pd.DataFrame(rows)


def session_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    cols = ["participant", "form", "first_unimodal", "fixed_order_regime",
            "duration_min", "min_text", "min_image", "min_pair",
            "min_demographics", "device", "age_band", "ml_familiarity",
            "english", "attention_failed", "focus_loss", "n_integrity"]
    return sessions[cols].round(2)


def escapes(pair: pd.DataFrame) -> pd.DataFrame:
    e = pair[pair.choice == "OTHER_CLASS"]
    return e[["stratum", "item_id", "sut", "anchor_word", "target_word",
              "other_text"]].reset_index(drop=True)


def quality(frames: dict, sessions_all: pd.DataFrame) -> pd.DataFrame:
    """Instrument-quality facts: funnel, attention, order arm, primacy, refs."""
    pair, text, image = frames["pair"], frames["text"], frames["image"]
    att = frames["attention"]
    comp = frames["sessions"]
    rows = []

    def add(k, v):
        rows.append({"measure": k, "value": v})

    started = len(sessions_all)
    any_trial = (sessions_all.n_trials > 0).sum()
    add("sessions started (consented)", started)
    add("sessions with >=1 trial", int(any_trial))
    add("sessions completed", len(comp))
    add("completion rate", round(len(comp) / started, 3))
    add("median session duration (min)", round(comp.duration_min.median(), 1))
    add("session duration IQR (min)",
        f"[{comp.duration_min.quantile(.25):.1f}, {comp.duration_min.quantile(.75):.1f}]")
    t_att = att[att.phase == "text"]
    p_att = att[att.phase == "pair"]
    add("attention text pass rate (scale<=2)", round((t_att.scale_value <= 2).mean(), 3))
    add("attention pair pass rate (choice=A)", round((p_att.choice == "ANCHOR_WORD").mean(), 3))
    add("raters failing both checks (exclusion rule)", 0)
    cb = ~comp.participant.isin({"P001", "P003", "P004", "P005"})
    add("order arms (counterbalanced): text-first / image-first",
        f'{(comp[cb].first_unimodal == "text").sum()} / {(comp[cb].first_unimodal == "image").sum()}')
    add("fixed text-first regime sessions (pre-switch)", int((~cb).sum()))
    for ph, df in [("text", text), ("image", image)]:
        x = df[df.first_unimodal == "text"].groupby("participant").scale_value.mean()
        y = df[df.first_unimodal == "image"].groupby("participant").scale_value.mean()
        d, p = st.cliffs_delta(x, y)
        add(f"order effect on {ph} score (cliffs delta, MW p)", f"{d:+.3f}, p={p:.3f}")
    a_first = pair[pair.anchor_displayed_first == True]  # noqa: E712
    a_second = pair[pair.anchor_displayed_first == False]  # noqa: E712
    add("P(choose A | A displayed first / second)",
        f'{(a_first.choice == "ANCHOR_WORD").mean():.3f} / {(a_second.choice == "ANCHOR_WORD").mean():.3f}')
    add("pair trials with word-reference opened", round((pair.n_refs_revealed > 0).mean(), 3))
    add("median RT text / image / pair (s)",
        f"{text.rt_s.median():.1f} / {image.rt_s.median():.1f} / {pair.rt_s.median():.1f}")
    add("answer changed before submit (any phase)",
        round((frames["trials"].n_changes > 0).mean(), 3))
    return pd.DataFrame(rows)


def write_tables(frames: dict) -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    pair, text, image = frames["pair"], frames["text"], frames["image"]
    tables = {
        "tab_pair_stratum": pair_stratum(pair),
        "tab_unimodal_stratum": unimodal_stratum(text, image),
        "tab_comparisons": comparisons(pair, text, image),
        "tab_agreement": agreement(pair, text, image),
        "tab_dose_response": dose_response(pair, text, image),
        "tab_items_pair": items_pair(pair),
        "tab_items_text": items_unimodal(text, ["clean", "low_drift", "medium_drift", "high_drift"]),
        "tab_items_image": items_unimodal(image, ["raw", "roundtrip", "boundary_joint", "image_heavy"]),
        "tab_participants": participants(frames["sessions"]),
        "tab_sessions": session_summary(frames["sessions"]),
        "tab_escapes": escapes(pair),
        "tab_quality": quality(frames, frames["sessions_all"]),
    }
    md = ["# HS-01 result tables\n"]
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        md.append(f"## {name}\n\n{df.to_markdown(index=False)}\n")
        print(f"  table: {name} ({len(df)} rows)")
    (OUT / "tables.md").write_text("\n".join(md))
    return tables
