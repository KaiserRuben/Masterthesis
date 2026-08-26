"""All HS-01 figures. Output: Obsidian Diary/assets/hs01/ (via style.asset_dir)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.core.style import apply_style, asset_dir, save_fig, subplot_label
from . import stats as st
from .load import CHOICES, CHOICE_LABELS, IMG_ORDER, PAIR_ORDER, TEXT_ORDER

ASSETS = None  # set in generate_all()

# house palette (analysis.core.style) — seaborn-muted family, no pipeline colors
LIKERT_COLORS = {1: "#C44E52", 2: "#DD8452", 3: "#CCCCCC", 4: "#64B5CD", 5: "#4C72B0"}
LIKERT_LABELS = {1: "strongly disagree", 2: "disagree", 3: "neutral",
                 4: "agree", 5: "strongly agree"}
PAIR_COLORS = {"ANCHOR_WORD": "#4C72B0", "TARGET_WORD": "#C44E52",
               "OTHER_CLASS": "#8172B3", "NOTHING_RECOGNIZABLE": "#937860",
               "CANT_TELL": "#CCB974"}
STRATUM_COLORS = {"baseline": "#4C72B0", "image_heavy": "#55A868",
                  "balanced": "#8172B3", "text_heavy": "#C44E52",
                  "clean": "#4C72B0", "low_drift": "#64B5CD",
                  "medium_drift": "#DD8452", "high_drift": "#C44E52",
                  "raw": "#4C72B0", "roundtrip": "#64B5CD",
                  "boundary_joint": "#8172B3"}
GRAY_NOTE = "#777777"


def _gray_subtitle(fig, text: str, y: float = 0.955):
    fig.text(0.5, y, text, ha="center", fontsize=8.5, color=GRAY_NOTE)
BOUNDARY = ["image_heavy", "balanced", "text_heavy"]

AGE_ORDER = ["18_24", "25_34", "35_44", "45_54", "55_plus"]
ML_ORDER = ["no_experience", "some_exposure", "regular_practice"]
EN_ORDER = ["B1", "B2", "C1", "C2", "native"]
NICE = {"18_24": "18–24", "25_34": "25–34", "35_44": "35–44", "45_54": "45–54",
        "55_plus": "55+", "no_experience": "none", "some_exposure": "some",
        "regular_practice": "regular", "native": "native",
        "image_heavy": "image-heavy", "text_heavy": "text-heavy",
        "low_drift": "low", "medium_drift": "medium", "high_drift": "high",
        "boundary_joint": "boundary", "roundtrip": "round-trip"}


# self-explanatory stratum definitions (thresholds from stage_pool_candidates.py;
# d_text = embedding distance prompt vs original, d_img = matrix distance image vs round-trip)
STRATUM_DEFS = {
    "clean": "original prompt",
    "low_drift": "d_text ≤ 0.30",
    "medium_drift": "d_text 0.30–0.55",
    "high_drift": "d_text > 0.55",
    "raw": "original photo",
    "roundtrip": "VQGAN encode–decode",
    "boundary_joint": "joint search",
    "image_heavy": "image-only search, d_text = 0",
    "baseline": "round-trip image + clean prompt",
    "balanced": "d_text 0.20–0.60, d_img > 0.001",
    "text_heavy": "d_text > 0.40, d_img < 0.01",
}


def _nice(v: str) -> str:
    return NICE.get(v, str(v))


def _deflabel(s: str) -> str:
    d = STRATUM_DEFS.get(s)
    return f"{_nice(s)} ({d})" if d else _nice(s)


def _counts_panel(ax, series: pd.Series, order: list[str], title: str, color: str):
    counts = series.value_counts().reindex(order).fillna(0)
    counts = counts[counts > 0]
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=color, height=0.65)
    ax.set_yticks(y, [_nice(v) for v in counts.index])
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + 0.3, i, f"{int(v)} ({v / len(series):.0%})", va="center", fontsize=8)
    ax.set_xlim(0, counts.max() * 1.35)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", visible=False)


def fig_participants(sessions: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.6))
    _counts_panel(axes[0, 0], sessions.age_band, AGE_ORDER, "Age band", "#4C72B0")
    _counts_panel(axes[0, 1], sessions.ml_familiarity, ML_ORDER, "ML familiarity", "#55A868")
    _counts_panel(axes[1, 0], sessions.english, EN_ORDER, "English proficiency", "#C44E52")
    _counts_panel(axes[1, 1], sessions.device, ["desktop", "mobile", "tablet"], "Device", "#8172B3")
    fig.suptitle(f"HS-01 — raters (n = {len(sessions)} completed sessions)", fontsize=12)
    return save_fig(fig, ASSETS / "hs01_participants.png")


def fig_process(sessions_all: pd.DataFrame, sessions: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
    # (a) sessions per day, stacked by outcome
    ax = axes[0, 0]
    df = sessions_all.copy()
    df["day"] = pd.to_datetime(df.started_utc).dt.strftime("%m-%d")
    days = sorted(df.day.unique())
    comp = df[df.status == "completed"].day.value_counts().reindex(days).fillna(0)
    aban = df[df.status == "abandoned"].day.value_counts().reindex(days).fillna(0)
    x = np.arange(len(days))
    ax.bar(x, comp.values, color="#4C72B0", label="completed")
    ax.bar(x, aban.values, bottom=comp.values, color="#CCCCCC", label="abandoned")
    ax.set_xticks(x, days, rotation=45, fontsize=8)
    ax.set_ylabel("sessions")
    ax.set_title("Collection window", fontsize=10)
    ax.legend(fontsize=8)
    subplot_label(ax, "a")
    # (b) funnel
    ax = axes[0, 1]
    stages = [("opened + consented", len(sessions_all)),
              ("answered ≥ 1 trial", int((sessions_all.n_trials > 0).sum())),
              ("completed", len(sessions))]
    y = np.arange(len(stages))
    vals = [v for _, v in stages]
    ax.barh(y, vals, color=["#999999", "#64B5CD", "#4C72B0"], height=0.6)
    ax.set_yticks(y, [s for s, _ in stages])
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v} ({v / vals[0]:.0%})", va="center", fontsize=9)
    ax.set_xlim(0, vals[0] * 1.25)
    ax.set_title("Completion funnel", fontsize=10)
    ax.grid(axis="y", visible=False)
    subplot_label(ax, "b")
    # (c) session duration
    ax = axes[1, 0]
    d = sessions.duration_min.dropna()
    ax.hist(d, bins=np.arange(0, np.ceil(d.max()) + 1, 1), color="#4C72B0", edgecolor="white")
    ax.axvline(d.median(), color="#C44E52", lw=1.5,
               label=f"median {d.median():.1f} min")
    ax.set_xlabel("session duration (min)")
    ax.set_ylabel("sessions")
    ax.set_title("Duration (completed)", fontsize=10)
    ax.legend(fontsize=8)
    subplot_label(ax, "c")
    # (d) minutes per phase
    ax = axes[1, 1]
    cols = [("min_text", "text"), ("min_image", "image"),
            ("min_pair", "pair"), ("min_demographics", "demogr.")]
    data = [sessions[c].dropna().values for c, _ in cols]
    bp = ax.boxplot(data, vert=False, tick_labels=[l for _, l in cols],
                    patch_artist=True, widths=0.55,
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="none", markeredgecolor="#555555"))
    for patch, color in zip(bp["boxes"], ["#C44E52", "#4C72B0", "#55A868", "#999999"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.invert_yaxis()
    ax.set_xlabel("minutes in phase")
    ax.set_title("Where the time goes", fontsize=10)
    ax.grid(axis="y", visible=False)
    subplot_label(ax, "d")
    fig.suptitle("HS-01 — collection and session flow", fontsize=12, y=1.0)
    return save_fig(fig, ASSETS / "hs01_process.png")


def _pair_stack(ax, pair: pd.DataFrame, order: list[str], annotate_ci: bool = True):
    ys = np.arange(len(order))
    for yi, s in enumerate(order):
        g = pair[pair.stratum == s]
        left = 0.0
        for c in CHOICES:
            share = (g.choice == c).mean()
            ax.barh(yi, share, left=left, color=PAIR_COLORS[c], height=0.62)
            if share >= 0.045:
                ax.text(left + share / 2, yi, f"{share * 100:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if c in ("ANCHOR_WORD", "TARGET_WORD",
                                               "NOTHING_RECOGNIZABLE", "OTHER_CLASS") else "black")
            left += share
        if annotate_ci:
            p, lo, hi = st.boot_ci(g, "is_valid")
            ax.text(1.02, yi, f"{p:.2f} [{lo:.2f}, {hi:.2f}]",
                    va="center", fontsize=8.5, transform=ax.get_yaxis_transform())
    labels = [f"{_deflabel(s)}\n{pair[pair.stratum == s].item_id.nunique()} items · "
              f"{len(pair[pair.stratum == s])} ratings" for s in order]
    ax.set_yticks(ys, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    if annotate_ci:
        ax.text(1.02, 1.0, "valid = P(A∨B)\n[95% CI]", fontsize=7.5,
                transform=ax.transAxes, style="italic", va="bottom")


def fig_pair_outcomes(pair: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    _pair_stack(ax, pair, PAIR_ORDER)
    handles = [Patch(color=PAIR_COLORS[c], label=CHOICE_LABELS[c]) for c in CHOICES]
    ax.legend(handles=handles, ncol=5, fontsize=7.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("HS-01 — pair phase: class judgment on the exact SUT input", fontsize=12)
    fig.subplots_adjust(right=0.82)
    return save_fig(fig, ASSETS / "hs01_pair_outcomes.png", tight=False)


def _likert_stack(ax, df: pd.DataFrame, order: list[str]):
    """Diverging stacked Likert bars, centered on the neutral midpoint."""
    for yi, s in enumerate(order):
        g = df[df.stratum == s]
        shares = g.scale_value.value_counts(normalize=True).reindex(range(1, 6)).fillna(0)
        left = -(shares[1] + shares[2] + shares[3] / 2)
        for k in range(1, 6):
            ax.barh(yi, shares[k], left=left, color=LIKERT_COLORS[k], height=0.62)
            if shares[k] >= 0.05:
                ax.text(left + shares[k] / 2, yi, f"{shares[k] * 100:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if k in (1, 5) else "black")
            left += shares[k]
        p, lo, hi = st.boot_ci(g.assign(v=g.scale_value.astype(float)), "v")
        ax.text(1.02, yi, f"{p:.2f} [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=8.5, transform=ax.get_yaxis_transform())
    labels = [f"{_deflabel(s)}\n{df[df.stratum == s].item_id.nunique()} items · "
              f"{len(df[df.stratum == s])} ratings" for s in order]
    ax.set_yticks(np.arange(len(order)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -.5, 0, .5, 1], ["100%", "50%", "0", "50%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.text(1.02, 1.0, "mean score\n[95% CI]", fontsize=7.5,
            transform=ax.transAxes, style="italic", va="bottom")


def _likert_legend(ax):
    handles = [Patch(color=LIKERT_COLORS[k], label=f"{k} – {LIKERT_LABELS[k]}")
               for k in range(1, 6)]
    ax.legend(handles=handles, ncol=5, fontsize=7.3, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), frameon=False)


def fig_text_likert(text: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    _likert_stack(ax, text, TEXT_ORDER)
    _likert_legend(ax)
    ax.set_title('HS-01 — text phase: "I can tell what this question is asking"', fontsize=12)
    fig.subplots_adjust(right=0.82)
    return save_fig(fig, ASSETS / "hs01_text_likert.png", tight=False)


def fig_image_likert(image: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    _likert_stack(ax, image, IMG_ORDER)
    _likert_legend(ax)
    ax.set_title('HS-01 — image phase: "I can tell what this image is displaying"', fontsize=12)
    fig.subplots_adjust(right=0.82)
    return save_fig(fig, ASSETS / "hs01_image_likert.png", tight=False)


def fig_validity_overview(text: pd.DataFrame, image: pd.DataFrame, pair: pd.DataFrame):
    """Single thesis-ready overview: all three phases, one visual language."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2),
                             gridspec_kw={"height_ratios": [4, 4, 4]})
    _likert_stack(axes[0], text, TEXT_ORDER)
    axes[0].set_title('Text — "I can tell what this question is asking"', fontsize=10.5)
    subplot_label(axes[0], "a", x=-0.28)
    _likert_stack(axes[1], image, IMG_ORDER)
    axes[1].set_title('Image — "I can tell what this image is displaying"', fontsize=10.5)
    subplot_label(axes[1], "b", x=-0.28)
    _likert_legend(axes[1])
    _pair_stack(axes[2], pair, PAIR_ORDER)
    axes[2].set_title("Pair — class judgment on the exact SUT input", fontsize=10.5)
    subplot_label(axes[2], "c", x=-0.28)
    handles = [Patch(color=PAIR_COLORS[c], label=CHOICE_LABELS[c]) for c in CHOICES]
    axes[2].legend(handles=handles, ncol=5, fontsize=7.3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.24), frameon=False)
    fig.subplots_adjust(right=0.8, hspace=0.75)
    return save_fig(fig, ASSETS / "hs01_validity_overview.png", tight=False)


def fig_dose_response(pair: pd.DataFrame, text: pd.DataFrame, image: pd.DataFrame):
    bitem = pair[pair.stratum.isin(BOUNDARY)].groupby("item_id").agg(
        valid=("is_valid", "mean"), stratum=("stratum", "first"),
        d_text=("d_text", "first"), d_img=("d_img", "first"),
        atg=("active_text_genes", "first"), tgtbal=("tgtbal", "first")).reset_index()
    bitem["log_tgtbal"] = np.log10(bitem.tgtbal.astype(float))
    titem = text.groupby("item_id").agg(
        score=("scale_value", "mean"), stratum=("stratum", "first"),
        d_text=("d_text", "first"), atg=("active_text_genes", "first")).reset_index()
    iitem = image[image.stratum.isin(["boundary_joint", "image_heavy"])].groupby("item_id").agg(
        score=("scale_value", "mean"), stratum=("stratum", "first"),
        d_img=("d_img", "first")).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    panels = [
        (axes[0, 0], bitem, "atg", "valid", "active text genes", "pair validity"),
        (axes[0, 1], bitem, "d_text", "valid", "text drift $d_{text}$", "pair validity"),
        (axes[0, 2], bitem, "log_tgtbal", "valid", "$\\log_{10}$ TgtBal (boundary distance)", "pair validity"),
        (axes[1, 0], titem, "atg", "score", "active text genes", "text score"),
        (axes[1, 1], titem, "d_text", "score", "text drift $d_{text}$", "text score"),
        (axes[1, 2], iitem, "d_img", "score", "image drift $d_{img}$", "image score"),
    ]
    canon = list(dict.fromkeys(PAIR_ORDER + TEXT_ORDER + IMG_ORDER))
    for i, (ax, df, xc, yc, xl, yl) in enumerate(panels):
        sub = df.dropna(subset=[xc, yc])
        strata = [s for s in canon if s in set(sub.stratum)]
        for s in strata:
            g = sub[sub.stratum == s]
            ax.scatter(g[xc], g[yc], s=30, alpha=0.85,
                       color=STRATUM_COLORS.get(s, "#999999"), label=_deflabel(s),
                       edgecolor="white", linewidth=0.5)
        rho, p, n = st.spearman(sub[xc], sub[yc])
        ax.set_title(f"ρ = {rho:+.2f}, p = {p:.3f}, n = {n}",
                     fontsize=9, color=GRAY_NOTE)
        subplot_label(ax, "abcdef"[i], x=-0.16)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.margins(y=0.18)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.9,
                  handletextpad=0.2, borderpad=0.25)
    fig.suptitle("HS-01 — item-level dose–response", fontsize=12, y=1.0)
    _gray_subtitle(fig, "human judgment vs manipulation heaviness and boundary "
                        "distance · one dot = one item · Spearman ρ", y=0.965)
    return save_fig(fig, ASSETS / "hs01_dose_response.png")


def fig_items_pair_heatmap(pair: pd.DataFrame):
    item = pair.groupby(["stratum", "item_id"]).agg(
        anchor=("anchor_word", "first"), target=("target_word", "first"),
        exp=("experiment_id", "first"),
        **{c: (f"is_{c}", "mean") for c in CHOICES}).reset_index()
    item["stratum"] = pd.Categorical(item.stratum, PAIR_ORDER, ordered=True)
    item = item.sort_values(["stratum", "ANCHOR_WORD"], ascending=[True, False])
    n = len(item)
    fig, ax = plt.subplots(figsize=(8.6, 0.30 * n + 1.8))
    mat = item[CHOICES].to_numpy(float)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for yi in range(n):
        for xi in range(5):
            v = mat[yi, xi]
            if v >= 0.005:
                ax.text(xi, yi, f"{v * 100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.55 else "#333333")
    labels = [f"{r.anchor} → {r.target}  ({str(r.exp).replace('Exp-', 'E')})"
              for r in item.itertuples()]
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xticks(range(5), ["A\n(anchor)", "B\n(target)", "another\nclass",
                             "nothing\nrecognizable", "can't\ntell"], fontsize=8)
    ax.tick_params(length=0)
    # stratum separators + right-margin labels
    y0 = 0
    for s in PAIR_ORDER:
        k = (item.stratum == s).sum()
        if y0:
            ax.axhline(y0 - 0.5, color="#C44E52", lw=1.2)
        ax.text(1.015, y0 + k / 2 - 0.5, _nice(s), rotation=270,
                va="center", ha="center", fontsize=9, fontweight="bold",
                color=STRATUM_COLORS[s], transform=ax.get_yaxis_transform())
        y0 += k
    ax.set_title("HS-01 — pair phase, per item: vote shares (%)", fontsize=12)
    fig.subplots_adjust(left=0.37, right=0.95)
    return save_fig(fig, ASSETS / "hs01_items_pair_heatmap.png", tight=False)


def fig_items_unimodal(text: pd.DataFrame, image: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, df, order, title in [(axes[0], text, TEXT_ORDER, "Text items"),
                                 (axes[1], image, IMG_ORDER, "Image items")]:
        item = df.groupby(["stratum", "item_id"]).agg(
            m=("scale_value", "mean"), sd=("scale_value", "std"),
            n=("scale_value", "size")).reset_index()
        item["stratum"] = pd.Categorical(item.stratum, order, ordered=True)
        item = item.sort_values(["stratum", "m"], ascending=[True, False]).reset_index(drop=True)
        ci = 1.96 * item.sd / np.sqrt(item.n)
        colors = item.stratum.map(STRATUM_COLORS)
        ax.errorbar(item.m, item.index, xerr=ci, fmt="none",
                    ecolor="#bbbbbb", elinewidth=1)
        ax.scatter(item.m, item.index, c=colors, s=30, zorder=3)
        y0 = 0
        for s in order:
            k = (item.stratum == s).sum()
            if y0:
                ax.axhline(y0 - 0.5, color="#dddddd", lw=1)
            ax.text(1.02, y0 + k / 2 - 0.5,
                    f"{_nice(s)}\n{STRATUM_DEFS.get(s, '')}", fontsize=7,
                    va="center", color=STRATUM_COLORS[s], fontweight="bold",
                    transform=ax.get_yaxis_transform())
            y0 += k
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(1, 5)
        ax.set_xlabel("mean score (± 95% CI)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("HS-01 — per-item mean scores by stratum", fontsize=12)
    return save_fig(fig, ASSETS / "hs01_items_unimodal.png")


def fig_rt(trials: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.2), sharex=True)
    specs = [("text", TEXT_ORDER), ("image", IMG_ORDER), ("pair", PAIR_ORDER)]
    for ax, (phase, order) in zip(axes, specs):
        df = trials[trials.phase == phase]
        data = [df[df.stratum == s].rt_s.dropna().values for s in order]
        bp = ax.boxplot(data, vert=False,
                        tick_labels=[_deflabel(s) for s in order],
                        patch_artist=True, widths=0.55,
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="none",
                                        markeredgecolor="#555555"))
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(STRATUM_COLORS[s])
            patch.set_alpha(0.65)
        for yi, s in enumerate(order):
            med = df[df.stratum == s].rt_s.median()
            ax.text(med, yi + 1.38, f"{med:.1f}s", fontsize=7.5, ha="center")
        ax.invert_yaxis()
        ax.set_title(f"{phase} phase", fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
    n_clip = int((trials.rt_s > 30).sum())
    axes[-1].set_xlim(0, 30)
    axes[-1].set_xlabel("time to response (s)")
    fig.suptitle("HS-01 — time to response by stratum", fontsize=12, y=1.045)
    _gray_subtitle(fig, "response_selected − stimulus onset, per trial · median "
                        f"annotated · axis clipped at 30 s ({n_clip} trials beyond)",
                   y=1.005)
    return save_fig(fig, ASSETS / "hs01_rt.png")


def fig_agreement(agreement_table: pd.DataFrame):
    df = agreement_table.copy()
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(df) + 1.6))
    y = np.arange(len(df))
    ax.scatter(df["pairwise agreement"], y, marker="o", s=55, facecolors="none",
               edgecolors="#4C72B0", linewidth=1.6, label="pairwise agreement", zorder=3)
    ax.scatter(df["gwet AC1"], y, marker="D", s=42, color="#55A868",
               label="Gwet AC1", zorder=3)
    ax.scatter(df["krippendorff alpha"], y, marker="s", s=42, color="#D64933",
               label="Krippendorff α", zorder=3)
    ax.axvline(0.67, color="#999999", ls="--", lw=1,
               label="α = 0.67 (design threshold)")
    ax.set_yticks(y, df.scope, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("agreement / reliability coefficient")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("HS-01 — inter-rater agreement by scope", fontsize=12, pad=22)
    ax.text(0.5, 1.015, "α collapses where consensus is high — near-unanimous "
                        "baseline votes leave α no variance (prevalence paradox)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=GRAY_NOTE)
    ax.grid(axis="y", visible=False)
    return save_fig(fig, ASSETS / "hs01_agreement.png")


def fig_order_effect(text: pd.DataFrame, image: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    rng = np.random.default_rng(3)
    for ax, (phase, df) in zip(axes, [("text", text), ("image", image)]):
        rater = df.groupby(["participant", "first_unimodal"]).scale_value.mean().reset_index()
        for xi, arm in enumerate(["text", "image"]):
            g = rater[rater.first_unimodal == arm]
            jitter = rng.uniform(-0.12, 0.12, len(g))
            ax.scatter(np.full(len(g), xi) + jitter, g.scale_value, s=26,
                       alpha=0.7, color="#4C72B0" if arm == "text" else "#55A868")
            ax.hlines(g.scale_value.mean(), xi - 0.22, xi + 0.22,
                      color="black", lw=2)
        x = rater[rater.first_unimodal == "text"].scale_value
        y = rater[rater.first_unimodal == "image"].scale_value
        d, p = st.cliffs_delta(x, y)
        ax.set_title(f"{phase} scores", fontsize=10, pad=18)
        ax.text(0.5, 1.02, f"δ = {d:+.2f}, p = {p:.2f}", transform=ax.transAxes,
                ha="center", fontsize=8.5, color=GRAY_NOTE)
        ax.set_xticks([0, 1], ["text first", "image first"])
        ax.set_ylim(1, 5)
    axes[0].set_ylabel("rater mean score")
    fig.suptitle("HS-01 — unimodal scores by phase-order arm", fontsize=12, y=1.02)
    _gray_subtitle(fig, "one dot = one rater's mean · bar = arm mean · "
                        "Mann-Whitney on rater means", y=0.93)
    return save_fig(fig, ASSETS / "hs01_order_effect.png")


def generate_all(frames: dict, tables: dict):
    global ASSETS
    apply_style()
    ASSETS = asset_dir("hs01")
    text, image, pair = frames["text"], frames["image"], frames["pair"]
    fig_participants(frames["sessions"])
    fig_process(frames["sessions_all"], frames["sessions"])
    fig_pair_outcomes(pair)
    fig_text_likert(text)
    fig_image_likert(image)
    fig_validity_overview(text, image, pair)
    fig_dose_response(pair, text, image)
    fig_items_pair_heatmap(pair)
    fig_items_unimodal(text, image)
    fig_rt(frames["trials"])
    fig_agreement(tables["tab_agreement"])
    fig_order_effect(text, image)
    return ASSETS
