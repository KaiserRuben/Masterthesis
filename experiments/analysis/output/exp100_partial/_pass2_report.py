"""Pass 2: cell-level tables + diagnostic figures from pass-1 artifacts."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"
s = pd.read_csv("/tmp/exp100_state/seed_summary.csv")
npz = np.load("/tmp/exp100_state/trajectories.npz")
sidx = npz["seed_idx"]
min_tb = npz["min_tgtbal"]      # (119, 200)
max_pb = npz["max_pb"]          # (119, 200)
hist_nb = npz["txt_hist_nb"]    # (119, 19) counts out of 50
hist_all = npz["txt_hist_all"]  # (119, 19) counts out of 6000

assert (s["seed_idx"].to_numpy() == sidx).all()

TARGETS = ["ostrich", "green iguana", "boa constrictor", "cello", "marimba"]

# ---------------- Task 1: crossing per cell ----------------
cell = s.groupby(["target", "la", "lt"]).agg(
    n=("seed_idx", "size"),
    n_crossed=("crossed", "sum"),
    mean_frac_crossed=("frac_crossed", "mean"),
    med_first_gen=("first_gen_crossed", lambda x: np.median(x[x >= 0]) if (x >= 0).any() else np.nan),
    med_max_pb=("max_p_class_b", "median"),
    med_min_tgtbal=("min_tgtbal", "median"),
).reset_index()
cell["cross_rate"] = cell["n_crossed"] / cell["n"]
cell.to_csv(os.path.join(OUT, "cell_summary.csv"), index=False)
print("=== cell summary ===")
print(cell.to_string(index=False))

print("\n=== by (la, lt) across targets ===")
lalt = s.groupby(["la", "lt"]).agg(
    n=("seed_idx", "size"), n_crossed=("crossed", "sum"),
    mean_frac_crossed=("frac_crossed", "mean"),
    med_min_tgtbal=("min_tgtbal", "median"),
    med_max_pb=("max_p_class_b", "median"),
    nb_mean_img=("nb_mean_img", "mean"), nb_mean_txt=("nb_mean_txt", "mean"),
    all_mean_img=("all_mean_img", "mean"), all_mean_txt=("all_mean_txt", "mean"),
    rho_img=("rho_tgtbal_img", "mean"), rho_txt=("rho_tgtbal_txt", "mean"),
).reset_index()
lalt["cross_rate"] = lalt["n_crossed"] / lalt["n"]
print(lalt.to_string(index=False))

print("\n=== by target ===")
bt = s.groupby("target").agg(
    n=("seed_idx", "size"), n_crossed=("crossed", "sum"),
    mean_frac_crossed=("frac_crossed", "mean"),
    med_min_tgtbal=("min_tgtbal", "median"),
    med_max_pb=("max_p_class_b", "median"),
    nb_mean_img=("nb_mean_img", "mean"), nb_mean_txt=("nb_mean_txt", "mean"),
).reset_index()
bt["cross_rate"] = bt["n_crossed"] / bt["n"]
print(bt.to_string(index=False))

print("\n=== seeds that never crossed ===")
nc = s[~s["crossed"]][["seed_idx", "target", "la", "lt", "label_a", "label_b",
                       "max_p_class_b", "min_tgtbal"]]
print(nc.to_string(index=False))

# ---------------- Task 2: modality ----------------
print("\n=== modality: near-boundary (top-50 TgtBal) vs full trace ===")
print(f"near-boundary mean active image genes: {s.nb_mean_img.mean():.1f} "
      f"(full-trace {s.all_mean_img.mean():.1f} of 222)")
print(f"near-boundary mean active text genes:  {s.nb_mean_txt.mean():.2f} "
      f"(full-trace {s.all_mean_txt.mean():.2f} of 19)")
print(f"spearman rho(TgtBal, n_act_img): mean {s.rho_tgtbal_img.mean():+.3f}, "
      f"median {s.rho_tgtbal_img.median():+.3f}, "
      f"frac p<0.01 {(s.p_rho_img < 0.01).mean():.2f}, "
      f"frac negative {(s.rho_tgtbal_img < 0).mean():.2f}")
print(f"spearman rho(TgtBal, n_act_txt): mean {s.rho_tgtbal_txt.mean():+.3f}, "
      f"median {s.rho_tgtbal_txt.median():+.3f}, "
      f"frac p<0.01 {(s.p_rho_txt < 0.01).mean():.2f}, "
      f"frac negative {(s.rho_tgtbal_txt < 0).mean():.2f}")

print("\nmodality by level_target:")
print(s.groupby("lt")[["nb_mean_img", "nb_mean_txt", "all_mean_img", "all_mean_txt",
                       "rho_tgtbal_img", "rho_tgtbal_txt"]].mean().to_string())
print("\nmodality by level_anchor:")
print(s.groupby("la")[["nb_mean_img", "nb_mean_txt", "all_mean_img", "all_mean_txt",
                       "rho_tgtbal_img", "rho_tgtbal_txt"]].mean().to_string())

# ---------------- Task 3: text gene usage ----------------
frac_nb = hist_nb.sum(0) / (len(s) * 50)     # activity rate per position near boundary
frac_all = hist_all.sum(0) / (len(s) * 6000)
print("\n=== text gene position activity (fraction of individuals active) ===")
labels = [f"mlm{i}" for i in range(3)] + [f"tog{i}" for i in range(13)] + \
         [f"op{i}" for i in range(3)]
tg = pd.DataFrame({"pos": range(19), "label": labels,
                   "near_boundary": frac_nb.round(3), "full_trace": frac_all.round(3),
                   "enrichment": (frac_nb / np.maximum(frac_all, 1e-9)).round(2)})
print(tg.to_string(index=False))
tg.to_csv(os.path.join(OUT, "text_gene_activity.csv"), index=False)

# ---------------- Task 5: trajectories ----------------
gens = np.arange(200)
# per (la,lt) mean of per-seed min-TgtBal cumulative best and per-gen values
s["lalt"] = list(zip(s["la"], s["lt"]))

# ---------------- figures ----------------
# Fig 1: crossing-rate heatmap (targets x (la,lt))
combos = [(la, lt) for la in (0, 1, 2) for lt in (0, 1, 2)]
M = np.full((len(TARGETS), len(combos)), np.nan)
F = np.full((len(TARGETS), len(combos)), np.nan)
for i, t in enumerate(TARGETS):
    for j, (la, lt) in enumerate(combos):
        r = cell[(cell["target"] == t) & (cell["la"] == la) & (cell["lt"] == lt)]
        if len(r):
            M[i, j] = r.cross_rate.iloc[0]
            F[i, j] = r.mean_frac_crossed.iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, D, title in [(axes[0], M, "seed crossing rate (any eval with p_b > p_a)"),
                     (axes[1], F, "mean fraction of evaluations crossed")]:
    im = ax.imshow(D, cmap="viridis", vmin=0, aspect="auto",
                   vmax=1 if D is M else np.nanmax(F))
    ax.set_xticks(range(len(combos)),
                  [f"la{la}\nlt{lt}" for la, lt in combos], fontsize=8)
    ax.set_yticks(range(len(TARGETS)), TARGETS, fontsize=9)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if not np.isnan(D[i, j]):
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if D[i, j] < 0.6 * np.nanmax(D) else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle("Exp-100 boundary crossing per abstraction cell (n=3 seeds/cell)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trace_crossing_heatmap.png"), dpi=150)
plt.close(fig)

# Fig 2: modality scatter near boundary
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
for lt in (0, 1, 2):
    m = s["lt"] == lt
    axes[0].scatter(100 * s.nb_mean_img_frac[m], s.nb_mean_txt[m], c=colors[lt],
                    label=f"level_target={lt}", alpha=0.75, s=28)
axes[0].scatter(100 * s.all_mean_img_frac.mean(), s.all_mean_txt.mean(), marker="*",
                s=240, c="red", label="full-trace mean", zorder=5)
axes[0].set_xlabel("% active image genes (of 222/276), 50 lowest-TgtBal")
axes[0].set_ylabel("mean active text genes (of 19), 50 lowest-TgtBal")
axes[0].legend(fontsize=8)
axes[0].set_title("near-boundary modality mix per seed")
bp_data = [s.rho_tgtbal_img, s.rho_tgtbal_txt]
axes[1].boxplot(bp_data, tick_labels=["rho(TgtBal, n_img)", "rho(TgtBal, n_txt)"])
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].set_ylabel("Spearman rho over full trace (per seed)")
axes[1].set_title("activity-count vs boundary distance")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trace_modality_scatter.png"), dpi=150)
plt.close(fig)

# Fig 3: text gene position histogram
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(19)
ax.bar(x - 0.2, frac_all, width=0.4, label="full trace", color="lightgray",
       edgecolor="gray")
ax.bar(x + 0.2, frac_nb, width=0.4, label="50 lowest-TgtBal per seed",
       color="tab:red", alpha=0.85)
ax.set_xticks(x, labels, rotation=45, fontsize=8)
ax.axvspan(-0.5, 2.5, color="tab:blue", alpha=0.08)
ax.axvspan(15.5, 18.5, color="tab:green", alpha=0.08)
ax.text(1, ax.get_ylim()[1] * 0.02, "MLM word-swap", fontsize=8, ha="center")
ax.set_ylabel("fraction of individuals with gene != 0")
ax.set_title("text gene position activity: near-boundary vs full trace (119 seeds)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trace_textgene_hist.png"), dpi=150)
plt.close(fig)

# Fig 4: per-(la,lt) average trajectories
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for (la, lt), grp in s.groupby("lalt"):
    rows = np.isin(sidx, grp.seed_idx.to_numpy())
    run_min = np.minimum.accumulate(min_tb[rows], axis=1)
    run_max = np.maximum.accumulate(max_pb[rows], axis=1)
    style = "-" if (la, lt) != (1, 1) else "-"
    lw = 2.6 if (la, lt) == (1, 1) else 1.3
    axes[0].semilogy(gens, run_min.mean(0), lw=lw, label=f"la{la},lt{lt}")
    axes[1].plot(gens, run_max.mean(0), lw=lw, label=f"la{la},lt{lt}")
axes[0].set_xlabel("generation"); axes[0].set_ylabel("running min TgtBal (mean over seeds)")
axes[0].set_title("approach to boundary"); axes[0].legend(fontsize=7, ncol=3)
axes[1].axhline(0.5, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("generation"); axes[1].set_ylabel("running max p_class_b (mean over seeds)")
axes[1].set_title("crossing depth"); axes[1].legend(fontsize=7, ncol=3)
fig.suptitle("Exp-100 trajectory shape per (level_anchor, level_target)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trace_trajectory_lalt.png"), dpi=150)
plt.close(fig)

# Fig 5: per-generation (non-cumulative) min TgtBal: smooth vs jumpy, per target
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for t in TARGETS:
    grp = s[s["target"] == t]
    rows = np.isin(sidx, grp.seed_idx.to_numpy())
    axes[0].semilogy(gens, np.median(min_tb[rows], axis=0), lw=1.4, label=t)
    axes[1].plot(gens, np.median(max_pb[rows], axis=0), lw=1.4, label=t)
axes[0].set_xlabel("generation"); axes[0].set_ylabel("per-gen population min TgtBal (median over seeds)")
axes[0].set_title("per-generation boundary proximity (non-cumulative)")
axes[1].axhline(0.5, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("generation"); axes[1].set_ylabel("per-gen population max p_class_b (median over seeds)")
axes[1].set_title("per-generation crossing depth")
for ax in axes:
    ax.legend(fontsize=8)
fig.suptitle("Exp-100 per-generation population extremes by target class", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trace_trajectory_target.png"), dpi=150)
plt.close(fig)

# jumpiness metric: per-seed std of per-gen min TgtBal diffs after gen 50 (log scale)
late = np.log10(np.maximum(min_tb[:, 50:], 1e-8))
jump = np.abs(np.diff(late, axis=1)).mean(axis=1)
s["jumpiness"] = jump
print("\n=== jumpiness (mean |delta log10 per-gen min TgtBal|, gens 50-199) ===")
print(s.groupby(["la", "lt"])["jumpiness"].median().to_string())
print("\nby target:")
print(s.groupby("target")["jumpiness"].median().to_string())

# (1,1) focus
print("\n=== (la=1,lt=1) vs others ===")
m11 = (s["la"] == 1) & (s["lt"] == 1)
for name, m in [("(1,1)", m11), ("others", ~m11)]:
    print(f"{name}: n={m.sum()}, cross_rate={s[m].crossed.mean():.2f}, "
          f"mean_frac_crossed={s[m].frac_crossed.mean():.3f}, "
          f"med_min_tgtbal={s[m].min_tgtbal.median():.5f}, "
          f"med_max_pb={s[m].max_p_class_b.median():.3f}, "
          f"nb_txt={s[m].nb_mean_txt.mean():.2f}, nb_img={s[m].nb_mean_img.mean():.1f}")

s.to_csv(os.path.join(OUT, "seed_summary.csv"), index=False)
print("\nfigures written.")
