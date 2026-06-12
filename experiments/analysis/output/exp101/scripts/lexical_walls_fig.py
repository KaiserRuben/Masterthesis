"""Slot-asymmetry figure: min_tgtbal_50 for cells containing 'snake'/'songbird',
split by which slot the word occupies, with matched controls."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
cell = pd.read_csv(f"{OUT}/exp101_per_cell.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

panels = {
    "snake": {
        "TARGET slot": [("junco->boa constrictor(0,1)", "junco anc."),
                        ("green iguana->boa constrictor(0,1)", "iguana anc."),
                        ("green iguana->boa constrictor(1,1)", "lizard anc."),
                        ("marimba->boa constrictor(1,1)", "percussion anc.")],
        "TARGET controls (constrictor/reptile)": [
            ("junco->boa constrictor(0,0)", "junco/constr"),
            ("junco->boa constrictor(0,2)", "junco/reptile"),
            ("green iguana->boa constrictor(0,0)", "iguana/constr"),
            ("marimba->boa constrictor(0,0)", "marimba/constr")],
        "ANCHOR slot": [("boa constrictor->green iguana(1,1)", "->lizard"),
                        ("boa constrictor->marimba(1,1)", "->percussion")],
        "ANCHOR controls (constrictor)": [
            ("boa constrictor->green iguana(0,0)", "->iguana"),
            ("boa constrictor->green iguana(0,1)", "->lizard"),
            ("boa constrictor->marimba(0,0)", "->marimba")],
    },
    "songbird": {
        "ANCHOR slot": [("junco->cello(1,0)", "->cello"),
                        ("junco->ostrich(1,1)", "->flightless")],
        "ANCHOR controls (sparrow/bird)": [
            ("junco->cello(0,0)", "sparrow->cello"),
            ("junco->cello(2,0)", "bird->cello"),
            ("junco->ostrich(0,1)", "sparrow->flightless")],
        "TARGET slot": [("cello->junco(0,1)", "cello anc."),
                        ("ostrich->junco(0,1)", "ratite anc."),
                        ("ostrich->junco(1,1)", "flightless anc."),
                        ("boa constrictor->junco(0,1)", "constrictor anc.")],
        "TARGET controls (sparrow)": [
            ("cello->junco(0,0)", "cello anc."),
            ("ostrich->junco(0,0)", "ratite anc."),
            ("boa constrictor->junco(0,0)", "constrictor anc.")],
    },
}
colors = {0: "#c0392b", 1: "#e8a298", 2: "#2471a3", 3: "#a4c8e0"}

for ax, (word, groups) in zip(axes, panels.items()):
    xt, xl = [], []
    x = 0
    for gi, (gname, items) in enumerate(groups.items()):
        for cid, lab in items:
            r = cell[cell.cell_id == cid].iloc[0]
            ax.bar(x, r["min_tgtbal_50"], color=colors[gi], width=0.8)
            xt.append(x)
            xl.append(lab)
            x += 1
        x += 0.7
    ax.set_yscale("log")
    ax.axhline(0.1, color="k", ls="--", lw=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=60, ha="right", fontsize=7)
    ax.set_title(f"'{word}'")
    ax.set_ylabel("min TgtBal @50 gens (nats, log)")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(4)]
axes[0].legend(handles,
               ["word in slot", "slot controls", "word other slot",
                "other-slot controls"], fontsize=7, loc="upper right")
fig.suptitle("Exp-101: wall words by prompt slot (dashed = stuck threshold 0.1)")
fig.tight_layout()
fig.savefig(f"{OUT}/walls_slot_asymmetry.png", dpi=150)
print("wrote", f"{OUT}/walls_slot_asymmetry.png")
