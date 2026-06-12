"""Exp-101 lexical-geometry analysis: label walls, word-vs-slot attachment.

Reads the canonical per-seed/per-cell tables, builds walls_table.csv, prints
all numbers needed for the wall-replication / slot-transfer / generalization
questions, and runs the Exp-100 matched-cell comparison.
"""
import numpy as np
import pandas as pd

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"

seed = pd.read_csv(f"{OUT}/exp101_per_seed.csv")
cell = pd.read_csv(f"{OUT}/exp101_per_cell.csv")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")

# ----------------------------------------------------------------------------
# 0. Sanity: word actually used per (class, level) slot
# ----------------------------------------------------------------------------
print("=" * 80)
print("WORD MAP (class, level) -> word, from anchor and target slots")
wa = seed[["anchor", "la", "anchor_word"]].rename(
    columns={"anchor": "cls", "la": "lvl", "anchor_word": "word"})
wt = seed[["target", "lt", "target_word"]].rename(
    columns={"target": "cls", "lt": "lvl", "target_word": "word"})
wordmap = pd.concat([wa, wt]).drop_duplicates().sort_values(["cls", "lvl"])
print(wordmap.to_string(index=False))
# check word is slot-independent
amb = wordmap.groupby(["cls", "lvl"])["word"].nunique()
print("\nany (class,level) with >1 word across slots?:", (amb > 1).any())

# ----------------------------------------------------------------------------
# 1. Sweeps
# ----------------------------------------------------------------------------
def cells(anchor, target, pairs):
    rows = []
    for la, lt in pairs:
        m = cell[(cell["anchor"] == anchor) & (cell["target"] == target)
                 & (cell["la"] == la) & (cell["lt"] == lt)]
        if len(m):
            rows.append(m.iloc[0])
    return pd.DataFrame(rows)

SWEEPS = {
    "S1 junco->boa  target sweep (la=0, lt 0/1/2)":
        ("junco", "boa constrictor", [(0, 0), (0, 1), (0, 2)]),
    "S2 junco->cello anchor sweep (lt=0, la 0/1/2)":
        ("junco", "cello", [(0, 0), (1, 0), (2, 0)]),
    "S3 boa->junco  target sweep (la=0, lt 0/1/2)":
        ("boa constrictor", "junco", [(0, 0), (0, 1), (0, 2)]),
    "S4 cello->junco target sweep (la=0, lt 0/1/2)":
        ("cello", "junco", [(0, 0), (0, 1), (0, 2)]),
    "S5 ostrich->junco (0,0)/(0,1)/(1,1)":
        ("ostrich", "junco", [(0, 0), (0, 1), (1, 1)]),
    "S6 iguana->boa (0,0)/(0,1)/(1,1)":
        ("green iguana", "boa constrictor", [(0, 0), (0, 1), (1, 1)]),
    "S7 boa->iguana (0,0)/(0,1)/(1,1)  [anchor constrictor vs snake]":
        ("boa constrictor", "green iguana", [(0, 0), (0, 1), (1, 1)]),
    "S8 boa->marimba (0,0)/(1,1) [anchor constrictor vs snake]":
        ("boa constrictor", "marimba", [(0, 0), (1, 1)]),
    "S9 marimba->boa (0,0)/(1,1) [target constrictor vs snake]":
        ("marimba", "boa constrictor", [(0, 0), (1, 1)]),
    "S10 junco->ostrich (0,0)/(0,1)/(1,1) [anchor sparrow vs songbird]":
        ("junco", "ostrich", [(0, 0), (0, 1), (1, 1)]),
}

COLS = ["cell_id", "anchor_word", "target_word", "n", "probe",
        "min_tgtbal_50", "crossed_50_any", "stuck_any",
        "gen_first_cross_mean"]

walls_rows = []
print("\n" + "=" * 80)
for name, (a, t, pairs) in SWEEPS.items():
    df = cells(a, t, pairs)
    print(f"\n--- {name} ---")
    print(df[COLS].to_string(index=False))
    # dex contrast vs easiest cell in the sweep
    base = df["min_tgtbal_50"].min()
    df = df.copy()
    df["dex_vs_sweep_easiest"] = np.log10(df["min_tgtbal_50"] / base)
    df["sweep"] = name.split()[0]
    print("dex_vs_sweep_easiest:",
          ", ".join(f"{w}={d:.2f}" for w, d in
                    zip(df["anchor_word"] + "/" + df["target_word"],
                        df["dex_vs_sweep_easiest"])))
    walls_rows.append(df)

# per-seed detail for the n=2 wall cells (replication spread)
print("\n" + "=" * 80)
print("PER-SEED detail for n=2 cells among sweep cells")
key_cells = pd.concat(walls_rows)["cell_id"].unique()
det = seed[seed.cell_id.isin(key_cells) & (seed.n_in_cell == 2)]
print(det[["cell_id", "seed_idx", "probe", "min_tgtbal_50", "crossed_50",
           "gen_first_cross", "stuck"]].to_string(index=False))

# ----------------------------------------------------------------------------
# 2. Word-attachment tables: every cell containing 'snake' / 'songbird'
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
for word in ["snake", "songbird"]:
    print(f"\n=== ALL cells containing word '{word}' (either slot) ===")
    m = cell[(cell.anchor_word == word) | (cell.target_word == word)].copy()
    m["slot"] = np.where(m.anchor_word == word, "ANCHOR", "TARGET")
    print(m[["cell_id", "slot", "anchor_word", "target_word", "n", "probe",
             "min_tgtbal_50", "crossed_50_any", "stuck_any"]]
          .sort_values(["slot", "min_tgtbal_50"]).to_string(index=False))

# anchor-direction context: all boa-anchored cells (basin check)
print("\n=== ALL boa-anchored cells (direction/basin context) ===")
m = cell[cell.anchor == "boa constrictor"]
print(m[COLS].to_string(index=False))

# ----------------------------------------------------------------------------
# 3. walls_table.csv artifact
# ----------------------------------------------------------------------------
wt_df = pd.concat(walls_rows, ignore_index=True)
wt_df = wt_df[["sweep", "cell_id", "anchor", "target", "la", "lt",
               "anchor_word", "target_word", "n", "probe", "min_tgtbal_50",
               "dex_eroded_50", "dex_vs_sweep_easiest", "crossed_50_any",
               "gen_first_cross_mean", "stuck_any"]]
wt_df = wt_df.rename(columns={"crossed_50_any": "crossed",
                              "stuck_any": "stuck"})
wt_df.to_csv(f"{OUT}/walls_table.csv", index=False)
print(f"\nwrote {OUT}/walls_table.csv  ({len(wt_df)} rows, "
      f"{wt_df.cell_id.nunique()} distinct cells)")

# ----------------------------------------------------------------------------
# 4. Exp-100 matched-cell comparison
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXP-100 MATCHED-CELL COMPARISON")
e100 = pd.read_parquet(
    "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/"
    "exp100_poc_aggregate.parquet")
print("exp100 columns:", list(e100.columns))
print("exp100 shape:", e100.shape)
print(e100.head(3).to_string())
