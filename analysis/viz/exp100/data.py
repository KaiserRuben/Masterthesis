"""The Exp-100 cartography store and its cell vocabulary.

One survey, three tables under ``experiments/analysis/output/cartography/
exp100/``:

points.parquet
    809k scored genotypes from every source: the evolutionary field
    (``smoo``, 2-option prompt regime) and the PDQ anchors, stage-1 probes
    and stage-2 walks (6-option regime).
straddle_pairs.parquet
    10.6k single-gene edits that flip a decision — exact, surveyed
    boundary points with midpoint descriptors.
transects.parquet
    79k stage-2 shrink-walk steps (path-constrained sampling).

All file access for the atlas goes through this module. So does the
vocabulary for addressing the experiment grid: a :class:`Cell` is one
(target class, anchor level, target level) label pair, and
:func:`is_wall` encodes the Exp-100 wall taxonomy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, NamedTuple

import pandas as pd

from .language import EASY_COLOR, WALL_COLOR

REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "experiments/analysis/output/cartography/exp100"
EXPLORE = REPO / "experiments/analysis/output/cartography/explore"
AGGREGATE = REPO / "experiments/analysis/output/exp100_poc_aggregate.parquet"

# Operator-group spans of the 19 text genes, by gene position.
TXT_GENE_GROUPS = {"mlm": (0, 3), "frag": (3, 8),
                   "charnoise": (8, 16), "saliency": (16, 19)}


# ---------------------------------------------------------------------------
# Cell vocabulary
# ---------------------------------------------------------------------------

def is_wall(target: str, level_anchor: int, level_target: int) -> bool:
    """The Exp-100 wall taxonomy.

    Walls hang on specific prompt words, not on classes: the boa wall on
    the target word 'snake' (Lt=1), the cello wall on the anchor word
    'songbird' (La=1).
    """
    return ((target == "boa constrictor" and level_target == 1)
            or (target == "cello" and level_anchor == 1))


class Cell(NamedTuple):
    """One cell of the Exp-100 grid: a (target class, La, Lt) label pair.

    Abstraction levels run 0 (concrete word) to 2 (generic word).
    """

    target: str
    level_anchor: int
    level_target: int

    @property
    def is_wall(self) -> bool:
        return is_wall(self.target, self.level_anchor, self.level_target)

    @property
    def tag(self) -> str:
        return "WALL" if self.is_wall else "EASY"

    @property
    def color(self) -> str:
        return WALL_COLOR if self.is_wall else EASY_COLOR

    @property
    def levels(self) -> str:
        return f"La{self.level_anchor}·Lt{self.level_target}"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return ((df.target_class == self.target)
                & (df.level_anchor == self.level_anchor)
                & (df.level_target == self.level_target))

    def words(self, df: pd.DataFrame) -> tuple[str, str]:
        """The (anchor, target) prompt words this cell was run with."""
        row = df[self.mask(df)].iloc[0]
        return row.anchor_word, row.target_word


# The canonical quartet: both wall species and two easy controls.
BOA_WALL = Cell("boa constrictor", 0, 1)    # target word 'snake'
CELLO_WALL = Cell("cello", 1, 1)            # anchor word 'songbird'
MARIMBA = Cell("marimba", 2, 1)
IGUANA = Cell("green iguana", 2, 0)
QUARTET = (BOA_WALL, CELLO_WALL, MARIMBA, IGUANA)


def cell_key(df: pd.DataFrame) -> pd.Series:
    """Per-row cell identifier, e.g. ``"cello (1,1)"`` — matches the key
    format of :func:`hardness_order`."""
    return (df.target_class + " (" + df.level_anchor.astype(str) + ","
            + df.level_target.astype(str) + ")")


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell key: prompt words, display label, wall flag, target class."""
    info = (df.assign(cell=cell_key(df))
            .groupby("cell")
            .agg(aw=("anchor_word", "first"), tw=("target_word", "first"),
                 tc=("target_class", "first"), la=("level_anchor", "first"),
                 lt=("level_target", "first")))
    info["label"] = ("'" + info["aw"] + "' vs '" + info["tw"] + "'  · "
                     + info["tc"].str.split().str[0])
    info["wall"] = [is_wall(t, a, l)
                    for t, a, l in zip(info["tc"], info["la"], info["lt"])]
    return info


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def points(columns: Iterable[str], **where: object) -> pd.DataFrame:
    """Rows of the survey. Keyword filters become parquet pushdown equality
    filters, e.g. ``points([...], prompt_regime="cat6")``."""
    filters = [(k, "==", v) for k, v in where.items()] or None
    return pd.read_parquet(STORE / "points.parquet",
                           columns=list(columns), filters=filters)


def evolutionary_field(extra: Iterable[str] = ()) -> pd.DataFrame:
    """The evolutionary survey around the junco anchor, ready for pooling.

    The semantic axes ``d_img_sem`` / ``d_txt_sem`` live on a different
    scale for every seed (up to 5× within one cell), so pooling seeds raw
    smears any structure. Each axis is therefore normalized per seed by its
    q99 and clipped at 1.25; the resulting ``*_n`` columns are the only
    semantic coordinates the atlas plots pooled.
    """
    cols = ["target_class", "level_anchor", "level_target", "anchor_word",
            "target_word", "seed_dir", "generation", "g_pair",
            "d_img_sem", "d_txt_sem"]
    cols += [c for c in extra if c not in cols]
    df = points(cols, source="smoo", anchor_class="junco")
    for c in ("d_img_sem", "d_txt_sem"):
        q = df.groupby("seed_dir")[c].transform(lambda s: s.quantile(0.99))
        df[c + "_n"] = (df[c] / q).clip(upper=1.25)
    return df


def straddles(*, kind: str | None = None) -> pd.DataFrame:
    """Surveyed boundary crossings, optionally one ``boundary_kind``:
    ``"argmax"`` (the 6-way prediction flips) or ``"pair_margin"`` (the
    anchor-vs-target margin changes sign)."""
    s = pd.read_parquet(STORE / "straddle_pairs.parquet")
    return s if kind is None else s[s.boundary_kind == kind].copy()


def transects(columns: Iterable[str], *,
              accepted_only: bool = True) -> pd.DataFrame:
    """Stage-2 shrink-walk steps; by default only the accepted ones."""
    cols = list(dict.fromkeys([*columns, "accepted"]))
    t = pd.read_parquet(STORE / "transects.parquet", columns=cols)
    return t[t.accepted].copy() if accepted_only else t


def hardness_order() -> pd.Series:
    """Cells ordered easy → hard: per-cell median of the best |P(A) − P(B)|
    reached under the 2-option regime (Exp-100 PoC aggregate)."""
    agg = pd.read_parquet(AGGREGATE)
    agg = agg[agg.run == "poc_boundary_pair"]
    key = (agg.target_class_concrete + " (" + agg.level_anchor.astype(str)
           + "," + agg.level_target.astype(str) + ")")
    return agg.groupby(key)["min_TgtBal"].median().sort_values()


def crispness_benchmark() -> pd.DataFrame:
    """kNN-AUC of side-separability per (projection × regime × cell), from
    the v3 axis-choice exploration."""
    return pd.read_csv(EXPLORE / "v3_crispness_benchmark.csv")
