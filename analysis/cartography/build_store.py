#!/usr/bin/env python3
"""Build the boundary-cartography store from boundary-pair run data.

See README.md in this directory for schema and design rationale.

Usage:
    conda run -n uni python -m analysis.cartography.build_store \
        --run-dir runs/Exp-100/poc_boundary_pair \
        --out experiments/analysis/output/cartography/exp100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)

N_TEXT = 19
# Text-gene groups, positions within the trailing 19-gene block
# (mapping established in the Exp-100 trace analysis / src/manipulator/text).
TXT_GROUPS = {
    "mlm": slice(0, 3),
    "frag": slice(3, 8),
    "charnoise": slice(8, 16),
    "saliency": slice(16, 19),
}


def _txt_group_counts(txt_block: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"txt_active_{name}": (txt_block[:, sl] != 0).sum(axis=1).astype(np.int16)
        for name, sl in TXT_GROUPS.items()
    }


def _genotype_features(genos: np.ndarray, image_dim: int,
                       bounds: np.ndarray) -> dict[str, np.ndarray]:
    img = genos[:, :image_dim]
    txt = genos[:, image_dim:]
    img_max = float(bounds[:image_dim].sum())
    txt_max = float(bounds[image_dim:].sum())
    feats = {
        "n_active_img": (img != 0).sum(axis=1).astype(np.int32),
        "n_active_txt": (txt != 0).sum(axis=1).astype(np.int16),
        "rank_sum_img": img.sum(axis=1).astype(np.int64),
        "rank_sum_txt": txt.sum(axis=1).astype(np.int64),
    }
    feats["rank_sum_img_norm"] = feats["rank_sum_img"] / img_max
    feats["rank_sum_txt_norm"] = feats["rank_sum_txt"] / txt_max
    feats["rank_bound_img"] = np.int64(img_max)
    feats["rank_bound_txt"] = np.int64(txt_max)
    feats.update(_txt_group_counts(txt))
    return feats


def _point_feats(g: np.ndarray, image_dim: int,
                 bounds: np.ndarray) -> dict[str, float]:
    """Minimal combinatorial descriptor set for a single genotype."""
    img, txt = g[:image_dim], g[image_dim:]
    return {
        "n_active_img": int((img != 0).sum()),
        "n_active_txt": int((txt != 0).sum()),
        "rank_sum_img_norm": float(img.sum() / bounds[:image_dim].sum()),
        "rank_sum_txt_norm": float(txt.sum() / bounds[image_dim:].sum()),
    }


def _meta_columns(stats: dict, seed_dir: Path, n: int) -> dict:
    md = stats["seed_metadata"]
    return {
        "run": seed_dir.parent.name,
        "seed_dir": seed_dir.name,
        "seed_idx": stats["seed_idx"],
        "anchor_class": md["anchor_class_concrete"],
        "target_class": md["target_class_concrete"],
        "level_anchor": md["level_anchor"],
        "level_target": md["level_target"],
        "common_ancestor_level": md["common_ancestor_level"],
        "seed_idx_in_class": md["seed_idx_in_class"],
        "anchor_word": md["anchor_label_in_prompt"],
        "target_word": md["target_label_in_prompt"],
        "image_dim": stats["image_dim"],
    }


# ---------------------------------------------------------------------------
# SMOO points (pair2 regime)
# ---------------------------------------------------------------------------

def smoo_points(seed_dir: Path) -> pd.DataFrame | None:
    evo = seed_dir / "evolutionary"
    try:
        stats = json.load(open(evo / "stats.json"))
        tr = pd.read_parquet(evo / "trace.parquet",
                             columns=["generation", "individual", "genotype",
                                      "logprobs", "p_class_a", "p_class_b",
                                      "predicted_class",
                                      "fitness_MatrixDistance_fro",
                                      "fitness_TextDist", "fitness_TgtBal"])
    except Exception as e:
        logger.warning("smoo skip %s: %s", seed_dir.name, e)
        return None

    genos = np.array(tr["genotype"].tolist(), dtype=np.int32)
    image_dim = stats["image_dim"]
    bounds = np.asarray(stats["gene_bounds"], dtype=np.int64)
    df = pd.DataFrame(_genotype_features(genos, image_dim, bounds))

    lp = np.array(tr["logprobs"].tolist(), dtype=np.float32)  # (n, 2) pair labels
    df["logprobs"] = list(lp)
    df["pred_label"] = tr["predicted_class"].values
    df["top_gap"] = np.abs(lp[:, 0] - lp[:, 1])
    df["pair_margin"] = lp[:, 0] - lp[:, 1]          # anchor-word side − target-word side
    df["g_pair"] = (tr["p_class_a"] - tr["p_class_b"]).values.astype(np.float32)
    df["d_img_sem"] = tr["fitness_MatrixDistance_fro"].values.astype(np.float32)
    df["d_txt_sem"] = tr["fitness_TextDist"].values.astype(np.float32)
    df["generation"] = tr["generation"].values.astype(np.int16)
    df["step"] = tr["individual"].values.astype(np.int16)
    df["row_ref"] = (tr["generation"].astype(str) + ":"
                     + tr["individual"].astype(str))
    df["candidate_id"] = -1
    df["hamming_to_anchor"] = np.int32(-1)
    df["source"] = "smoo"
    df["prompt_regime"] = "pair2"
    df["genotype"] = [g.astype(np.int16) for g in genos]
    for k, v in _meta_columns(stats, seed_dir, len(df)).items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# PDQ tables (cat6 regime): points + straddles + transects via stage-2 replay
# ---------------------------------------------------------------------------
# candidates.parquet persists only stage-1 candidates; stage-2 trajectory rows
# carry no candidate genotypes (candidate_id_before/after are null). Stage-2
# states are therefore RECONSTRUCTED by replay: start from the stage-1 flip
# genotype, apply each step's (target_gene -> new_value) to the current state;
# accepted steps advance the state, rejected ones are off-path probes. The
# replay was validated externally (end states match archive genotype_min).

def _txt_group_of(gene: int, image_dim: int) -> str | None:
    if gene < image_dim:
        return None
    pos = gene - image_dim
    return next((n for n, sl in TXT_GROUPS.items()
                 if sl.start <= pos < sl.stop), None)


def pdq_tables(
    seed_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    pdq = seed_dir / "pdq"
    evo = seed_dir / "evolutionary"
    try:
        stats = json.load(open(evo / "stats.json"))
        sut = pd.read_parquet(pdq / "sut_calls.parquet",
                              columns=["call_id", "candidate_id", "stage",
                                       "logprobs", "categories", "top1_label",
                                       "logprob_gap_1v2"])
        cand = pd.read_parquet(pdq / "candidates.parquet",
                               columns=["candidate_id", "genotype",
                                        "sut_call_id", "hamming_to_anchor",
                                        "image_pixel_L2", "text_cosine_sum"])
        s1f = pd.read_parquet(pdq / "stage1_flips.parquet",
                              columns=["flip_id", "candidate_id",
                                       "genotype_flipped"])
        arch = pd.read_parquet(pdq / "archive.parquet",
                               columns=["flip_id", "genotype_anchor"])
        traj = pd.read_parquet(pdq / "stage2_trajectories.parquet")
    except Exception as e:
        logger.warning("pdq skip %s: %s", seed_dir.name, e)
        return None, None, None

    md = stats["seed_metadata"]
    cats = list(sut["categories"].iloc[0])
    ai = cats.index(md["anchor_class_concrete"])
    ti = cats.index(md["target_class_concrete"])
    image_dim = stats["image_dim"]
    bounds = np.asarray(stats["gene_bounds"], dtype=np.int64)
    meta = _meta_columns(stats, seed_dir, 0)

    lp_by_call = dict(zip(sut["call_id"],
                          (np.asarray(v, dtype=np.float32)
                           for v in sut["logprobs"])))
    lbl_by_call = dict(zip(sut["call_id"], sut["top1_label"]))
    gap_by_call = dict(zip(sut["call_id"], sut["logprob_gap_1v2"]))

    # Collect (genotype, call_id, source, candidate_id, hamming, d_img, d_txt)
    genos: list[np.ndarray] = []
    rows: list[dict] = []

    def add_point(geno, call_id, source, candidate_id=-1, hamming=-1,
                  d_img=np.nan, d_txt=np.nan, step=-1):
        genos.append(np.asarray(geno, dtype=np.int32))
        rows.append({"call_id": call_id, "source": source,
                     "candidate_id": candidate_id, "hamming": hamming,
                     "d_img": d_img, "d_txt": d_txt, "step": step})

    # -- anchors --
    for ap in sorted((pdq / "anchors").glob("anchor_*.json")):
        a = json.load(open(ap))
        cid = a.get("anchor_call_id")
        if cid in lp_by_call:
            add_point(a["genotype"], cid, "pdq_anchor", hamming=0)

    # -- stage 1 --
    for _, c in cand.iterrows():
        if c["sut_call_id"] in lp_by_call:
            add_point(c["genotype"], c["sut_call_id"], "pdq_s1",
                      candidate_id=c["candidate_id"],
                      hamming=int(c["hamming_to_anchor"]),
                      d_img=c["image_pixel_L2"], d_txt=c["text_cosine_sum"])

    # -- stage 2 replay --
    cand_call = dict(zip(cand["candidate_id"], cand["sut_call_id"]))
    flip_geno = {r["flip_id"]: np.asarray(r["genotype_flipped"], dtype=np.int32)
                 for _, r in s1f.iterrows()}
    flip_init_call = {r["flip_id"]: cand_call.get(r["candidate_id"])
                      for _, r in s1f.iterrows()}
    anchor_geno = {r["flip_id"]: np.asarray(r["genotype_anchor"], dtype=np.int32)
                   for _, r in arch.iterrows()}

    s_rows: list[dict] = []
    t_rows: list[dict] = []
    traj = traj.sort_values(["flip_id", "step"])
    for flip_id, steps in traj.groupby("flip_id", sort=False):
        if flip_id not in flip_geno:
            continue
        state = flip_geno[flip_id].copy()
        ag = anchor_geno.get(flip_id)
        init_call = flip_init_call.get(flip_id)
        lp_cur = lp_by_call.get(init_call)
        lbl_cur = lbl_by_call.get(init_call)
        for _, r in steps.iterrows():
            call_id = r["sut_call_id"]
            lp_new = lp_by_call.get(call_id)
            if lp_new is None:
                continue
            gene = int(r["target_gene"])
            evaluated = state.copy()
            evaluated[gene] = int(r["new_value"])
            ham = int((evaluated != ag).sum()) if ag is not None else -1
            add_point(evaluated, call_id, "pdq_s2", hamming=ham,
                      step=int(r["step"]))

            margin_new = float(lp_new[ai] - lp_new[ti])
            lbl_new = lbl_by_call.get(call_id)
            if lp_cur is not None:
                margin_cur = float(lp_cur[ai] - lp_cur[ti])
                kinds = []
                if (margin_cur > 0) != (margin_new > 0):
                    kinds.append("pair_margin")
                if lbl_cur is not None and lbl_cur != lbl_new:
                    kinds.append("argmax")
                if kinds:
                    fb = _point_feats(state, image_dim, bounds)
                    fa = _point_feats(evaluated, image_dim, bounds)
                    mid = {f"m_{k}": (fb[k] + fa[k]) / 2 for k in fb}
                for kind in kinds:
                    s_rows.append({
                        **meta, "boundary_kind": kind, "flip_id": flip_id,
                        "step": int(r["step"]), "gene_idx": gene,
                        "gene_modality": "txt" if gene >= image_dim else "img",
                        "txt_group": _txt_group_of(gene, image_dim),
                        "value_before": int(state[gene]),
                        "value_after": int(r["new_value"]),
                        "margin_before": margin_cur,
                        "margin_after": margin_new,
                        "label_before": lbl_cur, "label_after": lbl_new,
                        "logprobs_before": lp_cur, "logprobs_after": lp_new,
                        "hamming_to_anchor_after": ham,
                        "call_id_after": call_id,
                        **mid,
                    })
            t_rows.append({
                **meta, "flip_id": flip_id, "step": int(r["step"]),
                "pass_name": r["pass_name"], "accepted": bool(r["accepted"]),
                "still_flipped": bool(r["still_flipped"]),
                "gene_idx": gene,
                "gene_modality": "txt" if gene >= image_dim else "img",
                "old_value": int(state[gene]),
                "new_value": int(r["new_value"]),
                "hamming_to_anchor": ham,
                "pair_margin": margin_new,
                "pred_label": lbl_new,
                "logprobs": lp_new,
                "n_active_img": int((evaluated[:image_dim] != 0).sum()),
                "n_active_txt": int((evaluated[image_dim:] != 0).sum()),
                "call_id": call_id,
            })
            if bool(r["accepted"]):
                state = evaluated
                lp_cur, lbl_cur = lp_new, lbl_new

    # -- assemble points dataframe --
    if not rows:
        return None, None, None
    G = np.vstack([g for g in genos])
    df = pd.DataFrame(_genotype_features(G, image_dim, bounds))
    aux = pd.DataFrame(rows)
    lp = np.vstack([lp_by_call[c] for c in aux["call_id"]])
    df["logprobs"] = list(lp.astype(np.float32))
    df["pred_label"] = [lbl_by_call[c] for c in aux["call_id"]]
    df["top_gap"] = np.array([gap_by_call[c] for c in aux["call_id"]],
                             dtype=np.float32)
    df["pair_margin"] = (lp[:, ai] - lp[:, ti]).astype(np.float32)
    pa = np.exp(lp[:, ai]); pb = np.exp(lp[:, ti])
    df["g_pair"] = ((pa - pb) / (pa + pb)).astype(np.float32)
    df["d_img_sem"] = aux["d_img"].values.astype(np.float32)
    df["d_txt_sem"] = aux["d_txt"].values.astype(np.float32)
    df["generation"] = np.int16(-1)
    df["step"] = aux["step"].values.astype(np.int32)
    df["row_ref"] = aux["call_id"].astype(str).values
    df["candidate_id"] = aux["candidate_id"].values
    df["hamming_to_anchor"] = aux["hamming"].values.astype(np.int32)
    df["source"] = aux["source"].values
    df["prompt_regime"] = "cat6"
    df["genotype"] = [g.astype(np.int16) for g in genos]
    for k, v in meta.items():
        df[k] = v

    straddles = pd.DataFrame(s_rows) if s_rows else None
    transects = pd.DataFrame(t_rows) if t_rows else None
    return df, straddles, transects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path,
                    default=REPO / "runs/Exp-100/poc_boundary_pair")
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/analysis/output/cartography/exp100")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N seed dirs (debug)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(p for p in args.run_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    if args.limit:
        seed_dirs = seed_dirs[:args.limit]

    pts, strads, trans = [], [], []
    for i, sd in enumerate(seed_dirs):
        if not (sd / "evolutionary/stats.json").exists():
            logger.info("incomplete, skipping %s", sd.name)
            continue
        p1 = smoo_points(sd)
        p2, s, t = pdq_tables(sd)
        for buf, df in ((pts, p1), (pts, p2), (strads, s), (trans, t)):
            if df is not None and len(df):
                buf.append(df)
        if (i + 1) % 10 == 0:
            logger.info("processed %d/%d seed dirs", i + 1, len(seed_dirs))

    points = pd.concat(pts, ignore_index=True)
    points.to_parquet(args.out / "points.parquet", compression="zstd")
    logger.info("points.parquet: %d rows", len(points))

    if strads:
        sp = pd.concat(strads, ignore_index=True)
        sp.to_parquet(args.out / "straddle_pairs.parquet", compression="zstd")
        logger.info("straddle_pairs.parquet: %d rows", len(sp))
    if trans:
        tr = pd.concat(trans, ignore_index=True)
        tr.to_parquet(args.out / "transects.parquet", compression="zstd")
        logger.info("transects.parquet: %d rows", len(tr))
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
