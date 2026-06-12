"""Exp-101 data-quality / integrity audit.

Checks (read-only on runs/Exp-101):
1. Completeness  - 46 dirs, trace.parquet 1500 rows (gens 0..49 x 30 ind),
                   stats.json, convergence.parquet, context.json present.
2. Budget        - generations=50, pop_size=30 everywhere; every trace reaches
                   gen 49 (early stop never fired); runtime distribution.
3. Backend       - which image backend actually ran (cone vs KNN):
                   context.json image_candidate_strategy / image_backend /
                   image_target_class; stats.json echo field; gene_bounds
                   structure; image_dim across runs.
4. Cell design   - reconstruct 40-cell x n design from seed_metadata, strata
                   counts, n=2 cells, anchor coverage.
5. Column sanity - NaNs, p_a+p_b<=1, TgtBal == |logprobs diff|, duplicate
                   (generation, individual) keys, cache_hit rates.

Outputs: per-run CSV + cell-design CSV + printed findings (stdout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/Users/kaiser/Projects/Masterarbeit/runs/Exp-101")
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101")

EXPECTED_FILTER_INDICES = [
    0, 1, 2, 6, 26, 28, 29, 30, 44, 50,
    56, 80, 82, 86, 88, 96, 196, 198, 202, 203,
    204, 205, 212, 240, 242, 243, 244, 276, 278, 282,
    302, 310, 320, 322, 324, 338, 354, 392, 393, 394,
    398, 454, 462, 472, 474, 478,
]

issues: list[tuple[str, str]] = []  # (severity, message)


def flag(severity: str, msg: str) -> None:
    issues.append((severity, msg))
    print(f"[{severity.upper()}] {msg}")


def main() -> None:
    dirs = sorted(RUNS.glob("exp101_margin_predictor_seed_*"))
    print(f"run dirs found: {len(dirs)}")
    if len(dirs) != 46:
        flag("blocker", f"expected 46 run dirs, found {len(dirs)}")

    rows = []
    sample_checked = 0
    for d in dirs:
        rec: dict = {"dir": d.name}
        # --- presence ---
        missing = [f for f in ("trace.parquet", "stats.json",
                               "convergence.parquet", "context.json")
                   if not (d / f).exists()]
        rec["missing_files"] = ";".join(missing)
        if missing:
            flag("blocker", f"{d.name}: missing {missing}")
            rows.append(rec)
            continue

        stats = json.loads((d / "stats.json").read_text())
        ctx = json.loads((d / "context.json").read_text())
        tr = pd.read_parquet(d / "trace.parquet")
        cv = pd.read_parquet(d / "convergence.parquet")

        # --- identity ---
        rec["seed_idx"] = stats["seed_idx"]
        name_idx = int(d.name.split("_seed_")[1].split("_")[0])
        rec["name_idx_match"] = (name_idx == stats["seed_idx"])
        if not rec["name_idx_match"]:
            flag("material", f"{d.name}: dir index {name_idx} != stats seed_idx {stats['seed_idx']}")

        # --- budget / shape ---
        rec["generations_cfg"] = stats["generations"]
        rec["pop_size_cfg"] = stats["pop_size"]
        rec["n_rows"] = len(tr)
        gens = np.sort(tr["generation"].unique())
        rec["gen_min"] = int(gens.min())
        rec["gen_max"] = int(gens.max())
        rec["n_gens"] = len(gens)
        per_gen = tr.groupby("generation")["individual"].count()
        rec["per_gen_count_uniform_30"] = bool((per_gen == 30).all())
        rec["dup_keys"] = int(tr.duplicated(subset=["generation", "individual"]).sum())
        rec["conv_rows"] = len(cv)
        if stats["generations"] != 50 or stats["pop_size"] != 30:
            flag("blocker", f"{d.name}: budget {stats['generations']}x{stats['pop_size']} != 50x30")
        if len(tr) != 1500 or rec["gen_max"] != 49 or rec["n_gens"] != 50:
            flag("blocker", f"{d.name}: trace shape rows={len(tr)} gens={rec['n_gens']} max_gen={rec['gen_max']}")
        if not rec["per_gen_count_uniform_30"]:
            flag("blocker", f"{d.name}: per-generation individual count not uniformly 30")
        if rec["dup_keys"]:
            flag("blocker", f"{d.name}: {rec['dup_keys']} duplicate (generation, individual) keys")

        rec["runtime_s"] = stats["runtime_seconds"]
        rec["n_pareto"] = stats["n_pareto"]

        # --- backend ---
        rec["ctx_backend"] = ctx.get("image_backend")
        rec["ctx_candidate_strategy"] = ctx.get("image_candidate_strategy")
        rec["ctx_target_class"] = ctx.get("image_target_class")
        rec["stats_candidate_strategy"] = stats.get("image_candidate_strategy")
        rec["stats_n_candidates"] = stats.get("image_n_candidates")
        rec["image_dim"] = stats["image_dim"]
        rec["text_dim"] = stats["text_dim"]
        gb = np.asarray(stats["gene_bounds"])
        rec["len_gene_bounds"] = len(gb)
        img_b = gb[: stats["image_dim"]]
        txt_b = gb[stats["image_dim"]:]
        rec["img_bounds_max"] = int(img_b.max())
        rec["img_bounds_median"] = float(np.median(img_b))
        rec["img_bounds_n_ge_1000"] = int((img_b >= 1000).sum())
        rec["img_bounds_n_eq_16383"] = int((img_b == 16383).sum())
        rec["txt_bounds"] = ",".join(map(str, txt_b.tolist()))
        rec["len_ok"] = (len(gb) == stats["image_dim"] + stats["text_dim"])
        # context candidate lists: per-position candidate counts should equal
        # gene_bound - 1 on cone path (bound = n_candidates + 1, "keep origin")
        cand = ctx.get("image_candidates")
        if cand is not None:
            cand_lens = np.array([len(c) for c in cand])
            rec["ctx_cand_match_bounds"] = bool(
                np.array_equal(cand_lens + 1, img_b)
            )
            rec["ctx_cand_max"] = int(cand_lens.max())
        # genotype length check on one row
        g0 = tr.iloc[0]["genotype"]
        rec["genotype_len"] = len(g0)

        # --- column sanity ---
        for col in ("fitness_TgtBal", "p_class_a", "p_class_b"):
            rec[f"nan_{col}"] = int(tr[col].isna().sum())
            if rec[f"nan_{col}"]:
                flag("blocker", f"{d.name}: {rec[f'nan_{col}']} NaNs in {col}")
        psum = tr["p_class_a"] + tr["p_class_b"]
        rec["psum_max"] = float(psum.max())
        rec["n_psum_gt_1"] = int((psum > 1.0 + 1e-9).sum())
        if rec["n_psum_gt_1"]:
            flag("material", f"{d.name}: {rec['n_psum_gt_1']} rows with p_a+p_b > 1 (max {rec['psum_max']:.6f})")
        rec["tgtbal_min"] = float(tr["fitness_TgtBal"].min())
        rec["tgtbal_neg"] = int((tr["fitness_TgtBal"] < 0).sum())

        # TgtBal vs logprobs consistency on a sample of 100 rows / run
        smp = tr.sample(n=100, random_state=0)
        lp = np.stack(smp["logprobs"].to_numpy())
        diff = np.abs(lp[:, 0] - lp[:, 1])
        err = np.abs(diff - smp["fitness_TgtBal"].to_numpy())
        rec["tgtbal_lp_maxerr"] = float(err.max())
        if rec["tgtbal_lp_maxerr"] > 1e-6:
            flag("material", f"{d.name}: TgtBal vs |logprob diff| max err {rec['tgtbal_lp_maxerr']:.3e}")
        sample_checked += len(smp)

        # cache hits
        rec["cache_hit_rate_trace"] = float(tr["cache_hit"].mean())
        rec["cache_hits_stats"] = stats["cache_hits"]
        rec["cache_misses_stats"] = stats["cache_misses"]

        # quick outcome cols (cross-agent sanity, not analysis)
        g0_rows = tr[tr["generation"] == 0]
        rec["probe_median_gen0"] = float(g0_rows["fitness_TgtBal"].median())
        rec["min_tgtbal_at_50"] = float(tr["fitness_TgtBal"].min())
        rec["crossed_at_50"] = bool((tr["p_class_b"] > tr["p_class_a"]).any())
        rec["stuck"] = rec["min_tgtbal_at_50"] > 0.1

        # --- metadata for design check ---
        sm = stats["seed_metadata"]
        for k in ("level_anchor", "level_target", "anchor_class_concrete",
                  "target_class_concrete", "anchor_label_in_prompt",
                  "target_label_in_prompt", "common_ancestor_level",
                  "seed_idx_in_class"):
            rec[k] = sm.get(k)
        rec["class_a"] = stats["class_a"]
        rec["class_b"] = stats["class_b"]
        rec["model_id"] = stats["model_id"]
        rec["score_full_categories"] = stats["score_full_categories"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "qc_per_run.csv", index=False)

    print("\n========== AGGREGATE ==========")
    print(f"TgtBal-vs-logprob rows checked: {sample_checked}")

    # filter indices
    got = sorted(df["seed_idx"].tolist())
    if got != sorted(EXPECTED_FILTER_INDICES):
        flag("blocker", f"seed_idx set != config filter_indices; missing="
             f"{sorted(set(EXPECTED_FILTER_INDICES)-set(got))} extra={sorted(set(got)-set(EXPECTED_FILTER_INDICES))}")
    else:
        print("seed_idx set matches config filter_indices exactly (46/46).")

    # budget uniformity
    print(f"generations unique: {df['generations_cfg'].unique().tolist()}, "
          f"pop_size unique: {df['pop_size_cfg'].unique().tolist()}")
    print(f"all traces reach gen 49: {bool((df['gen_max'] == 49).all())}")
    print(f"conv_rows unique: {sorted(df['conv_rows'].unique().tolist())}")

    rt = df["runtime_s"]
    q1, med, q3 = rt.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    out_mask = (rt < q1 - 1.5 * iqr) | (rt > q3 + 1.5 * iqr)
    print(f"runtime_s: min={rt.min():.0f} q1={q1:.0f} med={med:.0f} "
          f"q3={q3:.0f} max={rt.max():.0f} (IQR outliers: {int(out_mask.sum())})")
    if out_mask.any():
        for _, r in df[out_mask].iterrows():
            print(f"  outlier: {r['dir']} runtime={r['runtime_s']:.0f}s")

    # backend
    print("\n---------- backend ----------")
    print("context.json image_backend:", df["ctx_backend"].value_counts().to_dict())
    print("context.json image_candidate_strategy:", df["ctx_candidate_strategy"].value_counts().to_dict())
    print("stats.json   image_candidate_strategy:", df["stats_candidate_strategy"].value_counts().to_dict())
    print("image_target_class set in all runs:", bool(df["ctx_target_class"].notna().all()))
    print("ctx target class == target_class_concrete:",
          bool((df["ctx_target_class"] == df["target_class_concrete"]).all()))
    print("image_dim values:", df["image_dim"].value_counts().to_dict())
    print("text_dim values:", df["text_dim"].value_counts().to_dict())
    print("genotype_len == image_dim+text_dim everywhere:",
          bool((df["genotype_len"] == df["image_dim"] + df["text_dim"]).all()),
          "| len(gene_bounds) ok:", bool(df["len_ok"].all()))
    print("ctx candidate lists +1 == image gene_bounds everywhere:",
          bool(df["ctx_cand_match_bounds"].all()))
    print(f"img gene_bounds: median-of-medians={df['img_bounds_median'].median():.0f}, "
          f"max over runs={df['img_bounds_max'].max()}, "
          f"runs with any bound==16383: {int((df['img_bounds_n_eq_16383']>0).sum())}, "
          f"n bounds>=1000 per run: min={df['img_bounds_n_ge_1000'].min()} "
          f"med={df['img_bounds_n_ge_1000'].median():.0f} max={df['img_bounds_n_ge_1000'].max()}")
    print("text bounds patterns:", df["txt_bounds"].value_counts().to_dict())

    if set(df["ctx_candidate_strategy"].unique()) == {"cone_filter"}:
        print("VERDICT: cone_filter ran in ALL 46 runs (context.json is the "
              "runtime record); stats.json 'KNN' is a config-default echo "
              "(vlm_boundary_tester.py:213 writes config.image.candidate_strategy.name, "
              "which cone mode does not override).")
    else:
        flag("blocker", "mixed/non-cone candidate strategies in context.json: "
             + str(df["ctx_candidate_strategy"].value_counts().to_dict()))

    # model / scoring uniformity
    print("\nmodel_id:", df["model_id"].value_counts().to_dict())
    print("score_full_categories:", df["score_full_categories"].value_counts().to_dict())

    # cache
    print(f"\ncache_hit rate (trace): min={df['cache_hit_rate_trace'].min():.3f} "
          f"med={df['cache_hit_rate_trace'].median():.3f} max={df['cache_hit_rate_trace'].max():.3f}")
    print(f"stats cache hits+misses unique: "
          f"{sorted((df['cache_hits_stats']+df['cache_misses_stats']).unique().tolist())}")

    # ---------------- cell design ----------------
    print("\n---------- cell design ----------")
    df["cell"] = list(zip(df["anchor_class_concrete"], df["target_class_concrete"],
                          df["level_anchor"], df["level_target"]))
    df["within_bucket"] = df["common_ancestor_level"] == 2
    junco_involved = (df["anchor_class_concrete"] == "junco") | (df["target_class_concrete"] == "junco")
    df["stratum"] = np.where(df["within_bucket"], "within_bucket",
                     np.where(junco_involved, "wall_repl", "cross_breadth"))

    cells = df.groupby("cell").agg(
        n=("dir", "size"),
        stratum=("stratum", "first"),
        n_strata=("stratum", "nunique"),
    ).reset_index()
    print(f"distinct cells: {len(cells)} (expect 40)")
    n2 = cells[cells["n"] == 2]
    print(f"cells with n=2: {len(n2)} (expect 6)")
    for _, r in n2.iterrows():
        print("  n=2 cell:", r["cell"], "| stratum:", r["stratum"])
    if (cells["n_strata"] > 1).any():
        flag("material", "some cell maps to >1 stratum (metadata inconsistency)")
    bad_n = cells[~cells["n"].isin([1, 2])]
    if len(bad_n):
        flag("blocker", f"cells with n not in {{1,2}}: {bad_n['cell'].tolist()}")

    strata_counts = cells.groupby("stratum")["cell"].count().to_dict()
    print("strata cell counts:", strata_counts, "(expect within_bucket=18, wall_repl=12, cross_breadth=10)")
    if strata_counts != {"cross_breadth": 10, "wall_repl": 12, "within_bucket": 18}:
        flag("material", f"strata counts deviate from spec: {strata_counts}")

    anchors = sorted(df["anchor_class_concrete"].unique())
    print("anchor classes present:", anchors)
    if len(anchors) != 6:
        flag("material", f"expected 6 anchor classes, got {anchors}")
    nonjunco_cells = cells[[c[0] != "junco" for c in cells["cell"]]]
    print(f"non-junco-anchored cells: {len(nonjunco_cells)} (expect 31)")
    if len(nonjunco_cells) != 31:
        flag("material", f"non-junco cell count {len(nonjunco_cells)} != 31")

    # directed-pair table
    pair_tbl = df.groupby(["anchor_class_concrete", "target_class_concrete",
                           "level_anchor", "level_target", "stratum"]).size()
    pair_tbl.rename("n_runs").to_csv(OUT / "qc_cell_design.csv")

    # check expected stratum-2 / stratum-3 pair sets from the config comment
    s2_pairs = set(map(tuple, df[df["stratum"] == "wall_repl"][["anchor_class_concrete", "target_class_concrete"]].itertuples(index=False)))
    s3_pairs = set(map(tuple, df[df["stratum"] == "cross_breadth"][["anchor_class_concrete", "target_class_concrete"]].itertuples(index=False)))
    print("wall_repl directed pairs:", sorted(s2_pairs))
    print("cross_breadth directed pairs:", sorted(s3_pairs))
    exp_s2 = {("junco", "boa constrictor"), ("boa constrictor", "junco"),
              ("junco", "cello"), ("cello", "junco")}
    exp_s3 = {("ostrich", "green iguana"), ("green iguana", "cello"),
              ("marimba", "boa constrictor"), ("cello", "ostrich"),
              ("boa constrictor", "marimba")}
    if s2_pairs != exp_s2:
        flag("material", f"wall_repl pairs deviate: {s2_pairs ^ exp_s2}")
    if s3_pairs != exp_s3:
        flag("material", f"cross_breadth pairs deviate: {s3_pairs ^ exp_s3}")

    # within-bucket: 3 pairs x both directions x cells {(0,0),(0,1),(1,1)}
    wb = df[df["stratum"] == "within_bucket"]
    wb_cells = set(map(tuple, wb[["anchor_class_concrete", "target_class_concrete",
                                  "level_anchor", "level_target"]].itertuples(index=False)))
    exp_wb = set()
    for a, b in [("junco", "ostrich"), ("green iguana", "boa constrictor"), ("cello", "marimba")]:
        for x, y in [(a, b), (b, a)]:
            for la, lt in [(0, 0), (0, 1), (1, 1)]:
                exp_wb.add((x, y, la, lt))
    if wb_cells != exp_wb:
        flag("material", f"within-bucket cells deviate: missing={exp_wb - wb_cells} extra={wb_cells - exp_wb}")
    else:
        print("within-bucket cells match the 3-pairs x both-dirs x {(0,0),(0,1),(1,1)} grid exactly.")

    # n=2 cells expected set
    exp_n2 = {("junco", "ostrich", 0, 0), ("green iguana", "boa constrictor", 1, 1),
              ("cello", "marimba", 0, 0), ("junco", "boa constrictor", 0, 1),
              ("boa constrictor", "junco", 0, 1), ("green iguana", "cello", 0, 0)}
    got_n2 = set(n2["cell"])
    if got_n2 != exp_n2:
        flag("material", f"n=2 cells deviate: missing={exp_n2 - got_n2} extra={got_n2 - exp_n2}")
    else:
        print("n=2 cells match the pre-registered set exactly.")

    # quick descriptive outcome stats (for cross-agent agreement, not analysis)
    print("\n---------- outcome snapshot ----------")
    print(f"runs crossed_at_50: {int(df['crossed_at_50'].sum())}/46; "
          f"stuck (min TgtBal@50 > 0.1): {int(df['stuck'].sum())}/46")
    print(f"probe (gen-0 median TgtBal): min={df['probe_median_gen0'].min():.3f} "
          f"med={df['probe_median_gen0'].median():.3f} max={df['probe_median_gen0'].max():.3f}")
    print(f"negative TgtBal rows anywhere: {int(df['tgtbal_neg'].sum())}")
    print(f"max p_a+p_b over all runs: {df['psum_max'].max():.6f}")

    print("\n========== ISSUES ==========")
    if not issues:
        print("none")
    for sev, msg in issues:
        print(f"[{sev.upper()}] {msg}")

    # exit non-zero on blocker so the wrapper notices
    sys.exit(2 if any(s == "blocker" for s, _ in issues) else 0)


if __name__ == "__main__":
    main()
