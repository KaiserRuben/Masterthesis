#!/usr/bin/env python3
"""HS-01 stimulus candidate extraction: boundary-converged individuals.

Scans evolutionary run artifacts for individuals with fitness_TgtBal <= 1e-2
(the HS-01 boundary criterion) and extracts them — with genotype, prompt
text, objectives, drift metrics and full cell/pair provenance — into compact
per-experiment parquet files under experiments/HS-01/data_raw/.

Sources
-------
LLaVA (OpenVINO INT8):  runs/Exp-100 (poc_boundary_pair + smoke_boundary_pair),
                        runs/Exp-101, runs/Exp-102 (if synced).
Qwen  (torch/MPS):      runs/Exp-101q (if synced); fallback: the cone-enabled
                        Exp-27 pairA runs (cone05/10/20/40). Non-cone Qwen
                        runs (Exp-26 qwen baseline, Exp-27 baseline/stylegan)
                        are excluded by design and recorded as such.

The script is re-runnable: when Exp-101q / Exp-102 finish syncing, run it
again from the repo root and the new runs are picked up automatically.

Usage (from repo root):
    python experiments/HS-01/data_raw/extract_boundary_samples.py

Read-only with respect to runs/ and configs/; writes only into
experiments/HS-01/data_raw/.
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

THRESH = 1e-2          # hard boundary criterion
THRESH_TIGHT = 1e-3    # flagged subset

# Incremental mode: per-run results are cached under <out>/_cache/ so the
# script can be re-run (or resumed after interruption) cheaply. Set
# HS01_BUDGET_S to bound one invocation's processing time; on exhaustion it
# exits with code 3 ("PARTIAL") and the next invocation continues.
BUDGET_S = float(os.environ.get("HS01_BUDGET_S", "1e9"))
_T0 = time.time()


def out_of_time():
    return time.time() - _T0 > BUDGET_S


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def cached_run(cache_path, fn):
    """Return (rows, report) from cache, computing+caching via fn() if absent.
    Returns None if out of time budget."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            j = json.load(f)
        return j["rows"], j["report"]
    if out_of_time():
        return None
    rows, rep = fn()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"rows": rows, "report": rep}, f, default=_json_default)
    return rows, rep

TRACE_COLS = ["seed_id", "generation", "individual", "genotype", "logprobs",
              "decoded_text", "cache_hit", "predicted_class",
              "p_class_a", "p_class_b"]


# ───────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────

def read_trace(tp):
    """Return (df, fitness_cols) or (None, reason) if unreadable."""
    try:
        f = pq.ParquetFile(tp)
    except Exception as e:
        return None, f"unreadable trace ({type(e).__name__}: {str(e)[:60]})"
    fit_cols = [c.name for c in f.schema_arrow if c.name.startswith("fitness_")]
    cols = [c for c in TRACE_COLS if c in [x.name for x in f.schema_arrow]] + fit_cols
    df = f.read(columns=cols).to_pandas()
    return df, fit_cols


def load_pareto_index(run_dir):
    """genotype-tuple -> (pareto_idx, text, full_prompt, fitness)."""
    idx = {}
    for p in glob.glob(os.path.join(run_dir, "pareto_*.json")):
        i = int(os.path.basename(p)[len("pareto_"):-len(".json")])
        with open(p) as f:
            j = json.load(f)
        idx[tuple(j["genotype"])] = (i, j.get("text"), j.get("full_prompt"),
                                     j.get("fitness"))
    return idx


def gzip_copy(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)


def plain_copy(src, dst):
    if not os.path.exists(src) or os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def drift_class(n_img, n_txt):
    if n_img > 0 and n_txt > 0:
        return "mixed"
    if n_img > 0:
        return "image_only_drift"
    if n_txt > 0:
        return "text_only_drift"
    return "no_drift"


def extract_run(run_dir, evo_dir, stats, meta_common, keep_all=True,
                pareto_only=False):
    """Extract qualifying individuals from one run.

    Returns (rows, run_report). evo_dir holds trace.parquet + pareto_*.json.
    pareto_only: salvage mode for runs whose trace.parquet has no footer.
    """
    img_dim = stats.get("image_dim")
    txt_dim = stats.get("text_dim", 0)
    pareto = load_pareto_index(evo_dir)
    report = {"n_pareto_json": len(pareto)}
    rows = []

    if pareto_only:
        recs = []
        for g, (pidx, text, fp, fit) in pareto.items():
            recs.append(dict(generation=-1, individual=-1, genotype=list(g),
                             decoded_text=text, _pareto=(pidx, text, fp, fit),
                             _fit=fit))
        df = pd.DataFrame(recs)
        if df.empty:
            report.update(n_trace_rows=0, n_q1e2_unique=0, n_q1e3_unique=0,
                          min_tgtbal=None, source="pareto_json")
            return rows, report
        df["fitness_TgtBal"] = [f[-1] for f in df["_fit"]]
        df["fitness_MatrixDistance_fro"] = [f[0] for f in df["_fit"]]
        df["fitness_TextDist"] = [f[1] if len(f) == 3 else None for f in df["_fit"]]
        df["logprobs"] = None
        df["cache_hit"] = None
        df["predicted_class"] = None
        df["p_class_a"] = np.nan
        df["p_class_b"] = np.nan
        report["source"] = "pareto_json (trace.parquet footer missing)"
        trace_n = len(df)
    else:
        df, fit_cols = read_trace(os.path.join(evo_dir, "trace.parquet"))
        if df is None:
            report.update(error=fit_cols)
            return None, report
        report["source"] = "trace.parquet"
        trace_n = len(df)
        if "fitness_TextDist" not in df.columns:
            df["fitness_TextDist"] = np.nan

    report["n_trace_rows"] = trace_n
    report["min_tgtbal"] = float(df["fitness_TgtBal"].min())

    q = df[df["fitness_TgtBal"] <= THRESH].copy()
    if not keep_all and len(q) > 3:
        q = q.nsmallest(3, "fitness_TgtBal")

    # dedupe identical genotypes (re-evaluations / elites / cache hits)
    q["_gkey"] = q["genotype"].map(tuple)
    occ = q.groupby("_gkey").size()
    q = q.sort_values(["fitness_TgtBal", "generation"] if "generation" in q
                      else ["fitness_TgtBal"])
    q = q.drop_duplicates("_gkey", keep="first")

    for _, r in q.iterrows():
        g = r["_gkey"]
        garr = np.asarray(g)
        if img_dim is not None and txt_dim:
            n_img = int((garr[:img_dim] != 0).sum())
            n_txt = int((garr[img_dim:] != 0).sum())
        elif img_dim is not None:
            n_img = int((garr[:img_dim] != 0).sum())
            n_txt = 0
        else:
            n_img, n_txt = int((garr != 0).sum()), 0
        pidx, ptext, pfull, _ = pareto.get(g, (None, None, None, None))
        rows.append(dict(
            meta_common,
            generation=int(r["generation"]),
            individual=int(r["individual"]),
            tgtbal=float(r["fitness_TgtBal"]),
            q_le_1e3=bool(r["fitness_TgtBal"] <= THRESH_TIGHT),
            d_img_matrix=float(r["fitness_MatrixDistance_fro"]),
            d_text_embed=(None if pd.isna(r["fitness_TextDist"])
                          else float(r["fitness_TextDist"])),
            p_class_a=(None if pd.isna(r["p_class_a"]) else float(r["p_class_a"])),
            p_class_b=(None if pd.isna(r["p_class_b"]) else float(r["p_class_b"])),
            logprobs=(list(r["logprobs"]) if r["logprobs"] is not None else None),
            predicted_class=r["predicted_class"],
            cache_hit=(None if r["cache_hit"] is None else bool(r["cache_hit"])),
            decoded_text=r["decoded_text"],
            pareto_text=ptext,
            full_prompt=pfull,
            in_final_pareto=pidx is not None,
            pareto_idx=pidx,
            pareto_png=(os.path.join(meta_common["run_rel"],
                                     meta_common.get("evo_subdir", ""),
                                     f"pareto_{pidx}.png").replace("//", "/")
                        if pidx is not None else None),
            genotype=list(g),
            n_occurrences_in_trace=int(occ[g]),
            image_dim=img_dim, text_dim=txt_dim,
            n_active_img_genes=n_img, n_active_text_genes=n_txt,
            frac_active_img=(n_img / img_dim if img_dim else None),
            drift_class=drift_class(n_img, n_txt),
        ))
    report["n_q1e2_unique"] = len(rows)
    report["n_q1e3_unique"] = sum(r["q_le_1e3"] for r in rows)
    return rows, report


def stats_meta(stats):
    sm = stats.get("seed_metadata", {}) or {}
    return dict(
        seed_idx=stats.get("seed_idx"),
        class_a=stats.get("class_a"), class_b=stats.get("class_b"),
        prompt_template=stats.get("prompt_template"),
        answer_format=stats.get("answer_format"),
        sut_model_id=stats.get("model_id"),
        anchor_class_concrete=sm.get("anchor_class_concrete"),
        target_class_concrete=sm.get("target_class_concrete"),
        anchor_label_in_prompt=sm.get("anchor_label_in_prompt"),
        target_label_in_prompt=sm.get("target_label_in_prompt"),
        level_anchor=sm.get("level_anchor"), level_target=sm.get("level_target"),
        common_ancestor_level=sm.get("common_ancestor_level"),
        seed_idx_in_class=sm.get("seed_idx_in_class"),
        anchor_position=sm.get("anchor_position"),
        target_position=sm.get("target_position"),
    )


def write_parquet(rows, path):
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)


# ───────────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="experiments/HS-01/data_raw")
    args = ap.parse_args()
    root, out = os.path.abspath(args.repo_root), None
    out = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    runs = os.path.join(root, "runs")
    summary = {"threshold": THRESH, "threshold_tight": THRESH_TIGHT,
               "experiments": {}}
    n_pending = 0

    # ── LLaVA / Exp-100 (nested layout: <group>/<seed_dir>/evolutionary) ──
    for group in ["poc_boundary_pair", "smoke_boundary_pair"]:
        exp = "Exp-100"
        gdir = os.path.join(runs, exp, group)
        if not os.path.isdir(gdir):
            continue
        all_rows, reports = [], {}
        for sd in sorted(os.listdir(gdir)):
            run_dir = os.path.join(gdir, sd)
            evo = os.path.join(run_dir, "evolutionary")
            sp = os.path.join(evo, "stats.json")
            if not os.path.isdir(evo) or not os.path.exists(sp):
                reports[sd] = {"error": "missing evolutionary/stats.json"}
                continue
            with open(sp) as f:
                stats = json.load(f)
            meta = dict(
                experiment=exp, sut="llava", run_group=group, run_dir=sd,
                run_rel=f"runs/{exp}/{group}/{sd}", evo_subdir="evolutionary",
                modality="joint", sut_backend="openvino",
                image_backend="vqgan_codebook", cone_enabled=True,
                cone_alpha_deg=20.0, **stats_meta(stats))

            def work(run_dir=run_dir, evo=evo, stats=stats, meta=meta, sp=sp,
                     dst=os.path.join(out, "llava", exp, group, sd)):
                rows, rep = extract_run(run_dir, evo, stats, meta)
                if rows:  # provenance copies only for yielding runs
                    plain_copy(os.path.join(run_dir, "config.json"),
                               os.path.join(dst, "config.json"))
                    plain_copy(os.path.join(run_dir, "manifest.json"),
                               os.path.join(dst, "manifest.json"))
                    plain_copy(sp, os.path.join(dst, "evolutionary", "stats.json"))
                    gzip_copy(os.path.join(evo, "context.json"),
                              os.path.join(dst, "evolutionary", "context.json.gz"))
                return rows, rep

            res = cached_run(os.path.join(out, "_cache",
                                          f"{exp}__{group}__{sd}.json"), work)
            if res is None:
                n_pending += 1
                continue
            rows, rep = res
            reports[sd] = rep
            if rows:
                all_rows.extend(rows)

        # enrich Exp-100 poc rows with the tidy aggregate (cell_kind etc.)
        if group == "poc_boundary_pair" and all_rows:
            aggp = os.path.join(root, "experiments/analysis/output/"
                                "exp100_poc_aggregate.parquet")
            if os.path.exists(aggp):
                agg = pd.read_parquet(aggp)[
                    ["seed_dir", "bucket_anchor", "bucket_target", "cell_kind",
                     "cross_subkind", "is_diagonal", "is_forward"]]
                dfr = pd.DataFrame(all_rows).merge(
                    agg, left_on="run_dir", right_on="seed_dir", how="left")
                dfr = dfr.drop(columns=["seed_dir"])
                all_rows = dfr.to_dict("records")

        n = write_parquet(all_rows, os.path.join(
            out, "llava", exp, f"boundary_individuals_{group}.parquet"))
        summary["experiments"][f"{exp}/{group}"] = {
            "runs_found": len(os.listdir(gdir)),
            "n_extracted": n,
            "runs": reports,
        }

    # ── LLaVA / Exp-101 + Exp-102 + HS-GEN-01, Qwen / Exp-101q (flat) ─────
    # 4th tuple element = name_filter: when set, only run dirs whose name
    # contains it are scanned. HS-GEN-01 holds the 1024-run gap_filter SCREEN
    # (4-gen, no stimuli) alongside the 6 promoted full runs; only the promoted
    # runs are study stimulus sources. HS-GEN-01 is modality=image_only
    # (text_dim=0) → every qualifier is strictly image_only_drift.
    flat = [("Exp-101", "llava", dict(modality="joint", sut_backend="openvino",
                                      cone_alpha_deg=20.0), None),
            ("Exp-102", "llava", dict(modality="joint", sut_backend="openvino",
                                      cone_alpha_deg=20.0), None),
            ("Exp-101q", "qwen", dict(modality="joint", sut_backend="torch",
                                      cone_alpha_deg=20.0), None),
            ("HS-GEN-01", "llava", dict(modality="image_only",
                                        sut_backend="openvino",
                                        cone_alpha_deg=20.0), "promoted"),
            ("HS-GEN-02", "llava", dict(modality="joint",
                                        sut_backend="openvino",
                                        cone_alpha_deg=20.0), None),
            ("HS-GEN-03", "llava", dict(modality="joint",
                                        sut_backend="openvino",
                                        cone_alpha_deg=20.0), None)]
    for exp, sut, extra, name_filter in flat:
        edir = os.path.join(runs, exp)
        if not os.path.isdir(edir):
            summary["experiments"][exp] = {"runs_found": 0,
                                           "note": "directory missing"}
            continue
        all_rows, reports = [], {}
        for sd in sorted(os.listdir(edir)):
            run_dir = os.path.join(edir, sd)
            if not os.path.isdir(run_dir):
                continue
            if name_filter and name_filter not in sd:
                continue
            sp = os.path.join(run_dir, "stats.json")
            if not os.path.exists(sp):
                reports[sd] = {"error": "incomplete run (no stats.json; "
                                        "likely still running/syncing)"}
                continue
            with open(sp) as f:
                stats = json.load(f)
            meta = dict(
                experiment=exp, sut=sut, run_group=exp, run_dir=sd,
                run_rel=f"runs/{exp}/{sd}", evo_subdir="",
                image_backend="vqgan_codebook", cone_enabled=True,
                **extra, **stats_meta(stats))

            def work(run_dir=run_dir, stats=stats, meta=meta, sp=sp,
                     dst=os.path.join(out, sut, exp, sd)):
                rows, rep = extract_run(run_dir, run_dir, stats, meta)
                if rows:
                    plain_copy(sp, os.path.join(dst, "stats.json"))
                    gzip_copy(os.path.join(run_dir, "context.json"),
                              os.path.join(dst, "context.json.gz"))
                return rows, rep

            res = cached_run(os.path.join(out, "_cache",
                                          f"{exp}__{sd}.json"), work)
            if res is None:
                n_pending += 1
                continue
            rows, rep = res
            reports[sd] = rep
            if rows:
                all_rows.extend(rows)
        n = write_parquet(all_rows, os.path.join(
            out, sut, exp, "boundary_individuals.parquet"))
        summary["experiments"][exp] = {
            "runs_found": len([d for d in os.listdir(edir)
                               if os.path.isdir(os.path.join(edir, d))
                               and (not name_filter or name_filter in d)]),
            "n_extracted": n, "runs": reports}

    # ── Qwen / Exp-27 cone-enabled fallback ───────────────────────────────
    exp = "Exp-27"
    edir = os.path.join(runs, exp)
    cone_alpha = {"cone05": 5.0, "cone10": 10.0, "cone20": 20.0, "cone40": 40.0}
    all_rows, ref_rows, reports, excluded = [], [], {}, {}
    for sd in sorted(os.listdir(edir)) if os.path.isdir(edir) else []:
        run_dir = os.path.join(edir, sd)
        if not os.path.isdir(run_dir):
            continue
        tag = next((t for t in cone_alpha if t in sd), None)
        if "qwen" not in sd:
            excluded[sd] = "not a Qwen run"
            continue
        if tag is None:
            excluded[sd] = ("non-cone image backend (baseline/stylegan) — "
                            "excluded per HS-01 Qwen cone requirement")
            continue
        sp = os.path.join(run_dir, "stats.json")
        with open(sp) as f:
            stats = json.load(f)
        meta = dict(
            experiment=exp, sut="qwen", run_group=exp, run_dir=sd,
            run_rel=f"runs/{exp}/{sd}", evo_subdir="",
            modality="image_only", sut_backend="torch",
            image_backend="vqgan_codebook", cone_enabled=True,
            cone_alpha_deg=cone_alpha[tag], **stats_meta(stats))

        def work(run_dir=run_dir, stats=stats, meta=meta, sp=sp,
                 dst=os.path.join(out, "qwen", exp, sd)):
            # cone20's trace.parquet lacks a footer -> salvage via pareto JSONs
            pareto_only = False
            try:
                pq.ParquetFile(os.path.join(run_dir, "trace.parquet"))
            except Exception:
                pareto_only = True
            rows, rep = extract_run(run_dir, run_dir, stats, meta,
                                    pareto_only=pareto_only)
            refs = []
            # reference rows: best 3 by TgtBal even if non-qualifying, so the
            # manifest can document how far Qwen runs sit from the boundary.
            if rows is not None and not rows:
                if pareto_only:
                    pidx = load_pareto_index(run_dir)
                    fits = sorted(((v[3][-1], k, v) for k, v in pidx.items()))[:3]
                    for tb, g, (pi, text, fp, fit) in fits:
                        garr = np.asarray(g)
                        refs.append(dict(
                            meta, generation=-1, individual=-1, tgtbal=float(tb),
                            qualifies=False, d_img_matrix=float(fit[0]),
                            decoded_text=text, full_prompt=fp, pareto_idx=pi,
                            genotype=list(g),
                            n_active_img_genes=int((garr[:stats["image_dim"]] != 0).sum()),
                            n_active_text_genes=int((garr[stats["image_dim"]:] != 0).sum())))
                else:
                    df, _ = read_trace(os.path.join(run_dir, "trace.parquet"))
                    b = df.nsmallest(3, "fitness_TgtBal")
                    pidx = load_pareto_index(run_dir)
                    for _, r in b.iterrows():
                        g = tuple(r["genotype"])
                        garr = np.asarray(g)
                        pi = pidx.get(g, (None,))[0]
                        refs.append(dict(
                            meta, generation=int(r["generation"]),
                            individual=int(r["individual"]),
                            tgtbal=float(r["fitness_TgtBal"]), qualifies=False,
                            d_img_matrix=float(r["fitness_MatrixDistance_fro"]),
                            decoded_text=r["decoded_text"],
                            full_prompt=pidx.get(g, (None, None, None))[2],
                            pareto_idx=pi, genotype=list(g),
                            n_active_img_genes=int((garr[:stats["image_dim"]] != 0).sum()),
                            n_active_text_genes=int((garr[stats["image_dim"]:] != 0).sum())))
            # provenance for all scanned cone runs (small files)
            plain_copy(sp, os.path.join(dst, "stats.json"))
            plain_copy(os.path.join(run_dir, "context.json"),
                       os.path.join(dst, "context.json"))
            rep["_ref_rows"] = refs
            return rows, rep

        res = cached_run(os.path.join(out, "_cache", f"{exp}__{sd}.json"), work)
        if res is None:
            n_pending += 1
            continue
        rows, rep = res
        ref_rows.extend(rep.pop("_ref_rows", []) or [])
        reports[sd] = rep
        if rows:
            all_rows.extend(rows)
    write_parquet(all_rows, os.path.join(out, "qwen", exp,
                                         "boundary_individuals.parquet"))
    write_parquet(ref_rows, os.path.join(
        out, "qwen", exp, "NONQUALIFYING_best3_reference.parquet"))
    summary["experiments"][exp] = {
        "runs_found": len([d for d in (os.listdir(edir) if os.path.isdir(edir)
                                       else []) if os.path.isdir(os.path.join(edir, d))]),
        "n_extracted": len(all_rows), "n_reference_nonqualifying": len(ref_rows),
        "runs": reports, "excluded": excluded}

    # Exp-26 Qwen: baseline only -> excluded, recorded for completeness
    e26 = os.path.join(runs, "Exp-26")
    summary["experiments"]["Exp-26"] = {
        "qwen_runs": {d: "excluded: non-cone image backend (vqgan baseline)"
                      for d in (sorted(os.listdir(e26)) if os.path.isdir(e26)
                                else []) if "qwen" in d},
        "note": "Only Qwen run in Exp-26 is the non-cone vqgan baseline; the "
                "cone-enabled config qwen_mps_vqgan_cone.yaml was never run "
                "(no matching run dir)."}

    if n_pending:
        print(f"PARTIAL: {n_pending} runs still pending (time budget "
              f"HS01_BUDGET_S={BUDGET_S}s exhausted); re-run to continue.")
        sys.exit(3)

    # ── experiment-level configs (verbatim copies) ────────────────────────
    cfg_paths = (glob.glob(os.path.join(root, "configs/Exp-100/*.yaml"))
                 + glob.glob(os.path.join(root, "configs/Exp-101/*"))
                 + glob.glob(os.path.join(root, "configs/Exp-101q/*.yaml"))
                 + glob.glob(os.path.join(root, "configs/Exp-102/*"))
                 + glob.glob(os.path.join(root, "configs/HS-GEN-01/*.yaml"))
                 + glob.glob(os.path.join(root, "configs/HS-GEN-02/*.yaml"))
                 + glob.glob(os.path.join(root, "configs/HS-GEN-03/*.yaml"))
                 + glob.glob(os.path.join(root, "configs/Exp-26/qwen_*.yaml"))
                 + glob.glob(os.path.join(root, "configs/Exp-27/qwen_*.yaml")))
    for p in cfg_paths:
        rel = os.path.relpath(p, os.path.join(root, "configs"))
        plain_copy(p, os.path.join(out, "configs", rel))

    # ── summary json ──────────────────────────────────────────────────────
    total = 0
    for dp, _, fns in os.walk(out):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    summary["total_bytes_in_data_raw"] = total
    with open(os.path.join(out, "extract_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                      for k, v in summary["experiments"].items()}, indent=1,
                     default=str))
    print(f"total bytes in data_raw: {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
