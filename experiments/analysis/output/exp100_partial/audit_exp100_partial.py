"""Exp-100 poc_boundary_pair partial-run health audit (read-only on runs/).

Writes per-seed CSV + summary JSON to experiments/analysis/output/exp100_partial/.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/kaiser/Projects/Masterarbeit")
RUN = ROOT / "runs/Exp-100/poc_boundary_pair"
OUT = ROOT / "experiments/analysis/output/exp100_partial"
TARGET_SEEDS = 720
EXPECTED_TRACE = 6000
EXPECTED_CONV = 200

PDQ_PARQUETS = [
    "archive", "candidates", "stage1_flips", "stage2_trajectories", "sut_calls",
]

rows = []
config_hashes = {}
anchor_geno_stats = []
convergence_plateaus = []

seed_dirs = sorted(RUN.glob("seed_*"))
print(f"seed dirs found: {len(seed_dirs)}")

for sd in seed_dirs:
    name = sd.name
    seed_idx = int(name.split("_")[1])
    dir_ts = int(name.split("_")[2])
    evo = sd / "evolutionary"
    pdq = sd / "pdq"
    r: dict = {"seed": name, "seed_idx": seed_idx, "dir_ts": dir_ts}

    # ---- presence ---------------------------------------------------
    r["has_config"] = (sd / "config.json").exists()
    r["has_manifest"] = (sd / "manifest.json").exists()
    r["has_stats"] = (evo / "stats.json").exists()
    r["has_context"] = (evo / "context.json").exists()
    r["has_trace"] = (evo / "trace.parquet").exists()
    r["has_conv"] = (evo / "convergence.parquet").exists()
    r["n_pareto_json"] = len(list(evo.glob("pareto_*.json")))
    r["n_pareto_png"] = len(list(evo.glob("pareto_*.png")))
    for f in PDQ_PARQUETS:
        r[f"has_pdq_{f}"] = (pdq / f"{f}.parquet").exists()
    r["has_pdq_anchors_dir"] = (pdq / "anchors").is_dir()
    r["has_pdq_flips_dir"] = (pdq / "flips").is_dir()

    # ---- config hash ------------------------------------------------
    if r["has_config"]:
        cfg = json.loads((sd / "config.json").read_text())
        h = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:16]
        r["config_hash"] = h
        config_hashes.setdefault(h, []).append(name)
        r["config_schema_version"] = cfg.get("schema_version")

    # ---- row counts ---------------------------------------------------
    def nrows(p: Path):
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(p).metadata.num_rows
        except Exception as e:
            return f"ERR:{type(e).__name__}"

    r["trace_rows"] = nrows(evo / "trace.parquet") if r["has_trace"] else None
    r["conv_rows"] = nrows(evo / "convergence.parquet") if r["has_conv"] else None
    for f in PDQ_PARQUETS:
        r[f"pdq_{f}_rows"] = nrows(pdq / f"{f}.parquet") if r[f"has_pdq_{f}"] else None

    # ---- stats.json ---------------------------------------------------
    if r["has_stats"]:
        st = json.loads((evo / "stats.json").read_text())
        r["stats_schema_version"] = st.get("schema_version")
        r["evo_runtime_s"] = st.get("runtime_seconds")
        r["generations"] = st.get("generations")
        r["n_pareto"] = st.get("n_pareto")
        r["cache_hits"] = st.get("cache_hits")
        r["cache_misses"] = st.get("cache_misses")
        r["stats_has_early_stop_key"] = "early_stop" in st
        r["early_stop_trigger"] = (st.get("early_stop") or {}).get("trigger")
        r["class_a"] = st.get("class_a")
        r["class_b"] = st.get("class_b")
        sm = st.get("seed_metadata") or {}
        r["anchor_class_concrete"] = sm.get("anchor_class_concrete")
        r["target_class_concrete"] = sm.get("target_class_concrete")
        r["stats_mtime"] = (evo / "stats.json").stat().st_mtime
    if r["has_manifest"]:
        r["manifest_mtime"] = (sd / "manifest.json").stat().st_mtime
        man = json.loads((sd / "manifest.json").read_text())
        r["manifest_schema_version"] = man.get("schema_version")
        anchors = man.get("anchors", [])
        r["n_anchors"] = len(anchors)
        pa_pb = [(round(a["p_a"], 12), round(a["p_b"], 12)) for a in anchors]
        r["n_distinct_anchor_pa_pb"] = len(set(pa_pb))
        r["anchor_labels"] = "|".join(a["anchor_label"] for a in anchors)
        r["n_stage2_sut_calls_total"] = sum(a["n_stage2_sut_calls"] for a in anchors)
        r["n_stage1_flips_total"] = sum(a["n_stage1_flips"] for a in anchors)
        r["n_stage2_flips_total"] = sum(a["n_stage2_flips"] for a in anchors)

        # anchor genotype duplication from pdq/anchors/anchor_*.json
        genos = []
        fits = []
        for aj in sorted((pdq / "anchors").glob("anchor_*.json")):
            d = json.loads(aj.read_text())
            genos.append(tuple(d["genotype"]))
            fits.append(tuple(d["fitness"]))
        if genos:
            anchor_geno_stats.append({
                "seed": name,
                "n_anchor_json": len(genos),
                "n_distinct_genotypes": len(set(genos)),
                "n_distinct_fitness": len(set(fits)),
                "n_distinct_pa_pb": len(set(pa_pb)),
            })

    # ---- PDQ sut_calls cache + wall time ------------------------------
    if r["has_pdq_sut_calls"] and isinstance(r["pdq_sut_calls_rows"], int) and r["pdq_sut_calls_rows"] > 0:
        try:
            sc = pd.read_parquet(
                pdq / "sut_calls.parquet",
                columns=["cache_hit", "wall_time_s", "stage"],
            )
            r["pdq_calls"] = len(sc)
            r["pdq_cache_hit_rate"] = float(sc["cache_hit"].mean())
            r["pdq_sut_wall_s"] = float(sc["wall_time_s"].sum())
        except Exception as e:
            r["pdq_calls"] = f"ERR:{type(e).__name__}"

    # ---- convergence plateau audit -------------------------------------
    if r["has_conv"] and isinstance(r["conv_rows"], int) and r["conv_rows"] > 0:
        try:
            cv = pd.read_parquet(evo / "convergence.parquet")
            col = "pareto_min_TgtBal"
            if col in cv.columns:
                v = cv[col].to_numpy()
                r["final_min_TgtBal"] = float(v[-1])
                r["gen0_min_TgtBal"] = float(v[0])
                # longest streak of "no strict improvement of running best"
                best = np.inf
                streak = 0
                longest = 0
                last_improve_gen = -1
                for g, x in enumerate(v):
                    if x < best:
                        best = x
                        streak = 0
                        last_improve_gen = g
                    else:
                        streak += 1
                        longest = max(longest, streak)
                r["tgtbal_longest_plateau"] = longest
                r["tgtbal_tail_plateau"] = streak  # gens since last improvement
                r["tgtbal_last_improve_gen"] = last_improve_gen
                convergence_plateaus.append(
                    (name, longest, streak, last_improve_gen, float(v[-1]))
                )
        except Exception as e:
            r["tgtbal_longest_plateau"] = f"ERR:{type(e).__name__}"

    # ---- disk ----------------------------------------------------------
    total = 0
    for dirpath, _, filenames in os.walk(sd):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    r["disk_bytes"] = total

    rows.append(r)

df = pd.DataFrame(rows).sort_values("seed_idx").reset_index(drop=True)
df.to_csv(OUT / "per_seed_audit.csv", index=False)

# ======================= summaries =======================
print("\n================ COMPLETENESS ================")
complete = df[df.has_manifest & df.has_stats]
incomplete = df[~(df.has_manifest & df.has_stats)]
print(f"complete seeds (manifest+stats): {len(complete)}")
print(f"incomplete: {len(incomplete)} -> {incomplete.seed.tolist()}")
for _, x in incomplete.iterrows():
    present = [c for c in ["has_config", "has_trace", "has_conv", "has_context",
                           "has_stats", "has_manifest"] if x[c]]
    print(f"  {x.seed}: present={present} trace_rows={x.trace_rows} conv_rows={x.conv_rows} "
          f"pdq_rows={[x[f'pdq_{f}_rows'] for f in PDQ_PARQUETS]}")

bad_trace = complete[complete.trace_rows != EXPECTED_TRACE]
bad_conv = complete[complete.conv_rows != EXPECTED_CONV]
print(f"complete seeds w/ trace!={EXPECTED_TRACE}: {len(bad_trace)} {bad_trace.seed.tolist()}")
print(f"complete seeds w/ conv!={EXPECTED_CONV}: {len(bad_conv)} {bad_conv.seed.tolist()}")
zero_flip = complete[complete.pdq_archive_rows == 0]
print(f"zero-flip seeds (archive empty): {len(zero_flip)} {zero_flip.seed.tolist()}")
print("pareto json==png everywhere:",
      bool((complete.n_pareto_json == complete.n_pareto_png).all()))
print("n_pareto stats:", complete.n_pareto.describe()[["min", "mean", "max"]].to_dict())
print("archive rows stats:", complete.pdq_archive_rows.describe()[["min", "mean", "max"]].to_dict())
miss_ctx = complete[~complete.has_context]
print(f"complete seeds missing evolutionary/context.json: {len(miss_ctx)}")

print("\n================ CONFIG DRIFT ================")
print(f"distinct config hashes: {len(config_hashes)}")
for h, names in config_hashes.items():
    print(f"  {h}: {len(names)} seeds (e.g. {names[:2]})")
print("config schema_version values:", df.config_schema_version.dropna().unique().tolist())
print("stats schema_version values:", df.stats_schema_version.dropna().unique().tolist())
print("manifest schema_version values:", df.manifest_schema_version.dropna().unique().tolist())

print("\n================ EARLY-STOP ================")
print("generations values:", complete.generations.unique().tolist())
print("stats with early_stop key:", int(complete.stats_has_early_stop_key.sum()))
print("early_stop_trigger values:", complete.early_stop_trigger.dropna().unique().tolist())
pl = complete.dropna(subset=["tgtbal_longest_plateau"])
print(f"seeds with longest min-TgtBal plateau >= 50 gens: "
      f"{int((pl.tgtbal_longest_plateau >= 50).sum())}/{len(pl)}")
print(f"seeds with TAIL plateau >= 50 gens (no improvement in last 50+): "
      f"{int((pl.tgtbal_tail_plateau >= 50).sum())}/{len(pl)}")
print(f"seeds with TAIL plateau >= 100 gens: {int((pl.tgtbal_tail_plateau >= 100).sum())}")
print(f"seeds with TAIL plateau >= 150 gens: {int((pl.tgtbal_tail_plateau >= 150).sum())}")
print("last-improvement generation percentiles:",
      pl.tgtbal_last_improve_gen.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
print("tail plateau percentiles:",
      pl.tgtbal_tail_plateau.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
print("final min TgtBal percentiles:",
      pl.final_min_TgtBal.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict())
n_zero = int((pl.final_min_TgtBal <= 1e-30).sum())
print(f"seeds whose final min TgtBal <= 1e-30 (would have flip-triggered): {n_zero}")

print("\n================ ANCHOR DUPLICATION ================")
ag = pd.DataFrame(anchor_geno_stats)
print(f"seeds with anchor jsons: {len(ag)}")
print("distinct (p_a,p_b) among 3 anchors -> seed counts:")
print(ag.n_distinct_pa_pb.value_counts().sort_index().to_string())
print("distinct GENOTYPES among 3 anchors -> seed counts:")
print(ag.n_distinct_genotypes.value_counts().sort_index().to_string())
print("distinct FITNESS vectors among 3 anchors -> seed counts:")
print(ag.n_distinct_fitness.value_counts().sort_index().to_string())
ag.to_csv(OUT / "anchor_duplication.csv", index=False)

print("\n================ THROUGHPUT ================")
c = complete.copy()
c["seed_total_s"] = c.manifest_mtime - c.dir_ts
c["pdq_stage_s"] = c.manifest_mtime - c.stats_mtime
print("evo runtime_s:   ", c.evo_runtime_s.describe()[["min", "25%", "50%", "75%", "max", "mean"]].round(0).to_dict())
print("pdq stage wall_s (manifest-stats mtime):",
      c.pdq_stage_s.describe()[["min", "25%", "50%", "75%", "max", "mean"]].round(0).to_dict())
print("pdq SUT-call wall_s (sum wall_time_s):",
      c.pdq_sut_wall_s.describe()[["min", "50%", "max", "mean"]].round(0).to_dict())
print("seed total wall_s (manifest mtime - dir ts):",
      c.seed_total_s.describe()[["min", "25%", "50%", "75%", "max", "mean"]].round(0).to_dict())

# worker streams: even/odd interleave check
all_ts = df.sort_values("seed_idx")[["seed_idx", "dir_ts"]].to_numpy()
even = all_ts[all_ts[:, 0] % 2 == 0]
odd = all_ts[all_ts[:, 0] % 2 == 1]
print("even-idx stream monotone ts:", bool(np.all(np.diff(even[:, 1]) >= 0)))
print("odd-idx stream monotone ts:", bool(np.all(np.diff(odd[:, 1]) >= 0)))
for label, stream in [("even", even), ("odd", odd)]:
    gaps = np.diff(stream[:, 1])
    big = [(int(stream[i, 0]), int(stream[i + 1, 0]), int(g))
           for i, g in enumerate(gaps) if g > 6 * 3600]
    print(f"{label} stream: median inter-seed start gap {np.median(gaps)/3600:.2f} h, "
          f"max {gaps.max()/3600:.2f} h; gaps>6h: {big}")

span_s = df.dir_ts.max() - df.dir_ts.min()
n_started = len(df)
n_done = len(complete)
now = pd.Timestamp.now().timestamp()
elapsed = now - df.dir_ts.min()
rate_day = n_done / (elapsed / 86400)
remaining = TARGET_SEEDS - n_done
print(f"first start: {pd.to_datetime(df.dir_ts.min(), unit='s')}  "
      f"last start: {pd.to_datetime(df.dir_ts.max(), unit='s')}")
print(f"elapsed since first start: {elapsed/86400:.2f} days; complete={n_done}")
print(f"achieved rate: {rate_day:.2f} complete seeds/day (2 workers)")
print(f"remaining to {TARGET_SEEDS}: {remaining} -> linear ETA {remaining/rate_day:.1f} days "
      f"({(pd.Timestamp.now() + pd.Timedelta(days=remaining/rate_day)).date()})")
# alternative rate from per-seed wall time
med_total = c.seed_total_s.median()
print(f"median per-seed wall {med_total/3600:.2f} h -> 2 workers ideal rate "
      f"{2*86400/med_total:.2f} seeds/day -> ETA {remaining/(2*86400/med_total):.1f} days")

print("\n================ CACHE ================")
c["evo_hit_rate"] = c.cache_hits / (c.cache_hits + c.cache_misses)
print("evo cache hit-rate:", c.evo_hit_rate.describe()[["min", "25%", "50%", "75%", "max", "mean"]].round(4).to_dict())
# trend over seed_idx: correlation
corr = c[["seed_idx", "evo_hit_rate"]].corr().iloc[0, 1]
print(f"evo hit-rate vs seed_idx Pearson r = {corr:.3f}")
print("evo hit-rate by seed_idx decile:")
c["decile"] = pd.qcut(c.seed_idx, 10, labels=False)
print(c.groupby("decile").agg(idx_min=("seed_idx", "min"), idx_max=("seed_idx", "max"),
                              hit=("evo_hit_rate", "mean")).round(4).to_string())
print("pdq cache hit-rate:", c.pdq_cache_hit_rate.describe()[["min", "50%", "max", "mean"]].round(4).to_dict())
corr2 = c[["seed_idx", "pdq_cache_hit_rate"]].corr().iloc[0, 1]
print(f"pdq hit-rate vs seed_idx Pearson r = {corr2:.3f}")

print("\n================ DISK ================")
tot = df.disk_bytes.sum()
per_complete = complete.disk_bytes.mean()
print(f"total {tot/1e9:.2f} GB over {len(df)} dirs; mean complete seed {per_complete/1e6:.1f} MB "
      f"(min {complete.disk_bytes.min()/1e6:.1f}, max {complete.disk_bytes.max()/1e6:.1f})")
print(f"projection to {TARGET_SEEDS} seeds: {TARGET_SEEDS*per_complete/1e9:.1f} GB")

# anchored-class distribution sanity (pair-enumeration order claim)
print("\nanchor_class_concrete distribution:",
      complete.anchor_class_concrete.value_counts().to_dict())
print("target_class_concrete distribution:",
      complete.target_class_concrete.value_counts().to_dict())

summary = {
    "n_dirs": int(n_started),
    "n_complete": int(n_done),
    "incomplete": incomplete.seed.tolist(),
    "rate_per_day": float(rate_day),
    "eta_days_remaining_600": float(remaining / rate_day),
    "total_disk_gb": float(tot / 1e9),
    "projected_disk_gb_720": float(TARGET_SEEDS * per_complete / 1e9),
    "config_hashes": {h: len(v) for h, v in config_hashes.items()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nwrote", OUT / "per_seed_audit.csv", OUT / "anchor_duplication.csv", OUT / "summary.json")
