"""Backfill pair-space flip map from Exp-100 poc_boundary_pair PDQ logs.

Pair-flip criterion: lp[target_concrete] > lp[anchor_concrete] within the
6-option prompt (pair_margin = lp[anchor] - lp[target] < 0).
READ-ONLY on runs/; outputs under exp100_partial/ with backfill_ prefix.
"""
import json, glob, os, traceback
import numpy as np
import pandas as pd

RUNS = '/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair'
OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
N_TXT = 19

seed_rows, cand_rows, skip_log = [], [], []

dirs = sorted(glob.glob(f'{RUNS}/seed_*'))
for d in dirs:
    sname = os.path.basename(d).rsplit('_', 1)[0]  # seed_NNNN
    try:
        if not os.path.exists(f'{d}/pdq/sut_calls.parquet'):
            skip_log.append((sname, 'no sut_calls.parquet')); continue
        meta = json.load(open(f'{d}/evolutionary/stats.json'))['seed_metadata']
        man = json.load(open(f'{d}/manifest.json'))
        sc = pd.read_parquet(f'{d}/pdq/sut_calls.parquet')
        ca = pd.read_parquet(f'{d}/pdq/candidates.parquet')
        has_archive = os.path.exists(f'{d}/pdq/archive.parquet')
        ar = pd.read_parquet(f'{d}/pdq/archive.parquet') if has_archive else pd.DataFrame()
        s1 = pd.read_parquet(f'{d}/pdq/stage1_flips.parquet') if os.path.exists(f'{d}/pdq/stage1_flips.parquet') else pd.DataFrame()
        s2 = pd.read_parquet(f'{d}/pdq/stage2_trajectories.parquet') if os.path.exists(f'{d}/pdq/stage2_trajectories.parquet') else pd.DataFrame()
    except Exception as e:
        skip_log.append((sname, f'load error: {e}')); continue

    try:
        anc_cls, tgt_cls = meta['anchor_class_concrete'], meta['target_class_concrete']
        cats = list(sc['categories'].iloc[0])
        ia, it = cats.index(anc_cls), cats.index(tgt_cls)
        lp = np.vstack(sc['logprobs'].to_numpy())
        sc = sc.reset_index(drop=True)
        sc['pair_margin'] = lp[:, ia] - lp[:, it]
        seg = (sc.stage == 'anchor').cumsum() - 1
        pareto_by_seg = [a['pareto_idx'] for a in man['anchors']]
        n_seg = int(seg.max()) + 1 if len(sc) else 0
        seg_ok = n_seg == len(pareto_by_seg)

        # anchor genotypes
        anc_geno = {}
        for a in man['anchors']:
            aj = json.load(open(f"{d}/pdq/anchors/anchor_{a['pareto_idx']:03d}.json"))
            anc_geno[a['pareto_idx']] = np.asarray(aj['genotype'])

        ca_idx = ca.set_index('candidate_id') if len(ca) else None

        # --- hamming + genotype per call ---
        ham = np.full(len(sc), -1, dtype=int)
        img_diff = np.full(len(sc), -1, dtype=int)
        txt_diff = np.full(len(sc), -1, dtype=int)
        callpos = {cid: i for i, cid in enumerate(sc.call_id.values)}

        def set_diff(i, g, anc):
            dv = (g != anc)
            ham[i] = int(dv.sum())
            txt_diff[i] = int(dv[-N_TXT:].sum())
            img_diff[i] = ham[i] - txt_diff[i]

        # stage1 (and anchor=0)
        n_ham_recomputed = 0
        for i, row in sc.iterrows():
            if row.stage == 'anchor':
                ham[i] = img_diff[i] = txt_diff[i] = 0
            elif row.stage == 'stage1' and ca_idx is not None and row.candidate_id in ca_idx.index:
                g = np.asarray(ca_idx.loc[row.candidate_id, 'genotype'])
                pidx = pareto_by_seg[seg.iloc[i]] if seg_ok else pareto_by_seg[0]
                anc = anc_geno[pidx]
                set_diff(i, g, anc)
                stored = ca_idx.loc[row.candidate_id, 'hamming_to_anchor']
                if pd.isna(stored):
                    n_ham_recomputed += 1
                elif int(stored) != ham[i]:
                    # trust stored if mismatch vs segment anchor: use seed-level min over anchors
                    hmins = min(int((g != a).sum()) for a in anc_geno.values())
                    ham[i] = min(int(stored), hmins)

        # stage2 replay
        n_replay_fail = 0
        if len(s2) and len(s1) and len(ar):
            s1_idx = s1.set_index('flip_id')
            flip2anchor = ar.set_index('flip_id')['pareto_idx'].to_dict()
            for fid, grp in s2.groupby('flip_id'):
                try:
                    start_cand = s1_idx.loc[fid, 'candidate_id']
                    cur = np.asarray(ca_idx.loc[start_cand, 'genotype']).copy()
                    anc = anc_geno[flip2anchor[fid]]
                    for _, st in grp.sort_values('step').iterrows():
                        prop = cur.copy()
                        if prop[st.target_gene] != st.old_value:
                            raise RuntimeError(f'replay divergence flip {fid} step {st.step}')
                        prop[st.target_gene] = st.new_value
                        if st.sut_call_id in callpos:
                            set_diff(callpos[st.sut_call_id], prop, anc)
                        if st.accepted:
                            cur = prop
                except Exception:
                    n_replay_fail += 1

        sc['hamming'] = ham
        sc['img_diff'] = img_diff
        sc['txt_diff'] = txt_diff

        # per-segment anchor margin: margin of the segment's anchor-stage call.
        # genuine pair-flip = candidate crosses while its own anchor is on the
        # anchor side under the SAME 6-option prompt (excludes prompt-shift
        # reclassification artifacts).
        seg_anchor_margin = sc.loc[sc.stage == 'anchor'].set_index(seg[sc.stage == 'anchor'])['pair_margin']
        sc['anchor_margin_seg'] = seg.map(seg_anchor_margin)
        sc['genuine'] = (sc.pair_margin < 0) & (sc.anchor_margin_seg > 0)

        # --- pair-flip calls (exclude anchor-stage calls) ---
        body = sc[sc.stage != 'anchor']
        pf = body[body.pair_margin < 0]
        n_calls, n_pf = len(body), len(pf)
        gf = body[body.genuine]

        for _, r in pf.iterrows():
            cand_rows.append(dict(
                seed=sname, seed_idx=man['seed_idx'], target=tgt_cls,
                la=meta['level_anchor'], lt=meta['level_target'],
                call_id=int(r.call_id), candidate_id=int(r.candidate_id),
                stage=r.stage, pair_margin=float(r.pair_margin),
                anchor_margin_seg=float(r.anchor_margin_seg),
                genuine=bool(r.genuine),
                hamming=int(r.hamming), img_diff=int(r.img_diff),
                txt_diff=int(r.txt_diff), top1_label=r.top1_label,
                cat6_is_target=bool(r.top1_label == tgt_cls),
                cat6_is_boa=bool(r.top1_label == 'boa constrictor'),
            ))

        # --- stage2 waste ---
        s2_pf = s2_pf_rej = s2_pf_broken6 = s2_acc_nonpf = 0
        if len(s2):
            m2 = sc[sc.stage == 'stage2'].set_index('call_id')['pair_margin']
            s2v = s2[s2.sut_call_id.isin(m2.index)].copy()
            s2v['pm'] = m2.loc[s2v.sut_call_id].values
            s2_pf = int((s2v.pm < 0).sum())
            s2_pf_rej = int(((s2v.pm < 0) & (~s2v.accepted)).sum())
            s2_pf_broken6 = int(((s2v.pm < 0) & (~s2v.still_flipped)).sum())
            s2_acc_nonpf = int(((s2v.pm >= 0) & (s2v.accepted)).sum())

        # archive final minima: pair-flip status of genotype_min / genotype_flipped
        n_arch = len(ar)
        n_min_pf = n_disc_pf = 0
        if n_arch:
            lpm = np.vstack(ar['logprobs_min'].to_numpy())
            lpf = np.vstack(ar['logprobs_flipped'].to_numpy())
            n_min_pf = int((lpm[:, ia] - lpm[:, it] < 0).sum())
            n_disc_pf = int((lpf[:, ia] - lpf[:, it] < 0).sum())

        anchor_margins = sc.loc[sc.stage == 'anchor', 'pair_margin'].values
        seed_rows.append(dict(
            seed=sname, seed_idx=man['seed_idx'], run_dir=os.path.basename(d),
            target=tgt_cls, anchor=anc_cls,
            la=meta['level_anchor'], lt=meta['level_target'],
            common_ancestor_level=meta.get('common_ancestor_level'),
            seed_idx_in_class=meta.get('seed_idx_in_class'),
            label_a=meta.get('anchor_label_in_prompt'), label_b=meta.get('target_label_in_prompt'),
            n_anchors=len(man['anchors']),
            anchor_margin_mean=float(anchor_margins.mean()) if len(anchor_margins) else np.nan,
            anchor_margin_min=float(anchor_margins.min()) if len(anchor_margins) else np.nan,
            n_anchor_pairflipped=int((anchor_margins < 0).sum()),
            n_calls=n_calls, n_pairflips=n_pf,
            frac_pairflips=n_pf / n_calls if n_calls else np.nan,
            n_pf_stage1=int((pf.stage == 'stage1').sum()),
            n_pf_stage2=int((pf.stage == 'stage2').sum()),
            min_pf_hamming=int(pf.hamming[pf.hamming >= 0].min()) if (pf.hamming >= 0).any() else np.nan,
            min_pf_img_diff=int(pf.loc[pf.hamming[pf.hamming >= 0].idxmin(), 'img_diff']) if (pf.hamming >= 0).any() else np.nan,
            min_pf_txt_diff=int(pf.loc[pf.hamming[pf.hamming >= 0].idxmin(), 'txt_diff']) if (pf.hamming >= 0).any() else np.nan,
            min_pf_margin=float(pf.pair_margin.min()) if n_pf else np.nan,
            n_genuine=len(gf),
            min_gf_hamming=int(gf.hamming[gf.hamming >= 0].min()) if (gf.hamming >= 0).any() else np.nan,
            min_gf_img_diff=int(gf.loc[gf.hamming[gf.hamming >= 0].idxmin(), 'img_diff']) if (gf.hamming >= 0).any() else np.nan,
            min_gf_txt_diff=int(gf.loc[gf.hamming[gf.hamming >= 0].idxmin(), 'txt_diff']) if (gf.hamming >= 0).any() else np.nan,
            pf_boa_share=float((pf.top1_label == 'boa constrictor').mean()) if n_pf else np.nan,
            pf_target_share=float((pf.top1_label == tgt_cls).mean()) if n_pf else np.nan,
            n_stage2_steps=len(s2), s2_pairflip_steps=s2_pf,
            s2_pairflip_rejected=s2_pf_rej, s2_pairflip_broken6cat=s2_pf_broken6,
            s2_accepted_nonpairflip=s2_acc_nonpf,
            n_archive_flips=n_arch, n_min_is_pairflip=n_min_pf,
            n_discovery_is_pairflip=n_disc_pf,
            n_replay_fail=n_replay_fail, n_ham_recomputed=n_ham_recomputed,
            seg_ok=seg_ok,
        ))
    except Exception as e:
        skip_log.append((sname, f'process error: {e}'))
        traceback.print_exc()

per_seed = pd.DataFrame(seed_rows)
per_cand = pd.DataFrame(cand_rows)
per_seed.to_parquet(f'{OUT}/backfill_pairflips_per_seed.parquet', index=False)
per_cand.to_parquet(f'{OUT}/backfill_pairflip_candidates.parquet', index=False)

print(f"processed {len(per_seed)} seeds, skipped {len(skip_log)}: {skip_log}")
print(f"pair-flip calls total: {len(per_cand)}")
print(f"seeds with >=1 pair-flip: {(per_seed.n_pairflips > 0).sum()} / {len(per_seed)}")
print(f"replay failures: {per_seed.n_replay_fail.sum()}, seg_ok all: {per_seed.seg_ok.all()}")
print(f"hamming missing among pair-flips: {(per_cand.hamming < 0).sum() if len(per_cand) else 0}")
