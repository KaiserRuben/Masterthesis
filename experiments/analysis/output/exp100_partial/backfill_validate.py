import json, glob
import numpy as np
import pandas as pd

d = glob.glob('/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair/seed_0006_*')[0]
meta = json.load(open(f'{d}/evolutionary/stats.json'))['seed_metadata']
man = json.load(open(f'{d}/manifest.json'))
sc = pd.read_parquet(f'{d}/pdq/sut_calls.parquet')
ca = pd.read_parquet(f'{d}/pdq/candidates.parquet')
ar = pd.read_parquet(f'{d}/pdq/archive.parquet')
s1 = pd.read_parquet(f'{d}/pdq/stage1_flips.parquet')
s2 = pd.read_parquet(f'{d}/pdq/stage2_trajectories.parquet')

cats = list(sc['categories'].iloc[0])
# check categories constant
assert all(list(c) == cats for c in sc['categories'].iloc[::50]), "categories vary"
ia, it = cats.index(meta['anchor_class_concrete']), cats.index(meta['target_class_concrete'])
lp = np.vstack(sc['logprobs'].to_numpy())
pair_margin = lp[:, ia] - lp[:, it]
print("n calls:", len(sc), " pair-flips (margin<0):", int((pair_margin < 0).sum()))
print("pair-flips excluding anchor calls:", int((pair_margin[sc.stage.values != 'anchor'] < 0).sum()))
print("anchor call margins:", pair_margin[sc.stage.values == 'anchor'])

# segment attribution
seg = (sc.stage == 'anchor').cumsum() - 1
pareto_by_seg = [a['pareto_idx'] for a in man['anchors']]
print("segments:", seg.max() + 1, "pareto order:", pareto_by_seg)

# anchor genotypes
anc_geno = {}
for a in man['anchors']:
    aj = json.load(open(f"{d}/pdq/anchors/anchor_{a['pareto_idx']:03d}.json"))
    anc_geno[a['pareto_idx']] = np.array(aj['genotype'])
    print("anchor", a['pareto_idx'], "geno len", len(aj['genotype']), "keys:", list(aj.keys()))

# verify stage1 hamming
ca_idx = ca.set_index('candidate_id')
sc1 = sc[sc.stage == 'stage1']
ok, bad = 0, 0
for _, row in sc1.iterrows():
    g = np.array(ca_idx.loc[row.candidate_id, 'genotype'])
    pidx = pareto_by_seg[seg.loc[row.name]]
    h = int((g != anc_geno[pidx]).sum())
    stored = int(ca_idx.loc[row.candidate_id, 'hamming_to_anchor'])
    if h == stored: ok += 1
    else:
        bad += 1
        if bad < 4: print("MISMATCH cand", row.candidate_id, "recomputed", h, "stored", stored)
print(f"stage1 hamming verify: {ok} ok, {bad} mismatch")

# replay stage2 per flip_id
s1_idx = s1.set_index('flip_id')
flip2anchor = ar.set_index('flip_id')['pareto_idx'].to_dict()
ham_by_call = {}
geno_by_call = {}
for fid, grp in s2.groupby('flip_id'):
    start_cand = s1_idx.loc[fid, 'candidate_id']
    cur = np.array(ca_idx.loc[start_cand, 'genotype']).copy()
    anc = anc_geno[flip2anchor[fid]]
    grp = grp.sort_values('step')
    for _, st in grp.iterrows():
        prop = cur.copy()
        assert prop[st.target_gene] == st.old_value, f"replay mismatch flip {fid} step {st.step}: have {prop[st.target_gene]} expect {st.old_value}"
        prop[st.target_gene] = st.new_value
        ham_by_call[st.sut_call_id] = int((prop != anc).sum())
        geno_by_call[st.sut_call_id] = prop
        if st.accepted:
            cur = prop
    # check final equals genotype_min
    gmin = np.array(ar.set_index('flip_id').loc[fid, 'genotype_min'])
    if not np.array_equal(cur, gmin):
        print(f"flip {fid}: final replay != genotype_min, diff {int((cur!=gmin).sum())}")
print("replayed stage2 calls:", len(ham_by_call), "of", (sc.stage == 'stage2').sum())

# min hamming among pair-flip calls
sc = sc.assign(pair_margin=pair_margin, seg=seg.values)
ham = []
for _, row in sc.iterrows():
    if row.stage == 'anchor': ham.append(0)
    elif row.stage == 'stage1': ham.append(int(ca_idx.loc[row.candidate_id, 'hamming_to_anchor']))
    else: ham.append(ham_by_call.get(row.call_id, -1))
sc['hamming'] = ham
pf = sc[(sc.pair_margin < 0) & (sc.stage != 'anchor')]
print("\npair-flip calls:", len(pf), "min hamming:", pf.hamming.min(), "by stage:", pf.stage.value_counts().to_dict())
print("6-cat label of pair-flips:", pf.top1_label.value_counts().to_dict())
print("min-hamming pair-flip rows:\n", pf.nsmallest(5, 'hamming')[['call_id','stage','pair_margin','hamming','top1_label']])
