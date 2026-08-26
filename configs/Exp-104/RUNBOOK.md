# Exp-104 Phase B — Runbook

> This campaign has completed; the runbook is kept as the record of how it was
> executed. Machine names below (`this Mac`, the Arc workstation reached over a
> personal VPN alias) are the author's; substitute your own. Results are in
> `runs/Exp-104/` — see `docs/REPRODUCTION.md`.

Suite: raw-vs-PMI live A/B over an 8-cell Δ∅ dose-response ladder, cross-SUT
(Qwen and LLaVA), + null-image sensitivity. Full design + registry:
Obsidian `[[Exp-104-PMI-Calibration]]` §Phase-B-Design.

**Cells** (`filter_indices = [2, 26, 28, 96, 196, 278, 454, 474]`) — matched
across SUTs (enumeration is SUT-independent). Arms differ ONLY in the `pmi`
block; `pmi.apply_to_seedgen=false` + matched `--seed` ⇒ the search objective
is the sole difference.

---

## Qwen (Apple Silicon, MPS)

```bash
# E0 + E1 (prior map Δ∅×4 nulls + post-hoc PMI floors on archived Exp-101q):
conda run -n uni python experiments/analysis/exp104_pmi_calibration.py
#   → experiments/analysis/output/exp104/exp104_pmi.csv  (now with d0_white)

# E2 (live A/B, 48 runs, rep-1 both arms first, ~40–80 h):
bash configs/Exp-104/launch_qwen_ab.sh          # writes runs/Exp-104/qwen_{raw,pmi}_rep{1,2,3}/

# E3 (null sweep, after E0 picks divergent cells): PMI arm, black/white/noise,
#   2 cells × 1 matched seed — configs generated once E0 identifies the cells.
```

## LLaVA (Intel Arc workstation, OpenVINO backend)

Prereqs on the box: repo synced (this commit), env with the deps, VQGAN knn
cache `~/.cache/vqgan_knn/f8_16384_full.npz`, ImageNet cache, Redis up, and the
archived **Exp-101/102 LLaVA runs** present for E0·L/E1·L.

```bash
# In a tmux/screen (survives ssh drop):
cd <repo>            # workstation checkout
conda activate uni   # or the box's env

# E0·L + E1·L (LLaVA prior map + post-hoc; point --runs-glob at archived LLaVA runs):
python experiments/analysis/exp104_pmi_calibration.py \
    --runs-glob "runs/Exp-101/exp101_margin_predictor_seed_*" \
    --model OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov \
    --processor-id llava-hf/llava-v1.6-mistral-7b-hf \
    --backend openvino --device cpu --ov-device GPU \
    --out experiments/analysis/output/exp104_llava
#   → cross-SUT Δ∅ (walled lexeme flips snake↔constrictor) + P4 residual decomposition

# E2·L (live A/B, 48 runs, workers=2 OV-GPU+CPU):
bash configs/Exp-104/launch_llava_ab.sh         # writes runs/Exp-104/llava_{raw,pmi}_rep{1,2,3}/
```

---

## Analysis (after runs land, per SUT)

```bash
# Reach + 2D-Pareto structure, per (cell, arm, seed) + dose-response vs Δ∅:
conda run -n uni python experiments/analysis/exp104_phaseb_reach_hv.py \
    --runs-glob "runs/Exp-104/qwen_*_rep*/exp104_phaseb_qwen_*_seed_*" \
    --e0-csv experiments/analysis/output/exp104/exp104_pmi.csv \
    --out experiments/analysis/output/exp104/phaseb_qwen.csv
# LLaVA: swap the glob to runs/Exp-104/llava_* and --e0-csv to the LLaVA prior map.
```

**Verdict lens** (pre-registered, per Obsidian §Phase-B-Design): promises hold
if reach is preserved on non-wall cells, W-walls fall while G keeps a residual
floor, N cells get modestly harder, 2D-HV(PMI)≈HV(raw) on C controls and differs
elsewhere only in the Δ∅-predicted direction, and the sparse frontier still
spans a comparable n_active range everywhere.
