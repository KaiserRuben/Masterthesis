#!/usr/bin/env bash
# HS-GEN-01 lean expansion — run ONLY the 6 newly-promoted image-only pairs
# (idx 332/727/586/839/630/58) to widen the image_heavy anchor-image diversity
# for the HS-01 frozen pool. Sequential on the Arc A770; ~2.5 h/run.
# --resume is NOT wired for gap_filter (see chain script), so run configs directly.
set -u
cd ~/Projects/Masterarbeit
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uni
mkdir -p runs/HS-GEN-01/_logs
echo "=== LEAN6 START $(date) host $(hostname) python=$(which python) ==="
for idx in 332 727 586 839 630 58; do
  cfg=$(ls configs/HS-GEN-01/hs_gen01_promoted_idx${idx}_*.yaml 2>/dev/null | head -1)
  echo "=== idx${idx}: ${cfg:-MISSING}  $(date) ==="
  [ -z "$cfg" ] && { echo "MISSING config idx${idx}"; continue; }
  python experiments/runners/run_boundary_test.py "$cfg" || echo "FAIL idx${idx}"
done
echo "=== LEAN6 ALL DONE $(date) ==="
