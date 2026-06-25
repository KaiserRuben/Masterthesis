#!/usr/bin/env bash
# HS pool diversity campaign (smart/fast): widen DISTINCT anchor photos for the
# HS-01 frozen pool. Short budgets on fast-crossing, semantically-distant pairs.
#   1) HS-GEN-02 joint diversity — 108 cells, 30 gen (one invocation, workers=2)
#   2) HS-GEN-01 image-only distant expansion — 6 cross-super-category pairs, 40 gen
# Sequential on the Arc A770; ~17h total. Roster-mode HS-GEN-02 is --resume-safe.
set -u
cd ~/Projects/Masterarbeit
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uni
mkdir -p runs/HS-GEN-01/_logs runs/HS-GEN-02/_logs
echo "=== CAMPAIGN START $(date) host $(hostname) ==="
echo "=== [1] HS-GEN-02 joint diversity (108 cells, 30 gen) $(date) ==="
python experiments/runners/run_boundary_test.py configs/HS-GEN-02/hs_gen02_joint_diversity.yaml \
  || echo "FAIL hs-gen-02-joint"
for idx in 226 311 479 287 115 102; do
  cfg=$(ls configs/HS-GEN-01/hs_gen01_promoted_idx${idx}_*.yaml 2>/dev/null | head -1)
  echo "=== [2] image-only idx${idx}: ${cfg:-MISSING}  $(date) ==="
  [ -z "$cfg" ] && { echo "MISSING idx${idx}"; continue; }
  python experiments/runners/run_boundary_test.py "$cfg" || echo "FAIL idx${idx}"
done
echo "=== CAMPAIGN ALL DONE $(date) ==="
