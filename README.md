# VLM Boundary Testing

Search-based boundary testing for Vision-Language Models. Finds minimal input perturbations (text + image) that push a VLM to the decision boundary between two classes.

## How It Works

**Genotype** `[image_genes | text_genes]` — integer vector, 0 = keep original, k = k-th nearest candidate.

**Manipulators** apply the genotype: VQGAN codebook swaps (image) and PoS-aware synonym replacement (text).

**SUT** scores the manipulated input via teacher-forced log-prob decoding against all category labels.

**Optimizer** (AGE-MOEA-2) minimises four objectives simultaneously:

| Objective | Measures |
|-----------|----------|
| MatrixDistance | Pixel distance to reconstructed baseline |
| TextReplacementDistance | Cosine distance of replaced words |
| TargetedBalance | \|P(A) − P(B)\| → 0 at boundary |
| ArchiveSparsity | Genotype diversity across archive |

## Run

```bash
pip install -e tools/smoo
python experiments/run_boundary_test.py configs/boundary_test.yaml
```

## Structure

```
src/
├── config.py                  # ExperimentConfig (single YAML → dacite)
├── data/imagenet.py           # ImageNet labels + streaming
├── manipulator/
│   ├── image/                 # VQGAN codebook swaps
│   ├── text/                  # PoS-aware synonym replacement (spaCy + FastText)
│   └── vlm_manipulator.py     # Multi-modal bridge
├── objectives/                # MatrixDistance, TextReplacementDist, TargetedBalance, Concentration, GenomeDistance
├── optimizer/                 # Discrete AGE-MOEA-2 with per-gene int bounds
├── sut/                       # Teacher-forced log-prob scoring (VLMSUT + VLMScorer)
└── tester/                    # Orchestrator + seed generator (1-vs-all pairs)

experiments/
├── run_boundary_test.py       # Main entry point
├── seed_generation_test.py    # Seed triple generation
├── imagenet_vlm_probing/      # VLM classification baselines
└── question_rewrite/          # Text manipulation prototypes (archived)

tools/
├── smoo/                      # SMOO framework (git submodule, pip install -e)
└── VLTest/                    # Reference implementation

configs/
└── boundary_test.yaml         # Experiment config template
```

## SMOO Integration

```python
from smoo.objectives import Criterion, CriterionCollection
from smoo.optimizer import Optimizer
from smoo.sut import SUT
```

## Supervision

| Role | Name | Affiliation |
|------|------|-------------|
| Supervisor | Prof. Andrea Stocco | TUM / fortiss |
| Co-Supervisor | Oliver Weißl | fortiss |
