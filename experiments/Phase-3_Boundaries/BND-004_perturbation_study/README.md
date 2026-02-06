# BND-004: Input Modality Perturbation Study

## Research Questions

1. **Image vs Text Trust**: When text contradicts the image, does Alpamayo follow visual or textual input?
2. **Key Sensitivity**: Which semantic keys have the largest influence on trajectory prediction?
3. **Directional Effects**: Are some value transitions (e.g., clear→rainy) more impactful than others?

## Methodology

For each scene with known ground truth labels:

```
For each semantic key K (weather, road_type, ...):
    For each possible value V in key K's domain:
        1. Create perturbed text: "Scene context: clear weather, on a highway, ..."
           with K=V (may match or contradict the actual image)
        2. Run Alpamayo inference with (original image, perturbed text)
        3. Compute ADE between predicted and ground truth trajectory
        4. Record: {scene, key, original_value, perturbed_value, is_aligned, ADE}
```

## Files

```
BND-004_perturbation_study/
├── run_perturbations.py  # Main inference loop (requires GPU)
├── analyze_results.py    # Statistical analysis
├── visualize_3d.py       # 3D interactive visualizations
└── README.md             # This file
```

Configuration is derived directly from source definitions:
- `PERTURBATION_KEYS` from `pipeline.lib.schema.CLASSIFICATION_KEYS`
- `TEXT_VOCABULARY` from `pipeline.step_1_embed`

## Usage

### 1. Run Perturbation Study (requires GPU)

```bash
# Pilot study (10 scenes)
python run_perturbations.py --n-scenes 10

# Full study (50 scenes, ~1500 inferences)
python run_perturbations.py

# Resume from checkpoint
python run_perturbations.py --resume
```

### 2. Analyze Results

```bash
python analyze_results.py
```

### 3. Generate Visualizations

```bash
python visualize_3d.py
```

## Output

### Data Files (in `data/BND-004/`)

- `perturbation_results.parquet` - Raw results
- `perturbation_analysis.json` - Statistical analysis

### Visualizations (in `data/BND-004/figures/`)

- `alignment_effect_3d.html` - Aligned vs misaligned ADE by key
- `transition_heatmaps.html` - Value transition effects per key
- `key_value_ade_surface.html` - Key × Value × ADE surface
- `key_importance_3d.html` - Key importance ranking

## Expected Results

### Alignment Effect
- **Positive effect**: Misalignment increases ADE → Model trusts image over text
- **Negative effect**: Misalignment decreases ADE → Model trusts text over image
- **Zero effect**: Model ignores this key

### Key Sensitivity
- High sensitivity = Model predictions change significantly when this key varies
- Low sensitivity = Model ignores this key's text description

## Computational Cost

| Scenes | Keys | Values/Key | Total Inferences | Est. Time |
|--------|------|------------|------------------|-----------|
| 10     | 6    | ~5         | ~300             | ~50 min   |
| 50     | 6    | ~5         | ~1,500           | ~4 hours  |
| 100    | 6    | ~5         | ~3,000           | ~8 hours  |

*Assuming ~10 sec/inference on RTX 4090*
