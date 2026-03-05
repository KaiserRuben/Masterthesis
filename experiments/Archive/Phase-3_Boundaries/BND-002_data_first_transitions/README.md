# BND-002: Data-First Transition Analysis

## Experiment Metadata

| Field | Value |
|-------|-------|
| **ID** | BND-002 |
| **Status** | COMPLETED |
| **Date** | 2026-02-03 |
| **Output** | `data/BND-002/` |

## Research Question

**Which semantic attributes cause the largest changes in trajectory prediction error when varied between similar scenes?**

## Method: Data-First Boundary Detection

Unlike SLERP-based interpolation (deprecated BND-001), this experiment uses **real scene pairs** from the dataset that differ in semantic attributes, ensuring all comparisons are on-manifold.

### Pipeline

```
1. Load Data
   ├── CLS-001: 100 classified scenes (24 semantic keys)
   ├── EMB-001: OpenCLIP bigG embeddings (1280-dim)
   └── Alpamayo: Trajectory predictions + ADE

2. Build k-NN Graph (k=10)
   └── 100 nodes, 761 edges, cosine similarity

3. Find Pairs with Limited Key Differences
   ├── 1-key-diff: 13 pairs
   ├── 2-key-diff: 62 pairs
   └── 3-key-diff: 188 pairs
   Total: 263 pairs

4. Analyze Trajectory Transitions
   └── For each pair: compute |ΔADE| between scenes

5. Build Stability Map
   └── Aggregate |ΔADE| per semantic key (weighted by 1/n_diff)
```

## Key Findings

### Stability Map (Corrected: Relative |ΔADE|)

| Rank | Key | Abs |ΔADE| | Rel |ΔADE| | Confidence |
|------|-----|-------------|-------------|------------|
| 1 | cyclists_present | 4.48m | **120%** | medium |
| 2 | pedestrians_present | 1.37m | **99%** | medium |
| 3 | weather | 1.61m | **97%** | high |
| 4 | occlusion_level | 1.86m | **94%** | high |
| 5 | required_action | 1.39m | **93%** | high |
| 6 | construction_activity | 1.61m | 89% | medium |
| 7 | depth_complexity | 1.63m | 88% | medium |
| 8 | time_of_day | 1.08m | 85% | high |
| 9 | visual_degradation | 1.56m | 81% | high |
| 10 | traffic_situation | 1.52m | 75% | high |
| 11 | safety_criticality | 1.15m | 73% | high |
| 12 | similar_object_confusion | 1.00m | 72% | medium |
| 13 | road_type | 1.42m | 71% | medium |
| 14 | traffic_signals_visible | 1.09m | 70% | medium |

**Note:** "Relative |ΔADE|" = |ADE_b - ADE_a| / mean(ADE_a, ADE_b). This controls for baseline ADE differences between scene types.

### Key Insights

**⚠️ METHODOLOGICAL NOTE:** Initial analysis used absolute |ΔADE|, which confounds sensitivity with baseline ADE differences. Corrected analysis uses **relative |ΔADE|** = |ΔADE| / mean(ADE_a, ADE_b).

#### Corrected Findings (Relative Metric)

1. **Vulnerability-related attributes are most sensitive**: cyclists_present (120%), pedestrians_present (99%), required_action (93%) — all involve vulnerable road users or safety actions.

2. **Perceptual factors remain important**: weather (97%), occlusion_level (94%) rank high in both absolute and relative metrics.

3. **Road infrastructure is less sensitive than expected**: road_type (71%) and traffic_signals_visible (70%) dropped significantly when using relative metric.

4. **Time of day impact is larger than absolute metric suggested**: Moved from #13 to #8 when accounting for baseline differences.

5. **Sensitivity ratio is 1.7x (not 4.5x)**: The relative metric shows more uniform sensitivity across keys.

6. **Embedding similarity does NOT predict |ΔADE|** (r = 0.058). Scenes can be visually similar but have very different trajectory errors.

### Top Transitions by Impact

| Transition | Mean |ΔADE| |
|------------|--------------|
| cyclists_present: False → True | 4.48m |
| traffic_situation: complex → critical | 4.29m |
| weather: clear → rainy | 4.15m |
| road_type: highway → intersection | 3.74m |
| occlusion_level: severe → minimal | 2.73m |

## Limitations

1. **Small single-key-diff sample**: Only 13 pairs differ in exactly one key (0.3% of all pairs). Analysis relies on weighted multi-diff pairs (1/n_diff weighting).

2. **Imbalanced categories**: 68% of scenes are "clear" weather, 65% are "urban_street". Rare transitions (e.g., foggy → rainy) have low statistical power.

3. **Baseline ADE confounding**: Initial absolute |ΔADE| metric was confounded by inherent difficulty differences between scene types. Corrected with relative metric.

4. **Medium confidence for top findings**: cyclists_present (rank #1) and pedestrians_present (rank #2) both have only medium confidence (6.8 and 13.3 weighted pairs respectively).

5. **No trajectory class change detection**: Alpamayo output format doesn't include pre-classified trajectory behavior.

6. **All keys show high relative change (70-120%)**: This suggests substantial baseline noise in trajectory predictions, making it hard to isolate true sensitivity effects.

## Output Files

```
data/BND-002/
├── m1_data_exploration.json     # Data loading summary
├── knn_graph.pkl                # NetworkX graph
├── knn_graph_metadata.json      # Graph statistics
├── similarity_matrix.npz        # Pairwise cosine similarities
├── relaxed_pairs.json           # 263 pairs (1-3 key differences)
├── trajectory_analysis.json     # Per-pair trajectory metrics
├── stability_map.json           # Final sensitivity map
├── experiment_report.json       # Summary statistics
└── figures/
    ├── stability_map_bars.png   # Sensitivity ranking
    ├── transition_heatmap.png   # Top transitions
    ├── similarity_vs_ade.png    # Embedding vs trajectory correlation
    └── ade_by_key_count.png     # ΔADE distribution
```

## Reproducibility

```bash
cd experiments/Phase-3_Boundaries/BND-002_data_first_transitions/

# Run all milestones in sequence
python m1_data_exploration.py
python m2_knn_graph.py
python m3_relaxed_pairs.py
python m4_trajectory_analysis.py
python m5_stability_map.py
python m6_visualization.py
```

## Relation to Thesis

This experiment addresses **RQ2: Is the VLM decision landscape anisotropic?**

**Answer**: Yes. The trajectory model shows 4.5x sensitivity variation across semantic keys. Perceptual attributes (cyclists, occlusion, weather) are far more impactful than abstract safety concepts (criticality, required action).

## Scale-Up Results (BND-002b/c)

### BND-002b: Label Propagation (2,600 scenes)

| Metric | BND-002 | BND-002b |
|--------|---------|----------|
| Scenes | 100 | 2,600 |
| Single-key-diff pairs | 13 | 8,184 |
| Pairs with ADE | 63 | 63 (bottleneck) |

### BND-002c: Gap-Fill Embedding (2,647 scenes)

47 scenes had ADE data but no embeddings. After embedding:

| Metric | Before | After |
|--------|--------|-------|
| Single-key-diff pairs | 8,184 | 16,684 |
| Pairs with ADE | 63 | **224** |

### Final Stability Map (N=224)

| Rank | Key | Rel |ΔADE| | N |
|------|-----|-------------|-----|
| 1 | time_of_day | 108% | 4 |
| 2 | weather | 96% | 64 |
| 3 | required_action | 95% | 71 |
| 4 | depth_complexity | 89% | 16 |
| 5 | road_type | 70% | 50 |
| 6 | occlusion_level | 64% | 19 |

**Key Finding**: occlusion_level dropped from rank #2 to #6 with larger samples. BND-002 rankings were not robust.

### Ranking Stability

**Spearman ρ = 0.029** — Rankings do not replicate between BND-002 and BND-002c.

## Conclusion

1. **Rankings are unstable** and should not be used for definitive claims
2. **weather and required_action** show consistent moderate sensitivity (~95%)
3. **Embedding similarity does NOT predict trajectory error** (r = 0.058)
4. **Sample size matters**: Initial findings with N<20 were misleading
