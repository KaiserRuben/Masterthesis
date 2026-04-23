# Perturbation Selection Framework für VLM Robustness

## Problem
Wie wählt man relevante Perturbationen ohne a priori Wissen?

## Hypothese
**Relevante Perturbationen erzeugen strukturierte Boundaries**
- Konsistente Instabilitätsregionen
- Boundaries aligned mit Gradienten
- Clustering score >0.7
- Wenige (2-5) kohärente Cluster

**Irrelevante Perturbationen erzeugen Rauschen**
- Random scatter (was du jetzt siehst)
- Keine Gradienten-Alignment
- Clustering score <0.4
- Viele kleine Cluster

## Proposed Experiment

### Phase 1: Screening (5 samples, 5 categories, 5 pert each = 125 inferences)
Test auf den 5 instabilsten Samples aus deinen Daten:

1. **Photometric**: brightness, contrast, saturation, hue, gamma
2. **Geometric**: rotation (-15°, -7°, 0°, 7°, 15°), scale, translation  
3. **Quality**: blur, noise, JPEG compression, resolution
4. **Occlusion**: 10%, 20%, 30%, 40%, random masks
5. **Context**: background blur, crop, padding

**Metrics:**
- Clustering score
- Gradient alignment ratio
- Boundary rate
- IoU variance

**Selection criterion:** 
Top 2 categories mit höchstem clustering score + gradient alignment

### Phase 2: Deep Analysis (30 samples, 2 categories, 20 pert each = 1200 inferences)
- Dense sampling in relevanten Kategorien
- Full boundary detection analysis
- Characterize instability regions

### Phase 3: Validation (andere Task, z.B. classification)
- Testen ob gleiche Kategorien relevant sind
- Build task → perturbation mapping

## Expected Outcome

**Scenario A: Task-specific relevance**
- Grounding → Geometric relevant
- Classification → Quality relevant  
- Detection → Occlusion relevant
→ Framework zur task-based perturbation selection

**Scenario B: Model-specific relevance**
- Qwen3-VL sensitiv für X
- GPT-4V sensitiv für Y
→ Framework zur model benchmarking

**Scenario C: Universal patterns**
- Gewisse Perturbationen immer relevant
→ Standard perturbation suite

## Timeline
- Phase 1: 1 Tag (125 inferences)
- Phase 2: 2-3 Tage (1200 inferences)  
- Phase 3: 1 Woche (andere Task setup)

Total: ~2 Wochen statt 10 Tage für alle 99 samples

## Scientific Value
- Addresses fundamental methodological gap
- Systematic statt ad-hoc perturbation selection
- Generalizable framework
- Negative result (brightness/contrast) ist wichtiger Beitrag!
