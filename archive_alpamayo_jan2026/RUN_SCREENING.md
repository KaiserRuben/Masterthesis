# Perturbation Screening - Quick Start

## 1. Script ausführen

```bash
python3 perturbation_screening.py
```

**Dauer:** ~5-15 Minuten (45 Inferences)

## 2. Was wird getestet

**5 Samples:**
- Sample 2: "glass to far left" (hard, IoU=0.88)
- Sample 5: "bottom right white box truck" (hard, IoU=0.42)
- Sample 17: "TV on the bottom right" (hard, IoU=0.49)
- Sample 0: "giraffe front" (easy, IoU=0.81)
- Sample 1: "guy in plaid" (medium, IoU=0.58)

**3 Kategorien × 3 Perturbationen = 9 pro Sample:**

1. **Geometric**: Rotation -10°, 0°, 10°
2. **Occlusion**: 0%, 15%, 30% verdeckt
3. **Quality**: Blur 0, 2, 4 pixels

## 3. Output

Das Script gibt dir:

✓ **4 Metriken pro Kategorie:**
  - IoU Variance (>0.05 = Effekt vorhanden)
  - Boundary Rate (20-40% = meaningful)
  - Clustering Score (>0.7 = strukturiert)
  - Gradient Alignment (>0.5 = konsistent)

✓ **GO/NO-GO Entscheidung:**
  - GO = Alle 4 Kriterien erfüllt → Lohnt sich!
  - NO-GO = <4 Kriterien → Keine Struktur

✓ **Visualisierung:**
  - figures/screening_results.png

✓ **Gespeicherte Daten:**
  - data/screening_results.npz

## 4. Interpretation

**Wenn Geometric GO:**
→ Geometrische Transformationen zeigen Struktur
→ Deep analysis mit Rotation, Scale, Translation, Crop
→ Thesis: "Task-relevante Perturbationen für Grounding"

**Wenn Occlusion GO:**
→ Semantische Perturbationen zeigen Struktur
→ Deep analysis mit verschiedenen Occlusion-Patterns
→ Thesis: "Semantic robustness in VLM grounding"

**Wenn Quality GO:**
→ Bildqualität zeigt Struktur
→ Deep analysis mit Blur, Noise, Compression
→ Thesis: "Feature extraction robustness"

**Wenn NICHTS GO:**
→ Entweder VLM zu instabil
→ Oder: Andere Task versuchen (classification?)
→ Oder: Negatives Ergebnis dokumentieren

## 5. Nach dem Screening

Basierend auf dem Ergebnis:
- **GO:** Implementiere deep analysis für gewinnende Kategorie
- **NO-GO:** Überlege Pivot oder dokumentiere Limitation

