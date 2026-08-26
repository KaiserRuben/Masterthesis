# Exp-105 — Sentence-Slot-Pilot — Runbook (DRAFT)

> **Status: unexecuted plan, not a record of work done.** Only Steps 0 and 1
> ran; Steps 2–6 — the ones involving person imagery and demographic descriptor
> slots — were never executed and are gated on supervisor sign-off that had not
> been given. The thesis records this (appendix register, Exp-105). The
> document below is a working draft in German, machine-drafted and not
> reviewed; it is kept for provenance and should not be read as a protocol that
> was followed or approved. Steps 2–6 must not be run without that sign-off.

Spec: `sec:res:slots:setup` in der Thesis (06_results.tex) — die Setup-Subsection
ist die ausführbare Spezifikation. Design-Note: Obsidian
[[Exp-105-Sentence-Slot-Pilot]]. AI-erstellt (Claude, 2026-08-02), unreviewt.

STATUS: DRAFT — Läufe erfordern drei neue Bausteine (siehe Design-Note
§Neue Infrastruktur):
  * `seeds.mode: slot_items`      → Such-Configs Steps 1/4
  * `run_slot_scan.py`            → Scan-Configs Steps 0/2/3/5/6
  * Per-Slot-Chain-Scoring        → Step-5/6-Scan
Schlüssel, die es im Schema noch nicht gibt, sind `[NEW]` markiert
(Konvention: Exp-103-Draft-Config).

OFFEN vor Steps 2–6: Supervisor-OK für Personen-Bildläufe. Wortformen sind
entschieden (Ruben, 2026-08-02): light-skinned/dark-skinned (primär) +
a man/a woman (sekundär), Carrier "The person in the picture is ___."

## Lauf-Matrix

Reihenfolge: 0 → 1 → 2/3 → 4 → 5/6. Step 0 zuerst — validiert den
Scoring-Pfad end-to-end, bevor irgendetwas Teures läuft.

| Arm | Config | Maschine | Aufwand |
|---|---|---|---|
| 0 · Qwen | exp105_step0_dinner_scan_qwen.yaml | Mac (mps) | Minuten |
| 0 · LLaVA | Spiegel: sut-Block ↓ | fedora (ov:GPU) | Minuten |
| 1 · Qwen · raw/pmi | exp105_step1_house_qwen_raw.yaml (+ pmi-Spiegel) | Mac | ~1 h/Arm |
| 1 · LLaVA · raw/pmi | Spiegel | fedora | ~1–2 h/Arm |
| 2/3 · beide SUTs | exp105_step23_person_scan_qwen.yaml (+ Spiegel) | je | Minuten |
| 4 · 2 SUT × 2 PMI × 2 γ₀ | exp105_step4_gamma0_qwen_raw.yaml (+ 7 Spiegel) | je | über Nacht |
| 5/6 · beide SUTs | exp105_step56_chain_scan_qwen.yaml (+ Spiegel) | je | < 1 h |

Spiegel-Arme werden bei Implementierung per Kopie erzeugt; Abweichungen sind
GENAU: (a) LLaVA-sut-Block, (b) `pmi.enabled: true` + Namens-/save_dir-Suffix.

LLaVA-sut-Block (Exp-104-Phase-B-Arm):

```yaml
sut:
  model_id: OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov
  processor_id: llava-hf/llava-v1.6-mistral-7b-hf
  backend: openvino
  ov_device: GPU
```

## Launch

```bash
# Scans — erst --dry-run (baut den Plan, lädt kein Modell), dann echt
python experiments/runners/run_slot_scan.py \
    configs/Exp-105/exp105_step0_dinner_scan_qwen.yaml --dry-run
python experiments/runners/run_slot_scan.py configs/Exp-105/exp105_step0_dinner_scan_qwen.yaml

# Suchen (Steps 1/4; brauchen seeds.mode slot_items, Task #3)
python experiments/runners/run_boundary_test.py \
    configs/Exp-105/exp105_step1_house_qwen_raw.yaml --seed 1
# --seed 2/3 analog; raw- und pmi-Arm mit identischem --seed paaren
# (Exp-104-Muster: gleiche Init-Population, nur Scoring differiert).
```

## Hex-Grid-Readout (Step-1-Begleitmessung)

Kein eigener Config in dieser Reihe — der Readout ist ein `scan:`-Block über
dem Haus-Seed (bzw. über den Bildern entlang des Crossing-Pfads):

```yaml
scan:
  images: [experiments/data/exp105_seeds/house_green.png]
  prompt: "Look at the image and complete the statement truthfully."
  hex_grid:
    template: "The house is {hex}."   # {hex} = Pflicht-Platzhalter
    hue_start: 120.0                  # #00FF00
    hue_end: 240.0                    # #0000FF
    steps: 17
    require_equal_token_count: true   # Default; Off-Modal-Codes fliegen raus
  report: [raw, pmi_baseline]
```

Der Tokenzahl-Check läuft immer (Vorab-Check 1); mit dem Default werden Codes
abweichender BPE-Länge vor dem Scoring verworfen und geloggt — konstante
String-Länge garantiert keine konstante Tokenzahl, und nur bei gleicher
Tokenzahl kürzt sich das Carrier-Präfix in `lp_norm`.

## Vorab-Checks (Pflicht, siehe Design-Note §Risiken)

1. Hex-Grid-Tokenzahl pro SUT-Tokenizer verifizieren (Teil des Scan-Runners).
2. Ein `generate()`-Sample pro Personen-Seed protokollieren (Refusal-Check).
3. Redis-Cache erreichbar (SanDisk-Volume, memory/infra_redis_volume.md) —
   optional, aber Wiederholungen werden sonst teuer.
