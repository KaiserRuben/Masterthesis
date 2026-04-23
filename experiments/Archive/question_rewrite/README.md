# Question Rewrite Prototypes

Prototyping text manipulation strategies for the testing pipeline's text manipulator component.

## Purpose

Evaluates paraphrase models for seed expansion — generating semantically equivalent prompt variants that can serve as starting points for POS-aware synonym mutation in the optimizer loop.

## Models

| Script | Model | Approach |
|--------|-------|----------|
| `PEGASUS.py` | `tuner007/pegasus_paraphrase` | Sentence-level paraphrase (primary candidate) |
| `DIPPER.py` | DIPPER | Discourse-level paraphrase with controllable diversity |
| `PARROT.py` | Parrot | Paraphrase with fluency/diversity/adequacy control |

## Context

In the pipeline, sentence-level rewriting (Pegasus) is used for **seed expansion** only — generating diverse starting prompts. The optimizer then applies fine-grained **POS-aware synonym replacement** (spaCy + fastText) as the actual mutation operator during search.

This separation keeps the mutation operator chirurgical (single-word substitution, constant prompt length) while still allowing diverse seed coverage.
