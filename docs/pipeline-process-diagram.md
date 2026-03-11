# VLM Boundary Testing Pipeline — Process Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT CONFIG                        │
│  seeds: list[(image, class_A, class_B)]                     │
│  generations, pop_size, n_candidates, device, model_id      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   FOR EACH SEED TRIPLE │
              │  (image, class_A, B)   │
              └───────────┬────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPARE PHASE  (once per seed)            │
│                                                             │
│  ┌──────────────────────┐   ┌─────────────────────────────┐ │
│  │  ImageManipulator    │   │  TextManipulator            │ │
│  │  .prepare(image)     │   │  .prepare(prompt,           │ │
│  │                      │   │    exclude_words=categories)│ │
│  │                      │   │                             │ │
│  │  → ImageContext      │   │  → TextContext              │ │
│  │    · original grid   │   │    · original tokens        │ |
│  │    · patch selection │   │    · word selection         │ │
│  │    · patch candidates│   │    · synonym candidates     │ │
│  └──────────┬───────────┘   └──────────┬──────────────────┘ │
│             │                          │                    |
│             │                          │                    │
│             ▼                          ▼                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  VLMManipulator.prepare(image, prompt, exclude)      │   │
│  │                                                      │   │
│  │  Computes:                                           │   │
│  │   · gene_bounds = [img_bounds | txt_bounds]          │   │
│  │   · text_candidate_distances (cosine, precomputed)   │   │
│  │   · genotype_dim = image_dim + text_dim              │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                               │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             OPTIMIZER INIT  (once per seed)                  │
│                                                             │
│  DiscretePymooOptimizer(                                    │
│      gene_bounds = manipulator.gene_bounds,                 │
│      num_objectives = 5,                                    │
│      pop_size = 50,                                         │
│      algorithm = AGEMOEA2                                   │
│  )                                                          │
│  → draws initial population: int64[pop_size, genotype_dim]  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ FOR gen IN generations │
              └───────────┬────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. GET POPULATION                                      │ │
│  │    genotypes = optimizer.get_x_current()               │ │
│  │    shape: (pop_size, genotype_dim)  dtype: int64       │ │
│  │    gene 0 = keep original, gene k = k-th candidate     │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 2. MANIPULATE                                          │ │
│  │    images, texts = manipulator.manipulate(weights=gen) │ │
│  │                                                        │ │
│  │    For each individual:                                │ │
│  │    ┌─────────────────────────────────────────────┐     │ │
│  │    │genotype = [...img_genes... | ...txt_genes..]│     │ │
│  │    │              ◄─image_dim─►   ◄──text_dim──► │     │ │
│  │    └────────┬───────────────────────────┬────────┘     │ │
│  │             │                           │              │ │
│  │             ▼                           ▼              │ │
│  │    image_manip.apply(ctx, img_genes)   text_manip.apply│ │
│  │    → VQGAN codebook swaps              (ctx, txt_genes)│ │
│  │    → PIL Image                         → str (prompt)  │ │
│  │                                                        │ │
│  │    Output: list[PIL.Image], list[str]                  │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 3. SUT EVALUATION                                      │ │
│  │    For each (image, text) pair:                        │ │
│  │      logprobs = vlm_sut.process_input(image, text)     │ │
│  │                                                        │ │
│  │    Teacher-forced decoding:                            │ │
│  │      For each category label:                          │ │
│  │        log_prob_norm = Σ log P(token_i | prefix)       │ │
│  │                        ─────────────────────────       │ │
│  │                              n_tokens                  │ │
│  │                                                        │ │
│  │    Output: logits tensor (pop_size, n_categories)      │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 4. CONVERT PIL → TENSOR                                │ │
│  │    origin_batch = seed_image → (pop, 3, H, W) float    │ │
│  │    perturbed_batch = PIL images → (pop, 3, H, W) float │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 5. EVALUATE ALL OBJECTIVES                             │ │
│  │                                                        │ │
│  │    objectives.evaluate_all(                            │ │
│  │      images        = [origin_batch, perturbed_batch],  │ │
│  │      logits        = logits_tensor,                    │ │
│  │      text_genotypes = txt_genes_batch,                 │ │
│  │      text_candidate_distances = precomputed_dists,     │ │
│  │      solution_archive = genome_archive,                │ │
│  │      genome_target = current_genotypes,                │ │
│  │      genome_archive = archive_of_past_genotypes,       │ │
│  │      batch_dim     = 0,                                │ │
│  │    )                                                   │ │
│  │                                                        │ │
│  │    Each criterion picks what it needs, ignores rest:   │ │
│  │                                                        │ │
│  │    ┌─────────────────────────────────────────────────┐ │ │
│  │    │  #1 MatrixDistance        ← images              │ │ │
│  │    │      Frobenius norm of (origin - perturbed)     │ │ │
│  │    │      → minimize image perturbation              │ │ │
│  │    ├─────────────────────────────────────────────────┤ │ │
│  │    │  #2 TextReplacementDist   ← text_genotypes,     │ │ │
│  │    │                             text_candidate_dist │ │ │
│  │    │      Σ cosine_dist(original_word, replacement)  │ │ │
│  │    │      → minimize semantic text drift             │ │ │
│  │    ├─────────────────────────────────────────────────┤ │ │
│  │    │  #3 TargetedBalance       ← logits              │ │ │
│  │    │      |P(class_A) - P(class_B)|                  │ │ │
│  │    │      → minimize = push toward decision boundary │ │ │
│  │    ├─────────────────────────────────────────────────┤ │ │
│  │    │  #4 Concentration         ← logits              │ │ │
│  │    │      Σ P(c) for c ∉ {A, B}                      │ │ │
│  │    │      → minimize = keep mass on target pair      │ │ │
│  │    ├─────────────────────────────────────────────────┤ │ │
│  │    │  #5 ArchiveSparsity       ← genome_target,      │ │ │
│  │    │                             genome_archive      │ │ │
│  │    │      min dist to archive (on genotype space)    │ │ │
│  │    │      → minimize (inverted) = maximize diversity │ │ │
│  │    └─────────────────────────────────────────────────┘ │ │
│  │                                                        │ │
│  │    Output: dict[name → list[float]]  (pop_size each)   │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 6. ASSIGN FITNESS + PARETO EXTRACTION                  │ │
│  │    optimizer.assign_fitness(                           │ │
│  │      (obj1_vals, obj2_vals, obj3_vals, obj4_vals,      │ │
│  │       obj5_vals),                                      │ │
│  │      images_list, predictions_list                     │ │
│  │    )                                                   │ │
│  │                                                        │ │
│  │    Internally:                                         │ │
│  │      · Merge current pop fitness with historical best  │ │
│  │      · Non-dominated sorting → Pareto front            │ │
│  │      · Update best_candidates                          │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 7. UPDATE OPTIMIZER                                    │ │
│  │    optimizer.update()                                  │ │
│  │                                                        │ │
│  │    · Tell fitness to PyMoo (StaticProblem + Evaluator) │ │
│  │    · AGE-MOEA-2 selection + SBX crossover + PM mut     │ │
│  │    · Ask for next generation                           │ │
│  │    · Sanitize: round, clip to [0, gene_bounds-1]       │ │
│  │                                                        │ │
│  │    → new population ready for next iteration           │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│                   ┌──────────────┐                          │
│                   │ next gen?    │──── yes ──→ loop to (1)  │
│                   └──────┬───────┘                          │
│                          │ no                               │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOGGING (continuous, per generation)      │
│                                                             │
│  Every SUT interaction is recorded:                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  trace.parquet  — one row per individual per generation │ │
│  │                                                        │ │
│  │  Columns:                                              │ │
│  │   · seed_id, generation, individual                    │ │
│  │   · genotype (int64 array — reconstructable to img+txt)│ │
│  │   · logprobs (float array — full SUT output)           │ │
│  │   · decoded_text (str — the mutated prompt)            │ │
│  │   · fitness_img_dist, fitness_txt_dist,                │ │
│  │     fitness_balance, fitness_concentration,            │ │
│  │     fitness_sparsity                                   │ │
│  │   · predicted_class, P(class_A), P(class_B)            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Images are NOT saved per-individual (reconstructable       │
│  from genotype + context deterministically).                │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    SAVE RESULTS (per seed, at end)           │
│                                                             │
│  · Pareto-optimal candidate images + texts (decoded)        │
│  · Origin image + original prompt for reference             │
│  · Stats JSON (runtime, budget, seed config, gene_bounds)   │
│  · Context metadata (patch positions, word positions,       │
│    candidate lists — for reconstruction)                    │
│                                                             │
│  optimizer.reset()  →  next seed triple                     │
└─────────────────────────────────────────────────────────────┘
```

## Component Inventory

```
 COMPONENT                  STATUS      LOCATION
 ─────────────────────────  ──────────  ────────────────────────────────────
 ImageManipulator           DONE        src/manipulator/image/
 TextManipulator            DONE        src/manipulator/text/
 VLMManipulator (bridge)    DONE        src/manipulator/vlm_manipulator.py
 VLMSUT                     DONE        src/sut/vlm_sut.py
 DiscretePymooOptimizer     DONE        src/optimizer/discrete_pymoo_optimizer.py
 #1 MatrixDistance          REUSE SMOO  tools/smoo/.../image_criteria/
 #2 TextReplacementDist     DONE        src/objectives/text_replacement_distance.py
 #3 TargetedBalance         DONE        src/objectives/targeted_balance.py
 #4 Concentration           DONE        src/objectives/concentration.py
 #5 ArchiveSparsity         REUSE SMOO  tools/smoo/.../auxiliary_criteria/
 VLMBoundaryTester          TODO        src/tester/  (new)
 ExperimentConfig           TODO        src/tester/  (new)
 Entry point / run script   TODO        experiments/ (new)
```

## Data Flow Summary

```
seed (image, prompt, class_A, class_B)
        │
        ▼
    PREPARE ──→ contexts + gene_bounds + text_distances
        │
        ▼
    OPTIMIZER ──→ int64 genotypes [img_genes | txt_genes]
        │
        ▼
    MANIPULATE ──→ (PIL images, prompt strings)
        │
        ▼
    SUT ──→ log_prob_norm tensor (pop_size, n_categories)
        │
        ▼
    5 OBJECTIVES ──→ 5 fitness arrays (all minimized)
        │
        ▼
    PARETO FRONT ──→ non-dominated boundary candidates
```
