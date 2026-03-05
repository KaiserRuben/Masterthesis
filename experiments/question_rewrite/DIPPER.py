import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time

# --- Setup ---
torch_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {torch_device}")

# DIPPER is T5-XXL (~11B params, ~22GB fp32 / ~11GB fp16)
# If OOM on MPS, try: torch_dtype=torch.float16
MODEL_NAME = "kalpeshk2011/dipper-paraphraser-xxl"
print(f"Loading {MODEL_NAME} (this may take a while)...")

tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # fp16 to fit in memory
).to(torch_device)
model.eval()

ADEQUACY_THRESHOLD = 0.5
NUM_RETURN = 5

print("Loading similarity model for adequacy filtering...")
sim_model = SentenceTransformer("all-MiniLM-L6-v2", device=torch_device)


def dipper_paraphrase(
    text: str,
    lex_diversity: int = 40,   # 0-100, step 20
    order_diversity: int = 0,  # 0-100, step 20
    num_return: int = NUM_RETURN,
) -> list[tuple[str, float]]:
    """
    DIPPER control codes:
      lex_diversity:   0 = minimal lexical change, 100 = maximum
      order_diversity: 0 = preserve word order, 100 = maximum reordering
    """
    # DIPPER input format
    prefix = f"lexical = {lex_diversity}, order = {order_diversity}"
    input_text = f"{prefix} <sent> {text} </sent>"

    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).input_ids.to(torch_device)

    outputs = model.generate(
        input_ids,
        max_length=256,
        do_sample=True,
        top_p=0.75,
        top_k=None,
        num_return_sequences=num_return,
    )

    candidates = list(set(tokenizer.batch_decode(outputs, skip_special_tokens=True)))
    candidates = [c.strip() for c in candidates if c.strip().lower() != text.strip().lower()]

    if not candidates:
        return []

    src_emb = sim_model.encode(text, convert_to_tensor=True)
    cand_embs = sim_model.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(src_emb, cand_embs)[0].cpu().tolist()

    results = [
        (cand, score)
        for cand, score in zip(candidates, scores)
        if score >= ADEQUACY_THRESHOLD
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def lexical_diversity(original: str, candidates: list[str]) -> float:
    orig_tokens = set(original.lower().split())
    distances = []
    for c in candidates:
        cand_tokens = set(c.lower().split())
        union = orig_tokens | cand_tokens
        distances.append(1 - len(orig_tokens & cand_tokens) / len(union) if union else 0)
    return np.mean(distances) if distances else 0.0


def pairwise_diversity(candidates: list[str]) -> float:
    if len(candidates) < 2:
        return 0.0
    distances = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            t_i = set(candidates[i].lower().split())
            t_j = set(candidates[j].lower().split())
            union = t_i | t_j
            distances.append(1 - len(t_i & t_j) / len(union) if union else 0)
    return np.mean(distances)


# --- Diversity settings to sweep ---
CONFIGS = [
    {"lex_diversity": 0,  "order_diversity": 0,  "label": "lex=0, ord=0 (minimal)"},
    {"lex_diversity": 20, "order_diversity": 0,  "label": "lex=20, ord=0"},
    {"lex_diversity": 40, "order_diversity": 0,  "label": "lex=40, ord=0"},
    {"lex_diversity": 60, "order_diversity": 0,  "label": "lex=60, ord=0"},
    {"lex_diversity": 60, "order_diversity": 40, "label": "lex=60, ord=40"},
    {"lex_diversity": 80, "order_diversity": 60, "label": "lex=80, ord=60 (aggressive)"},
    {"lex_diversity": 100,"order_diversity": 100,"label": "lex=100, ord=100 (max)"},
]

test_questions = [
    "Is there a pedestrian crossing the road?",
    "How many vehicles are visible in this image?",
    "What abnormalities are present in the chest X-ray?",
    "What is the weather condition in this driving scene?",
]

print(f"\nModel loaded. Running sweep...\n")

for question in test_questions:
    print("=" * 100)
    print(f"ORIGINAL: {question}")
    print("=" * 100)

    for config in CONFIGS:
        label = config["label"]
        t0 = time.time()

        results = dipper_paraphrase(
            question,
            lex_diversity=config["lex_diversity"],
            order_diversity=config["order_diversity"],
        )

        elapsed = time.time() - t0
        phrases = [r[0] for r in results]
        lex_d = lexical_diversity(question, phrases)
        pair_d = pairwise_diversity(phrases)
        avg_sim = np.mean([r[1] for r in results]) if results else 0

        print(f"\n  [{label}]  lex_div={lex_d:.3f}  pair_div={pair_d:.3f}  avg_sim={avg_sim:.3f}  ({elapsed:.1f}s)")
        if results:
            for i, (phrase, score) in enumerate(results, 1):
                is_q = phrase.strip().endswith("?")
                print(f"    {i}. {phrase}  (sim: {score:.3f}) {'✓' if is_q else '✗'}")
        else:
            print("    No paraphrases met threshold.")

    print()