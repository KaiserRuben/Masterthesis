import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- Setup ---
torch_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {torch_device}\n")

model_name = "prithivida/parrot_paraphraser_on_T5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)
sim_model = SentenceTransformer("all-MiniLM-L6-v2", device=torch_device)

ADEQUACY_THRESHOLD = 0.75
MAX_DISPLAY = 5

# --- Generation strategies ---
STRATEGIES = {
    "beam_baseline": dict(
        num_beams=10,
        num_return_sequences=10,
        do_sample=False,
    ),
    "sampling_mild": dict(
        do_sample=True,
        top_k=30,
        top_p=0.9,
        temperature=1.0,
        num_return_sequences=20,
    ),
    "sampling_moderate": dict(
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.3,
        num_return_sequences=20,
    ),
    "sampling_aggressive": dict(
        do_sample=True,
        top_k=100,
        top_p=0.98,
        temperature=1.8,
        num_return_sequences=20,
    ),
    "sampling_extreme": dict(
        do_sample=True,
        top_k=200,
        top_p=0.99,
        temperature=2.5,
        num_return_sequences=30,
    ),
}


def lexical_diversity(original: str, candidates: list[str]) -> float:
    """Average token-level Jaccard distance from original."""
    orig_tokens = set(original.lower().split())
    distances = []
    for c in candidates:
        cand_tokens = set(c.lower().split())
        intersection = orig_tokens & cand_tokens
        union = orig_tokens | cand_tokens
        distances.append(1 - len(intersection) / len(union) if union else 0)
    return np.mean(distances) if distances else 0.0


def pairwise_diversity(candidates: list[str]) -> float:
    """Average pairwise Jaccard distance among candidates."""
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


def paraphrase(text: str, strategy: dict) -> list[tuple[str, float]]:
    input_ids = tokenizer(
        f"paraphrase: {text}",
        return_tensors="pt",
        max_length=128,
        truncation=True,
    ).input_ids.to(torch_device)

    outputs = model.generate(input_ids, max_length=128, **strategy)
    candidates = list(set(tokenizer.batch_decode(outputs, skip_special_tokens=True)))
    candidates = [c for c in candidates if c.strip().lower() != text.strip().lower()]

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
    return results[:MAX_DISPLAY]


# --- Test ---
test_questions = [
    "Is there a pedestrian crossing the road?",
    "How many vehicles are visible in this image?",
    "What abnormalities are present in the chest X-ray?",
    "What is the weather condition in this driving scene?",
]

for question in test_questions:
    print("=" * 90)
    print(f"ORIGINAL: {question}")
    print("=" * 90)

    for name, strategy in STRATEGIES.items():
        results = paraphrase(question, strategy)
        phrases = [r[0] for r in results]

        lex_div = lexical_diversity(question, phrases)
        pair_div = pairwise_diversity(phrases)
        avg_sim = np.mean([r[1] for r in results]) if results else 0

        print(f"\n  [{name}]  lex_div={lex_div:.3f}  pair_div={pair_div:.3f}  avg_sim={avg_sim:.3f}")
        if results:
            for i, (phrase, score) in enumerate(results, 1):
                is_q = phrase.strip().endswith("?")
                print(f"    {i}. {phrase}  (sim: {score:.3f}) {'✓' if is_q else '✗'}")
        else:
            print("    No paraphrases met threshold.")

    print()