"""Compare sentence-distance behaviour across embedding backends.

Motivation: the current TextReplacementDistance objective sums per-word
fasttext cosine distances. Word-level distances conflate antonym/negation
with synonymy ("main" <-> "non-main" sit very close in distributional
space). This script quantifies the gap across three backends:

    1. fasttext (current) -- sum of per-word cosine distances, matching
       TextReplacementDistance; also mean-pooled sentence-level variant
       for an apples-to-apples sentence comparison.
    2. Qwen3.5-4B  -- mean-pooled last-hidden-state of the SUT itself.
    3. A modern BERT-class sentence encoder -- BAAI/bge-base-en-v1.5.

The reference pairs cover:
    - the motivating case (main vs non-main),
    - a true paraphrase (main vs primary),
    - a content-word swap (cat vs dog),
    - self-identity (sanity baseline).

Each pair reports cosine distance (1 - cos_sim). A healthy backend
separates "non-main" (semantic negation) from "primary" (synonym); the
hypothesis is that fasttext does not, while Qwen / bge do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


SENTENCES = {
    "orig":       "What is the main subject of this image",
    "negated":    "What is the non-main subject of this image",
    "synonym":    "What is the primary subject of this image",
    "content":    "What is the main object in this photograph",
    "unrelated":  "How many people are walking on the street",
    "identical":  "What is the main subject of this image",
}

PAIRS = [
    ("orig", "negated",  "negation   (main -> non-main)"),
    ("orig", "synonym",  "synonym    (main -> primary)"),
    ("orig", "content",  "content    (subject/image -> object/photograph)"),
    ("orig", "unrelated", "unrelated (different question)"),
    ("orig", "identical", "identical  (sanity)"),
]


@dataclass
class Result:
    backend: str
    distances: dict[str, float]
    load_s: float
    encode_s: float


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


# -------------------------------------------------------------- fasttext

def run_fasttext() -> tuple[Result, Result]:
    """Return (sentence-mean, token-sum) results for fasttext."""
    import gensim.downloader as api

    t0 = time.time()
    kv = api.load("fasttext-wiki-news-subwords-300")
    load = time.time() - t0

    def sent_vec(s: str) -> np.ndarray:
        toks = [t.lower() for t in s.split()]
        vecs = [kv[t] for t in toks if t in kv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size)

    def token_sum_dist(a: str, b: str) -> float:
        """Per-aligned-token cosine distance sum (only where tokens differ)."""
        ta = [t.lower() for t in a.split()]
        tb = [t.lower() for t in b.split()]
        if len(ta) != len(tb):
            return float("nan")
        total = 0.0
        for x, y in zip(ta, tb):
            if x == y:
                continue
            if x not in kv or y not in kv:
                total += float("nan")
                continue
            total += cos_dist(kv[x], kv[y])
        return total

    t0 = time.time()
    vecs = {k: sent_vec(v) for k, v in SENTENCES.items()}
    enc = time.time() - t0

    sent_dists = {lbl: cos_dist(vecs[a], vecs[b]) for a, b, lbl in PAIRS}
    tok_dists  = {lbl: token_sum_dist(SENTENCES[a], SENTENCES[b])
                  for a, b, lbl in PAIRS}

    return (
        Result("fasttext-sent-mean", sent_dists, load, enc),
        Result("fasttext-tok-sum (current objective)", tok_dists, 0.0, 0.0),
    )


# -------------------------------------------------------------- Qwen 3.5-4B

def run_qwen() -> Result:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3.5-4B"
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        output_hidden_states=False,
    ).to(device).eval()
    load = time.time() - t0

    @torch.no_grad()
    def embed(s: str) -> np.ndarray:
        batch = tok(s, return_tensors="pt").to(device)
        out = model(**batch, output_hidden_states=True)
        last = out.hidden_states[-1][0]         # (seq, dim)
        mask = batch["attention_mask"][0].unsqueeze(-1).float()
        pooled = (last * mask).sum(0) / mask.sum().clamp(min=1)
        return pooled.float().cpu().numpy()

    t0 = time.time()
    vecs = {k: embed(v) for k, v in SENTENCES.items()}
    enc = time.time() - t0

    dists = {lbl: cos_dist(vecs[a], vecs[b]) for a, b, lbl in PAIRS}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return Result("qwen3.5-4b-mean-last", dists, load, enc)


# -------------------------------------------------------------- BGE (modern BERT sentence encoder)

def run_bge() -> Result:
    from transformers import AutoModel, AutoTokenizer

    model_id = "BAAI/bge-base-en-v1.5"
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    load = time.time() - t0

    @torch.no_grad()
    def embed(s: str) -> np.ndarray:
        batch = tok(s, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**batch)
        cls = out.last_hidden_state[:, 0]   # bge uses CLS
        cls = F.normalize(cls, dim=-1)
        return cls[0].float().cpu().numpy()

    t0 = time.time()
    vecs = {k: embed(v) for k, v in SENTENCES.items()}
    enc = time.time() - t0

    dists = {lbl: cos_dist(vecs[a], vecs[b]) for a, b, lbl in PAIRS}

    return Result("bge-base-en-v1.5 (CLS)", dists, load, enc)


# --------------------------------------------------------------

def print_result(r: Result) -> None:
    print(f"\n=== {r.backend} ===  load {r.load_s:.1f}s  encode {r.encode_s:.2f}s")
    for lbl, d in r.distances.items():
        bar = "#" * int(min(d, 1.0) * 40)
        print(f"  {lbl:50s}  {d:+.4f}  {bar}")


def main() -> None:
    results: list[Result] = []

    print("loading fasttext...")
    sent_ft, tok_ft = run_fasttext()
    results += [tok_ft, sent_ft]

    print("loading qwen3.5-4b...")
    results.append(run_qwen())

    print("loading bge-base-en-v1.5...")
    results.append(run_bge())

    print("\n" + "=" * 72)
    print("COSINE DISTANCE  (higher = more different; lower = more similar)")
    print("=" * 72)
    for r in results:
        print_result(r)

    print("\nInterpretation target:")
    print("  Healthy backend => d(negation) > d(synonym).")
    print("  Broken backend  => d(negation) <= d(synonym),")
    print("                     i.e. 'non-main' is treated as a paraphrase of 'main'.")


if __name__ == "__main__":
    main()
