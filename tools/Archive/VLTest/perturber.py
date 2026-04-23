import os
import sys
import time
import platform
import argparse
import random

import json
from collections import Counter

import torch
import numpy as np
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from omegaconf import OmegaConf
import PIL
from PIL import Image
from taming.models.vqgan import VQModel
from main import instantiate_from_config
from tqdm import tqdm
from glob import glob
from queue import Queue
from threading import Thread

from transformers import (
    BlipProcessor,
    BlipForImageTextRetrieval,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from sentence_transformers import SentenceTransformer, util

import spacy
import fasttext
import itertools

import re

# Ensure local imports work
sys.path.append(".")

# Disable gradients to save memory
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda")

# Hyperparameters
DATA_ROOT = "data"
TASK_REGISTRY = {
    "itm": {
        "name": "image_text_matching",
        "dataset": "coco",
        "models": ["albef", "blip"],
    },
    "vqa": {
        "name": "visual_question_answering",
        "dataset": "vqav2",
        "models": ["blip2", "lxmert"],
    },
    "nlvr": {
        "name": "visual_reasoning",
        "dataset": "nlvr2",
        "models": ["vilbert", "lxmert"],
    },
}


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x


def norm01(img):
    img = (img + 1.0) / 2.0
    return img


def preprocess(img, target_image_size=256):
    s = min(img.size)  # shortest side

    # Just allow upscaling instead of raising
    # (for larger images this will downscale; for smaller, it will upscale)
    r = target_image_size / s
    new_h = round(r * img.size[1])
    new_w = round(r * img.size[0])

    img = TF.resize(img, (new_h, new_w), interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)

    return img


def load_or_build_codebook_knn(vqgan, cache_name, metric="cosine"):
    """
    For each codeword, build a sorted KNN list (nearest first).
    Only keep top-k neighbors.
    """

    cache_path = os.path.join("logs", cache_name)

    if os.path.exists(cache_path):
        print(f"[Codebook] Loading cached KNN from {cache_path}")
        data = np.load(cache_path)
        return data["knn"]  # shape: [n_embed, k]

    print("[Codebook] Cache not found, building KNN...")

    # 1. Extract codebook
    codebook = vqgan.quantize.embedding.weight.detach().cpu().numpy().astype(np.float32)
    n_embed, dim = codebook.shape
    # print(f"[Codebook] Codebook shape: {codebook.shape}")

    # 2. Compute distance matrix
    if metric == "cosine":
        # normalize
        codebook_norm = codebook / np.linalg.norm(codebook, axis=1, keepdims=True)
        dist = 1.0 - np.matmul(codebook_norm, codebook_norm.T)
    else:
        from sklearn.metrics import pairwise_distances

        dist = pairwise_distances(codebook, metric="euclidean")

    # 3. Sort neighbors (remove self)
    order = np.argsort(dist, axis=1)
    order = order[:, 1:]  # Remove self keep all else

    # 4. Save cache
    np.savez(cache_path, knn=order)
    print(f"[Codebook] Saved KNN cache to {cache_path}")

    return order


@torch.no_grad()
def swap_codewords(
    model,  # taming VQModel (eval mode)
    z_e,  # [B, D, Hq, Wq] from encoder/quant_conv
    indices,  # [B*Hq*Wq] or [Hq*Wq] flattened indices
    old_id: int,
    new_id: int,
):

    # infer shapes
    B, D, Hq, Wq = z_e.shape
    e_dim = model.quantize.e_dim
    n_embed = model.quantize.n_e
    assert 0 <= old_id < n_embed and 0 <= new_id < n_embed, "index out of range"

    # swap old → new
    device = indices.device
    idx_swapped = torch.where(
        indices == old_id,
        torch.as_tensor(new_id, device=device, dtype=indices.dtype),
        indices,
    )

    # rebuild z_q from edited indices
    z_q_edit = model.quantize.get_codebook_entry(
        idx_swapped, (B, Hq, Wq, e_dim)
    )  # -> [B, D, Hq, Wq]
    return z_q_edit, idx_swapped


def imgid_to_key(x):
    res = ""
    if isinstance(x, str):
        x = os.path.basename(x)  # Extract filename
        x = os.path.splitext(x)[0]  # Remove .jpg / .png
    res = f"{int(x):012d}.jpg"
    return res


@torch.no_grad()
def rewrite_sentence_pegasus(
    text: str,
    num_candidates: int = 5,
    max_new_tokens: int = 64,
    max_rounds: int = 5,
):
    inputs = rewrite_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)

    collected = []
    seen = set()

    for _ in range(max_rounds):
        need = num_candidates - len(collected)
        if need <= 0:
            break

        outputs = rewrite_model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=need,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )

        candidates = [
            rewrite_tokenizer.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

        for c in candidates:
            key = c.lower()
            if key != text.lower() and key not in seen:
                seen.add(key)
                collected.append(c)

    return collected


def select_perturbable_tokens(doc):
    tokens = []
    for token in doc:
        if (
            token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
            and token.is_alpha
            and not token.ent_type_
        ):
            tokens.append(token)
    return tokens


def clean_fasttext_substitutes(
    word: str,
    top_k: int = 3,
):
    try:
        neighbors = ft_model.get_nearest_neighbors(word, k=top_k * 200)
    except Exception:
        return []

    cleaned = []
    word_lower = word.lower()

    def too_similar(a, b):
        if a == b:
            return True
        if a.startswith(b) or b.startswith(a):
            return True
        if abs(len(a) - len(b)) <= 3:
            if a[:3] == b[:3]:
                return True
        if re.search(r"(.)\1\1", a):
            return True
        return False

    for _, w in neighbors:
        w_lower = w.lower()

        if not w_lower.isalpha():
            continue

        if len(w_lower) < 3:
            continue

        if too_similar(w_lower, word_lower):
            continue

        skip = False
        for prev in cleaned:
            if too_similar(w_lower, prev.lower()):
                skip = True
                break
        if skip:
            continue

        cleaned.append(w)

        if len(cleaned) >= top_k:
            break

    return cleaned


def lexical_cartesian_expand(
    rewrite_sentences,
    max_tokens_per_sentence=5,
    candidates_per_token=3,
):
    """
    Input:
        rewrite_sentences: List[str]
    Output:
        Dict[str, List[str]]  mapping each rewrite -> lexical variants
    """

    # ---------- Stage 1: collect global vocab ----------
    global_vocab = set()
    docs = {}

    for sent in rewrite_sentences:
        doc = nlp(sent)
        docs[sent] = doc
        for tok in doc:
            if (
                tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                and tok.is_alpha
                and not tok.ent_type_
            ):
                global_vocab.add(tok.text.lower())

    # ---------- Stage 2: build fastText cache ----------
    fasttext_cache = {}
    for word in global_vocab:
        subs = clean_fasttext_substitutes(word, top_k=candidates_per_token)
        if subs:
            fasttext_cache[word] = subs
    # ---------- Stage 3: per-sentence cartesian expansion ----------
    results = {}

    for sent, doc in docs.items():
        tokens = [tok for tok in doc if tok.text.lower() in fasttext_cache]

        if not tokens:
            results[sent] = []
            continue

        if len(tokens) > max_tokens_per_sentence:
            tokens = random.sample(tokens, max_tokens_per_sentence)

        positions = [tok.i for tok in tokens]
        subs_lists = [fasttext_cache[tok.text.lower()] for tok in tokens]

        variants = []
        for combo in itertools.product(*subs_lists):
            words = [t.text for t in doc]
            for idx, new_w in zip(positions, combo):
                words[idx] = new_w
            variant = " ".join(words)
            if variant.lower() != sent.lower():
                variants.append(variant)

        results[sent] = variants

    return results, fasttext_cache


def embedding_prune(
    seed_text: str,
    candidates: list,
    beta: int = 20,
):
    """
    Keep top-beta candidates closest to seed_text in embedding space.
    """

    if not candidates:
        return []

    texts = [seed_text] + candidates

    embeddings = sem_model.encode(
        texts, convert_to_tensor=True, normalize_embeddings=True
    )

    seed_emb = embeddings[0]
    cand_embs = embeddings[1:]

    # cosine distance = 1 - cosine similarity
    distances = 1 - util.cos_sim(seed_emb, cand_embs)[0]

    ranked = sorted(zip(candidates, distances.tolist()), key=lambda x: x[1])

    pruned = [t for t, _ in ranked[:beta]]
    return pruned


save_queue = Queue(maxsize=512)


def saver():
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.task_done()
            break
        img, path = item
        save_image(img, path)
        save_queue.task_done()


save_thread = Thread(target=saver)
save_thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUN THE PROBE CONFIGURATION")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASK_REGISTRY.keys(),
        help=(
            "Testing task:\n"
            "  itm  : Image-Text Matching (mscoco, CLIP vs BLIP)\n"
            "  vqa  : Visual Question Answering (vqav2, BLIP2 vs LXMERT)\n"
            "  nlvr : Visual Reasoning (nlvr2, ViLBERT vs LXMERT)"
        ),
    )
    parser.add_argument(
        "--perturb_mode",
        type=str,
        choices=["image", "text", "joint"],
        default="joint",
        help="Perturbation mode: image-only, text-only, or joint image-text perturbation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of test samples to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
        help="Output path for generated results",
    )
    parser.add_argument(
        "--pert_budget",
        type=int,
        default=1000,
        help="Maximum size of the perturbation candidate pool",
    )
    parser.add_argument(
        "--pict_top_p_ratio",
        type=float,
        default=0.1,
        help="Ratio of top-frequency codewords to perturb (e.g., 0.05, 0.1, 0.2)",
    )
    parser.add_argument(
        "--pict_pert_time",
        type=int,
        default=25,
        help="Number of perturbations per codeword (KNN depth)",
    )
    parser.add_argument(
        "--pict_pert_mode",
        type=str,
        default="knn",
        choices=["knn", "uniform", "kfn"],
        help=(
            "Perturbation mode based on semantic distance:\n"
            "  knn : nearest neighbors (front of KNN list)\n"
            "  uniform : uniform sampling\n"
            "  kfn : farthest neighbors (back of KNN list)"
        ),
    )
    parser.add_argument(
        "--text_variant_num",
        type=int,
        default=10,
        help="Number of text variants generated per original sentence",
    )
    parser.add_argument(
        "--text_max_word",
        type=int,
        default=50,
        help="Maximum number of word-level perturbation word",
    )
    parser.add_argument(
        "--text_max_word_attempts",
        type=int,
        default=50,
        help="Maximum number of word-level perturbation word attempts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=True,
        help="random seed for initialization",
    )

    # Parse arguments from the command line
    args = parser.parse_args()
    set_seed(args.seed)

    if args.perturb_mode == "image":
        do_image = True
        do_text = False
    elif args.perturb_mode == "text":
        do_image = False
        do_text = True
    else:
        do_image = True
        do_text = True

    PER_METHOD = args.perturb_mode
    MAX_IMAGES = args.max_samples
    OUTPUT_PATH = args.output_path
    PER_BUDGET = args.pert_budget
    PIC_R = args.pict_top_p_ratio
    PIC_T = args.pict_pert_time
    PIC_M = args.pict_pert_mode
    TEXT_N = args.text_variant_num
    TEXT_W = args.text_max_word
    TEXT_T = args.text_max_word_attempts

    # Build the task context
    task = TASK_REGISTRY[args.task]
    ctx = {
        "task": args.task,
        "task_name": task["name"],
        "dataset": task["dataset"],
        "dataset_path": os.path.join(DATA_ROOT, task["dataset"]),
        "models": task["models"],
    }

    print("\n====== Task Context ======")
    print(f" Task    : {ctx['task_name']}")
    print(f" Dataset : {ctx['dataset']}")
    print(f" Models  : {ctx['models'][0]}  vs  {ctx['models'][1]}")
    print("==========================\n")

    print("[VQGAN] Loaded checkpoint...")

    # VQGAN models load

    vqgan_cfg_path = "configs/coco_scene_images_transformer.yaml"
    vqgan_ckpt_path = "logs/coco_epoch117.ckpt"
    print(f"[VQGAN] Using COCO/OpenImages-8k-VQGAN for task `{ctx['task']}`")
    cfg = OmegaConf.load(vqgan_cfg_path)
    first_stage_cfg = cfg.model.params.first_stage_config
    first_stage_cfg.params.ckpt_path = vqgan_ckpt_path
    vqgan = instantiate_from_config(first_stage_cfg).eval().to(DEVICE)

    # VQGAN - Load checkpoint weights
    sd = torch.load(vqgan_ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = vqgan.load_state_dict(sd, strict=False)
    print("[VQGAN] Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    vqgan = vqgan.to(DEVICE).eval()

    # VQGAN - Build or load codebook KNN (top-100 for each codeword), 8192 for coco-vqgan-8k
    CODEWORD_KNN = load_or_build_codebook_knn(
        vqgan=vqgan, cache_name=f"coco_vqgan_8k_knn.npz", metric="cosine"
    )
    NUM_CODEWORDS = CODEWORD_KNN.shape[0]

    # print("[Codebook] KNN shape:", CODEWORD_KNN.shape)
    # Expected: (8192, 100)

    # TEXT perturbation model load
    TEXT_REWRITE_MODEL_NAME = "tuner007/pegasus_paraphrase"

    rewrite_tokenizer = AutoTokenizer.from_pretrained(
        TEXT_REWRITE_MODEL_NAME, use_fast=True
    )

    rewrite_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            TEXT_REWRITE_MODEL_NAME,
            use_safetensors=True,
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        .to(DEVICE)
        .eval()
    )

    # spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser"])

    # fastText
    FASTTEXT_MODEL_PATH = "cc.en.300.bin"
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    # BERT model
    SEM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    sem_model = SentenceTransformer(SEM_MODEL_NAME, device=DEVICE)

    # IMAGE Load
    IMAGE_DATA_PATH = os.path.join(DATA_ROOT, ctx["dataset"], "val")
    TEXT_DATA_PATH = None

    image_paths = sorted(
        glob(os.path.join(IMAGE_DATA_PATH, "**", "*.jpg"), recursive=True)
    )
    image_paths += sorted(
        glob(os.path.join(IMAGE_DATA_PATH, "**", "*.png"), recursive=True)
    )

    prep = T.Compose(
        [
            T.Resize(256, interpolation=Image.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
        ]
    )
    print(len(image_paths), "images found.")

    # TEXT Load
    caption_path = os.path.join(DATA_ROOT, task["dataset"], "image2captions.json")

    with open(caption_path, "r", encoding="utf-8") as f:
        caption_dict = json.load(f)

    # output folder Load
    dir_name = f"{OUTPUT_PATH}/{ctx['task']}-{PER_METHOD}-{PER_BUDGET}-{PIC_R}-{PIC_T}-{PIC_M}-{TEXT_N}-{TEXT_T}-seed{MAX_IMAGES}"
    os.makedirs(dir_name, exist_ok=True)
    event_image_path = os.path.join(dir_name, f"images.jsonl")
    event_image = open(event_image_path, "w", encoding="utf-8")
    event_text_path = os.path.join(dir_name, f"texts.jsonl")
    event_text = open(event_text_path, "w", encoding="utf-8")

    start_time = time.time()

    for count, img_path in enumerate(
        tqdm(list(image_paths)[:MAX_IMAGES], desc="Processing images")
    ):
        img_name = os.path.basename(img_path).split(".")[0]
        subdir_name = f"{dir_name}/{img_name}"
        os.makedirs(subdir_name, exist_ok=True)

        img_key = imgid_to_key(img_path)
        if img_key not in caption_dict:
            raise KeyError(f"Caption not found for image {img_key}")

        orig_caption = caption_dict[img_key]["question"]
        # print(orig_caption)

        # Preprocess image
        img = Image.open(img_path).convert("RGB")
        x_vqgan = preprocess(img)
        x = preprocess_vqgan(x_vqgan).to(DEVICE)
        z_q, _, [_, _, indices] = vqgan.encode(x)
        latent_grid = indices.reshape(16, 16)
        # print(latent_grid)

        # Save undoctored reconstruction
        orig_image = norm01(vqgan.decode(z_q))
        path = f"{subdir_name}/0_orig_image.png"
        save_image(orig_image, path)

        # fallback: only seed text
        orig_image_event = {
            "seed_id": count,
            "seed_name": img_name,
            "image_path": path,
            "ori_tag": True,
        }
        event_image.write(json.dumps(orig_image_event) + "\n")

        if do_text:
            rewritten_caption = rewrite_sentence_pegasus(
                orig_caption, num_candidates=TEXT_N
            )

            lexical_variants_map, word_per_dict = lexical_cartesian_expand(
                rewritten_caption,
                max_tokens_per_sentence=TEXT_W,
                candidates_per_token=TEXT_T,
            )

            all_text_candidates = []
            for r in rewritten_caption:
                all_text_candidates.append(r)
                all_text_candidates.extend(lexical_variants_map[r])

            text_pool = embedding_prune(
                seed_text=orig_caption, candidates=all_text_candidates, beta=PER_BUDGET
            )

            # print("Before pruning:", len(all_text_candidates))
            # print("After pruning:", len(text_pool))
            # print(text_pool[:10])

            text_event = {
                "seed_id": count,
                "seed_name": img_name,
                "seed_text": orig_caption,
                "rewritten_captions": rewritten_caption,  # List[str]
                "word_per_dict": word_per_dict,  # Dict[str, int]
                "text_pool": text_pool,  # List[str]
            }
            event_text.write(json.dumps(text_event) + "\n")
        else:
            # fallback: only seed text
            text_event = {
                "seed_id": count,
                "seed_name": img_name,
                "seed_text": orig_caption,
            }

            event_text.write(json.dumps(text_event) + "\n")

        if do_image:

            # High frequency sorting
            flat_indices = indices.view(-1).cpu().tolist()  # 16*16 = 256 tokens
            codeword_counter = Counter(flat_indices)
            codeword_list = codeword_counter.most_common()
            codeword_ranked = [cw for cw, cnt in codeword_list]

            # Take top p percent
            n = max(1, int(256 * PIC_R))
            # print(n)
            codeword_ranked = codeword_ranked[:n]
            # print(codeword_ranked)

            # Perturbation starts
            ori_latent = z_q.clone()
            ori_latent_ids = indices.clone()

            for i, old_id in enumerate(
                tqdm(codeword_ranked, desc="Processing codewords", leave=False)
            ):
                knn_new_ids = set()
                knn_list = CODEWORD_KNN[old_id]
                if PIC_M == "knn":
                    selected = knn_list[:PIC_T]

                elif PIC_M == "uniform":
                    total = len(knn_list)
                    indexs = np.linspace(0, total - 1, PIC_T, dtype=int)
                    selected = knn_list[indexs]

                elif PIC_M == "kfn":
                    selected = knn_list[-PIC_T:]

                else:
                    raise ValueError(f"Unknown PERT_MODE: {PIC_M}")

                # print(f"\n{old_id}: "+" ".join(map(str, selected)))

                for j in range(PIC_T):
                    cur_latent = ori_latent
                    cur_latent_ids = ori_latent_ids
                    new_close_id = selected[j]

                    cur_latent, cur_latent_ids = swap_codewords(
                        vqgan,
                        cur_latent,
                        cur_latent_ids,
                        old_id=old_id,
                        new_id=new_close_id,
                    )

                    rc = norm01(vqgan.decode(cur_latent))
                    pic_name = f"No{i+1}_{old_id}_{PIC_M}_pert_{j+1}_{new_close_id}"
                    path = f"{subdir_name}/{pic_name}.png"
                    save_queue.put((rc, path))

                    image_event = {
                        "seed_id": count,  # Cur seed index
                        "seed_name": img_name,  # Cur seed name
                        "image_path": path,  # Cur mutation save path
                        "perturb_index": j + 1,  # jth perturbation
                        "cur_codewword_index": i + 1,  # Cur codewword index
                        "src_codeword": int(old_id),  # Replaced codeword
                        "dst_codeword": int(new_close_id),  # Replacement codeword
                        "perturb_type": PIC_M,  # knn / kfn / uniform sampling
                    }
                    event_image.write(json.dumps(image_event) + "\n")

    end_time = time.time()

    # ===============================
    # Flush image saving queue
    # ===============================
    print("[Saver] Waiting for all images to be saved...")
    save_queue.join()  # Wait for all save_image to complete

    print("[Saver] Shutting down saver thread...")
    save_queue.put(None)  # Notify saver to exit
    save_thread.join()  # Wait for saver to fully terminate
    print("[Saver] Saver thread exited cleanly.")
    # ===============================

    event_image.close()
    event_text.close()

    time_taken = end_time - start_time
    time_per_seed = time_taken / MAX_IMAGES

    print(f"ALL seeds time cost: {time_taken:.4f}s")
    print(f"PER seed time cost: {time_per_seed:.4f}s")
    print("========================================")
    # ===============================
    # Save timing summary (minimal)
    # ===============================
    timing_summary = {
        "total_time_sec": round(time_taken, 4),
        "avg_time_per_seed_sec": round(time_per_seed, 4),
    }

    timing_path = os.path.join(dir_name, "timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"[Timing] Saved timing summary to {timing_path}")
