import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import math
import numpy as np
from collections import defaultdict
import torch
import torchvision.transforms as T
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CODEBOOK_SIZE = 8192
MODEL = "image"
RESULT_JSON_PATH = "summary_metrics.json"


# ========== Utility functions ==========


def safe_round(x, ndigits=4):
    if x is None:
        return None
    return round(float(x), ndigits)


def cosine_dist(a, b, eps=1e-8):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)


# ========== Metric 1 Latent Space Coverage ==========


def compute_lsc(image_records):
    """
    image_records: list of dicts (same seed)
    """
    new_ids = set()
    for r in image_records:
        if "dst_codeword" not in r:
            continue
        cw = r["dst_codeword"]
        if isinstance(cw, list):
            new_ids.update(cw)  # Multiple codewords
        else:
            new_ids.add(cw)  # Single codeword int
    # print(len(new_ids))
    return len(new_ids) / CODEBOOK_SIZE


# ========== Metric 2：Feature Coverage & Concentration ==========


def compute_fc_cc(features, bins):
    """
    features: [N, d]
    bins: list of d arrays, each of size B+1
    """
    N, d = features.shape
    B = len(bins[0]) - 1

    activated = np.zeros((d, B))
    counts = np.zeros((d, B))

    for feat in features:
        for k in range(d):
            b = np.digitize(feat[k], bins[k]) - 1
            b = max(0, min(B - 1, b))
            activated[k, b] = 1
            counts[k, b] += 1

    # FC
    FC = activated.sum() / (d * B)

    # CC
    CCs = []
    for k in range(d):
        pk = counts[k] / max(counts[k].sum(), 1)
        H = -np.sum([p * math.log(p) for p in pk if p > 0])
        CCs.append(1 - H / math.log(B))

    CC = np.mean(CCs)
    return FC, CC


# ========== Metric 3：Textual Diversity ==========


def compute_textual_diversity(seed_text_emb, pool_embs):
    """
    seed_text_emb: [d]
    pool_embs: [m, d]
    """
    dists = np.array([cosine_dist(seed_text_emb, e) for e in pool_embs])
    SDev = dists.mean()
    SDisp = dists.std()
    return SDev, SDisp


def load_dino(device):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_features_for_seed(image_paths, model, transform, device):
    feats = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        x = transform(img).unsqueeze(0).to(device)
        f = model(x).squeeze(0).cpu().numpy()
        feats.append(f)

    if len(feats) == 0:
        return None

    return np.stack(feats)  # [N_seed, d]


def build_transform():
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# ========== main ==========

if __name__ == "__main__":
    output_root = "./outputs"

    images_events = glob.glob(os.path.join(output_root, "*", "images.jsonl"))
    texts_events = glob.glob(os.path.join(output_root, "*", "texts.jsonl"))

    if len(images_events) != len(texts_events):
        raise ValueError(
            f"Paired events mismatch: "
            f"{len(images_events)} images.jsonl vs {len(texts_events)} texts.jsonl"
        )

    images_events = sorted(images_events)
    texts_events = sorted(texts_events)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sent_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    sent_encoder.eval()

    bins = np.load("data/coco/features/bins_dino_B10.npz", allow_pickle=True)["bins"]

    dino = load_dino(device)
    transform = build_transform()

    all_results = []

    for exp_id in tqdm(range(len(images_events)), desc="Processing experiments"):
        avg_lsc = None
        fc = None
        cc = None
        avg_SDev = None
        avg_SDisp = None
        images_event = images_events[exp_id]
        texts_event = texts_events[exp_id]
        print(images_event)
        exp_name = os.path.basename(os.path.dirname(images_event))

        parts = exp_name.split("-")

        perturb_task = parts[0]  # itm
        perturb_mode = parts[1]  # image

        print(f"\n=== Processing experiment {exp_name} ===")

        with open(images_event, "r", encoding="utf-8") as f:
            image_records = [json.loads(line) for line in f if line.strip()]
        with open(texts_event, "r", encoding="utf-8") as f:
            text_records = [json.loads(line) for line in f if line.strip()]

        # ========== Group by seed ==========
        images_by_seed = defaultdict(list)
        for r in image_records:
            images_by_seed[r["seed_id"]].append(r)

        texts_by_seed = {r["seed_id"]: r for r in text_records}

        if perturb_mode != "text":
            # ========== 1. Latent Space Coverage ==========
            lsc_list = []
            for seed_id, records in images_by_seed.items():
                lsc = compute_lsc(records)
                lsc_list.append(lsc)
                # print(lsc)

            avg_lsc = float(np.mean(lsc_list))
            # print(avg_lsc)

            # ========== 2. Feature Diversity ==========
            FC_list = []
            CC_list = []

            for seed_id, records in images_by_seed.items():
                image_paths = [
                    r["image_path"]
                    for r in records
                    if os.path.exists(r["image_path"]) and not r.get("ori_tag", False)
                ]

                if len(image_paths) == 0:
                    continue

                feats = extract_features_for_seed(image_paths, dino, transform, device)

                if feats is None:
                    continue

                FC_seed, CC_seed = compute_fc_cc(feats, bins)
                FC_list.append(FC_seed)
                CC_list.append(CC_seed)

            fc = float(np.mean(FC_list)) if len(FC_list) > 0 else None
            cc = float(np.mean(CC_list)) if len(CC_list) > 0 else None

            # print(FC,CC)
        if perturb_mode != "image":
            # ========== 3. Textual Diversity ==========
            SDev_list, SDisp_list = [], []

            for seed_id, text_obj in texts_by_seed.items():
                seed_text = text_obj["seed_text"]
                text_pool = text_obj["text_pool"]

                if len(text_pool) == 0:
                    continue

                # ---- sentence embedding ----
                # g(t0)
                seed_emb = sent_encoder.encode(
                    seed_text, convert_to_numpy=True, normalize_embeddings=True
                )  # [d]

                # g(tj)
                pool_embs = sent_encoder.encode(
                    text_pool,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )  # [m, d]

                # ---- compute SDev & SDisp ----
                SDev, SDisp = compute_textual_diversity(seed_emb, pool_embs)

                SDev_list.append(SDev)
                SDisp_list.append(SDisp)

            # ---- aggregate over seeds ----
            avg_SDev = float(np.mean(SDev_list)) if len(SDev_list) > 0 else None
            avg_SDisp = float(np.mean(SDisp_list)) if len(SDisp_list) > 0 else None

        result = {
            "experiment": os.path.basename(os.path.dirname(images_event)),
            "LSC": safe_round(avg_lsc),
            "FC": safe_round(fc),
            "CC": safe_round(cc),
            "SDev": safe_round(avg_SDev),
            "SDisp": safe_round(avg_SDisp),
        }

        all_results.append(result)

        # ---- incremental save (safe) ----
        with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary metrics to {RESULT_JSON_PATH}")

    # ========== SUMMRY ==========
    print("\n=== Final Results ===")
    for r in all_results:
        print(json.dumps(r, indent=2))
