import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob
import math
import numpy as np
from collections import defaultdict
import torch
import torchvision.transforms as T
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
from transformers import XLMRobertaTokenizer
from beit3.modeling_finetune import beit3_large_patch16_224_nlvr2

SEED = 123456
PAIR_BUDGET = 1000
TEXT_POOL_BUDGET = 100
DEVICE = torch.device("cuda")
TEST_ROOT = "diff0"

BEIT3_TRANSFORM = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_distance(a, b, eps=1e-8):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)


def repl_root(path, old_root="outputs"):
    rel = os.path.relpath(path, old_root)
    return os.path.join(TEST_ROOT, rel)


def load_ref(seed_id):
    filename = f"{seed_id}.png"
    return os.path.join("data", "nlvr2", "ref_val_cache", filename)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def load_itm_models(device):
    from transformers import (
        ViltProcessor,
        ViltForImageAndTextRetrieval,
        BlipProcessor,
        BlipForImageTextRetrieval,
    )

    # ===== ViLT (ITM) =====
    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    vilt_model = (
        ViltForImageAndTextRetrieval.from_pretrained(
            "dandelin/vilt-b32-finetuned-coco", use_safetensors=True
        )
        .to(device)
        .eval()
    )

    # ===== BLIP (ITM) =====
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    blip_model = (
        BlipForImageTextRetrieval.from_pretrained(
            "Salesforce/blip-itm-base-coco", use_safetensors=True
        )
        .to(device)
        .eval()
    )

    return {
        "primary": {
            "name": "vilt",
            "model": vilt_model,
            "processor": vilt_processor,
        },
        "secondary": {
            "name": "blip",
            "model": blip_model,
            "processor": blip_processor,
        },
    }


def load_vqa_models(device):
    from transformers import (
        AutoProcessor,
        PaliGemmaForConditionalGeneration,
        Blip2Processor,
        Blip2ForConditionalGeneration,
    )

    # =========================
    # PaLI-Gemma (primary, VQA)
    # =========================
    paligemma_id = "google/paligemma-3b-ft-vqav2-224"

    paligemma_processor = AutoProcessor.from_pretrained(paligemma_id)

    paligemma_model = (
        PaliGemmaForConditionalGeneration.from_pretrained(
            paligemma_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )

    # =========================
    # BLIP-2 (secondary, VQA)
    # =========================
    blip2_id = "Salesforce/blip2-flan-t5-xl"

    blip2_processor = Blip2Processor.from_pretrained(blip2_id)

    blip2_model = (
        Blip2ForConditionalGeneration.from_pretrained(
            blip2_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )

    return {
        "primary": {
            "name": "paligemma",
            "model": paligemma_model,
            "processor": paligemma_processor,
        },
        "secondary": {
            "name": "blip2",
            "model": blip2_model,
            "processor": blip2_processor,
        },
    }


def load_beit3_nlvr2(device):
    """
    Load finetuned BEiT-3 Large for NLVR2 (discriminative)
    """
    # 1. build model structure
    model = beit3_large_patch16_224_nlvr2(pretrained=False)

    # 2. load finetuned checkpoint
    ckpt = torch.load("beit3_large_patch16_224_nlvr2.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    model.to(device).eval()

    # 3. tokenizer (must match training)
    tokenizer = XLMRobertaTokenizer("beit3.spm")

    return model, tokenizer


def load_vr_models(device):
    from transformers import (
        ViltProcessor,
        ViltForImagesAndTextClassification,
    )

    # ===== ViLT (primary, NLVR2) =====
    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
    vilt_model = (
        ViltForImagesAndTextClassification.from_pretrained(
            "dandelin/vilt-b32-finetuned-nlvr2",
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )

    # ===== BEiT-3 (secondary, NLVR2) =====
    beit3_model, beit3_tokenizer = load_beit3_nlvr2(device)

    return {
        "primary": {
            "name": "vilt",
            "model": vilt_model,
            "processor": vilt_processor,
        },
        "secondary": {
            "name": "beit3",
            "model": beit3_model,
            "tokenizer": beit3_tokenizer,
        },
    }


@torch.no_grad()
def run_itm_pair(image_path, text, models, device):
    img = Image.open(image_path).convert("RGB")

    # ===== ViLT =====
    vilt_inputs = models["primary"]["processor"](img, text, return_tensors="pt").to(
        device
    )

    vilt_outputs = models["primary"]["model"](**vilt_inputs)
    vilt_logits = vilt_outputs.logits

    # ViLT: logits shape could be [1] or [1,1]
    if vilt_logits.ndim == 2:
        vilt_logit = vilt_logits[0, 0].item()
    else:
        vilt_logit = vilt_logits[0].item()

    vilt_pred = int(vilt_logit > 0)

    # ===== BLIP =====
    blip_inputs = models["secondary"]["processor"](img, text, return_tensors="pt").to(
        device
    )

    blip_outputs = models["secondary"]["model"](**blip_inputs)
    blip_logits = blip_outputs.itm_score  # [1, 2]

    blip_pred = int(blip_logits.argmax(dim=-1).item())
    blip_match_logit = blip_logits[0, 1].item()

    return {
        "vilt_pred": vilt_pred,  # 0 / 1
        "blip_pred": blip_pred,  # 0 / 1
        "vilt_logit": float(vilt_logit),
        "blip_match_logit": float(blip_match_logit),
        "disagree": vilt_pred != blip_pred,
    }


@torch.no_grad()
def run_vqa_pair(image_path, question, models, device, sbert):
    img = Image.open(image_path).convert("RGB")

    # =========================
    # PaLI-Gemma (primary)
    # =========================
    prompt = f"<image> question: {question} short answer:"

    pg_inputs = models["primary"]["processor"](
        text=prompt, images=img, return_tensors="pt"
    ).to(device)

    input_len = pg_inputs["input_ids"].shape[-1]

    pg_gen = models["primary"]["model"].generate(
        **pg_inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
    )

    pg_gen = pg_gen[0][input_len:]

    paligemma_answer = (
        models["primary"]["processor"].decode(pg_gen, skip_special_tokens=True).strip()
    )

    # =========================
    # BLIP-2 (secondary)
    # =========================
    blip_prompt = f"Question: {question}\n" f"Answer with ONE word or ONE number only."

    blip_inputs = models["secondary"]["processor"](
        images=img, text=blip_prompt, return_tensors="pt"
    ).to(device)

    blip_gen = models["secondary"]["model"].generate(
        **blip_inputs,
        max_new_tokens=4,
        do_sample=False,
        num_beams=1,
    )

    blip2_answer = (
        models["secondary"]["processor"]
        .batch_decode(blip_gen, skip_special_tokens=True)[0]
        .strip()
    )

    # =========================
    # Semantic oracle
    # =========================
    emb1 = sbert.encode(paligemma_answer, normalize_embeddings=True)
    emb2 = sbert.encode(blip2_answer, normalize_embeddings=True)
    sim = float(np.dot(emb1, emb2))

    from vqa_normalizer_exact import VQANormalizerExact

    vqa_norm = VQANormalizerExact()

    a_norm = vqa_norm.normalize(paligemma_answer)
    b_norm = vqa_norm.normalize(blip2_answer)

    disagree = a_norm != b_norm

    return {
        "paligemma_answer": paligemma_answer,
        "blip2_answer": blip2_answer,
        "similarity": sim,
        "disagree": disagree,
    }


@torch.no_grad()
def run_nlvr2_pair(image1_path, image2_path, statement, models, device):
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # ======================
    # ViLT (primary)
    # ======================
    vilt_processor = models["primary"]["processor"]
    vilt_model = models["primary"]["model"]

    encoding = vilt_processor(
        [img1, img2], statement, return_tensors="pt", truncation=True, max_length=40
    )

    vilt_outputs = vilt_model(
        input_ids=encoding.input_ids.to(device),
        pixel_values=encoding.pixel_values.unsqueeze(0).to(device),
    )

    vilt_logits = vilt_outputs.logits  # [1, 2]
    vilt_true_logit = vilt_logits[0, 1].item()
    vilt_pred = int(vilt_logits.argmax(dim=-1).item())  # 0 / 1

    # ======================
    # BEiT-3 (secondary)
    # ======================
    beit3 = models["secondary"]["model"]
    tokenizer = models["secondary"]["tokenizer"]

    img1_t = BEIT3_TRANSFORM(img1).unsqueeze(0).to(device)
    img2_t = BEIT3_TRANSFORM(img2).unsqueeze(0).to(device)

    enc = tokenizer(
        statement,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=40,
    )

    beit3_logits = beit3(
        img1_t,
        img2_t,
        enc["input_ids"].to(device),
        enc["attention_mask"].eq(0).to(device),  # padding_mask
    )  # [1, 2]

    beit3_true_logit = beit3_logits[0, 1].item()
    beit3_pred = int(beit3_logits.argmax(dim=-1).item())

    # ======================
    # Return (ITM-style)
    # ======================
    return {
        "vilt_pred": vilt_pred,  # 0 / 1
        "beit3_pred": beit3_pred,  # 0 / 1
        "vilt_logit": float(vilt_true_logit),  # True-class logit
        "beit3_logit": float(beit3_true_logit),  # True-class logit
        "disagree": vilt_pred != beit3_pred,
    }


def compute_dfb_threshold(failure_outputs, rho):
    """
    failure_outputs: List[np.ndarray], each is o(x)
    rho: distance threshold ρ
    return: DFB (int)
    """

    n = len(failure_outputs)
    if n == 0:
        return 0

    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(failure_outputs[i], failure_outputs[j])
            if d <= rho:
                uf.union(i, j)

    # count distinct clusters
    roots = set(uf.find(i) for i in range(n))
    return len(roots)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


MODEL_FACTORY = {
    "itm": load_itm_models,
    "vqa": load_vqa_models,
    "nlvr": load_vr_models,
}

if __name__ == "__main__":
    set_seed(123456)
    test_root = TEST_ROOT

    images_events = glob.glob(os.path.join(test_root, "*", "images.jsonl"))
    texts_events = glob.glob(os.path.join(test_root, "*", "texts.jsonl"))
    taming_events = glob.glob(os.path.join(test_root, "*", "timing.json"))

    if len(images_events) != len(texts_events):
        raise ValueError(
            f"Paired events mismatch: "
            f"{len(images_events)} images.jsonl vs {len(texts_events)} texts.jsonl"
        )

    images_events = sorted(images_events)
    texts_events = sorted(texts_events)
    taming_events = sorted(taming_events)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sbert = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    loaded_models = {}

    for exp_id in range(len(images_events)):
        images_event = images_events[exp_id]
        texts_event = texts_events[exp_id]
        taming_event = taming_events[exp_id]

        exp_name = os.path.basename(os.path.dirname(images_event))

        parts = exp_name.split("-")

        perturb_task = parts[0]  # itm
        perturb_mode = parts[1]  # image

        # Load model
        if perturb_task not in loaded_models:
            print(f"[Load Models] task = {perturb_task}")
            loaded_models[perturb_task] = MODEL_FACTORY[perturb_task](DEVICE)

        models = loaded_models[perturb_task]

        print(f"\n=== Processing experiment {exp_name} ===")

        with open(images_event, "r", encoding="utf-8") as f:
            image_records = [json.loads(line) for line in f if line.strip()]
        with open(texts_event, "r", encoding="utf-8") as f:
            text_records = [json.loads(line) for line in f if line.strip()]
        with open(taming_event, "r", encoding="utf-8") as f:
            taming = json.load(f)

        # ========== Group by seed ==========
        images_by_seed = defaultdict(list)
        for r in image_records:
            images_by_seed[r["seed_id"]].append(r)

        texts_by_seed = {r["seed_id"]: r for r in text_records}

        seed_diff_stats = []
        baseline_sims = []
        all_seed_num = len(texts_by_seed)
        ori_seed_fail = 0
        failure_outputs = []

        for seed_id, records in tqdm(
            images_by_seed.items(), desc="Processing seeds", total=len(images_by_seed)
        ):
            # prepare image
            ori_record = None
            pert_image_paths = []

            for r in records:
                if r.get("ori_tag") is True:
                    ori_record = r
                else:
                    pert_image_paths.append(r)

            # prepare text
            ori_text = texts_by_seed[seed_id]["seed_text"]
            if perturb_mode == "image":
                text_pool = [ori_text]
            else:
                pert_texts = texts_by_seed[seed_id]["text_pool"]
                text_pool = pert_texts

            # Load pools
            if perturb_mode == "text":
                image_pool = [ori_record]  # Only orig image
                text_pool = pert_texts

            elif perturb_mode == "image":
                image_pool = pert_image_paths
                text_pool = [ori_text]  # Only orig image

            else:  # joint / both
                image_pool = pert_image_paths
                text_pool = pert_texts

            n_img = len(image_pool)
            n_txt = min(TEXT_POOL_BUDGET, len(text_pool))
            text_pool = text_pool[:n_txt]
            # print(f"[Pool Size] image: {n_img}, text: {len(text_pool)}")

            pairs = []

            # RR Sampling

            # Step 1: image coverage
            for i in range(n_img):
                j = i % n_txt
                pairs.append((image_pool[i], text_pool[j]))

            # Step 2: text coverage
            used_text_ids = {i % n_txt for i in range(n_img)}
            for j in range(n_txt):
                if j not in used_text_ids:
                    i = j % n_img
                    pairs.append((image_pool[i], text_pool[j]))

            # Step 3: fill remaining (safe, finite)
            max_pairs = min(PAIR_BUDGET, n_img * n_txt)
            for i in range(n_img):
                for j in range(n_txt):
                    if len(pairs) >= max_pairs:
                        break
                    pair = (image_pool[i], text_pool[j])
                    if pair not in pairs:
                        pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break

            random.shuffle(pairs)

            results = []
            for img_rec, text in tqdm(pairs, desc="Processing pairs", total=len(pairs)):
                image_path = repl_root(img_rec["image_path"])

                if perturb_task == "itm":
                    baseline_out = run_itm_pair(
                        repl_root(ori_record["image_path"]), ori_text, models, device
                    )
                    if baseline_out["disagree"]:
                        # print("[Skip]")
                        break
                    out = run_itm_pair(image_path, text, models, device)
                    if out["disagree"]:
                        # output difference representation
                        o = np.array(
                            [
                                out.get("vilt_logit", 0.0)
                                - out.get("blip_match_logit", 0.0)
                            ]
                        )
                        failure_outputs.append(o)

                elif perturb_task == "vqa":
                    baseline_out = run_vqa_pair(
                        repl_root(ori_record["image_path"]),
                        ori_text,
                        models,
                        device,
                        sbert,
                    )
                    baseline_sims.append(baseline_out["similarity"])
                    if baseline_out["disagree"]:
                        print(baseline_out)
                        # print("[Skip]")
                        break
                    out = run_vqa_pair(image_path, text, models, device, sbert)
                    if out["disagree"]:
                        emb1 = sbert.encode(
                            out["paligemma_answer"], normalize_embeddings=True
                        )
                        emb2 = sbert.encode(
                            out["blip2_answer"], normalize_embeddings=True
                        )
                        o = emb1 - emb2  # output difference representation
                        failure_outputs.append(o)

                elif perturb_task == "nlvr":
                    baseline_out = run_nlvr2_pair(
                        repl_root(ori_record["image_path"]),
                        load_ref(ori_record["seed_name"]),
                        ori_text,
                        models,
                        device,
                    )
                    if baseline_out["disagree"]:
                        # print("[Skip]")
                        break
                    out = run_nlvr2_pair(
                        image_path, load_ref(img_rec["seed_name"]), text, models, device
                    )
                    if out["disagree"]:
                        # output difference representation
                        o = np.array(
                            [out.get("vilt_logit", 0.0) - out.get("beit3_logit", 0.0)]
                        )
                        failure_outputs.append(o)

                else:
                    raise ValueError(f"Unknown task: {perturb_task}")

                out.update(
                    {
                        "seed_id": seed_id,
                        "image": image_path,
                        "text": text,
                    }
                )

                results.append(out)

            # ===== quick sanity statistics =====
            if len(results) != 0:
                num_pairs = len(results)
                num_disagree = sum(r["disagree"] for r in results)
                seed_diff_rate = num_disagree / num_pairs * 100
                seed_diff_stats.append(
                    {
                        "num_pairs": num_pairs,
                        "num_disagree": num_disagree,
                        "diff_rate": seed_diff_rate,
                    }
                )
            else:
                ori_seed_fail += 1

        ttb = taming["avg_time_per_seed_sec"]

        ratio = (
            (all_seed_num - ori_seed_fail) / all_seed_num * 100
            if all_seed_num > 0
            else 0.0
        )

        total_disagree = sum(s["num_disagree"] for s in seed_diff_stats)
        total_pairs = sum(s["num_pairs"] for s in seed_diff_stats)

        per_seed_num_disagree = [s["num_disagree"] for s in seed_diff_stats]
        mean_num_disagree_per_seed = (
            np.mean(per_seed_num_disagree) if per_seed_num_disagree else 0.0
        )

        avg_diff_rate = (
            np.mean([s["diff_rate"] for s in seed_diff_stats])
            if seed_diff_stats
            else 0.0
        )
        dfb_results = {
            rho: compute_dfb_threshold(failure_outputs, rho) for rho in [0.5]
        }

        tpf = ttb / avg_diff_rate if avg_diff_rate > 0 else float("inf")

        print(f"Seeds Num={all_seed_num}")
        print(f"ValidSeeds={all_seed_num - ori_seed_fail} ({ratio:.2f}%)")

        print(f"FRI(avg)={avg_diff_rate:.2f}%")
        print(
            f"FRI(global)={mean_num_disagree_per_seed:.2f}% "
            f"(disagree={total_disagree}, pairs={total_pairs})"
        )

        print("DFB=" + ", ".join(f"{rho}:{dfb}" for rho, dfb in dfb_results.items()))
        print(f"TTB={ttb:.2f}s")
        print(f"TPF={tpf:.2f}s")

        # ===== save per-experiment json =====
        exp_metrics = {
            "exp_name": exp_name,
            "Seeds Num": all_seed_num,
            "ValidSeeds": {
                "count": all_seed_num - ori_seed_fail,
                "ratio_percent": round(ratio, 2),
            },
            # === Failure Rate Indicators ===
            "FRI": {
                "avg_percent": round(avg_diff_rate, 2),
                "mean_num_disagree_per_seed*": round(mean_num_disagree_per_seed, 2),
                "total_disagree": total_disagree,
                "total_pairs": total_pairs,
            },
            "DFB*": {str(rho): dfb_results[rho] for rho in dfb_results},
            "TTB_sec*": round(ttb, 2),
            "TPF_sec*": round(tpf, 2),
        }
        json_path = f"{exp_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(exp_metrics, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {json_path}")
