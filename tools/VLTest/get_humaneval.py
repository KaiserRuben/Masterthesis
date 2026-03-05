import json
import os
import glob
import math
import numpy as np
from collections import defaultdict
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import random
import shutil

SEED = 123456
PAIR_BUDGET = 5
DEVICE = torch.device("cuda")
SEED_IMG_DIR = "./images_src"
os.makedirs(SEED_IMG_DIR, exist_ok=True)
RESULT_JSONL = "selected_samples.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(123456)
    test_root = "./diff"

    images_events = glob.glob(os.path.join(test_root, "*", "images.jsonl"))
    texts_events = glob.glob(os.path.join(test_root, "*", "texts.jsonl"))

    if len(images_events) != len(texts_events):
        raise ValueError(
            f"Paired events mismatch: "
            f"{len(images_events)} images.jsonl vs {len(texts_events)} texts.jsonl"
        )

    images_events = sorted(images_events)
    texts_events = sorted(texts_events)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uni_index = 1
    all_records = []
    out_f = open(RESULT_JSONL, "w", encoding="utf-8")
    for exp_id in tqdm(range(len(images_events)), desc="Processing experiments"):
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

        for seed_id, records in images_by_seed.items():
            # prepare image
            ori_path = None
            pert_image_paths = []

            for r in records:
                if r.get("ori_tag") is True:
                    ori_path = r["image_path"]
                else:
                    pert_image_paths.append(r["image_path"])
            image_pool = pert_image_paths

            ori_img_dst = os.path.join(
                SEED_IMG_DIR, f"{perturb_task}_ori_{seed_id}.png"
            )
            rel_path = os.path.relpath(ori_path, start="outputs")
            new_path = os.path.join(test_root, rel_path)
            shutil.copy(new_path, ori_img_dst)

            if perturb_task == "nlvr":
                target = seed_id + 1
                ref_src = os.path.join(
                    "data", "nlvr2", "ref_val_cache", f"{target:012d}.png"
                )

                ref_dst = os.path.join(
                    SEED_IMG_DIR, f"{perturb_task}_ori_{seed_id}_ref.png"
                )

                if os.path.exists(ref_src):
                    if not os.path.exists(ref_dst):
                        shutil.copy(ref_src, ref_dst)
                else:
                    print(f"[WARN] NLVR2 ref image not found: {ref_src}")

            # prepare text
            ori_text = texts_by_seed[seed_id]["seed_text"]
            if perturb_mode == "image":
                text_pool = [ori_text]
            else:
                pert_texts = texts_by_seed[seed_id]["text_pool"]
                text_pool = pert_texts

            n_img = len(image_pool)
            n_txt = len(text_pool)

            # print(f"[Pool Size] image: {n_img}, text: {n_txt}")

            all_pairs = [(i, j) for i in range(n_img) for j in range(n_txt)]

            sampled = random.sample(all_pairs, k=min(PAIR_BUDGET, len(all_pairs)))

            pairs = [(image_pool[i], text_pool[j]) for i, j in sampled]
            # print(pairs)

            for img_rec, adv_text in pairs:
                adv_image_path = img_rec
                adv_img_dst = os.path.join(SEED_IMG_DIR, f"adv_{uni_index:012d}.png")
                rel_path = os.path.relpath(adv_image_path, start="outputs")
                new_path = os.path.join(test_root, rel_path)
                shutil.copy(new_path, adv_img_dst)

                record = {
                    "Index": uni_index,
                    "Task": perturb_task,
                    "method": perturb_mode,
                    "orig_image": f"{perturb_task}_ori_{seed_id}.png",
                    "orig_text": ori_text,
                    "adv_image": f"adv_{uni_index:012d}.png",
                    "adv_text": adv_text,
                }
                if perturb_task == "nlvr":
                    record["ref_image"] = f"{perturb_task}_ori_{seed_id}_ref.png"

                all_records.append(record)

                uni_index += 1

    with open(RESULT_JSONL, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
