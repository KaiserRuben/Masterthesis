"""Throwaway OpenVINO VLM smoke test on Intel Arc A770.

Loads each model via optimum-intel with INT4 weight compression, runs one
image+prompt, prints latency + output. Persists the OV-IR conversion under
``$HF_HOME/ov_ir/<safe_name>/`` so re-runs skip conversion.

Not part of the SUT pipeline. Will be deleted once the OV scorer is wired
into ``src/sut/scorer.py``.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

# HF cache lives on the bulk volume.
os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/storage/huggingface/hub")

from PIL import Image
from transformers import AutoProcessor

from optimum.intel import OVModelForVisualCausalLM


@dataclass
class ModelSpec:
    name: str            # short label
    ir_id: str           # HF repo holding the OV IR (skip local conversion)
    processor_id: str    # HF repo for the processor (often same as original FP16)
    family: str          # "llava_next" | "qwen2_5_vl"


# Pre-converted OV IRs from the OpenVINO HF org. Skips the broken in-process
# export path (Qwen2.5-VL fails to trace under transformers 4.57).
MODELS: list[ModelSpec] = [
    ModelSpec(
        name="LLaVA-NeXT-Mistral-7B",
        ir_id="OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov",
        processor_id="llava-hf/llava-v1.6-mistral-7b-hf",
        family="llava_next",
    ),
    ModelSpec(
        name="Qwen2.5-VL-7B-Instruct",
        ir_id="OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov",
        processor_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen2_5_vl",
    ),
]

DEVICE = "GPU"  # OpenVINO label for Arc A770
MAX_NEW_TOKENS = 64
PROMPT = "Describe this image in one short sentence."


def build_messages(spec: ModelSpec, image: Image.Image, prompt: str) -> list[dict]:
    """Both LLaVA-NeXT and Qwen2.5-VL accept the same chat-template shape."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_ir(spec: ModelSpec) -> tuple[OVModelForVisualCausalLM, AutoProcessor, float]:
    """Load a pre-converted OpenVINO IR straight from HF (no local export)."""
    t0 = time.perf_counter()
    print(f"  [load]    {spec.ir_id} on {DEVICE}")
    model = OVModelForVisualCausalLM.from_pretrained(spec.ir_id, device=DEVICE)
    processor = AutoProcessor.from_pretrained(spec.processor_id)
    return model, processor, time.perf_counter() - t0


def run_one(spec: ModelSpec, image: Image.Image) -> dict:
    print(f"\n=== {spec.name} ({spec.ir_id}) ===")
    model, processor, t_load = load_ir(spec)

    msgs = build_messages(spec, image, PROMPT)
    text = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    # Warmup: first generate() pass triggers OV compilation on GPU.
    t0 = time.perf_counter()
    _ = model.generate(**inputs, max_new_tokens=4)
    t_warmup = time.perf_counter() - t0

    t0 = time.perf_counter()
    gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    t_gen = time.perf_counter() - t0

    prefix_len = inputs["input_ids"].shape[1]
    new_tokens = gen[0, prefix_len:]
    answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "name": spec.name,
        "load_s": t_load,
        "warmup_s": t_warmup,
        "gen_s": t_gen,
        "n_new_tokens": int(new_tokens.shape[0]),
        "tok_per_s": float(new_tokens.shape[0]) / max(t_gen, 1e-6),
        "answer": answer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default="/home/kaiser/Projects/Masterarbeit/runs/Exp-09/"
                "exp09_M0_n16383_shark_seed_5_1776512034/pareto_2.png",
    )
    ap.add_argument("--only", choices=[m.name for m in MODELS], default=None,
                    help="Run a single model by short name.")
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    print(f"image: {args.image}  size={image.size}")
    print(f"prompt: {PROMPT!r}")
    print(f"device: {DEVICE} (Intel Arc A770)")
    print(f"HF_HOME: {os.environ['HF_HOME']}")

    results = []
    for spec in MODELS:
        if args.only and spec.name != args.only:
            continue
        try:
            results.append(run_one(spec, image))
        except Exception as e:
            print(f"  [error] {spec.name}: {type(e).__name__}: {e}")
            results.append({"name": spec.name, "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"{r['name']:>26}: ERROR: {r['error']}")
            continue
        print(
            f"{r['name']:>26}: load={r['load_s']:6.1f}s  "
            f"warmup={r['warmup_s']:5.1f}s  "
            f"gen({r['n_new_tokens']:>3}tok)={r['gen_s']:5.2f}s  "
            f"={r['tok_per_s']:5.1f} tok/s"
        )
        print(f"{'':>26}  -> {r['answer']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
