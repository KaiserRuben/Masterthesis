# Third-party material

This repository redistributes, or instructs you to fetch, material that is not
covered by [LICENSE](LICENSE) or [LICENSE-DATA](LICENSE-DATA) and that the
copyright holder of this repository cannot relicense. Each item below keeps its
own terms. Where those terms are more restrictive than this repository's
licenses, the more restrictive terms win for that material.

## Redistributed in this repository

### Reference photographs — `experiments/HS-01/app/config/refs/` (33 PNG files)

Photographs of ImageNet classes, curated by
`experiments/HS-01/tools/build_references.py` from an ImageNet image cache and
used as reference stimuli in the HS-01 rating study.

ImageNet does not own the copyright in these photographs; they were collected
from the web and are made available under terms permitting **non-commercial
research and educational use only**. They are included here so the HS-01 study
application and its figures can be reproduced, and are not licensed by the
copyright holder of this repository. Do not redistribute them outside that
purpose. See <https://www.image-net.org/download.php> for the current terms.

The same applies to the VQGAN-manipulated imagery under `samples/output/` and
`samples/output_aesthetic/`, which is derived from ImageNet photographs. (Those
directories are pruned from the `repro/thesis-v1` branch.)

### SMOO — `tools/smoo` (git submodule)

Tracks `KaiserRuben/SMOO` branch `masterarbeit`, a fork of
`oliverweissl/SMOO` carrying three commits: packaging metadata, timm 1.0
compatibility for the StyleGAN-XL pickles, and `inference_mode` on the
manipulator entry points.

> **Unresolved:** neither the fork nor upstream `oliverweissl/SMOO` carries a
> license file. Absent an express license, the upstream author retains all
> rights, and this repository has no terms to pass on to you for that code. The
> three fork commits above are the only part licensed under
> [LICENSE](LICENSE). Anyone reproducing this work should treat the submodule
> as all-rights-reserved upstream code until a license is added there.

### VLTest — `tools/Archive/VLTest/`

MIT License, Copyright (c) 2026 Alin. Full text at
[`tools/Archive/VLTest/LICENSE`](tools/Archive/VLTest/LICENSE). Retained for
reference only; not used by either pipeline. Pruned from the
`repro/thesis-v1` branch.

### Alpamayo side research — `archive_alpamayo_jan2026/`, `infrastructure/`

An unrelated January 2026 research thread on VLM referring-expression
grounding. Its notebooks reference the RefCOCO/COCO datasets and NVIDIA
Alpamayo-R1, each under its own terms; consult those sources directly. Pruned
from the `repro/thesis-v1` branch.

Note that `tools/alpamayo/` is a separate git repository present on disk but
gitignored — it is not part of this distribution.

## Fetched at runtime, not redistributed

Model weights are downloaded when a pipeline runs; none are stored in this
repository. Their licenses are set by their publishers and can change — check
the model card or repository before relying on any of them, especially for
anything beyond noncommercial research.

| Model | Used for | Terms |
|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | SUT (VLM under test) | Qwen model card |
| `OpenVINO/Qwen2.5-VL-7B-Instruct-int{4,8}-ov` | quantized SUT | inherits Qwen terms |
| `OpenVINO/llava-v1.6-mistral-7b-hf-int{4,8}-ov` | quantized SUT | LLaVA / Mistral terms |
| `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` | auxiliary scoring | Qwen model card |
| `answerdotai/ModernBERT-large` | MLM-Synonym text manipulator | model card |
| `sentence-transformers/all-MiniLM-L6-v2` | embedding distances | model card |
| `dalle-mini/vqgan_imagenet_f16_16384`, `thomwolf/vqgan_imagenet_f16_1024` | VQGAN image manipulator | model card |
| StyleGAN-XL `imagenet256.pkl` | image-space probes | **NVIDIA Source Code License-NC — noncommercial only** |
| spaCy `en_core_web_sm` | tokenization | spaCy model terms |

The StyleGAN-XL checkpoint derives from NVIDIA StyleGAN3 code, released under a
license restricting use to non-commercial purposes. Any pipeline path that
touches it is bound by that restriction independently of this repository's
licenses.

Python dependencies installed from PyPI via `experiments/requirements.txt` and
`experiments/requirements-openvino.txt` are not redistributed here and carry
their own licenses.

## Reporting a problem

If you hold rights in any material listed here and believe it is
inappropriately included, contact Ruben.Kaiser@tum.de and it will be removed.
