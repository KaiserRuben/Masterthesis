# VLTest
VLTest is a black-box testing framework for vision–language models that systematically generates semantics-preserving test cases. It explores discrete visual latent spaces and bounded textual neighborhoods to uncover behavioral inconsistencies across VLM tasks, datasets, and architectures.

![License](https://img.shields.io/badge/License-MIT-green.svg)

- [Overview](#overview)
    - [Mutation-Based Testing Approach](#mutation-based-testing-approach)
    - [Folder Structure](#folder-structure)
    - [Dataset and Models](#dataset-and-models)
    - [Models and configs/logs download](#models-and-configslogs-download)
- [Experiments](#experiments)
    - [Environment Configuration](#environment-configuration)
    - [Data Preprocessing](#data-preprocessing)
    - [Test Case Generation Pipeline](#test-case-generation-pipeline)
    - [RQ1: Exploration Capability Analysis](#rq1-exploration-capability-analysis)
    - [RQ2: Semantic Validity (Human Evaluation)](#rq2-semantic-validity-human-evaluation)
    - [RQ3: Failure Discovery Effectiveness and Efficiency](#rq3-failure-discovery-effectiveness-and-efficiency)


# Overview

## Folder Structure
```
├── README.md
├── VLTest
    ├─beit3
    │  └─get_started
    ├─configs
    ├─data
    │  ├─coco
    │  │  ├─annotations
    │  │  ├─features
    │  │  ├─train
    │  │  └─val
    │  ├─nlvr2
    │  │  ├─annotations
    │  │  ├─features
    │  │  ├─ref_val_cache
    │  │  ├─train
    │  │  │  ├─images_left
    │  │  │  └─images_right
    │  │  └─val
    │  │      ├─images_left
    │  │      └─images_right
    │  └─vqav2
    │      ├─annotations
    │      ├─features
    │      ├─train
    │      └─val
    ├── environment.yml
    ├─humaneval
    └─taming
        ├─data
        │  └─conditional_builder
        ├─models
        └─modules
```

## Dataset and Models
We evaluate VLTest on three representative vision–language tasks, covering image–text matching, visual reasoning, and visual question answering. All models under test are publicly available, while VQGAN is used solely as a visual generator to support discrete latent-space mutation and is not treated as a model under test.
[VQGAN](https://github.com/CompVis/taming-transformers)
| Task | Dataset | Models |
|------|---------|-------|
| Image–Text Matching | [MSCOCO](https://cocodataset.org/#home) | [ViLT_COCO](https://huggingface.co/dandelin/vilt-b32-finetuned-coco) & [BLIP](https://huggingface.co/Salesforce/blip-itm-base-coco)|
| Visual Reasoning   | [NLVR2](https://lil.nlp.cornell.edu/nlvr/)  | [PaliGemma](https://huggingface.co/google/paligemma-3b-ft-vqav2-224) & [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xl)|
| Visual Question Answering | [VQA v2](https://visualqa.org/) | [ViLT_NLVR2](https://huggingface.co/dandelin/vilt-b32-finetuned-nlvr2) & [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)|


## Models and configs/ckpt download

- [VQGAN](https://github.com/CompVis/taming-transformers)
- [ViLT_COCO](https://huggingface.co/dandelin/vilt-b32-finetuned-coco)
- [BLIP](https://huggingface.co/Salesforce/blip-itm-base-coco)
- [PaliGemma](https://huggingface.co/google/paligemma-3b-ft-vqav2-224)
- [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)
- [ViLT_NLVR2](https://huggingface.co/dandelin/vilt-b32-finetuned-nlvr2)


# Experiments
***We follow the order in paper RQ to explain usage operations.***

## Environment Configuration

1. Clone this repository (if you haven't already):

```
git clone https://anonymous.4open.science/r/VLTest-7738.git
cd VLTest
```

2. Create a Conda environment from the environment.yml file:

```
conda env create -f environment.yml
conda activate taming_ada
```

or

```
conda env create -f environment.yml -n your_custom_env_name
conda activate your_custom_env_name
```

## Data Preprocessing
Each folder contains a preprocessing script for the corresponding dataset. Please follow the script name and its inner instructions to perform data preprocessing The data source link can be found in [Dataset and Models](#dataset-and-models)

- `get_image_captions.py` extracts captions corresponding to the target images from the dataset-provided JSON files.
- `download_coco_images.py` and `collect_orig_images.py` are used for **VQA v2** and **NLVR2**, respectively, to retrieve the original images from the dataset JSON files.

## RQ1: Exploration Capability Analysis
Run `python run_vltest.py` to perturb images and apply the same operations for baseline perturbation(please run `python run_vltest.py`)

In RQ1 we obtain experimental results by adjusting parameters in these two files You can quickly reproduce the results by modifying the same parameters

eg.
```
import os

os.system(
    "CUDA_VISIBLE_DEVICES=1 python perturber.py \  # run on GPU 1
    --output_path=outputs\                         # output directory
    --task=itm \                                   # task: Image–Text Matching
    --perturb_mode=joint \                         # joint image–text perturbation
    --max_samples=100 \                            # number of seed samples
    --pert_budget=100 \                            # total perturbation budget for each modality (image and text)
    --pict_top_p_ratio=0.1\                        # top-p ratio for image codebook candidates
    --pict_pert_time=40 \                          # max image perturbation attempts
    --pict_pert_mode=uniform \                     # uniform image perturbation
    --text_variant_num=5 \                         # text variants per seed
    --text_max_word=5 \                            # max words changed per variant
    --text_max_word_attempts=5 \                   # max attempts per word
    --seed 123456"                                 # random seed for reproducibility
)
```

```
import os

os.system(
    "CUDA_VISIBLE_DEVICES=0 python perturber_baseline.py \  # run baseline on GPU 0
    --output_path=outputs\                               # output directory
    --task=itm \                                         # task: Image–Text Matching
    --seed_nums=35 \                                     # number of seeds to use
    --pre_seed_num=10 \                                  # number of seeds actually perturbed (remaining are reference seeds)
    --pre_level_start=0\                                 # initial perturbation level
    --pre_level_step=0.0005 \                            # perturbation level increment
    --pre_time=400 \                                     # total baseline perturbation steps
    --seed 123456"                                       # random seed for reproducibility
)
```

## RQ2: Semantic Validity (Human Evaluation)

Step 1 — Install dependencies

    pip install -r requirements.txt

Step 2 — Run the evaluation tool

    python eval.py

Step 3 — Enter your username when prompted.  
    
    A file `<username>_results.json` will be created automatically to store your scores.

Step 4 — Rate all samples
    Check both panels, then rate each Image-Text pair (1–5) for:

    * Image Semantic Preservation (Img-SemPres)
    * Text Semantic Preservation (Txt-SemPres)
    * Image–Text Alignment (ImgTxt-Align)

    Use “Save and Next” to move through samples.

For more details refer to [Here](https://github.com/ALinrunrun/Human-ImgTxt-Evaluation-GUI-Panel)
    
## RQ3: Failure Discovery Effectiveness and Efficiency
Here, we primarily conduct metric statistics and differential testing on the perturbed images. We mainly rely on two files, `metrics.py` and `diff_test.py`.
We take the ITM task as an example to demonstrate how to use them.

For `metrics.py`
- Before computing the metrics, run `compute_bins_dino.py` in each dataset directory to obtain the model’s distribution, which is required for subsequent metric computation.
- Make sure all generated files are saved under `./outputs`.
- Set `MODEL` in the metrics file according to the current task.
- Specify the `RESULT_JSON_PATH`.

```
MODEL = "image"
RESULT_JSON_PATH = "summary_metrics.json"
OUTPUT = "./outputs"
```

3. Specify the name of the output JSON file.

For `diff_test.py`
- We continue to use automatic folder iteration: all contents under `TEST_ROOT` will be evaluated sequentially. Please ensure the directory structure is correct.
- `PAIR_BUDGET` and `TEXT_POOL_BUDGET` correspond to the settings in **RQ3** and should be adjusted accordingly.

```
PAIR_BUDGET = 1000
TEXT_POOL_BUDGET = 100
TEST_ROOT = "diff0"
```
