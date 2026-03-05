import os

os.system(
    "CUDA_VISIBLE_DEVICES=1 python perturber.py \
    --output_path=outputs\
    --task=itm \
    --perturb_mode=joint \
    --max_samples=100 \
    --pert_budget=100 \
    --pict_top_p_ratio=0.1\
    --pict_pert_time=40 \
    --pict_pert_mode=uniform \
    --text_variant_num=5 \
    --text_max_word=5 \
    --text_max_word_attempts=5 \
    --seed 123456"
)
