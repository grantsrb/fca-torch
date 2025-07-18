#!/bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python huggingface_sft.py \
    --model_name_or_path gpt2  \
    --dataset_name trl-lib/Capybara  \
    --learning_rate 2.0e-5  \
    --num_train_epochs 1  \
    --per_device_train_batch_size 12  \
    --gradient_accumulation_steps 8  \
    --gradient_checkpointing  \
    --eos_token '<|im_end|>'  \
    --logging_steps 25  \
    --eval_strategy steps  \
    --eval_steps 100  \
    --output_dir Qwen2-0.5B-SFT
