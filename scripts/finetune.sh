#!/usr/bin/env bash
# Scenix ALOHA finetune on GR00T N1.7 (mirror of the n1.6.1-rtc branch script).
#
# Differences vs n1.6:
#   - base model nvidia/GR00T-N1.7-3B
#   - NO --rtc-max-delay: n1.7 has no training-time RTC; RTC is inference-time
#     inpainting driven per-call via Gr00tPolicy options (any ckpt supports it).
#   - action horizon is 40 (see examples/ALOHA/*_config.py delta_indices).
#
# Run inside this repo's own uv env (py3.10, torch 2.7.1): `uv sync`, then
# `uv run bash scripts/finetune.sh`. Do NOT use manipulation_gym's .venv-n17.
set -euo pipefail

DATASET_REPO=shashuo0104/260617_aloha_pipe_insert_80Hz_v2
DATASET_LOCAL=./data/260617_aloha_pipe_insert_80Hz_v2

# Download dataset from HuggingFace if not already present
if [ ! -d "$DATASET_LOCAL" ]; then
    echo "Downloading dataset from HuggingFace..."
    huggingface-cli download $DATASET_REPO \
        --repo-type dataset \
        --local-dir $DATASET_LOCAL
fi

export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $DATASET_LOCAL \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/ALOHA/scenix_aloha_single_arm_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir checkpoints/260617_gr00t_n17_aloha_pipe_insert_80Hz_v2 \
    --save-total-limit 5 \
    --save-steps 10000 \
    --max-steps 50000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
