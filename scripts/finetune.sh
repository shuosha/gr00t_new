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
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path $DATASET_LOCAL \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/ALOHA/scenix_aloha_single_arm_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir checkpoints/260617_gr00t_aloha_pipe_insert_rtc12_80Hz_v2 \
    --save-total-limit 5 \
    --save-steps 10000 \
    --max-steps 50000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --color-jitter-p 0.5 \
    --random-rotation-angle 5 \
    --geometric-p 0.5 \
    --state-input-noise-scale 0.01 \
    --dataloader-num-workers 4 \
    --rtc-max-delay 12