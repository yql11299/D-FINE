#!/bin/bash
# Multi-GPU Training Script (Linux)
# Set CUDA_DEVICE_ORDER to ensure GPU IDs match nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Set the GPU IDs you want to use (e.g., 0,3)
export CUDA_VISIBLE_DEVICES=0,3

echo "Starting Multi-GPU Training on GPUs: $CUDA_VISIBLE_DEVICES"

# Define Output Dir (Must match config)
OUTPUT_DIR="output/dfine_hgnetv2_n_custom_unified"

# Check for resume flag
ARGS=""
if [ "$1" == "--resume" ]; then
    if [ -f "$OUTPUT_DIR/last.pth" ]; then
        echo "Resuming from $OUTPUT_DIR/last.pth"
        ARGS="-r $OUTPUT_DIR/last.pth"
    else
        echo "Warning: Resume requested but $OUTPUT_DIR/last.pth not found. Starting from scratch."
    fi
fi

# Note: --nproc_per_node should match the number of GPUs in CUDA_VISIBLE_DEVICES
torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/dfine/custom/unified_train_config_n.yml --use-amp --seed=0 $ARGS
