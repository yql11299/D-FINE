#!/bin/bash
# Single GPU Training Script (Linux)
# Set CUDA_DEVICE_ORDER to ensure GPU IDs match nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "Starting Single GPU Training..."

# Define Output Dir (Must match config)
OUTPUT_DIR="output/dfine_hgnetv2_s_custom_unified"

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

python train.py -c configs/dfine/custom/unified_train_config.yml --use-amp --seed=0 $ARGS
