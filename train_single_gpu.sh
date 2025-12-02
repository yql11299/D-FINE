#!/bin/bash
# Single GPU Training Script (Linux)
# Set CUDA_DEVICE_ORDER to ensure GPU IDs match nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "Starting Single GPU Training..."
python train.py -c configs/dfine/custom/unified_train_config.yml --use-amp --seed=0
