@echo off
REM Single GPU Training Script (Windows)
REM Set CUDA_DEVICE_ORDER to ensure GPU IDs match nvidia-smi
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0

echo Starting Single GPU Training...
python train.py -c configs/dfine/custom/unified_train_config.yml --use-amp --seed=0

pause
