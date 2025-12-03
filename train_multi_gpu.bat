@echo off
REM Multi-GPU Training Script (Windows)
REM Set CUDA_DEVICE_ORDER to ensure GPU IDs match nvidia-smi
set CUDA_DEVICE_ORDER=PCI_BUS_ID
REM Set the GPU IDs you want to use (e.g., 0,3)
set CUDA_VISIBLE_DEVICES=0,3

echo Starting Multi-GPU Training on GPUs: %CUDA_VISIBLE_DEVICES%

REM Define Output Dir (Must match config)
set OUTPUT_DIR=output\dfine_hgnetv2_s_custom_unified
set ARGS=

if "%1"=="--resume" (
    if exist "%OUTPUT_DIR%\last.pth" (
        echo Resuming from %OUTPUT_DIR%\last.pth
        set ARGS=-r %OUTPUT_DIR%\last.pth
    ) else (
        echo Warning: Resume requested but %OUTPUT_DIR%\last.pth not found. Starting from scratch.
    )
)

REM Note: --nproc_per_node should match the number of GPUs in CUDA_VISIBLE_DEVICES
torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/dfine/custom/unified_train_config.yml --use-amp --seed=0 %ARGS%

pause
