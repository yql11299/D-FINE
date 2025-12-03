@echo off
REM Single GPU Training Script (Windows)
REM Set CUDA_DEVICE_ORDER to ensure GPU IDs match nvidia-smi
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0

echo Starting Single GPU Training...

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

python train.py -c configs/dfine/custom/unified_train_config.yml --use-amp --seed=0 %ARGS%

pause
