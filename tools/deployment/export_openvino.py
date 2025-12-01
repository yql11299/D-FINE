"""
D-FINE: Export OpenVINO IR from an ONNX model
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
import shutil
import subprocess
import nncf
import cv2
from typing import List
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


import json



def main(args):
    """Convert ONNX to OpenVINO IR. For INT8, uses Post-Training Optimization Tool (POT)."""
    precision = args.precision.upper()
    onnx_path = args.onnx
    input_shape = [1, 3, args.img_size[0], args.img_size[1]]

    try:
        import openvino as ov
        from nncf import quantize, QuantizationPreset
    except ImportError:
        print("OpenVINO NNCF API not found. 请确保已安装 openvino-dev >=2024.6.0，且包含 NNCF。\n安装命令: pip install openvino-dev nncf")
        raise

    os.makedirs(args.output_dir, exist_ok=True)

    if precision in ["FP32", "FP16"]:
        xml_path = os.path.join(args.output_dir, f"{args.model_name}_{precision.lower()}.xml")
        print(f"Converting ONNX → OpenVINO IR\n  input: {onnx_path}\n  output: {xml_path}\n  precision: {args.precision}")

        ov_model = ov.convert_model(onnx_path)

        compress_to_fp16 = precision == "FP16"
        ov.save_model(ov_model, xml_path, compress_to_fp16=compress_to_fp16)

        print(f"Saved OpenVINO IR: {xml_path}")
        return

    elif precision == "INT8":
        print("--- Starting INT8 Quantization using OpenVINO Quantize API ---")

        # 1. Load the ONNX model
        print("\n--- Step 1: Loading ONNX model ---")
        ov_model = ov.convert_model(onnx_path)
        print("Model loaded successfully.")

        # 2. Create the calibration dataset
        print("\n--- Step 2: Creating calibration dataset ---")
        
        input_shape = [1, 3, args.img_size[0], args.img_size[1]]  # [1, 3, H, W]
        input_h, input_w = input_shape[2], input_shape[3]
        input_name = ov_model.input(0).any_name

        def transform_fn(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (input_w, input_h))
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 0)
            image = image.astype(np.float32) / 255.0
            return {input_name: image, "orig_target_sizes": np.array([[input_h, input_w]], dtype=np.float32)}

        # 3. Quantize the model
        print("\n--- Step 3: Running quantization ---")
        quantized_model = quantize(
            ov_model,
            calibration_dataset=nncf.Dataset(
                sorted([os.path.join(args.calib_dir, f) for f in os.listdir(args.calib_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]),
                transform_func=transform_fn
            ),
            preset=QuantizationPreset.PERFORMANCE,
            subset_size=min(args.subset, len(os.listdir(args.calib_dir))) # Ensure subset_size is not larger than available images
        )
        print("Quantization complete.")
        
        

        # 4. Save the quantized model
        print("\n--- Step 4: Saving quantized model ---")
        output_model_name = f"{args.model_name}_int8"
        output_path = os.path.join(args.output_dir, f"{output_model_name}.xml")
        ov.save_model(quantized_model, output_path)

        print(f"\n--- INT8 quantization successful! ---")
        print(f"Saved quantized OpenVINO IR: {output_path}")

    else:
        raise ValueError(f"Unsupported precision: {args.precision}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX to OpenVINO IR, using POT for INT8 quantization.")
    parser.add_argument("--onnx", type=str, default=r"output\dfine_hgnetv2_s_custom\best_stg1.onnx", help="Path to the input ONNX model")
    parser.add_argument(
        "--output-dir", type=str, default=r"output", help="Directory to save OpenVINO IR files"
    )
    parser.add_argument(
        "--model-name", type=str, default="model", help="Output model base name (without suffix)"
    )
    parser.add_argument(
        "--precision", type=str, default="int8", choices=["FP32", "FP16", "INT8"], help="IR precision"
    )
    parser.add_argument(
        "--calib-dir", type=str, default=r"G:\datasets\chupin\nmsr\train_data\merge\images", help="Calibration images directory for INT8"
    )
    parser.add_argument(
        "--img-size", nargs=2, type=int, default=[320, 320], help="Export H W for inputs (Note: Not used by POT simplified engine, but kept for compatibility)"
    )
    parser.add_argument(
        "--subset", type=int, default=200, help="Number of calibration samples for POT"
    )
    args = parser.parse_args()
    main(args)

