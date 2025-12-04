"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from src.core import YAMLConfig


class CustomPostProcessor(nn.Module):
    def __init__(self, postprocessor):
        super().__init__()
        self.postprocessor = postprocessor

    def forward(self, outputs, orig_target_sizes):
        # Assuming outputs from the model are in a format that can be directly converted
        # to boxes and scores without NMS.
        # The original postprocessor might perform NMS, so we need to bypass or modify it.
        # For simplicity, let's assume outputs['pred_boxes'] and outputs['scores'] are available
        # and we just need to concatenate them.
        
        # The original postprocessor expects a list of dicts, where each dict contains
        # 'boxes' and 'scores'. We need to adapt this.
        
        # This is a placeholder. The actual implementation will depend on the exact
        # output format of cfg.model.deploy() and how to extract raw boxes and scores.
        
        # Example: if outputs is a dict with 'pred_boxes' and 'scores'
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']
        
        # Apply sigmoid to pred_logits to get scores
        scores = pred_logits.sigmoid()
        
        # Reshape scores to match pred_boxes for concatenation
        # Assuming scores are [batch_size, num_queries, num_classes]
        # and we want the score for the predicted class, or max score.
        # For simplicity, let's assume scores are already [batch_size, num_queries]
        # or we take the max score per box.
        
        # If scores are [batch_size, num_queries, num_classes], take max over classes
        if scores.dim() == 3:
            scores, _ = scores.max(dim=-1) # [batch_size, num_queries]
            
        # Concatenate boxes and scores
        # pred_boxes: [batch_size, num_queries, 4]
        # scores: [batch_size, num_queries]
        # Result: [batch_size, num_queries, 5]
        
        # Ensure scores has an extra dimension for concatenation
        scores = scores.unsqueeze(-1) # [batch_size, num_queries, 1]
        
        # Concatenate boxes and scores
        output_tensor = torch.cat([pred_boxes, scores], dim=-1)
        
        return output_tensor


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume, eval_spatial_size=args.img_size)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            # Use a custom postprocessor that does not perform NMS
            self.postprocessor = CustomPostProcessor(cfg.postprocessor.deploy())

        def forward(self, images): # Removed orig_target_sizes from input
            # Input is 1*320*320*3, need to transpose to 1*3*320*320
            images = images.permute(0, 3, 1, 2)
            
            outputs = self.model(images)
            
            # The original postprocessor expects orig_target_sizes.
            # Since we are removing NMS and simplifying output, we might not need it.
            # However, if the model itself (self.model) still requires it, we need to pass a dummy.
            # For now, let's assume self.model does not need it for raw outputs.
            # If self.postprocessor still needs it, we can pass a dummy.
            
            # Let's create a dummy orig_target_sizes for the custom postprocessor if needed
            # Assuming batch size is 1 and image size is 320x320
            batch_size = images.shape[0]
            h, w = images.shape[2:] # Get H, W from transposed image
            orig_target_sizes = torch.tensor([[w, h]], dtype=torch.float32).repeat(batch_size, 1)
            
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    h, w = args.img_size
    data = torch.rand(1, h, w, 3)

    # orig_target_sizes is no longer an input to the model's forward method
    # size = torch.tensor([[w, h]], dtype=torch.float32).repeat(batch, 1)

    _ = model(data)  # Pass only data

    output_file = args.resume.replace(".pth", "_custom.onnx") if args.resume else "model_custom.onnx"

    torch.onnx.export(
        model,
        (data,),  # Only pass data as input
        output_file,
        input_names=["images"],
        output_names=["output"],  # Single output for 1*n*5
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
        dynamic_axes=None,  # Export with fixed size
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        # Simplify with fixed shape
        onnx_model_simplify, check = onnxsim.simplify(output_file)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify onnx model {check}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=r"configs\dfine\custom\unified_train_config.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        default=r"D:\valid_data\num_merge\model\dfine\dfine_hgnetv2_s_custom_unified\best_stg2.pth",
        type=str,
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[320, 320],
        help="Image size for export (height, width)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()
    main(args)
