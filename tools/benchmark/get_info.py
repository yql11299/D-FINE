"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import argparse

import torch
import torch.nn as nn
from calflops import calculate_flops

from src.core import YAMLConfig


def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=None)

    class Model_for_flops(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            outputs = self.model(images)
            return outputs

    model = Model_for_flops().eval()

    # Resolve input size: default from config's collate_fn.base_size, can be overridden by CLI
    try:
        base_size = cfg.train_dataloader.collate_fn.base_size
    except Exception:
        base_size = 640
    if args.img_size is not None:
        if isinstance(args.img_size, (list, tuple)):
            if len(args.img_size) == 1:
                h = w = int(args.img_size[0])
            elif len(args.img_size) >= 2:
                h, w = int(args.img_size[0]), int(args.img_size[1])
            else:
                h = w = int(base_size)
        else:
            h = w = int(args.img_size)
    else:
        h = w = int(base_size)

    flops, macs, _ = calculate_flops(
        model=model,
        input_shape=(1, 3, h, w),
        output_as_string=True,
        output_precision=4,
        print_detailed=args.detailed,
    )
    params = sum(p.numel() for p in model.parameters())
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="configs/dfine/dfine_hgnetv2_l_coco.yml", type=str
    )
    parser.add_argument(
        "--img-size",
        "-s",
        nargs="+",
        type=int,
        help="输入尺寸，单值表示方形 H=W，或两个值 H W",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="打印每一层的 FLOPs/参数 详细信息",
    )
    args = parser.parse_args()

    main(args)
