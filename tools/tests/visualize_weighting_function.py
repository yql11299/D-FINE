#!/usr/bin/env python3
import argparse
import os
import sys

import torch

# 直接在此处定义与源码一致的 weighting_function，避免包导入链触发相对导入错误
def weighting_function(reg_max, up, reg_scale, deploy=False):
    """生成非均匀刻度序列 W(n) 供分布积分使用"""
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.tensor(0.0)]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def main():
    """可视化 weighting_function 生成的刻度序列并打印数值"""
    parser = argparse.ArgumentParser(description="Visualize weighting function sequence")
    parser.add_argument("--reg_max", type=int, default=32, help="离散桶最大数量")
    parser.add_argument("--up", type=float, default=0.5, help="权重函数上下界幅度控制")
    parser.add_argument("--reg_scale", type=float, default=4.0, help="权重函数曲率控制")
    parser.add_argument("--output", type=str, default="./weighting_function.png", help="保存图像路径")
    args = parser.parse_args()

    # 构造参数张量
    up = torch.tensor([args.up], dtype=torch.float32)
    values = weighting_function(args.reg_max, up, args.reg_scale, deploy=True)

    # 打印数值序列
    print("W(n) 序列 (长度=%d):" % values.numel())
    print(values.tolist())

    # 绘制并保存图像（延迟导入 matplotlib，避免环境缺失导致导入失败）
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        xs = list(range(values.numel()))
        ys = values.cpu().numpy()
        plt.plot(xs, ys, marker="o", linewidth=1.5)
        plt.title("Weighting Function W(n)\nreg_max=%d, up=%.3f, reg_scale=%.3f" % (args.reg_max, args.up, args.reg_scale))
        plt.xlabel("n (bin index)")
        plt.ylabel("W(n)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        plt.savefig(args.output)
        print("已保存可视化到:", args.output)
    except Exception as e:
        print("matplotlib 不可用或渲染失败：", e)
        print("已仅打印数值序列，若需图像请安装 matplotlib：pip install matplotlib")


if __name__ == "__main__":
    main()
