"""
数据增强容器与调度策略
---------------------------------
职责：
- 承载一组可配置的图像/标签变换（`transforms`），并按策略动态决定执行；
- 支持三种前向策略：默认执行、按 `epoch` 停止特定变换、按样本计数停止特定变换；
- 通过全局注册机制从 YAML 配置中解析变换实例。

使用方式：
- 在数据集的 `transforms` 字段中声明：
  - `type: Compose`，`ops: [...]` 变换列表；
  - 可选 `policy`：`{"name": "default"|"stop_epoch"|"stop_sample", ...}`。

策略说明：
- `default`：所有变换顺序执行；
- `stop_epoch`：达到设定 `epoch` 后，跳过 `policy.ops` 指定的变换；
- `stop_sample`：累计样本数达到设定 `sample` 后，跳过 `policy.ops` 指定的变换。
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform

torchvision.disable_beta_transforms_warning()


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        """根据配置构建变换管线并设置执行策略

        参数：
        - `ops`: 变换列表；支持字典（经注册表实例化）或已构建的 `nn.Module`
        - `policy`: 执行策略字典（见模块说明）
        """
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop("type")
                    transform = getattr(
                        GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"]
                    )(**op)
                    transforms.append(transform)
                    op["type"] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError("")
        else:
            transforms = [
                EmptyTransform(),
            ]

        super().__init__(transforms=transforms)

        if policy is None:
            policy = {"name": "default"}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        """入口方法，按策略选择对应的执行函数"""
        return self.get_forward(self.policy["name"])(*inputs)

    def get_forward(self, name):
        """返回策略对应的前向函数"""
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
            "stop_sample": self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        """默认：顺序执行所有变换"""
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        """按 epoch 阈值停止指定变换

        依赖：`policy` 中包含 `ops`（需要停止的变换类名列表）与 `epoch`（阈值）。
        当当前数据集 `epoch >= policy.epoch` 时，对属于 `policy.ops` 的变换执行跳过。
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def stop_sample_forward(self, *inputs: Any):
        """按样本计数阈值停止指定变换

        依赖：`policy` 中包含 `ops`（需要停止的变换类名列表）与 `sample`（阈值）。
        当累计样本数 `global_samples >= policy.sample` 时，对属于 `policy.ops` 的变换执行跳过。
        每次调用后自增 `global_samples`。
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_sample = self.policy["sample"]

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
