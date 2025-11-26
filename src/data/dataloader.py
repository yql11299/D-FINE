"""
数据加载与批次打包模块
---------------------------------
职责：
- 封装 `torch.utils.data.DataLoader`，支持在训练周期内动态设置 `epoch`；
- 提供仅拼接图像的 `batch_image_collate_fn` 与具备多尺度插值的 `BatchImageCollateFunction`；
- 为 D-FINE 的训练/验证循环提供一致的数据打包与变换入口。

设计要点：
- `DataLoader` 通过 `__inject__` 依赖注入配置的 `dataset` 与 `collate_fn`；
- 每个 `epoch` 开始时调用 `set_epoch`，将周期信息同步到数据集与打包函数；
- `BatchImageCollateFunction` 支持在早期训练阶段进行随机多尺度重采样，以增强鲁棒性；
- 仅在目标包含 `masks` 时才进行对应的插值（当前实现抛出未实现以提示补充）。
"""

import random
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from torch.utils.data import default_collate
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as VF

from ..core import register

torchvision.disable_beta_transforms_warning()


__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    """扩展版数据加载器

    - 通过依赖注入构建底层数据集与打包函数；
    - 提供 `set_epoch`/`epoch` 属性，便于与分布式采样器/自定义策略联动；
    - `shuffle` 作为布尔属性控制随机顺序（与采样器行为相协同）。
    """

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """仅拼接图像维度，保持 targets 列表不变

    - 输入：`items` 为形如 `(image_tensor, target_dict)` 的列表；
    - 输出：`(batched_images, targets)` 其中 `batched_images` 为 `NCHW` 拼接张量；
    - 适用于无需对 `targets` 做复杂对齐的检测任务。
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


def generate_scales(base_size, base_size_repeat):
    """生成多尺度列表用于随机插值

    - 以 `base_size` 为中心，按 0.75 与 1.25 的比例构建尺度序列；
    - `base_size_repeat` 控制基准尺度重复次数，以提高出现频率；
    - 返回按 32 对齐的整数尺度列表。
    """
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = (
            generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        # 预留插值模式 `self.interpolation`：可在需要时扩展到不同插值策略

    def __call__(self, items):
        """将输入样本打包为批次并进行可选的多尺度重采样

        - 将图像拼接为 `NCHW` 张量，目标以列表形式返回；
        - 若设置了 `scales` 且当前 `epoch < stop_epoch`，则对图像执行随机插值；
        - 当目标包含 `masks` 时，对掩码进行相同尺度插值（当前未实现）。
        """
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")
                raise NotImplementedError("")

        return images, targets
