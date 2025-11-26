"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_xyxy_to_cxcywh


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """生成非均匀刻度的权重函数 W(n)

    作用：
    - 为分布回归提供一条“非线性刻度曲线”，刻度序列在中心更密集、两端更稀疏，使模型对近中心的小偏移更敏感、对极端值更稳健。

    参数：
    - `reg_max`：离散桶最大数量（总桶数约为 `reg_max+1`，含两端与中心）
    - `up`：控制刻度上下界的张量（通常为单元素张量），刻度范围与图像尺度相关，最大偏移约为 `± up * H / W`
    - `reg_scale`：控制曲线“弯曲度”，绝对值越大，中心附近更平缓、两端变化更陡；影响步长计算与两端上界
    - `deploy`：部署模式开关。在部署模式下，返回值以 `torch.tensor` 构造，便于图优化与导出；训练模式下使用 `torch.cat` 拼接张量序列

    返回：
    - `Tensor`：长度约为 `reg_max+1` 的刻度序列，包含：左端上界→左侧刻度递进→中心零刻度→右侧刻度递进→右端上界

    细节：
    - `upper_bound1/upper_bound2`：由 `abs(up[0]) * abs(reg_scale)` 推导出的两端界，右端取两倍以对称；
    - `step`：按照 `2/(reg_max-2)` 指数关系生成左右刻度的几何级进，确保越靠近中心步长越小；
    - 中心刻度使用 `0`，以对齐“零偏移”位置；左右刻度通过幂次序列生成，再在两端补上界值，形成完整的非均匀序列。
    """
    if deploy:
        # 部署模式：以 Python list 组合并一次性转为 Tensor，便于导出
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        # 左侧刻度：从靠近中心到更远，值为 -(step^i)+1，使其单调递减至负端
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        # 右侧刻度：从靠近中心到更远，值为 step^i - 1，使其单调递增至正端
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        # 训练模式：直接拼接张量序列，保持计算图梯度属性
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


def translate_gt(gt, reg_max, reg_scale, up):
    """将连续 GT 边距值投影到非均匀离散分布桶，并计算插值权重

    目的：
    - 将连续值（例如相对参考点的四边距离）映射到由 `weighting_function` 生成的非均匀刻度序列上，得到最贴近的“左桶”索引与左右插值权重，以便训练分布回归。

    输入参数：
    - `gt`：形状为 `(N,)` 的连续目标值张量；N 可表示批内所有边距的拉直集合
    - `reg_max`：离散桶的最大数量（`reg_max+1` 个取值，包含两端与中心）
    - `reg_scale`：刻度曲率控制参数，数值越大曲线越平缓，越小两端越陡峭（影响权重函数形状）
    - `up`：控制权重函数上下界的张量（通常形如 `[u]`），决定允许的最大偏移幅度

    返回：
    - `indices`：每个 `gt` 对应的左侧桶索引（浮点表示，内部包含边界修正），形状 `(N,)`
    - `weight_right`：右侧桶的插值权重，形状 `(N,)`
    - `weight_left`：左侧桶的插值权重，形状 `(N,)`

    细节说明：
    - 非均匀刻度由 `weighting_function(reg_max, up, reg_scale)` 产生，中心更密集、两端跨度更大，有助于提升近中心区域的拟合精度同时稳定极端值
    - 插值权重基于目标值到左右桶刻度的距离做归一化分配，保证 `weight_left + weight_right = 1`
    - 边界情况（目标值落在最小/最大刻度之外）通过掩码处理并赋予合理的权重与索引修正
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # 计算每个 gt 相对每个刻度的差值，并找到“最靠左且不超过 gt”的刻度索引
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # 将索引转为浮点以便后续边界修正与插值赋值
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    # 有效索引掩码：在合法桶区间内的元素参与标准插值计算
    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # 取出左右刻度值，计算与 gt 的距离用于插值权重
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # 有效权重：右权重按左/右距离比计算，左权重为补值
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # 边界处理：目标值落在最小刻度左侧（负向越界）
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    # 边界处理：目标值落在最大刻度右侧（正向越界）
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    将 GT 框坐标转换为相对参考点的四边距离（用于 FDR 标签）

    参数:
        points (Tensor): 形状 (n, 4)，格式 [x, y, w, h]，其中 (x, y) 为参考点中心
        bbox   (Tensor): 形状 (n, 4)，GT 框，"xyxy" 格式
        reg_max (float): 离散桶最大上界，用于限制目标距离范围
        reg_scale (float): 权重函数 W(n) 的曲率控制参数，影响归一化与平移项
        up (Tensor): 控制 W(n) 上下界的张量
        eps (float): 极小值，避免目标距离恰等于 reg_max 导致边界问题

    返回:
        Tuple[Tensor, Tensor, Tensor]:
            - 展平的一维目标距离张量 (n*4,)
            - 右侧桶插值权重 (n*4,)
            - 左侧桶插值权重 (n*4,)
    """
    # 取绝对值，保证缩放方向正确
    reg_scale = abs(reg_scale)
    # 计算左边距离：参考点中心到 GT 左边的归一化偏移，再减去 0.5*reg_scale 做基准平移
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    # 计算上边距离：参考点中心到 GT 上边的归一化偏移，再减去 0.5*reg_scale 做基准平移
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    # 计算右边距离：GT 右边到参考点中心的归一化偏移，再减去 0.5*reg_scale 做基准平移
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    # 计算下边距离：GT 下边到参考点中心的归一化偏移，再减去 0.5*reg_scale 做基准平移
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    # 拼接为四边距离张量，形状 (n, 4)
    four_lens = torch.stack([left, top, right, bottom], -1)
    # 将连续距离投影到非均匀离散桶，并计算左右插值权重
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    # 边界截断，保证目标距离不超过 reg_max - eps
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    # 展平为 (n*4,) 并断开梯度，权重同样断开梯度，避免监督支路回传
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
