"""
FiftyOne 可视化与评估工具
---------------------------------
用途：
- 通过 FiftyOne 加载公开数据集（默认 COCO-2017 validation）并启动 Web UI；
- 从 D-FINE 的 YAML 配置与 checkpoint 加载模型与后处理器；
- 对样本运行推理，生成 `predictions{eval_idx}` 标签字段并可选过滤置信度；
- 使用 FiftyOne 的内置评估接口计算 mAP 等指标；
- 将可视化/评估视图导出以便后续复现。

主要流程：
1) 预处理：尝试关闭遗留的 `mongod` 进程，避免与 FiftyOne 的后端数据库冲突；
2) 数据加载：使用 `fiftyone.zoo` 加载 COCO-2017 验证集并持久化；
3) 模型装载：读取 YAML 配置（`YAMLConfig`）与 `--resume` 指向的权重，构建推理模型与后处理器；
4) 推理与评估：对采样视图运行推理，写入 `predictions{L}` 字段，过滤低置信度并计算评估指标；
5) 导出与展示：导出预测视图到磁盘，打开 Web UI 并保持会话运行。

注意：
- 本工具默认将输入图片缩放到 `(640, 640)` 与模型预期一致，输出框以 640 归一化到 [0,1]；
- 需要 GPU 环境（`.cuda()` 调用），如需 CPU 请自行修改模型与张量设备；
- `--resume` 必须指定一个包含 `model` 或 `ema.module` 键的 checkpoint。

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import os
import subprocess
import sys
import time

import fiftyone as fo
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.zoo as foz
import torch
import torchvision.transforms as transforms
import tqdm
from fiftyone import ViewField as F
from PIL import Image

from src.core import YAMLConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def kill_existing_mongod():
    """关闭已有的 mongod 进程

    FiftyOne 使用 MongoDB 作为后端存储；若之前启动的 mongod 未正常退出，
    可能导致端口/数据目录占用从而影响新的会话。这里通过扫描进程列表，
    查找带有 `mongod` 且包含 `--dbpath` 参数的进程并强制终止。

    建议：生产环境中应优先优雅关闭；此处为了工具稳健性采取强杀策略。
    """
    try:
        result = subprocess.run(["ps", "aux"], stdout=subprocess.PIPE)
        processes = result.stdout.decode("utf-8").splitlines()

        for process in processes:
            if "mongod" in process and "--dbpath" in process:
                # find mongod PID
                pid = int(process.split()[1])
                print(f"Killing existing mongod process with PID: {pid}")
                # kill mongod session
                os.kill(pid, 9)
    except Exception as e:
        print(f"Error occurred while killing mongod: {e}")


kill_existing_mongod()


label_map = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorbike",
    5: "aeroplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "trafficlight",
    11: "firehydrant",
    12: "streetsign",
    13: "stopsign",
    14: "parkingmeter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eyeglasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sportsball",
    38: "kite",
    39: "baseballbat",
    40: "baseballglove",
    41: "skateboard",
    42: "surfboard",
    43: "tennisracket",
    44: "bottle",
    45: "plate",
    46: "wineglass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hotdog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "sofa",
    64: "pottedplant",
    65: "bed",
    66: "mirror",
    67: "diningtable",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cellphone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddybear",
    89: "hairdrier",
    90: "toothbrush",
    91: "hairbrush",
}


class CustomModel(fom.Model):
    """将 D-FINE 模型封装为 FiftyOne 兼容的推理模型

    - 读取 YAML 配置的 `model` 与 `postprocessor`，置于 eval 模式并迁移到 GPU；
    - 使用 `torchvision.transforms` 将输入图像转为张量并缩放到 `(640, 640)`；
    - `predict`/`predict_all` 方法返回 FiftyOne 的 `Detections` 对象用于可视化；
    - `_convert_predictions` 负责将模型输出的框/类别/分数转换为归一化的检测结果。
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.eval().cuda()
        self.postprocessor = cfg.postprocessor.eval().cuda()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((640, 640)),  # Resize to the size expected by your model
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    @property
    def media_type(self):
        return "image"

    @property
    def has_logits(self):
        return False

    @property
    def has_embeddings(self):
        return False

    @property
    def ragged_batches(self):
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return True

    @preprocess.setter
    def preprocess(self, value):
        pass

    def _convert_predictions(self, predictions):
        """将模型输出转换为 FiftyOne `Detections`

        参数：
        - `predictions`: D-FINE 后处理器输出的列表，每个元素包含 `labels`、`boxes`、`scores`

        行为：
        - 将 `labels` 通过 `label_map` 映射为字符串类别；
        - 将 `boxes` 从像素坐标转换为 [0,1] 归一化坐标（除以 640）；
        - 组装为 `fol.Detections` 返回。
        """
        class_labels, bboxes, scores = (
            predictions[0]["labels"],
            predictions[0]["boxes"],
            predictions[0]["scores"],
        )

        detections = []
        for label, bbox, score in zip(class_labels, bboxes, scores):
            detection = fol.Detection(
                label=label_map[label.item()],
                bounding_box=[
                    bbox[0] / 640,  # Normalized coordinates
                    bbox[1] / 640,
                    (bbox[2] - bbox[0]) / 640,
                    (bbox[3] - bbox[1]) / 640,
                ],
                confidence=score,
            )
            detections.append(detection)

        return fol.Detections(detections=detections)

    def predict(self, image):
        """对单张图片进行推理并返回 `Detections`

        - 输入：`image` 为 Numpy 数组（FiftyOne 传入）
        - 过程：PIL 转换 → 预处理（ToTensor+Resize）→ 前向 → 后处理 → 转换
        - 输出：`fol.Detections`
        """
        image = Image.fromarray(image).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).cuda()
        outputs = self.model(image_tensor)
        orig_target_sizes = torch.tensor([[640, 640]]).cuda()
        predictions = self.postprocessor(outputs, orig_target_sizes)
        return self._convert_predictions(predictions)

    def predict_all(self, images):
        """对一批图片进行推理

        - 输入：`images` 列表（Numpy 数组）
        - 输出：`List[Detections]` 与 `apply_model` 的批量接口兼容
        """
        image_tensors = []
        for image in images:
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors).cuda()
        outputs = self.model(image_tensors)
        orig_target_sizes = torch.tensor([[640, 640] for image in images]).cuda()
        predictions = self.postprocessor(outputs, orig_target_sizes)
        converted_predictions = [self._convert_predictions(pred) for pred in predictions]

        # Ensure the output is a list of lists of Detections
        return converted_predictions


def filter_by_predictions5_confidence(predictions_view, confidence_threshold=0.3):
    """使用另一评估层的置信度更新预测结果

    该函数示例化了跨不同 `eval_idx` 的置信度对齐逻辑：当 `predictions0` 与
    `predictions5` 在阈值两侧出现“交叉”，则用更高置信度替换低置信度，
    并记录原始置信度以便后续恢复。

    参数：
    - `predictions_view`: FiftyOne 视图对象（含 `predictions0` 与 `predictions5` 字段）
    - `confidence_threshold`: 判定交叉的阈值（默认 0.3）
    """
    for j, sample in tqdm.tqdm(enumerate(predictions_view), total=len(predictions_view)):
        has_modified = False
        for i, detection in enumerate(sample["predictions0"].detections):
            if "original_confidence" not in detection:
                detection["original_confidence"] = detection["confidence"]

            if (
                detection["confidence"] <= confidence_threshold
                and sample["predictions5"].detections[i]["confidence"] >= confidence_threshold
            ) or (
                detection["confidence"] >= confidence_threshold
                and sample["predictions5"].detections[i]["confidence"] <= confidence_threshold
            ):
                sample["predictions0"].detections[i]["confidence"] = sample[
                    "predictions5"
                ].detections[i]["confidence"]
                has_modified = True
        if has_modified:
            sample.save()


def restore_confidence(predictions_view):
    """恢复 `filter_by_predictions5_confidence` 修改过的置信度

    遍历视图，将带有 `original_confidence` 的检测置信度回滚到原值。
    """
    for j, sample in tqdm.tqdm(enumerate(predictions_view), total=len(predictions_view)):
        for i, detection in enumerate(sample["predictions0"].detections):
            if "original_confidence" in detection:
                detection["confidence"] = detection["original_confidence"]
        sample.save()


def fast_iou(bbox1, bbox2):
    """计算两个归一化框的 IoU（xywh 格式）

    参数：
    - `bbox1`/`bbox2`: `[x, y, w, h]`，归一化坐标
    返回：IoU 浮点数
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def assign_iou_diff(predictions_view):
    """为不同评估层的检测结果分配 IoU 差异标记

    - 从 `predictions0` 与 `predictions5` 读取评估结果 `eval0_iou` 与 `eval5_iou`；
    - 若两个框的交并比（互相重叠程度）大于 0.5，则记录两者 IoU 的绝对差；
    - 将差异写入每个检测的 `iou_diff` 字段，便于基于差异的筛选。
    """
    for sample in predictions_view:
        ious_0 = [
            detection.eval0_iou if "eval0_iou" in detection else None
            for detection in sample["predictions0"].detections
        ]
        ious_5 = [
            detection.eval5_iou if "eval5_iou" in detection else None
            for detection in sample["predictions5"].detections
        ]
        bbox_0 = [detection.bounding_box for detection in sample["predictions0"].detections]
        bbox_5 = [detection.bounding_box for detection in sample["predictions5"].detections]
        # iou_diffs = [abs(iou_5 - iou_0) if iou_0 is not None and iou_5 is not None else -1 for iou_0, iou_5 in zip(ious_0, ious_5)]
        iou_inter = [fast_iou(b0, b5) for b0, b5 in zip(bbox_0, bbox_5)]
        iou_diffs = [
            abs(iou_5 - iou_0)
            if iou_0 is not None and iou_5 is not None and iou_inter > 0.5
            else -1
            for iou_0, iou_5, iou_inter in zip(ious_0, ious_5, iou_inter)
        ]

        for detection, iou_diff in zip(sample["predictions0"].detections, iou_diffs):
            detection["iou_diff"] = iou_diff
        for detection, iou_diff in zip(sample["predictions5"].detections, iou_diffs):
            detection["iou_diff"] = iou_diff
        # for detection, iou_diff in zip(sample["predictions100"].detections, iou_diffs):
        #     detection["iou_diff"] = iou_diff
        sample.save()


def main(args):
    """主执行流程：加载数据、模型，推理、评估、导出与展示

    - 若存在已导出的视图目录 `saved_predictions_view`/`saved_filtered_view`，直接载入；
    - 否则：
      1) 从 FiftyOne zoo 加载 COCO-2017 验证集并启动 App；
      2) 使用 `YAMLConfig` 读入配置与 checkpoint，构建模型并加载权重；
      3) 选取 100 张样本组成视图，按当前解码器评估层 `eval_idx=L` 运行推理；
      4) 对 `predictions{L}` 字段按置信度过滤并计算 mAP；
      5) 导出预测视图到 `saved_predictions_view` 目录；
    - 最后将会话置为该视图并保持运行，直至进程结束。
    """
    try:
        if os.path.exists("saved_predictions_view") and os.path.exists("saved_filtered_view"):
            print("Loading saved predictions and filtered views...")
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                dataset_name="evaluate-detections-tutorial",
                dataset_dir="data/fiftyone",
            )

            dataset.persistent = True
            session = fo.launch_app(dataset)

            predictions_view = fo.Dataset.from_dir(
                dataset_dir="saved_predictions_view", dataset_type=fo.types.FiftyOneDataset
            ).view()
            filtered_view = fo.Dataset.from_dir(
                dataset_dir="saved_filtered_view", dataset_type=fo.types.FiftyOneDataset
            ).view()
        else:
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                dataset_name="evaluate-detections-tutorial",
                dataset_dir="data/fiftyone",
            )

            dataset.persistent = True

            session = fo.launch_app(dataset)
            cfg = YAMLConfig(args.config, resume=args.resume)
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            if args.resume:
                checkpoint = torch.load(args.resume, map_location="cpu")
                if "ema" in checkpoint:
                    state = checkpoint["ema"]["module"]
                else:
                    state = checkpoint["model"]
            else:
                raise AttributeError("only support resume to load model.state_dict by now.")

            # NOTE load train mode state -> convert to deploy mode
            cfg.model.load_state_dict(state)
            predictions_view = dataset.take(100, seed=51)

            model = CustomModel(cfg)
            L = model.model.decoder.decoder.eval_idx
            # Apply models and save predictions in different label fields
            for i in [L]:
                model.model.decoder.decoder.eval_idx = i
                label_field = "predictions{:d}".format(i)
                predictions_view.apply_model(model, label_field=label_field)

            # filter_by_predictions5_confidence(predictions_view, confidence_threshold=0.3)
            for i in [L]:
                label_field = "predictions{:d}".format(i)
                predictions_view = predictions_view.filter_labels(
                    label_field, F("confidence") > 0.5, only_matches=False
                )
                eval_key = "eval{:d}".format(i)
                _ = predictions_view.evaluate_detections(
                    label_field,
                    gt_field="ground_truth",
                    eval_key=eval_key,
                    compute_mAP=True,
                )

            # assign_iou_diff(predictions_view)

            # filtered_view = predictions_view.filter_labels("predictions0", F("iou_diff") > 0.05, only_matches=True)
            # filtered_view = filtered_view.filter_labels("predictions5", F("iou_diff") > 0.05, only_matches=True)
            # restore_confidence(filtered_view)

            predictions_view.export(
                export_dir="saved_predictions_view", dataset_type=fo.types.FiftyOneDataset
            )
            # filtered_view.export(
            #     export_dir="saved_filtered_view",
            #     dataset_type=fo.types.FiftyOneDataset
            # )

        # Display the filtered view
        session.view = predictions_view

        # Keep the session open
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down session")
        if "session" in locals():
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--resume", "-r", type=str)
    args = parser.parse_args()

    main(args)
