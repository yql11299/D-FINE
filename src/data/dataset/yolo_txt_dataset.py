from typing import Optional, List
import os
from PIL import Image
import torch

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset


@register()
class YOLOTxtDetection(DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(
        self,
        list_file: str,
        transforms: Optional[object] = None,
    ):
        with open(list_file, "r") as f:
            self.images: List[str] = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def _label_path_from_image(self, image_path: str) -> str:
        # Replace the first occurrence of '/images/' with '/labels/'
        # and change extension to .txt. If 'images' segment not found,
        # assume sibling directory 'labels' under the image root.
        img_dir, img_name = os.path.split(image_path)
        stem, _ = os.path.splitext(img_name)
        if os.sep + "images" + os.sep in image_path:
            label_path = image_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            label_path = os.path.splitext(label_path)[0] + ".txt"
        else:
            label_path = os.path.join(os.path.dirname(img_dir), "labels", stem + ".txt")
        return label_path

    def _read_yolo_label(self, label_path: str) -> List[List[float]]:
        boxes = []
        labels = []
        if not os.path.exists(label_path):
            return boxes, labels
        with open(label_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h])
                labels.append(cls)
        return boxes, labels

    def load_item(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        label_path = self._label_path_from_image(image_path)
        yolo_boxes, yolo_labels = self._read_yolo_label(label_path)

        xyxy_boxes = []
        areas = []
        for cx, cy, bw, bh in yolo_boxes:
            # YOLO format is normalized [0,1]; convert to pixels
            cx_px = cx * w
            cy_px = cy * h
            bw_px = bw * w
            bh_px = bh * h
            x1 = cx_px - bw_px / 2
            y1 = cy_px - bh_px / 2
            x2 = cx_px + bw_px / 2
            y2 = cy_px + bh_px / 2
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))
            if x2 > x1 and y2 > y1:
                xyxy_boxes.append([x1, y1, x2, y2])
                areas.append((x2 - x1) * (y2 - y1))

        output = {}
        output["image_id"] = torch.tensor([index])
        output["image_path"] = image_path
        if len(xyxy_boxes) > 0:
            boxes = torch.tensor(xyxy_boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        output["boxes"] = convert_to_tv_tensor(boxes, key="boxes", spatial_size=[h, w])

        if len(yolo_labels) > 0:
            output["labels"] = torch.tensor(yolo_labels, dtype=torch.int64)
        else:
            output["labels"] = torch.zeros((0,), dtype=torch.int64)

        if len(areas) > 0:
            output["area"] = torch.tensor(areas, dtype=torch.float32)
        else:
            output["area"] = torch.zeros((0,), dtype=torch.float32)

        output["iscrowd"] = torch.zeros((output["boxes"].shape[0],), dtype=torch.int64)
        output["orig_size"] = torch.tensor([w, h])

        return image, output

