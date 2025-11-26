"""
批量推理与导出工具（图像/视频）
---------------------------------
用途：
- 对单张图像、文件夹内图像或记录图像路径的 txt 文件执行批量推理；
- 可选保存可视化结果（绘制框与类别/分数到图像）；
- 可选导出标签文件，支持 VOC（XML）或 YOLO（txt）格式；
- 可选按类别过滤输出，仅保留指定类别的检测结果。

默认行为：
- 不保存可视化与标签（`--visualize` 关闭、`--label-format none`）；
- 不进行类别过滤（`--classe` 不指定或为空）。

输入与识别：
- `-i/--input` 可以是：
  - 单张图像路径（jpg/png/bmp/webp）；
  - 图像目录（自动收集支持扩展名）；
  - txt 文件（每行一个图像绝对/相对路径）。

输出目录结构：
- `--output-dir` 下创建：
  - `vis/` 用于保存可视化图像（当 `--visualize` 开启时）；
  - `labels/` 用于保存标签文件（当 `--label-format` 非 `none` 时）。

模型与后处理：
- 从 YAML 配置与 `--resume` checkpoint 加载 D-FINE 模型；
- 使用 `deploy()` 模式的后处理器返回每张图像的 `labels`、`boxes`、`scores`。

过滤规则：
- 置信度阈值 `--score-thresh`（默认 0.4）；
- 类别过滤 `--classe`：列表（整型 ID），仅输出属于该集合的目标；
- 注意：`labels` 为内部类别索引（非 COCO 类别 ID），如需映射请参考项目配置。

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_label2category, mscoco_category2name


def _filter_by_thresh_and_classes(labs, bxs, scrs, thrh, allowed_classes):
    """根据分数阈值与类别集合过滤检测结果

    返回：过滤后的 (labels, boxes, scores) 列表，长度一致
    """
    labs_list = [int(x.item()) for x in labs]
    out_l, out_b, out_s = [], [], []
    for lab, box, sc in zip(labs_list, bxs.tolist(), scrs.tolist()):
        if sc <= thrh:
            continue
        if allowed_classes and lab not in allowed_classes:
            continue
        out_l.append(lab)
        out_b.append(box)
        out_s.append(sc)
    return out_l, out_b, out_s


def draw(images, labels, boxes, scores, thrh=0.4, allowed_classes=None, remap_map=None):
    out_images = []
    palette = [
        (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 214, 56), (255, 56, 214),
        (56, 214, 255), (255, 160, 56), (160, 56, 255), (56, 160, 255), (160, 255, 56),
    ]
    for i, im in enumerate(images):
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        thickness = max(2, int(round(min(h, w) / 300)))
        font_scale = max(0.6, thickness / 2)
        overlay = img.copy()

        labs, bxs, scrs = labels[i], boxes[i], scores[i]
        labs_f, bxs_f, scrs_f = _filter_by_thresh_and_classes(labs, bxs, scrs, thrh, allowed_classes)

        for lab, b, sc in zip(labs_f, bxs_f, scrs_f):
            color = palette[int(lab) % len(palette)]
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

            label_text = remap_map.get(lab, lab) if remap_map is not None else lab
            text = f"{label_text} {sc:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness // 2))
            bx1, by1 = x1, max(0, y1 - th - 6)
            bx2, by2 = x1 + tw + 8, y1
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(img, text, (bx1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, thickness // 2), lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        out_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    return out_images


def save_yolo_labels(path, boxes, labels, scores, w, h, thrh, allowed_classes=None, remap_map=None):
    """保存为 YOLO txt 标签（cls cx cy w h，归一化）"""
    lines = []
    for b, lab_t, sc in zip(boxes, labels, scores):
        lab = int(lab_t.item())
        if sc <= thrh:
            continue
        if allowed_classes and lab not in allowed_classes:
            continue
        if remap_map is not None:
            val = remap_map.get(lab, lab)
            if isinstance(val, int):
                lab = val
        x1, y1, x2, y2 = b.tolist()
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{lab} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_voc_labels(path, img_filename, boxes, labels, scores, w, h, thrh, allowed_classes=None, remap_map=None):
    """保存为 VOC XML 标签（数值类别）"""
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = os.path.basename(img_filename)
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    for b, lab_t, sc in zip(boxes, labels, scores):
        lab = int(lab_t.item())
        if sc <= thrh:
            continue
        if allowed_classes and lab not in allowed_classes:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        obj = ET.SubElement(ann, "object")
        name_val = remap_map.get(lab, lab) if remap_map is not None else lab
        ET.SubElement(obj, "name").text = str(name_val)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(max(0, x1))
        ET.SubElement(bb, "ymin").text = str(max(0, y1))
        ET.SubElement(bb, "xmax").text = str(min(w, x2))
        ET.SubElement(bb, "ymax").text = str(min(h, y2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree = ET.ElementTree(ann)
    tree.write(path, encoding="utf-8")


def process_image(model, device, file_path, visualize=False, out_vis_dir=None, label_format="none", out_label_dir=None, score_thresh=0.4, allowed_classes=None, remap_map=None):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    labels, boxes, scores = model(im_data, orig_size)

    if visualize and out_vis_dir is not None:
        vis_list = draw([im_pil], labels, boxes, scores, thrh=score_thresh, allowed_classes=allowed_classes, remap_map=remap_map)
        os.makedirs(out_vis_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        vis_list[0].save(os.path.join(out_vis_dir, base + ".jpg"))

    if label_format != "none" and out_label_dir is not None:
        os.makedirs(out_label_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        if label_format == "yolo":
            save_yolo_labels(os.path.join(out_label_dir, base + ".txt"), boxes[0], labels[0], scores[0], w, h, score_thresh, allowed_classes=allowed_classes, remap_map=remap_map)
        elif label_format == "voc":
            save_voc_labels(os.path.join(out_label_dir, base + ".xml"), file_path, boxes[0], labels[0], scores[0], w, h, score_thresh, allowed_classes=allowed_classes, remap_map=remap_map)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        vis_list = draw([frame_pil], labels, boxes, scores)

        frame = cv2.cvtColor(np.array(vis_list[0]), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def collect_image_paths(input_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isdir(input_path):
        return [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[-1].lower() in exts
        ]
    if os.path.isfile(input_path) and os.path.splitext(input_path)[-1].lower() == ".txt":
        with open(input_path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    if os.path.isfile(input_path) and os.path.splitext(input_path)[-1].lower() in exts:
        return [input_path]
    return []


def resolve_remap(remap_arg):
    """解析重映射参数为 label_index->(name|id) 的字典

    支持：
    - None：使用 COCO 默认映射（label_index -> 类别名称）
    - 文件路径：
      * .txt：每行一个类别名称，按行号（0-based）映射
      * .json：字典，严格按提供的键值对使用
    - JSON 字符串：与 .json 文件相同逻辑
    """
    import json
    if remap_arg is None:
        return {li: mscoco_category2name[mscoco_label2category[li]] for li in mscoco_label2category}
    if isinstance(remap_arg, str) and os.path.exists(remap_arg):
        if remap_arg.lower().endswith('.txt'):
            with open(remap_arg, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return {i: lines[i] for i in range(len(lines))}
        if remap_arg.lower().endswith('.json'):
            with open(remap_arg, 'r') as f:
                d = json.load(f)
            return d
    if isinstance(remap_arg, str):
        try:
            d = json.loads(remap_arg)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {li: mscoco_category2name[mscoco_label2category[li]] for li in mscoco_label2category}


def main(args):
    """Main function"""
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
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    file_path = args.input
    paths = collect_image_paths(file_path)
    if len(paths) > 0:
        out_dir = args.output_dir
        vis_dir = os.path.join(out_dir, "vis") if args.visualize else None
        label_dir = os.path.join(out_dir, "labels") if args.label_format != "none" else None
        remap_map = resolve_remap(args.remap)
        for p in paths:
            process_image(
                model,
                device,
                p,
                visualize=args.visualize,
                out_vis_dir=vis_dir,
                label_format=args.label_format,
                out_label_dir=label_dir,
                score_thresh=args.score_thresh,
                allowed_classes=args.classe,
                remap_map=remap_map,
            )
        print(f"Processed {len(paths)} images.")
    else:
        process_video(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, default="/media/ai/Data_SSD1/yql/datasets/kc/customized/nmsr/old/images")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="./output/torch_inf")
    parser.add_argument("--label-format", type=str, choices=["none", "voc", "yolo"], default="none")
    parser.add_argument("--score-thresh", type=float, default=0.6)
    parser.add_argument("--classe", type=int, nargs='*', default=[0], help="可选类别过滤列表（整数 ID），为空表示不过滤")
    parser.add_argument("--remap", type=str, default=None, help="类别重映射：支持 txt(每行一个类别名)、json(字典)，或 JSON 字符串；为空使用 COCO 默认映射")
    args = parser.parse_args()
    main(args)
    
def resolve_remap(remap_arg):
    """解析重映射参数为 label_index->(name|id) 的字典

    支持：
    - None：使用 COCO 默认映射（label_index -> 类别名称）
    - 文件路径：
      * .txt：每行一个类别名称，按行号（0-based）映射
      * .json：字典，若键为 label_index 直接映射；若键为 COCO 类别ID（1..90），则通过 `mscoco_label2category` 转换
    - JSON 字符串：与 .json 文件相同逻辑
    """
    import json
    if remap_arg is None:
        return {li: mscoco_category2name[mscoco_label2category[li]] for li in mscoco_label2category}
    # 文件路径
    if isinstance(remap_arg, str) and os.path.exists(remap_arg):
        if remap_arg.lower().endswith('.txt'):
            with open(remap_arg, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return {i: lines[i] for i in range(len(lines))}
        if remap_arg.lower().endswith('.json'):
            with open(remap_arg, 'r') as f:
                d = json.load(f)
            return d
    # 尝试解析 JSON 字符串
    if isinstance(remap_arg, str):
        try:
            d = json.loads(remap_arg)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    # 兜底默认
    return {li: mscoco_category2name[mscoco_label2category[li]] for li in mscoco_label2category}
