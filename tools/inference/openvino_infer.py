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
- `-i/--input` ：
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
import time

import cv2  # Added for video processing
import numpy as np
import openvino
from openvino.runtime import Core
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_label2category, mscoco_category2name


class OvInfer:
    def __init__(self, model_path, device="CPU", 
                 label_format="voc", out_vis_dir=None, out_label_dir=None, remap_map=None):
        self.resized_image = None
        self.ratio = None
        self.pad_w = None
        self.pad_h = None
        self.ori_image = None
        self.device = device
        self.model_path = model_path
        self.core = Core()
        self.available_device = self.core.available_devices
        
        # 根据模型文件后缀判断数据类型
        model_name = os.path.basename(model_path).lower()
        model_ext = os.path.splitext(model_path)[1].lower()
        
        if "int8" in model_name:
            self.data_type = "int8"
            self.normalize = True
            self.input_dtype = np.float32
        elif "fp16" in model_name:
            self.data_type = "fp16"
            self.normalize = True
            self.input_dtype = np.float16
        else:
            # 默认FP32
            self.data_type = "fp32"
            self.normalize = True
            self.input_dtype = np.float32
        
        print(f"检测到模型类型: {self.data_type}, 归一化: {self.normalize}")
        
        # 模型初始化计时
        start_time = time.time()
        # 显式设置性能建议，让 OpenVINO 自动优化线程和调度
        # "PERFORMANCE_HINT": "LATENCY" 适合实时性要求高的场景（单张推理）
        # "PERFORMANCE_HINT": "THROUGHPUT" 适合批量处理
        config = {"PERFORMANCE_HINT": "LATENCY"}
        self.compile_model = self.core.compile_model(self.model_path, config=config)
        self.init_time = time.time() - start_time
        
        self.target_size = [
            self.compile_model.inputs[0].get_partial_shape()[2].get_length(),
            self.compile_model.inputs[0].get_partial_shape()[3].get_length(),
        ]
        self.query_num = self.compile_model.outputs[0].get_partial_shape()[1].get_length()
        self.label_format = label_format
        self.out_vis_dir = out_vis_dir
        self.out_label_dir = out_label_dir
        self.remap_map = remap_map
        
        # 计时统计
        self.preprocess_times = []
        self.inference_times = []
        self.postprocess_times = []
        self.io_times = []

    def infer(self, inputs: dict):
        infer_request = self.compile_model.create_infer_request()
        for input_name, input_data in inputs.items():
            input_tensor = openvino.Tensor(input_data)
            infer_request.set_tensor(input_name, input_tensor)
        infer_request.infer()
        return infer_request.get_tensor("labels").data, infer_request.get_tensor("boxes").data, infer_request.get_tensor("scores").data
    
    def postprocess(self, boxes, scores, labels, keep_ratio: bool, score_thresh: float):
        """后处理函数，根据 keep_ratio 选择是否修正框坐标"""
        if keep_ratio:
            # 修正框坐标
            boxes = correct_box(boxes, self.ratio, self.pad_w, self.pad_h)
        # boxes = correct_box(boxes, self.ratio, self.pad_w, self.pad_h)
        # 过滤低置信度框
        mask = scores >= score_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        return boxes, scores, labels
    
    def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
        """Resizes an image while maintaining aspect ratio and pads it."""
        original_width, original_height = image.size
        ratio = min(size / original_width, size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        image = image.resize((new_width, new_height), interpolation)

        # Create a new image with the desired size and paste the resized image onto it
        new_image = Image.new("RGB", (size, size))
        new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
        return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2
        
    def process_image(self, image_path, keep_ratio: bool = True, visualize: bool = True, allowed_classes=None, score_thresh: float = 0.6):
        """处理单张图片，包括缩放、填充与归一化"""
        # 预处理计时
        preprocess_start = time.time()
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        self.ori_image = np.array(image)
        h, w = self.ori_image.shape[:2]
        
        # INT8模型使用与export脚本相同的预处理（无aspect ratio padding）
        if self.data_type == "int8":
            # 使用CV2进行resize，与export脚本保持一致
            self.resized_image = cv2.resize(
                self.ori_image,
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            # keep_ratio = False  # INT8模型不使用aspect ratio padding
            keep_ratio = True
        if keep_ratio:
            self.resized_image, self.ratio, self.pad_w, self.pad_h = resize_with_aspect_ratio(
                image, self.target_size[0], interpolation=Image.BILINEAR
            )
        else:
            self.resized_image = cv2.resize(
                self.ori_image,
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        
        orig_size = np.array([self.target_size[1], self.target_size[0]], dtype=np.float32).reshape(
            1, 2
        )
        
        # 根据数据类型决定是否进行归一化
        if self.normalize:
            # FP32/FP16模型需要归一化到[0,1]
            image_data = np.array(self.resized_image).astype(np.float32) / 255.0
        else:
            # INT8模型不需要归一化，保持0-255范围
            image_data = np.array(self.resized_image).astype(np.float32)
        
        inputs = {
            "images": np.transpose(image_data, (2, 0, 1))[None, :],  # 构造 NCHW: (1,3,320,320)
            "orig_target_sizes": orig_size,
        }
        
        preprocess_time = time.time() - preprocess_start
        self.preprocess_times.append(preprocess_time)
        
        # 推理计时
        inference_start = time.time()
        outputs = self.infer(inputs)
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        
        # 后处理计时
        postprocess_start = time.time()
        labels, boxes, scores = outputs
        boxes, scores, labels = self.postprocess(boxes, scores, labels, keep_ratio, score_thresh)
        postprocess_time = time.time() - postprocess_start
        self.postprocess_times.append(postprocess_time)
        
        # IO操作计时
        io_start = time.time()
        
        if visualize and self.out_vis_dir is not None:
            vis_list = draw(self.ori_image, labels, boxes, scores, thrh=score_thresh, allowed_classes=allowed_classes, remap_map=self.remap_map)
            os.makedirs(self.out_vis_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            vis_list.save(os.path.join(self.out_vis_dir, base + ".jpg"))

        if self.label_format != "none" and self.out_label_dir is not None:
            os.makedirs(self.out_label_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            if self.label_format == "yolo":
                save_yolo_labels(os.path.join(self.out_label_dir, base + ".txt"), boxes, labels, scores, 
                                 w, h, score_thresh, allowed_classes=allowed_classes, remap_map=self.remap_map)
            elif self.label_format == "voc":    
                save_voc_labels(os.path.join(self.out_label_dir, base + ".xml"), os.path.basename(image_path), boxes, labels, scores, 
                                w, h, score_thresh, allowed_classes=allowed_classes, remap_map=self.remap_map)
        
        io_time = time.time() - io_start
        self.io_times.append(io_time)
        
        return labels, boxes, scores
        

    def get_available_device(self):
        return self.available_device
    
    def print_timing_stats(self):
        """打印计时统计信息"""
        if not self.preprocess_times:
            return
            
        total_images = len(self.preprocess_times)
        
        print("\n" + "="*60)
        print("OpenVINO 推理性能统计")
        print("="*60)
        print(f"模型初始化时间: {self.init_time:.4f} 秒")
        print(f"处理图片数量: {total_images}")
        print("-"*60)
        
        # 计算平均值
        avg_preprocess = sum(self.preprocess_times) / total_images
        avg_inference = sum(self.inference_times) / total_images
        avg_postprocess = sum(self.postprocess_times) / total_images
        avg_io = sum(self.io_times) / total_images
        
        # 计算FPS
        fps = 1.0 / (avg_preprocess + avg_inference + avg_postprocess)
        
        print(f"平均预处理时间: {avg_preprocess:.4f} 秒")
        print(f"平均推理时间: {avg_inference:.4f} 秒")
        print(f"平均后处理时间: {avg_postprocess:.4f} 秒")
        print(f"平均IO操作时间: {avg_io:.4f} 秒")
        print("-"*60)
        print(f"平均单张图片总时间: {avg_preprocess + avg_inference + avg_postprocess + avg_io:.4f} 秒")
        print(f"推理FPS: {fps:.2f}")
        print("="*60)

def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2



def _filter_by_thresh_and_classes(labs, bxs, scrs, thrh, allowed_classes):
    """根据分数阈值与类别集合过滤检测结果

    返回：过滤后的 (labels, boxes, scores) 列表，长度一致
    """
    print(labs, bxs, scrs)
    labs_list = labs.tolist()
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

def correct_box(batch_boxes, ratio, pad_w, pad_h):
    """根据缩放比例与填充修正框坐标，支持 numpy 数组输入"""
    
    # 检查输入类型，确保是 numpy 数组
    if not isinstance(batch_boxes, np.ndarray):
        batch_boxes = np.array(batch_boxes)
    
    # 确保至少是2维数组
    if batch_boxes.ndim < 2:
        raise ValueError("batch_boxes 至少应该是2维数组")
    
    # 保存原始形状以便后续恢复
    original_shape = batch_boxes.shape
    
    # 重塑为 [num_boxes, 4] 的形状
    if batch_boxes.ndim > 2:
        batch_boxes = batch_boxes.reshape(-1, 4)
    
    # 向量化计算，一次性处理所有框
    batch_boxes[:, 0] = (batch_boxes[:, 0] - pad_w) / ratio
    batch_boxes[:, 1] = (batch_boxes[:, 1] - pad_h) / ratio
    batch_boxes[:, 2] = (batch_boxes[:, 2] - pad_w) / ratio
    batch_boxes[:, 3] = (batch_boxes[:, 3] - pad_h) / ratio
    
    # # 向量化计算，一次性处理所有框
    # batch_boxes[:, 0] = (batch_boxes[:, 0] - pad_w) / ratio
    # batch_boxes[:, 1] = (batch_boxes[:, 1] - pad_h) 
    # batch_boxes[:, 2] = (batch_boxes[:, 2] - pad_w) / ratio
    # batch_boxes[:, 3] = (batch_boxes[:, 3] - pad_h) 
    
    # 恢复原始形状
    batch_boxes = batch_boxes.reshape(original_shape)
    
    return batch_boxes


def draw(image, labels, boxes, scores, thrh=0.4, allowed_classes=None, remap_map=None):
    palette = [
        (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 214, 56), (255, 56, 214),
        (56, 214, 255), (255, 160, 56), (160, 56, 255), (56, 160, 255), (160, 255, 56),
    ]
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    thickness = max(2, int(round(min(h, w) / 300)))
    font_scale = max(0.6, thickness / 2)
    overlay = img.copy()

    labs_f, bxs_f, scrs_f = _filter_by_thresh_and_classes(labels, boxes, scores, thrh, allowed_classes)

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
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


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
    # Load the ONNX model
    out_dir = args.output_dir
    vis_dir = os.path.join(out_dir, "vis") if args.visualize else None
    label_dir = os.path.join(out_dir, "labels") if args.label_format != "none" else None
    remap_map = resolve_remap(args.remap)
    
    # 调试输出，验证remap参数是否正确解析
    print(f"Remap argument: {args.remap}")
    print(f"Parsed remap map: {remap_map}")
    
    OVDetector = OvInfer(args.model, label_format=args.label_format, device=args.device, 
                         out_vis_dir=vis_dir, out_label_dir=label_dir, remap_map=remap_map)

    file_path = args.input
    paths = collect_image_paths(file_path)
    if len(paths) > 0: 
        for p in paths[:]:
            try:
                OVDetector.process_image(
                    p,
                    visualize=args.visualize,
                    score_thresh=args.score_thresh,
                    allowed_classes=args.classe,
                )
            except Exception as e:
                print(f"Error processing image {p}: {e}")
                continue
        print(f"Processed {len(paths)} images.")
        
        # 打印计时统计
        OVDetector.print_timing_stats()
    else:
        raise ValueError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--model", type=str, default=r"D:\valid_data\num_merge\model\dfine\dfine_hgnetv2_s_custom_unified\model_int8.xml")
    parser.add_argument("-i", "--input", type=str, default=r"D:\valid_data\num_merge\0630_train\eval\images")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default=r"D:\valid_data\num_merge\0630_train\eval\test_result\dfine_s_openvino_int8")
    parser.add_argument("--label-format", type=str, choices=["none", "voc", "yolo"], default="voc")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--classe", type=int, nargs='*', default=[0], help="可选类别过滤列表（整数 ID），为空表示不过滤")
    parser.add_argument("--remap", type=str, default='{0: "kc"}', help="类别重映射：支持 txt(每行一个类别名)、json(字典)，或 JSON 字符串；为空使用 COCO 默认映射")
    args = parser.parse_args()
    main(args)
