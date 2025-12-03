from typing import Optional, List, Dict, Any
import os
import time
import hashlib
import logging
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

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
        min_size: int = 10, 
    ):
        self.list_file = list_file
        self.transforms = transforms
        self.min_size = min_size
        
        # Read image list
        with open(list_file, "r") as f:
            self.all_images: List[str] = [ln.strip() for ln in f.readlines() if ln.strip()]
            
        # Cache logic
        self.cache_dir = os.path.dirname(list_file)
        list_filename = os.path.basename(list_file).split('.')[0]
        self.cache_path = os.path.join(self.cache_dir, f"{list_filename}.cache")
        
        self.images = self._check_and_load_cache()

    def _check_and_load_cache(self) -> List[str]:
        """
        Check if cache exists and is valid. If so, load it.
        Otherwise, scan dataset, verify labels, and save cache.
        """
        if os.path.exists(self.cache_path):
            try:
                cache = torch.load(self.cache_path)
                # 1. Check if image list itself changed (fast check)
                list_hash = self._get_list_content_hash()
                if cache.get("list_hash") != list_hash:
                     print("Image list changed, rescanning...")
                else:
                     # 2. Check if any label file changed (slower check, but necessary)
                     print("Checking label files for updates...")
                     labels_hash = self._get_labels_state_hash()
                     if cache.get("labels_hash") == labels_hash and cache.get("version") == 1.0:
                         print(f"Loaded dataset cache from {self.cache_path}")
                         return cache["images"]
                     else:
                         print("Label files updated or mismatch, rescanning...")
            except Exception as e:
                print(f"Failed to load cache: {e}, rescanning...")
        
        # Rescan and verify
        valid_images = self._scan_and_verify()
        
        # Recalculate hashes for the new valid set (or the original full set? usually original list defines the scope)
        # We should store the hash of the INPUT list, so next time we verify against the same INPUT list.
        list_hash = self._get_list_content_hash()
        labels_hash = self._get_labels_state_hash()
        
        # Save cache
        cache_data = {
            "list_hash": list_hash,
            "labels_hash": labels_hash,
            "version": 1.0,
            "images": valid_images,
        }
        try:
            torch.save(cache_data, self.cache_path)
            print(f"Saved dataset cache to {self.cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save dataset cache: {e}")
            
        return valid_images

    def _get_list_content_hash(self) -> str:
        """Hash of the image list file content"""
        return hashlib.md5("".join(self.all_images).encode('utf-8')).hexdigest()

    def _get_labels_state_hash(self) -> str:
        """
        Calculate hash based on label files' mtime and size.
        This ensures any modification to label files invalidates cache.
        """
        hash_content = []
        # We use tqdm here because for large datasets this stat operation can take a moment
        for img_path in tqdm(self.all_images, desc="Checking file states", leave=False):
            label_path = self._label_path_from_image(img_path)
            try:
                stat = os.stat(label_path)
                # Use size and mtime for uniqueness
                hash_content.append(f"{stat.st_size}_{stat.st_mtime}")
            except FileNotFoundError:
                # If label missing, mark as missing in hash
                hash_content.append("missing")
        
        return hashlib.md5("".join(hash_content).encode('utf-8')).hexdigest()

    def _scan_and_verify(self) -> List[str]:
        print(f"Scanning and verifying {len(self.all_images)} images...")
        valid_images = []
        invalid_count = 0
        empty_label_count = 0  # Count for background images (no label file or empty file)
        
        # Use tqdm for progress bar
        pbar = tqdm(self.all_images, desc="Verifying data")
        
        for img_path in pbar:
            try:
                is_valid, is_empty = self._verify_item(img_path)
                if is_valid:
                    valid_images.append(img_path)
                    if is_empty:
                        empty_label_count += 1
                else:
                    invalid_count += 1
            except Exception as e:
                print(f"\nError verifying {img_path}: {e}")
                invalid_count += 1
                
        print(f"\nScan completed. Valid: {len(valid_images)}, Invalid: {invalid_count}")
        print(f"Background images (no labels): {empty_label_count}")
        return valid_images

    def _verify_item(self, image_path: str) -> tuple[bool, bool]:
        """
        Returns:
            (is_valid, is_empty_label)
        """
        # 1. Check image existence
        if not os.path.exists(image_path):
            print(f"\n[Missing Image] {image_path}")
            return False, False
            
        try:
            # Verify image file integrity (Fast check: only read header)
            # We avoid img.verify() as it reads the whole file which is slow.
            # Just opening and getting size is enough to catch empty/non-image files.
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"\n[Corrupt Image] {image_path}: {e}")
            return False, False
            
        # 2. Check label existence
        label_path = self._label_path_from_image(image_path)
        if not os.path.exists(label_path):
            # Missing label file -> Background image -> Valid
            return True, True

        # 3. Verify label content
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            # If file is empty, it's a valid background image
            if not lines:
                return True, True
                
            for i, ln in enumerate(lines):
                ln = ln.strip()
                if not ln:
                    continue
                    
                parts = ln.split()
                if len(parts) < 5:
                    print(f"\n[Invalid Format] Line {i+1} in {label_path}: insufficient parts")
                    return False, False
                    
                # Parse
                try:
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    print(f"\n[Parse Error] Line {i+1} in {label_path}: value error")
                    return False, False
                
                # Check 1: Normalized range [0, 1] with small tolerance
                # Allowing slightly out of bound due to floating point issues or augmentation
                tol = 1e-6
                if not (-tol <= cx <= 1+tol and -tol <= cy <= 1+tol and 
                        0 < bw <= 1+tol and 0 < bh <= 1+tol):
                    print(f"\n[Out of Bounds] {label_path} Line {i+1}: cx={cx}, cy={cy}, w={bw}, h={bh}")
                    return False, False
                    
                # Check 2: Box validity (xmin < xmax, ymin < ymax) implicitly checked by w>0, h>0
                # Convert to pixel coords to be sure
                w_px = bw * w
                h_px = bh * h
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                
                # Check 3: Min size check
                if w_px < self.min_size or h_px < self.min_size:
                    print(f"\n[Small Box] {label_path} Line {i+1}: size={w_px:.1f}x{h_px:.1f} < min_size={self.min_size}")
                    return False, False

                # Check 4: Pixel boundary check (strict)
                # Allow small tolerance for rounding
                if x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1:
                     print(f"\n[Box Out of Image] {label_path} Line {i+1}: box={x1,y1,x2,y2}, img={w,h}")
                     return False, False
                
            return True, False # Valid and Not Empty

        except Exception as e:
             print(f"\n[Read Error] {label_path}: {e}")
             return False, False

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

