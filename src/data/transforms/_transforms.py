"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        # Center padding
        left = w // 2
        right = w - left
        top = h // 2
        bottom = h - top
        self.padding = [left, top, right, bottom]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)
        self.fill = fill

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self.fill
        if isinstance(inpt, Mask):
            fill = 0
        elif isinstance(fill, dict):
            fill = fill.get(type(inpt), 0)
        
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        inpt = inputs if len(inputs) > 1 else inputs[0]
        image, target = inpt
        boxes = target.get("boxes")
        labels = target.get("labels")

        # Attempt to crop
        if boxes is not None and labels is not None:
            # Add index to boxes to track which ones are kept
            n = boxes.shape[0]
            if n == 0:
                return super().forward(*inputs)
                
            device = boxes.device
            indices = torch.arange(n, dtype=boxes.dtype, device=device).unsqueeze(1)
            # Use cat to append index column. 
            # We assume boxes are (N, 4). We make them (N, 5).
            # Note: We must wrap this back into BoundingBoxes to preserve metadata for torchvision transforms
            boxes_with_idx = torch.cat([boxes, indices], dim=1)
            
            # Wrap into BoundingBoxes (tv_tensor)
            # We temporarily treat the 5th column as part of the box data. 
            # Most torchvision transforms operate on the coordinates (first 4), 
            # but we need to check if RandomIoUCrop respects the extra dimension.
            # T.RandomIoUCrop usually calls crop, which handles tensors generically.
            from torchvision.tv_tensors import BoundingBoxes
            boxes_with_idx_tv = BoundingBoxes(
                boxes_with_idx, 
                format=boxes.format, 
                canvas_size=boxes.canvas_size
            )
            
            # Construct a temporary target for the transform
            temp_target = target.copy()
            temp_target["boxes"] = boxes_with_idx_tv
            
            try:
                # Perform the crop
                # super().forward accepts (image, target) or (image, boxes, labels) etc.
                # Here we pass (image, temp_target)
                out = super().forward(image, temp_target)
                
                # Unpack output
                new_image, new_target = out
                new_boxes_with_idx = new_target["boxes"]
                
                # Recover valid indices
                if len(new_boxes_with_idx) > 0:
                    # The last column is our index
                    kept_indices = new_boxes_with_idx[:, 4].long()
                    
                    # Filter labels
                    new_labels = labels[kept_indices]
                    
                    # Restore boxes to (N, 4)
                    new_boxes = new_boxes_with_idx[:, :4]
                    
                    # Re-wrap boxes to ensure correct metadata
                    new_boxes = BoundingBoxes(
                        new_boxes, 
                        format=new_boxes_with_idx.format, 
                        canvas_size=new_boxes_with_idx.canvas_size
                    )
                    
                    # Update target
                    target["boxes"] = new_boxes
                    target["labels"] = new_labels
                    
                    return new_image, target
                else:
                    # If all boxes are removed, return empty
                    target["boxes"] = BoundingBoxes(
                        torch.empty((0, 4), dtype=boxes.dtype, device=device),
                        format=boxes.format, 
                        canvas_size=boxes.canvas_size # Should be updated canvas size? 
                        # Actually RandomIoUCrop updates canvas_size in new_boxes_with_idx, so we should use that if available
                        # But here new_boxes_with_idx is empty.
                        # We can just use the new image size.
                    )
                    # Update canvas size based on new image
                    h, w = F.get_size(new_image)
                    target["boxes"].canvas_size = (h, w)
                    
                    target["labels"] = torch.empty((0,), dtype=labels.dtype, device=device)
                    return new_image, target

            except Exception as e:
                print(f"Warning: RandomIoUCrop failed with index hack, skipping crop. Error: {e}")
                return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)



@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt
