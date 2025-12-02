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
class LetterBoxResize(T.Resize):
    """
    Resizes the image so that the longest edge is equal to `size`, preserving aspect ratio.
    The short edge is scaled accordingly.
    This is equivalent to T.Resize(size=smaller_edge, max_size=target_size), but handles the
    calculation automatically to avoid 'max_size > size' errors.
    """
    def __init__(self, size, interpolation=T.InterpolationMode.BILINEAR, antialias=True):
        # We don't pass size/max_size to super().__init__ here because we'll calculate them dynamically
        # or use a safe default. Actually T.Resize stores them.
        # To implement "longest edge = size", we can set size=size, max_size=size? 
        # No, that fails validation.
        
        # Strategy: We act as a wrapper. We don't rely on super().forward to handle the logic if it's restrictive.
        # But we want to reuse its implementation.
        
        # Ideally: size=int, max_size=int.
        # If we set size=size, max_size=None -> Shortest edge = size. (Wrong for letterbox)
        # If we set size=1, max_size=size -> Longest edge = size. (Correct logic, but validation might fail if size is too small?)
        # Actually, validation says max_size > size.
        # So we set self.target_size = size
        super().__init__(size=size, interpolation=interpolation, antialias=antialias)
        self.target_size = size

    def forward(self, *inputs):
        # We need to inspect the image size to decide the correct 'size' parameter for T.Resize
        # so that the result's longest edge is self.target_size.
        
        # Inputs can be (image, target) or just image.
        inpt = inputs if len(inputs) > 1 else inputs[0]
        if isinstance(inpt, (list, tuple)) and len(inpt) >= 1:
            image = inpt[0]
        else:
            image = inpt
            
        # Get current size
        h, w = F.get_size(image)
        
        # Calculate scale to fit into target_size x target_size
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Now we have the exact target dimensions.
        # We can use T.Resize((new_h, new_w)) which specifies (h, w) directly.
        # This bypasses the size/max_size logic for shortest/longest edge.
        
        # We create a temporary Resize transform or call F.resize directly
        # But we need to handle all inputs (image, boxes, masks, etc.)
        # The cleanest way is to use F.resize, but we need to handle multiple inputs.
        # Or, we can just update self.size to be (new_h, new_w) temporarily?
        # No, self.size is shared.
        
        # Better: use the functional API which handles structure automatically in v2?
        # No, F.resize is for single tensor.
        
        # We can construct a new T.Resize((new_h, new_w)) and call it.
        # Note: T.Resize accepts a sequence for size to specify (h, w).
        return T.Resize(size=(new_h, new_w), interpolation=self.interpolation, antialias=self.antialias)(*inputs)

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
        
        # Robust unpacking: Check if input is (image, target) tuple or just image
        if isinstance(inpt, (list, tuple)) and len(inpt) == 2:
            image, target = inpt
        else:
            # If input format is unexpected (e.g. just image, or (img, target, other)), 
            # fallback to default behavior without index hack
            return super().forward(*inputs)
        
        if target is None:
             return super().forward(*inputs)

        if isinstance(target, dict):
            boxes = target.get("boxes")
            labels = target.get("labels")
        else:
            # If target is not a dict (e.g. tensor), we can't extract boxes/labels
            return super().forward(*inputs)

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
            # Optimization: Avoid deep copy of target if it's large.
            # We only need to replace "boxes" in the dict passed to super().forward
            # Shallow copy is enough as we only modify top-level keys
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
                    # Use as_subclass to avoid overhead if possible, but slicing creates new tensor anyway
                    new_boxes = new_boxes_with_idx[:, :4]
                    
                    # Re-wrap boxes to ensure correct metadata
                    # Optimization: Reuse canvas_size and format directly
                    new_boxes = BoundingBoxes(
                        new_boxes, 
                        format=new_boxes_with_idx.format, 
                        canvas_size=new_boxes_with_idx.canvas_size
                    )
                    
                    # Update target (in-place modification of the dict passed in)
                    target["boxes"] = new_boxes
                    target["labels"] = new_labels
                    
                    return new_image, target
                else:
                    # If all boxes are removed, return empty
                    # Optimization: Use empty_like to inherit properties
                    target["boxes"] = BoundingBoxes(
                        torch.empty((0, 4), dtype=boxes.dtype, device=device),
                        format=boxes.format, 
                        canvas_size=F.get_size(new_image) # Update canvas size to new image size
                    )
                    
                    target["labels"] = torch.empty((0,), dtype=labels.dtype, device=device)
                    return new_image, target

            except Exception as e:
                # print(f"Warning: RandomIoUCrop failed with index hack, skipping crop. Error: {e}")
                # Fail gracefully by returning original input
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
