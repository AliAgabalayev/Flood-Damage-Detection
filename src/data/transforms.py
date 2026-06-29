from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    import albumentations as A
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:  
    _ALBUMENTATIONS_AVAILABLE = False
    A = None  



class Sen1FloodTransform:

    def __init__(self, albu_transform: Optional[object] = None) -> None:
        if albu_transform is not None and not _ALBUMENTATIONS_AVAILABLE:
            raise RuntimeError(
                "albumentations is required for Sen1FloodTransform but is not installed. "
                "Install it with: pip install albumentations"
            )
        self._transform: Optional[object] = albu_transform


    def __call__(
        self,
        image: Tensor,
        label: Tensor,
        valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self._transform is None:
            return image, label, valid_mask

        # image: (C, H, W) -> (H, W, C)
        image_np: np.ndarray = image.numpy().transpose(1, 2, 0)          # (H, W, C)
        label_np: np.ndarray = label.numpy().astype(np.int32)            # (H, W)
        mask_np:  np.ndarray = valid_mask.numpy().astype(np.uint8)       # (H, W)

        # Apply geometric transform — label and valid_mask ride as "additional_targets"
        result = self._transform( 
            image=image_np,
            mask=label_np,
            valid_mask=mask_np,
        )

        # Back to (C, H, W) tensors
        image_out = torch.from_numpy(
            result["image"].transpose(2, 0, 1).astype(np.float32)
        )
        label_out = torch.from_numpy(result["mask"].astype(np.int64)).long()
        mask_out  = torch.from_numpy(result["valid_mask"].astype(bool)).bool()

        return image_out.float(), label_out.long(), mask_out.bool()



def build_train_transforms(image_size: int = 512) -> Optional[Sen1FloodTransform]:
    
    if _ALBUMENTATIONS_AVAILABLE:
         pipeline = A.Compose(
             [
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 A.RandomCrop(height=image_size, width=image_size, p=1.0),
             ],
             additional_targets={"valid_mask": "mask"},
         )
         return Sen1FloodTransform(pipeline)
    
    _ = image_size  
    return None


def build_val_transforms() -> Optional[Sen1FloodTransform]:
    """Build a validation/test transform pipeline (identity by default).

    Validation and test sets use no augmentation; this factory exists so that
    the DataModule can call it symmetrically with :func:`build_train_transforms`
    and future non-geometric pre-processing (e.g. padding to a fixed size) can
    be added here without touching the DataModule.

    Returns:
        ``None`` (identity / no-op).
    """
    return None
