import numpy as np
import random
import albumentations as A
from typing import Sequence, Tuple, Union

# Flip and brightness pipeline
_geom_tf = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,  # ±10% brightness
            contrast_limit=0.00,
            p=0.0
        ),
    ],
    additional_targets={"mask": "mask"},
    is_check_shapes=False,
)

class Augmenter:
    """
    Provides:
      1) Patch-wise mosaic (not implemented here)
      2) Random flip and brightness ±10%

    Args:
      mosaic_prob: probability of applying mosaic (0~1)
    """
    def __init__(self, mosaic_prob: float = 0.0):
        self.mosaic_prob = mosaic_prob # Not implemented in this snippet

    def __call__(
        self,
        imgs: Union[np.ndarray, Sequence[np.ndarray]],
        masks: Union[np.ndarray, Sequence[np.ndarray]],
        epoch: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        single = False
        if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
            imgs_list = [imgs]
            masks_list = [masks]  # type: ignore
            single = True
        else:
            imgs_list = list(imgs)  # type: ignore
            masks_list = list(masks)  # type: ignore

        out_imgs, out_masks = [], []
        for im, mk in zip(imgs_list, masks_list):
            aug = _geom_tf(image=im, mask=mk)
            out_imgs.append(aug["image"].astype(im.dtype))
            out_masks.append(aug["mask"].astype(mk.dtype))

        if single:
            return out_imgs[0], out_masks[0]
        return np.stack(out_imgs), np.stack(out_masks)
