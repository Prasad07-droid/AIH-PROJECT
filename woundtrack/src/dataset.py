"""PyTorch dataset, image normalization, and augmentations for wound masks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from woundtrack.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
)

SplitName = Literal["train", "val", "test"]


def build_augmentations(train: bool = True) -> A.Compose:
    """Build spatially-safe augmentations and ImageNet normalization.

    Albumentations applies spatial transforms to the image and mask together;
    photometric transforms and normalization affect the image only.
    """
    transforms: list[A.BasicTransform] = []
    if train:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.3,
            ),
            A.RandomBrightnessContrast(p=0.35),
            A.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.15,
                hue=0.03,
                p=0.25,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ]
    transforms.append(
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        )
    )
    # A fixed seed makes data-loader worker augmentation repeatable when the
    # caller also seeds PyTorch and NumPy.
    return A.Compose(transforms, seed=RANDOM_SEED)


def resize_mask_to_original(
    mask: np.ndarray,
    original_shape: Sequence[int],
) -> np.ndarray:
    """Resize a binary 2-D mask back to an image's original height/width.

    Nearest-neighbor interpolation preserves the binary boundary labels.
    ``original_shape`` may be an ``(height, width)`` pair or an image shape
    such as ``(height, width, channels)``. The result is uint8 with values 0/1.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2-D (or singleton-channel), got {mask_array.shape}")
    if len(original_shape) < 2:
        raise ValueError("original_shape must include height and width")

    height, width = int(original_shape[0]), int(original_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"original height/width must be positive, got {(height, width)}")
    if mask_array.shape == (height, width):
        return (mask_array > 0).astype(np.uint8)

    binary_mask = (mask_array > 0).astype(np.uint8)
    restored = cv2.resize(
        binary_mask,
        dsize=(width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (restored > 0).astype(np.uint8)


class WoundSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load the prepared 256x256 WoundTrack image/mask pairs.

    ``split`` is one of ``train``, ``val`` or ``test``. Images are returned as
    ImageNet-normalized float tensors shaped ``[3, H, W]``; masks are binary
    float tensors shaped ``[1, H, W]``. Training augmentation is enabled only
    for the training split unless ``augment`` is explicitly supplied.
    """

    def __init__(
        self,
        split: SplitName,
        *,
        augment: bool | None = None,
        images_dir: Path = PROCESSED_IMAGES_DIR,
        masks_dir: Path = PROCESSED_MASKS_DIR,
        splits_dir: Path = SPLITS_DIR,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"unsupported split {split!r}; use train, val, or test")

        self.split = split
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        split_file = Path(splits_dir) / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"Missing split file {split_file}; run "
                "`python -m woundtrack.src.prepare_data` first."
            )
        self.sample_ids = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.sample_ids:
            raise ValueError(f"split {split!r} is empty: {split_file}")

        use_augmentation = split == "train" if augment is None else augment
        self.transform = build_augmentations(train=use_augmentation)

    def __len__(self) -> int:
        """Return the number of image/mask pairs in this split."""
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Read, augment, normalize, and convert one pair to PyTorch tensors."""
        sample_id = self.sample_ids[index]
        image_path = self.images_dir / f"{sample_id}.png"
        mask_path = self.masks_dir / f"{sample_id}.png"

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")
        if image_bgr.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"image/mask shape mismatch for {sample_id}: "
                f"{image_bgr.shape[:2]} != {mask.shape[:2]}"
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        binary_mask = (mask > 0).astype(np.uint8)
        transformed = self.transform(image=image_rgb, mask=binary_mask)
        image_array = np.ascontiguousarray(
            transformed["image"].transpose(2, 0, 1), dtype=np.float32
        )
        mask_array = np.ascontiguousarray(
            (transformed["mask"] > 0).astype(np.float32)[None, ...]
        )
        return torch.from_numpy(image_array), torch.from_numpy(mask_array)
