"""CPU-first inference for wound masks using the trained U-Net checkpoint."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from woundtrack.config import (
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from woundtrack.src.dataset import resize_mask_to_original
from woundtrack.src.model import build_unet


def load_model(
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load the trained model for CPU inference.

    ``device`` is retained for compatibility, but inference is deliberately
    CPU-only regardless of the requested value.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}; run "
            "`python -m woundtrack.src.train` first."
        )
    model = build_unet(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Force CPU even when a caller requests an accelerator.
    model.to(torch.device("cpu"))
    model.eval()
    return model


def predict_mask(
    image_rgb: np.ndarray,
    model: nn.Module | None = None,
    *,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict a binary wound mask at the input image's original resolution.

    The returned array is uint8 with values 0/1. The U-Net operates on a
    256x256 view and uses ImageNet normalization; nearest-neighbor resizing
    maps its binary mask back to the original image height and width. ``device``
    is retained for compatibility but ignored: inference always runs on CPU.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape HxWx3, got {image.shape}")
    if image.size == 0:
        raise ValueError("image_rgb cannot be empty")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    original_shape = image.shape[:2]
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    image_float = resized.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (image_float - mean) / std
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)

    inference_model = model if model is not None else load_model(device="cpu")
    target_device = torch.device("cpu")
    inference_model.to(target_device)
    inference_model.eval()
    with torch.inference_mode():
        logits = inference_model(tensor.to(target_device))
        binary_small = (torch.sigmoid(logits)[0, 0] >= threshold).cpu().numpy().astype(np.uint8)
    return resize_mask_to_original(binary_small, original_shape)
