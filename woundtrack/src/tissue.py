"""Heuristic HSV tissue-color proportions within a wound segmentation mask."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TissueComposition:
    """Percent of wound-mask pixels assigned to each heuristic color group."""

    granulation_pct: float
    slough_pct: float
    necrotic_pct: float
    other_pct: float
    analyzed_pixels: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a simple serializable representation."""
        return asdict(self)


def analyze_tissue(image_rgb: np.ndarray, wound_mask: np.ndarray) -> TissueComposition:
    """Estimate granulation/slough/necrotic fractions using HSV thresholds.

    Thresholds are deliberately simple and are not a validated clinical
    tissue classifier. RGB input is expected; image lighting and skin tone
    strongly affect these heuristic estimates.
    """
    image = np.asarray(image_rgb)
    mask = np.asarray(wound_mask)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape HxWx3, got {image.shape}")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"wound_mask must be 2-D, got {mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image and wound mask must match spatially, got {image.shape[:2]} and {mask.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    inside = mask > 0
    total_pixels = int(inside.sum())
    if total_pixels == 0:
        return TissueComposition(0.0, 0.0, 0.0, 0.0, 0)

    # Priority masks keep the three percentages mutually exclusive.
    necrotic = inside & (value <= 55)
    slough = (
        inside
        & ~necrotic
        & (hue >= 15)
        & (hue <= 40)
        & (saturation >= 35)
        & (value >= 60)
    )
    red_or_pink = (hue <= 12) | (hue >= 160)
    granulation = (
        inside
        & ~necrotic
        & ~slough
        & red_or_pink
        & (saturation >= 25)
        & (value >= 60)
    )

    granulation_pct = 100.0 * int(granulation.sum()) / total_pixels
    slough_pct = 100.0 * int(slough.sum()) / total_pixels
    necrotic_pct = 100.0 * int(necrotic.sum()) / total_pixels
    other_pct = max(0.0, 100.0 - granulation_pct - slough_pct - necrotic_pct)
    return TissueComposition(
        granulation_pct=granulation_pct,
        slough_pct=slough_pct,
        necrotic_pct=necrotic_pct,
        other_pct=other_pct,
        analyzed_pixels=total_pixels,
    )
