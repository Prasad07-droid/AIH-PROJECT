"""Visit-to-visit wound area trends and optional ORB visual alignment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AreaTrend:
    """Signed area changes and status compared with baseline/previous visit."""

    change_from_baseline_pct: float | None
    change_from_previous_pct: float | None
    reduction_from_baseline_pct: float | None
    healing_rate_cm2_per_day: float | None
    status_label: str


def _percent_change(current: float, reference: float) -> float | None:
    """Return signed percent change (negative indicates area reduction)."""
    if reference == 0:
        return None
    return (current - reference) / reference * 100.0


def area_status(reduction_from_baseline_pct: float | None) -> str:
    """Map area reduction versus baseline to the requested simple status label."""
    if reduction_from_baseline_pct is None:
        return "Stagnant"
    if reduction_from_baseline_pct >= 50.0:
        return "Healing well"
    if reduction_from_baseline_pct >= 10.0:
        return "Slow progress"
    if reduction_from_baseline_pct >= -10.0:
        return "Stagnant"
    return "Worsening, consult a doctor"


def compare_areas(
    current_area_cm2: float,
    baseline_area_cm2: float,
    previous_area_cm2: float | None = None,
    days_since_previous: float | None = None,
) -> AreaTrend:
    """Calculate percent changes, healing rate, and baseline status.

    Percent-change fields use the common signed convention: a negative value
    means the current area is smaller. ``reduction_from_baseline_pct`` is the
    inverse sign used for the requested status thresholds. Healing rate is
    positive when area decreases, in cm²/day.
    """
    values = [current_area_cm2, baseline_area_cm2]
    if previous_area_cm2 is not None:
        values.append(previous_area_cm2)
    if not all(np.isfinite(value) and value >= 0 for value in values):
        raise ValueError("areas must be finite, non-negative numbers")

    baseline_change = _percent_change(current_area_cm2, baseline_area_cm2)
    reduction = -baseline_change if baseline_change is not None else None
    previous_change = (
        _percent_change(current_area_cm2, previous_area_cm2)
        if previous_area_cm2 is not None
        else None
    )

    healing_rate: float | None = None
    if previous_area_cm2 is not None and days_since_previous is not None:
        if not np.isfinite(days_since_previous) or days_since_previous <= 0:
            raise ValueError("days_since_previous must be positive")
        healing_rate = (previous_area_cm2 - current_area_cm2) / days_since_previous

    return AreaTrend(
        change_from_baseline_pct=baseline_change,
        change_from_previous_pct=previous_change,
        reduction_from_baseline_pct=reduction,
        healing_rate_cm2_per_day=healing_rate,
        status_label=area_status(reduction),
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image to OpenCV grayscale."""
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        gray = image_array
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"image must be grayscale or RGB, got shape {image_array.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def align_later_to_earlier(
    earlier_rgb: np.ndarray,
    later_rgb: np.ndarray,
) -> np.ndarray | None:
    """Align a later image to an earlier one with ORB + homography.

    This is for visual side-by-side comparison only, not for area calculation.
    Returns ``None`` silently when there are too few reliable feature matches
    or a stable homography cannot be estimated.
    """
    earlier = np.asarray(earlier_rgb)
    later = np.asarray(later_rgb)
    if earlier.ndim not in (2, 3) or later.ndim not in (2, 3):
        return None
    if earlier.size == 0 or later.size == 0:
        return None
    try:
        earlier_gray = _to_gray(earlier)
        later_gray = _to_gray(later)
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints_earlier, descriptors_earlier = orb.detectAndCompute(earlier_gray, None)
        keypoints_later, descriptors_later = orb.detectAndCompute(later_gray, None)
        if descriptors_earlier is None or descriptors_later is None:
            return None

        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
            descriptors_later, descriptors_earlier, k=2
        )
        good_matches = [
            first
            for pair in matches
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]
        if len(good_matches) < 8:
            return None

        source_points = np.float32(
            [keypoints_later[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        target_points = np.float32(
            [keypoints_earlier[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if homography is None or inlier_mask is None:
            return None
        inliers = int(inlier_mask.sum())
        if inliers < 8 or inliers / len(good_matches) < 0.25:
            return None

        height, width = earlier_gray.shape[:2]
        warped = cv2.warpPerspective(later, homography, (width, height))
        return warped
    except (cv2.error, ValueError, TypeError):
        return None
