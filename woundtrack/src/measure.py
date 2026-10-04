"""Binary-mask area, perimeter, and reference-marker calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

COIN_DIAMETER_CM = 2.5  # Indian one-rupee coin nominal diameter.
SQUARE_MARKER_SIDE_CM = 2.0
CalibrationSource = Literal["1-rupee coin", "2 cm square marker", "manual estimate"]


@dataclass(frozen=True)
class CalibrationResult:
    """Pixels-per-centimeter estimate and its source/trust label."""

    pixels_per_cm: float
    source: CalibrationSource
    calibrated: bool

    @property
    def display_label(self) -> str:
        """Return a user-facing calibration label."""
        if self.calibrated:
            return f"Calibrated from {self.source}"
        return "Uncalibrated / relative (manual pixels-per-cm estimate)"


@dataclass(frozen=True)
class AreaMeasurement:
    """Wound pixel area, optional scaled area, and perimeter values."""

    area_pixels: int
    area_cm2: float | None
    perimeter_pixels: float
    perimeter_cm: float | None
    pixels_per_cm: float | None
    calibrated: bool
    calibration_method: str

    @property
    def display_label(self) -> str:
        """Return a clear label for the measurement's scale reliability."""
        if self.calibrated:
            return "Calibrated area"
        return "Uncalibrated / relative"


def _as_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image to uint8 grayscale safely."""
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"image must be grayscale or RGB, got shape {image.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def detect_square_marker_pixels_per_cm(
    image_rgb: np.ndarray,
    side_cm: float = SQUARE_MARKER_SIDE_CM,
) -> CalibrationResult | None:
    """Detect a printed square marker and estimate pixels per centimeter.

    Looks for a convex quadrilateral with approximately equal opposing side
    lengths in Otsu-thresholded and edge maps. This is a pragmatic CV detector;
    users should ensure the full marker is visible and approximately face-on.
    """
    if side_cm <= 0:
        raise ValueError("square marker side length must be positive")
    gray = _as_gray(image_rgb)
    min_side = max(18.0, min(gray.shape) * 0.025)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_map = cv2.Canny(blurred, 50, 150)
    candidates = (threshold, cv2.bitwise_not(threshold), edge_map)

    best_side_length: float | None = None
    best_area = 0.0
    border_margin = max(3, int(round(min(gray.shape) * 0.01)))
    height, width = gray.shape[:2]
    for candidate in candidates:
        contours, _ = cv2.findContours(candidate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter <= 0:
                continue
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points = polygon.reshape(4, 2).astype(np.float32)
            if (
                points[:, 0].min() < border_margin
                or points[:, 1].min() < border_margin
                or points[:, 0].max() >= width - border_margin
                or points[:, 1].max() >= height - border_margin
            ):
                continue
            side_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
            if float(side_lengths.min()) < min_side:
                continue
            opposing_ratio = max(
                float(np.mean(side_lengths[[0, 2]])),
                float(np.mean(side_lengths[[1, 3]])),
            ) / max(
                min(
                    float(np.mean(side_lengths[[0, 2]])),
                    float(np.mean(side_lengths[[1, 3]])),
                ),
                1e-6,
            )
            if opposing_ratio > 1.35:
                continue
            contour_area = float(cv2.contourArea(polygon))
            if contour_area <= best_area:
                continue
            best_area = contour_area
            best_side_length = float(np.mean(side_lengths))

    if best_side_length is None:
        return None
    return CalibrationResult(
        pixels_per_cm=best_side_length / side_cm,
        source="2 cm square marker",
        calibrated=True,
    )


def detect_coin_pixels_per_cm(
    image_rgb: np.ndarray,
    diameter_cm: float = COIN_DIAMETER_CM,
) -> CalibrationResult | None:
    """Use Hough circles to detect a face-on one-rupee coin in an RGB image."""
    if diameter_cm <= 0:
        raise ValueError("coin diameter must be positive")
    gray = _as_gray(image_rgb)
    if min(gray.shape) < 32:
        return None
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
    min_radius = max(8, int(round(min(gray.shape) * 0.015)))
    max_radius = max(min_radius + 1, int(round(min(gray.shape) * 0.20)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(gray.shape) // 5),
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or circles.size == 0:
        return None
    detections = circles[0]
    # Prefer a strong, larger round reference object over small wound details.
    radius = float(max(detections, key=lambda circle: float(circle[2]))[2])
    return CalibrationResult(
        pixels_per_cm=(2.0 * radius) / diameter_cm,
        source="1-rupee coin",
        calibrated=True,
    )


def detect_reference_scale(image_rgb: np.ndarray) -> CalibrationResult | None:
    """Try the 2 cm square marker and then a 1-rupee coin."""
    square = detect_square_marker_pixels_per_cm(image_rgb)
    if square is not None:
        return square
    return detect_coin_pixels_per_cm(image_rgb)


def manual_calibration(pixels_per_cm: float) -> CalibrationResult:
    """Create a manual scale estimate that remains visibly uncalibrated."""
    if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        raise ValueError("pixels_per_cm must be a finite positive number")
    return CalibrationResult(
        pixels_per_cm=float(pixels_per_cm),
        source="manual estimate",
        calibrated=False,
    )


def _perimeter_pixels(binary_mask: np.ndarray) -> float:
    """Sum external contour perimeters for all wound components."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.arcLength(contour, closed=True) for contour in contours))


def measure_mask(
    mask: np.ndarray,
    calibration: CalibrationResult | None = None,
) -> AreaMeasurement:
    """Measure mask pixels and, when possible, convert to cm² and centimeters.

    Pixel area is the count of nonzero pixels. A manually entered scale produces
    a numerical cm² estimate but is deliberately marked ``calibrated=False``
    so the UI can label it as relative rather than a verified real-world value.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2-D (or singleton-channel), got {mask_array.shape}")
    binary = mask_array > 0
    area_pixels = int(binary.sum())
    perimeter_pixels = _perimeter_pixels(binary.astype(np.uint8))

    if calibration is None:
        return AreaMeasurement(
            area_pixels=area_pixels,
            area_cm2=None,
            perimeter_pixels=perimeter_pixels,
            perimeter_cm=None,
            pixels_per_cm=None,
            calibrated=False,
            calibration_method="uncalibrated / relative",
        )

    pixels_per_cm = float(calibration.pixels_per_cm)
    if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        raise ValueError("calibration pixels_per_cm must be a finite positive number")
    return AreaMeasurement(
        area_pixels=area_pixels,
        area_cm2=area_pixels / (pixels_per_cm**2),
        perimeter_pixels=perimeter_pixels,
        perimeter_cm=perimeter_pixels / pixels_per_cm,
        pixels_per_cm=pixels_per_cm,
        calibrated=calibration.calibrated,
        calibration_method=calibration.source,
    )
