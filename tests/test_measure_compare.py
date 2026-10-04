"""Fast unit tests for calibrated measurement and longitudinal comparison."""

import cv2
import numpy as np

from woundtrack.src.compare import align_later_to_earlier, area_status, compare_areas
from woundtrack.src.measure import (
    detect_reference_scale,
    manual_calibration,
    measure_mask,
)


def test_manual_area_conversion_is_explicitly_relative() -> None:
    """A known binary mask converts correctly but stays uncalibrated."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:15, 7:17] = 1
    measurement = measure_mask(mask, manual_calibration(10.0))

    assert measurement.area_pixels == 100
    assert measurement.area_cm2 == 1.0
    assert measurement.perimeter_pixels == 36.0
    assert measurement.perimeter_cm == 3.6
    assert measurement.calibrated is False
    assert measurement.display_label == "Uncalibrated / relative"


def test_coin_marker_gives_sensible_real_world_area() -> None:
    """A synthetic face-on 1-rupee coin produces a plausible wound area."""
    image = np.full((512, 512, 3), 128, dtype=np.uint8)
    cv2.circle(image, (400, 400), 50, (225, 225, 225), thickness=-1)
    cv2.circle(image, (400, 400), 47, (30, 30, 30), thickness=3)
    wound_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(wound_mask, (150, 150), 20, 1, thickness=-1)

    calibration = detect_reference_scale(image)
    assert calibration is not None
    assert calibration.source == "1-rupee coin"
    assert 30.0 <= calibration.pixels_per_cm <= 50.0
    measurement = measure_mask(wound_mask, calibration)
    assert measurement.area_cm2 is not None
    assert 0.5 <= measurement.area_cm2 <= 1.5
    assert measurement.calibrated is True


def test_compare_areas_calculates_signed_changes_and_healing_rate() -> None:
    """A smaller area yields negative changes and positive healing rate."""
    trend = compare_areas(
        current_area_cm2=4.0,
        baseline_area_cm2=10.0,
        previous_area_cm2=5.0,
        days_since_previous=7,
    )

    assert trend.change_from_baseline_pct == -60.0
    assert trend.change_from_previous_pct == -20.0
    assert trend.reduction_from_baseline_pct == 60.0
    assert round(trend.healing_rate_cm2_per_day or 0.0, 6) == round(1.0 / 7.0, 6)
    assert trend.status_label == "Healing well"


def test_area_status_thresholds() -> None:
    """Status labels follow the baseline reduction threshold boundaries."""
    assert area_status(50.0) == "Healing well"
    assert area_status(10.0) == "Slow progress"
    assert area_status(0.0) == "Stagnant"
    assert area_status(-10.0) == "Stagnant"
    assert area_status(-10.1) == "Worsening, consult a doctor"


def test_orb_alignment_silently_skips_blank_images() -> None:
    """Featureless photos do not raise or produce a false homography."""
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    assert align_later_to_earlier(blank, blank) is None
