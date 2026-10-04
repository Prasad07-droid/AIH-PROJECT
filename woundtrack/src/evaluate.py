"""Evaluate the best wound model and save ten test-set visual overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from woundtrack.config import (
    BEST_MODEL_PATH,
    EVALUATION_DIR,
    METRICS_PATH,
    PROCESSED_IMAGES_DIR,
)
from woundtrack.src.dataset import WoundSegmentationDataset
from woundtrack.src.model import build_unet


def _sample_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute binary Dice and IoU for one pair of 2-D masks."""
    predicted = prediction.astype(bool)
    expected = target.astype(bool)
    intersection = int(np.logical_and(predicted, expected).sum())
    predicted_area = int(predicted.sum())
    target_area = int(expected.sum())
    union = int(np.logical_or(predicted, expected).sum())
    dice = (2.0 * intersection) / (predicted_area + target_area) if predicted_area + target_area else 1.0
    iou = intersection / union if union else 1.0
    return dice, iou


def _overlay_panel(
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray,
) -> np.ndarray:
    """Return an RGB overlay: target boundary green, prediction fill red."""
    overlay = image_rgb.copy()
    pred = prediction_mask.astype(bool)
    overlay[pred] = (0.6 * overlay[pred] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
    contours, _ = cv2.findContours(
        (target_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, color=(30, 230, 60), thickness=2)
    return overlay


def _save_overlay(
    sample_id: str,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """Save an original/ground-truth/prediction triptych for one test sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_bgr = cv2.imread(str(PROCESSED_IMAGES_DIR / f"{sample_id}.png"), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Missing processed image for overlay: {sample_id}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred_overlay = _overlay_panel(image_rgb, target, prediction)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Expert binary mask")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction (red), expert edge (green)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(sample_id)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(
    checkpoint_path: Path = BEST_MODEL_PATH,
    *,
    batch_size: int = 4,
    save_overlays: int = 10,
) -> dict[str, float | int | str]:
    """Evaluate on the held-out split and save aggregate metrics and overlays."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}; run "
            "`python -m woundtrack.src.train` first."
        )
    if batch_size < 1 or save_overlays < 0:
        raise ValueError("batch_size must be positive and save_overlays cannot be negative")

    # Evaluation is model inference: keep it on CPU regardless of hardware availability.
    device = torch.device("cpu")
    model = build_unet(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = WoundSegmentationDataset("test", augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_image_dice: list[float] = []
    per_image_iou: list[float] = []
    total_intersection = 0
    total_predicted = 0
    total_target = 0
    predictions_for_overlay: list[tuple[str, np.ndarray, np.ndarray]] = []

    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits).cpu().numpy()[:, 0] >= 0.5)
            targets = (masks.numpy()[:, 0] >= 0.5)
            for batch_index, (prediction, target) in enumerate(zip(predictions, targets)):
                sample_index = len(per_image_dice)
                dice, iou = _sample_metrics(prediction, target)
                per_image_dice.append(dice)
                per_image_iou.append(iou)
                total_intersection += int(np.logical_and(prediction, target).sum())
                total_predicted += int(prediction.sum())
                total_target += int(target.sum())
                if len(predictions_for_overlay) < save_overlays:
                    sample_id = dataset.sample_ids[sample_index]
                    predictions_for_overlay.append((sample_id, target, prediction))

    if not per_image_dice:
        raise ValueError("The held-out test split is empty")
    global_dice = (
        2.0 * total_intersection / (total_predicted + total_target)
        if total_predicted + total_target
        else 1.0
    )
    global_iou = (
        total_intersection / (total_predicted + total_target - total_intersection)
        if total_predicted + total_target - total_intersection
        else 1.0
    )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for overlay_index, (sample_id, target, prediction) in enumerate(predictions_for_overlay, start=1):
        _save_overlay(
            sample_id,
            target,
            prediction,
            EVALUATION_DIR / f"prediction_{overlay_index:02d}_{sample_id}.png",
        )
    metrics: dict[str, float | int | str] = {
        "num_test_images": len(per_image_dice),
        "mean_dice": float(np.mean(per_image_dice)),
        "mean_iou": float(np.mean(per_image_iou)),
        "global_dice": float(global_dice),
        "global_iou": float(global_iou),
        "threshold": 0.5,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "overlays_saved": len(predictions_for_overlay),
        "overlay_directory": str(EVALUATION_DIR),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Test images: {len(per_image_dice)}")
    print(f"Mean test Dice: {metrics['mean_dice']:.4f}")
    print(f"Mean test IoU: {metrics['mean_iou']:.4f}")
    print(f"Global test Dice: {metrics['global_dice']:.4f}")
    print(f"Global test IoU: {metrics['global_iou']:.4f}")
    print(f"Saved {len(predictions_for_overlay)} prediction overlays to {EVALUATION_DIR}")
    return metrics


def main() -> None:
    """Parse optional checkpoint and evaluation-size overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overlays", type=int, default=10)
    args = parser.parse_args()
    evaluate_model(args.checkpoint, batch_size=args.batch_size, save_overlays=args.overlays)


if __name__ == "__main__":
    main()
