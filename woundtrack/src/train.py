"""Train a U-Net with a ResNet-34 ImageNet encoder on prepared FUSeg data."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from woundtrack.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    RANDOM_SEED,
    TRAINING_HISTORY_PATH,
)
from woundtrack.src.dataset import WoundSegmentationDataset
from woundtrack.src.model import build_unet


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bce_dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Return BCE-with-logits plus a smoothed soft Dice loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    reduce_dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + (1.0 - dice.mean())


def _mean_hard_dice(logits: Tensor, targets: Tensor) -> float:
    """Compute mean per-image Dice after thresholding model logits at zero."""
    predictions = logits >= 0.0
    targets_bool = targets >= 0.5
    reduce_dims = (1, 2, 3)
    intersection = (predictions & targets_bool).sum(dim=reduce_dims).float()
    denominator = predictions.sum(dim=reduce_dims) + targets_bool.sum(dim=reduce_dims)
    dice = (2.0 * intersection + 1.0) / (denominator.float() + 1.0)
    return float(dice.mean().item())


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Run one optimizer epoch and return its average loss."""
    model.train()
    running_loss = 0.0
    batch_count = 0
    for batch_count, (images, masks) in enumerate(loader, start=1):
        if max_batches is not None and batch_count > max_batches:
            batch_count -= 1
            break
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
    if batch_count == 0:
        raise ValueError("Training loader produced no batches")
    return running_loss / batch_count


def _run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Return mean validation loss and per-image hard Dice."""
    model.eval()
    loss_total = 0.0
    dice_total = 0.0
    seen_samples = 0
    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            batch_size = int(images.shape[0])
            loss_total += float(bce_dice_loss(logits, masks).item()) * batch_size
            dice_total += _mean_hard_dice(logits, masks) * batch_size
            seen_samples += batch_size
    if seen_samples == 0:
        raise ValueError("Validation loader produced no samples")
    return loss_total / seen_samples, dice_total / seen_samples


def train_model(
    *,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    resume_from: Path | None = None,
) -> dict[str, object]:
    """Train, early-stop on validation Dice, and save the best checkpoint."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    set_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WoundSegmentationDataset("train")
    validation_dataset = WoundSegmentationDataset("val")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    resume_checkpoint: dict[str, object] | None = None
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    model = build_unet(pretrained=resume_checkpoint is None).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    if TRAINING_HISTORY_PATH.is_file():
        history = json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    best_dice = float(resume_checkpoint.get("best_val_dice", -1.0)) if resume_checkpoint else -1.0
    best_epoch = int(resume_checkpoint.get("best_epoch", 0)) if resume_checkpoint else 0
    epochs_without_improvement = 0
    start_epoch = best_epoch + 1 if resume_checkpoint else 1

    print(
        f"Training U-Net/ResNet34 on {device}; "
        f"{len(train_dataset)} train and {len(validation_dataset)} validation samples"
        + (f"; resuming from epoch {best_epoch}" if resume_checkpoint else "")
    )
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.perf_counter()
        train_loss = _run_train_epoch(
            model, train_loader, optimizer, device, max_batches=max_train_batches
        )
        val_loss, val_dice = _run_validation(
            model, validation_loader, device, max_batches=max_val_batches
        )
        elapsed = time.perf_counter() - epoch_start
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "seconds": elapsed,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{start_epoch + epochs - 1:02d} — train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val Dice {val_dice:.4f}, {elapsed:.1f}s"
        )

        if val_dice > best_dice + 1e-4:
            best_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_dice,
                    "best_epoch": best_epoch,
                    "model_config": {
                        "architecture": "Unet",
                        "encoder_name": "resnet34",
                        "encoder_weights": "imagenet",
                        "in_channels": 3,
                        "classes": 1,
                    },
                    "image_size": IMAGE_SIZE,
                    "normalization": "imagenet",
                    "seed": seed,
                },
                BEST_MODEL_PATH,
            )
            print(f"  Saved best checkpoint: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"  No validation Dice improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print("  Early stopping on validation Dice.")
                break

        TRAINING_HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    TRAINING_HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    if not BEST_MODEL_PATH.is_file():
        raise RuntimeError("Training ended without producing models/best.pt")
    result: dict[str, object] = {
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "epochs_run": len(history),
        "checkpoint": str(BEST_MODEL_PATH),
        "device": str(device),
    }
    print(
        f"Training complete — best validation Dice {best_dice:.4f} at epoch {best_epoch}; "
        f"checkpoint: {BEST_MODEL_PATH}"
    )
    return result


def main() -> None:
    """Parse CLI flags and train the configured model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None, help="continue from a saved checkpoint")
    parser.add_argument("--max-train-batches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-batches", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
