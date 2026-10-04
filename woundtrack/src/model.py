"""Shared U-Net model factory and verified ImageNet-weight fallback."""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from woundtrack.config import (
    RESNET34_WEIGHT_MIRROR_URL,
    RESNET34_WEIGHT_NAME,
    RESNET34_WEIGHT_SHA256_PREFIX,
    RESNET34_WEIGHT_SIZE,
)


def _weight_cache_path() -> Path:
    """Return the standard Torch Hub cache path for the ResNet-34 checkpoint."""
    return Path(torch.hub.get_dir()) / "checkpoints" / RESNET34_WEIGHT_NAME


def _sha256(path: Path) -> str:
    """Hash a file incrementally so model weights need not fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_cached_weight(path: Path) -> bool:
    """Check the known file size and official PyTorch SHA-256 prefix."""
    return (
        path.is_file()
        and path.stat().st_size == RESNET34_WEIGHT_SIZE
        and _sha256(path).startswith(RESNET34_WEIGHT_SHA256_PREFIX)
    )


def _download_verified_weight_mirror() -> Path:
    """Fetch official ResNet-34 weights through a checksum-verified GitHub mirror.

    Direct ``download.pytorch.org`` downloads are attempted by SMP first. The
    pinned GitHub API mirror is only used if that request fails, and the file
    must match both the known size and SHA-256 prefix before it enters Torch's
    cache.
    """
    cache_path = _weight_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_valid_cached_weight(cache_path):
        return cache_path
    cache_path.unlink(missing_ok=True)

    temporary_path = cache_path.with_suffix(cache_path.suffix + ".part")
    request = urllib.request.Request(
        RESNET34_WEIGHT_MIRROR_URL,
        headers={
            "User-Agent": "WoundTrack-MVP-model-setup",
            "Accept": "application/vnd.github.raw+json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                with temporary_path.open("wb") as output_file:
                    while chunk := response.read(1024 * 1024):
                        output_file.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
            if byte_count != RESNET34_WEIGHT_SIZE:
                raise OSError(
                    f"Expected {RESNET34_WEIGHT_SIZE} bytes, received {byte_count}"
                )
            if not digest.hexdigest().startswith(RESNET34_WEIGHT_SHA256_PREFIX):
                raise OSError("ResNet-34 checkpoint SHA-256 did not match the expected prefix")
            temporary_path.replace(cache_path)
            return cache_path
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Could not retrieve verified ResNet-34 ImageNet weights: {last_error}") from last_error


def build_unet(pretrained: bool = True) -> smp.Unet:
    """Create the fixed U-Net/ResNet34 model, optionally with ImageNet weights."""
    encoder_weights = "imagenet" if pretrained else None
    try:
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
    except Exception as initial_error:
        if not pretrained:
            raise
        try:
            _download_verified_weight_mirror()
            return smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "Unable to initialize the ResNet-34 ImageNet encoder. Check network access "
                "or populate the Torch Hub checkpoint cache with the verified weights. "
                f"Original error: {initial_error}; fallback error: {fallback_error}"
            ) from fallback_error
