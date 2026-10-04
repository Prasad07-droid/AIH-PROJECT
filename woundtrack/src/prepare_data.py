"""Download and prepare the labeled FUSeg data for WoundTrack Step 1.

Run from the repository root with::

    python -m woundtrack.src.prepare_data

The downloader fetches only the labeled official FUSeg train/validation
folders. The official challenge test folder has no public masks and is not
used for supervised training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from woundtrack.config import (
    FUSEG_ANNOTATED_SPLITS,
    FUSEG_CHALLENGE_DIR,
    FUSEG_RAW_DIR,
    FUSEG_REPOSITORY,
    FUSEG_SOURCE_REVISION,
    FUSEG_SOURCE_TREE_SHA,
    IMAGE_SIZE,
    MANIFEST_PATH,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RANDOM_SEED,
    SAMPLE_GRID_PATH,
    SPLITS_DIR,
    SPLIT_SUMMARY_PATH,
)

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_DOWNLOAD_ATTEMPTS = 4


def _fetch_url(url: str, *, accept: str = "application/octet-stream") -> bytes:
    """Fetch a URL with a small retry budget for transient network failures."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WoundTrack-MVP-data-preparation",
            "Accept": accept,
        },
    )
    last_error: Exception | None = None
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Unable to fetch {url}: {last_error}") from last_error


def _source_files() -> tuple[list[dict[str, Any]], list[tuple[str, str, dict[str, Any], dict[str, Any]]]]:
    """Read the pinned GitHub tree and return downloadable labeled pairs."""
    tree_url = (
        f"https://api.github.com/repos/{FUSEG_REPOSITORY}/git/trees/"
        f"{FUSEG_SOURCE_TREE_SHA}?recursive=1"
    )
    payload = json.loads(_fetch_url(tree_url, accept="application/vnd.github+json"))
    entries = payload.get("tree", [])
    if payload.get("truncated"):
        raise RuntimeError("The GitHub data tree response was truncated; cannot safely prepare FUSeg.")

    entries_by_split: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        split: {"images": {}, "labels": {}} for split in FUSEG_ANNOTATED_SPLITS
    }
    for entry in entries:
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        for split in FUSEG_ANNOTATED_SPLITS:
            prefix = f"{FUSEG_CHALLENGE_DIR}/{split}/"
            if not path.startswith(prefix):
                continue
            relative = Path(path[len(prefix) :])
            if len(relative.parts) != 2:
                continue
            kind, filename = relative.parts
            if kind not in ("images", "labels"):
                continue
            if Path(filename).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            entries_by_split[split][kind][Path(filename).stem.casefold()] = entry

    files_to_download: list[dict[str, Any]] = []
    pairs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for split in FUSEG_ANNOTATED_SPLITS:
        images = entries_by_split[split]["images"]
        labels = entries_by_split[split]["labels"]
        common_names = sorted(images.keys() & labels.keys())
        if not common_names:
            raise RuntimeError(f"No image/label pairs found in the public FUSeg {split} folder.")
        if images.keys() != labels.keys():
            missing_labels = sorted(images.keys() - labels.keys())
            missing_images = sorted(labels.keys() - images.keys())
            if missing_labels or missing_images:
                print(
                    f"Warning: {split} has {len(missing_labels)} images without labels and "
                    f"{len(missing_images)} labels without images; unmatched files are skipped."
                )
        for stem in common_names:
            image_entry = images[stem]
            label_entry = labels[stem]
            pairs.append((split, stem, image_entry, label_entry))
            files_to_download.extend((image_entry, label_entry))

    return files_to_download, pairs


def _download_source_archive() -> Path:
    """Download the pinned GitHub archive through the API/codeload endpoint."""
    archive_path = FUSEG_RAW_DIR.parent / f"FUSeg-{FUSEG_SOURCE_REVISION[:8]}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file() and zipfile.is_zipfile(archive_path):
        return archive_path

    temporary_path = archive_path.with_suffix(".zip.part")
    archive_url = (
        f"https://api.github.com/repos/{FUSEG_REPOSITORY}/zipball/"
        f"{FUSEG_SOURCE_REVISION}"
    )
    request = urllib.request.Request(
        archive_url,
        headers={
            "User-Agent": "WoundTrack-MVP-data-preparation",
            "Accept": "application/zip",
        },
    )
    last_error: Exception | None = None
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                with temporary_path.open("wb") as archive_file:
                    shutil.copyfileobj(response, archive_file, length=1024 * 1024)
            with zipfile.ZipFile(temporary_path) as archive:
                corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise OSError(f"Downloaded GitHub archive contains a corrupt file: {corrupt_member}")
            temporary_path.replace(archive_path)
            size_mb = archive_path.stat().st_size / (1024 * 1024)
            print(f"Downloaded and verified source archive ({size_mb:.1f} MiB).")
            return archive_path
        except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            if attempt + 1 < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Unable to download the FUSeg source archive: {last_error}") from last_error


def _extract_dataset_files(archive_path: Path, entries: list[dict[str, Any]]) -> int:
    """Extract only requested FUSeg images and labels from the source archive."""
    challenge_prefix = f"{FUSEG_CHALLENGE_DIR}/"
    expected = {str(entry["path"]): entry for entry in entries}
    extracted_paths: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or "/" not in member.filename:
                continue
            source_path = member.filename.split("/", maxsplit=1)[1]
            entry = expected.get(source_path)
            if entry is None:
                continue
            if not source_path.startswith(challenge_prefix):
                raise ValueError(f"Unexpected path in the GitHub archive: {source_path}")

            local_relative = source_path[len(challenge_prefix) :]
            destination = FUSEG_RAW_DIR / local_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_file, destination.open("wb") as output_file:
                shutil.copyfileobj(source_file, output_file, length=1024 * 1024)
            expected_size = int(entry["size"])
            if destination.stat().st_size != expected_size:
                destination.unlink(missing_ok=True)
                raise OSError(
                    f"Extracted size mismatch for {source_path}: expected {expected_size} bytes"
                )
            extracted_paths.add(source_path)

    missing = set(expected) - extracted_paths
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"The source archive is missing {len(missing)} expected files: {examples}")
    return len(extracted_paths)


def download_fuseg() -> tuple[int, int]:
    """Download labeled FUSeg train/validation files, skipping valid cached files."""
    print(f"Reading FUSeg file index from {FUSEG_REPOSITORY}@{FUSEG_SOURCE_REVISION[:8]}...")
    files, pairs = _source_files()
    print(
        f"Found {len(pairs)} labeled image/mask pairs "
        f"({len(files)} files; official unlabeled challenge test images are excluded)."
    )

    missing_entries = []
    challenge_prefix = f"{FUSEG_CHALLENGE_DIR}/"
    for entry in files:
        source_path = str(entry["path"])
        local_path = FUSEG_RAW_DIR / source_path[len(challenge_prefix) :]
        if not local_path.is_file() or local_path.stat().st_size != int(entry["size"]):
            missing_entries.append(entry)

    downloaded = 0
    if missing_entries:
        print(
            f"Preparing {len(missing_entries)} missing files from the pinned source archive "
            "(raw.githubusercontent.com is not required)."
        )
        archive_path = _download_source_archive()
        downloaded = _extract_dataset_files(archive_path, missing_entries)
        archive_path.unlink(missing_ok=True)
    else:
        print("All expected raw image and label files are already present; download skipped.")

    print(f"Raw data ready at {FUSEG_RAW_DIR} ({downloaded} files downloaded this run).")
    return len(pairs), downloaded


def _collect_local_pairs() -> list[tuple[str, Path, Path]]:
    """Pair locally available FUSeg images and labels by filename stem."""
    pairs: list[tuple[str, Path, Path]] = []
    for source_split in FUSEG_ANNOTATED_SPLITS:
        split_dir = FUSEG_RAW_DIR / source_split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        images = {
            path.stem.casefold(): path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        }
        labels = {
            path.stem.casefold(): path
            for path in labels_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        }
        for stem in sorted(images.keys() & labels.keys()):
            pairs.append((source_split, images[stem], labels[stem]))
        missing_labels = images.keys() - labels.keys()
        missing_images = labels.keys() - images.keys()
        if missing_labels or missing_images:
            print(
                f"Warning: local {source_split} folder has {len(missing_labels)} unmatched "
                f"images and {len(missing_images)} unmatched labels; these are skipped."
            )

    if not pairs:
        raise FileNotFoundError(
            f"No labeled FUSeg image/mask pairs were found under {FUSEG_RAW_DIR}. "
            "Run the downloader, or place files in train/images + train/labels and "
            "validation/images + validation/labels."
        )
    return pairs


def _resize_and_save_pair(
    source_split: str,
    image_path: Path,
    mask_path: Path,
) -> dict[str, Any]:
    """Resize one original pair, save it, and return manifest metadata."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Unable to decode source image: {image_path}")
    if mask is None:
        raise OSError(f"Unable to decode source label: {mask_path}")
    original_height, original_width = image.shape[:2]
    if mask.shape[:2] != (original_height, original_width):
        raise ValueError(
            f"Image/label dimension mismatch for {image_path.name}: "
            f"{(original_height, original_width)} vs {mask.shape[:2]}"
        )

    resized_image = cv2.resize(
        image,
        dsize=(IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    resized_mask = cv2.resize(
        binary_mask,
        dsize=(IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    sample_id = f"fuseg_{source_split}_{image_path.stem}"
    image_out = PROCESSED_IMAGES_DIR / f"{sample_id}.png"
    mask_out = PROCESSED_MASKS_DIR / f"{sample_id}.png"
    if not cv2.imwrite(str(image_out), resized_image):
        raise OSError(f"Unable to save processed image: {image_out}")
    if not cv2.imwrite(str(mask_out), resized_mask):
        raise OSError(f"Unable to save processed mask: {mask_out}")

    return {
        "sample_id": sample_id,
        "image_path": image_out.relative_to(FUSEG_RAW_DIR.parent.parent).as_posix(),
        "mask_path": mask_out.relative_to(FUSEG_RAW_DIR.parent.parent).as_posix(),
        "original_height": original_height,
        "original_width": original_width,
        "source_split": source_split,
    }


def _write_splits(sample_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Create deterministic 70/15/15 splits by image, not by source folder."""
    shuffled_ids = list(sample_ids)
    random.Random(seed).shuffle(shuffled_ids)
    train_end = int(0.70 * len(shuffled_ids))
    validation_end = train_end + int(0.15 * len(shuffled_ids))
    splits = {
        "train": shuffled_ids[:train_end],
        "val": shuffled_ids[train_end:validation_end],
        "test": shuffled_ids[validation_end:],
    }
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (SPLITS_DIR / f"{split}.txt").write_text(
            "\n".join(ids) + "\n",
            encoding="utf-8",
        )
    return splits


def _write_manifest(rows: list[dict[str, Any]]) -> None:
    """Save source and original-dimension metadata for each prepared sample."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "image_path",
        "mask_path",
        "original_height",
        "original_width",
        "source_split",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_sample_grid(rows: list[dict[str, Any]], seed: int) -> None:
    """Save eight processed image/mask examples in a two-column grid."""
    selected_count = min(8, len(rows))
    selected = random.Random(seed + 1).sample(rows, selected_count)
    fig, axes = plt.subplots(selected_count, 2, figsize=(8, 3 * selected_count), squeeze=False)

    for row_index, row in enumerate(selected):
        sample_id = row["sample_id"]
        image_bgr = cv2.imread(str(PROCESSED_IMAGES_DIR / f"{sample_id}.png"))
        mask = cv2.imread(
            str(PROCESSED_MASKS_DIR / f"{sample_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if image_bgr is None or mask is None:
            raise OSError(f"Unable to read processed sample {sample_id} for preview grid")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axes[row_index, 0].imshow(image_rgb)
        axes[row_index, 0].set_title(f"Image: {sample_id}", fontsize=9)
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title("Ground-truth wound mask", fontsize=9)
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")

    fig.suptitle("FUSeg samples after 256x256 preprocessing", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    SAMPLE_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLE_GRID_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_dataset(seed: int = RANDOM_SEED) -> dict[str, list[str]]:
    """Resize all local pairs, split them, and write a sample grid and manifest."""
    pairs = _collect_local_pairs()
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        _resize_and_save_pair(source_split, image_path, mask_path)
        for source_split, image_path, mask_path in pairs
    ]
    _write_manifest(rows)
    splits = _write_splits([row["sample_id"] for row in rows], seed)
    _save_sample_grid(rows, seed)

    summary = {
        "dataset": "FUSeg (labeled train + validation images)",
        "source_repository": f"https://github.com/{FUSEG_REPOSITORY}",
        "source_revision": FUSEG_SOURCE_REVISION,
        "seed": seed,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "normalization": "ImageNet mean/std, applied in WoundSegmentationDataset",
        "total_labeled_pairs": len(rows),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_proportions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "sample_grid": str(SAMPLE_GRID_PATH),
    }
    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nWoundTrack Step 1 — data preparation check")
    print(f"  Labeled image/mask pairs: {len(rows)}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Validation: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    print(f"  Prepared resolution: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("  Input normalization: ImageNet mean/std in Dataset loader")
    print(f"  Eight-pair sample grid: {SAMPLE_GRID_PATH}")
    print(f"  Split lists and manifest: {SPLITS_DIR} and {MANIFEST_PATH}")
    return splits


def _run_dataset_smoke_check(splits: dict[str, list[str]]) -> None:
    """Load one actual training sample and verify tensor/mask/restore shapes."""
    from woundtrack.src.dataset import WoundSegmentationDataset, resize_mask_to_original

    import torch

    dataset = WoundSegmentationDataset("train", augment=False)
    image_tensor, mask_tensor = dataset[0]
    expected_tensor_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    expected_mask_shape = (1, IMAGE_SIZE, IMAGE_SIZE)
    if tuple(image_tensor.shape) != expected_tensor_shape:
        raise AssertionError(f"unexpected image tensor shape: {tuple(image_tensor.shape)}")
    if tuple(mask_tensor.shape) != expected_mask_shape:
        raise AssertionError(f"unexpected mask tensor shape: {tuple(mask_tensor.shape)}")
    if not torch.isfinite(image_tensor).all():
        raise AssertionError("normalized image tensor contains NaN or infinite values")
    if not set(torch.unique(mask_tensor).tolist()).issubset({0.0, 1.0}):
        raise AssertionError("dataset mask is not binary")

    sample_id = dataset.sample_ids[0]
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as csv_file:
        metadata = next(row for row in csv.DictReader(csv_file) if row["sample_id"] == sample_id)
    original_shape = (int(metadata["original_height"]), int(metadata["original_width"]))
    restored_mask = resize_mask_to_original(mask_tensor.squeeze(0).numpy(), original_shape)
    if restored_mask.shape != original_shape:
        raise AssertionError(
            f"restored mask shape {restored_mask.shape} does not match {original_shape}"
        )
    if not set(np.unique(restored_mask).tolist()).issubset({0, 1}):
        raise AssertionError("restored mask is not binary")

    augmented_dataset = WoundSegmentationDataset("train")
    augmented_image, augmented_mask = augmented_dataset[0]
    if tuple(augmented_image.shape) != expected_tensor_shape:
        raise AssertionError("augmented image has an unexpected tensor shape")
    if tuple(augmented_mask.shape) != expected_mask_shape:
        raise AssertionError("augmented mask has an unexpected tensor shape")
    if not torch.isfinite(augmented_image).all():
        raise AssertionError("augmented image tensor contains NaN or infinite values")
    if not set(torch.unique(augmented_mask).tolist()).issubset({0.0, 1.0}):
        raise AssertionError("augmented mask is not binary")

    if len(dataset) != len(splits["train"]):
        raise AssertionError("PyTorch Dataset count does not match the train split")
    print(
        "  Dataset smoke check: PASS — "
        f"image {tuple(image_tensor.shape)}, mask {tuple(mask_tensor.shape)}, "
        f"restored mask {restored_mask.shape}; train augmentation tensor checks passed"
    )


def main() -> None:
    """Run download, preprocessing, split generation, and the Step 1 check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use FUSeg files already placed in the configured raw-data folder",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="fixed split seed")
    args = parser.parse_args()

    if not args.skip_download:
        download_fuseg()
    splits = prepare_dataset(seed=args.seed)
    _run_dataset_smoke_check(splits)


if __name__ == "__main__":
    main()
