"""Central paths and data configuration for WoundTrack.

The data root can be moved outside the repository by setting
``WOUNDTRACK_DATA_DIR``. All other paths are derived from this module.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("WOUNDTRACK_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser().resolve()
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR = PROCESSED_DIR / "masks"
SPLITS_DIR = DATA_ROOT / "splits"
SAMPLE_GRID_PATH = PROCESSED_DIR / "sample_grid.png"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
SPLIT_SUMMARY_PATH = SPLITS_DIR / "summary.json"
UPLOADS_DIR = DATA_ROOT / "uploads"
REPORTS_DIR = DATA_ROOT / "reports"
DATABASE_PATH = DATA_ROOT / "woundtrack.sqlite3"

MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best.pt"
EVALUATION_DIR = MODEL_DIR / "evaluation"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"

RESNET34_WEIGHT_NAME = "resnet34-333f7ec4.pth"
RESNET34_WEIGHT_SHA256_PREFIX = "333f7ec4"
RESNET34_WEIGHT_SIZE = 87_306_240
RESNET34_WEIGHT_MIRROR_URL = (
    "https://api.github.com/repos/fregu856/deeplabv3/contents/"
    "pretrained_models/resnet/resnet34-333f7ec4.pth"
    "?ref=415d983ec8a3e4ab6977b316d8f553371a415739"
)

TRAINING_SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
MEDICAL_DISCLAIMER = (
    "This tool is for monitoring support only and is not a medical diagnosis."
)

FUSEG_RAW_DIR = RAW_DIR / "FUSeg"
FUSEG_REPOSITORY = "uwm-bigdata/wound-segmentation"
# Pin the public source revision so repeated downloads use the same data tree.
FUSEG_SOURCE_REVISION = "42a272dfe0679f20675e826385925cb7562934b6"
FUSEG_SOURCE_TREE_SHA = "6cc3ba2391f647ccda78a58593fd53e9abd9dcdf"
FUSEG_CHALLENGE_DIR = "data/Foot Ulcer Segmentation Challenge"
FUSEG_ANNOTATED_SPLITS = ("train", "validation")

IMAGE_SIZE = 256
RANDOM_SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
