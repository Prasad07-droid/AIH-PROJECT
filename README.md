# WoundTrack

WoundTrack is a working MVP for monitoring wound-healing progress from serial smartphone images. It segments the wound, measures pixel and scale-derived area, compares visits, estimates tissue color composition, stores records locally, and creates a PDF report.

> **This tool is for monitoring support only and is not a medical diagnosis.**

## Quick start

From the repository root with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m woundtrack.src.prepare_data
python -m woundtrack.src.train
python -m woundtrack.src.evaluate
python -m streamlit run woundtrack/app.py
```

The final command starts the dashboard at `http://localhost:8501`. In Arena, the dashboard is also available in the live preview. **The exact app command is `python -m streamlit run woundtrack/app.py`.**

The first data-preparation run downloads about 337 MiB of FUSeg source archive. Model training on CPU is possible but slow; the checked run below took about 32 minutes for three full epochs. Training on a GPU/Colab is recommended. **App prediction and held-out evaluation run on CPU regardless of available accelerator hardware.**

If data is already in place, rerun preparation without internet access:

```bash
python -m woundtrack.src.prepare_data --skip-download
```

Move all raw, processed, upload, report, and database files outside the repository by setting `WOUNDTRACK_DATA_DIR`, for example:

```bash
WOUNDTRACK_DATA_DIR=/mnt/woundtrack-data python -m woundtrack.src.prepare_data
WOUNDTRACK_DATA_DIR=/mnt/woundtrack-data python -m streamlit run woundtrack/app.py
```

## Step 1 — data

The pipeline uses the labeled **FUSeg (Foot Ulcer Segmentation Challenge)** subset published in the public [UWM wound-segmentation repository](https://github.com/uwm-bigdata/wound-segmentation) and described on the [FUSeg challenge page](https://fusc.grand-challenge.org/FUSeg-2021/). It fetches a [pinned source revision](https://github.com/uwm-bigdata/wound-segmentation/tree/42a272dfe0679f20675e826385925cb7562934b6/data/Foot%20Ulcer%20Segmentation%20Challenge) and organizes its labeled `train` and `validation` folders. The official challenge `test` folder has images but no public masks, so it is excluded from supervised training.

The script prepares **1,010 labeled pairs** (810 source-train + 200 source-validation), resizes to 256×256, applies ImageNet mean/std normalization in the dataset loader, and splits by image with seed 42: **707 train / 151 validation / 152 test**. `manifest.csv` preserves each source image's original height and width; `resize_mask_to_original()` maps a predicted binary mask back with nearest-neighbor interpolation.

Augmentations used on training samples: horizontal and vertical flips, small rotation, affine zoom/translation, brightness/contrast, color jitter, and Gaussian blur. Validation/test use normalization only.

Generated outputs:

- `woundtrack/data/raw/FUSeg/{train,validation}/{images,labels}/`
- `woundtrack/data/processed/images/` and `masks/`
- `woundtrack/data/processed/manifest.csv`
- `woundtrack/data/processed/sample_grid.png` — eight image/mask pairs
- `woundtrack/data/splits/{train,val,test}.txt` and `summary.json`

The visual grid is created by `prepare_data.py` and is also available in the local workspace at `woundtrack/data/processed/sample_grid.png` after preparation. Model prediction overlays are generated under `woundtrack/models/evaluation/` after evaluation. These data-derived screenshots are local generated outputs (and are not committed to Git).

### Data sample screenshot

![Eight FUSeg image/mask examples](woundtrack/data/processed/sample_grid.png)

### Data decisions

- FUSeg was selected because it contains expert pixel masks. The release is associated with the AZH Wound and Vascular Center; a duplicate AZH copy is not needed.
- Medetec stock wound photos are not mixed into supervised training because they do not come with paired segmentation masks. Optional manually obtained images belong in `woundtrack/data/raw/Medetec/images/`; matching masks would need to go in `woundtrack/data/raw/Medetec/labels/` and an adapter added before training. No manual download is needed for FUSeg.
- This follows the requested image-level split, not a patient-level split. The release does not expose patient IDs to this pipeline, so repeated-patient leakage across splits cannot be ruled out.
- Downloaded data, personal records, model checkpoints, and generated reports are ignored by Git. Check the data source's current terms before redistribution or clinical use.

## Step 2 — segmentation model

`woundtrack/src/train.py` trains a PyTorch `segmentation_models_pytorch` U-Net with a ResNet-34 ImageNet encoder using BCE + Dice loss, Adam at `1e-4`, and early stopping on validation Dice. The best checkpoint is saved at `woundtrack/models/best.pt`. If direct PyTorch weight download is blocked, the model helper uses a pinned GitHub API mirror and verifies the official file size and SHA-256 prefix before loading it.

`woundtrack/src/evaluate.py` reports mean per-image and global Dice/IoU on the held-out test set and saves ten expert-mask/prediction overlays under `woundtrack/models/evaluation/`.

### Checked metrics (2026-10-04)

The initial two-epoch fit had mean test Dice **0.5946** / mean IoU **0.4677**. Per the requested `<0.70` improvement rule, I added affine zoom/translation augmentation and resumed for one epoch. The final checkpoint metrics are:

| Metric | Test result |
|---|---:|
| Test images | 152 |
| Mean per-image Dice | **0.7253** |
| Mean per-image IoU | **0.6228** |
| Global Dice | **0.8342** |
| Global IoU | **0.7156** |
| Best validation Dice | **0.7861** (epoch 3) |
| Prediction overlays | 10 |

The final mean test Dice is above the 0.70 target. The model checkpoint and evaluation artifacts are local outputs and are not committed.

![Held-out test prediction overlay](woundtrack/models/evaluation/prediction_01_fuseg_train_0335.png)

## Step 3 — area measurement and scale

`woundtrack/src/measure.py` counts mask pixels at original image resolution. It can estimate pixels/cm using Hough-circle detection for a nominal **25 mm one-rupee coin** or contour detection for a **2 cm square marker**. The user may enter pixels/cm manually; manual estimates are always marked **“Uncalibrated / relative”** even though a numerical cm² estimate is shown. Without any scale, the app reports pixels only.

Synthetic coin check: a detected 1-rupee marker at 40.0 px/cm converted a 1,257-pixel wound mask to **0.79 cm²**, a sensible result for the test geometry. Coin/square detection is approximate; keep the reference object flat, fully visible, and near the wound plane.

## Step 4 — visit comparison

`compare.py` stores signed percentage change from baseline and previous visit (negative means area reduction), reduction from baseline for status, and healing rate `(previous area − current area) / days` in cm²/day. Status thresholds are:

- ≥50% reduction: **Healing well**
- 10–50%: **Slow progress**
- −10–10%: **Stagnant**
- <−10%: **Worsening, consult a doctor**

ORB + RANSAC homography alignment is attempted only for the visual baseline/latest photo comparison. On failure it silently displays the unaligned photos; alignment never changes area measurement.

## Step 5 — tissue analysis

`woundtrack/src/tissue.py` uses simple HSV thresholds inside the wound mask for granulation (red/pink), slough (yellow), and necrotic (dark) pixels. The dashboard shows a stacked bar over time and keeps the remaining pixels as “other / unclassified.” **These thresholds are heuristic, sensitive to lighting/camera/skin tone, and are not a validated clinical tissue classifier.**

## Step 6 — local database

SQLite is stored at `woundtrack/data/woundtrack.sqlite3` by default. `db.py` provides patient and visit CRUD. Visit records include date, image/mask paths, pixel and optional cm² area, perimeter/unit, calibration flag, area-change metrics, status, and tissue percentages. Deleting a patient cascades its visit records but deliberately does not erase stored image files.

## Step 7 — Streamlit dashboard and PDF

Pages:

1. **Patients** — add/select a patient.
2. **New Visit** — upload or take a photo, set visit date, run cached CPU segmentation, review original/mask/overlay, auto-detect a marker or enter a manual scale, inspect area/perimeter and heuristic tissue percentages, then save.
3. **Progress** — Plotly area trend, latest healing rate/status, stacked tissue chart, baseline/latest comparison, and an overlay timeline.
4. **Report** — generate/download a ReportLab PDF with patient summary, area/tissue graphs, visit details, photo comparison, and the required disclaimer.

Photos, masks, database, and reports are stored locally. Use a protected machine and do not use identifiable patient data without appropriate authorization and safeguards.

## Implementation map

```text
woundtrack/
  app.py
  config.py
  src/
    dataset.py        # Dataset, augmentation/normalization, original-size masks
    prepare_data.py  # FUSeg download, preparation, splits, sample grid
    model.py         # U-Net factory and checked ImageNet weight fallback
    train.py         # BCE + Dice / Adam / early stopping
    evaluate.py      # Test Dice/IoU and ten overlays
    segment.py       # CPU predict_mask(image_rgb)
    measure.py       # Pixel area, marker/manual scale, perimeter
    compare.py       # Change metrics and optional ORB alignment
    tissue.py        # HSV heuristic color proportions
    db.py            # SQLite patient/visit CRUD
    report.py        # PDF summary, plots, photos, disclaimer
```

## Step 8 — tests, limitations, and current status

Run the full unit suite with:

```bash
python -m pytest -q
```

The six tests cover measurement/comparison edge cases and CPU-only prediction when a caller passes a model and requests an accelerator.

Limitations: this is a small single-dataset MVP; test performance is not clinical validation; image-level splits may leak patient-specific characteristics; real-world measurements depend on correct marker placement and detection; tissue color estimates are heuristic; photos with different viewpoint/lighting can distort comparisons; and model performance can shift on other wound types, cameras, skin tones, or image quality. This project is monitoring support only, not diagnosis or treatment advice.

All requested steps 1–8 are implemented. Data/model outputs remain local; source code, configuration, pinned direct dependencies, and unit tests are in the repository.
