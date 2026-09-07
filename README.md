# AI bird-detection model

Initial image-only object-detection project for the autonomous bird-deterrence drone. This repository intentionally excludes tracking, flight control, audio control, video processing, and Raspberry Pi deployment.

## Status

- Project-local Python 3.11 environment created at `.venv`.
- No packages, datasets, models, or training outputs have been downloaded or created.
- Recommended initial dataset/class: **COCO 2017, one class: `bird`**. See `dataset_info.md`.

## Layout

`datasets/public/` holds immutable upstream downloads. `data/` holds derived YOLO-ready data. Training artifacts belong in `runs/`, curated weights in `models/trained/`, and reports/plots in `results/`.

## Next approved stage

1. Download the selected COCO 2017 source files to `datasets/public/coco2017/`.
2. Filter only images with COCO category `bird` into a derived YOLO dataset without changing the source files.
3. Run image/annotation validation, exact-duplicate detection, and grouped train/validation/test splitting.
4. Create bounding-box visual QA samples and review them before installing/training Ultralytics YOLO Nano.

## Environment activation

```powershell
& 'D:\HIWI_SENSIG\AI_MODEL\.venv\Scripts\Activate.ps1'
```

Do not run package installation, downloads, or training until the dataset choice is approved.
