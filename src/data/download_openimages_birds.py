"""Download Open Images V7 Bird + Raven boxes via FiftyOne and convert to YOLO.

Replaces the earlier raw-Flickr-URL downloader, which failed on ~90% of
requests (dead OriginalURL links). FiftyOne's zoo downloader pulls images
from a Google-hosted CDN mirror, which is far more reliable, and reuses
images already cached locally under datasets/public/fiftyone/.

Raven ('/m/06j2d') is pulled alongside the generic Bird ('/m/015p6') class
because the deployment target is Common Raven (Kolkrabe); both are still
written out as class 0 ("bird") per the single-class detector decision.
"""
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "open_images_bird"
)

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

CLASSES = ["Bird", "Raven"]

# Cap agreed in configs/dataset_plan.yaml: never the full 561 GB release.
MAX_IMAGES = 20000

SEED = 42


# ============================================================
# STEP 1: LOAD (DOWNLOADS ONLY WHAT'S MISSING)
# ============================================================

print("=" * 70)
print("OPEN IMAGES V7 BIRD + RAVEN DOWNLOAD (via FiftyOne)")
print("=" * 70)

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=MAX_IMAGES,
    seed=SEED,
    shuffle=True,
    dataset_name="open_images_bird_raven_train",
)

print(f"\nLoaded samples: {len(dataset):,}")


# ============================================================
# STEP 2: CONVERT TO YOLO LABELS
# ============================================================

successful = 0
skipped = 0
bird_boxes = 0
raven_boxes = 0

for sample in dataset.iter_samples(progress=True):

    image_path = Path(sample.filepath)

    if not image_path.exists():
        skipped += 1
        continue

    detections = sample.ground_truth
    if detections is None:
        skipped += 1
        continue

    lines = []

    for det in detections.detections:

        if det.label not in CLASSES:
            continue

        # FiftyOne bounding_box: [top-left-x, top-left-y, width, height],
        # all normalized to [0, 1].
        x, y, w, h = det.bounding_box

        x_center = x + w / 2
        y_center = y + h / 2

        lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
        )

        if det.label == "Bird":
            bird_boxes += 1
        else:
            raven_boxes += 1

    if not lines:
        skipped += 1
        continue

    dest_image = IMAGE_DIR / image_path.name
    dest_label = LABEL_DIR / f"{image_path.stem}.txt"

    if not dest_image.exists():
        dest_image.write_bytes(image_path.read_bytes())

    dest_label.write_text("".join(lines), encoding="utf-8")

    successful += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"Images with usable boxes: {successful:,}")
print(f"Skipped (no boxes/missing file): {skipped:,}")
print(f"Bird boxes:  {bird_boxes:,}")
print(f"Raven boxes: {raven_boxes:,}")

print(f"\nImages:\n{IMAGE_DIR}")
print(f"\nYOLO labels:\n{LABEL_DIR}")

print("\nClass mapping:")
print("0 = bird (includes Open Images 'Bird' and 'Raven' source boxes)")
