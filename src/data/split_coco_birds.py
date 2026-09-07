from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

SOURCE = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "coco2017"
    / "bird_subset"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coco_bird"
)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(SEED)

source_images = SOURCE / "images"
source_labels = SOURCE / "labels"

if not source_images.exists():
    raise FileNotFoundError(source_images)

if not source_labels.exists():
    raise FileNotFoundError(source_labels)

images = [
    p for p in source_images.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

random.shuffle(images)

total = len(images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:],
}

print("=" * 60)
print("COCO BIRD DATASET SPLIT")
print("=" * 60)
print(f"Total images: {total}")
print(f"Seed: {SEED}")
print()

for split in splits:
    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

for split, split_images in splits.items():

    copied = 0

    for image_path in split_images:

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(
                f"WARNING: Missing label for {image_path.name}"
            )
            continue

        shutil.copy2(
            image_path,
            OUTPUT / "images" / split / image_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT / "labels" / split / label_path.name
        )

        copied += 1

    print(
        f"{split}: {copied} images"
    )

print()
print("COCO dataset split complete.")
print(f"Output: {OUTPUT}")
print("=" * 60)