from pathlib import Path
import shutil
import hashlib


PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

# ------------------------------------------------------------
# INPUT DATASETS
# ------------------------------------------------------------

OPEN_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "open_images_bird"
)

COCO = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coco_bird"
)

# ------------------------------------------------------------
# FINAL COMBINED DATASET
# ------------------------------------------------------------

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bird_detection"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

def file_hash(path: Path) -> str:
    """Return SHA256 hash of an image."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_images(dataset: Path, split: str):
    """Return all images in a dataset split."""

    image_dir = dataset / "images" / split

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing image directory: {image_dir}"
        )

    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_output():

    for split in SPLITS:

        (OUTPUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def copy_dataset(
    dataset: Path,
    dataset_name: str,
    split: str,
    existing_hashes: dict,
):

    images = get_images(
        dataset,
        split,
    )

    source_labels = (
        dataset
        / "labels"
        / split
    )

    copied = 0
    skipped = 0

    for image_path in images:

        label_path = (
            source_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: Missing label: "
                f"{label_path}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Check image content for duplicates
        # ----------------------------------------------------

        image_hash = file_hash(
            image_path
        )

        if image_hash in existing_hashes:

            previous = existing_hashes[
                image_hash
            ]

            print(
                "WARNING: Duplicate image detected:"
            )

            print(
                f"  Current : {image_path}"
            )

            print(
                f"  Existing: {previous}"
            )

            skipped += 1
            continue

        existing_hashes[image_hash] = (
            dataset_name,
            split,
            image_path.name,
        )

        # ----------------------------------------------------
        # Prefix filenames with dataset name
        # ----------------------------------------------------

        new_stem = (
            f"{dataset_name}_{image_path.stem}"
        )

        new_image = (
            OUTPUT
            / "images"
            / split
            / f"{new_stem}{image_path.suffix.lower()}"
        )

        new_label = (
            OUTPUT
            / "labels"
            / split
            / f"{new_stem}.txt"
        )

        # ----------------------------------------------------
        # Copy files
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            new_image,
        )

        shutil.copy2(
            label_path,
            new_label,
        )

        copied += 1

    return copied, skipped


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

print("=" * 70)
print("COMBINING BIRD DETECTION DATASETS")
print("=" * 70)

print(
    f"Open Images:\n{OPEN_IMAGES}"
)

print(
    f"\nCOCO:\n{COCO}"
)

print(
    f"\nOutput:\n{OUTPUT}"
)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

for dataset in (
    OPEN_IMAGES,
    COCO,
):

    if not dataset.exists():

        raise FileNotFoundError(
            f"Dataset not found: {dataset}"
        )


prepare_output()

# Hashes are maintained separately for each split.
#
# This prevents an image appearing in train and test,
# even if it comes from different source datasets.

split_hashes = {
    split: {}
    for split in SPLITS
}

totals = {
    split: {
        "open_images": 0,
        "coco": 0,
        "skipped": 0,
    }
    for split in SPLITS
}


# ------------------------------------------------------------
# Combine datasets
# ------------------------------------------------------------

for split in SPLITS:

    print()
    print("-" * 70)
    print(f"SPLIT: {split}")
    print("-" * 70)

    # Open Images

    copied, skipped = copy_dataset(
        OPEN_IMAGES,
        "openimages",
        split,
        split_hashes[split],
    )

    totals[split][
        "open_images"
    ] = copied

    totals[split][
        "skipped"
    ] += skipped

    print(
        f"Open Images: {copied} copied, "
        f"{skipped} skipped"
    )

    # COCO

    copied, skipped = copy_dataset(
        COCO,
        "coco",
        split,
        split_hashes[split],
    )

    totals[split][
        "coco"
    ] = copied

    totals[split][
        "skipped"
    ] += skipped

    print(
        f"COCO:        {copied} copied, "
        f"{skipped} skipped"
    )


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("COMBINATION COMPLETE")
print("=" * 70)

grand_total = 0

for split in SPLITS:

    open_count = totals[split][
        "open_images"
    ]

    coco_count = totals[split][
        "coco"
    ]

    total = (
        open_count
        + coco_count
    )

    grand_total += total

    print(
        f"{split:5s}: "
        f"{total:4d} images "
        f"({open_count} Open Images + "
        f"{coco_count} COCO)"
    )

print()
print(
    f"TOTAL: {grand_total} images"
)

print()
print(
    f"Output:\n{OUTPUT}"
)

print("=" * 70)