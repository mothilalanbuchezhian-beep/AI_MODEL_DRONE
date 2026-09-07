from pathlib import Path
import json
import random
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

ANNOTATIONS = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "coco2017"
    / "annotations"
    / "instances_train2017.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "coco2017"
    / "bird_subset"
)

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BIRD_CATEGORY_ID = 16

# TEST FIRST: only 50 images
TARGET_IMAGES = 3237

RANDOM_SEED = 42

# HTTP because HTTPS has certificate problems on this machine
IMAGE_URL = (
    "http://images.cocodataset.org/train2017/"
    "{filename}"
)


# ============================================================
# LOAD COCO ANNOTATIONS
# ============================================================

print("=" * 70)
print("COCO BIRD DATASET TEST DOWNLOAD")
print("=" * 70)

print("\nLoading COCO annotations...")

with ANNOTATIONS.open(
    "r",
    encoding="utf-8",
) as f:
    coco = json.load(f)


# ============================================================
# FIND BIRD ANNOTATIONS
# ============================================================

bird_annotations = {}

for annotation in coco["annotations"]:

    if annotation["category_id"] != BIRD_CATEGORY_ID:
        continue

    image_id = annotation["image_id"]

    bird_annotations.setdefault(
        image_id,
        []
    ).append(annotation)


print(
    f"COCO Bird images available: "
    f"{len(bird_annotations):,}"
)

print(
    f"COCO Bird bounding boxes: "
    f"{sum(len(v) for v in bird_annotations.values()):,}"
)


# ============================================================
# IMAGE METADATA
# ============================================================

image_info = {
    image["id"]: image
    for image in coco["images"]
}


# ============================================================
# SELECT 50 IMAGES
# ============================================================

image_ids = list(bird_annotations.keys())

random.seed(RANDOM_SEED)
random.shuffle(image_ids)

selected_ids = image_ids[:TARGET_IMAGES]

print(
    f"\nSelected for TEST: "
    f"{len(selected_ids)} images"
)


# ============================================================
# DOWNLOAD
# ============================================================

session = requests.Session()

successful = 0
failed = 0

print("\nDownloading COCO Bird test images...\n")

for image_id in tqdm(selected_ids):

    info = image_info[image_id]

    filename = info["file_name"]

    image_path = IMAGE_DIR / filename
    label_path = LABEL_DIR / (
        Path(filename).stem + ".txt"
    )

    if image_path.exists() and label_path.exists():
        successful += 1
        continue

    url = IMAGE_URL.format(
        filename=filename
    )

    try:

        response = session.get(
            url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image.save(
            image_path,
            "JPEG",
            quality=95,
        )

        # ====================================================
        # COCO -> YOLO
        #
        # COCO bbox:
        # [x_min, y_min, width, height]
        #
        # YOLO:
        # class x_center y_center width height
        # ====================================================

        image_width = info["width"]
        image_height = info["height"]

        with label_path.open(
            "w",
            encoding="utf-8",
        ) as label_file:

            for annotation in bird_annotations[
                image_id
            ]:

                x, y, w, h = annotation["bbox"]

                x_center = (
                    x + w / 2
                ) / image_width

                y_center = (
                    y + h / 2
                ) / image_height

                width = w / image_width
                height = h / image_height

                label_file.write(
                    f"0 "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

        successful += 1

    except Exception as exc:

        failed += 1

        print(
            f"\nFAILED: {filename}"
        )
        print(
            f"Reason: {exc}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COCO TEST DOWNLOAD COMPLETE")
print("=" * 70)

print(
    f"Successful images: {successful}"
)

print(
    f"Failed images:     {failed}"
)

print(
    f"\nImages:\n{IMAGE_DIR}"
)

print(
    f"\nLabels:\n{LABEL_DIR}"
)