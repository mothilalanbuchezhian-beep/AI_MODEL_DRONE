"""Add COCO val2017 bird images into the same bird_subset pool as train2017.

train2017's 3,237 bird images are already fully used (see
download_coco_birds.py). val2017 contributes a further ~125 bird images
that have not been pulled in yet; they are written into the same
bird_subset/images and bird_subset/labels folders (COCO train/val image
IDs never collide) so split_coco_birds.py can re-split the combined pool.
"""
from pathlib import Path
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

ANNOTATIONS = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "coco2017"
    / "annotations"
    / "instances_val2017.json"
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

BIRD_CATEGORY_ID = 16

IMAGE_URL = "http://images.cocodataset.org/val2017/{filename}"


print("=" * 70)
print("COCO val2017 BIRD DOWNLOAD (supplement to train2017)")
print("=" * 70)

with ANNOTATIONS.open("r", encoding="utf-8") as f:
    coco = json.load(f)

bird_annotations = {}

for annotation in coco["annotations"]:
    if annotation["category_id"] != BIRD_CATEGORY_ID:
        continue
    bird_annotations.setdefault(annotation["image_id"], []).append(annotation)

print(f"val2017 bird images available: {len(bird_annotations):,}")
print(f"val2017 bird boxes: {sum(len(v) for v in bird_annotations.values()):,}")

image_info = {image["id"]: image for image in coco["images"]}

session = requests.Session()
successful = 0
failed = 0

for image_id, annotations in tqdm(bird_annotations.items(), total=len(bird_annotations)):

    info = image_info[image_id]
    filename = info["file_name"]

    image_path = IMAGE_DIR / filename
    label_path = LABEL_DIR / (Path(filename).stem + ".txt")

    if image_path.exists() and label_path.exists():
        successful += 1
        continue

    url = IMAGE_URL.format(filename=filename)

    try:
        response = session.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(image_path, "JPEG", quality=95)

        image_width = info["width"]
        image_height = info["height"]

        with label_path.open("w", encoding="utf-8") as label_file:
            for annotation in annotations:
                x, y, w, h = annotation["bbox"]
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width = w / image_width
                height = h / image_height
                label_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        successful += 1

    except Exception as exc:
        failed += 1
        print(f"\nFAILED: {filename}\nReason: {exc}")

print("\n" + "=" * 70)
print("val2017 BIRD DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Successful: {successful}")
print(f"Failed:     {failed}")
