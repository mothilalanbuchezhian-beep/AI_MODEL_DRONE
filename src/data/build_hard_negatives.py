"""Build a hard-negative (background) image set for the bird detector.

The farm-video test showed the detector boxing lambs as "bird". This pulls COCO
train2017 images that show look-alike animals (sheep first, then cow/horse/dog)
and contain NO bird, writes them with empty YOLO label files, and splits them
70/20/10 into data/processed/hard_negatives/. The original dataset is not
modified; the training YAMLs list both folders.

COCO annotations are CC BY 4.0, but the images carry mixed Flickr licenses, so
manifest.json records each image's COCO id, license id and Flickr URL for
publishing ID lists instead of images.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")
ANNOTATIONS = PROJECT_ROOT / "datasets" / "public" / "coco2017" / "annotations" / "instances_train2017.json"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "hard_negatives"

BIRD_ID = 16
# name -> (COCO category id, image quota). Order matters: an image is assigned
# to the first class it matches, so sheep images are never counted as cow/dog.
NEGATIVE_CLASSES = {"sheep": (20, 1500), "cow": (21, 350), "horse": (19, 200), "dog": (18, 200)}

IMAGE_URL = "http://images.cocodataset.org/train2017/{filename}"  # HTTPS has certificate problems here
SPLITS = (("train", 0.70), ("val", 0.20), ("test", 0.10))


def select_images(coco: dict, rng: random.Random) -> list[dict]:
    categories_by_image: dict[int, set[int]] = {}
    for ann in coco["annotations"]:
        categories_by_image.setdefault(ann["image_id"], set()).add(ann["category_id"])
    info = {img["id"]: img for img in coco["images"]}

    selected, taken = [], set()
    for name, (category_id, quota) in NEGATIVE_CLASSES.items():
        candidates = [
            image_id for image_id, cats in categories_by_image.items()
            if category_id in cats and BIRD_ID not in cats and image_id not in taken
        ]
        rng.shuffle(candidates)
        print(f"{name:6s}: {len(candidates):5d} bird-free images available, taking {min(quota, len(candidates))}")
        for image_id in candidates[:quota]:
            taken.add(image_id)
            img = info[image_id]
            selected.append({
                "coco_id": image_id, "file_name": img["file_name"], "source_class": name,
                "license": img.get("license"), "flickr_url": img.get("flickr_url"),
            })
    return selected


def assign_splits(items: list[dict], rng: random.Random) -> None:
    rng.shuffle(items)
    total = len(items)
    train_end = int(total * SPLITS[0][1])
    val_end = train_end + int(total * SPLITS[1][1])
    for index, item in enumerate(items):
        item["split"] = "train" if index < train_end else "val" if index < val_end else "test"


def download(item: dict, session: requests.Session) -> bool:
    split, filename = item["split"], item["file_name"]
    image_path = OUTPUT / "images" / split / filename
    label_path = OUTPUT / "labels" / split / (Path(filename).stem + ".txt")
    if image_path.exists() and label_path.exists():
        return True
    try:
        response = session.get(IMAGE_URL.format(filename=filename), timeout=60,
                               headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        Image.open(BytesIO(response.content)).convert("RGB").save(image_path, "JPEG", quality=95)
        label_path.write_text("", encoding="utf-8")
        return True
    except Exception as exc:
        print(f"FAILED {filename}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split, _ in SPLITS:
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations (large file)...")
    coco = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    items = select_images(coco, rng)
    del coco
    assign_splits(items, rng)

    session = requests.Session()
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item, session): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                ok += 1
            else:
                futures[future]["failed"] = True

    items = [i for i in items if not i.get("failed")]
    (OUTPUT / "manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"\nDownloaded {ok}/{len(futures)}")
    for split, _ in SPLITS:
        print(f"  {split}: {sum(1 for i in items if i['split'] == split)} images")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
