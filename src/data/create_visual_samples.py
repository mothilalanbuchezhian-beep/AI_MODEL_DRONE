"""Render deterministic YOLO box samples for a human QA review."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/plots/annotation_samples"))
    args = parser.parse_args()
    image_dir, label_dir = args.dataset / "images" / args.split, args.dataset / "labels" / args.split
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    chosen = random.Random(42).sample(images, min(args.count, len(images)))
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in chosen:
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for line in label_path.read_text(encoding="utf-8").splitlines():
                _, x, y, box_width, box_height = line.split()
                x, y, box_width, box_height = map(float, (x, y, box_width, box_height))
                left, top = (x - box_width / 2) * width, (y - box_height / 2) * height
                right, bottom = (x + box_width / 2) * width, (y + box_height / 2) * height
                draw.rectangle((left, top, right, bottom), outline="lime", width=max(2, width // 300))
                draw.text((left, max(0, top - 15)), "bird", fill="lime")
            image.save(args.output / f"{image_path.stem}.jpg", quality=95)
    print(f"Wrote {len(chosen)} visual QA samples to {args.output}")


if __name__ == "__main__":
    main()
