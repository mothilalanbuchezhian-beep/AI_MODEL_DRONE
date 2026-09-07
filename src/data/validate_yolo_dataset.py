"""Validate a YOLO detection dataset before training.

Checks corrupt images, missing labels, invalid boxes, class IDs and duplicate
image bytes. It deliberately fails on cross-split duplicates to prevent leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label(path: Path, classes: int) -> tuple[list[str], Counter[int]]:
    errors: list[str] = []
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric value")
            continue
        if not 0 <= class_id < classes:
            errors.append(f"{path}:{line_number}: class {class_id} outside [0, {classes - 1}]")
        if not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            errors.append(f"{path}:{line_number}: invalid normalized box")
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            errors.append(f"{path}:{line_number}: box extends outside image")
        counts[class_id] += 1
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="YOLO dataset root containing images/ and labels/")
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("results/metrics/dataset_validation.json"))
    args = parser.parse_args()

    report = {"splits": {}, "errors": [], "cross_split_duplicates": []}
    hashes: dict[str, tuple[str, str]] = {}
    total_boxes: Counter[int] = Counter()

    for split in ("train", "val", "test"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        split_report = {"images": 0, "missing_labels": 0, "corrupt_images": 0, "boxes": 0}
        if not image_dir.exists():
            report["errors"].append(f"Missing image directory: {image_dir}")
            report["splits"][split] = split_report
            continue
        for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            split_report["images"] += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError):
                split_report["corrupt_images"] += 1
                report["errors"].append(f"Corrupt image: {image_path}")
                continue
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                split_report["missing_labels"] += 1
                report["errors"].append(f"Missing label: {label_path}")
            else:
                errors, counts = validate_label(label_path, args.classes)
                report["errors"].extend(errors)
                split_report["boxes"] += sum(counts.values())
                total_boxes.update(counts)
            digest = sha256(image_path)
            if digest in hashes and hashes[digest][0] != split:
                report["cross_split_duplicates"].append({"first": hashes[digest][1], "duplicate": str(image_path)})
            else:
                hashes[digest] = (split, str(image_path))
        report["splits"][split] = split_report

    report["class_box_counts"] = dict(total_boxes)
    report["valid"] = not report["errors"] and not report["cross_split_duplicates"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
