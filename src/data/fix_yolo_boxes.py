from pathlib import Path

PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "open_images_bird"
)

SPLITS = ["train", "val", "test"]

fixed_boxes = 0
removed_boxes = 0

for split in SPLITS:

    label_dir = DATASET / "labels" / split

    for label_file in label_dir.glob("*.txt"):

        lines = label_file.read_text(
            encoding="utf-8"
        ).splitlines()

        new_lines = []

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                removed_boxes += 1
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                removed_boxes += 1
                continue

            # Original box limits
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Clamp box to image boundaries
            original = (x1, y1, x2, y2)

            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # Check whether box is still valid
            if x2 <= x1 or y2 <= y1:
                removed_boxes += 1
                continue

            new_x_center = (x1 + x2) / 2
            new_y_center = (y1 + y2) / 2
            new_width = x2 - x1
            new_height = y2 - y1

            if original != (x1, y1, x2, y2):
                fixed_boxes += 1

            # 10 decimal places avoids tiny floating-point boundary
            # errors that would otherwise fail strict [0, 1] validation.
            new_lines.append(
                f"{class_id} "
                f"{new_x_center:.10f} "
                f"{new_y_center:.10f} "
                f"{new_width:.10f} "
                f"{new_height:.10f}"
            )

        label_file.write_text(
            "\n".join(new_lines) + (
                "\n" if new_lines else ""
            ),
            encoding="utf-8"
        )

print("=" * 60)
print("YOLO BOUNDING BOX FIX COMPLETE")
print("=" * 60)
print(f"Boxes corrected: {fixed_boxes}")
print(f"Invalid boxes removed: {removed_boxes}")