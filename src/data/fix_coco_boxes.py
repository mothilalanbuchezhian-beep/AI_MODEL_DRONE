from pathlib import Path

PROJECT_ROOT = Path(r"D:\HIWI_SENSIG\AI_MODEL")

DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coco_bird"
)

SPLITS = ["train", "val", "test"]

fixed_boxes = 0
removed_boxes = 0


for split in SPLITS:

    label_dir = DATASET / "labels" / split

    if not label_dir.exists():
        print(f"WARNING: Missing directory: {label_dir}")
        continue

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

            # ------------------------------------------------
            # YOLO -> corner coordinates
            # ------------------------------------------------

            x1 = x_center - width / 2
            y1 = y_center - height / 2

            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # ------------------------------------------------
            # Clip to image boundaries
            # ------------------------------------------------

            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))

            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # ------------------------------------------------
            # Check valid area
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:
                removed_boxes += 1
                continue

            # ------------------------------------------------
            # Convert back to YOLO
            # ------------------------------------------------

            new_x_center = (x1 + x2) / 2
            new_y_center = (y1 + y2) / 2

            new_width = x2 - x1
            new_height = y2 - y1

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Use 10 decimal places instead of 6.
            # This avoids tiny floating-point boundary errors.
            # ------------------------------------------------

            new_lines.append(
                f"{class_id} "
                f"{new_x_center:.10f} "
                f"{new_y_center:.10f} "
                f"{new_width:.10f} "
                f"{new_height:.10f}"
            )

            fixed_boxes += 1

        # ----------------------------------------------------
        # Save cleaned annotations
        # ----------------------------------------------------

        label_file.write_text(
            "\n".join(new_lines)
            + ("\n" if new_lines else ""),
            encoding="utf-8"
        )


print("=" * 60)
print("COCO BIRD BOUNDING BOX CLEANUP COMPLETE")
print("=" * 60)
print(f"Boxes processed:   {fixed_boxes}")
print(f"Invalid removed:   {removed_boxes}")
print(f"Dataset:           {DATASET}")
print("=" * 60)