"""Reproducible YOLO Nano training entry point."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics choose a safe batch size")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="mixed_birds_yolo26n")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, seed=args.seed, deterministic=True, project=str(root / "runs" / "training"),
        name=args.name, exist_ok=False, workers=2, cache=False,
    )
    best = root / "runs" / "training" / args.name / "weights" / "best.pt"
    if best.exists():
        destination = root / "models" / "trained" / f"{args.name}_best.pt"
        shutil.copy2(best, destination)
        print(f"Saved curated model: {destination}")


if __name__ == "__main__":
    main()
