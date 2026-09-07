"""Evaluate a trained detector and persist only actual reported metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="mixed_birds_evaluation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data), split="test", device=args.device, imgsz=args.imgsz,
        project=str(root / "runs" / "evaluation"), name=args.name, plots=True,
    )
    values = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    output = root / "results" / "metrics" / f"{args.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
