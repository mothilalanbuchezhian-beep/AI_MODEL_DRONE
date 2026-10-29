"""Measure bird-detector false positives on bird-free images and on a farm video.

Images: a bird-free folder (e.g. data/processed/hard_negatives/images/test); every
detection there is a false positive, so we report the share of images with >=1
detection and the mean detections per image.

Video: no ground truth exists, so this reports (a) the share of sampled frames
with >=1 detection and (b) the share of detections lying mostly inside a sheep
box found by the stock COCO model. (b) is a PROXY for "lamb mistaken for bird";
a real raven standing in front of a lamb can also count, so check frames by eye
and label a small subset before quoting numbers in a paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def inside_fraction(box, zone) -> float:
    ix1, iy1 = max(box[0], zone[0]), max(box[1], zone[1])
    ix2, iy2 = min(box[2], zone[2]), min(box[3], zone[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def eval_images(model: YOLO, folder: Path, conf: float, imgsz: int, device: str) -> dict:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    with_detection = total_boxes = 0
    for result in model.predict(source=[str(p) for p in paths], conf=conf, imgsz=imgsz,
                                device=device, stream=True, verbose=False):
        n = len(result.boxes)
        with_detection += n > 0
        total_boxes += n
    return {"images": len(paths), "images_with_detection": with_detection,
            "pct_images_with_false_positive": round(100 * with_detection / max(1, len(paths)), 2),
            "mean_detections_per_image": round(total_boxes / max(1, len(paths)), 3)}


def eval_video(model: YOLO, sheep_model: YOLO, video: Path, conf: float, imgsz: int,
               device: str, stride: int, sheep_overlap: float) -> dict:
    sheep_id = next(k for k, v in sheep_model.names.items() if v == "sheep")
    cap = cv2.VideoCapture(str(video))
    frames = frames_with_det = dets = dets_on_sheep = index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames += 1
            birds = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            sheep = sheep_model.predict(frame, conf=0.25, imgsz=imgsz, device=device,
                                        classes=[sheep_id], verbose=False)[0]
            sheep_boxes = sheep.boxes.xyxy.tolist()
            bird_boxes = birds.boxes.xyxy.tolist()
            frames_with_det += bool(bird_boxes)
            for box in bird_boxes:
                dets += 1
                dets_on_sheep += any(inside_fraction(box, z) >= sheep_overlap for z in sheep_boxes)
        index += 1
    cap.release()
    return {"frames_sampled": frames, "pct_frames_with_detection": round(100 * frames_with_det / max(1, frames), 2),
            "detections": dets, "detections_mostly_inside_sheep_box": dets_on_sheep,
            "pct_detections_on_sheep_proxy": round(100 * dets_on_sheep / max(1, dets), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--negatives", type=Path, default=Path("data/processed/hard_negatives/images/test"))
    parser.add_argument("--video", type=Path, default=Path("test video/IMG_5091.MP4"))
    parser.add_argument("--sheep-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=5, help="Use every Nth video frame")
    parser.add_argument("--sheep-overlap", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=None, help="Write results JSON here")
    args = parser.parse_args()

    sheep_model = YOLO(str(args.sheep_model))
    report = {}
    for model_path in args.models:
        model = YOLO(str(model_path))
        report[str(model_path)] = {}
        for conf in args.confs:
            entry = {}
            if args.negatives.exists():
                entry["negative_images"] = eval_images(model, args.negatives, conf, args.imgsz, args.device)
            if args.video.exists():
                entry["farm_video"] = eval_video(model, sheep_model, args.video, conf, args.imgsz,
                                                 args.device, args.stride, args.sheep_overlap)
            report[str(model_path)][f"conf_{conf}"] = entry
            print(f"\n{model_path}  conf={conf}")
            print(json.dumps(entry, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
