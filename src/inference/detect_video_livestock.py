"""Report a bird only when it's near a sheep/lamb in a video.

Runs two models per frame:
  - our custom single-class detector for "bird" (the raven-biased model)
  - the stock COCO-pretrained yolo11n.pt (already in the repo root) for
    "sheep", since COCO already has a solid sheep class and no new
    training or download is needed for that half.

"Near" is a 2D heuristic, not a real-world distance measurement (no
depth/calibration here): each sheep box is expanded outward by
--near-margin times its own size, and a bird is flagged as "near" if its
box overlaps that expanded zone. This is a first-pass gate to cut down
false triggers from birds elsewhere in the sky, not a precise metric.

No tracking (e.g. ByteTrack) yet -- this is a per-frame check. Adding a
tracker would give persistent IDs and smooth out single-frame flicker;
worth doing once this heuristic is validated on real footage.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_BIRD_MODEL = Path("models/trained/yolo11n_bird_cluster_50ep_best.pt")
DEFAULT_LIVESTOCK_MODEL = Path("yolo11n.pt")  # stock COCO weights
SHEEP_CLASS_ID = 18  # COCO class index in the stock model, verified at runtime


@dataclass
class ProximityEvent:
    start_frame: int
    start_time: float
    end_frame: int = 0
    end_time: float = 0.0
    max_bird_confidence: float = 0.0
    frame_count: int = 0

    def to_dict(self) -> dict:
        return {
            "start_time_s": round(self.start_time, 2),
            "end_time_s": round(self.end_time, 2),
            "duration_s": round(self.end_time - self.start_time, 2),
            "max_bird_confidence": round(self.max_bird_confidence, 3),
            "frames": self.frame_count,
        }


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{int(minutes):02d}:{secs:05.2f}"


def expand_box(box, margin_ratio: float, frame_w: int, frame_h: int):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    dx, dy = width * margin_ratio, height * margin_ratio
    return (
        max(0, x1 - dx), max(0, y1 - dy),
        min(frame_w, x2 + dx), min(frame_h, y2 + dy),
    )


def boxes_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to a video file")
    parser.add_argument("--bird-model", type=Path, default=DEFAULT_BIRD_MODEL)
    parser.add_argument("--livestock-model", type=Path, default=DEFAULT_LIVESTOCK_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--near-margin", type=float, default=1.0,
                         help="Sheep-box expansion ratio defining the 'near' zone")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu",
                         help="Defaults to CPU so it doesn't fight the GPU training job")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-video-out", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    for model_path in (args.bird_model, args.livestock_model):
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

    output_dir = args.output_dir or Path("results/predictions") / f"{args.source.stem}_livestock"
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = cv2.VideoCapture(str(args.source))
    if not probe.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    bird_model = YOLO(str(args.bird_model))
    livestock_model = YOLO(str(args.livestock_model))
    bird_class_name = bird_model.names.get(0, "bird")
    sheep_ids = [k for k, v in livestock_model.names.items() if v == "sheep"]
    if not sheep_ids:
        raise RuntimeError(f"'sheep' class not found in {args.livestock_model}")
    sheep_class_id = sheep_ids[0]

    print("=" * 70)
    print(f"Video:            {args.source}")
    print(f"Bird model:       {args.bird_model} (class: {bird_class_name})")
    print(f"Livestock model:  {args.livestock_model} (sheep class id: {sheep_class_id})")
    print(f"FPS: {fps:.2f}  Frames: {total_frames}  Size: {frame_width}x{frame_height}")
    print(f"Confidence threshold: {args.conf}  Near-margin: {args.near_margin}x sheep box size")
    print(f"Device: {args.device}")
    print("=" * 70)

    writer = None
    if not args.no_video_out:
        video_path = output_dir / f"{args.source.stem}_livestock_annotated.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

    events: list[ProximityEvent] = []
    frame_log: list[dict] = []
    active_event: ProximityEvent | None = None
    frames_with_bird = 0
    frames_with_sheep = 0
    frames_with_proximity = 0
    frame_index = 0

    bird_results = bird_model.predict(
        source=str(args.source), conf=args.conf, imgsz=args.imgsz,
        device=args.device, stream=True, verbose=False,
    )
    livestock_results = livestock_model.predict(
        source=str(args.source), conf=args.conf, imgsz=args.imgsz,
        device=args.device, classes=[sheep_class_id], stream=True, verbose=False,
    )

    for bird_result, livestock_result in zip(bird_results, livestock_results):
        timestamp = frame_index / fps
        frame = bird_result.orig_img.copy()

        bird_boxes = bird_result.boxes.xyxy.tolist() if bird_result.boxes is not None else []
        bird_confs = bird_result.boxes.conf.tolist() if bird_result.boxes is not None else []
        sheep_boxes = livestock_result.boxes.xyxy.tolist() if livestock_result.boxes is not None else []

        if bird_boxes:
            frames_with_bird += 1
        if sheep_boxes:
            frames_with_sheep += 1

        expanded_sheep = [expand_box(b, args.near_margin, frame_width, frame_height) for b in sheep_boxes]
        near_pairs = [
            (bird_box, conf)
            for bird_box, conf in zip(bird_boxes, bird_confs)
            if any(boxes_overlap(bird_box, zone) for zone in expanded_sheep)
        ]

        for x1, y1, x2, y2 in sheep_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, "sheep", (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for box, conf in zip(bird_boxes, bird_confs):
            x1, y1, x2, y2 = box
            is_near = box in [b for b, _ in near_pairs]
            color = (0, 0, 255) if is_near else (0, 200, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{bird_class_name} NEAR SHEEP" if is_near else bird_class_name
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if near_pairs:
            frames_with_proximity += 1
            max_conf = max(conf for _, conf in near_pairs)
            frame_log.append({
                "frame": frame_index, "time_s": round(timestamp, 2),
                "bird_near_sheep_count": len(near_pairs), "max_confidence": round(max_conf, 3),
            })
            if active_event is None:
                active_event = ProximityEvent(start_frame=frame_index, start_time=timestamp)
                print(f"[{format_timestamp(timestamp)}] {bird_class_name.upper()} NEAR SHEEP "
                      f"(confidence {max_conf:.2f})")
            active_event.frame_count += 1
            active_event.max_bird_confidence = max(active_event.max_bird_confidence, max_conf)
            active_event.end_frame = frame_index
            active_event.end_time = timestamp
        elif active_event is not None:
            print(f"[{format_timestamp(timestamp)}] clear "
                  f"(proximity lasted {active_event.end_time - active_event.start_time:.1f}s)")
            events.append(active_event)
            active_event = None

        if writer is not None:
            writer.write(frame)

        frame_index += 1

    if active_event is not None:
        events.append(active_event)
    if writer is not None:
        writer.release()

    summary = {
        "video": str(args.source),
        "bird_model": str(args.bird_model),
        "livestock_model": str(args.livestock_model),
        "confidence_threshold": args.conf,
        "near_margin_ratio": args.near_margin,
        "fps": fps,
        "total_frames": frame_index,
        "frames_with_bird": frames_with_bird,
        "frames_with_sheep": frames_with_sheep,
        "frames_with_bird_near_sheep": frames_with_proximity,
        "proximity_events": [event.to_dict() for event in events],
    }
    (output_dir / "livestock_proximity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "frame_log.json").write_text(json.dumps(frame_log, indent=2), encoding="utf-8")

    print("=" * 70)
    if events:
        print(f"RESULT: bird-near-sheep in {len(events)} event(s), "
              f"{frames_with_proximity}/{frame_index} frames "
              f"({100 * frames_with_proximity / max(1, frame_index):.1f}%).")
    else:
        print("RESULT: no bird-near-sheep proximity detected in this video.")
    print(f"(for reference: bird seen in {frames_with_bird} frames, "
          f"sheep seen in {frames_with_sheep} frames)")
    print(f"Summary: {output_dir / 'livestock_proximity_summary.json'}")
    if writer is not None:
        print(f"Annotated video: {output_dir / (args.source.stem + '_livestock_annotated.mp4')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
