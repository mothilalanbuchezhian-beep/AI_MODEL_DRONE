"""Run the trained detector over a recorded video file.

This is the offline prototype for the eventual pipeline: RGB camera on
the drone -> this detector on the Raspberry Pi -> a "target detected"
signal that commands the drone toward the bird/raven. Point it at a
downloaded farm clip to see whether/when the current model fires.

The model is a single-class detector (class 0, currently named "bird" in
configs/bird_detection.yaml) -- it was not trained to tell a crow/raven
apart from other bird species, so detections are reported using the
model's own class name rather than being relabeled "crow". See
docs/mixed_dataset_decision.md for why the class stayed generic.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = Path("models/trained/yolo11n_bird_50ep_best.pt")


@dataclass
class DetectionEvent:
    start_frame: int
    start_time: float
    end_frame: int = 0
    end_time: float = 0.0
    max_confidence: float = 0.0
    frame_count: int = 0

    def to_dict(self) -> dict:
        return {
            "start_time_s": round(self.start_time, 2),
            "end_time_s": round(self.end_time, 2),
            "duration_s": round(self.end_time - self.start_time, 2),
            "max_confidence": round(self.max_confidence, 3),
            "frames": self.frame_count,
        }


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{int(minutes):02d}:{secs:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to a video file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Defaults to results/predictions/<video stem>")
    parser.add_argument("--no-video-out", action="store_true",
                         help="Skip writing the annotated output video (log/summary only)")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model weights not found: {args.model}\n"
            "Train a model first (src/training) or pass --model explicitly."
        )

    output_dir = args.output_dir or Path("results/predictions") / args.source.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = cv2.VideoCapture(str(args.source))
    if not probe.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    model = YOLO(str(args.model))
    class_name = model.names.get(0, "bird")

    print("=" * 70)
    print(f"Video:      {args.source}")
    print(f"Model:      {args.model}")
    print(f"Class:      {class_name} (id 0)")
    print(f"FPS:        {fps:.2f}  Frames: {total_frames}  Size: {frame_width}x{frame_height}")
    print(f"Confidence threshold: {args.conf}")
    print("=" * 70)

    writer = None
    if not args.no_video_out:
        video_path = output_dir / f"{args.source.stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

    events: list[DetectionEvent] = []
    frame_log: list[dict] = []
    active_event: DetectionEvent | None = None
    frames_with_detection = 0
    frame_index = 0

    results = model.predict(
        source=str(args.source), conf=args.conf, imgsz=args.imgsz,
        device=args.device, stream=True, verbose=False,
    )

    for result in results:
        timestamp = frame_index / fps
        boxes = result.boxes
        detected = boxes is not None and len(boxes) > 0
        confidence = float(boxes.conf.max()) if detected else 0.0

        if detected:
            frames_with_detection += 1
            frame_log.append({
                "frame": frame_index, "time_s": round(timestamp, 2),
                "count": len(boxes), "max_confidence": round(confidence, 3),
            })
            if active_event is None:
                active_event = DetectionEvent(start_frame=frame_index, start_time=timestamp)
                print(f"[{format_timestamp(timestamp)}] {class_name.upper()} DETECTED "
                      f"(confidence {confidence:.2f})")
            active_event.frame_count += 1
            active_event.max_confidence = max(active_event.max_confidence, confidence)
            active_event.end_frame = frame_index
            active_event.end_time = timestamp
        elif active_event is not None:
            print(f"[{format_timestamp(timestamp)}] clear "
                  f"(was visible for {active_event.end_time - active_event.start_time:.1f}s)")
            events.append(active_event)
            active_event = None

        if writer is not None:
            writer.write(result.plot())

        frame_index += 1

    if active_event is not None:
        events.append(active_event)

    if writer is not None:
        writer.release()

    summary = {
        "video": str(args.source),
        "model": str(args.model),
        "class_name": class_name,
        "confidence_threshold": args.conf,
        "fps": fps,
        "total_frames": frame_index,
        "frames_with_detection": frames_with_detection,
        "detection_events": [event.to_dict() for event in events],
    }

    (output_dir / "detection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "frame_log.json").write_text(json.dumps(frame_log, indent=2), encoding="utf-8")

    print("=" * 70)
    if events:
        print(f"RESULT: {class_name} detected in {len(events)} event(s), "
              f"{frames_with_detection}/{frame_index} frames "
              f"({100 * frames_with_detection / max(1, frame_index):.1f}%).")
    else:
        print(f"RESULT: no {class_name} detected in this video at confidence >= {args.conf}.")
    print(f"Summary:   {output_dir / 'detection_summary.json'}")
    if writer is not None:
        print(f"Annotated video: {output_dir / (args.source.stem + '_annotated.mp4')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
