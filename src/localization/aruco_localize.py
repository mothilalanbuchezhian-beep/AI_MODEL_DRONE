"""ArUco marker detection and pose estimation for drone landing/docking guidance.

Subcommands:
  generate  write printable marker PNGs
  detect    run on a video file or camera index, print/draw marker pose
  selftest  render a synthetic marker at a known pose and check the recovered pose

Pose is the marker position in the camera frame (x right, y down, z forward, meters).
Accurate metric pose needs a calibrated camera (--calib file.npz with keys
camera_matrix, dist_coeffs); without it a rough pinhole model is assumed
(horizontal FOV --fov-deg) and distances are approximate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50


def make_detector() -> cv2.aruco.ArucoDetector:
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTIONARY), cv2.aruco.DetectorParameters()
    )


def approx_intrinsics(width: int, height: int, fov_deg: float):
    fx = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    camera_matrix = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)
    return camera_matrix, np.zeros(5)


def marker_object_points(length: float) -> np.ndarray:
    h = length / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def estimate_poses(corners, ids, length, camera_matrix, dist_coeffs):
    poses = []
    obj = marker_object_points(length)
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(
            obj, marker_corners.reshape(4, 2), camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if ok:
            poses.append((int(marker_id), rvec, tvec.flatten()))
    return poses


def cmd_generate(args) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    for marker_id in args.ids:
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, args.size_px)
        img = cv2.copyMakeBorder(img, args.border_px, args.border_px, args.border_px,
                                 args.border_px, cv2.BORDER_CONSTANT, value=255)
        path = out / f"aruco_4x4_50_id{marker_id}.png"
        cv2.imwrite(str(path), img)
        print(f"Wrote {path}")
    print("Print with the white border intact and measure the black square's side "
          "length -- that is --marker-length for detect.")


def cmd_detect(args) -> None:
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.calib:
        data = np.load(args.calib)
        camera_matrix, dist_coeffs = data["camera_matrix"], data["dist_coeffs"]
    else:
        camera_matrix, dist_coeffs = approx_intrinsics(width, height, args.fov_deg)
        print(f"No calibration given: assuming {args.fov_deg} deg horizontal FOV, "
              "distances are approximate.")

    detector = make_detector()
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    log, frame_index, frames_with_marker = [], 0, 0
    while args.max_frames == 0 or frame_index < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            frames_with_marker += 1
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_id, rvec, tvec in estimate_poses(corners, ids, args.marker_length,
                                                        camera_matrix, dist_coeffs):
                x, y, z = tvec
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec,
                                  args.marker_length * 0.5)
                text = f"id{marker_id} x={x:+.2f} y={y:+.2f} z={z:.2f} m"
                cv2.putText(frame, text, (10, 30 + 28 * (marker_id % 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                log.append({"frame": frame_index, "time_s": round(frame_index / fps, 2),
                            "id": marker_id, "x_m": round(float(x), 3),
                            "y_m": round(float(y), 3), "z_m": round(float(z), 3)})
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("aruco", frame)
            if frame_index == 0:
                cv2.setWindowProperty("aruco", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_index += 1

    cap.release()
    if writer:
        writer.release()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Frames: {frame_index}, with marker: {frames_with_marker} "
          f"({100 * frames_with_marker / max(1, frame_index):.1f}%)")


def cmd_selftest(args) -> None:
    width, height = 1280, 720
    camera_matrix, dist_coeffs = approx_intrinsics(width, height, 70.0)
    length = 0.20
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 400)
    marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

    cases = [
        ("straight below, 1.5 m", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.5])),
        ("offset + tilted, 2.0 m", np.array([0.3, -0.2, 0.1]), np.array([0.25, -0.15, 2.0])),
        ("far + tilted, 4.0 m", np.array([-0.4, 0.3, 0.2]), np.array([-0.5, 0.3, 4.0])),
    ]
    detector = make_detector()
    worst = 0.0
    for name, rot, trans in cases:
        h = length * 600 / 400 / 2
        pts3d = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], dtype=np.float32)
        pts2d, _ = cv2.projectPoints(pts3d, rot, trans, camera_matrix, dist_coeffs)
        src = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(src, pts2d.reshape(4, 2).astype(np.float32))
        frame = np.full((height, width), 180, dtype=np.uint8)
        warped = cv2.warpPerspective(marker, homography, (width, height),
                                     flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full(marker.shape, 255, np.uint8), homography,
                                   (width, height), flags=cv2.INTER_NEAREST)
        frame[mask > 0] = warped[mask > 0]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            print(f"[FAIL] {name}: marker not detected")
            worst = float("inf")
            continue
        _, _, tvec = estimate_poses(corners, ids, length, camera_matrix, dist_coeffs)[0]
        error = float(np.linalg.norm(tvec - trans))
        relative = error / float(np.linalg.norm(trans))
        worst = max(worst, relative)
        print(f"[{'OK' if relative < 0.03 else 'BAD'}] {name}: true={trans.round(3)} "
              f"est={tvec.round(3)} error={error * 100:.1f} cm ({relative * 100:.1f}% of distance)")
    print("SELFTEST PASSED" if worst < 0.03 else "SELFTEST FAILED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    gen.add_argument("--size-px", type=int, default=600)
    gen.add_argument("--border-px", type=int, default=100)
    gen.add_argument("--output-dir", type=Path, default=Path("results/aruco_markers"))
    gen.set_defaults(func=cmd_generate)

    det = sub.add_parser("detect")
    det.add_argument("source", help="Video file path or camera index (e.g. 0)")
    det.add_argument("--marker-length", type=float, required=True, help="Black square side in meters")
    det.add_argument("--calib", type=Path, default=None)
    det.add_argument("--fov-deg", type=float, default=70.0)
    det.add_argument("--output", type=Path, default=None, help="Annotated video path")
    det.add_argument("--log", type=Path, default=None, help="JSON pose log path")
    det.add_argument("--show", action="store_true", help="Live window (Esc to quit)")
    det.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    det.set_defaults(func=cmd_detect)

    st = sub.add_parser("selftest")
    st.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
