"""Replay detector output through the mission logic and a (fake or SITL) drone.

Input is the frame_log.json + livestock_proximity_summary.json written by
src/inference/detect_video_livestock.py: one entry per frame where a bird was
near a sheep. A frame counts as a "hit" when its confidence >= --min-conf.

  Fake drone (no simulator needed, runs in simulated time):
    python src/simulation/run_simulation.py results/predictions/IMG_5091_livestock

  ArduPilot SITL (run in the Ubuntu VM, sim_vehicle.py already running, real time):
    python3 src/simulation/run_simulation.py <results folder> --backend ardupilot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backends import ArduPilotDrone, FakeDrone  # noqa: E402
from mission import Config, Mission, State  # noqa: E402


def load_hits(folder: Path, min_conf: float) -> tuple[list[bool], float]:
    summary = json.loads((folder / "livestock_proximity_summary.json").read_text(encoding="utf-8"))
    frame_log = json.loads((folder / "frame_log.json").read_text(encoding="utf-8"))
    fps, total = float(summary["fps"]), int(summary["total_frames"])
    hit_frames = {e["frame"] for e in frame_log if e["max_confidence"] >= min_conf}
    return [i in hit_frames for i in range(total)], fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("--backend", choices=["fake", "ardupilot"], default="fake")
    parser.add_argument("--connection", default="udpin:0.0.0.0:14550")
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--battery-start", type=float, default=100.0, help="Fake drone only")
    parser.add_argument("--tail-s", type=float, default=900.0, help="Keep running after the video ends")
    parser.add_argument("--output", type=Path, default=Path("results/simulation/mission_summary.json"))
    args = parser.parse_args()

    hits, fps = load_hits(args.results_folder, args.min_conf)
    drone = FakeDrone(battery_pct=args.battery_start) if args.backend == "fake" else ArduPilotDrone(args.connection)
    mission = Mission(drone, Config())
    realtime = args.backend == "ardupilot"

    print(f"Video: {len(hits)} frames at {fps:.1f} fps, {sum(hits)} hit frames (conf >= {args.min_conf})")
    dt, shown = 1.0 / fps, 0
    total_steps = len(hits) + int(args.tail_s * fps)
    trace = []
    for i in range(total_steps):
        now = i * dt
        mission.step(now, hits[i] if i < len(hits) else False)
        if i % 5 == 0:
            trace.append((round(now, 2), drone.n, drone.e, drone.alt, mission.state.value))
        while shown < len(mission.events):
            t, state, message = mission.events[shown]
            print(f"[{t:8.2f}s] {state:8s} {message}")
            shown += 1
        if i >= len(hits) and mission.state == State.IDLE:
            break
        if realtime:
            time.sleep(dt)

    outcomes = [m["outcome"] for m in mission.missions]
    summary = {
        "backend": args.backend, "min_conf": args.min_conf, "video_frames": len(hits),
        "hit_frames": sum(hits), "missions": mission.missions, "events": mission.events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{len(outcomes)} mission(s): {outcomes if outcomes else 'none'}  -> {args.output}")

    try:
        from plot_mission import plot_mission
    except ImportError:
        print("matplotlib not installed here, skipping the figure")
        return
    png = args.output.with_suffix(".png")
    if mission.missions:
        tmax = mission.missions[-1]["end_s"] + 15.0
        trace = [p for p in trace if p[0] <= tmax]
    cfg = mission.config
    label = "fake kinematic drone" if args.backend == "fake" else "ArduPilot SITL"
    plot_mission(trace, mission.events, hits, fps, (cfg.target_north_m, cfg.target_east_m),
                 f"Raven-deterrence mission replay ({label}, farm video detections, conf >= {args.min_conf})", png)
    print(f"Figure: {png}")


if __name__ == "__main__":
    main()
