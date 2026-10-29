"""Scenario tests for the mission logic, using the fake drone. Run: python src/simulation/test_mission.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backends import FakeDrone  # noqa: E402
from mission import Config, Mission, State  # noqa: E402

FPS = 25.0


def run(hit_at, seconds: float, drone: FakeDrone | None = None, config: Config | None = None) -> Mission:
    mission = Mission(drone or FakeDrone(), config or Config())
    for i in range(int(seconds * FPS)):
        now = i / FPS
        mission.step(now, hit_at(now))
    return mission


def outcomes(mission: Mission) -> list:
    return [m["outcome"] for m in mission.missions]


def test_no_detections_stays_idle() -> None:
    m = run(lambda t: False, 60)
    assert m.state == State.IDLE and not m.missions


def test_single_frame_false_positive_is_ignored() -> None:
    m = run(lambda t: abs(t - 5.0) < 0.5 / FPS, 60)
    assert m.state == State.IDLE and not m.missions, m.events


def test_short_burst_is_ignored() -> None:
    m = run(lambda t: 5.0 <= t < 5.4, 60)  # 10 frames, below the 20-observation minimum
    assert m.state == State.IDLE and not m.missions


def test_raven_leaves_during_first_level() -> None:
    m = run(lambda t: 5.0 <= t < 30.0, 400)
    assert outcomes(m) == ["cleared"], m.events
    assert m.missions[0]["levels_used"] >= 1
    assert m.backend.alt == 0.0 and m.backend.mode == "ground"


def test_stubborn_raven_escalates_then_gives_up() -> None:
    m = run(lambda t: 5.0 <= t < 60.0, 400)
    assert outcomes(m) == ["gave_up"], m.events
    assert m.missions[0]["levels_used"] == 3
    assert len(m.backend.actions) == 3


def test_persistent_raven_relaunches_after_cooldown() -> None:
    m = run(lambda t: 5.0 <= t < 200.0, 600)
    assert len(m.missions) >= 2 and outcomes(m)[0] == "gave_up", m.events


def test_low_battery_blocks_launch() -> None:
    m = run(lambda t: t >= 5.0, 60, drone=FakeDrone(battery_pct=40.0))
    assert not m.missions and m.backend.alt == 0.0
    assert any("too low" in e[2] for e in m.events)


def test_battery_drain_aborts_mission() -> None:
    m = run(lambda t: t >= 5.0, 600, drone=FakeDrone(drain_pct_per_s=2.0))
    assert outcomes(m)[:1] == ["battery"], m.events


def test_cooldown_prevents_immediate_relaunch() -> None:
    m = run(lambda t: 5.0 <= t < 20.0 or 70.0 <= t < 85.0, 130)
    assert len(m.missions) == 1, m.events
    assert any(e[1] == "cooldown" for e in m.events)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
