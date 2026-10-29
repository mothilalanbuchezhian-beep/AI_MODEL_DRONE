"""Backend-independent mission logic for the raven-deterrence drone.

States: IDLE -> LAUNCH -> APPROACH -> DETER -> RETURN -> LAND -> COOLDOWN -> IDLE.
The detector only supplies one boolean per frame ("raven near sheep"); this class
decides when that is trustworthy enough to launch, how to escalate, and when to
give up. The drone itself sits behind the DroneBackend interface so the same
logic runs against a fake drone (tests) or ArduPilot SITL (backends.py).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class State(str, Enum):
    IDLE = "idle"
    LAUNCH = "launch"
    APPROACH = "approach"
    DETER = "deter"
    RETURN = "return"
    LAND = "land"
    COOLDOWN = "cooldown"


class DroneBackend(Protocol):
    def tick(self, now: float) -> None: ...
    def takeoff(self, alt_m: float) -> None: ...
    def at_altitude(self) -> bool: ...
    def goto(self, north_m: float, east_m: float, alt_m: float) -> None: ...
    def arrived(self) -> bool: ...
    def deter(self, action: str) -> None: ...
    def return_home(self) -> None: ...
    def at_home(self) -> bool: ...
    def land(self) -> None: ...
    def landed(self) -> bool: ...
    def battery_pct(self) -> float: ...


@dataclass
class Config:
    # Confirmation: this share of observations in the window must be hits.
    confirm_window_s: float = 2.0
    confirm_min_obs: int = 20
    confirm_hit_ratio: float = 0.6
    # Where the flock is, relative to the dock (north/east metres), and flight height.
    target_north_m: float = 30.0
    target_east_m: float = 20.0
    cruise_alt_m: float = 10.0
    # Graded deterrence: (action, hold seconds). Escalates when the raven stays.
    deter_levels: tuple = (("sound", 8.0), ("sound+light", 8.0), ("close approach+sound+light", 10.0))
    clear_s: float = 5.0            # no hit for this long = raven gone
    cooldown_s: float = 120.0       # ignore new triggers after a mission
    launch_min_battery_pct: float = 50.0
    return_battery_pct: float = 30.0
    max_mission_s: float = 600.0


@dataclass
class Mission:
    backend: DroneBackend
    config: Config = field(default_factory=Config)
    state: State = State.IDLE
    events: list = field(default_factory=list)
    missions: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self._window: deque = deque()
        self._level = 0
        self._level_start = 0.0
        self._last_seen = 0.0
        self._mission_start = 0.0
        self._cooldown_until = 0.0
        self._outcome = ""
        self._levels_used = 0

    def _log(self, now: float, message: str) -> None:
        self.events.append((round(now, 2), self.state.value, message))

    def _confirmed(self, now: float, hit: bool) -> bool:
        cfg = self.config
        self._window.append((now, hit))
        while self._window and now - self._window[0][0] > cfg.confirm_window_s:
            self._window.popleft()
        if len(self._window) < cfg.confirm_min_obs:
            return False
        hits = sum(1 for _, h in self._window if h)
        return hits / len(self._window) >= cfg.confirm_hit_ratio

    def _go(self, state: State, now: float, message: str) -> None:
        self.state = state
        self._log(now, message)

    def _start_return(self, now: float, outcome: str, message: str) -> None:
        self._outcome = outcome
        self.backend.return_home()
        self._go(State.RETURN, now, message)

    def step(self, now: float, hit: bool) -> None:
        cfg, drone = self.config, self.backend
        drone.tick(now)

        if self.state == State.IDLE:
            if self._confirmed(now, hit):
                if drone.battery_pct() < cfg.launch_min_battery_pct:
                    self._log(now, f"raven confirmed but battery {drone.battery_pct():.0f}% too low, staying docked")
                    self._window.clear()
                    return
                self._mission_start = now
                self._level = self._levels_used = 0
                self._outcome = ""
                drone.takeoff(cfg.cruise_alt_m)
                self._go(State.LAUNCH, now, "raven near sheep confirmed, launching")

        elif self.state == State.LAUNCH:
            if drone.at_altitude():
                drone.goto(cfg.target_north_m, cfg.target_east_m, cfg.cruise_alt_m)
                self._go(State.APPROACH, now, "at cruise altitude, flying to the flock")

        elif self.state == State.APPROACH:
            if self._abort_needed(now):
                return
            if drone.arrived():
                self._last_seen = now
                self._level_start = now
                self._levels_used = 1
                drone.deter(cfg.deter_levels[0][0])
                self._go(State.DETER, now, f"deterrence level 1: {cfg.deter_levels[0][0]}")

        elif self.state == State.DETER:
            if hit:
                self._last_seen = now
            if self._abort_needed(now):
                return
            if now - self._last_seen >= cfg.clear_s:
                self._start_return(now, "cleared", "raven gone, returning to dock")
            elif now - self._level_start >= cfg.deter_levels[self._level][1]:
                if self._level + 1 < len(cfg.deter_levels):
                    self._level += 1
                    self._levels_used = self._level + 1
                    self._level_start = now
                    action = cfg.deter_levels[self._level][0]
                    drone.deter(action)
                    self._log(now, f"raven still there, escalating to level {self._level + 1}: {action}")
                else:
                    self._start_return(now, "gave_up", "all levels used, raven still there: returning, alert the farmer")

        elif self.state == State.RETURN:
            if drone.at_home():
                drone.land()
                self._go(State.LAND, now, "above dock, landing")

        elif self.state == State.LAND:
            if drone.landed():
                self.missions.append({
                    "start_s": round(self._mission_start, 2), "end_s": round(now, 2),
                    "duration_s": round(now - self._mission_start, 2),
                    "outcome": self._outcome, "levels_used": self._levels_used,
                })
                self._cooldown_until = now + cfg.cooldown_s
                self._window.clear()
                self._go(State.COOLDOWN, now, f"landed, mission {self._outcome}")

        elif self.state == State.COOLDOWN:
            if now >= self._cooldown_until:
                self._go(State.IDLE, now, "cooldown over, watching again")

    def _abort_needed(self, now: float) -> bool:
        cfg = self.config
        if self.backend.battery_pct() <= cfg.return_battery_pct:
            self._start_return(now, "battery", f"battery {self.backend.battery_pct():.0f}%, aborting and returning")
            return True
        if now - self._mission_start >= cfg.max_mission_s:
            self._start_return(now, "timeout", "max mission time reached, returning")
            return True
        return False


def distance(a: tuple, b: tuple) -> float:
    return math.dist(a, b)
