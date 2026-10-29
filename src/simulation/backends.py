"""Drone backends for mission.py: a fake drone for tests and ArduPilot SITL via pymavlink.

FakeDrone is a simple kinematic model (no wind, no dynamics) driven by the
simulation clock. ArduPilotDrone talks MAVLink to a running SITL instance, so it
needs ArduPilot's sim_vehicle.py running and `pip install pymavlink`; it has NOT
been run against a real SITL yet.
"""
from __future__ import annotations

import math
import time


class FakeDrone:
    def __init__(self, speed_mps: float = 5.0, climb_mps: float = 2.5, descend_mps: float = 1.5,
                 drain_pct_per_s: float = 0.04, battery_pct: float = 100.0) -> None:
        self.n = self.e = self.alt = 0.0
        self.mode = "ground"
        self.target = (0.0, 0.0, 0.0)
        self.target_alt = 0.0
        self.batt = battery_pct
        self.speed, self.climb, self.descend, self.drain = speed_mps, climb_mps, descend_mps, drain_pct_per_s
        self.last = None
        self.actions: list = []

    def tick(self, now: float) -> None:
        dt = 0.0 if self.last is None else max(0.0, now - self.last)
        self.last = now
        if dt == 0.0 or self.mode == "ground":
            return
        self.batt = max(0.0, self.batt - self.drain * dt)
        if self.mode == "climb":
            self.alt = min(self.target_alt, self.alt + self.climb * dt)
        elif self.mode == "goto":
            tn, te, ta = self.target
            dist = math.hypot(tn - self.n, te - self.e)
            step = self.speed * dt
            if dist <= step:
                self.n, self.e = tn, te
            elif dist > 0:
                self.n += (tn - self.n) / dist * step
                self.e += (te - self.e) / dist * step
            dz = ta - self.alt
            self.alt += math.copysign(min(abs(dz), self.climb * dt), dz)
        elif self.mode == "descend":
            self.alt = max(0.0, self.alt - self.descend * dt)
            if self.alt == 0.0:
                self.mode = "ground"

    def takeoff(self, alt_m: float) -> None:
        self.mode, self.target_alt = "climb", alt_m

    def at_altitude(self) -> bool:
        return self.alt >= self.target_alt - 0.5

    def goto(self, north_m: float, east_m: float, alt_m: float) -> None:
        self.mode, self.target = "goto", (north_m, east_m, alt_m)

    def arrived(self) -> bool:
        tn, te, ta = self.target
        return math.dist((self.n, self.e, self.alt), (tn, te, ta)) <= 1.0

    def deter(self, action: str) -> None:
        self.actions.append((self.last, action))

    def return_home(self) -> None:
        self.goto(0.0, 0.0, self.alt)

    def at_home(self) -> bool:
        return math.hypot(self.n, self.e) <= 1.0

    def land(self) -> None:
        self.mode = "descend"

    def landed(self) -> bool:
        return self.mode == "ground" and self.alt <= 0.05

    def battery_pct(self) -> float:
        return self.batt


class ArduPilotDrone:
    """ArduCopter SITL over MAVLink. Local NED frame, origin at the start position."""

    def __init__(self, connection: str = "udpin:0.0.0.0:14550", arm_timeout_s: float = 90.0) -> None:
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(connection)
        print(f"Waiting for heartbeat on {connection} ...")
        self.m.wait_heartbeat()
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
        self.n = self.e = self.alt = 0.0
        self.batt = 100.0
        self.target = (0.0, 0.0, 0.0)
        self.target_alt = 0.0
        self.arm_timeout_s = arm_timeout_s
        self.actions: list = []

    def _poll(self) -> None:
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "LOCAL_POSITION_NED":
                self.n, self.e, self.alt = msg.x, msg.y, -msg.z
            elif kind == "SYS_STATUS" and msg.battery_remaining >= 0:
                self.batt = float(msg.battery_remaining)

    def _set_mode(self, name: str) -> None:
        self.m.set_mode(self.m.mode_mapping()[name])

    def tick(self, now: float) -> None:
        self._poll()

    def takeoff(self, alt_m: float) -> None:
        self.target_alt = alt_m
        self._set_mode("GUIDED")
        deadline = time.time() + self.arm_timeout_s
        while True:
            self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                         self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            retry_at = time.time() + 5.0
            while time.time() < retry_at and not self.m.motors_armed():
                self._poll()
                self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if self.m.motors_armed():
                break
            if time.time() > deadline:
                raise RuntimeError("Could not arm: wait for the simulator's GPS/EKF to be ready and retry")
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     self.mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt_m)

    def at_altitude(self) -> bool:
        return self.alt >= self.target_alt - 0.7

    def goto(self, north_m: float, east_m: float, alt_m: float) -> None:
        self.target = (north_m, east_m, alt_m)
        # type_mask 3576: use position only, ignore velocity, acceleration, yaw and yaw rate.
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, 3576,
            north_m, east_m, -alt_m, 0, 0, 0, 0, 0, 0, 0, 0)

    def arrived(self) -> bool:
        tn, te, ta = self.target
        return math.dist((self.n, self.e, self.alt), (tn, te, ta)) <= 2.0

    def deter(self, action: str) -> None:
        print(f"   [DETERRENT] {action}")
        self.actions.append((time.time(), action))

    def return_home(self) -> None:
        self.goto(0.0, 0.0, self.alt)

    def at_home(self) -> bool:
        return math.hypot(self.n, self.e) <= 2.0

    def land(self) -> None:
        self._set_mode("LAND")

    def landed(self) -> bool:
        return self.alt <= 0.3

    def battery_pct(self) -> float:
        return self.batt
