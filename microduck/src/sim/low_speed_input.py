"""Slew-limited keyboard commands for the local low-speed simulation profile."""
from constants import VX_MAX, VX_MAX_BACKWARD, VY_MAX, VTHETA_MAX_MOVING, VTHETA_MAX_STATIONARY
from sim.mujoco_input import MuJoCoInputSource
import math


class LowSpeedKeyboardInput(MuJoCoInputSource):
    FORWARD_LIMIT = 0.05
    BACKWARD_LIMIT = 0.025
    TURN_LIMIT = 0.10
    STARTUP_STAND_S = 2.0
    LINEAR_SLEW = 0.025
    ANGULAR_SLEW = 0.05

    HEADING_GAIN = 1.0
    HEADING_CORRECTION_LIMIT = 0.30

    def __init__(self, *, time_source, stop_flag_path, initial_moves=None, forward_limit=0.05,
                 heading_source=None):
        if forward_limit not in (0.05, 0.10):
            raise ValueError("Supported forward limits are .05 and .10 m/s")
        super().__init__({"v": "walk"}, stop_flag_path=stop_flag_path, initial_moves=initial_moves)
        self.FORWARD_LIMIT = forward_limit
        self._time_source = time_source
        self._physical_target = {"vx": 0., "vy": 0., "vtheta": 0.}
        self._physical_current = dict(self._physical_target)
        self._last_time = None
        self._heading_source = heading_source
        self._heading_reference = None
        self.heading_correction = 0.

    def start(self):
        print("LOW-SPEED KEYBOARD INPUT: slew-limited simulation controls; not hardware validation")
        print(f"  [up/down] change forward/back target by .05 m/s (initial limits +{self.FORWARD_LIMIT:.2f}/-.025)")
        print("  [left/right] change turn target by .05 rad/s (limit +/- .10)")
        print("  [8] .05 m/s forward limit, [9] .10 m/s forward limit; targets change gradually")
        if self._heading_source is not None:
            print("  Fast forward uses heading assistance; X/idle cancels automatic heading correction")
        print("  [x] decelerate to standing, [r] reset + clear motion, [q] quit")
        print("  Commands latch after release and ramp gradually; startup stands for 2 seconds")

    def clear_velocity(self):
        with self._lock:
            self._state.velocity = {"vx": 0., "vy": 0., "vtheta": 0.}
            self._physical_target = dict(self._state.velocity)
            self._physical_current = dict(self._state.velocity)
            self._last_time = None
            self._heading_reference = None
            self.heading_correction = 0.

    def key_callback(self, keycode):
        # MuJoCo uses digits 0..5 for geometry visibility; 2 hides this robot's meshes.
        if keycode in (ord("8"), ord("9")):
            limit = 0.05 if keycode == ord("8") else 0.10
            with self._lock:
                self.FORWARD_LIMIT = limit
                if self._physical_target["vx"] > 0.:
                    self._physical_target["vx"] = limit
            print(f"Forward speed limit: {limit:.2f} m/s (ramped)")
            return
        if keycode in (262, 263, 264, 265):
            axis = "vx" if keycode in (264, 265) else "vtheta"
            delta = 0.05 * (1 if keycode in (262, 265) else -1)
            lower, upper = ((-self.BACKWARD_LIMIT, self.FORWARD_LIMIT) if axis == "vx"
                            else (-self.TURN_LIMIT, self.TURN_LIMIT))
            with self._lock:
                self._physical_target[axis] = max(lower, min(upper, self._physical_target[axis] + delta))
                target = self._physical_target[axis]
            print(f"target {axis}={target:+.3f} {'m/s' if axis == 'vx' else 'rad/s'} (ramped)")
            return
        if keycode == ord("X"):
            with self._lock:
                self._physical_target = {"vx": 0., "vy": 0., "vtheta": 0.}
            print("Decelerating to standing")
            return
        super().key_callback(keycode)

    def read(self):
        state = super().read()
        now = self._time_source()
        with self._lock:
            dt = 0. if self._last_time is None else max(0., min(.1, now - self._last_time))
            self._last_time = now
            ready = now >= self.STARTUP_STAND_S and "walk" in state.active_moves
            for axis in self._physical_current:
                target = self._physical_target[axis] if ready else 0.
                rate = self.ANGULAR_SLEW if axis == "vtheta" else self.LINEAR_SLEW
                current = self._physical_current[axis]
                step = rate * dt
                self._physical_current[axis] = current + max(-step, min(step, target - current))
            physical = dict(self._physical_current)
            assist_active = (self._heading_source is not None and ready
                             and self.FORWARD_LIMIT > .05
                             and self._physical_target["vx"] > 0. and physical["vx"] > 1e-6)
            if assist_active:
                heading = self._heading_source()
                if not math.isfinite(heading):
                    raise ValueError("Simulated heading must be finite")
                if self._heading_reference is None:
                    self._heading_reference = heading
                self._heading_reference += physical["vtheta"] * dt
                difference = self._heading_reference - heading
                error = math.atan2(math.sin(difference), math.cos(difference))
                self.heading_correction = max(-self.HEADING_CORRECTION_LIMIT,
                                              min(self.HEADING_CORRECTION_LIMIT, self.HEADING_GAIN * error))
                physical["vtheta"] = max(-self.HEADING_CORRECTION_LIMIT,
                                          min(self.HEADING_CORRECTION_LIMIT,
                                              physical["vtheta"] + self.heading_correction))
            else:
                # X must take precedence over chasing an old heading: the
                # standing policy cannot reliably turn on the spot.
                self._heading_reference = None
                self.heading_correction = 0.
        vx_limit = VX_MAX if physical["vx"] >= 0 else VX_MAX_BACKWARD
        normalized = {"vx": physical["vx"] / vx_limit, "vy": physical["vy"] / VY_MAX}
        # Invert the same moving/stationary choice as scale_velocity, so adding
        # translation cannot suddenly halve an existing physical turn command.
        moving = abs(normalized["vx"]) > 1e-6 or abs(normalized["vy"]) > 1e-6
        normalized["vtheta"] = physical["vtheta"] / (VTHETA_MAX_MOVING if moving else VTHETA_MAX_STATIONARY)
        state.velocity = normalized
        return state
