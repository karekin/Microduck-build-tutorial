"""Real keyboard-callback physics probes; outputs are LOCAL_SIM evidence only.

Run directly with --scenario A1, A10, B, or B-conservative. This does not
instantiate a hardware controller or modify runtime/configuration source.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imu_reader import quat_apply_inverse
from input.input_source import InputSource
from moves.walk import WalkMove
from scheduler import Scheduler
from sim.low_speed_input import LowSpeedKeyboardInput
from sim.mujoco_controller import MuJoCoController

UP, DOWN, RIGHT, LEFT, X, R = 265, 264, 262, 263, ord("X"), ord("R")
EVENTS_B = [(100, UP, 1), (600, X, 1), (900, DOWN, 1), (1400, X, 1),
            (1700, RIGHT, 2), (2200, UP, 1), (2700, X, 1),
            (3000, LEFT, 2), (3500, R, 1)]


class EventKeyboardInput(InputSource):
    """Replay actual keyboard callbacks by global tick, even after an R reset."""
    def __init__(self, keyboard, events, *, mixed_forward_cap=None):
        self.keyboard = keyboard
        self.events = {tick: (key, count) for tick, key, count in events}
        self.tick = 0
        self.last_events = []
        self.mixed_forward_cap = mixed_forward_cap
        self.original_forward_cap = keyboard.FORWARD_LIMIT

    def start(self):
        self.keyboard.start()

    def stop(self):
        self.keyboard.stop()

    @property
    def show_torque(self):
        return self.keyboard.show_torque

    def consume_reset(self):
        return self.keyboard.consume_reset()

    def clear_velocity(self):
        self.keyboard.clear_velocity()

    def read(self):
        self.last_events = []
        event = self.events.get(self.tick)
        if event:
            key, count = event
            # The single explicitly authorized conservative comparison alters
            # only the mixed forward cap, not policy/gains/protection settings.
            if self.mixed_forward_cap is not None and key == UP:
                turning = abs(self.keyboard._physical_target["vtheta"]) > 1e-9
                self.keyboard.FORWARD_LIMIT = self.mixed_forward_cap if turning else self.original_forward_cap
            for _ in range(count):
                self.keyboard.key_callback(key)
            self.last_events = [{"keycode": key, "count": count,
                                 "global_tick": self.tick,
                                 "global_time_s": self.tick / 50.}]
        self.tick += 1
        return self.keyboard.read()


def run_scenario(name, folder):
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"keyboard_callback_{name}.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Preserve existing evidence; use a fresh --output-dir: {log_path}")
    stop_path = folder / f"keyboard_callback_{name}.stop"
    events, total_ticks = ([(0, UP, 1 if name == "A1" else 10)], 1500) if name in ("A1", "A10") else (EVENTS_B, 4000)
    controller = MuJoCoController(str(SRC / "model/mjcf/scene.xml"), headless=True,
                                  scheduler_driven=True, frequency_hz=50.,
                                  initial_height_m=.12, physics_timestep_s=.005,
                                  policy_kp=200, stop_flag_path=str(stop_path),
                                  delay_act_steps=0, delay_pos_ticks=0, delay_vel_ticks=0,
                                  delay_gyro_ticks=0, delay_quat_ticks=0)
    keyboard = LowSpeedKeyboardInput(time_source=lambda: controller.simulation_time,
                                    stop_flag_path=str(stop_path), initial_moves={"walk"})
    source = EventKeyboardInput(keyboard, events,
                               mixed_forward_cap=.025 if name == "B-conservative" else None)
    controller._reset_source = source
    walk = WalkMove(controller=controller)
    scheduler = Scheduler(controller=controller, input_source=source,
                          moves={"walk": walk}, stop_flag_path=str(stop_path))
    scheduler.observer.observe_current = True
    records, candidate = [], []
    epoch = 0
    last_sim_time = -1.
    with log_path.open("w", encoding="utf-8") as log:
        header = {"type": "header", "scenario": name, "evidence": "ACTUAL_KEYBOARD_CALLBACK_LOCAL_SIM",
                  "hardware_verified": False, "normal_startup": True,
                  "parameters": {"hz": 50., "policy_kp": 200, "height_m": .12,
                                 "physics_timestep_s": .005, "all_sensor_actuation_delays": 0,
                                 "mixed_forward_cap": source.mixed_forward_cap},
                  "events": events, "requested_ticks": total_ticks,
                  "input_source_sha256": hashlib.sha256((SRC / "sim/low_speed_input.py").read_bytes()).hexdigest(),
                  "policy_sha256": hashlib.sha256((SRC / "agents/walk.onnx").read_bytes()).hexdigest()}
        log.write(json.dumps(header) + "\n")

        def capture_observation(obs):
            nonlocal candidate
            candidate = walk.build_observation(obs)

        def capture_tick(tick, obs, command):
            nonlocal epoch, last_sim_time
            sim_time = controller.simulation_time
            if sim_time < last_sim_time:
                epoch += 1
            last_sim_time = sim_time
            gravity = quat_apply_inverse(obs.robot_state.body_quat, [0., 0., -1.])
            tilt = math.degrees(math.acos(max(-1., min(1., -gravity[2]))))
            height = float(controller._data.qpos[2])
            body_v = quat_apply_inverse(obs.robot_state.body_quat, controller._data.qvel[:3].tolist())
            row = {"type": "tick", "tick": tick, "global_time_s": (tick + 1) / 50.,
                   "simulation_time_s": sim_time, "reset_epoch": epoch,
                   "events": source.last_events, "move_state": walk.state.name,
                   "physical_target": dict(keyboard._physical_target),
                   "physical_input_current": dict(keyboard._physical_current),
                   "actual_scheduler_command": dict(obs.user_input.velocity),
                   "firmware_kp": controller._bam.model.actuator.kp,
                   "root_height_m": height, "body_tilt_deg": tilt,
                   "fallen": height < .1 or tilt > 60.,
                   "qpos": controller._data.qpos.tolist(), "qvel": controller._data.qvel.tolist(),
                   "body_velocity_m_s": body_v,
                   "body_quat": obs.robot_state.body_quat, "projected_up": obs.robot_state.projected_gravity,
                   "gyro": obs.robot_state.gyro, "policy_observation_candidate": candidate,
                   "last_action": list(walk._last_action),
                   "targets": {key: float(value) for key, value in command.target_angles.items()},
                   "applied_torque_nm": controller._data.actuator_force.tolist(),
                   "simulated_currents_a": obs.robot_state.motor_currents}
            records.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")

        completed = scheduler.run(max_ticks=total_ticks, realtime=False,
                                  time_source=lambda: controller.simulation_time,
                                  observation_callback=capture_observation, tick_callback=capture_tick)
        fallen = [r for r in records if r["fallen"]]
        motion_stages = []
        boundaries = sorted(set([0, *[tick for tick, _, _ in events], total_ticks]))
        for lo, hi in zip(boundaries, boundaries[1:]):
            samples = [r for r in records if lo + 150 <= r["tick"] < hi]
            if not samples:
                continue
            stage = {"from_global_s": lo / 50., "to_global_s": hi / 50.,
                     "command_at_end": samples[-1]["actual_scheduler_command"],
                     "mean_body_vx_m_s": sum(r["body_velocity_m_s"][0] for r in samples) / len(samples),
                     "mean_body_vy_m_s": sum(r["body_velocity_m_s"][1] for r in samples) / len(samples),
                     "mean_body_yaw_rate_rad_s": sum(r["qvel"][5] for r in samples) / len(samples),
                     "fallen_ticks": sum(r["fallen"] for r in samples)}
            motion_stages.append(stage)
        summary = {"type": "summary", "scenario": name, "evidence": header["evidence"],
                   "hardware_verified": False, "completed_ticks": completed,
                   "requested_ticks_completed": completed == total_ticks,
                   "global_elapsed_s": completed / 50., "simulation_time_s": controller.simulation_time,
                   "reset_epochs": epoch, "minimum_height_m": min(r["root_height_m"] for r in records),
                   "max_tilt_deg": max(r["body_tilt_deg"] for r in records),
                   "fallen_ticks": len(fallen),
                   "first_fall": {k: fallen[0][k] for k in ["tick", "global_time_s", "simulation_time_s",
                                    "reset_epoch", "root_height_m", "body_tilt_deg", "actual_scheduler_command"]} if fallen else None,
                   "actual_kp_values": sorted(set(r["firmware_kp"] for r in records)),
                   "event_readback": [r for r in records if r["events"]],
                   "motion_stages": motion_stages}
        log.write(json.dumps(summary, allow_nan=False) + "\n")
    controller.shutdown()
    report = dict(summary)
    report["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    (folder / f"keyboard_callback_{name}_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "event_readback"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=("A1", "A10", "B", "B-conservative"))
    parser.add_argument("--output-dir", type=Path,
                        default=SRC.parent / ".omx/artifacts/keyboard-sim-20261010")
    options = parser.parse_args()
    print(json.dumps(run_scenario(options.scenario, options.output_dir), allow_nan=False))
