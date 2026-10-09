# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud
"""Local CPU physics + deployment-policy simulation; never connects to hardware."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from constants import KP_RL
from moves.move import MoveState
from input.input_source import InputSource, UserInput
from scheduler import Scheduler
from sim.mujoco_input import MuJoCoInputSource
from sim.mujoco_controller import MuJoCoController
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove


class FixedSimulationInput(InputSource):
    def __init__(self, walk: bool, velocity: list[float], *, delay_s: float = 0.,
                 ramp_s: float = 0., time_source=None):
        self.walk = walk
        self.velocity = dict(zip(("vx", "vy", "vtheta"), velocity))
        self.delay_s = delay_s
        self.ramp_s = ramp_s
        self.time_source = time_source or (lambda: 0.)

    def read(self) -> UserInput:
        elapsed = self.time_source() - self.delay_s
        progress = (0. if elapsed < 0 else
                    min(elapsed / self.ramp_s, 1.) if self.ramp_s > 0 else 1.)
        return UserInput(active_moves={"walk"} if self.walk else set(),
                         velocity={name: value * progress for name, value in self.velocity.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--physics-timestep", type=float,
                        help="Explicit physics timestep in seconds; omitted uses MJCF, current training config uses .005")
    parser.add_argument("--headless", action="store_true", help="No viewer; use finite --ticks")
    parser.add_argument("--ticks", type=int, help="Stop after this many control ticks")
    parser.add_argument("--realtime", action="store_true", help="Pace headless ticks to wall time")
    parser.add_argument("--walk", action="store_true", help="Activate walk from tick zero in headless mode")
    parser.add_argument("--policy-immediately", action="store_true",
                        help="Headless walk only: begin policy immediately at its default pose, bypassing hardware startup ramp")
    parser.add_argument("--velocity", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("VX", "VY", "THETA"), help="Normalized commands in [-1,1], centrally scaled")
    parser.add_argument("--command-delay", type=float, default=0., help="Headless: stand for this many simulation seconds before commanding motion")
    parser.add_argument("--command-ramp", type=float, default=0., help="Headless: ramp velocity linearly over this many simulation seconds")
    parser.add_argument("--policy-kp", type=int, choices=(KP_RL, 200), default=KP_RL,
                        help="Simulation walking firmware gain: deployment default or sourced upstream training value 200")
    parser.add_argument("--log", type=Path, help="Write JSONL ticks and final summary (simulation evidence only)")
    parser.add_argument("--delay-act", type=int, default=2, help="Actuation delay in physics substeps")
    parser.add_argument("--delay-pos", type=int, default=0)
    parser.add_argument("--delay-vel", type=int, default=1)
    parser.add_argument("--delay-gyro", type=int, default=3)
    parser.add_argument("--delay-quat", type=int, default=4)
    parser.add_argument("--trunk-com-offset", type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument("--initial-height", type=float, default=0.165,
                        help="Explicit initial root height in meters; preserved tutorial default=.165, upstream STAND=.12")
    args = parser.parse_args()
    if not math.isfinite(args.hz) or args.hz <= 0:
        parser.error("--hz must be finite and positive")
    if args.physics_timestep is not None and (not math.isfinite(args.physics_timestep) or args.physics_timestep <= 0):
        parser.error("--physics-timestep must be finite and positive")
    if args.ticks is not None and args.ticks <= 0:
        parser.error("--ticks must be positive")
    if args.headless and args.ticks is None:
        parser.error("--headless requires a finite --ticks")
    if args.policy_immediately and not (args.headless and args.walk):
        parser.error("--policy-immediately requires --headless --walk")
    if any(v < 0 for v in (args.delay_act, args.delay_pos, args.delay_vel, args.delay_gyro, args.delay_quat)):
        parser.error("delays must be nonnegative")
    if any(not math.isfinite(v) or abs(v) > 1 for v in args.velocity):
        parser.error("--velocity values must be finite normalized commands in [-1,1]")
    if any(not math.isfinite(v) or v < 0 for v in (args.command_delay, args.command_ramp)):
        parser.error("command delay and ramp must be finite and nonnegative")
    if any(not math.isfinite(v) for v in args.trunk_com_offset):
        parser.error("--trunk-com-offset must be finite")
    if not math.isfinite(args.initial_height) or args.initial_height <= 0:
        parser.error("--initial-height must be finite and positive")
    if args.log and not args.log.is_absolute():
        args.log = PROJECT_ROOT / args.log
    model_scene = SRC_ROOT / "model/mjcf/scene.xml"
    policy_file = SRC_ROOT / "agents/walk.onnx"

    # All stop files are private to this run, never the robot's public stop path.
    with tempfile.TemporaryDirectory(prefix="microduck-sim-") as directory:
        stop_path = str(Path(directory) / "stop")
        input_source = (FixedSimulationInput(args.walk, args.velocity, delay_s=args.command_delay,
                                            ramp_s=args.command_ramp,
                                            time_source=lambda: controller.simulation_time) if args.headless else
                        MuJoCoInputSource(move_keys={"h": "head", "s": "squat", "v": "walk"},
                                          stop_flag_path=stop_path))
        try:
            controller = MuJoCoController(
                mjcf_path=str(model_scene), headless=args.headless, policy_kp=args.policy_kp,
                initial_height_m=args.initial_height,
                physics_timestep_s=args.physics_timestep,
                frequency_hz=args.hz, scheduler_driven=True, stop_flag_path=stop_path,
                key_callback=None if args.headless else input_source.key_callback,
                reset_source=None if args.headless else input_source,
                delay_act_steps=args.delay_act, delay_pos_ticks=args.delay_pos,
                delay_vel_ticks=args.delay_vel, delay_gyro_ticks=args.delay_gyro,
                delay_quat_ticks=args.delay_quat, trunk_com_offset=tuple(args.trunk_com_offset))
        except RuntimeError as exc:
            if sys.platform == "darwin" and not args.headless and "mjpython" in str(exc):
                parser.exit(2, "macOS viewer requires: PYTHONPATH=src .venv/bin/mjpython src/sim/sim_main.py\n")
            raise
        if not args.headless:
            input_source.set_viewer_opt(controller.viewer_opt)

        log_file = None
        try:
            walk = WalkMove(controller=controller)
            if args.policy_immediately:
                controller.reset(pose=walk._default_pose)
                controller.set_kp(args.policy_kp)
                walk.state = MoveState.ACTIVE
            scheduler = Scheduler(frequency_hz=args.hz, controller=controller,
                                  input_source=input_source, stop_flag_path=stop_path,
                                  moves={"head": RotateHeadMove(), "squat": SquatMove(), "walk": walk})
            # CPU sensor reads have no bus latency. Protect using the simulated
            # BAM current (applied torque / kt), rather than a hardware proxy.
            scheduler.observer.observe_current = True
            if args.log:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                log_file = args.log.open("w", encoding="utf-8")
            policy_candidate = []
            minimum_height = float("inf")
            fallen_ticks = 0
            action_ticks = 0

            def before_dispatch(obs):
                nonlocal policy_candidate
                policy_candidate = walk.build_observation(obs)

            def record(tick, obs, command):
                nonlocal minimum_height, fallen_ticks, action_ticks
                height = float(controller._data.qpos[2])
                minimum_height = min(minimum_height, height)
                fallen = height < 0.10
                fallen_ticks += int(fallen)
                action_ticks += int(walk.state.name == "ACTIVE")
                row = {"type": "tick", "tick": tick, "observation_time_s": obs.robot_state.time_s,
                       "simulation_time_s": controller.simulation_time,
                       "move_state": walk.state.name,
                       "command_velocity": obs.user_input.velocity,
                       "firmware_kp": controller._bam.model.actuator.kp,
                       "motor_positions": obs.robot_state.motor_positions,
                       "motor_velocities": obs.robot_state.motor_velocities,
                       "simulated_motor_currents_a": obs.robot_state.motor_currents,
                       "imu_quat": obs.robot_state.quat,
                       "body_quat": obs.robot_state.body_quat,
                       "projected_up": obs.robot_state.projected_gravity,
                       "gyro": obs.robot_state.gyro,
                       "policy_observation_candidate": policy_candidate,
                       "last_action": list(walk._last_action),
                       "targets": {name: float(angle) for name, angle in command.target_angles.items()},
                       "qpos_after_step": controller._data.qpos.tolist(),
                       "qvel_after_step": controller._data.qvel.tolist(),
                       "bam_command_torques": controller._data.ctrl.tolist(),
                       "torques_after_step": controller._data.actuator_force.tolist(),
                       "root_height_m": height, "below_training_height_threshold": fallen}
                encoded = json.dumps(row, allow_nan=False)
                if log_file:
                    log_file.write(encoded + "\n")

            if log_file:
                log_file.write(json.dumps({"type": "header", "evidence": "LOCAL_SIM_POLICY_CLOSED_LOOP",
                    "hardware_verified": False, "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                    "policy_path": str(policy_file), "model_scene": str(model_scene), "hz": args.hz,
                    "command_delay_s": args.command_delay, "command_ramp_s": args.command_ramp,
                    "target_normalized_velocity": args.velocity, "policy_firmware_kp": args.policy_kp,
                    "physics_timestep_s": controller._model.opt.timestep,
                    "substeps_per_tick": controller._steps_per_tick,
                    "requested_ticks": args.ticks,
                    "policy_immediately": args.policy_immediately,
                    "initial_root_height_m": args.initial_height,
                    "bam_voltage_drop_resistance_ohm": controller._bam.vin_drop_resistance,
                    "bam_max_current_a": controller._bam.model.actuator.max_current,
                    "bam_firmware_kp_at_start": controller._bam.model.actuator.kp,
                    "bam_cpu_friction_matches_constraint_dof_ids": True,
                    "bam_torque_actuators_gain_one_bias_none": True,
                    "overcurrent_source": "simulated_BAM_torque_divided_by_kt",
                    "delays": {"act": args.delay_act, "pos": args.delay_pos, "vel": args.delay_vel,
                               "gyro": args.delay_gyro, "quat": args.delay_quat},
                    "simulation_voltage_is_placeholder": True}) + "\n")
            ticks = scheduler.run(max_ticks=args.ticks,
                                  realtime=not args.headless or args.realtime,
                                  observation_callback=before_dispatch, tick_callback=record,
                                  time_source=lambda: controller.simulation_time)
            summary = {"type": "summary", "evidence": "LOCAL_SIM_POLICY_CLOSED_LOOP",
                       "hardware_verified": False, "ticks": ticks,
                       "requested_ticks_completed": args.ticks is None or ticks == args.ticks,
                       "simulation_time_s": controller.simulation_time,
                       "minimum_root_height_m": minimum_height if ticks else None,
                       "below_training_height_threshold_ticks": fallen_ticks,
                       "active_move_ticks": action_ticks}
            if log_file:
                log_file.write(json.dumps(summary, allow_nan=False) + "\n")
            print(json.dumps(summary, allow_nan=False))
            if args.ticks is not None and ticks != args.ticks:
                raise SystemExit(2)
        finally:
            if log_file:
                log_file.close()
            controller.shutdown()


if __name__ == "__main__":
    main()
