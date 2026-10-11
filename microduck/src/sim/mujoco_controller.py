# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
import math
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer

if TYPE_CHECKING:
    from sim.mujoco_input import MuJoCoInputSource

from bam.model import load_model as bam_load_model
from bam.mujoco import MujocoController as BamController

from constants import MOTOR_TO_ID, ID_TO_MOTOR, NEUTRAL_POSE, KP_DEFAULT, KP_RL, BAM_VIN, BAM_VOLTAGE_DROP_GAIN, BAM_VIN_MIN, BAM_MAX_CURRENT


class _DelayBuffer:
    """Returns values delayed by n_steps ticks (0 = no delay)."""

    def __init__(self, initial, n_steps: int) -> None:
        size = max(1, n_steps + 1)
        self._buf: deque = deque([initial] * size, maxlen=size)

    def push_and_read(self, value):
        self._buf.appendleft(value)
        return self._buf[-1]

    def fill(self, value) -> None:
        for i in range(len(self._buf)):
            self._buf[i] = value


class MuJoCoController:
    """MuJoCo-backed controller."""

    def __init__(
        self,
        mjcf_path: str,
        key_callback: Callable[[int], None] | None = None,
        stop_flag_path: str = "/tmp/microduck_scheduler.stop",
        reset_source: "MuJoCoInputSource | None" = None,
        # Actuation delay (command → motor), in simulator steps (default timestep: 0.005 s)
        delay_act_steps: int = 0,
        # Sensor delays (motor/IMU → observation), in scheduler ticks (default: 0.02 s)
        delay_pos_ticks: int = 0,
        delay_vel_ticks: int = 0,
        delay_gyro_ticks: int = 0,
        delay_quat_ticks: int = 0,
        trunk_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        frequency_hz: float = 50.0,
        headless: bool = False,
        scheduler_driven: bool = False,
        initial_height_m: float = 0.165,
        physics_timestep_s: float | None = None,
        policy_kp: int = KP_RL,
        follow_body_camera: bool = False,
    ) -> None:
        self._stop_flag_path = Path(stop_flag_path)
        self._reset_source = reset_source
        self._scheduler_driven = scheduler_driven
        self._policy_kp = policy_kp
        self._follow_body_camera = follow_body_camera
        if not math.isfinite(initial_height_m) or initial_height_m <= 0:
            raise ValueError("Initial root height must be finite and positive")
        self._initial_height_m = initial_height_m
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        if physics_timestep_s is not None:
            if not math.isfinite(physics_timestep_s) or physics_timestep_s <= 0:
                raise ValueError("Physics timestep must be finite and positive")
            self._model.opt.timestep = physics_timestep_s
        self._data = mujoco.MjData(self._model)

        self._name_to_actuator_idx: dict[str, int] = {}
        self._name_to_qpos_idx: dict[str, int] = {}
        self._name_to_qvel_idx: dict[str, int] = {}
        for name in MOTOR_TO_ID:
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in MJCF model {mjcf_path!r}")
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MJCF model {mjcf_path!r}")
            self._name_to_actuator_idx[name] = actuator_id
            self._name_to_qpos_idx[name] = self._model.jnt_qposadr[joint_id]
            self._name_to_qvel_idx[name] = self._model.jnt_dofadr[joint_id]
        # Number of physics sub-steps per scheduler tick (scheduler runs at 50 Hz)
        self._steps_per_tick = self.substeps_for_period(frequency_hz, self._model.opt.timestep)
        self._torque_interval = 0.1
        self._last_torque_print = 0.0

        # Apply CoM offset on trunk body (simulates inertial model error)
        if any(v != 0.0 for v in trunk_com_offset):
            trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
            if trunk_id < 0:
                raise ValueError("CoM offset requires the matching trunk_base body")
            self._model.body_ipos[trunk_id, 0] += trunk_com_offset[0]
            self._model.body_ipos[trunk_id, 1] += trunk_com_offset[1]
            self._model.body_ipos[trunk_id, 2] += trunk_com_offset[2]

        # Set initial pose to neutral so the robot starts upright
        self._data.qpos[2] = self._initial_height_m
        for name, angle in NEUTRAL_POSE.items():
            if name in self._name_to_qpos_idx:
                self._data.qpos[self._name_to_qpos_idx[name]] = angle
        mujoco.mj_forward(self._model, self._data)

        # Delay buffers — simulate sensor/communication latency
        self._delay_pos = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_pos_ticks,
            )
            for mid in MOTOR_TO_ID.values()
        }
        self._delay_vel = {
            mid: _DelayBuffer(0.0, delay_vel_ticks)
            for mid in MOTOR_TO_ID.values()
        }
        self._delay_gyro = _DelayBuffer((0.0, 0.0, 0.0), delay_gyro_ticks)
        self._delay_quat = _DelayBuffer((1.0, 0.0, 0.0, 0.0), delay_quat_ticks)
        self._delay_act = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_act_steps,
            )
            for mid in MOTOR_TO_ID.values()
        }

        # BAM motor model — XL330 m6 (DC motor + Stribeck + load-dependent friction)
        bam_model = bam_load_model(motor_name="xl330", model="m6")
        bam_model.actuator.kp = KP_DEFAULT
        bam_model.actuator.vin = BAM_VIN
        bam_model.actuator.max_current = BAM_MAX_CURRENT
        # BAM produces torque, not position setpoints. Match training's
        # set_to_motor(): otherwise an XML position actuator applies its own
        # kp * (BAM torque - q), silently adding a second control law.
        torque_limit = bam_model.kt.value * BAM_MAX_CURRENT
        for index in self._name_to_actuator_idx.values():
            self._model.actuator_dyntype[index] = mujoco.mjtDyn.mjDYN_NONE
            self._model.actuator_gaintype[index] = mujoco.mjtGain.mjGAIN_FIXED
            self._model.actuator_biastype[index] = mujoco.mjtBias.mjBIAS_NONE
            self._model.actuator_dynprm[index, :] = 0.0
            self._model.actuator_gainprm[index, :] = 0.0
            self._model.actuator_gainprm[index, 0] = 1.0
            self._model.actuator_biasprm[index, :] = 0.0
            self._model.actuator_gear[index, :] = 0.0
            self._model.actuator_gear[index, 0] = 1.0
            self._model.actuator_ctrllimited[index] = True
            self._model.actuator_ctrlrange[index] = (-torque_limit, torque_limit)
            self._model.actuator_forcelimited[index] = True
            self._model.actuator_forcerange[index] = (-torque_limit, torque_limit)
        self._bam = BamController(
            bam_model,
            list(MOTOR_TO_ID.keys()),
            self._model,
            self._data,
            # Older BAM used Vdrop = gain * sum(|torque|). BAM 1.0 uses
            # Vdrop = resistance * sum(|torque|) / kt, so preserve the units.
            vin_drop_resistance=BAM_VOLTAGE_DROP_GAIN * bam_model.kt.value,
            vin_min=BAM_VIN_MIN,
        )
        self._reset_motor_model()

        # Native viewer Reset uses qpos0, which otherwise places this floating
        # robot's trunk at floor height rather than its configured standing pose.
        # Preserve hinge qpos0: it also defines each joint's geometric reference.
        self._model.qpos0[2] = self._initial_height_m

        self._viewer = None if headless else mujoco.viewer.launch_passive(
            self._model, self._data, key_callback=key_callback
        )
        if self._viewer is not None:
            with self._viewer.lock():
                self._viewer.cam.azimuth = 135
                self._viewer.cam.elevation = -20
                self._viewer.cam.distance = 0.6
                self._viewer.cam.lookat[:] = (0., 0., 0.14)

        # Sensor indices for IMU readout
        self._sensor_orientation = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self._sensor_gyro = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        if self._sensor_orientation < 0 or self._sensor_gyro < 0:
            raise ValueError("Simulation requires orientation and angular-velocity sensors")
        self._targets = {name: angle for name, angle in NEUTRAL_POSE.items()}
        # BAM calls mj_setConst while configuring armatures; MuJoCo restores
        # qpos0 there. Seed the actual starting pose only after that setup.
        self.reset()

    @staticmethod
    def substeps_for_period(frequency_hz: float, timestep: float) -> int:
        if not math.isfinite(frequency_hz) or frequency_hz <= 0 or timestep <= 0:
            raise ValueError("Simulation frequency and timestep must be positive")
        steps = 1.0 / frequency_hz / timestep
        rounded = round(steps)
        if rounded < 1 or not math.isclose(steps, rounded, abs_tol=1e-8):
            raise ValueError("Control period must be an integer multiple of the physics timestep")
        return rounded

    @property
    def simulation_time(self) -> float:
        return float(self._data.time)

    @property
    def viewer_opt(self) -> mujoco.MjvOption | None:
        return None if self._viewer is None else self._viewer.opt

    def set_kp(self, kp: float) -> None:
        self._bam.model.actuator.kp = kp

    def read_body_heading(self) -> float:
        """True simulated trunk yaw for optional command-level heading assistance."""
        w, x, y, z = self._data.qpos[3:7]
        return math.atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        kp = int(self._bam.model.actuator.kp)
        return [kp] * len(ids)

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        gain = gains[0]
        self._bam.model.actuator.kp = self._policy_kp if gain == KP_RL else gain

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        pass

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        pass

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if self._viewer is not None and not self._viewer.is_running():
            self._stop_flag_path.write_text("stop\n", encoding="ascii")
            return

        self._targets.update({ID_TO_MOTOR[mid]: pos for mid, pos in zip(ids, positions)})
        if not self._scheduler_driven:
            self._advance_physics()

    def begin_tick(self) -> bool:
        native_reset = self._data.time + 1e-9 < self._last_physics_time
        keyboard_reset = self._reset_source is not None and self._reset_source.consume_reset()
        if native_reset or keyboard_reset:
            if self._reset_source is not None:
                self._reset_source.clear_velocity()
            self.reset()
            return True
        return False

    def end_tick(self) -> None:
        if self._scheduler_driven:
            self._advance_physics()

    def _advance_physics(self) -> None:
        if self._viewer is not None and not self._viewer.is_running():
            self._stop_flag_path.write_text("stop\n", encoding="ascii")
            return

        for _ in range(self._steps_per_tick):
            for name, pos in self._targets.items():
                motor_id = MOTOR_TO_ID[name]
                delayed_pos = self._delay_act[motor_id].push_and_read(pos)
                self._bam.set_q_target(ID_TO_MOTOR[motor_id], delayed_pos)
            self._bam.update()
            # BAM 1.0.2 CPU matches FRICTION_DOF efc_id against joint IDs.
            # With a floating root these differ from DOF IDs. Recompute the
            # load-dependent budget with the actual per-DOF motor friction
            # removed, matching BAM's training-side external-load semantics.
            dofs = self._bam.dof_indexes
            frictionloss, damping = self._bam.model.compute_frictions(
                self._data.qfrc_actuator[dofs],
                self._external_torque_without_motor_friction(),
                self._data.qvel[dofs],
            )
            self._model.dof_frictionloss[dofs] = frictionloss
            self._model.dof_damping[dofs] = damping
            mujoco.mj_step(self._model, self._data)

        self._last_physics_time = float(self._data.time)

        if self._reset_source is not None and self._reset_source.show_torque:
            now = time.monotonic()
            if now - self._last_torque_print >= self._torque_interval:
                total = float(sum(abs(f) for f in self._data.actuator_force))
                print(f"Torque sum: {total:.3f} Nm")
                self._last_torque_print = now

        if self._viewer is not None:
            if self._follow_body_camera:
                with self._viewer.lock():
                    self._viewer.cam.lookat[:] = (self._data.qpos[0], self._data.qpos[1], 0.14)
            self._viewer.sync()

    def shutdown(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    def _external_torque_without_motor_friction(self):
        dofs = self._bam.dof_indexes
        selector = (self._data.efc_id[None, :] == dofs[:, None]) & (
            self._data.efc_type[None, :] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        )
        motor_friction = (self._data.efc_force[None, :] * selector).sum(axis=1)
        return -self._data.qfrc_bias[dofs] + self._data.qfrc_constraint[dofs] - motor_friction

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        return [
            self._delay_pos[mid].push_and_read(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_position(self, motor_id: int) -> float:
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_pos[motor_id].push_and_read(
            self._data.qpos[self._name_to_qpos_idx[name]]
        ))

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        return [
            self._delay_vel[mid].push_and_read(
                self._data.qvel[self._name_to_qvel_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_velocity(self, motor_id: int) -> float:
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_vel[motor_id].push_and_read(
            self._data.qvel[self._name_to_qvel_idx[name]]
        ))

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        # Motor current from the torque applied by the bam model: I = torque / kt.
        # Read the actually applied actuator torque, after MuJoCo force limits.
        kt = self._bam.model.kt.value
        return [
            float(self._data.actuator_force[self._name_to_actuator_idx[ID_TO_MOTOR[mid]]] / kt)
            for mid in ids
        ]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        return [80.0] * len(ids)

    def read_present_input_voltage(self, motor_id: int) -> float:
        return 80.0

    def read_acc(self) -> tuple[float, float, float]:
        """Return pseudo-accelerometer (ax, ay, az) in g from the 'orientation' sensor."""
        if self._sensor_orientation < 0:
            return 0.0, 0.0, -1.0
        adr = self._model.sensor_adr[self._sensor_orientation]
        w, x, y, z = self._data.sensordata[adr:adr + 4]
        # Gravity in world is (0, 0, -1) g; rotate into IMU frame using conjugate quat
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = w * w - x * x - y * y + z * z
        return float(gx), float(gy), float(-gz)

    def read_gyro(self) -> tuple[float, float, float]:
        if self._sensor_gyro < 0:
            current = (0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_gyro]
            gx, gy, gz = self._data.sensordata[adr:adr + 3]
            current = (float(gx), float(gy), float(gz))
        return self._delay_gyro.push_and_read(current)

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        if self._sensor_orientation < 0:
            current = (1.0, 0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_orientation]
            w, x, y, z = self._data.sensordata[adr:adr + 4]
            current = (float(w), float(x), float(y), float(z))
        return self._delay_quat.push_and_read(current)

    def reset(self, *, pose: dict[str, float] | None = None) -> None:
        """Reset the simulation to the initial neutral standing pose."""
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[2] = self._initial_height_m
        initial_pose = NEUTRAL_POSE if pose is None else pose
        for name, angle in initial_pose.items():
            if name in self._name_to_qpos_idx:
                self._data.qpos[self._name_to_qpos_idx[name]] = angle
        self._reset_motor_model()
        self._last_physics_time = float(self._data.time)
        mujoco.mj_forward(self._model, self._data)
        self._targets = dict(initial_pose)
        for name, index in self._name_to_qpos_idx.items():
            mid = MOTOR_TO_ID[name]
            neutral = self._data.qpos[index]
            self._delay_act[mid].fill(neutral)
            self._delay_pos[mid].fill(neutral)
            self._delay_vel[mid].fill(0.0)
        self._fill_sensor_delays()
        if self._viewer is not None:
            self._viewer.sync()

    def _fill_sensor_delays(self) -> None:
        quat = (1.0, 0.0, 0.0, 0.0)
        gyro = (0.0, 0.0, 0.0)
        if self._sensor_orientation >= 0:
            adr = self._model.sensor_adr[self._sensor_orientation]
            quat = tuple(float(v) for v in self._data.sensordata[adr:adr + 4])
        if self._sensor_gyro >= 0:
            adr = self._model.sensor_adr[self._sensor_gyro]
            gyro = tuple(float(v) for v in self._data.sensordata[adr:adr + 3])
        self._delay_quat.fill(quat)
        self._delay_gyro.fill(gyro)

    def _reset_motor_model(self) -> None:
        # Match BAM's training edit_spec initial state. CPU update() then
        # replaces these fields with its computed budget before each step.
        self._model.dof_frictionloss[self._bam.dof_indexes] = 0.0
        self._model.dof_damping[self._bam.dof_indexes] = 0.0
        self._bam.model.reset()
        self._bam.q_target = self._data.qpos[self._bam.qpos_indexes].copy()
        self._bam.last_ts = self._data.time
