"""Scheduler and simulator regression checks; fake controllers are unit fixtures only."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scheduler import Scheduler
from observer import RobotState
from moves.move import Move, MoveState
from input.input_source import InputSource, UserInput
from sim.mujoco_controller import MuJoCoController, _DelayBuffer
from constants import MOTOR_TO_ID, NEUTRAL_POSE
from constants import IMU_MOUNT_QUAT
from imu_reader import quat_apply_inverse
from sim.sim_main import FixedSimulationInput


class EmptyMove(Move):
    def step(self, obs, command):
        command.target_angles.clear()


class FixedInput(InputSource):
    def read(self):
        return UserInput(active_moves={"empty"})


class TickController:
    def __init__(self):
        self.ticks = 0
        self.disabled = False
    def end_tick(self):
        self.ticks += 1
    def sync_write_goal_position(self, ids, targets):
        pass
    def sync_write_torque_enable(self, ids, values):
        self.disabled = not any(values)


class ResetMove(EmptyMove):
    def __init__(self):
        super().__init__()
        self.history = [9.0]
    def on_stop(self, obs, command):
        self.history.clear()
        super().on_stop(obs, command)


class SimRuntimeTests(unittest.TestCase):
    def test_real_imu_gyro_and_up_vector_match_training_mount_on_all_axes(self):
        import mujoco
        import numpy as np
        scene = Path(__file__).resolve().parents[1] / "src/model/mjcf/scene.xml"
        controller = MuJoCoController(str(scene), headless=True)
        mount = np.array(IMU_MOUNT_QUAT)
        mount /= np.linalg.norm(mount)
        angular_velocity_body = np.array([0.2, 0.4, -0.3])
        try:
            for axis in np.eye(3):
                body_quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(body_quat, axis, 0.25)
                controller._data.qpos[3:7] = body_quat
                controller._data.qvel[3:6] = angular_velocity_body
                mujoco.mj_forward(controller._model, controller._data)
                expected_gyro = np.zeros(3)
                mujoco.mju_rotVecQuat(expected_gyro, angular_velocity_body, mount)
                np.testing.assert_allclose(controller.read_gyro(), expected_gyro, atol=1e-7)
                expected_up = np.zeros(3)
                up_body = np.array(quat_apply_inverse(body_quat, [0., 0., 1.]))
                mujoco.mju_rotVecQuat(expected_up, up_body, mount)
                actual_up = quat_apply_inverse(controller.read_quat(0.02), [0., 0., 1.])
                np.testing.assert_allclose(actual_up, expected_up, atol=1e-7)
        finally:
            controller.shutdown()

    def test_command_ramp_uses_simulation_time_and_never_changes_target(self):
        now = [0.0]
        source = FixedSimulationInput(True, [0.2, -0.1, 0.3], delay_s=2., ramp_s=2.,
                                      time_source=lambda: now[0])
        self.assertEqual(source.read().velocity, {"vx": 0., "vy": 0., "vtheta": 0.})
        now[0] = 3.
        self.assertEqual(source.read().velocity, {"vx": 0.1, "vy": -0.05, "vtheta": 0.15})
        now[0] = 8.
        self.assertEqual(source.read().velocity, {"vx": 0.2, "vy": -0.1, "vtheta": 0.3})

    def test_bam_torque_is_applied_directly_without_xml_position_feedback(self):
        import mujoco
        import numpy as np
        scene = Path(__file__).resolve().parents[1] / "src/model/mjcf/scene.xml"
        controller = MuJoCoController(str(scene), headless=True)
        try:
            controller._bam.update()
            indices = controller._bam.act_indexes
            np.testing.assert_allclose(controller._data.ctrl[indices], 0., atol=1e-10)
            mujoco.mj_forward(controller._model, controller._data)
            np.testing.assert_allclose(controller._data.actuator_force[indices], 0., atol=1e-10)
            controller._data.ctrl[indices] = 0.10
            mujoco.mj_forward(controller._model, controller._data)
            np.testing.assert_allclose(controller._data.actuator_force[indices], 0.10, atol=1e-10)
            controller._data.ctrl[indices] = 1e3
            mujoco.mj_forward(controller._model, controller._data)
            limit = controller._bam.model.kt.value * 1.75
            np.testing.assert_allclose(controller._data.actuator_force[indices], limit, atol=1e-10)
            np.testing.assert_allclose(controller.sync_read_present_current(list(MOTOR_TO_ID.values())),
                                       1.75, atol=1e-10)
        finally:
            controller.shutdown()

    def test_floating_root_motor_friction_is_matched_by_dof_not_joint_id(self):
        import mujoco
        import numpy as np
        controller = MuJoCoController.__new__(MuJoCoController)
        controller._bam = SimpleNamespace(dof_indexes=np.array([6, 7]), joint_indexes=[1, 2])
        controller._data = SimpleNamespace(
            qfrc_bias=np.zeros(8), qfrc_constraint=np.array([0., 0., 0., 0., 0., 0., 5., 6.]),
            efc_id=np.array([6, 7, 3]), efc_force=np.array([1.5, 2.5, 9.]),
            efc_type=np.array([mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                               mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                               mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL]))
        np.testing.assert_allclose(controller._external_torque_without_motor_friction(), [3.5, 3.5])

    def test_explicit_upstream_height_and_training_timestep_are_honored(self):
        scene = Path(__file__).resolve().parents[1] / "src/model/mjcf/scene.xml"
        controller = MuJoCoController(str(scene), headless=True, scheduler_driven=True,
                                      initial_height_m=0.12, physics_timestep_s=0.005)
        try:
            self.assertAlmostEqual(controller._data.qpos[2], 0.12)
            self.assertEqual(controller._steps_per_tick, 4)
            controller.end_tick()
            self.assertAlmostEqual(controller.simulation_time, 0.02)
            controller.reset()
            self.assertAlmostEqual(controller._data.qpos[2], 0.12)
        finally:
            controller.shutdown()

    def test_real_model_initial_pose_survives_bam_setup_and_idle_tick(self):
        scene = Path(__file__).resolve().parents[1] / "src/model/mjcf/scene.xml"
        with tempfile.TemporaryDirectory() as directory:
            controller = MuJoCoController(str(scene), headless=True, scheduler_driven=True,
                                          stop_flag_path=str(Path(directory) / "stop"))
            try:
                initial = controller.sync_read_present_position(list(MOTOR_TO_ID.values()))
                for name, angle in zip(MOTOR_TO_ID, initial):
                    self.assertAlmostEqual(angle, NEUTRAL_POSE[name])
                self.assertAlmostEqual(controller._data.qpos[2], 0.165)
                self.assertTrue(all(v == 0 for v in controller._model.dof_damping[controller._bam.dof_indexes]))
                self.assertTrue(all(v == 0 for v in controller._model.dof_frictionloss[controller._bam.dof_indexes]))
                self.assertAlmostEqual(controller._bam.vin_drop_resistance,
                                       0.2 * controller._bam.model.kt.value)
                controller.end_tick()
                self.assertAlmostEqual(controller.simulation_time, 0.02)
                controller.sync_write_goal_position(list(MOTOR_TO_ID.values()), [2.0] * len(MOTOR_TO_ID))
                for _ in range(5):
                    controller.end_tick()
                    currents = controller.sync_read_present_current(list(MOTOR_TO_ID.values()))
                    self.assertLessEqual(max(abs(v) for v in currents), 1.75 + 1e-8)
                controller.reset()
                self.assertAlmostEqual(controller.simulation_time, 0.)
                self.assertAlmostEqual(controller._data.qpos[3], 1.)
            finally:
                controller.shutdown()

    def test_controller_reset_clears_scheduler_move_and_safety_history(self):
        controller = TickController()
        resets = iter([True, False])
        controller.begin_tick = lambda: next(resets)
        move = ResetMove()
        move.state = MoveState.ACTIVE
        with tempfile.TemporaryDirectory() as directory:
            scheduler = Scheduler(controller=controller, input_source=FixedInput(),
                                  moves={"empty": move},
                                  stop_flag_path=str(Path(directory) / "stop"))
            scheduler.observer.read_state = lambda dt: RobotState()
            scheduler._cmd_history.append({"invalid_old_target": 9.})
            scheduler._overcurrent_ticks = 1
            first_history = []
            scheduler.run(max_ticks=2, realtime=False,
                          observation_callback=lambda obs: first_history.append(list(scheduler._cmd_history)))
        self.assertEqual(move.history, [])
        self.assertEqual(first_history[0], [])
        self.assertEqual(controller.ticks, 2)

    def test_bounded_scheduler_advances_even_with_empty_commands(self):
        controller = TickController()
        move = EmptyMove()
        move.state = MoveState.ACTIVE
        with tempfile.TemporaryDirectory() as directory:
            scheduler = Scheduler(controller=controller, input_source=FixedInput(),
                                  moves={"empty": move},
                                  stop_flag_path=str(Path(directory) / "stop"))
            scheduler.observer.read_state = lambda dt: RobotState()
            records = []
            scheduler.run(max_ticks=3, realtime=False,
                          tick_callback=lambda tick, obs, cmd: records.append(tick))
        self.assertEqual(controller.ticks, 3)
        self.assertEqual(records, [0, 1, 2])
        self.assertTrue(controller.disabled)

    def test_period_substeps_match_hz_and_reject_fractional_steps(self):
        self.assertEqual(MuJoCoController.substeps_for_period(50, 0.005), 4)
        self.assertEqual(MuJoCoController.substeps_for_period(100, 0.005), 2)
        with self.assertRaises(ValueError):
            MuJoCoController.substeps_for_period(60, 0.005)
        with self.assertRaises(ValueError):
            MuJoCoController.substeps_for_period(0, 0.005)

    def test_reset_restores_free_quaternion_and_all_delays(self):
        import numpy as np
        controller = MuJoCoController.__new__(MuJoCoController)
        controller._model = SimpleNamespace(qpos0=np.array([0., 0., 0., 1., 0., 0., 0.]),
                                            dof_frictionloss=np.ones(6), dof_damping=np.ones(6))
        controller._initial_height_m = 0.165
        controller._data = SimpleNamespace(qpos=np.ones(7), qvel=np.ones(6), ctrl=np.ones(1), time=5.)
        controller._name_to_qpos_idx = {}
        controller._delay_act = {}
        controller._delay_pos = {}
        controller._delay_vel = {}
        controller._delay_gyro = _DelayBuffer((1., 2., 3.), 3)
        controller._delay_quat = _DelayBuffer((0., 1., 0., 0.), 4)
        controller._sensor_orientation = -1
        controller._sensor_gyro = -1
        controller._bam = SimpleNamespace(model=SimpleNamespace(reset=lambda: None), qpos_indexes=[], dof_indexes=[])
        controller._viewer = None
        controller._targets = {}
        with patch("sim.mujoco_controller.mujoco.mj_resetData") as reset, \
             patch("sim.mujoco_controller.mujoco.mj_forward"):
            def reset_data(model, data):
                data.qpos[:] = model.qpos0
                data.qvel[:] = 0
                data.ctrl[:] = 0
                data.time = 0
            reset.side_effect = reset_data
            controller.reset()
        self.assertEqual(controller._data.qpos[3], 1.)
        self.assertEqual(controller._data.time, 0.)
        self.assertEqual(controller._delay_gyro.push_and_read((0., 0., 0.)), (0., 0., 0.))
        self.assertEqual(controller._delay_quat.push_and_read((1., 0., 0., 0.)), (1., 0., 0., 0.))


if __name__ == "__main__":
    unittest.main()
