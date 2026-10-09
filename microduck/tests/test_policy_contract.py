"""Exercise the shipped ONNX policy and observation convention without hardware."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import OBSERVATION_DOF_ORDER, IMU_MOUNT_QUAT
from imu_reader import quat_apply_inverse
from input.input_source import UserInput
from moves.move import MotorCommand
from moves.walk import WalkMove
from observer import Observation, RobotState


class PolicyContractTests(unittest.TestCase):
    def test_real_policy_loads_outside_project_and_executes_upright_observation(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                walk = WalkMove()
        finally:
            os.chdir(previous)
        self.assertEqual(walk._ort_session.get_inputs()[0].shape, [1, 51])
        self.assertEqual(walk._ort_session.get_outputs()[0].shape, [1, 14])
        # Sensor->body is the inverse of the configured body->sensor mount.
        imu_quat = [IMU_MOUNT_QUAT[0], -IMU_MOUNT_QUAT[1],
                    -IMU_MOUNT_QUAT[2], -IMU_MOUNT_QUAT[3]]
        projected_up = quat_apply_inverse(imu_quat, [0., 0., 1.])
        np.testing.assert_allclose(projected_up, [1., 0., 0.], atol=1e-6)
        state = RobotState(gyro=[0., 0., 0.], projected_gravity=projected_up,
                           body_quat=[1., 0., 0., 0.],
                           motor_positions=dict(walk._default_pose),
                           motor_velocities={n: 0. for n in OBSERVATION_DOF_ORDER})
        obs = Observation(robot_state=state, user_input=UserInput())
        vector = walk.build_observation(obs)
        self.assertEqual(len(vector), 51)
        np.testing.assert_allclose(vector[3:6], [1., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(vector[6:48], np.zeros(42))
        command = MotorCommand()
        walk.step(obs, command)
        self.assertTrue(np.isfinite(walk._last_action).all())
        self.assertEqual(set(command.target_angles), set(OBSERVATION_DOF_ORDER))
        self.assertTrue(np.isfinite(list(command.target_angles.values())).all())
        # Prior policy action, rather than physical joint targets, is fed back.
        np.testing.assert_allclose(walk.build_observation(obs)[34:48], walk._last_action)
        old_targets = dict(command.target_angles)
        state.gyro[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            walk.step(obs, command)
        self.assertEqual(command.target_angles, old_targets)


if __name__ == "__main__":
    unittest.main()
