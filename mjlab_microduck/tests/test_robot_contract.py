"""Training configuration contracts; require declared MjLab/BAM dependencies.

Entity compilation exercises actual MjLab/BAM code without starting training or
requiring a CUDA device. Environment stepping needs a separate runtime check.
"""

import unittest

# Discover task entry points before importing robot submodules, so the plugin
# loader does not encounter a partially initialized robot configuration.
import mjlab
import mujoco
import numpy as np
from bam.model import load_model
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ROBOT_CFG,
    MICRODUCK_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


class TrainingRobotContractTests(unittest.TestCase):
    def test_legacy_config_alias_builds_real_14dof_entity(self):
        self.assertIs(MICRODUCK_ROBOT_CFG, MICRODUCK_WALK_ROBOT_CFG)
        model = MICRODUCK_ROBOT_CFG.build().compile()
        self.assertEqual((model.nu, model.nq, model.nv), (14, 21, 20))
        self.assertEqual(len(set(model.actuator(i).name for i in range(model.nu))), 14)
        self.assertGreater(model.nkey, 0)
        np.testing.assert_allclose(model.key_qpos[0, 3:7], [1, 0, 0, 0])
        np.testing.assert_allclose(model.site_quat[model.site("imu").id], [1, 0, 0, 0])
        for name in ("orientation", "angular-velocity", "imu_accel"):
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name), 0)

    def test_bam_voltage_sag_units_preserve_upstream_torque_gain(self):
        actuator = MICRODUCK_ROBOT_CFG.articulation.actuators[0]
        self.assertEqual(actuator.kp_fw, 200.0)
        self.assertEqual((actuator.delay_min_lag, actuator.delay_max_lag), (3, 6))
        if hasattr(actuator, "vin_drop_gain_range"):
            self.assertEqual(actuator.vin_drop_gain_range, (0.0, 0.2))
        else:
            kt = load_model(motor_name="xl330", model="m6").kt.value
            resistance = actuator.vin_drop_resistance_range
            np.testing.assert_allclose(np.array(resistance) / kt, [0.0, 0.2])

    def test_train_and_play_request_both_bam_per_environment_fields(self):
        self.assertIn("Mjlab-Velocity-Microduck", list_tasks())
        for play in (False, True):
            with self.subTest(play=play):
                event = make_microduck_velocity_env_cfg(play=play).events["bam_init"]
                self.assertEqual(event.mode, "startup")
                self.assertEqual(set(event.func.model_fields), {"dof_frictionloss", "dof_damping"})


if __name__ == "__main__":
    unittest.main()
