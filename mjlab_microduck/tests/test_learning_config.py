# SPDX-License-Identifier: Apache-2.0
"""Fast checks for teaching-task isolation; the pipeline checks actual physics."""

import unittest

import mjlab  # noqa: F401 — task registration before plugin imports.

from mjlab_microduck.tasks.microduck_learning_env_cfg import make_microduck_learning_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


class LearningTaskIsolationTest(unittest.TestCase):
    def test_learning_changes_do_not_leak_to_velocity_task(self):
        before = make_microduck_velocity_env_cfg()
        learning = make_microduck_learning_env_cfg()
        after = make_microduck_velocity_env_cfg()
        self.assertEqual(set(before.rewards), set(after.rewards))
        self.assertGreater(len(after.rewards), 3)
        self.assertTrue(after.curriculum)
        self.assertEqual(after.scene.entities["robot"].articulation.actuators[0].vin_range,
                         before.scene.entities["robot"].articulation.actuators[0].vin_range)
        learning.rewards["upright"].weight = 999.0
        self.assertEqual(after.rewards["upright"].weight, 2.0)
        self.assertTrue(after.observations["actor"].enable_corruption)

    def test_standing_task_has_no_hidden_random_sampling_or_privileged_critic(self):
        cfg = make_microduck_learning_env_cfg()
        command = cfg.commands["twist"]
        self.assertEqual(command.rel_standing_envs, 1.0)
        self.assertEqual(command.rel_forward_envs, 0.0)
        self.assertEqual(command.ranges.lin_vel_x, (0.0, 0.0))
        self.assertEqual(set(cfg.events), {"bam_init", "reset_base", "reset_robot_joints"})
        self.assertFalse(cfg.curriculum)
        self.assertFalse(cfg.scene.sensors)
        self.assertEqual(set(cfg.observations["actor"].terms), set(cfg.observations["critic"].terms))
        self.assertEqual(len(cfg.rewards), 3)
        for term in cfg.observations["actor"].terms.values():
            self.assertIsNone(term.noise)
            self.assertEqual((term.delay_min_lag, term.delay_max_lag), (0, 0))


if __name__ == "__main__":
    unittest.main()
