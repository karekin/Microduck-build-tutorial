"""Viewer startup library lookup and real 14-joint reset regressions."""
import os
from pathlib import Path
import sys
import sysconfig
import tempfile
import unittest
import mujoco
import numpy as np
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import MOTOR_TO_ID, NEUTRAL_POSE
from moves.move import MoveState
from scheduler import Scheduler
from sim.launch_sim import launch_environment
from sim.mujoco_controller import MuJoCoController
from sim.mujoco_input import MuJoCoInputSource
from sim.sim_main import simulation_moves, ScriptedDemoInput


class ViewerStartupTests(unittest.TestCase):
    def test_reset_clears_latched_translation_and_rotation(self):
        source = MuJoCoInputSource({"v": "walk"})
        source.key_callback(265)
        source.key_callback(262)
        self.assertNotEqual(source.read().velocity["vx"], 0.)
        self.assertNotEqual(source.read().velocity["vtheta"], 0.)
        source.key_callback(ord("R"))
        self.assertTrue(source.consume_reset())
        self.assertFalse(any(source.read().velocity.values()))

    def test_native_viewer_reset_has_legal_pose_and_reinitializes_feedback(self):
        scene = Path(__file__).resolve().parents[1] / "src/model/mjcf/scene.xml"
        source = MuJoCoInputSource({"v": "walk"})
        controller = MuJoCoController(str(scene), headless=True, scheduler_driven=True, reset_source=source)
        try:
            reference_model = mujoco.MjModel.from_xml_path(str(scene))
            reference_data = mujoco.MjData(reference_model)
            reference_data.qpos[2] = .165
            for name, value in NEUTRAL_POSE.items():
                reference_data.qpos[reference_model.jnt_qposadr[reference_model.joint(name).id]] = value
            mujoco.mj_forward(reference_model, reference_data)
            for name in ("left_foot", "right_foot"):
                np.testing.assert_allclose(controller._data.site(name).xpos,
                                           reference_data.site(name).xpos, atol=1e-8)
            controller.end_tick()
            source.key_callback(265)
            mujoco.mj_resetData(controller._model, controller._data)
            self.assertAlmostEqual(controller._data.qpos[2], .165)
            self.assertTrue(controller.begin_tick())
            self.assertFalse(any(source.read().velocity.values()))
            self.assertEqual(controller._bam.last_ts, 0.)
            self.assertEqual(controller._targets, {n: controller._data.qpos[i]
                                                   for n, i in controller._name_to_qpos_idx.items()})
        finally:
            controller.shutdown()

    def test_scripted_demo_ignores_arrow_mixing_and_can_stop_and_replay(self):
        now = [4.]
        source = ScriptedDemoInput([.05 / .7, 0., 0.], time_source=lambda: now[0],
                                  delay_s=2., ramp_s=2., stop_flag_path="/tmp/unused-unit-fixture")
        source.key_callback(262)
        self.assertEqual(source.read().velocity["vtheta"], 0.)
        self.assertAlmostEqual(source.read().velocity["vx"], .05 / .7)
        source.key_callback(ord("X"))
        self.assertFalse(any(source.read().velocity.values()))
        source.key_callback(ord("R"))
        now[0] = 0.
        self.assertFalse(any(source.read().velocity.values()))
        now[0] = 3.
        self.assertAlmostEqual(source.read().velocity["vx"], .5 * .05 / .7)

    def test_uv_macos_shared_library_path_is_preserved_in_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "libpython3.12.dylib").touch()
            values = {"LIBDIR": directory, "LDLIBRARY": "libpython3.12.dylib"}
            with patch("sim.launch_sim.sys.platform", "darwin"), \
                    patch("sim.launch_sim.sysconfig.get_config_var", side_effect=values.get), \
                    patch.dict(os.environ, {"DYLD_FALLBACK_LIBRARY_PATH": "/existing/path"}):
                env = launch_environment()
            self.assertEqual(env["DYLD_FALLBACK_LIBRARY_PATH"], directory + ":/existing/path")

    def test_running_policy_and_reset_use_only_real_microduck_joint_names(self):
        scene = Path(__file__).resolve().parents[1] / "src/model/mjcf/scene.xml"
        with tempfile.TemporaryDirectory() as directory:
            stop = str(Path(directory) / "stop")
            source = MuJoCoInputSource({"v": "walk"}, stop_flag_path=stop, initial_moves={"walk"})
            controller = MuJoCoController(str(scene), headless=True, scheduler_driven=True,
                                          reset_source=source, stop_flag_path=stop)
            moves = simulation_moves(controller)
            scheduler = Scheduler(controller=controller, input_source=source, moves=moves,
                                  stop_flag_path=stop)
            scheduler.observer.observe_current = True
            active_seen = []
            def record(tick, obs, command):
                self.assertTrue(set(command.target_angles).issubset(MOTOR_TO_ID))
                active_seen.append(moves["walk"].state == MoveState.ACTIVE)
                if tick == 80:
                    source.key_callback(ord("R"))
            try:
                source.key_callback(ord("R"))
                count = scheduler.run(max_ticks=90, realtime=False,
                                      time_source=lambda: controller.simulation_time,
                                      tick_callback=record)
                self.assertEqual(count, 90)
                self.assertTrue(any(active_seen))
                self.assertLess(controller.simulation_time, 0.3)
            finally:
                controller.shutdown()


if __name__ == "__main__":
    unittest.main()
