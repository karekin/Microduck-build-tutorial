"""Physical keyboard limits, warmup, and slew control independent of policy success."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sim.low_speed_input import LowSpeedKeyboardInput
from input.input_source import scale_velocity


class LowSpeedInputTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.source = LowSpeedKeyboardInput(time_source=lambda: self.now,
                                           stop_flag_path="/tmp/unused-keyboard-unit-fixture",
                                           initial_moves={"walk"})

    def advance(self, until):
        samples = []
        while self.now + .02 <= until + 1e-8:
            self.now += .02
            samples.append(scale_velocity(self.source.read().velocity))
        return samples

    def test_first_up_and_repeats_are_bounded_and_ramp_after_warmup(self):
        self.source.read()
        for _ in range(10):
            self.source.key_callback(265)
        early = self.advance(1.98)
        self.assertTrue(all(r["vx"] == 0. for r in early))
        values = self.advance(5.)
        previous = 0.
        for sample in values:
            self.assertLessEqual(sample["vx"], .05 + 1e-9)
            self.assertLessEqual(abs(sample["vx"] - previous), .025 * .02 + 1e-9)
            previous = sample["vx"]
        self.assertAlmostEqual(values[-1]["vx"], .05)

    def test_add_translation_without_a_turn_command_jump_and_stop_gradually(self):
        self.source.read()
        self.source.key_callback(262)
        self.source.key_callback(262)
        self.advance(5.)
        before = scale_velocity(self.source.read().velocity)
        self.assertAlmostEqual(before["vtheta"], .10)
        self.source.key_callback(265)
        after = self.advance(5.02)[-1]
        self.assertAlmostEqual(after["vtheta"], before["vtheta"])
        self.advance(8.)
        self.source.key_callback(ord("X"))
        first = self.advance(8.02)[-1]
        self.assertGreater(first["vx"], 0.)
        self.assertGreater(first["vtheta"], 0.)
        last = self.advance(11.)[-1]
        self.assertTrue(all(abs(v) < 1e-8 for v in last.values()))

    def test_reset_discards_target_and_slew_history(self):
        self.source.read()
        self.source.key_callback(265)
        self.advance(5.)
        self.source.key_callback(ord("R"))
        self.assertTrue(self.source.consume_reset())
        self.now = 0.
        self.assertTrue(all(v == 0. for v in self.source.read().velocity.values()))
        self.assertTrue(all(r["vx"] == 0. for r in self.advance(5.)))

    def test_speed_selection_accelerates_and_decelerates_without_jumps(self):
        self.source.read()
        self.source.key_callback(265)
        self.advance(5.)
        before = scale_velocity(self.source.read().velocity)["vx"]
        self.source.key_callback(ord("9"))
        self.assertAlmostEqual(scale_velocity(self.source.read().velocity)["vx"], before)
        faster = self.advance(8.)
        self.assertAlmostEqual(faster[-1]["vx"], .10)
        previous = before
        for row in faster:
            self.assertLessEqual(abs(row["vx"] - previous), .025 * .02 + 1e-9)
            self.assertLessEqual(row["vx"], .10 + 1e-9)
            previous = row["vx"]
        self.source.key_callback(ord("8"))
        self.assertAlmostEqual(scale_velocity(self.source.read().velocity)["vx"], .10)
        self.assertAlmostEqual(self.advance(11.)[-1]["vx"], .05)

    def test_initial_fast_limit_and_repeated_up_cannot_exceed_point_one(self):
        self.source = LowSpeedKeyboardInput(time_source=lambda: self.now,
                                           stop_flag_path="/tmp/unused-keyboard-unit-fixture",
                                           initial_moves={"walk"}, forward_limit=.10)
        self.source.read()
        for _ in range(20):
            self.source.key_callback(265)
        values = self.advance(8.)
        self.assertAlmostEqual(values[-1]["vx"], .10)
        self.assertTrue(all(row["vx"] <= .10 + 1e-9 for row in values))
        with self.assertRaises(ValueError):
            LowSpeedKeyboardInput(time_source=lambda: self.now, stop_flag_path="/tmp/unused-fixture", forward_limit=.20)

    def test_heading_assist_cancels_on_stop_and_rebases_when_forward_resumes(self):
        heading = [0.]
        self.source = LowSpeedKeyboardInput(time_source=lambda: self.now,
                                           stop_flag_path="/tmp/unused-keyboard-unit-fixture",
                                           initial_moves={"walk"}, forward_limit=.10,
                                           heading_source=lambda: heading[0])
        self.source.read()
        self.source.key_callback(265)
        self.source.key_callback(265)
        self.advance(8.)
        heading[0] = .2
        self.advance(8.02)
        self.assertAlmostEqual(self.source.heading_correction, -.2)
        self.source.key_callback(ord("X"))
        first = self.advance(8.04)[-1]
        self.assertEqual(self.source.heading_correction, 0.)
        self.assertIsNone(self.source._heading_reference)
        self.assertEqual(first["vtheta"], 0.)
        self.assertGreater(first["vx"], 0.)  # Braking is still gradual.
        self.advance(13.)
        self.source.key_callback(265)
        self.advance(13.02)
        self.assertAlmostEqual(self.source._heading_reference, .2)
        self.assertEqual(self.source.heading_correction, 0.)

    def test_heading_assist_uses_shortest_wrap_and_rejects_invalid_heading(self):
        heading = [3.13]
        self.source = LowSpeedKeyboardInput(time_source=lambda: self.now,
                                           stop_flag_path="/tmp/unused-keyboard-unit-fixture",
                                           initial_moves={"walk"}, forward_limit=.10,
                                           heading_source=lambda: heading[0])
        self.source.read()
        self.source.key_callback(265)
        self.advance(5.)
        heading[0] = -3.13
        self.advance(5.02)
        self.assertLess(abs(self.source.heading_correction), .03)
        heading[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.advance(5.04)


if __name__ == "__main__":
    unittest.main()
