"""The one-hour rental must keep its authorized hardware and cost boundary."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("hf_hour", Path(__file__).resolve().parents[1]/"hf_hour.py")
hour = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hour
spec.loader.exec_module(hour)


class CostBoundaryTest(unittest.TestCase):
    def test_l4_minute_price_matches_authorized_hour(self):
        hardware = SimpleNamespace(name="l4x1", unit_label="minute", unit_cost_micro_usd=13333)
        self.assertLessEqual(float(hour.hourly_price(hardware)), 0.8)
        self.assertEqual(hour.MAX_SECONDS, 3600)

    def test_upgrade_or_price_increase_requires_new_authorization(self):
        for values in (("a100-large", "minute", 13333), ("l4x1", "hour", 800000),
                       ("l4x1", "minute", 13334), ("l4x1", "minute", 15000)):
            with self.assertRaises(ValueError):
                hour.hourly_price(SimpleNamespace(name=values[0], unit_label=values[1], unit_cost_micro_usd=values[2]))


if __name__ == "__main__":
    unittest.main()
