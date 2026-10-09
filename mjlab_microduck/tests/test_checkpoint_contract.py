"""The bundled old checkpoint must not be silently adapted to a new robot."""
import unittest
from pathlib import Path
from mjlab_microduck.scripts.export_onnx import validate_checkpoint_dimensions


class CheckpointContractTests(unittest.TestCase):
    def test_bundled_18_action_checkpoint_is_rejected_for_14_joint_task(self):
        path = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/agents/velocity.pt"
        with self.assertRaisesRegex(ValueError, "63 observations / 18 actions"):
            validate_checkpoint_dimensions(path)
        validate_checkpoint_dimensions(path, obs_dim=63, action_dim=18)


if __name__ == "__main__":
    unittest.main()
