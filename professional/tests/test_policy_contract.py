"""Exercise the official ONNX interface gate with existing optional ONNX tools."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "professional/upstream/microduck_rl/src/mjlab_microduck/publish/manifest.py"
spec = importlib.util.spec_from_file_location("official_manifest", SOURCE)
manifest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = manifest
spec.loader.exec_module(manifest)
HAS_ONNX = importlib.util.find_spec("onnx") is not None


@unittest.skipUnless(HAS_ONNX, "用已安装ONNX的训练环境Python运行此组检查")
class OfficialContractTests(unittest.TestCase):
    def model(self, directory, observations):
        import onnx
        from onnx import TensorProto, helper
        path = Path(directory) / f"shape-{observations}.onnx"
        graph = helper.make_graph(
            [helper.make_node("Slice", ["obs", "starts", "ends", "axes"], ["actions"])],
            "interface-test",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, observations])],
            [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
            initializer=[helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
                         helper.make_tensor("ends", TensorProto.INT64, [1], [14]),
                         helper.make_tensor("axes", TensorProto.INT64, [1], [1])])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        return path

    def test_official_61_input_contract_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            shape = manifest.check_onnx(self.model(directory, 61))
            self.assertEqual((shape.obs_len, shape.action_len), (61, 14))

    def test_legacy_51_input_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                manifest.check_onnx(self.model(directory, 51))


if __name__ == "__main__":
    unittest.main()
