"""Regression checks for GPU claims and pinned-source identification."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("gpu_pipeline", Path(__file__).resolve().parents[1] / "gpu_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class GpuEvidenceTest(unittest.TestCase):
    def test_no_cuda_never_silently_becomes_cpu(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
            pipeline.device_identity(fake_torch, "cuda:0")
        self.assertFalse(pipeline.device_identity(fake_torch, "cpu")["gpu_execution"])

    def test_parent_git_head_is_not_mislabeled_as_official_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src/mjlab_microduck").mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='official-test'\n")
            source = root / "src/mjlab_microduck/__init__.py"
            source.write_text("x=1\n")
            self.assertEqual(pipeline.official_root(root / "src"), root.resolve())
            first = pipeline.source_identity(root, "a" * 40)
            self.assertEqual(first["commit"], "a" * 40)
            self.assertEqual(first["commit_origin"], "caller_pinned_archive")
            source.write_text("x=2\n")
            second = pipeline.source_identity(root, "a" * 40)
            self.assertNotEqual(first["python_and_lock_sha256"], second["python_and_lock_sha256"])


if __name__ == "__main__":
    unittest.main()
