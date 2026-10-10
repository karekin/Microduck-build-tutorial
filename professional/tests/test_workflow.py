"""Source integrity and experiment separation checks without cloud calls."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("professional_run", SCRIPT)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def args(self, **changes):
        settings = dict(profile="smoke", num_envs=None, iterations=None, task="walk",
                        seed=42, backend="local", flavor="l4x1", timeout_minutes=60,
                        namespace=None)
        return argparse.Namespace(**(settings | changes))

    def test_pinned_full_source_is_intact(self):
        result = workflow.verify_source()
        self.assertEqual(result["policy_contract"]["actor_observations"], 61)
        self.assertGreater(result["files_verified"], 200)

    def test_integrity_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, "文件集变化"):
                workflow.verify_source(Path(path))

    def test_cloud_command_is_bounded_and_local_does_not_submit(self):
        local = workflow.training_command(self.args())
        self.assertNotIn("--hf-jobs", local)
        cloud = workflow.training_command(self.args(backend="hf", namespace="example"))
        self.assertIn("--hf-jobs", cloud)
        self.assertEqual(cloud[cloud.index("--timeout")+1], "60m")
        self.assertEqual(cloud[cloud.index("--agent.max-iterations")+1], "5")
        self.assertIn("--no-wandb", cloud)

    def test_seed_matches_environment_and_optimizer_run(self):
        command = workflow.training_command(self.args(seed=43))
        self.assertEqual(command[command.index("--env.seed")+1], "43")
        self.assertEqual(command[command.index("--agent.seed")+1], "43")
        self.assertIn("professional-walk-smoke-seed43", command)

    def test_negative_or_unbounded_work_is_rejected(self):
        for changes in ({"num_envs": 0}, {"iterations": -1}, {"num_envs": 1000000}):
            with self.assertRaises(ValueError):
                workflow.training_command(self.args(**changes))

    def test_failed_fetch_does_not_publish_half_initialized_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "official"

            def run(command, **kwargs):
                if "fetch" in command:
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(workflow, "PREPARED", prepared), patch.object(workflow.subprocess, "run", side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    workflow.prepare_workspace()
            self.assertFalse(prepared.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
