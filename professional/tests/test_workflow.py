"""Source integrity and experiment separation checks without cloud calls."""
import argparse
import ast
import contextlib
import io
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

    def test_aliases_cover_exact_pinned_official_registry(self):
        # Read the actual registration declarations without importing mjlab/CUDA.
        # This independently catches omitted tasks and invented Backlash twins.
        registry = workflow.VENDOR / "src/mjlab_microduck/tasks/__init__.py"
        tree = ast.parse(registry.read_text())
        registered = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_mjlab_task":
                for keyword in node.keywords:
                    if keyword.arg == "task_id" and isinstance(keyword.value, ast.Constant):
                        registered.add(keyword.value.value)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_BACKLASH_TASKS" for target in node.targets):
                registered.update(item.elts[0].value for item in node.value.elts)
        self.assertEqual(len(registered), 33)
        self.assertEqual(set(workflow.TASKS.values()), registered)
        self.assertEqual(len(workflow.TASKS), len(registered))

    def test_all_tasks_generate_bounded_commands(self):
        for alias, task_id in workflow.TASKS.items():
            with self.subTest(task=alias):
                command = workflow.training_command(self.args(task=alias, backend="hf", namespace="example"))
                self.assertEqual(command[4], task_id)
                self.assertEqual(command[command.index("--agent.max-iterations") + 1], "5")
                self.assertEqual(command[command.index("--timeout") + 1], "60m")

    def test_existing_aliases_keep_their_task_contracts(self):
        self.assertEqual({alias: workflow.TASKS[alias] for alias in ("walk", "walk-backlash", "recovery", "standup", "sitstand", "kick")}, {
            "walk": "Mjlab-Velocity-Flat-MicroDuck", "walk-backlash": "Mjlab-Velocity-Flat-Backlash-MicroDuck",
            "recovery": "Mjlab-VelStand-Flat-MicroDuck", "standup": "Mjlab-StandUp-Flat-MicroDuck",
            "sitstand": "Mjlab-SitStand-Flat-MicroDuck", "kick": "Mjlab-BallKick-Flat-MicroDuck"})

    def test_catalog_identifies_teacher_and_hardware_prerequisites(self):
        catalog = {item["alias"]: item for item in workflow.task_catalog()}
        for alias in ("recovery", "recovery-rough", "recovery-backlash", "recovery-rough-backlash"):
            self.assertTrue(catalog[alias]["expert_checkpoints_required"])
        self.assertFalse(catalog["standup"]["expert_checkpoints_required"])
        for alias in ("rollers", "swizzle-backlash", "roller-standup", "spin"):
            self.assertEqual(catalog[alias]["hardware"], "roller_wheels")
        self.assertEqual(catalog["roulade"]["hardware"], "standard_feet")

    def test_default_new_task_command_does_not_execute_or_prepare(self):
        with patch("sys.argv", [str(SCRIPT), "train", "--task", "ground-pick"]), \
             patch.object(workflow, "prepare_workspace", side_effect=AssertionError("must not prepare")), \
             patch.object(workflow.subprocess, "run", side_effect=AssertionError("must not execute")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.main(), 0)
        self.assertIn("Mjlab-GroundPick-Flat-MicroDuck", output.getvalue())

    def test_catalog_is_read_only(self):
        with patch("sys.argv", [str(SCRIPT), "catalog"]), \
             patch.object(workflow.subprocess, "run", side_effect=AssertionError("must not execute")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.main(), 0)
        self.assertIn("expert_checkpoints_required", output.getvalue())

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
