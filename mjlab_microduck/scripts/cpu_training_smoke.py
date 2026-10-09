# SPDX-License-Identifier: Apache-2.0
"""One CPU PPO update and fresh checkpoint-to-ONNX export, never hardware use.

Run with the installed training environment's Python. Every output is a fresh
UNTRAINED_DEMO_NOT_FOR_HARDWARE artifact. Default networks, rewards and PPO
settings are preserved; the bounded run uses one environment and 24 samples.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


REPO = Path(__file__).resolve().parents[2]
TASK = "Mjlab-Velocity-Microduck"
LABEL = "UNTRAINED_DEMO_NOT_FOR_HARDWARE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, report: dict) -> None:
    # Logging stays local even if the caller has W&B credentials configured.
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["MJLAB_WARP_QUIET"] = "1"

    import mjlab  # Registers task entry points before robot-module imports.
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from importlib.metadata import version

    torch.set_num_threads(1)
    agent_cfg = load_rl_cfg(TASK)
    env_cfg = load_env_cfg(TASK)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    agent_cfg.max_iterations = 1
    agent_cfg.num_steps_per_env = 24
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.save_interval = 1
    torch.manual_seed(agent_cfg.seed)

    report.update(
        task=TASK,
        device="cpu",
        cuda_available=torch.cuda.is_available(),
        versions={name: version(name) for name in ("mjlab", "better-actuator-models", "torch", "rsl-rl-lib", "onnx")},
        overrides={"num_envs": 1, "max_iterations": 1, "num_steps_per_env": 24,
                   "logger": "tensorboard", "upload_model": False,
                   "save_interval": 1, "torch_cpu_threads": 1},
        agent_configuration=asdict(agent_cfg),
        seed=agent_cfg.seed,
        changes_to_rewards_terminations_curriculum=False,
    )
    original_paths = (
        REPO / "mjlab_microduck/src/mjlab_microduck/agents/velocity.pt",
        REPO / "microduck/src/agents/walk.onnx",
    )
    original_hashes = {str(p.relative_to(REPO)): sha256(p) for p in original_paths}
    report["original_asset_sha256"] = original_hashes
    report["phase"] = "environment"
    env = ManagerBasedRlEnv(env_cfg, device="cpu")
    wrapped = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK)
        runner = runner_cls(wrapped, asdict(agent_cfg), str(output / "local_logs"), device="cpu")
        policy = runner.alg.get_policy()
        before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        report["phase"] = "ppo_update"
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        changes = {name: float((p.detach() - before[name]).abs().max())
                   for name, p in policy.named_parameters()}
        if not all(bool(torch.isfinite(p).all()) for p in policy.parameters()):
            raise RuntimeError("Non-finite actor parameters after PPO update")
        if not any(delta > 0 for delta in changes.values()):
            raise RuntimeError("No actor parameter changed during the PPO update")
        report["actor_parameter_max_abs_updates"] = changes
        report["control_steps"] = env.common_step_counter
        if env.common_step_counter != 24:
            raise RuntimeError(f"Expected 24 control steps, got {env.common_step_counter}")
        checkpoint = output / f"{LABEL}.pt"
        runner.save(str(checkpoint), infos={"artifact_status": LABEL,
                                          "training_iterations_completed": 1,
                                          "not_for_hardware": True})
        report["checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
    finally:
        (wrapped if wrapped is not None else env).close()

    report["phase"] = "fresh_checkpoint_export"
    exporter = REPO / "mjlab_microduck/src/mjlab_microduck/scripts/export_onnx.py"
    command = [sys.executable, str(exporter), "--checkpoint", str(checkpoint), "--device", "cpu"]
    result = subprocess.run(command, cwd=output, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (output / "export.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Fresh checkpoint export failed; see {output / 'export.log'}")
    exported = output / "policy.onnx"
    labelled = output / f"{LABEL}.onnx"
    exported.rename(labelled)
    attach_metadata_to_onnx(str(labelled), {
        "artifact_status": LABEL, "not_for_hardware": "true",
        "training_iterations_completed": "1", "training_control_steps": "24",
        "training_device": "cpu", "training_purpose": "pipeline_smoke_only",
    })
    # The velocity runner also saves intermediate checkpoints/ONNX files. Mark
    # those local byproducts too, so none can be mistaken for a usable gait.
    for other in output.rglob("*.onnx"):
        if other != labelled:
            attach_metadata_to_onnx(str(other), {
                "artifact_status": LABEL, "not_for_hardware": "true",
                "training_iterations_completed": "1",
            })
    for other in output.rglob("*.pt"):
        if other != checkpoint:
            saved = torch.load(other, map_location="cpu", weights_only=False)
            saved["infos"] = {**(saved.get("infos") or {}),
                              "artifact_status": LABEL, "not_for_hardware": True,
                              "training_iterations_completed": 1}
            torch.save(saved, other)
    model = onnx.load(labelled)
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    inputs = np.zeros((1, 51), dtype=np.float32)
    outputs = evaluator.run(None, {model.graph.input[0].name: inputs})[0]
    if outputs.shape != (1, 14) or not np.isfinite(outputs).all():
        raise RuntimeError(f"Exported ONNX has invalid output: {outputs.shape}")
    if {str(p.relative_to(REPO)): sha256(p) for p in original_paths} != original_hashes:
        raise RuntimeError("Original checkpoint or deployment ONNX changed")
    report.update(phase="complete", status="PASS", original_assets_unchanged=True,
                  onnx={"path": str(labelled), "sha256": sha256(labelled),
                        "input_shape": [1, 51], "output_shape": list(outputs.shape),
                        "checker": "PASS", "reference_inference_finite": True},
                  interpretation="One PPO update validates the training/export pipeline only. No stable gait or hardware claim.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = REPO / ".omx/artifacts/training-preflight" / (
        LABEL + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--output-dir", type=Path, default=default)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"artifact_status": LABEL, "status": "RUNNING", "not_for_hardware": True}
    try:
        with (output / "training.log").open("w") as log, contextlib.redirect_stdout(log):
            run(output, report)
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "artifact_status": LABEL,
                      "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
