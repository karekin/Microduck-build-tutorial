# SPDX-License-Identifier: Apache-2.0
"""CPU PPO -> fresh checkpoint -> ONNX -> physical rollout, never hardware."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO / "learning_artifacts"
LABEL = "LEARNING_ONLY_NOT_FOR_HARDWARE"
TASK = "Mjlab-Learning-Stand-Microduck"


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, output, report):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["MJLAB_WARP_QUIET"] = "1"
    import mjlab  # noqa: F401 — discovers the local task plugin.
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from importlib.metadata import version

    torch.set_num_threads(1)
    agent_cfg = load_rl_cfg(TASK)
    agent_cfg.max_iterations = args.iterations
    env_cfg = load_env_cfg(TASK)
    env_cfg.seed = agent_cfg.seed
    torch.manual_seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    originals = [REPO / "mjlab_microduck/src/mjlab_microduck/agents/velocity.pt",
                 REPO / "microduck/src/agents/walk.onnx"]
    original_hashes = {str(p.relative_to(REPO)): checksum(p) for p in originals}
    report.update(task=TASK, device="cpu", seed=agent_cfg.seed,
                  versions={name: version(name) for name in
                            ("mjlab", "torch", "rsl-rl-lib", "onnx", "better-actuator-models")},
                  rewards={name: {"weight": term.weight, "function": term.func.__name__}
                           for name, term in env_cfg.rewards.items()},
                  configuration=asdict(agent_cfg), original_asset_sha256=original_hashes)
    if len(env_cfg.rewards) != 3 or env_cfg.curriculum or env_cfg.scene.num_envs != 1:
        raise RuntimeError("Learning task must remain three rewards, no curriculum, one environment")
    report["phase"] = "construct_cpu_simulation"
    print("[1/4] CPU MuJoCo: one robot, fixed zero-speed command, three rewards", flush=True)
    env = ManagerBasedRlEnv(env_cfg, device="cpu")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device="cpu")
        if args.view:
            checkpoint = args.checkpoint or Path(json.loads(
                (ARTIFACT_ROOT / "latest.json").read_text())["checkpoint"])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            infos = saved.get("infos", {})
            if not infos.get("not_for_hardware") or infos.get("task") != TASK:
                raise ValueError("Viewer accepts only a labelled checkpoint for the learning task")
            runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
            from mjlab.viewer.viser import ViserPlayViewer
            import viser
            server = viser.ViserServer(host="127.0.0.1", label=LABEL)
            try:
                ViserPlayViewer(wrapped, runner.get_inference_policy(device="cpu"),
                                viser_server=server).run()
            finally:
                server.stop()
            return

        policy = runner.alg.get_policy()
        before = {name: parameter.detach().clone()
                  for name, parameter in policy.named_parameters()}
        report["phase"] = "ppo_train"
        print(f"[2/4] Real PPO: {args.iterations} iterations × 24 control steps", flush=True)
        started = time.perf_counter()
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        report["training_seconds"] = time.perf_counter() - started
        changes = {name: float((parameter.detach() - before[name]).abs().max())
                   for name, parameter in policy.named_parameters()}
        if not any(delta > 0 for name, delta in changes.items() if name.startswith("mlp.")):
            raise RuntimeError("PPO did not update any actor MLP weight")
        if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
            raise RuntimeError("Non-finite trained policy parameter")
        report["actor_parameter_max_abs_updates"] = changes
        trained_steps = env.common_step_counter
        if trained_steps != args.iterations * agent_cfg.num_steps_per_env:
            raise RuntimeError("Unexpected number of real training control steps")
        report["training_control_steps"] = trained_steps
        checkpoint = output / f"{LABEL}.pt"
        runner.save(str(checkpoint), infos={"not_for_hardware": True, "task": TASK,
                                          "artifact_status": LABEL,
                                          "training_iterations_completed": args.iterations})

        # Reload the saved actor before export: the checkpoint is part of the loop.
        report["phase"] = "reload_checkpoint_and_export"
        print("[3/4] Save and reload .pt; export and check ONNX", flush=True)
        # RSL-RL updates normalizer buffers under inference_mode during learn;
        # loading them back into the same runner must use that mode as well.
        with torch.inference_mode():
            runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
        filename = f"{LABEL}.onnx"
        runner.export_policy_to_onnx(str(output), filename)
        model_path = output / filename
        metadata = get_base_metadata(env, run_path=output.name)
        metadata.update(artifact_status=LABEL, not_for_hardware="true", task=TASK,
                        training_control_steps=str(trained_steps), training_device="cpu")
        attach_metadata_to_onnx(str(model_path), metadata)
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        evaluator = ReferenceEvaluator(model)
        input_name = model.graph.input[0].name
        inference = runner.get_inference_policy(device="cpu")

        # Compare on observations obtained from physics, then drive physics with
        # ONNX output (not a shipped policy, zero actions, or a prerecorded clip).
        report["phase"] = "fresh_onnx_physical_rollout"
        print(f"[4/4] Drive CPU physics with the new ONNX for {args.rollout_steps} steps", flush=True)
        with torch.inference_mode():
            wrapped.reset()
            obs = wrapped.get_observations()
        qpos_before = env.sim.data.qpos.detach().clone()
        max_error = 0.0
        reward_sum = 0.0
        resets = 0
        physical_delta = 0.0
        rollout_started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.rollout_steps):
                values = obs["actor"].detach().cpu().numpy().astype(np.float32)
                actions = evaluator.run(None, {input_name: values})[0]
                expected = inference(obs).detach().cpu().numpy()
                if values.shape != (1, 51) or actions.shape != (1, 14):
                    raise RuntimeError("The learning observation/action contract changed")
                np.testing.assert_allclose(actions, expected, rtol=1e-4, atol=1e-5)
                if not np.isfinite(actions).all():
                    raise RuntimeError("Non-finite ONNX action")
                max_error = max(max_error, float(np.max(np.abs(actions - expected))))
                obs, reward, done, _ = wrapped.step(torch.from_numpy(actions))
                if not bool(torch.isfinite(env.sim.data.qpos).all()) or not bool(torch.isfinite(reward).all()):
                    raise RuntimeError("Non-finite physical state or reward")
                reward_sum += float(reward.sum())
                resets += int(done.sum())
                # A final auto-reset may restore the initial qpos exactly;
                # retain evidence of motion throughout the trajectory instead.
                physical_delta = max(physical_delta, float((env.sim.data.qpos - qpos_before).abs().max()))
        if physical_delta <= 0 or env.common_step_counter != trained_steps + args.rollout_steps:
            raise RuntimeError(f"Physical rollout check failed: qpos_delta={physical_delta}, "
                               f"counter={env.common_step_counter}, expected={trained_steps + args.rollout_steps}")
        if {str(p.relative_to(REPO)): checksum(p) for p in originals} != original_hashes:
            raise RuntimeError("Original policy assets changed")
        report.update(status="PASS", phase="complete", original_assets_unchanged=True,
                      checkpoint={"path": str(checkpoint), "sha256": checksum(checkpoint)},
                      onnx={"path": str(model_path), "sha256": checksum(model_path),
                            "checker": "PASS", "input_shape": [1, 51], "output_shape": [1, 14],
                            "pytorch_comparison_max_abs_error": max_error},
                      rollout={"control_steps": args.rollout_steps, "simulated_seconds": args.rollout_steps * env.step_dt,
                               "wall_seconds": time.perf_counter() - rollout_started,
                               "finite_state_and_actions": True, "qpos_max_abs_change": physical_delta,
                               "reward_sum": reward_sum, "episode_resets": resets},
                      interpretation="Actual PPO updates and fresh-policy CPU physics only; no stable gait or hardware qualification.")
        (ARTIFACT_ROOT / "latest.json").write_text(json.dumps(
            {"checkpoint": str(checkpoint), "onnx": str(model_path), "report": str(output / "result.json")}, indent=2) + "\n")
    finally:
        wrapped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--view", action="store_true", help="View the latest learning checkpoint in a local browser")
    parser.add_argument("--checkpoint", type=Path, help="Optional labelled learning checkpoint for --view")
    args = parser.parse_args()
    if args.iterations < 1 or args.rollout_steps < 1:
        parser.error("iterations and rollout-steps must be positive")
    if args.view:
        if args.checkpoint is None and not (ARTIFACT_ROOT / "latest.json").is_file():
            parser.error("Run ./learn.sh once before opening the latest learning checkpoint")
        run(args, None, {})
        return
    output = (args.output_dir or ARTIFACT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "artifact_status": LABEL, "not_for_hardware": True}
    started = time.perf_counter()
    try:
        run(args, output, report)
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output_dir": str(output),
                      "total_seconds": report["total_seconds"], "not_for_hardware": True}, indent=2))


if __name__ == "__main__":
    main()
