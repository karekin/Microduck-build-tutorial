"""Bounded official Microduck PPO pipeline. Defaults to CUDA; never falls back.

Copy this script into the pinned official checkout and run from that checkout.
CPU is an explicit developer self-test option, never evidence of GPU training.
No account access, upload, installation or hardware control happens here.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

TASK = "Mjlab-Velocity-Flat-MicroDuck"
LABEL = "PIPELINE_VALIDATION_NOT_FOR_HARDWARE"


def official_root(cwd: Path) -> Path:
    for candidate in (cwd.resolve(), *cwd.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/mjlab_microduck/__init__.py").is_file():
            return candidate
    raise FileNotFoundError("Run from the official checkout containing pyproject.toml and src/mjlab_microduck")


def source_identity(root: Path, supplied_commit: str | None) -> dict:
    root = root.resolve()
    commit = None
    origin = "unavailable"
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=False)
        if result.returncode == 0 and top.returncode == 0 and Path(top.stdout.strip()).resolve() == root:
            commit, origin = result.stdout.strip(), "checkout_git_head"
    except FileNotFoundError:
        pass  # Archives can run in a container without the Git executable.
    if supplied_commit:
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", supplied_commit):
            raise ValueError("source-commit must be a full lowercase Git SHA")
        if commit and supplied_commit != commit:
            raise ValueError("Supplied source commit differs from the official checkout HEAD")
        commit, origin = supplied_commit, origin if commit else "caller_pinned_archive"
    digest = hashlib.sha256()
    files = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py"))]
    if (root / "uv.lock").is_file():
        files.append(root / "uv.lock")
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return {"commit": commit, "commit_origin": origin,
            "python_and_lock_sha256": digest.hexdigest(),
            "hashed_files": len(files), "hash_scope": "pyproject, uv.lock if present, src/**/*.py"}


def device_identity(torch, requested: str) -> dict:
    if requested == "cpu":
        return {"device": "cpu", "gpu_execution": False,
                "interpretation": "Explicit local CPU self-test, not GPU training evidence"}
    if requested != "cuda:0":
        raise ValueError("Only cuda:0 or an explicit cpu self-test is supported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is forbidden")
    props = torch.cuda.get_device_properties(0)
    return {"device": "cuda:0", "gpu_execution": True, "gpu_name": props.name,
            "total_memory_bytes": props.total_memory, "compute_capability": [props.major, props.minor],
            "torch_cuda_version": torch.version.cuda}


@contextlib.contextmanager
def phase(report, name, synchronize=lambda: None):
    report["phase"] = name
    print(f"[pipeline] {name}", flush=True)
    synchronize()
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize()
        report.setdefault("timings_seconds", {})[name] = time.perf_counter() - started


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def run(args, root, output, report):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["MJLAB_WARP_QUIET"] = "1"
    sys.path.insert(0, str(root / "src"))
    wrapped = None
    with phase(report, "bootstrap"):
        import torch
        import numpy as np
        import onnx
        from onnx.reference import ReferenceEvaluator
        import mjlab  # noqa: F401 — discovers official tasks.
        import mjlab_microduck
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.os import dump_yaml

        if not Path(mjlab_microduck.__file__).resolve().is_relative_to(root / "src"):
            raise RuntimeError("Task package did not load from this official checkout's src")
        report["source"] = source_identity(root, args.source_commit)
        report["execution"] = device_identity(torch, args.device)
        report["versions"] = {}
        for package in ("mjlab", "torch", "rsl-rl-lib", "mujoco", "mujoco-warp", "onnx", "better-actuator-models"):
            try:
                report["versions"][package] = version(package)
            except PackageNotFoundError:
                report["versions"][package] = None
        report["bootstrap_scope"] = "Script imports and training-env construction; excludes HF scheduling and dependency installation"
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        synchronize = (lambda: torch.cuda.synchronize(0)) if args.device.startswith("cuda") else (lambda: None)
        cfg = load_env_cfg(TASK)
        cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
        agent_cfg = load_rl_cfg(TASK)
        agent_cfg.seed, agent_cfg.max_iterations = args.seed, args.iterations
        agent_cfg.logger, agent_cfg.upload_model = "tensorboard", False
        train_cfg = asdict(agent_cfg)
        params = output / "params"
        params.mkdir()
        # Match the official CLI: serialize before the runner injects _env.
        dump_yaml(params / "env.yaml", asdict(cfg))
        dump_yaml(params / "agent.yaml", train_cfg)
        report.update(task=TASK, seed=args.seed, num_envs=args.num_envs,
                      iterations=args.iterations, num_steps_per_env=agent_cfg.num_steps_per_env,
                      official_rewards_randomization_algorithm_preserved=True,
                      override_scope="environment count, seed, iteration limit, local logging only")
        env = ManagerBasedRlEnv(cfg, device=args.device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner_cls = load_runner_cls(TASK)
            runner = runner_cls(wrapped, train_cfg, device=args.device)
        except BaseException:
            wrapped.close()
            raise

    try:
        with phase(report, "training", synchronize):
            actor = runner.alg.get_policy()
            before = {name: param.detach().clone() for name, param in actor.named_parameters()
                      if name.startswith("mlp.")}
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            changes = {name: float((param.detach() - before[name]).abs().max())
                       for name, param in actor.named_parameters() if name in before}
            if not changes or not any(change > 0 for change in changes.values()):
                raise RuntimeError("PPO did not change any actor MLP weight")
            if not all(bool(torch.isfinite(param).all()) for param in actor.parameters()):
                raise RuntimeError("Actor contains non-finite parameters")
            control_steps = args.iterations * agent_cfg.num_steps_per_env
            if env.common_step_counter != control_steps:
                raise RuntimeError("Actual training control-step count differs from requested run")
            report["training"] = {"control_steps_per_environment": control_steps,
                                  "samples": control_steps * args.num_envs,
                                  "actor_mlp_parameter_max_abs_changes": changes}
            report["gpu_training_completed"] = args.device == "cuda:0"
        report["training"]["samples_per_second"] = report["training"]["samples"] / report["timings_seconds"]["training"]

        with phase(report, "export", synchronize):
            checkpoint = output / f"model_{args.iterations}.pt"
            # The velocity save hook's legacy logger_type access is incompatible
            # with RSL-RL 5 when no logging writer is used. Its official base
            # persists the identical full state; export explicitly below.
            MjlabOnPolicyRunner.save(runner, str(checkpoint), infos={
                "artifact_status": LABEL, "not_for_hardware": True,
                "task": TASK, "training_iterations_completed": args.iterations})
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
            if not required.issubset(saved):
                raise RuntimeError("Checkpoint is missing full training state")
            del saved
            # Buffers updated inside learn's inference mode must reload in that mode.
            with torch.inference_mode():
                runner.load(str(checkpoint), map_location=args.device)
            runner.export_policy_to_onnx(str(output), "policy.onnx")
            metadata = get_base_metadata(env, run_path=output.name)
            metadata.update(artifact_status=LABEL, not_for_hardware="true", task=TASK,
                            source_commit=report["source"]["commit"] or "unavailable",
                            training_device=args.device)
            for model_path in output.glob("*.onnx"):
                attach_metadata_to_onnx(str(model_path), metadata)
            model = onnx.load(output / "policy.onnx")
            onnx.checker.check_model(model)
            dims = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
            if dims(model.graph.input[0]) != [1, 61] or dims(model.graph.output[0]) != [1, 14]:
                raise RuntimeError("Official export must preserve the 61 -> 14 contract")
            evaluator = ReferenceEvaluator(model)
            with torch.inference_mode():
                obs = wrapped.get_observations()[:1]
                values = obs["actor"].detach().cpu().numpy().astype(np.float32)
                exported = evaluator.run(None, {model.graph.input[0].name: values})[0]
                expected = runner.get_inference_policy(device=args.device)(obs).detach().cpu().numpy()
                np.testing.assert_allclose(exported, expected, rtol=1e-4, atol=1e-5)
            report["export"] = {"checkpoint": checkpoint.name, "checkpoint_full_training_state": True,
                                "onnx": "policy.onnx", "checker": "PASS", "normalization_included": True,
                                "input_shape": [1, 61], "output_shape": [1, 14],
                                "real_training_observation_max_abs_error": float(np.abs(exported - expected).max())}
    finally:
        wrapped.close()
    del runner, actor, wrapped, env, before
    if args.device == "cuda:0":
        torch.cuda.empty_cache()

    with phase(report, "play_environment_bootstrap", synchronize):
        play_cfg = load_env_cfg(TASK, play=True)
        play_cfg.scene.num_envs, play_cfg.seed = 1, args.seed
        play_env = ManagerBasedRlEnv(play_cfg, device=args.device)
        play = RslRlVecEnvWrapper(play_env, clip_actions=agent_cfg.clip_actions)
    try:
        with phase(report, "rollout", synchronize), torch.inference_mode():
            play_runner = runner_cls(play, asdict(agent_cfg), device=args.device)
            play_runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
            # Checkpoint restoration loads the training step counter. Measure a delta.
            start_counter = play_env.common_step_counter
            obs, _ = play.reset()
            initial_qpos = play_env.sim.data.qpos.detach().clone()
            infer = play_runner.get_inference_policy(device=args.device)
            robot = play_env.scene["robot"]
            body_ids, _ = robot.find_bodies("trunk_base")
            if len(body_ids) != 1:
                raise RuntimeError("Cannot identify a unique trunk body for rollout statistics")
            heights, tilts, resets, max_error, qpos_change = [], [], 0, 0.0, 0.0
            for _ in range(args.rollout_steps):
                values = obs["actor"].detach().cpu().numpy().astype(np.float32)
                actions = evaluator.run(None, {model.graph.input[0].name: values})[0]
                expected = infer(obs).detach().cpu().numpy()
                np.testing.assert_allclose(actions, expected, rtol=1e-4, atol=1e-5)
                if actions.shape != (1, 14) or not np.isfinite(actions).all():
                    raise RuntimeError("Non-finite or incorrect ONNX action")
                max_error = max(max_error, float(np.abs(actions - expected).max()))
                obs, reward, done, _ = play.step(torch.from_numpy(actions).to(args.device))
                tensors = [reward, play_env.sim.data.qpos, play_env.sim.data.qvel, *obs.values()]
                if not all(bool(torch.isfinite(value).all()) for value in tensors):
                    raise RuntimeError("Non-finite physical rollout state")
                quat = robot.data.body_link_quat_w[0, body_ids[0]].detach().cpu().numpy()
                cosine = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
                tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
                heights.append(float(robot.data.body_link_pos_w[0, body_ids[0], 2]))
                resets += int(done.sum())
                qpos_change = max(qpos_change, float((play_env.sim.data.qpos - initial_qpos).abs().max()))
            if play_env.common_step_counter - start_counter != args.rollout_steps or qpos_change <= 0:
                raise RuntimeError("Fresh ONNX policy did not advance the physical simulation")
            report["rollout"] = {"fresh_play_environment": True, "num_envs": 1,
                                 "control_steps": args.rollout_steps,
                                 "simulated_seconds": args.rollout_steps * play_env.step_dt,
                                 "onnx_drives_physics": True, "finite": True, "episode_resets": resets,
                                 "qpos_max_abs_change": qpos_change, "pytorch_max_abs_error": max_error,
                                 "trunk_height_m": {"min": min(heights), "max": max(heights), "mean": sum(heights) / len(heights)},
                                 "trunk_tilt_degrees": {"min": min(tilts), "max": max(tilts), "mean": sum(tilts) / len(tilts)},
                                 "statistics_scope": "Post-step state after any automatic reset; terminal minima may not be observed",
                                 "stable_gait_verified": False, "hardware_verified": False}
    finally:
        play.close()
    if args.device == "cuda:0":
        report["execution"]["peak_allocated_memory_bytes"] = torch.cuda.max_memory_allocated(0)
        report["execution"]["peak_reserved_memory_bytes"] = torch.cuda.max_memory_reserved(0)
    report.update(status="PASS", phase="complete")


def manifest(output: Path, report: dict) -> dict:
    return {"artifact_status": LABEL, "status": report["status"], "not_for_hardware": True,
            "source": report.get("source"), "execution": report.get("execution"),
            "files": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--source-commit", help="Pinned full Git SHA for an archive without .git")
    args = parser.parse_args(argv)
    if min(args.num_envs, args.iterations, args.rollout_steps) < 1:
        parser.error("num-envs, iterations and rollout-steps must be positive")
    root = official_root(Path.cwd())
    output = root / "logs/rsl_rl/professional_gpu_pipeline" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "requested_device": args.device, "artifact_status": LABEL,
              "not_for_hardware": True, "hardware_verified": False, "stable_gait_verified": False,
              "gpu_training_completed": False}
    started = time.perf_counter()
    print(f"[pipeline] artifacts: {output}", flush=True)
    with (output / "pipeline.log").open("w") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                run(args, root, output, report)
            except Exception as exc:
                report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
    report["timings_seconds"]["total"] = time.perf_counter() - started
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest(output, report), indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output / "result.json"),
                      "gpu_training_completed": report.get("gpu_training_completed", False)}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
