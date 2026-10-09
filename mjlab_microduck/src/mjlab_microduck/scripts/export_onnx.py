# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Export a trained .pt checkpoint to ONNX.

Usage:
    # Latest checkpoint of the latest run:
    uv run python src/mjlab_microduck/scripts/export_onnx.py

    # Specific checkpoint file:
    uv run python src/mjlab_microduck/scripts/export_onnx.py --checkpoint logs/rsl_rl/mjlab_microduck_velocity/2026-06-25_21-24-43/model_5000.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path

TASK = "Mjlab-Velocity-Microduck"
LOG_ROOT = Path("logs/rsl_rl/mjlab_microduck_velocity")
OUTPUT_FILENAME = "policy.onnx"


def validate_checkpoint_dimensions(path: Path, obs_dim: int = 51, action_dim: int = 14) -> None:
    """Reject another robot's weights before constructing the simulation."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    actor = checkpoint.get("actor_state_dict")
    if actor is None:
        actor = {k.removeprefix("actor."): v
                 for k, v in checkpoint.get("model_state_dict", {}).items()
                 if k.startswith("actor.")}
    weights = [(int(k.split(".")[-2]), v) for k, v in actor.items()
               if k.endswith(".weight") and v.ndim == 2
               and k.split(".")[-2].isdigit()]
    if not weights:
        raise ValueError("Checkpoint has no supported MLP actor weights")
    weights.sort(key=lambda item: item[0])
    found_obs, found_actions = weights[0][1].shape[1], weights[-1][1].shape[0]
    if (found_obs, found_actions) != (obs_dim, action_dim):
        raise ValueError(
            f"Checkpoint is for {found_obs} observations / {found_actions} actions; "
            f"this Microduck task requires {obs_dim} / {action_dim}. "
            "Use a matching 14-joint checkpoint; do not truncate or reshape weights. "
            "The shipped microduck/src/agents/walk.onnx is a separate deployment policy."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to a specific .pt file (default: latest)")
    p.add_argument("--device", type=str, default="cpu", help="Device for model loading (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = get_checkpoint_path(LOG_ROOT)

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    validate_checkpoint_dimensions(checkpoint_path)

    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner_cls = load_runner_cls(TASK)
        runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=args.device)
        output_dir = Path.cwd()
        runner.export_policy_to_onnx(str(output_dir), OUTPUT_FILENAME)
        onnx_path = output_dir / OUTPUT_FILENAME
        metadata = get_base_metadata(wrapped.unwrapped, run_path=str(checkpoint_path.parent.name))
        attach_metadata_to_onnx(str(onnx_path), metadata)
        print(f"[INFO] Exported: {onnx_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
