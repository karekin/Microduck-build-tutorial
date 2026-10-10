"""Three physical steps of the pinned official task, with no training or hardware."""
from __future__ import annotations

from datetime import datetime, timezone
import contextlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "professional/runs" / datetime.now(timezone.utc).strftime("cpu-check-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "device": "cpu", "gpu_training": "not_run", "hardware_verified": False,
              "source_commit": json.loads((ROOT / "professional/upstream.lock.json").read_text())["commit"]}
    try:
        with (output / "environment.log").open("w") as log, contextlib.redirect_stdout(log):
            import torch
            import mjlab  # Registers the prepared official package, not tutorial tasks.
            from mjlab.tasks.registry import load_env_cfg
            from mjlab.envs import ManagerBasedRlEnv
            torch.set_num_threads(1)
            task = "Mjlab-Velocity-Flat-MicroDuck"
            cfg = load_env_cfg(task, play=True)
            cfg.scene.num_envs = 1
            cfg.seed = 42
            torch.manual_seed(42)
            env = ManagerBasedRlEnv(cfg, device="cpu")
            try:
                env.reset()
                for _ in range(3):
                    obs, rewards, _, _, _ = env.step(torch.zeros(1, env.action_manager.total_action_dim))
                    if not torch.isfinite(rewards).all() or not all(torch.isfinite(x).all() for x in obs.values()):
                        raise ValueError("官方任务产生非有限观测或奖励。")
                if list(obs["actor"].shape) != [1, 61] or env.action_manager.total_action_dim != 14:
                    raise ValueError("官方61输入/14动作接口不匹配。")
                report.update(status="PASS", task=task, actor_shape=list(obs["actor"].shape),
                              critic_shape=list(obs["critic"].shape), actions=14, physical_control_steps=3,
                              cuda_available=torch.cuda.is_available())
            finally:
                env.close()
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output / "result.json")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
