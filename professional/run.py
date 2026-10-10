#!/usr/bin/env python3
"""Pinned official training workspace and explicit, bounded experiment commands."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "professional/upstream.lock.json"
VENDOR = ROOT / "professional/upstream/microduck_rl"
PREPARED = ROOT / "professional/runs/official"
TASKS = {
    "walk": "Mjlab-Velocity-Flat-MicroDuck",
    "walk-backlash": "Mjlab-Velocity-Flat-Backlash-MicroDuck",
    "recovery": "Mjlab-VelStand-Flat-MicroDuck",
    "standup": "Mjlab-StandUp-Flat-MicroDuck",
    "sitstand": "Mjlab-SitStand-Flat-MicroDuck",
    "kick": "Mjlab-BallKick-Flat-MicroDuck",
}
PROFILES = {"smoke": (64, 5), "baseline": (1024, 4000)}


def verify_source(vendor: Path = VENDOR) -> dict:
    lock = json.loads(LOCK.read_text())
    expected = {item["path"] for item in lock["files"]}
    actual = {
        path.relative_to(vendor).as_posix() for path in vendor.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != expected:
        raise ValueError(f"官方源码文件集变化: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for item in lock["files"]:
        path = vendor / item["path"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"官方源码校验失败: {item['path']}")
    return {"source_commit": lock["commit"], "files_verified": len(expected),
            "policy_contract": lock["policy_contract"],
            "gpu_training": "not_run", "hardware": "not_verified"}


def prepare_workspace() -> Path:
    """HF's packager uses the Git root, so give upstream its own exact checkout."""
    verify_source()
    lock = json.loads(LOCK.read_text())
    sha = lock["commit"]
    if PREPARED.exists():
        current = subprocess.check_output(["git", "-C", str(PREPARED), "rev-parse", "HEAD"], text=True).strip()
        if current != sha:
            raise ValueError("已有训练工作目录版本不同，请保留它并另建目录，不能静默覆盖。")
        dirty = subprocess.check_output(["git", "-C", str(PREPARED), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise ValueError("训练工作目录存在改动；请先记录实验改动，不能覆盖。")
        return PREPARED
    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    # A fresh clone contains the vendored files but not necessarily the original
    # upstream commit object. Use the local object when available, otherwise
    # fetch that exact public upstream commit (never the mutable develop tip).
    has_local_commit = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{sha}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    source = str(ROOT) if has_local_commit else lock["repository"]
    # Publish the directory only after checkout succeeds. A failed download
    # cleans up only this call's temporary files and leaves retry possible.
    with tempfile.TemporaryDirectory(prefix="official-prepare-", dir=PREPARED.parent) as temporary:
        staged = Path(temporary) / "source"
        subprocess.run(["git", "init", "--quiet", str(staged)], check=True)
        subprocess.run(["git", "-C", str(staged), "fetch", "--quiet", "--no-tags", "--depth=1", source, sha], check=True)
        subprocess.run(["git", "-C", str(staged), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
        staged.rename(PREPARED)
    return PREPARED


def training_command(args: argparse.Namespace) -> list[str]:
    envs, iterations = PROFILES[args.profile]
    envs = args.num_envs if args.num_envs is not None else envs
    iterations = args.iterations if args.iterations is not None else iterations
    if not 1 <= envs <= 16384 or not 1 <= iterations <= 50000:
        raise ValueError("并行环境须为1..16384，迭代须为1..50000。")
    command = ["uv", "run", "--frozen", "train", TASKS[args.task],
               "--env.scene.num-envs", str(envs), "--agent.max-iterations", str(iterations),
               "--env.seed", str(args.seed), "--agent.seed", str(args.seed),
               "--agent.logger", "tensorboard", "--agent.upload-model", "False",
               "--agent.save-interval", "5" if args.profile == "smoke" else "100",
               "--agent.run-name", f"professional-{args.task}-{args.profile}-seed{args.seed}"]
    if args.backend == "hf":
        command += ["--hf-jobs", "--flavor", args.flavor, "--timeout", f"{args.timeout_minutes}m", "--no-wandb"]
        if args.namespace:
            command += ["--namespace", args.namespace]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="专业训练入口；默认只生成命令，不启动训练或云任务。")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify", help="校验固定版本的完整官方源码")
    sub.add_parser("prepare", help="建立独立官方源码工作目录，避免云端打包教程仓库")
    sub.add_parser("cpu-check", help="用独立官方环境做3步CPU物理预检，不训练或连接硬件")
    train = sub.add_parser("train", help="生成或执行固定配置的训练命令")
    train.add_argument("--task", choices=TASKS, default="walk")
    train.add_argument("--profile", choices=PROFILES, default="smoke")
    train.add_argument("--backend", choices=("local", "hf"), default="local")
    train.add_argument("--num-envs", type=int)
    train.add_argument("--iterations", type=int)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--flavor", choices=("l4x1", "a10g-large", "a100-large"), default="l4x1")
    train.add_argument("--timeout-minutes", type=int, default=60)
    train.add_argument("--namespace")
    train.add_argument("--execute", action="store_true", help="实际启动；HF后端会产生云计算费用")
    args = parser.parse_args()
    try:
        if args.action == "verify":
            print(json.dumps(verify_source(), ensure_ascii=False, indent=2))
        elif args.action == "prepare":
            print(prepare_workspace())
        elif args.action == "cpu-check":
            prepare_workspace()
            python = PREPARED / ".venv/bin/python"
            if not python.exists():
                raise ValueError("先在prepare输出目录运行 uv sync --frozen。")
            return subprocess.run([str(python), str(ROOT / "professional/cpu_check.py")], cwd=PREPARED).returncode
        else:
            verify_source()
            if not 1 <= args.timeout_minutes <= 720:
                raise ValueError("云任务超时须为1..720分钟。")
            command = training_command(args)
            print(f"工作目录: {PREPARED}\n命令: {shlex.join(command)}", flush=True)
            if not args.execute:
                print("仅生成命令。先 prepare，再在工作目录 uv sync --frozen；无GPU时可选择HF。")
                return 0
            prepare_workspace()
            if not (PREPARED / ".venv/bin/python").exists():
                raise ValueError("先运行 prepare，并在输出目录执行 uv sync --frozen 安装独立环境。")
            if args.backend == "local":
                subprocess.run([str(PREPARED / ".venv/bin/python"), "-c",
                                "import torch; assert torch.cuda.is_available(), '本地专业训练需要CUDA GPU'"], check=True)
            elif not args.namespace:
                raise ValueError("执行HF任务时必须指定 --namespace，明确使用哪个账户计费。")
            return subprocess.run(command, cwd=PREPARED).returncode
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
