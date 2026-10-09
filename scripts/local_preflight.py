#!/usr/bin/env python3
"""Check the local simulation and optional training CPU setup. No hardware I/O."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="Also test the installed training environment on CPU")
    parser.add_argument("--training-update", action="store_true", help="Also perform one PPO update and export a NOT_FOR_HARDWARE demo policy")
    args = parser.parse_args()
    directory = ROOT / ".omx/artifacts/preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    checks = []

    def run(label, command, cwd=ROOT):
        print(f"Checking {label}...", flush=True)
        result = subprocess.run([str(v) for v in command], cwd=cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = directory / f"{label}.log"
        output.write_text(result.stdout, encoding="utf-8")
        checks.append({"check": label, "passed": result.returncode == 0,
                       "exit_code": result.returncode, "log": str(output.relative_to(ROOT))})
        if result.returncode:
            print(result.stdout[-5000:], file=sys.stderr)
        return result.returncode == 0

    runtime = ROOT / "microduck/.venv/bin/python"
    training = ROOT / "mjlab_microduck/.venv/bin/python"
    if not runtime.exists() or ((args.training or args.training_update) and not training.exists()):
        parser.exit(2, "Install the virtual environments first; see docs/复刻前软件运行指南.md\n")
    run("deployment-tests", [runtime, "-m", "unittest", "discover", "-s", "microduck/tests", "-v"])
    simulation = directory / "standing.jsonl"
    if run("standing-10s", [runtime, ROOT / "microduck/src/sim/sim_main.py",
                           "--headless", "--walk", "--ticks", "500", "--log", simulation]):
        rows = [json.loads(line) for line in simulation.read_text().splitlines()]
        ticks = [row for row in rows if row["type"] == "tick"]
        summary = rows[-1]
        stable = (len(ticks) == 500 and summary["requested_ticks_completed"]
                  and summary["below_training_height_threshold_ticks"] == 0
                  and summary["active_move_ticks"] > 0)
        checks.append({"check": "standing-stability", "passed": stable, "metrics": summary})
    if args.training:
        run("training-tests", [training, "-m", "unittest", "discover", "-s", "tests", "-v"],
            ROOT / "mjlab_microduck")
        smoke = '''import json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
env = ManagerBasedRlEnv(make_microduck_velocity_env_cfg(play=True), device="cpu")
try:
    obs, _ = env.reset()
    assert all(torch.isfinite(v).all() for v in obs.values())
    for _ in range(3):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 14)))
        assert all(torch.isfinite(v).all() for v in obs.values())
        assert torch.isfinite(reward).all()
        assert not terminated.any() and not truncated.any()
    print(json.dumps({"device":"cpu", "cuda_available":torch.cuda.is_available(),
                      "steps":3, "shapes":{k:list(v.shape) for k,v in obs.items()}}))
finally:
    env.close()
'''
        run("training-cpu-smoke", [training, "-c", smoke])
    if args.training_update:
        run("training-update-export", [training, ROOT / "mjlab_microduck/scripts/cpu_training_smoke.py",
                                       "--output-dir", directory / "UNTRAINED_DEMO_NOT_FOR_HARDWARE"])
    report = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
              "scope": "local software / simulated standing only",
              "hardware_verified": False, "walking_stability_verified": False,
              "policy_sha256": hashlib.sha256((ROOT / "microduck/src/agents/walk.onnx").read_bytes()).hexdigest(),
              "checks": checks}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f'{report["status"]}: {directory / "report.json"}')
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
