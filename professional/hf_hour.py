"""One bounded HF L4 job for the complete GPU training/export/physics chain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
FLAVOR = "l4x1"
MAX_SECONDS = 3600
BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates xz-utils >/dev/null
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
mkdir -p /work && cd /work
tar -xzf "/src/$SRC_ARCHIVE"
uv sync --frozen --python 3.12 --no-progress
set +e
uv run --frozen python gpu_pipeline.py --device cuda:0 --num-envs 64 --iterations 5 --rollout-steps 48 --source-commit "$SOURCE_COMMIT"
PIPELINE_RC=$?
set -e
# Final artifact upload also preserves failed reports, without uploading env or credentials.
uv run --frozen python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(
    repo_id=os.environ['RESULT_REPO'], repo_type='model',
    folder_path='logs/rsl_rl/professional_gpu_pipeline', path_in_repo='runs',
    allow_patterns=['**/model_*.pt','**/*.onnx','**/result.json','**/manifest.json',
                    '**/params/*.yaml','**/params/*.json','**/pipeline.log'],
    commit_message='Preserve bounded GPU pipeline artifacts and validation results')
print('[job] checkpoint, ONNX and validation artifacts uploaded')
PY
exit "$PIPELINE_RC"
"""


def build_archive(output: Path) -> tuple[Path, dict]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("professional_workflow", ROOT / "professional/run.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    workflow.verify_source()
    lock = json.loads((ROOT / "professional/upstream.lock.json").read_text())
    source = ROOT / lock["vendor_directory"]
    archive = output / "source.tar.gz"
    pipeline = ROOT / "professional/gpu_pipeline.py"
    with tarfile.open(archive, "w:gz") as tar:
        for entry in lock["files"]:
            tar.add(source / entry["path"], arcname=entry["path"], recursive=False)
        tar.add(pipeline, arcname="gpu_pipeline.py", recursive=False)
    return archive, {"official_commit": lock["commit"],
                     "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                     "pipeline_sha256": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
                     "official_files": len(lock["files"]), "archive_bytes": archive.stat().st_size}


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def hourly_price(hardware) -> Decimal:
    if hardware.name != FLAVOR or hardware.unit_label != "minute":
        raise ValueError("Unexpected HF hardware or billing period; refusing to submit")
    hourly = Decimal(hardware.unit_cost_micro_usd) * 60 / Decimal(1000000)
    if hourly > Decimal("0.80"):
        raise ValueError("Current L4 hourly price exceeds the authorized $0.80 limit")
    return hourly


def main():
    parser = argparse.ArgumentParser(description="最多一小时L4任务；默认只准备，不提交。")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--receipt", type=Path, help="查询已有任务状态，不重复提交")
    parser.add_argument("--download", action="store_true", help="回收已有任务的全部模型结果")
    args = parser.parse_args()
    from huggingface_hub import HfApi, Volume, get_token, snapshot_download
    api = HfApi()
    if args.receipt:
        state = json.loads(args.receipt.read_text())
        job = api.inspect_job(job_id=state["job_id"], namespace=state["namespace"])
        state["last_status"] = job.status.stage
        save(args.receipt, state)
        print(json.dumps({"job_id": job.id, "status": job.status.stage, "url": state["job_url"]}, indent=2))
        if args.download:
            folder = args.receipt.parent / "downloaded"
            snapshot_download(repo_id=state["result_repo"], repo_type="model", local_dir=folder)
            reports = list(folder.rglob("result.json"))
            if len(reports) != 1:
                raise RuntimeError("Expected one immutable pipeline result")
            report = json.loads(reports[0].read_text())
            manifest = json.loads((reports[0].parent / "manifest.json").read_text())
            required = {"result.json", "policy.onnx", report.get("export", {}).get("checkpoint", "missing_checkpoint")}
            verified = set()
            for item in manifest["files"]:
                path = (reports[0].parent / item["path"]).resolve()
                if not path.is_relative_to(reports[0].parent.resolve()):
                    raise RuntimeError("Artifact manifest contains an unsafe path")
                # Optional training byproducts are not part of our upload allowlist.
                if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"Artifact hash mismatch: {item['path']}")
                if path.is_file():
                    verified.add(item["path"])
            if report.get("status") == "PASS" and not required.issubset(verified):
                raise RuntimeError("Successful result is missing a required checkpoint, ONNX or report")
            state["artifact_download"] = str(folder)
            state["gpu_chain_verified"] = (report.get("status") == "PASS"
                and report.get("gpu_training_completed") is True
                and report.get("source", {}).get("commit") == state["source"]["official_commit"])
            save(args.receipt, state)
            print(json.dumps({"artifact_download": str(folder), "gpu_chain_verified": state["gpu_chain_verified"]}, indent=2))
        return
    output = ROOT / "professional/runs" / datetime.now(timezone.utc).strftime("hf-hour-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    archive, source = build_archive(output)
    state = {"run_id": uuid.uuid4().hex, "flavor": FLAVOR, "timeout_seconds": MAX_SECONDS,
             "maximum_compute_usd": 0.80, "source": source, "submitted": False}
    receipt = output / "receipt.json"
    save(receipt, state)
    print(json.dumps({"receipt": str(receipt), "archive_bytes": source["archive_bytes"],
                      "flavor": FLAVOR, "timeout_seconds": MAX_SECONDS, "maximum_compute_usd": 0.80}, indent=2), flush=True)
    if not args.submit:
        return
    if not get_token():
        raise RuntimeError("Hugging Face CLI login is required; no job was submitted")
    account = api.whoami()
    namespace = account["name"]
    suffix = state["run_id"][:12]
    src_repo = f"{namespace}/microduck-hour-source-{suffix}"
    result_repo = f"{namespace}/microduck-hour-result-{suffix}"
    hardware = next(h for h in api.list_jobs_hardware() if h.name == FLAVOR)
    price = hourly_price(hardware)
    state.update(namespace=namespace, source_repo=src_repo, result_repo=result_repo,
                 api_hourly_price_usd=str(price),
                 phase="creating_private_artifact_repositories")
    save(receipt, state)
    api.create_repo(src_repo, repo_type="dataset", private=True, exist_ok=False)
    api.upload_file(path_or_fileobj=archive, path_in_repo="source.tar.gz", repo_id=src_repo, repo_type="dataset")
    api.create_repo(result_repo, repo_type="model", private=True, exist_ok=False)
    state["phase"] = "submission_in_progress_do_not_resubmit"
    save(receipt, state)
    # One submission only. An uncertain response must be recovered via list_jobs,
    # never retried blindly, to preserve the user's cumulative one-hour limit.
    job = api.run_job(image=IMAGE, command=["bash", "-c", BOOTSTRAP],
                     flavor=FLAVOR, timeout=MAX_SECONDS, namespace=namespace,
                     env={"SRC_ARCHIVE": "source.tar.gz", "SOURCE_COMMIT": source["official_commit"],
                          "RESULT_REPO": result_repo, "MICRODUCK_IN_HF_JOB": "1", "MICRODUCK_RUN_ID": state["run_id"]},
                     secrets={"HF_TOKEN": get_token()},
                     labels={"project": "microduck", "run_id": state["run_id"]},
                     volumes=[Volume(type="dataset", source=src_repo, mount_path="/src", read_only=True)])
    state.update(submitted=True, job_id=job.id, job_url=job.url, phase="submitted", last_status=job.status.stage)
    save(receipt, state)
    print(json.dumps({"receipt": str(receipt), "job_id": job.id, "job_url": job.url,
                      "result_repo": result_repo, "timeout_seconds": MAX_SECONDS}, indent=2))


if __name__ == "__main__":
    main()
