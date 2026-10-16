#!/usr/bin/env python3
"""Small connected-runner metadata probe. No parent weights, neural inference, or scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PARENTS = ("state-spaces/mamba-130m-hf", "Qwen/Qwen2.5-Math-1.5B")


def inspect(model_id: str) -> dict:
    start = time.monotonic()
    info = HfApi().model_info(model_id, files_metadata=True)
    card = info.card_data.to_dict() if info.card_data else {}
    path = Path(hf_hub_download(model_id, "config.json", revision=info.sha))
    raw = path.read_bytes()
    config = json.loads(raw)
    files = []
    for sibling in info.siblings:
        if sibling.rfilename.endswith((".safetensors", ".bin", ".index.json")):
            files.append(
                {
                    "name": sibling.rfilename,
                    "size_bytes": sibling.size,
                    "lfs_sha256": sibling.lfs.sha256 if sibling.lfs else None,
                }
            )
    licenses = []
    for sibling in info.siblings:
        if sibling.rfilename.upper().startswith(("LICENSE", "NOTICE")):
            license_path = Path(hf_hub_download(model_id, sibling.rfilename, revision=info.sha))
            licenses.append(
                {
                    "name": sibling.rfilename,
                    "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
                    "text": license_path.read_text(),
                }
            )
    return {
        "status": "PASS",
        "measurement_kind": "remote_metadata",
        "model_id": model_id,
        "revision": info.sha,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config": config,
        "license_tag": card.get("license"),
        "license_documents": licenses,
        "parameter_count": info.safetensors.total if info.safetensors else None,
        "parameter_count_source": "Hub safetensors metadata" if info.safetensors else "unavailable",
        "weight_files": files,
        "weights_downloaded": False,
        "inference_run": False,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    parents = []
    for model_id in PARENTS:
        try:
            parents.append(inspect(model_id))
        except Exception as error:
            parents.append(
                {
                    "status": "BLOCKED",
                    "model_id": model_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "weights_downloaded": False,
                    "inference_run": False,
                }
            )
    report = {
        "schema_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "kind": "parent_metadata_probe",
        "parents": parents,
        "source_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "environment": {
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "disk_free_bytes": shutil.disk_usage(".").free,
            "gpu_present": bool(shutil.which("nvidia-smi")),
        },
        "benchmarks": {"status": "NOT_RUN", "reason": "Metadata inspection only"},
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n")
    print(payload)
    if args.post_pr:
        repository = os.environ["GITHUB_REPOSITORY"]
        body = (
            "<!-- heterogeneous-parent-probe -->\n"
            "Connected-runner parent metadata only; no weights or benchmark scores.\n"
            "SHA-256: `"
            + hashlib.sha256((payload + "\n").encode()).hexdigest()
            + "`\n```json\n"
            + payload
            + "\n```\n"
        )
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/issues/{args.post_pr}/comments",
            data=json.dumps({"body": body}).encode(),
            method="POST",
            headers={
                "Authorization": "Bearer " + os.environ["GH_TOKEN"],
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            print("Evidence comment posted:", response.status)


if __name__ == "__main__":
    main()
