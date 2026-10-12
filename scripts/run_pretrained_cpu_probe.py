#!/usr/bin/env python3
"""Bounded REAL pretrained structural/calibration probe, not a capability benchmark.

Downloads the two audited immutable checkpoints on a connected >=12 GiB CPU runner.
Only new interfaces are trained on short real WikiText TRAIN chunks. No synthetic
parents, teacher outputs, or parent weights are uploaded. Full benchmarks stay NOT RUN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import HfApi
from prepare_parents import acquire
from safetensors.torch import load_file

from equylaptatemps.artifacts import save_artifact, write_json
from equylaptatemps.audit import frozen_fingerprint, hash_state, module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, load_benchmarks, text_chunks
from equylaptatemps.evaluation import greedy_answer
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.references import save_reference
from equylaptatemps.sectors import sector_inventory, trace_child
from equylaptatemps.training import TrainingOptions, fit_encoded_seams

PINS = {
    "state-spaces/mamba-130m-hf": "1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
    "Qwen/Qwen2.5-Math-1.5B": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
}


def publish(report: dict, pr: int):
    # Public evidence only: no model/interface tensors, secrets, or raw corpus files in comments.
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    body = (
        "<!-- heterogeneous-pretrained-cpu-probe -->\n"
        "REAL pretrained structural/calibration probe; benchmarks and inheritance NOT RUN.\n"
        "SHA-256: `"
        + hashlib.sha256(payload.encode()).hexdigest()
        + "`\n```json\n"
        + payload.rstrip()
        + "\n```\n"
    )
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/issues/{pr}/comments",
        data=json.dumps({"body": body}).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print("Probe evidence posted:", response.status, flush=True)


def execute(output: Path, report: dict):
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    report["environment"] = {
        "ram_bytes": ram,
        "disk_free_bytes": shutil.disk_usage(output).free,
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "dtype": "float32",
        "quantization": "none",
        "threads": 2,
    }
    if ram < 12 * 1024**3 or shutil.disk_usage(output).free < 12 * 1024**3:
        raise RuntimeError("Probe requires >=12 GiB RAM and free disk before any model download")
    torch.set_num_threads(2)
    print("Verifying licenses and immutable source revisions", flush=True)
    upstream = HfApi().model_info("state-spaces/mamba-130m")
    original_card = upstream.card_data.to_dict() if upstream.card_data else {}
    upstream_license = original_card.get("license")
    report["mamba_license"] = {
        "hf_conversion_license_tag": None,
        "upstream_model_id": upstream.id,
        "upstream_revision": upstream.sha,
        "upstream_license_tag": upstream_license,
        "limitation": (
            "HF conversion omits a license tag/document. "
            "Original official checkpoint supplies lineage license evidence. "
            "No parent weights redistributed."
        ),
    }
    if upstream_license != "apache-2.0":
        raise RuntimeError(
            "Original official Mamba checkpoint license could not be verified as Apache-2.0"
        )
    q_info = HfApi().model_info("Qwen/Qwen2.5-Math-1.5B", revision=PINS["Qwen/Qwen2.5-Math-1.5B"])
    if not q_info.card_data or q_info.card_data.to_dict().get("license") != "apache-2.0":
        raise RuntimeError("Qwen model license could not be verified")
    specs = [acquire(model_id, output / "models", revision) for model_id, revision in PINS.items()]
    recipe = Recipe(specs[0], specs[1], LayerSpan(11, 13), LayerSpan(13, 15), seed=42)
    write_json(output / "recipe.json", recipe.to_dict())
    report["phase"] = "loading_real_weights"
    write_json(output / "results.json", report)
    print("Loading real parents at common float32 precision", flush=True)
    parents = load_parents(recipe, dtype=torch.float32)
    parents.recipient.requires_grad_(False)
    parents.donor.requires_grad_(False)
    parent_hashes = {
        "recipient": module_fingerprint(parents.recipient),
        "donor": module_fingerprint(parents.donor),
    }
    report["parents"] = {
        "recipient": {
            "id": specs[0].model_id,
            "revision": specs[0].revision,
            "parameters": sum(p.numel() for p in parents.recipient.parameters()),
            "weights_loaded": True,
        },
        "donor": {
            "id": specs[1].model_id,
            "revision": specs[1].revision,
            "parameters": sum(p.numel() for p in parents.donor.parameters()),
            "weights_loaded": True,
        },
    }
    report["snapshot_evidence"] = {
        role: {
            "revision_evidence": data["revision_evidence"],
            "snapshot_files_sha256": data["snapshot_files_sha256"],
        }
        for role, data in parents.provenance["snapshots"].items()
    }
    child = child_from_parents(recipe, parents.recipient, parents.donor)
    report["structure"] = child.structure_report()
    probe_ids = torch.tensor(
        [parents.tokenizer.encode("The study of mathematics", add_special_tokens=False)]
    )
    with torch.inference_mode(), trace_child(child) as calls:
        initial_logits = child(probe_ids).logits
    if not torch.isfinite(initial_logits).all() or any(
        n != 1 for role in calls.values() for n in role
    ):
        raise RuntimeError(
            "Real child did not execute every intended pretrained block exactly once"
        )
    report["execution_trace"] = calls
    report["initial_forward"] = {
        "status": "PASS",
        "finite_logits": True,
        "shape": list(initial_logits.shape),
    }
    report["sector_inventory"] = {
        "recipient": sector_inventory(parents.recipient, specs[0], [recipe.recipient_cut]),
        "donor": sector_inventory(parents.donor, specs[1], [recipe.donor_sector]),
    }
    # Full tensor-shape inventory stays in the artifact; comments carry its hash.
    write_json(output / "sector-inventory.json", report.pop("sector_inventory"))
    report["sector_inventory_sha256"] = hashlib.sha256(
        (output / "sector-inventory.json").read_bytes()
    ).hexdigest()
    report["phase"] = "real_text_calibration"
    write_json(output / "results.json", report)
    print("Preparing pinned real WikiText train/development data", flush=True)
    acquire_benchmarks(output / "datasets")
    manifest, data = load_benchmarks(output / "datasets")
    report["dataset_sources"] = {
        name: {
            "repository": source["repository"],
            "revision": source["revision"],
            "license": source["license"],
            "files": {split: item["sha256"] for split, item in source["local_files"].items()},
        }
        for name, source in manifest["sources"].items()
    }
    train_chunks = text_chunks(data["wikitext2"]["train"], characters=64, limit=4)
    dev_chunks = text_chunks(data["wikitext2"]["validation"], characters=64, limit=4)
    if set(train_chunks) & set(dev_chunks):
        raise RuntimeError("Real text calibration/development chunks overlap")
    encoded = encode_text(train_chunks, parents.tokenizer, 64)
    heldout = encode_text(dev_chunks, parents.tokenizer, 64)
    frozen_before = frozen_fingerprint(child)
    print("Calibrating only dense interfaces for 8 steps on actual WikiText train text", flush=True)
    calibration = fit_encoded_seams(
        child,
        parents.tokenizer,
        encoded,
        TrainingOptions(
            steps=8, batch_size=1, learning_rate=0.001, max_length=64, gradient_checkpointing=True
        ),
        heldout=heldout,
        journal=output / "calibration.jsonl",
    )
    report["calibration"] = {
        "status": "PASS",
        "kind": "measured",
        "dataset": "WikiText-2 TRAIN",
        "steps": 8,
        "train_chunks": 4,
        "dev_chunks": 4,
        "characters_per_chunk": 64,
        "objective": calibration["objective"],
        "optimizer": calibration["optimizer"],
        "seed": 42,
        "learning_rate": 0.001,
        "elapsed_seconds": calibration["elapsed_seconds"],
        "initial_loss": calibration["initial_train"]["groups"]["overall"]["token_nll"],
        "final_loss": calibration["final_train"]["groups"]["overall"]["token_nll"],
        "development_loss": calibration["validation"]["groups"]["overall"]["token_nll"],
        "warning": "Tiny train/development diagnostic, NOT a WikiText/GSM8K performance benchmark",
    }
    write_json(output / "calibration-report.json", calibration)
    save_artifact(
        child,
        parents.tokenizer,
        output / "private-child",
        parents.provenance,
        training_status="seam_calibrated",
        metrics=calibration,
    )
    save_reference(child, output / "reference-child", parents.provenance)
    report["phase"] = "fresh_process_reload"
    write_json(output / "results.json", report)
    with torch.inference_mode():
        expected_logits = child(probe_ids).logits
    write_json(output / "probe-input.json", probe_ids.tolist())
    print(
        "Verifying self-contained load in a fresh process with parent loading forbidden", flush=True
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/fresh_child_check.py",
            "--child",
            str(output / "private-child"),
            "--ids",
            str(output / "probe-input.json"),
            "--out",
            str(output / "fresh-process"),
        ],
        check=True,
        timeout=180,
    )
    actual = load_file(str(output / "fresh-process/logits.safetensors"))["logits"]
    report["fresh_process_reload"] = json.loads((output / "fresh-process/check.json").read_text())
    report["fresh_process_reload"]["identical_logits"] = bool(torch.equal(expected_logits, actual))
    report["fresh_process_reload"]["close_logits"] = bool(
        torch.allclose(expected_logits, actual, atol=1e-5, rtol=1e-5)
    )
    if not report["fresh_process_reload"]["close_logits"]:
        raise RuntimeError("Fresh-process real-child outputs changed")
    report["generation_sample"] = {
        "prompt": "The study of mathematics",
        "output": greedy_answer(
            child, parents.tokenizer, "The study of mathematics", max_new_tokens=4
        ),
        "benchmark": False,
        "quality_validated": False,
    }
    report["frozen_invariance"] = {
        "child_unchanged": frozen_fingerprint(child) == frozen_before,
        "recipient_unchanged": module_fingerprint(parents.recipient) == parent_hashes["recipient"],
        "donor_unchanged": module_fingerprint(parents.donor) == parent_hashes["donor"],
        "frozen_sha256": frozen_before,
        "graft_sha256": module_fingerprint(child.graft.layers),
        "calibrated_logits_sha256": hash_state({"logits": expected_logits}),
    }
    if not all(
        report["frozen_invariance"][key]
        for key in ("child_unchanged", "recipient_unchanged", "donor_unchanged")
    ):
        raise RuntimeError("A real pretrained tensor changed")
    report["checkpoint"] = {
        "self_contained_private_checkpoint_tested": True,
        "public_artifact": "reference-child: new interfaces + immutable parent references only",
        "parent_weights_uploaded": False,
    }
    report["status"] = "PASS"
    report["phase"] = "finished"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Use a new output directory; preserve failed-run evidence")
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "kind": "real_pretrained_structural_probe",
        "source_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "status": "NOT_RUN",
        "phase": "preflight",
        "benchmarks": {
            "gsm8k": {"status": "NOT_RUN", "score": None},
            "wikitext2": {"status": "NOT_RUN", "score": None},
        },
        "inheritance": {"status": "NOT_RUN", "claim": "none; structural/calibration evidence only"},
    }
    try:
        execute(out, report)
    except Exception as error:
        report.update(
            {
                "status": "FAIL" if report.get("parents") else "BLOCKED",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        print("Probe did not complete:", type(error).__name__, str(error), flush=True)
    report["total_elapsed_seconds"] = time.monotonic() - started
    write_json(out / "results.json", report)
    if args.post_pr:
        publish(report, args.post_pr)
    print(json.dumps({"status": report["status"], "phase": report["phase"]}), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
