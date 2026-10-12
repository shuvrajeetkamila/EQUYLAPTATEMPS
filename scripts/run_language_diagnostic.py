#!/usr/bin/env python3
"""Registered language diagnosis, matched random-sector control and fresh references."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from prepare_parents import acquire
from run_pretrained_study import publish_bundle
from safetensors.torch import load_file, save_file

from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import frozen_fingerprint, hash_state, module_fingerprint, sha256_file
from equylaptatemps.benchdata import acquire_benchmarks, load_benchmarks, text_chunks
from equylaptatemps.benchmarks import generate_tokens, observe_language
from equylaptatemps.controls import random_qwen_sector_control
from equylaptatemps.diagnostics import validate_language_diagnostic
from equylaptatemps.evidence import paired_interval, read_rows
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import Recipe
from equylaptatemps.references import load_reference, save_reference
from equylaptatemps.sectors import trace_child
from equylaptatemps.training import TrainingOptions, fit_encoded_seams


def execute(config, out, result):
    if config["dtype"] != "float32" or config["device"] != "cpu":
        raise ValueError("This registered diagnostic supports common float32 CPU only")
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    result["hardware"] = {
        "ram_bytes": ram,
        "disk_free_bytes": shutil.disk_usage(out).free,
        "gpu": False,
        "dtype": "float32",
        "torch": torch.__version__,
        "threads": 2,
    }
    if ram < 12 * 1024**3 or shutil.disk_usage(out).free < 12 * 1024**3:
        raise RuntimeError("Need >=12 GiB RAM/free disk before acquiring genuine parents")
    torch.set_num_threads(2)
    reference = Path(config["reference"])
    descriptor = json.loads((reference / "reference.json").read_text())
    if sha256_file(reference / "interfaces.safetensors") != config["reference_interfaces_sha256"]:
        raise ValueError("Published selected child interfaces changed")
    local_recipe = json.loads(json.dumps(descriptor["recipe"]))
    for role in ("recipient", "donor"):
        original = local_recipe[role]
        local_recipe[role] = asdict(
            acquire(original["model_id"], Path(config["parent_cache"]), original["revision"])
        )
    parents = load_parents(Recipe.from_dict(local_recipe), dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    child = load_reference(reference, parents)
    parent_hashes = {
        role: module_fingerprint(getattr(parents, role)) for role in ("recipient", "donor")
    }
    child_hash = frozen_fingerprint(child)
    write_json(out / "provenance.json", parents.provenance)
    write_json(out / "topology.json", child.recipe.to_dict())
    result["parents"] = {
        role: {
            "model_id": local_recipe[role]["model_id"],
            "revision": local_recipe[role]["revision"],
            "parameters": sum(p.numel() for p in getattr(parents, role).parameters()),
        }
        for role in ("recipient", "donor")
    }
    result["structure"] = child.structure_report()
    result["status"] = "NOT_RUN"
    result["phase"] = "verified_real_reference"
    write_json(out / "results.json", result)
    acquire_benchmarks(config["dataset_cache"])
    manifest, data = load_benchmarks(config["dataset_cache"])
    write_json(out / "dataset-provenance.json", manifest)
    prior = Path("reports/pretrained/run-38293921543")
    splits = json.loads((prior / "split-selection.json").read_text())
    prior_config = json.loads((prior / "registration.json").read_text())
    budget = config["training_budget"]
    for key in ("steps", "batch_size", "learning_rate", "max_length", "gradient_checkpointing"):
        if budget[key] != prior_config["calibration"][key]:
            raise ValueError("Random-sector calibration must match published selected-child budget")
    if config["seed"] != prior_config["seed"]:
        raise ValueError("Matched control must use the original interface/training seed")
    math_rows = [dict(data["gsm8k"]["train"][i], source_index=i) for i in splits["gsm_train"]]
    wiki_train = text_chunks(data["wikitext2"]["train"], characters=192, limit=8)
    if [hashlib.sha256(text.encode()).hexdigest() for text in wiki_train] != splits["wiki_windows"][
        "train"
    ]:
        raise ValueError("Matched control WikiText training content differs from original child")
    train_math, omitted = encode_math(math_rows, parents.tokenizer, budget["max_length"])
    if omitted:
        raise ValueError("Do not silently change matched calibration examples")
    train = train_math + encode_text(wiki_train, parents.tokenizer, budget["max_length"])
    benchmark = config["benchmark"]
    chunks = text_chunks(
        data["wikitext2"]["test"],
        characters=benchmark["window_characters"],
        limit=benchmark["window_offset"] + benchmark["window_count"],
    )[benchmark["window_offset"] :]

    def normalize(text):
        return " ".join(text.lower().split())

    if set(map(normalize, chunks)) & set(map(normalize, wiki_train)):
        raise ValueError("New test windows overlap matched calibration text")
    if set(chunks) & set(text_chunks(data["wikitext2"]["test"], characters=192, limit=16)):
        raise ValueError("Do not reuse previously observed test windows")
    write_json(
        out / "data-selection.json",
        {
            "math_train_indices": splits["gsm_train"],
            "wiki_train_window_count": len(wiki_train),
            "test_window_offset": benchmark["window_offset"],
            "test_window_count": len(chunks),
            "previous_test_windows_excluded": True,
        },
    )
    diagnostic_ids = torch.tensor(
        [parents.tokenizer.encode("The study of mathematics", add_special_tokens=False)]
    )
    with torch.inference_mode(), trace_child(child) as calls:
        logits = child(diagnostic_ids).logits
    if not torch.isfinite(logits).all() or any(n != 1 for values in calls.values() for n in values):
        raise RuntimeError("Published reference failed real complete-block execution")
    result["real_execution"] = {"status": "PASS", "native_calls": calls, "finite_logits": True}
    result["generation_samples"] = []
    for prompt in config["generation"]["prompts"]:
        text, tokens = generate_tokens(
            child, parents.tokenizer, prompt, max_input_tokens=512, max_new_tokens=8
        )
        result["generation_samples"].append(
            {"prompt": prompt, "output": text, "ids": tokens, "benchmark": False}
        )
    save_reference(child, out / "reference-v2", parents.provenance)
    save_file({"logits": logits}, str(out / "expected-logits.safetensors"))
    write_json(
        out / "fresh-inputs.json",
        {"ids": diagnostic_ids.tolist(), "prompts": config["generation"]["prompts"]},
    )
    result["phase"] = "matched_random_sector_calibration"
    write_json(out / "results.json", result)
    control = random_qwen_sector_control(
        child.recipe,
        parents.recipient,
        parents.donor,
        initialization_seed=config["random_sector_seed"],
    )
    if module_fingerprint(control.graft.layers) == module_fingerprint(child.graft.layers):
        raise ValueError("Matched random sector was not independently initialized")
    control_before = frozen_fingerprint(control)
    options = TrainingOptions(
        **{
            k: budget[k]
            for k in (
                "steps",
                "batch_size",
                "learning_rate",
                "max_length",
                "gradient_checkpointing",
            )
        },
        seed=config["seed"],
    )
    fit = fit_encoded_seams(
        control, parents.tokenizer, train, options, journal=out / "control-calibration.jsonl"
    )
    write_json(out / "control-calibration.json", fit)
    result["matched_control"] = {
        "status": "PASS",
        "graft_kind": "random initialization, NOT pretrained",
        "initialization_seed": config["random_sector_seed"],
        "parent_kind": "real frozen recipient; random donor control",
        "same_calibration_examples": True,
        "same_trainable_names_budget_seed": True,
        "original_child_training_file": "reports/pretrained/run-38293921543/calibration-1.json",
    }
    uncalibrated = child_from_parents(child.recipe, parents.recipient, parents.donor)
    recipe_data = child.recipe.to_dict()
    recipe_data["seed"] += 1
    random_interfaces = child_from_parents(
        Recipe.from_dict(recipe_data), parents.recipient, parents.donor
    )
    variants = {
        "recipient_parent": (parents.recipient, parents.tokenizer, "none"),
        "donor_parent": (parents.donor, parents.donor_tokenizer, "none"),
        "published_calibrated_child": (child, parents.tokenizer, "none"),
        "uncalibrated_child": (uncalibrated, parents.tokenizer, "none"),
        "random_interface": (random_interfaces, parents.tokenizer, "none"),
        "matched_random_sector": (control, parents.tokenizer, "none"),
    }
    for mode in (
        "bypass_graft",
        "bypass_prefix",
        "bypass_suffix",
        "zero_input_seam",
        "zero_output_seam",
    ):
        variants[mode] = (child, parents.tokenizer, mode)
    result["comparisons"] = {}
    result["phase"] = "fresh_heldout_language"
    for role, (model, tokenizer, mode) in variants.items():
        path = out / "observations" / f"{role}.jsonl"
        metrics = observe_language(model, tokenizer, chunks, path, max_length=512, ablation=mode)
        result["comparisons"][role] = {
            "status": "PASS",
            "kind": "measured",
            "metrics": metrics,
            "records": str(path.relative_to(out)),
            "records_sha256": sha256_file(path),
        }
        write_json(out / "results.json", result)
        print(role, metrics, flush=True)
    paired = {}
    full = read_rows(out / result["comparisons"]["published_calibrated_child"]["records"])
    for role in ("matched_random_sector", "bypass_graft", "bypass_prefix", "bypass_suffix"):
        rows = read_rows(out / result["comparisons"][role]["records"])
        paired[role] = paired_interval(rows, full, "wikitext2", seed=config["seed"])
    result["paired_language"] = paired
    result["frozen_invariance"] = {
        "status": "PASS",
        "all_original_parents_unchanged": all(
            module_fingerprint(getattr(parents, r)) == v for r, v in parent_hashes.items()
        ),
        "published_child_frozen_unchanged": frozen_fingerprint(child) == child_hash,
        "random_sector_and_recipient_control_frozen_unchanged": frozen_fingerprint(control)
        == control_before,
    }
    if not all(v for k, v in result["frozen_invariance"].items() if k != "status"):
        raise RuntimeError("A frozen tensor changed")
    child_bpb = result["comparisons"]["published_calibrated_child"]["metrics"]["bits_per_byte"]
    host_bpb = result["comparisons"]["recipient_parent"]["metrics"]["bits_per_byte"]
    enough = len(chunks) >= config["criteria"]["minimum_windows"]
    retained = enough and child_bpb <= min(
        config["criteria"]["maximum_language_bpb"],
        config["criteria"]["language_retention_ratio"] * host_bpb,
    )
    result["language_retention"] = {
        "status": "PASS" if retained else "FAIL",
        "kind": "derived",
        "child_to_recipient_bpb_ratio": child_bpb / host_bpb,
        "reason": "Registered language ceiling, not dual-capability inheritance",
    }
    beneficial = enough and paired["matched_random_sector"]["bootstrap_95"][0] > 0
    result["learned_sector_advantage"] = {
        "status": "PASS" if beneficial else "FAIL",
        "kind": "derived",
        "reason": "Registered matched-budget paired language comparison only",
    }
    result["inheritance"] = {
        "status": "NOT_RUN",
        "reason": "Math not assessed; language-only controlled diagnosis",
    }
    result["phase"] = "fresh_reference_process"
    write_json(out / "results.json", result)
    # Returning releases all model/control references before a second full-parent CPU load.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrained-language-diagnostic.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    write_json(out / "registration.json", config)
    result = {
        "schema_version": 2,
        "kind": "real_pretrained_language_diagnostic",
        "project": config["project"],
        "status": "NOT_RUN",
        "phase": "preflight",
        "source_commit": (
            os.environ.get("GITHUB_SHA")
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        ),
        "software_versions": {
            name: version(name)
            for name in ("torch", "transformers", "huggingface-hub", "safetensors")
        },
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "registration_sha256": sha256_file(out / "registration.json"),
        "math_benchmark": config["math_benchmark"],
        "reverse": config["reverse"],
    }
    write_json(out / "results.json", result)
    started = time.monotonic()
    try:
        execute(config, out, result)
        gc.collect()
        subprocess.run(
            [
                sys.executable,
                "scripts/check_reference_process.py",
                "--reference",
                str(out / "reference-v2"),
                "--parent-cache",
                config["parent_cache"],
                "--inputs",
                str(out / "fresh-inputs.json"),
                "--out",
                str(out / "fresh-reference"),
            ],
            check=True,
            timeout=240,
        )
        expected = load_file(str(out / "expected-logits.safetensors"))["logits"]
        fresh = load_file(str(out / "fresh-reference/logits.safetensors"))["logits"]
        result["fresh_reference"] = {
            "status": "PASS",
            "identical_logits": bool(torch.equal(expected, fresh)),
            "logits_sha256": hash_state({"logits": fresh}),
            "generation_equal": json.loads((out / "fresh-reference/results.json").read_text())[
                "generation_samples"
            ]
            == [
                {k: v for k, v in sample.items() if k != "benchmark"}
                for sample in result["generation_samples"]
            ],
        }
        if (
            not result["fresh_reference"]["identical_logits"]
            or not result["fresh_reference"]["generation_equal"]
        ):
            raise RuntimeError("Fresh reference reconstruction output changed")
        result["status"] = "PASS"
        result["phase"] = "complete"
        write_json(out / "results.json", result)
        result["integrity"] = validate_language_diagnostic(
            out, dataset_root=config["dataset_cache"]
        )
    except Exception as error:
        result["status"] = "BLOCKED" if result["phase"] == "preflight" else "FAIL"
        result["error"] = {"type": type(error).__name__, "reason": str(error)}
        print("Diagnostic failed", result["error"], flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "results.json", result)
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
