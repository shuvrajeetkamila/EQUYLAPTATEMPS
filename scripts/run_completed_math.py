#!/usr/bin/env python3
"""Genuine unchanged parents/child, longer completed math answers, native-cache oracle gated."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
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

from equylaptatemps.artifacts import save_artifact, write_json
from equylaptatemps.audit import frozen_fingerprint, module_fingerprint, sha256_file
from equylaptatemps.benchdata import acquire_benchmarks, load_benchmarks, text_chunks
from equylaptatemps.benchmarks import numeric_answer
from equylaptatemps.completion import complete_answer, final_answer_stop, summarize_completed_math
from equylaptatemps.controls import random_qwen_sector_control
from equylaptatemps.decoding import CachedDecoder, cached_generate
from equylaptatemps.evaluation import predict
from equylaptatemps.math_evidence import registered_prompt as prompt_for
from equylaptatemps.math_evidence import validate_completed_math
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import Recipe
from equylaptatemps.references import load_reference
from equylaptatemps.training import TrainingOptions, fit_encoded_seams


@torch.inference_mode()
def cache_oracle(model, tokenizer, text, config, *, ablation="none"):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    n = config["prefill_tokens"]
    steps = config["decode_tokens"]
    if len(encoded) < n + steps:
        raise ValueError("Oracle must have enough real source tokens")
    ids = torch.tensor([encoded[: n + steps]], dtype=torch.long)
    decoder = CachedDecoder(model, ablation=ablation)
    differences = []
    argmax_equal = True
    started = time.monotonic()
    prefill = decoder.prefill(ids[:, :n]).logits
    full = predict(model, {"input_ids": ids[:, :n]}, ablation).logits
    if not torch.allclose(prefill, full, atol=config["atol"], rtol=config["rtol"]):
        raise RuntimeError("Native cache prefill disagrees with cache-free oracle")
    differences.append(float((prefill - full).abs().max()))
    for index in range(n, n + steps):
        cached = decoder.step(ids[:, index : index + 1]).logits[0, -1]
        expected = predict(model, {"input_ids": ids[:, : index + 1]}, ablation).logits[0, -1]
        differences.append(float((cached - expected).abs().max()))
        argmax_equal = argmax_equal and int(cached.argmax()) == int(expected.argmax())
        if not torch.allclose(cached, expected, atol=config["atol"], rtol=config["rtol"]):
            raise RuntimeError("Incremental cache changed full-prefix model computation")
    if config["require_argmax_equal"] and not argmax_equal:
        raise RuntimeError("Native cached oracle changed greedy token decisions")
    return {
        "status": "PASS",
        "max_absolute_logit_error": max(differences),
        "argmax_equal": argmax_equal,
        "prefill_tokens": n,
        "forced_decode_tokens": steps,
        "elapsed_seconds": time.monotonic() - started,
        "source": "reported prefix from HUMAN GSM8K TRAIN question, not benchmark truncation",
    }


def execute(config, out, result):
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if ram < 12 * 1024**3 or shutil.disk_usage(out).free < 12 * 1024**3:
        raise RuntimeError("Need >=12GiB RAM/free disk before original parent loading")
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("Only CPUfloat32 native caching is registered/validated here")
    torch.set_num_threads(config["threads"])
    result["hardware"] = {
        "ram_bytes": ram,
        "disk_free_bytes": shutil.disk_usage(out).free,
        "gpu": False,
        "threads": config["threads"],
        "dtype": "float32",
    }
    reference = Path(config["reference"])
    if sha256_file(reference / "interfaces.safetensors") != config["reference_interfaces_sha256"]:
        raise ValueError("Published real child interface checkpoint changed")
    descriptor = json.loads((reference / "reference.json").read_text())
    local = json.loads(json.dumps(descriptor["recipe"]))
    for role in ("recipient", "donor"):
        spec = local[role]
        local[role] = asdict(
            acquire(spec["model_id"], Path(config["parent_cache"]), spec["revision"])
        )
    parents = load_parents(Recipe.from_dict(local), dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    child = load_reference(reference, parents)
    before = {r: module_fingerprint(getattr(parents, r)) for r in ("recipient", "donor")}
    child_before = module_fingerprint(child)
    write_json(out / "provenance.json", parents.provenance)
    write_json(out / "topology.json", child.recipe.to_dict())
    result["parents"] = {
        r: {
            "model_id": local[r]["model_id"],
            "revision": local[r]["revision"],
            "parameters": sum(p.numel() for p in getattr(parents, r).parameters()),
        }
        for r in ("recipient", "donor")
    }
    result["structure"] = child.structure_report()
    result["phase"] = "actual_parents_restored"
    write_json(out / "results.json", result)
    acquire_benchmarks(config["dataset_cache"])
    manifest, data = load_benchmarks(config["dataset_cache"])
    write_json(out / "dataset-provenance.json", manifest)
    prior = Path("reports/pretrained/run-38293921543")
    prior_splits = json.loads((prior / "split-selection.json").read_text())
    prior_config = json.loads((prior / "registration.json").read_text())
    budget = config["matched_control_budget"]
    for key in ("steps", "learning_rate", "batch_size", "max_length", "gradient_checkpointing"):
        if budget[key] != prior_config["calibration"][key]:
            raise ValueError("Matched budget drift")
    if config["calibration_seed"] != prior_config["seed"]:
        raise ValueError("Matched training seed drift")
    train = [dict(data["gsm8k"]["train"][i], source_index=i) for i in prior_splits["gsm_train"]]
    excluded_train = set(prior_splits["gsm_train"] + prior_splits["gsm_dev"])
    shot_indices = [i for i in range(len(data["gsm8k"]["train"])) if i not in excluded_train][
        : config["benchmark"]["shots"]
    ]
    fewshots = [data["gsm8k"]["train"][i] for i in shot_indices]
    order = [
        i
        for i in range(len(data["gsm8k"]["test"]))
        if i not in config["benchmark"]["exclude_previously_observed_test_ids"]
    ]
    random.Random(config["test_seed"]).shuffle(order)
    ids = order[: config["benchmark"]["sample_count"]]
    tests = [dict(data["gsm8k"]["test"][i], source_index=i) for i in ids]

    def normalize(text):
        return " ".join(text.lower().split())

    if {normalize(r["question"]) for r in tests} & {
        normalize(r["question"]) for r in train + fewshots
    }:
        raise ValueError("Calibration/fewshot/test prompt leakage")
    write_json(
        out / "data-selection.json",
        {
            "test_ids": ids,
            "fewshot_train_ids": shot_indices,
            "calibration_train_ids": prior_splits["gsm_train"],
            "prior_test_ids_excluded": True,
        },
    )
    result["phase"] = "matched_control_training"
    write_json(out / "results.json", result)
    control = random_qwen_sector_control(
        child.recipe,
        parents.recipient,
        parents.donor,
        initialization_seed=config["random_sector_seed"],
    )
    control_before = frozen_fingerprint(control)
    train_math, omitted = encode_math(train, parents.tokenizer, budget["max_length"])
    if omitted:
        raise ValueError("Do not silently change matched data")
    train_wiki = text_chunks(data["wikitext2"]["train"], characters=192, limit=8)
    if [hashlib.sha256(t.encode()).hexdigest() for t in train_wiki] != prior_splits["wiki_windows"][
        "train"
    ]:
        raise ValueError("Matched WikiText data changed")
    encoded = train_math + encode_text(train_wiki, parents.tokenizer, budget["max_length"])
    options = TrainingOptions(
        **{
            k: budget[k]
            for k in (
                "steps",
                "learning_rate",
                "batch_size",
                "max_length",
                "gradient_checkpointing",
            )
        },
        seed=config["calibration_seed"],
    )
    fit = fit_encoded_seams(
        control, parents.tokenizer, encoded, options, journal=out / "control-calibration.jsonl"
    )
    write_json(out / "control-calibration.json", fit)
    uncalibrated = child_from_parents(child.recipe, parents.recipient, parents.donor)
    random_recipe = child.recipe.to_dict()
    random_recipe["seed"] += 1
    random_interface = child_from_parents(
        Recipe.from_dict(random_recipe), parents.recipient, parents.donor
    )
    variants = {
        "recipient_parent": (parents.recipient, parents.tokenizer, "none"),
        "donor_parent": (parents.donor, parents.donor_tokenizer, "none"),
        "published_calibrated_child": (child, parents.tokenizer, "none"),
        "uncalibrated_child": (uncalibrated, parents.tokenizer, "none"),
        "random_interface": (random_interface, parents.tokenizer, "none"),
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
    variants = {name: entry for name, entry in variants.items() if name in config["variants"]}
    result["cache_oracles"] = {}
    result["comparisons"] = {}
    result["phase"] = "cache_oracle_gate"
    for name, (model, tok, mode) in variants.items():
        result["cache_oracles"][name] = cache_oracle(
            model, tok, train[0]["question"], config["cache_oracle"], ablation=mode
        )
        write_json(out / "results.json", result)
    # Also test unchanged real original blocks in the reverse structural direction.
    reverse_recipe = child.recipe.to_dict()
    reverse_recipe["recipient"], reverse_recipe["donor"] = (
        reverse_recipe["donor"],
        reverse_recipe["recipient"],
    )
    reverse_recipe["recipient_cut"], reverse_recipe["donor_sector"] = (
        reverse_recipe["donor_sector"],
        reverse_recipe["recipient_cut"],
    )
    reverse = child_from_parents(Recipe.from_dict(reverse_recipe), parents.donor, parents.recipient)
    result["reverse_cache_oracle"] = cache_oracle(
        reverse, parents.donor_tokenizer, train[0]["question"], config["cache_oracle"]
    )
    del reverse
    result["phase"] = "completed_math_generation"
    write_json(out / "results.json", result)
    for name, (model, tok, mode) in variants.items():
        path = out / "observations" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for example in tests:
            prompt = prompt_for(example["question"], fewshots)
            started = time.monotonic()
            generation = cached_generate(
                model,
                tok,
                prompt,
                max_new_tokens=config["benchmark"]["max_new_tokens"],
                max_input_tokens=config["benchmark"]["max_input_tokens"],
                ablation=mode,
                stop_when=final_answer_stop,
            )
            prediction = complete_answer(generation.text, generation.stop_reason)
            gold = numeric_answer(example["answer"], strict=True)
            if gold is None:
                raise ValueError("Invalid original GSM8K reference")
            row = {
                "id": example["source_index"],
                "split": "test",
                "question_sha256": hashlib.sha256(example["question"].encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "reference": example["answer"],
                "response": generation.text,
                "generated_ids": generation.generated_ids,
                "stop_reason": generation.stop_reason,
                "input_tokens": generation.input_tokens,
                "prediction": prediction,
                "correct": prediction == gold,
                "elapsed_seconds": time.monotonic() - started,
            }
            rows.append(row)
            with path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            print(
                name,
                example["source_index"],
                row["correct"],
                row["stop_reason"],
                len(row["generated_ids"]),
                flush=True,
            )
        result["comparisons"][name] = {
            "status": "PASS",
            "kind": "measured",
            "metrics": summarize_completed_math(rows),
            "records": str(path.relative_to(out)),
            "records_sha256": sha256_file(path),
        }
        write_json(out / "results.json", result)
    result["frozen_invariance"] = {
        "status": "PASS",
        "original_parents_unchanged": all(
            module_fingerprint(getattr(parents, r)) == v for r, v in before.items()
        ),
        "published_child_all_tensors_unchanged": module_fingerprint(child) == child_before,
        "random_control_frozen_unchanged": frozen_fingerprint(control) == control_before,
    }
    if not all(v is True for k, v in result["frozen_invariance"].items() if k != "status"):
        raise RuntimeError("A pretrained/frozen or published interface tensor changed")
    save_artifact(
        child,
        parents.tokenizer,
        out / "private-child",
        parents.provenance,
        training_status="seam_calibrated",
    )
    proof_prompt = prompt_for(tests[0]["question"], fewshots)
    expected = cached_generate(
        child, parents.tokenizer, proof_prompt, max_input_tokens=2048, max_new_tokens=16
    )
    write_json(
        out / "fresh-input.json",
        {
            "prompt": proof_prompt,
            "max_new_tokens": 16,
            "text": expected.text,
            "generated_ids": expected.generated_ids,
            "stop_reason": expected.stop_reason,
        },
    )
    result["phase"] = "fresh_cached_checkpoint"
    write_json(out / "results.json", result)
    # Locals and all full-parent/control instances are released before the fresh child-only process.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrained-completed-math-validated.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    write_json(out / "registration.json", config)
    result = {
        "schema_version": 3,
        "kind": "real_pretrained_completed_math",
        "project": config["project"],
        "status": "NOT_RUN",
        "phase": "preflight",
        "source_commit": os.environ.get("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "software_versions": {
            name: version(name)
            for name in ("torch", "transformers", "huggingface-hub", "safetensors")
        },
        "registration_sha256": sha256_file(out / "registration.json"),
        "inheritance": config["inheritance"],
        "reverse": config["reverse"],
        "published_child_training": config["training_of_published_child"],
    }
    write_json(out / "results.json", result)
    started = time.monotonic()
    try:
        execute(config, out, result)
        gc.collect()
        subprocess.run(
            [
                sys.executable,
                "scripts/check_cached_child_process.py",
                "--child",
                str(out / "private-child"),
                "--inputs",
                str(out / "fresh-input.json"),
                "--out",
                str(out / "fresh-process.json"),
            ],
            check=True,
            timeout=180,
        )
        fresh = json.loads((out / "fresh-process.json").read_text())
        if not fresh["generation_identical"]:
            raise RuntimeError("Fresh cached child generation changed")
        result["fresh_process"] = fresh
        result["status"] = "PASS"
        result["phase"] = "complete"
        write_json(out / "results.json", result)
        result["integrity"] = validate_completed_math(out, dataset_root=config["dataset_cache"])
    except Exception as error:
        result["status"] = "BLOCKED" if result["phase"] == "preflight" else "FAIL"
        result["error"] = {"type": type(error).__name__, "reason": str(error)}
        print("Completed-math diagnostic stopped", result["error"], flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "results.json", result)
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
