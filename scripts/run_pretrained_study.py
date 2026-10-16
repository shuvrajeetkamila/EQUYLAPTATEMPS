#!/usr/bin/env python3
"""Execute the registered real-parent study. Opt in explicitly; never part of ordinary CI."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import random
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
from equylaptatemps.audit import frozen_fingerprint, module_fingerprint, sha256_file
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.benchmarks import observe_language, observe_math
from equylaptatemps.evaluation import evaluate_encoded
from equylaptatemps.evidence import (
    comparison_template,
    inheritance_criteria,
    pending,
    validate_results,
)
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.references import save_reference
from equylaptatemps.sectors import (
    activation_statistics,
    annotate_selection,
    identity_sector,
    sector_inventory,
    trace_child,
)
from equylaptatemps.training import TrainingOptions, fit_encoded_seams

STAGES = (
    "preflight",
    "parents",
    "datasets",
    "parent_baselines",
    "sector_discovery",
    "calibration",
    "surgery",
    "reload",
    "benchmarks",
    "controls",
    "reverse",
    "integrity",
)


def seal(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", config)
    commit = (
        os.environ.get("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    result = {
        "schema_version": 1,
        "project": config["project"],
        "parent_kind": "pretrained",
        "source_commit": commit,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "registration_file": "registration.json",
        "registration_sha256": sha256_file(output / "registration.json"),
        "stages": {name: pending() for name in STAGES},
        "comparisons": comparison_template(),
        "sample_counts": {
            "gsm8k": config["benchmark"]["gsm_test_samples"],
            "wikitext2": config["benchmark"]["wiki_test_samples"],
        },
        "inheritance": pending("Held-out benchmarks not executed"),
    }
    write_json(output / "results.json", result)
    return result


def pilot_config(config: dict) -> dict:
    # Exploratory preset, sealed before held-out evaluation; cannot pass the min64 criteria.
    config = json.loads(json.dumps(config))
    config["scope"] = "CPU exploratory pilot; underpowered for registered inheritance criteria"
    config["device"] = "cpu"
    config["donor_candidates"] = [[13, 15], [20, 22]]
    config["calibration"].update(
        steps=16,
        gsm_train_rows=16,
        gsm_dev_rows=4,
        wiki_train_chunks=8,
        wiki_dev_chunks=4,
        max_length=512,
    )
    config["benchmark"].update(gsm_test_samples=4, wiki_test_samples=16, max_new_tokens=32)
    return config


def run(config: dict, out: Path, result: dict):
    def checkpoint(stage, status="PASS", **fields):
        result["stages"][stage] = {"status": status, "kind": "measured", **fields}
        write_json(out / "results.json", result)
        print(stage, status, flush=True)

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    device = config["device"]
    if ram < 12 * 1024**3 or shutil.disk_usage(out).free < 12 * 1024**3:
        raise RuntimeError("Need >=12 GiB RAM and free disk before real-parent acquisition")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Registered GPU execution unavailable; choose a GPU Colab runtime")
    torch.set_num_threads(2)
    dtype = torch.float32
    if config["dtype"] != "float32" or device not in {"cpu", "cuda"}:
        raise ValueError("This registered comparison supports common float32 CPU/CUDA only")
    result["environment"] = {
        "ram_bytes": ram,
        "device": device,
        "torch": torch.__version__,
        "dtype": "float32",
        "quantization": "none",
        "threads": 2,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    checkpoint("preflight")
    origin = config["mamba_license_origin"]
    info = HfApi().model_info(origin["model_id"], revision=origin["revision"])
    if not info.card_data or info.card_data.to_dict().get("license") != origin["license"]:
        raise RuntimeError("Mamba original-checkpoint licensing evidence not verified")
    specs = [
        acquire(
            config["parents"][role]["model_id"],
            Path(config["parent_cache"]),
            config["parents"][role]["revision"],
        )
        for role in ("recipient", "donor")
    ]
    spans = [LayerSpan(*span) for span in config["donor_candidates"]]
    recipe = Recipe(
        specs[0], specs[1], LayerSpan(*config["recipient_cut"]), spans[0], seed=config["seed"]
    )
    if recipe.synthetic:
        raise ValueError("A real-pretrained study cannot use synthetic parents")
    parents = load_parents(recipe, dtype=dtype)
    for model in (parents.recipient, parents.donor):
        model.to(device).requires_grad_(False).eval()
    before = {
        "recipient": module_fingerprint(parents.recipient),
        "donor": module_fingerprint(parents.donor),
    }
    result["provenance"] = parents.provenance
    write_json(out / "provenance.json", parents.provenance)
    result["provenance"] = {
        "file": "provenance.json",
        "sha256": sha256_file(out / "provenance.json"),
    }
    checkpoint(
        "parents",
        ids=[spec.model_id for spec in specs],
        revisions=[spec.revision for spec in specs],
    )
    acquire_benchmarks(Path(config["dataset_cache"]))
    manifest, datasets = load_benchmarks(Path(config["dataset_cache"]))
    write_json(out / "dataset-provenance.json", manifest)
    cal = config["calibration"]
    train_math, dev_math = gsm_splits(
        datasets["gsm8k"]["train"],
        seed=config["seed"],
        train_count=cal["gsm_train_rows"],
        dev_count=cal["gsm_dev_rows"],
    )
    train_text = text_chunks(
        datasets["wikitext2"]["train"],
        characters=cal["wiki_characters"],
        limit=cal["wiki_train_chunks"],
    )
    dev_text = text_chunks(
        datasets["wikitext2"]["validation"],
        characters=cal["wiki_characters"],
        limit=cal["wiki_dev_chunks"],
    )
    test_text = text_chunks(
        datasets["wikitext2"]["test"],
        characters=cal["wiki_characters"],
        limit=config["benchmark"]["wiki_test_samples"],
    )
    order = list(range(len(datasets["gsm8k"]["test"])))
    random.Random(config["seed"]).shuffle(order)
    test_math = [
        dict(datasets["gsm8k"]["test"][index], source_index=index)
        for index in order[: config["benchmark"]["gsm_test_samples"]]
    ]

    def normalize(text):
        return " ".join(text.lower().split())

    training_questions = {normalize(row["question"]) for row in train_math + dev_math}
    if training_questions & {normalize(row["question"]) for row in test_math}:
        raise ValueError("Calibration/development/test math prompt leakage")
    if any(
        set(map(normalize, a)) & set(map(normalize, b))
        for a, b in ((train_text, dev_text), (train_text, test_text), (dev_text, test_text))
    ):
        raise ValueError("Calibration/development/test text window leakage")
    write_json(
        out / "split-selection.json",
        {
            "gsm_train": [r["source_index"] for r in train_math],
            "gsm_dev": [r["source_index"] for r in dev_math],
            "gsm_test": [r["source_index"] for r in test_math],
            "wiki_windows": {
                "train": [hashlib.sha256(t.encode()).hexdigest() for t in train_text],
                "dev": [hashlib.sha256(t.encode()).hexdigest() for t in dev_text],
                "test": [hashlib.sha256(t.encode()).hexdigest() for t in test_text],
            },
        },
    )
    checkpoint(
        "datasets",
        leakage_checks="selected normalized questions/text windows; no pretraining decontamination",
    )

    def comparison(role, model, tokenizer, ablation="none"):
        for task in ("gsm8k", "wikitext2"):
            path = out / "observations" / f"{role}-{task}.jsonl"
            try:
                metrics = (
                    observe_math(
                        model,
                        tokenizer,
                        test_math,
                        path,
                        ablation=ablation,
                        max_new_tokens=config["benchmark"]["max_new_tokens"],
                        max_input_tokens=config["benchmark"]["max_input_tokens"],
                    )
                    if task == "gsm8k"
                    else observe_language(
                        model,
                        tokenizer,
                        test_text,
                        path,
                        ablation=ablation,
                        max_length=cal["max_length"],
                    )
                )
                entry = {
                    "status": "PASS",
                    "kind": "measured",
                    "metrics": metrics,
                    "records": str(path.relative_to(out)),
                    "records_sha256": sha256_file(path),
                }
            except Exception as error:
                entry = {
                    "status": "FAIL",
                    "kind": "unavailable",
                    "metrics": None,
                    "reason": type(error).__name__ + ": " + str(error),
                    "partial_records": str(path.relative_to(out)) if path.exists() else None,
                }
            result["comparisons"][role][task] = entry
            write_json(out / "results.json", result)
            print(role, task, entry["status"], entry.get("metrics"), flush=True)

    comparison("recipient_parent", parents.recipient, parents.tokenizer)
    comparison("donor_parent", parents.donor, parents.donor_tokenizer)
    baseline_ok = all(
        result["comparisons"][role][task]["status"] == "PASS"
        for role in ("recipient_parent", "donor_parent")
        for task in ("gsm8k", "wikitext2")
    )
    checkpoint(
        "parent_baselines",
        status="PASS" if baseline_ok else "FAIL",
        reason="All original-parent measurements required",
    )
    donor_inventory = sector_inventory(parents.donor, specs[1], spans)
    host_inventory = sector_inventory(parents.recipient, specs[0], [recipe.recipient_cut])
    for model, tokenizer, inventory in (
        (parents.recipient, parents.tokenizer, host_inventory),
        (parents.donor, parents.donor_tokenizer, donor_inventory),
    ):
        dev_encoded, _ = encode_math(dev_math, tokenizer, cal["max_length"], filter_oversize=True)
        dev_encoded += encode_text(dev_text, tokenizer, cal["max_length"])
        baseline = evaluate_encoded(model, dev_encoded, tokenizer)
        diagnostic = [{"input_ids": torch.tensor([r.ids], device=device)} for r in dev_encoded[:2]]
        activations = activation_statistics(model, diagnostic)
        for sector in inventory["sectors"]:
            span = LayerSpan(*sector["block_boundaries"])
            with identity_sector(model, span):
                ablated = evaluate_encoded(model, dev_encoded, tokenizer)
            sector.update(
                baseline_performance=baseline,
                ablation_performance=ablated,
                activation_statistics=activations["blocks"][span.start : span.stop],
                capability_label_status="hypothesis only; development NLL is not math accuracy",
            )
    write_json(
        out / "sector-inventory.json", {"recipient": host_inventory, "donor": donor_inventory}
    )
    checkpoint("sector_discovery", status="PASS", selection_method=cal["selection"])
    train_encoded, omitted = encode_math(
        train_math, parents.tokenizer, cal["max_length"], filter_oversize=True
    )
    dev_encoded, dev_omitted = encode_math(
        dev_math, parents.tokenizer, cal["max_length"], filter_oversize=True
    )
    if not train_encoded or not dev_encoded:
        raise ValueError(
            "No context-eligible real math train/development samples; do not silently omit a domain"
        )
    train_encoded += encode_text(train_text, parents.tokenizer, cal["max_length"])
    dev_encoded += encode_text(dev_text, parents.tokenizer, cal["max_length"])
    options = TrainingOptions(
        **{
            key: cal[key]
            for key in (
                "steps",
                "batch_size",
                "learning_rate",
                "max_length",
                "gradient_checkpointing",
            )
        },
        seed=config["seed"],
    )
    candidates = []
    best_score, best_child = float("inf"), None
    for index, span in enumerate(spans):
        candidate_recipe = Recipe(
            specs[0], specs[1], recipe.recipient_cut, span, seed=config["seed"]
        )
        child = child_from_parents(candidate_recipe, parents.recipient, parents.donor).to(device)
        fit = fit_encoded_seams(
            child,
            parents.tokenizer,
            train_encoded,
            options,
            heldout=dev_encoded,
            journal=out / f"calibration-{index}.jsonl",
        )
        write_json(out / f"calibration-{index}.json", fit)
        groups = fit["validation"]["groups"]
        score = (groups["domain:math"]["token_nll"] + groups["domain:language"]["token_nll"]) / 2
        candidates.append(
            {
                "span": [span.start, span.stop],
                "development_selection_score": score,
                "training_file": f"calibration-{index}.json",
                "status": "PASS",
            }
        )
        if score < best_score:
            best_score, best_child = score, child
        print("candidate", index, "development score", score, flush=True)
    child = best_child
    result["search"] = {
        "candidates": candidates,
        "selected": [child.recipe.donor_sector.start, child.recipe.donor_sector.stop],
        "selection_reason": cal["selection"],
        "omitted_oversize_train_ids": omitted,
        "omitted_oversize_dev_ids": dev_omitted,
    }
    donor_inventory = annotate_selection(donor_inventory, result["search"])
    write_json(
        out / "sector-inventory.json", {"recipient": host_inventory, "donor": donor_inventory}
    )
    result["structure"] = child.structure_report()
    checkpoint("calibration")
    probe = torch.tensor(
        [parents.tokenizer.encode("The study of mathematics", add_special_tokens=False)],
        device=device,
    )
    with torch.inference_mode(), trace_child(child) as calls:
        logits = child(probe).logits
    if not torch.isfinite(logits).all() or any(n != 1 for ns in calls.values() for n in ns):
        raise RuntimeError("Integrated real child failed complete-sector execution trace")
    checkpoint("surgery", native_block_calls=calls, finite_logits=True)
    save_artifact(
        child,
        parents.tokenizer,
        out / "private-child",
        parents.provenance,
        training_status="seam_calibrated",
    )
    save_reference(child, out / "reference-child", parents.provenance)
    write_json(out / "probe-input.json", probe.cpu().tolist())
    # Fresh process reload stays on CPU, avoiding a second full GPU copy.
    subprocess.run(
        [
            sys.executable,
            "scripts/fresh_child_check.py",
            "--child",
            str(out / "private-child"),
            "--ids",
            str(out / "probe-input.json"),
            "--out",
            str(out / "fresh-process"),
        ],
        check=True,
        timeout=180,
    )
    fresh = load_file(str(out / "fresh-process/logits.safetensors"))["logits"]
    if not torch.allclose(
        logits.cpu(),
        fresh,
        atol=1e-3 if device == "cuda" else 1e-5,
        rtol=1e-3 if device == "cuda" else 1e-5,
    ):
        raise RuntimeError("Fresh-process real-child inference disagreed with original")
    checkpoint(
        "reload",
        parents_loaded_in_fresh_process=False,
        identical_logits=bool(torch.equal(logits.cpu(), fresh)),
    )
    comparison("calibrated_child", child, parents.tokenizer)
    uncalibrated = child_from_parents(child.recipe, parents.recipient, parents.donor).to(device)
    comparison("uncalibrated_child", uncalibrated, parents.tokenizer)
    for ablation in (
        "bypass_graft",
        "bypass_prefix",
        "bypass_suffix",
        "zero_input_seam",
        "zero_output_seam",
    ):
        comparison(ablation, child, parents.tokenizer, ablation)
    control_recipe = Recipe(
        specs[0], specs[1], recipe.recipient_cut, child.recipe.donor_sector, seed=config["seed"] + 1
    )
    control = child_from_parents(control_recipe, parents.recipient, parents.donor).to(device)
    comparison("random_interface", control, parents.tokenizer)
    complete = all(
        entry["status"] == "PASS"
        for tasks in result["comparisons"].values()
        for entry in tasks.values()
    )
    checkpoint(
        "benchmarks",
        status="PASS" if complete else "FAIL",
        reason="All required held-out comparisons must complete",
    )
    checkpoint(
        "controls",
        status="PASS" if complete else "FAIL",
        reason="Required control measurements must complete",
        warning="Independent random interfaces, not training-budget matched",
    )
    # Resource-bounded opposite direction is structural only. No extra sectors scaled blindly.
    reverse_recipe = Recipe(
        specs[1], specs[0], child.recipe.donor_sector, recipe.recipient_cut, seed=config["seed"]
    )
    reverse = child_from_parents(reverse_recipe, parents.donor, parents.recipient).to(device)
    reverse_ids = torch.tensor(
        [parents.donor_tokenizer.encode("A mathematical argument", add_special_tokens=False)],
        device=device,
    )
    with torch.inference_mode():
        reverse_finite = bool(torch.isfinite(reverse(reverse_ids).logits).all())
    checkpoint(
        "reverse",
        status="PASS" if reverse_finite else "FAIL",
        reason="Structural-only feasibility; no reverse capability claim",
    )
    if any(module_fingerprint(getattr(parents, role)) != value for role, value in before.items()):
        raise RuntimeError("An original pretrained parent changed; all scientific claims refused")
    result["frozen_invariance"] = {
        "status": "PASS",
        "all_original_parent_tensors_unchanged": True,
        "child_frozen_sha256": frozen_fingerprint(child),
    }
    result["inheritance"] = inheritance_criteria(result, out, config["criteria"])
    checkpoint("integrity", **validate_results(out, dataset_root=config["dataset_cache"]))


def publish_bundle(out: Path, pr: int):
    files = {}
    for file in out.rglob("*"):
        relative = str(file.relative_to(out))
        if (
            file.is_file()
            and file.suffix in {".json", ".jsonl"}
            and not relative.startswith(("models/", "datasets/", "private-child/"))
        ):
            files[relative] = file.read_text()
    raw = json.dumps(files, sort_keys=True).encode()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    parts = [encoded[i : i + 40000] for i in range(0, len(encoded), 40000)]
    for index, part in enumerate(parts):
        data = {
            "kind": "pretrained-study-evidence",
            "encoding": "gzip+base64",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "run_id": os.environ["GITHUB_RUN_ID"],
            "source_commit": os.environ["GITHUB_SHA"],
            "part": index,
            "parts": len(parts),
            "content": part,
        }
        body = (
            "<!-- heterogeneous-study-evidence -->\n"
            "No parent tensors or raw WikiText.\n```json\n" + json.dumps(data) + "\n```"
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
            print("Evidence bundle part posted", index, response.status, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrained-study.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu-pilot", action="store_true")
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.cpu_pilot:
        config = pilot_config(config)
    out = Path(args.out)
    result = seal(config, out)
    started = time.monotonic()
    try:
        run(config, out, result)
    except Exception as error:
        next_stage = next(
            (name for name in STAGES if result["stages"][name]["status"] == "NOT_RUN"), "integrity"
        )
        result["stages"][next_stage] = {
            "status": "BLOCKED" if next_stage == "preflight" else "FAIL",
            "reason": type(error).__name__ + ": " + str(error),
            "kind": "unavailable",
        }
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        print("Study stopped:", type(error).__name__, str(error), flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "results.json", result)
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    if any(stage["status"] != "PASS" for stage in result["stages"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
