#!/usr/bin/env python3
"""Campaign 10 awake-control runner: Mamba-130M host, SmolLM-135M (Llama) donor.

Frozen design (campaigns/campaign_10/preregistration.md): recipient cut
[11, 13), contiguous late donor sector [24, 30) of 30 Llama blocks, dense
full-size LayerNorm+Linear seams (768<->576), initialization "random"
(DenseSeam default), 32 steps, seed-matched pretrained vs random-sector twin.
Development data only; no TEST metrics; promotion only via the frozen gates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from campaign02_align import measure_initial
from campaign03_screen import macro
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.alignment import initialize_conditioned, install_interfaces
from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.campaign_registry import CampaignRegistry
from equylaptatemps.campaign_training import fit_resumable, stable_hash
from equylaptatemps.controls import (
    calibrate_awake_sector,
    native_sector_activity,
    random_qwen_sector_control,
)
from equylaptatemps.evaluation import evaluate_encoded
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.references import save_reference
from equylaptatemps.sequence_seams import (
    initialize_bottleneck,
    install_bottleneck_interfaces,
    install_sequence_interfaces,
    sequence_design_report,
)
from equylaptatemps.training import TrainingOptions


def install(child, interface):
    if interface.get("family") == "sequence":
        return install_sequence_interfaces(child, interface)
    if interface.get("family") == "bottleneck":
        return install_bottleneck_interfaces(child, interface)
    install_interfaces(
        child,
        {"architecture": interface["architecture"], "initialization": "covariance_orthogonal"},
    )
    return child


def graft_delta(child, tokenizer, encoded_texts):
    """Relative donor-sector update ||out-in||/||in|| on real dev text.

    Near zero means the trained bridges use the sector as a near-identity
    passenger; native-scale values mean it is genuinely transforming signal.
    """
    captured = {}

    def hook(_module, args, output):
        captured["input"] = args[0].detach()
        captured["output"] = output.detach()

    handle = child.graft.register_forward_hook(hook)
    ratios, input_rms, output_rms = [], [], []
    try:
        with torch.inference_mode():
            for text in encoded_texts:
                ids = torch.tensor(
                    [tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long
                )
                child(ids)
                delta = captured["output"] - captured["input"]
                ratios.append(float(delta.norm() / captured["input"].norm().clamp_min(1e-9)))
                input_rms.append(float(captured["input"].float().square().mean().sqrt()))
                output_rms.append(float(captured["output"].float().square().mean().sqrt()))
    finally:
        handle.remove()
    return {
        "mean_relative_update": sum(ratios) / len(ratios),
        "mean_input_rms": sum(input_rms) / len(input_rms),
        "mean_output_rms": sum(output_rms) / len(output_rms),
        "texts": len(encoded_texts),
        "interpretation": "descriptive only; magnitude alone is not usefulness evidence",
    }


def run(args):
    config = json.loads(Path(args.registry).read_text())
    out = Path(args.out)
    registry = CampaignRegistry(
        out, config["experiments"], filename="campaign_10_results.json", campaign=10
    )
    write_json(out / "registration.json", config)
    clock = time.monotonic()
    deadline = clock + config["compute"]["max_session_seconds"]
    source = (
        os.environ.get("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    if (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        < config["compute"]["minimum_ram_bytes"]
    ):
        raise RuntimeError("Insufficient RAM before real parent download")
    torch.set_num_threads(2)
    roles = []
    for role in ("recipient", "donor"):
        spec = config["parents"][role]
        roles.append(acquire(spec["id"], Path("models/pretrained"), spec["revision"]))
    base = Recipe(roles[0], roles[1], LayerSpan(11, 13), LayerSpan(24, 30), 42)
    parents = load_parents(base, dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    frozen = {r: module_fingerprint(getattr(parents, r)) for r in ("recipient", "donor")}
    write_json(out / "provenance.json", parents.provenance)
    acquire_benchmarks("data/local/pretrained-benchmarks")
    manifest, data = load_benchmarks("data/local/pretrained-benchmarks")
    write_json(out / "data_provenance.json", manifest)
    dc = config["data"]
    train_math, dev_math = gsm_splits(
        data["gsm8k"]["train"],
        seed=dc["seed"],
        train_count=dc["gsm_train_rows"],
        dev_count=dc["gsm_dev_rows"],
    )
    train_text = text_chunks(
        data["wikitext2"]["train"], characters=dc["wiki_characters"], limit=dc["wiki_train_windows"]
    )
    dev_text = text_chunks(
        data["wikitext2"]["validation"],
        characters=dc["wiki_characters"],
        limit=dc["wiki_dev_windows"],
    )
    stats_text = train_text[: dc["statistic_text_windows"]]
    write_json(
        out / "data_selection.json",
        {
            "train_math": [r["source_index"] for r in train_math],
            "dev_math": [r["source_index"] for r in dev_math],
            "train_text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in train_text],
            "dev_text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in dev_text],
            "TEST_used_for_selection": False,
        },
    )
    awake_target = native_sector_activity(
        parents.donor, parents.donor_tokenizer, stats_text, LayerSpan(24, 30)
    )
    write_json(out / "awake_calibration_target.json", awake_target)
    native_dev = {}
    for name, model, tok in [
        ("recipient", parents.recipient, parents.tokenizer),
        ("donor", parents.donor, parents.donor_tokenizer),
    ]:
        encoded, _ = encode_math(
            dev_math, tok, config["training"]["max_length"], filter_oversize=True
        )
        encoded += encode_text(dev_text, tok, config["training"]["max_length"])
        native_dev[name] = evaluate_encoded(model, encoded, tok)
    write_json(out / "native_development.json", native_dev)

    results = {}
    for exp in config["experiments"]:
        exp_id = exp["id"]
        if registry.verified_complete(exp_id):
            results[exp_id] = registry.state["records"][exp_id]["metrics"]
            continue
        if time.monotonic() >= deadline:
            registry.update(
                exp_id, "BLOCKED", reason="Registered sequential wall-clock budget exhausted"
            )
            continue
        path = out / "runs" / exp_id
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "config.json", exp)
        registry.update(
            exp_id,
            "RUNNING",
            reason="Campaign 5 run begun",
            metadata={"source_commit": source},
        )
        try:
            recipe = Recipe(
                roles[0],
                roles[1],
                LayerSpan(*exp["recipient_cut"]),
                LayerSpan(*exp["donor_sector"]),
                exp["seed"],
            )
            if exp.get("random_sector"):
                child = random_qwen_sector_control(
                    recipe,
                    parents.recipient,
                    parents.donor,
                    initialization_seed=exp["random_sector_seed"],
                )
            else:
                child = child_from_parents(recipe, parents.recipient, parents.donor)
            install(child, exp["interface"])
            training, omitted = encode_math(
                train_math,
                parents.tokenizer,
                config["training"]["max_length"],
                filter_oversize=True,
            )
            validation, dev_omitted = encode_math(
                dev_math, parents.tokenizer, config["training"]["max_length"], filter_oversize=True
            )
            training += encode_text(train_text, parents.tokenizer, config["training"]["max_length"])
            validation += encode_text(dev_text, parents.tokenizer, config["training"]["max_length"])
            stats = measure_initial(child, parents, stats_text)
            write_json(path / "initial_statistics.json", stats["summary"])
            if exp["initialization"] == "bottleneck_covariance":
                init = initialize_bottleneck(child, stats)
            else:
                init = initialize_conditioned(
                    child, exp["initialization"], statistics=stats, seed=exp["seed"]
                )
            del stats
            if exp.get("awake"):
                init["awake_calibration"] = calibrate_awake_sector(
                    child,
                    parents.tokenizer,
                    stats_text,
                    target=awake_target["mean_relative_update"],
                )
            write_json(path / "initialization.json", init)
            opts = TrainingOptions(
                **{
                    k: config["training"][k]
                    for k in ("batch_size", "learning_rate", "max_length", "gradient_checkpointing")
                },
                steps=exp.get("steps", config["training"]["steps"]),
                seed=exp["seed"],
            )
            initial = evaluate_encoded(child, validation, parents.tokenizer)
            fit = fit_resumable(
                child,
                parents.tokenizer,
                training,
                opts,
                validation=validation,
                output=path / "calibration",
                contract={"experiment": exp, "registry_sha256": stable_hash(config)},
                diagnostic_records=training[: config["training"]["diagnostic_train_rows"]],
                checkpoint_every=config["training"]["checkpoint_every"],
                deadline=deadline,
            )
            report = {
                "training_tokens_processed": sum(
                    int((torch.tensor(training[i].labels) != -100).sum())
                    for event in fit["trace"]
                    for i in event["record_indices"]
                )
                if fit.get("trace")
                else None,
                "source_commit": source,
                "config_sha256": stable_hash(exp),
                "structure": child.structure_report(),
                "sequence_design": sequence_design_report(child),
                "random_sector_control": bool(exp.get("random_sector")),
                "initial_validation": initial,
                "training": fit,
                "math_oversize_train_ids": omitted,
                "math_oversize_dev_ids": dev_omitted,
                "parent_weights_frozen": True,
                "test_metrics": None,
            }
            if fit["execution_status"] != "COMPLETED":
                write_json(path / "results.json", report)
                registry.update(
                    exp_id,
                    "INTERRUPTED",
                    reason="Budget stop with verified NEW-parameter checkpoint",
                    paths=[
                        path / "config.json",
                        path / "results.json",
                        path / "calibration/state.json",
                        path / "calibration/state.safetensors",
                    ],
                )
                break
            report["validation"] = fit["validation"]
            report["ablations"] = {
                name: evaluate_encoded(child, validation, parents.tokenizer, ablation=name)
                for name in ("bypass_graft", "zero_graft")
            }
            report["graft_delta"] = graft_delta(child, parents.tokenizer, dev_text)
            paths = [path / "config.json", path / "results.json"]
            if not exp.get("random_sector"):
                if not (path / "reference").exists():
                    save_reference(child, path / "reference", parents.provenance)
                paths += [
                    path / "reference/reference.json",
                    path / "reference/interfaces.safetensors",
                ]
            write_json(path / "results.json", report)
            metrics = dict(fit["validation"])
            metrics["ablations"] = report["ablations"]
            metrics["graft_delta"] = report["graft_delta"]
            metrics["initial_validation"] = initial
            results[exp_id] = metrics
            registry.update(
                exp_id,
                "COMPLETED",
                reason="Campaign 10 frozen-protocol run complete; gates evaluated by publisher",
                metrics=metrics,
                paths=paths,
                metadata={
                    "source_commit": source,
                    "structure": report["structure"],
                    "reproduction": (
                        f"python scripts/campaign10_runner.py --registry "
                        f"{args.registry} --out {args.out}"
                    ),
                },
            )
        except Exception as error:
            write_json(
                path / "failure.json",
                {"type": type(error).__name__, "reason": str(error), "source_commit": source},
            )
            registry.update(
                exp_id,
                "FAILED",
                evaluation_status="BLOCKED",
                reason=type(error).__name__ + ": " + str(error),
                paths=[path / "failure.json"],
            )
        finally:
            if "child" in locals():
                del child
            gc.collect()
    for role, digest in frozen.items():
        if module_fingerprint(getattr(parents, role)) != digest:
            raise ValueError("Original pretrained tensor changed")
    host_language = native_dev["recipient"]["groups"]["domain:language"]["token_nll"]
    summary = {
        "host_language_dev_nll": host_language,
        "candidates": {
            exp_id: {
                "macro_dev_nll": macro(metrics),
                "language_dev_nll": metrics["groups"]["domain:language"]["token_nll"],
                "math_dev_nll": metrics["groups"]["domain:math"]["token_nll"],
                "language_ratio_vs_host": metrics["groups"]["domain:language"]["token_nll"]
                / host_language,
                "bypass_macro": macro(metrics["ablations"]["bypass_graft"]),
                "zero_macro": macro(metrics["ablations"]["zero_graft"]),
                "graft_relative_update": metrics["graft_delta"]["mean_relative_update"],
            }
            for exp_id, metrics in results.items()
        },
        "final_TEST_status": "NOT_RUN_NO_PROMOTION unless frozen Campaign 7 gates pass",
    }
    write_json(out / "phase_summary.json", summary)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "runtime_seconds": time.monotonic() - clock,
            "parents_unchanged": True,
            "registry": registry.summary(),
            "phase_summary": summary,
            "final_TEST_status": "NOT_RUN_NO_PROMOTION unless frozen Campaign 7 gates pass",
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="campaigns/campaign_10/experiment_registry.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(
            out / "session_failure.json",
            {"type": type(error).__name__, "reason": str(error), "status": "BLOCKED"},
        )
        if args.post_pr:
            publish_bundle(out, args.post_pr)
        raise


if __name__ == "__main__":
    main()
