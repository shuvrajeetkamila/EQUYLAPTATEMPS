#!/usr/bin/env python3
"""Campaign 16 paired connector panel (derived from the Campaign 14 runner).

Same-family self-control at the Campaign 13 placement: residual-identity
vs historical connectors, seeds {42, 123, 31415}, 96 steps, validation
at steps 0/8/16/32/64/96 via segment-driven hash-verified resumes.
Frozen design: campaigns/campaign_16/preregistration.md.
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
    row_shuffled_sector_control,
    shuffled_sector_control,
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
        out, config["experiments"], filename="campaign_16_results.json", campaign=16
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
    base = Recipe(
        roles[0], roles[1], LayerSpan(11, 13), LayerSpan(24, 30), 42, positive_control=True
    )
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
                positive_control=True,
            )
            if exp.get("row_shuffled_sector"):
                child = row_shuffled_sector_control(
                    recipe,
                    parents.recipient,
                    parents.donor,
                    shuffle_seed=exp["row_shuffle_seed"],
                )
            elif exp.get("shuffled_sector"):
                child = shuffled_sector_control(
                    recipe,
                    parents.recipient,
                    parents.donor,
                    shuffle_seed=exp["shuffle_seed"],
                )
            elif exp.get("random_sector"):
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
            elif exp["initialization"] == "residual_identity":
                init = {
                    "kind": "residual_identity_zero_init",
                    "interfaces": child.interfaces_config,
                }
            else:
                init = initialize_conditioned(
                    child, exp["initialization"], statistics=stats, seed=exp["seed"]
                )
            del stats
            if getattr(child, "control_record", None):
                init["sector_control"] = child.control_record
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
                    for k in ("batch_size", "max_length", "gradient_checkpointing")
                },
                learning_rate=exp.get("learning_rate", config["training"]["learning_rate"]),
                steps=exp.get("steps", config["training"]["steps"]),
                seed=exp["seed"],
            )
            initial = evaluate_encoded(child, validation, parents.tokenizer)
            # Detached optimization diagnostics (Campaign 16): snapshots of
            # the seam parameters at step 0; update magnitudes and graft
            # activity are measured between segments without touching
            # training mathematics.
            seam_snapshot = {
                n: p.detach().cpu().clone()
                for n, p in child.named_parameters()
                if n.startswith(("input_seam.", "output_seam."))
            }

            def checkpoint_diagnostics(step, _child=child, _snap=seam_snapshot):
                updates = {}
                changed = total = 0
                for group in ("input_seam", "output_seam"):
                    sq = psq = 0.0
                    for n, p in _child.named_parameters():
                        if n.startswith(group + "."):
                            delta = p.detach().cpu() - _snap[n]
                            sq += float(delta.square().sum())
                            psq += float(_snap[n].square().sum())
                            changed += int((delta != 0).sum())
                            total += delta.numel()
                    base = psq**0.5
                    updates[group] = {
                        "update_l2": sq**0.5,
                        "update_to_param_ratio": (sq**0.5) / base if base > 0 else None,
                    }
                return {
                    "step": step,
                    "seam_updates": updates,
                    "fraction_parameters_changed": changed / total if total else 0.0,
                    "graft_activity": graft_delta(_child, parents.tokenizer, stats_text),
                }

            checkpoint_diagnostics_list = [checkpoint_diagnostics(0)]
            # Segment-driven scheduled evaluation (Campaign 16): interrupt
            # at each preregistered mark, evaluate, resume via the library's
            # hash-verified mechanism. Training identity is unchanged.
            checkpoint_evaluations = [{"step": 0, "validation": initial}]
            fit_kwargs = {
                "validation": validation,
                "output": path / "calibration",
                "contract": {"experiment": exp, "registry_sha256": stable_hash(config)},
                "diagnostic_records": training[: config["training"]["diagnostic_train_rows"]],
                "checkpoint_every": config["training"]["checkpoint_every"],
                "deadline": deadline,
            }
            for mark in (8, 16, 32, 64):
                if mark < opts.steps:
                    fit_resumable(
                        child,
                        parents.tokenizer,
                        training,
                        opts,
                        interrupt_after=mark,
                        **fit_kwargs,
                    )
                    checkpoint_evaluations.append(
                        {
                            "step": mark,
                            "validation": evaluate_encoded(child, validation, parents.tokenizer),
                        }
                    )
                    checkpoint_diagnostics_list.append(checkpoint_diagnostics(mark))
            fit = fit_resumable(child, parents.tokenizer, training, opts, **fit_kwargs)
            checkpoint_evaluations.append({"step": opts.steps, "validation": fit["validation"]})
            checkpoint_diagnostics_list.append(checkpoint_diagnostics(opts.steps))
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
                "checkpoint_evaluations": checkpoint_evaluations,
                "checkpoint_diagnostics": checkpoint_diagnostics_list,
                "learning_rate": opts.learning_rate,
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
                reason="Campaign 16 frozen-protocol run complete; gates evaluated by publisher",
                metrics=metrics,
                paths=paths,
                metadata={
                    "source_commit": source,
                    "structure": report["structure"],
                    "reproduction": (
                        f"python scripts/campaign12_runner.py --registry "
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
    parser.add_argument(
        "--registry", default="campaigns/campaign_16/experiment_registry_screen32.json"
    )
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
