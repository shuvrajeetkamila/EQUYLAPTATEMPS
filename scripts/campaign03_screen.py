#!/usr/bin/env python3
"""Campaign 3 screen: sequence-aware bridges vs dense-covariance control.

Development-only selection under the frozen preregistration gates. Direct
child TRAIN loss only; no teacher targets, no hidden-activation imitation,
no TEST metrics. Sequential, resumable via the campaign registry.
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
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.alignment import initialize_conditioned, install_interfaces
from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.campaign_registry import CampaignRegistry
from equylaptatemps.campaign_training import fit_resumable, stable_hash
from equylaptatemps.evaluation import evaluate_encoded
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.references import load_reference, save_reference
from equylaptatemps.sectors import trace_child
from equylaptatemps.sequence_seams import install_sequence_interfaces, sequence_design_report
from equylaptatemps.training import TrainingOptions


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def install(child, interface):
    if interface.get("family") == "sequence":
        return install_sequence_interfaces(child, interface)
    install_interfaces(
        child,
        {"architecture": interface["architecture"], "initialization": "covariance_orthogonal"},
    )
    return child


def tokenizer_diagnostics(parents, texts):
    host_vocab = parents.tokenizer.get_vocab()
    donor_vocab = parents.donor_tokenizer.get_vocab()
    shared = set(host_vocab) & set(donor_vocab)
    rows = []
    for text in texts:
        host_ids = parents.tokenizer.encode(text, add_special_tokens=False)
        donor_ids = parents.donor_tokenizer.encode(text, add_special_tokens=False)
        rows.append(
            {
                "bytes": len(text.encode()),
                "host_tokens": len(host_ids),
                "donor_tokens": len(donor_ids),
                "same_id_sequence": host_ids == donor_ids,
            }
        )
    return {
        "kind": "measured_tokenizer_comparison_not_a_compatibility_claim",
        "host_vocab_size": len(host_vocab),
        "donor_vocab_size": len(donor_vocab),
        "shared_token_strings": len(shared),
        "host_only_strings": len(set(host_vocab) - set(donor_vocab)),
        "donor_only_strings": len(set(donor_vocab) - set(host_vocab)),
        "child_pathway": "host tokenizer + host embedding only; donor IDs never enter the child",
        "per_text": rows,
        "mean_host_tokens_per_byte": sum(r["host_tokens"] for r in rows)
        / max(sum(r["bytes"] for r in rows), 1),
        "mean_donor_tokens_per_byte": sum(r["donor_tokens"] for r in rows)
        / max(sum(r["bytes"] for r in rows), 1),
    }


def run(args):
    config = json.loads(Path(args.registry).read_text())
    out = Path(args.out)
    registry = CampaignRegistry(
        out, config["experiments"], filename="campaign_03_results.json", campaign=3
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
    base = Recipe(roles[0], roles[1], LayerSpan(11, 13), LayerSpan(20, 22), 42)
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
    write_json(out / "tokenizer_diagnostics.json", tokenizer_diagnostics(parents, dev_text))
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

    # Representation diagnostics on one dense child before any training.
    diag_child = child_from_parents(base, parents.recipient, parents.donor)
    install_interfaces(diag_child, {"architecture": "layernorm_linear", "initialization": "random"})
    stats = measure_initial(diag_child, parents, stats_text)
    slim = {
        name: {
            k: v
            for k, v in summary.items()
            if k not in {"covariance", "channel_mean", "channel_variance"}
        }
        for name, summary in stats["summary"]["streams"].items()
    }
    write_json(
        out / "representation_diagnostics.json",
        {"kind": "measured_unpaired_aggregate_diagnostics_not_teacher_targets", "streams": slim},
    )
    del diag_child, stats
    gc.collect()

    opts = TrainingOptions(
        **{
            k: config["training"][k]
            for k in (
                "steps",
                "batch_size",
                "learning_rate",
                "max_length",
                "gradient_checkpointing",
            )
        },
        seed=42,
    )
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
            reason="Campaign 3 screen run begun",
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
            child = child_from_parents(recipe, parents.recipient, parents.donor)
            install(child, exp["interface"])
            training, omitted = encode_math(
                train_math, parents.tokenizer, opts.max_length, filter_oversize=True
            )
            validation, dev_omitted = encode_math(
                dev_math, parents.tokenizer, opts.max_length, filter_oversize=True
            )
            training += encode_text(train_text, parents.tokenizer, opts.max_length)
            validation += encode_text(dev_text, parents.tokenizer, opts.max_length)
            stats = measure_initial(child, parents, stats_text)
            write_json(path / "initial_statistics.json", stats["summary"])
            init = initialize_conditioned(
                child, exp["initialization"], statistics=stats, seed=exp["seed"]
            )
            del stats
            write_json(path / "initialization.json", init)
            probe = torch.tensor(
                [parents.tokenizer.encode("Language is a way to", add_special_tokens=False)]
            )
            with torch.inference_mode(), trace_child(child) as calls:
                finite = bool(torch.isfinite(child(probe).logits).all())
            if not finite or any(n != 1 for ns in calls.values() for n in ns):
                raise ValueError("Invalid integrated native-block execution")
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
                "source_commit": source,
                "config_sha256": stable_hash(exp),
                "structure": child.structure_report(),
                "sequence_design": sequence_design_report(child)
                if exp["interface"].get("family") == "sequence"
                else None,
                "native_calls": calls,
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
            if not (path / "reference").exists():
                save_reference(child, path / "reference", parents.provenance)
            restored = load_reference(path / "reference", parents)
            with torch.inference_mode():
                identical = bool(torch.equal(child(probe).logits, restored(probe).logits))
            del restored
            report["reference_reconstruction_identical_logits"] = identical
            if not identical:
                raise ValueError("Bridge-only reference reconstruction mismatch")
            write_json(path / "results.json", report)
            metrics = dict(fit["validation"])
            metrics["ablations"] = report["ablations"]
            metrics["initial_validation"] = initial
            results[exp_id] = metrics
            registry.update(
                exp_id,
                "COMPLETED",
                reason="Development-only screen run complete, no TEST claim",
                metrics=metrics,
                paths=[
                    path / "config.json",
                    path / "results.json",
                    path / "reference/reference.json",
                    path / "reference/interfaces.safetensors",
                ],
                metadata={
                    "source_commit": source,
                    "structure": report["structure"],
                    "reproduction": (
                        f"python scripts/campaign03_screen.py --registry "
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
    promotion = {
        "status": "NOT_RUN",
        "selected": None,
        "selection_data": "development only",
        "reason": "Screen incomplete or dense control missing",
    }
    dense = results.get("c03_dense_cov_42")
    if dense:
        dense_language = dense["groups"]["domain:language"]["token_nll"]
        dense_math = dense["groups"]["domain:math"]["token_nll"]
        checks = {}
        for exp_id, metrics in results.items():
            if exp_id == "c03_dense_cov_42":
                continue
            language = metrics["groups"]["domain:language"]["token_nll"]
            math_nll = metrics["groups"]["domain:math"]["token_nll"]
            full_macro = macro(metrics)
            checks[exp_id] = {
                "G1_language_retention_1.50x_host": language <= 1.5 * host_language,
                "language_ratio_vs_host": language / host_language,
                "G2_beats_dense_control": language <= 0.95 * dense_language
                and math_nll <= 1.05 * dense_math,
                "G3a_donor_used_vs_ablations": all(
                    full_macro <= 0.98 * macro(metrics["ablations"][name])
                    for name in ("bypass_graft", "zero_graft")
                ),
                "G4_math_signal_0.90x_dense": math_nll <= 0.90 * dense_math,
                "macro_dev_nll": full_macro,
            }
        eligible = [
            (value["macro_dev_nll"], exp_id)
            for exp_id, value in checks.items()
            if all(
                value[k]
                for k in (
                    "G1_language_retention_1.50x_host",
                    "G2_beats_dense_control",
                    "G3a_donor_used_vs_ablations",
                    "G4_math_signal_0.90x_dense",
                )
            )
        ]
        promotion = {
            "status": "SCREEN_PASS" if eligible else "FAIL",
            "selected": min(eligible)[1] if eligible else None,
            "selection_data": "development only",
            "reason": (
                "Frozen preregistration gates G1,G2,G3a,G4 (G5 checked per-run by "
                "reference reconstruction); final promotion still requires confirm phase G3b+G6"
            )
            if eligible
            else "No sequence-aware candidate satisfies the frozen screen gates",
            "checks": checks,
            "host_language_dev_nll": host_language,
            "dense_control": {
                "language": dense_language,
                "math": dense_math,
                "macro": macro(dense),
            },
        }
    write_json(out / "screen_promotion.json", promotion)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "runtime_seconds": time.monotonic() - clock,
            "parents_unchanged": True,
            "registry": registry.summary(),
            "screen_promotion": promotion,
            "final_TEST_status": (
                "NOT_RUN; held-out confirmation requires final promotion in confirm phase"
            ),
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="campaigns/campaign_03/experiment_registry.json")
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
