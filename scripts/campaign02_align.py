#!/usr/bin/env python3
"""Sequential bounded alignment matrix; direct child TRAIN loss, development-only promotion."""

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
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.alignment import initialize_conditioned, install_interfaces
from equylaptatemps.alignment_stats import capture_streams
from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.backbones import extract_parent
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.campaign_registry import CampaignRegistry
from equylaptatemps.campaign_training import fit_resumable, stable_hash
from equylaptatemps.evaluation import evaluate_encoded
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.references import save_reference
from equylaptatemps.sectors import trace_child
from equylaptatemps.training import TrainingOptions


def measure_initial(child, parents, chunks, *, reverse=False):
    host = parents.donor if reverse else parents.recipient
    donor = parents.recipient if reverse else parents.donor
    ht = parents.donor_tokenizer if reverse else parents.tokenizer
    dt = parents.tokenizer if reverse else parents.donor_tokenizer
    hp = extract_parent(host)
    dp = extract_parent(donor)
    cut = child.recipe.recipient_cut
    span = child.recipe.donor_sector
    groups = [
        (
            host,
            ht,
            {"recipient_native_output": (hp.layers[cut.stop - 1], "output", hp.config.hidden_size)},
        ),
        (
            donor,
            dt,
            {"donor_native_input": (dp.layers[span.start], "input", dp.config.hidden_size)},
        ),
        (
            child,
            ht,
            {
                "child_input_normalized": (child.input_seam.norm, "output", hp.config.hidden_size),
                "child_output_normalized": (
                    child.output_seam.norm,
                    "output",
                    dp.config.hidden_size,
                ),
                "child_donor_input": (child.graft, "input", dp.config.hidden_size),
                "child_host_suffix_input": (child.output_seam, "output", hp.config.hidden_size),
            },
        ),
    ]
    summaries = {}
    tensors = {}
    for model, tok, specs in groups:
        with torch.inference_mode(), capture_streams(specs) as meters:
            for text in chunks:
                ids = torch.tensor([tok.encode(text, add_special_tokens=False)], dtype=torch.long)
                if model is child:
                    model(ids)
                else:
                    model(ids, use_cache=False)
        for name, m in meters.items():
            summaries[name] = m.summary()
            tensors[name + "/covariance"] = m.covariance_matrix()
    return {"summary": {"streams": summaries, "teacher_targets": False}, "tensors": tensors}


def run(args):
    config = json.loads(Path(args.registry).read_text())
    out = Path(args.out)
    registry = CampaignRegistry(out, config["experiments"])
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
    for exp in config["experiments"]:
        id = exp["id"]
        stage = exp["stage"]
        if registry.verified_complete(id):
            continue
        if stage in {"deferred", "conditional", "invalid_control"}:
            registry.update(id, "NOT_RUN", reason=exp["reason"])
            continue
        if time.monotonic() >= deadline:
            registry.update(
                id,
                "BLOCKED",
                evaluation_status="BLOCKED",
                reason="Registered sequential wall-clock budget exhausted",
            )
            continue
        path = out / "runs" / id
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "config.json", exp)
        registry.update(
            id, "RUNNING", reason="Registered run begun", metadata={"source_commit": source}
        )
        try:
            reverse = exp.get("reverse", False)
            recipe = Recipe(
                roles[1] if reverse else roles[0],
                roles[0] if reverse else roles[1],
                LayerSpan(*exp["recipient_cut"]),
                LayerSpan(*exp["donor_sector"]),
                exp["seed"],
            )
            child = child_from_parents(
                recipe,
                parents.donor if reverse else parents.recipient,
                parents.recipient if reverse else parents.donor,
            )
            install_interfaces(
                child,
                {"architecture": exp["architecture"], "initialization": exp["initialization"]},
            )
            tok = parents.donor_tokenizer if reverse else parents.tokenizer
            training, omitted = encode_math(train_math, tok, opts.max_length, filter_oversize=True)
            validation, dev_omitted = encode_math(
                dev_math, tok, opts.max_length, filter_oversize=True
            )
            training += encode_text(train_text, tok, opts.max_length)
            validation += encode_text(dev_text, tok, opts.max_length)
            if exp["initialization"] != "random":
                stats = measure_initial(child, parents, stats_text, reverse=reverse)
                write_json(path / "initial_statistics.json", stats["summary"])
                init = initialize_conditioned(
                    child, exp["initialization"], statistics=stats, seed=exp["seed"]
                )
                del stats
            else:
                init = initialize_conditioned(child, "random")
            write_json(path / "initialization.json", init)
            input_ids = torch.tensor([tok.encode("Language is a way to", add_special_tokens=False)])
            with torch.inference_mode(), trace_child(child) as calls:
                finite = bool(torch.isfinite(child(input_ids).logits).all())
            if not finite or any(n != 1 for ns in calls.values() for n in ns):
                raise ValueError("Invalid integrated native-block execution")
            initial = evaluate_encoded(child, validation, tok)
            report = {
                "source_commit": source,
                "config_sha256": stable_hash(exp),
                "structure": child.structure_report(),
                "native_calls": calls,
                "initial_validation": initial,
                "math_oversize_train_ids": omitted,
                "math_oversize_dev_ids": dev_omitted,
                "parent_weights_frozen": True,
                "test_metrics": None,
                "evaluation_status": "NOT_RUN",
            }
            if stage == "calibration":
                fit = fit_resumable(
                    child,
                    tok,
                    training,
                    opts,
                    validation=validation,
                    output=path / "calibration",
                    contract={"experiment": exp, "registry_sha256": stable_hash(config)},
                    diagnostic_records=training[: config["training"]["diagnostic_train_rows"]],
                    checkpoint_every=config["training"]["checkpoint_every"],
                    deadline=deadline,
                )
                report["training"] = fit
                if fit["execution_status"] != "COMPLETED":
                    write_json(path / "results.json", report)
                    registry.update(
                        id,
                        "INTERRUPTED",
                        reason="Budget stop with verified NEW-parameter/optimizer checkpoint",
                        paths=[
                            path / "config.json",
                            path / "results.json",
                            path / "calibration/state.json",
                            path / "calibration/state.safetensors",
                        ],
                    )
                    break
                report["validation"] = fit["validation"]
            else:
                report["validation"] = initial
            if not (path / "reference").exists():
                save_reference(child, path / "reference", parents.provenance)
            write_json(path / "results.json", report)
            registry.update(
                id,
                "COMPLETED",
                reason="Development-only run complete, no TEST inheritance claim",
                metrics=report["validation"],
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
                        f"python scripts/campaign02_align.py --registry {args.registry} "
                        f"--out {args.out}"
                    ),
                },
            )
        except Exception as error:
            write_json(
                path / "failure.json",
                {"type": type(error).__name__, "reason": str(error), "source_commit": source},
            )
            registry.update(
                id,
                "FAILED",
                evaluation_status="BLOCKED",
                reason=type(error).__name__ + ": " + str(error),
                paths=[path / "failure.json"],
            )
        finally:
            if "child" in locals():
                del child
            gc.collect()
    for r, h in frozen.items():
        if module_fingerprint(getattr(parents, r)) != h:
            raise ValueError("Original pretrained tensor changed")
    completed = {
        id: r
        for id, r in registry.state["records"].items()
        if r["execution_status"] == "COMPLETED" and registry.configs[id]["stage"] == "calibration"
    }
    promotion = {
        "status": "NOT_RUN",
        "reason": "No completed candidates",
        "selected": None,
        "selection_data": "development only",
    }
    if "primary_legacy_42" in completed:
        legacy = completed["primary_legacy_42"]["metrics"]["groups"]["domain:language"]["token_nll"]
        host = native_dev["recipient"]["groups"]["domain:language"]["token_nll"]
        eligible = []
        for id, r in completed.items():
            value = r["metrics"]["groups"]["domain:language"]["token_nll"]
            if value / host <= config["selection"][
                "maximum_dev_language_parent_ratio"
            ] and value <= legacy * (
                1 - config["selection"]["minimum_dev_improvement_vs_matched_legacy"]
            ):
                eligible.append((value, id))
        if eligible:
            promotion = {
                "status": "PASS",
                "selected": min(eligible)[1],
                "reason": "Predeclared development gate, not TEST outcome",
                "selection_data": "development only",
            }
        else:
            promotion = {
                "status": "FAIL",
                "selected": None,
                "reason": "No alignment passes registered development gates",
                "selection_data": "development only",
            }
    write_json(out / "promotion.json", promotion)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "runtime_seconds": time.monotonic() - clock,
            "parents_unchanged": True,
            "registry": registry.summary(),
            "promotion": promotion,
            "final_TEST_status": "NOT_RUN; confirmatory job requires development promotion",
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default="campaigns/campaign_02/experiment_registry.json")
    p.add_argument("--out", required=True)
    p.add_argument("--post-pr", type=int)
    args = p.parse_args()
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
