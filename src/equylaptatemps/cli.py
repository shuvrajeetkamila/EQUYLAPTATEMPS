"""Research CLI: validate -> build/search -> calibrate -> evaluate -> generate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .artifacts import load_artifact, read_manifest, save_artifact, write_json
from .data import read_records
from .evaluation import evaluate_answers, evaluate_records, greedy_answer
from .parents import DTYPES, build_local, load_local_model, load_parents
from .recipe import Recipe
from .search import search_sectors
from .synthetic import run_demo
from .tokenization import check_tokenizer, load_tokenizer, tokenizer_fingerprint
from .training import TrainingOptions, fit_seams


def _training_arguments(parser):
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")


def _options(args):
    return TrainingOptions(
        args.steps,
        args.batch_size,
        args.lr,
        args.max_length,
        args.seed,
        args.gradient_checkpointing,
    )


def _new_output(path):
    if Path(path).exists():
        raise FileExistsError(f"Use a new output directory; {path} already exists")


def _emit(report, path=None):
    if path:
        write_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="equylaptatemps",
        description=(
            "Experimental stranger-model whole-block surgery, not a validated foundation model."
        ),
    )
    root.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    sub = root.add_subparsers(dest="command", required=True)
    validate = sub.add_parser(
        "validate", help="Check a recipe's reviewed family/lineage constraints"
    )
    validate.add_argument("recipe")
    build = sub.add_parser("build", help="Transplant intact blocks from two local parent snapshots")
    build.add_argument("--recipe", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--dtype", choices=DTYPES, default="float32")
    build.add_argument("--device", default="cpu")
    build.add_argument("--report")
    demo = sub.add_parser("demo", help="Offline, tiny RANDOM Mamba/Qwen2 architecture smoke test")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--steps", type=int, default=40)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--report")
    train = sub.add_parser(
        "train", help="Calibrate full-rank seams using ground-truth JSONL answers"
    )
    train.add_argument("--child", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--validation-data")
    train.add_argument("--out", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--report")
    _training_arguments(train)
    search = sub.add_parser(
        "search", help="Rank full donor windows by disjoint validation-set loss"
    )
    search.add_argument("--recipe", required=True)
    search.add_argument("--data", required=True)
    search.add_argument("--validation-data", required=True)
    search.add_argument("--donor-depth", type=int, required=True)
    search.add_argument("--stride", type=int, default=1)
    search.add_argument("--out", required=True)
    search.add_argument("--dtype", choices=DTYPES, default="float32")
    search.add_argument("--device", default="cpu")
    search.add_argument("--report")
    _training_arguments(search)
    evaluate = sub.add_parser(
        "evaluate", help="Held-out loss, graft ablations and native-parent baselines"
    )
    evaluate.add_argument("--child", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--max-length", type=int, default=256)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--ablations", action="store_true")
    evaluate.add_argument("--generate-answers", action="store_true")
    evaluate.add_argument("--max-new-tokens", type=int, default=64)
    evaluate.add_argument("--recipient-baseline")
    evaluate.add_argument("--donor-baseline")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--report")
    generate = sub.add_parser("generate", help="Greedy, uncached generation with a portable child")
    generate.add_argument("--child", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--device", default="cpu")
    inspect = sub.add_parser(
        "inspect", help="Show checkpoint structure/provenance; optionally verify tensors"
    )
    inspect.add_argument("child")
    inspect.add_argument("--verify", action="store_true")
    return root


def _evaluate(args):
    stored = load_artifact(args.child, device=args.device)
    records = read_records(args.data)
    report = {
        "capability_status": stored.manifest["capability_status"],
        "child": evaluate_records(
            stored.model,
            stored.tokenizer,
            records,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
    }
    if args.ablations:
        report["ablations"] = {
            ablation: evaluate_records(
                stored.model,
                stored.tokenizer,
                records,
                max_length=args.max_length,
                batch_size=args.batch_size,
                ablation=ablation,
            )
            for ablation in ("bypass_graft", "zero_graft")
        }
    if args.generate_answers:
        report["child_answers"] = evaluate_answers(
            stored.model, stored.tokenizer, records, max_new_tokens=args.max_new_tokens
        )
    for role in ("recipient", "donor"):
        path = getattr(args, f"{role}_baseline")
        if not path:
            continue
        tokenizer = load_tokenizer(path)
        model = load_local_model(path, DTYPES[stored.manifest["dtype"]]).to(args.device)
        spec = getattr(stored.model.recipe, role)
        if model.config.model_type != spec.identity.family:
            raise ValueError(f"{role} baseline architecture does not match the recipe")
        check_tokenizer(tokenizer, model.config.vocab_size)
        expected_hash = stored.manifest["provenance"]["tokenizers"][f"{role}_sha256"]
        if tokenizer_fingerprint(tokenizer) != expected_hash:
            raise ValueError(f"{role} baseline tokenizer does not match the recorded parent")
        baseline = evaluate_records(
            model, tokenizer, records, max_length=args.max_length, batch_size=args.batch_size
        )
        baseline["comparison_warning"] = (
            "Recipient NLL is directly comparable to child NLL. Donor-token perplexity is not "
            "comparable across different tokenizers; use task scores and manual error analysis."
        )
        if args.generate_answers:
            baseline["answers"] = evaluate_answers(
                model, tokenizer, records, max_new_tokens=args.max_new_tokens
            )
        report[f"{role}_baseline"] = baseline
        del model
    report["warning"] = "Use a final test split not used for seam calibration or sector selection."
    _emit(report, args.report)


def execute(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "validate":
        recipe = Recipe.load(args.recipe)
        _emit(
            {
                "recipe": recipe.to_dict(),
                "stranger_checks": recipe.stranger_report(),
                "local_weights_checked": False,
            }
        )
    elif args.command == "build":
        _new_output(args.out)
        recipe = Recipe.load(args.recipe)
        child, tokenizer, provenance = build_local(
            recipe, dtype=DTYPES[args.dtype], device=args.device
        )
        manifest = save_artifact(child, tokenizer, args.out, provenance)
        _emit(
            {
                "artifact": args.out,
                "training_status": manifest["training_status"],
                "capability_status": manifest["capability_status"],
                "audit": manifest["audit"],
                "provenance": provenance,
            },
            args.report,
        )
    elif args.command == "demo":
        _new_output(args.out)
        child, tokenizer, provenance, report = run_demo(steps=args.steps, seed=args.seed)
        manifest = save_artifact(
            child,
            tokenizer,
            args.out,
            provenance,
            training_status="seam_calibrated",
            metrics=report,
        )
        restored = load_artifact(args.out)
        probe = torch.tensor([tokenizer.encode("1+1=\n\nAnswer:")])
        with torch.inference_mode():
            identical = torch.equal(child(probe).logits, restored.model(probe).logits)
        if not identical:
            raise RuntimeError("Portable child reload changed the demo logits")
        report["artifact_reload_identical"] = identical
        report["weights_sha256"] = manifest["weights_sha256"]
        report["artifact"] = args.out
        _emit(report, args.report)
    elif args.command == "train":
        _new_output(args.out)
        stored = load_artifact(args.child, device=args.device)
        records = read_records(args.data)
        validation = read_records(args.validation_data) if args.validation_data else None
        report = fit_seams(
            stored.model, stored.tokenizer, records, _options(args), validation=validation
        )
        save_artifact(
            stored.model,
            stored.tokenizer,
            args.out,
            stored.manifest["provenance"],
            training_status="seam_calibrated",
            metrics=report,
        )
        _emit(report, args.report)
    elif args.command == "search":
        _new_output(args.out)
        recipe = Recipe.load(args.recipe)
        parents = load_parents(recipe, dtype=DTYPES[args.dtype], device=args.device)
        train, validation = read_records(args.data), read_records(args.validation_data)
        child, provenance, report = search_sectors(
            recipe,
            parents,
            train,
            validation,
            _options(args),
            donor_depth=args.donor_depth,
            stride=args.stride,
            progress=lambda candidate: print(json.dumps(candidate), file=sys.stderr),
        )
        save_artifact(
            child,
            parents.tokenizer,
            args.out,
            provenance,
            training_status="seam_calibrated",
            metrics=report,
        )
        _emit(report, args.report)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "generate":
        stored = load_artifact(args.child, device=args.device)
        print(
            f"Status: {stored.manifest['capability_status']}; not a validated capability model.",
            file=sys.stderr,
        )
        print(
            greedy_answer(
                stored.model, stored.tokenizer, args.prompt, max_new_tokens=args.max_new_tokens
            )
        )
    elif args.command == "inspect":
        manifest = read_manifest(args.child)
        if args.verify:
            manifest = load_artifact(args.child).manifest
        _emit(
            {
                "training_status": manifest["training_status"],
                "capability_status": manifest["capability_status"],
                "audit": manifest["audit"],
                "provenance": manifest["provenance"],
                "checksums_verified": args.verify,
            }
        )


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        execute(args)
    except (ValueError, OSError, RuntimeError) as error:
        root.exit(2, f"error: {error}\n")
