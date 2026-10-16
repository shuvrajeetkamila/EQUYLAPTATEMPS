#!/usr/bin/env python3
"""Campaign 19 foreign-graft pilot runner (authorized execution of the
frozen preregistration at b0599b1).

--mode gates: D0 donor validity, A native, B identity gate,
structural/gradient audit, C/D/E step-0 (seed 42). No training.
--mode train: A/B recomputed, C/D/E x seeds {42,123,31415} + H(42)
trained 96 steps (evals 0/32/96), F/G per seed, frozen primary
decision, held-out slice opened in-job only if the rule passes.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from campaign03_screen import macro
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.backbones import extract_parent
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.campaign_training import fit_resumable, stable_hash
from equylaptatemps.evaluation import collate, evaluate_encoded, predict
from equylaptatemps.native_placement import (
    build_foreign_graft_child,
    build_native_depth_child,
    randomize_blocks,
    scramble_blocks,
)
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.training import TrainingOptions

RECIPIENT_ID = "HuggingFaceTB/SmolLM-135M"
RECIPIENT_REV = "1d461723eec654e65efdc40cf49301c89c0c92f4"
DONOR_ID = "state-spaces/mamba-130m-hf"
DONOR_REV = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
LR = 3e-4
SEEDS = (42, 123, 31415)
SPAN = (11, 13)


def per_example_scores(model, encoded, tokenizer):
    scores = []
    model.eval()
    with torch.no_grad():
        for record in encoded:
            batch = collate([record], tokenizer.pad_token_id, next(model.parameters()).device)
            output = predict(model, batch, "none")
            targets = batch["labels"][:, 1:]
            losses = F.cross_entropy(
                output.logits[:, :-1].float().transpose(1, 2),
                targets,
                ignore_index=-100,
                reduction="none",
            )
            mask = targets != -100
            scores.append(
                {
                    "domain": record.domain,
                    "target_tokens": int(mask.sum()),
                    "token_nll": float(losses.sum() / mask.sum()),
                }
            )
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["gates", "train"], required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(out, args)
    except BaseException:
        import traceback

        write_json(
            out / "failure.json",
            {"stage": getattr(main, "_stage", "startup"), "traceback": traceback.format_exc()},
        )
        if args.post_pr:
            publish_bundle(out, args.post_pr)
        raise


def run(out, args):
    main._stage = "setup"
    source = (
        os.environ.get("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    torch.set_num_threads(2)
    recipient_spec = acquire(RECIPIENT_ID, Path("models/pretrained"), RECIPIENT_REV)
    donor_spec = acquire(DONOR_ID, Path("models/pretrained"), DONOR_REV)
    base = Recipe(recipient_spec, donor_spec, LayerSpan(*SPAN), LayerSpan(*SPAN), 42)
    parents = load_parents(base, dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    recipient_before = module_fingerprint(parents.recipient)
    donor_before = module_fingerprint(parents.donor)
    write_json(out / "provenance.json", parents.provenance)

    acquire_benchmarks("data/local/pretrained-benchmarks")
    _, data = load_benchmarks("data/local/pretrained-benchmarks")
    train_math, dev_math = gsm_splits(data["gsm8k"]["train"], seed=42, train_count=32, dev_count=16)
    train_text = text_chunks(data["wikitext2"]["train"], characters=192, limit=32)
    dev_text = text_chunks(data["wikitext2"]["validation"], characters=192, limit=16)
    dev_encoded, _ = encode_math(dev_math, parents.tokenizer, 512, filter_oversize=True)
    dev_encoded += encode_text(dev_text, parents.tokenizer, 512)
    train_encoded, _ = encode_math(train_math, parents.tokenizer, 512, filter_oversize=True)
    train_encoded += encode_text(train_text, parents.tokenizer, 512)
    batch_records = dev_encoded[:4]

    def native_macro(model, encoded, tok):
        return macro(evaluate_encoded(model, encoded, tok))

    report = {
        "source_commit": source,
        "mode": args.mode,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    clock = time.monotonic()

    def build_foreign(condition, seed, donor_span=SPAN):
        recipe = Recipe(recipient_spec, donor_spec, LayerSpan(*SPAN), LayerSpan(*donor_span), seed)
        graft_blocks = None
        donor_blocks = list(extract_parent(parents.donor).layers[donor_span[0] : donor_span[1]])
        if condition == "D":
            graft_blocks, _ = scramble_blocks(donor_blocks, seed=seed)
        elif condition == "E":
            graft_blocks, _ = randomize_blocks(donor_blocks, seed=seed)
        child, manifest = build_foreign_graft_child(
            parents.recipient,
            parents.donor,
            recipient_span=SPAN,
            donor_span=donor_span,
            recipe=recipe,
            graft_blocks=graft_blocks,
        )
        return child, manifest

    if args.mode == "gates":
        main._stage = "A_native"
        report["A_native_macro"] = native_macro(parents.recipient, dev_encoded, parents.tokenizer)

        main._stage = "D0_donor_validity"
        donor_dev, _ = encode_math(dev_math, parents.donor_tokenizer, 512, filter_oversize=True)
        donor_dev += encode_text(dev_text, parents.donor_tokenizer, 512)
        intact = native_macro(parents.donor, donor_dev, parents.donor_tokenizer)
        scrambled_donor = copy.deepcopy(parents.donor)
        blocks, scramble_record = scramble_blocks(
            list(extract_parent(scrambled_donor).layers[SPAN[0] : SPAN[1]]), seed=1900
        )
        layers = list(scrambled_donor.backbone.layers)
        layers[SPAN[0] : SPAN[1]] = blocks
        scrambled_donor.backbone.layers = torch.nn.ModuleList(layers)
        scrambled = native_macro(scrambled_donor, donor_dev, parents.donor_tokenizer)
        del scrambled_donor
        gc.collect()
        report["D0"] = {
            "intact_macro": intact,
            "scrambled_sector_macro": scrambled,
            "margin_intact_advantage": scrambled - intact,
            "required_margin": 0.05,
            "pass": bool(scrambled - intact >= 0.05),
            "scramble_all_multisets_preserved": scramble_record["all_multisets_preserved"],
        }

        main._stage = "B_identity_gate"
        recipe_b = Recipe(
            recipient_spec,
            recipient_spec,
            LayerSpan(*SPAN),
            LayerSpan(*SPAN),
            42,
            positive_control=True,
        )
        # load_parents bound recipient_spec to SmolLM already; donor role
        # model here is the same recipient object for the self control.
        child_b, _ = build_native_depth_child(
            parents.recipient,
            execution_order=list(range(30)),
            graft_span=SPAN,
            seam_mode="residual",
            recipe=recipe_b,
            donor_parent=parents.recipient,
        )
        b_macro = native_macro(child_b, dev_encoded, parents.tokenizer)
        diffs = []
        for record in batch_records:
            ids = torch.tensor([record.ids])
            with torch.no_grad():
                ref = parents.recipient(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                got = child_b(ids, torch.ones_like(ids)).logits
            diffs.append(
                {
                    "max_abs_diff": float((got - ref).abs().max()),
                    "mean_abs_diff": float((got - ref).abs().mean()),
                }
            )
        report["B_identity"] = {
            "macro": b_macro,
            "delta_vs_native": b_macro - report["A_native_macro"],
            "logit_max_abs": max(d["max_abs_diff"] for d in diffs),
            "logit_mean_abs_max": max(d["mean_abs_diff"] for d in diffs),
            "pass": bool(
                abs(b_macro - report["A_native_macro"]) <= 0.01
                and max(d["max_abs_diff"] for d in diffs) <= 0.02
                and max(d["mean_abs_diff"] for d in diffs) <= 0.001
            ),
        }
        del child_b
        gc.collect()

        main._stage = "structural_gradient_audit"
        child, manifest = build_foreign("C", 42)
        graft_fp_before = module_fingerprint(child.graft)
        captured = {}

        def hook(_m, _i, output):
            captured["rms"] = float(output.float().pow(2).mean().sqrt())

        handle = child.graft.register_forward_hook(hook)
        ids = torch.tensor([batch_records[0].ids])
        child.train()
        logits = child(ids, torch.ones_like(ids)).logits
        loss = logits.float().mean()
        loss.backward()
        handle.remove()
        seam_grads = {
            n: float(p.grad.norm())
            for n, p in child.named_parameters()
            if n.startswith(("input_seam.", "output_seam."))
        }
        donor_grads = [
            n
            for n, p in child.named_parameters()
            if n.startswith("graft.") and p.grad is not None and float(p.grad.abs().sum()) > 0
        ]
        report["structural_audit"] = {
            "manifest": manifest,
            "trainable_parameters": sum(p.numel() for p in child.parameters() if p.requires_grad),
            "graft_output_rms": captured.get("rms"),
            "seam_gradient_norms": seam_grads,
            "donor_parameters_with_nonzero_gradient": donor_grads,
            "donor_fingerprint_unchanged_by_backward": module_fingerprint(child.graft)
            == graft_fp_before,
        }
        step0 = {}
        for condition in ("C", "D", "E"):
            if condition != "C":
                child_c, _ = build_foreign(condition, 42)
            else:
                child_c = child
            step0[condition] = native_macro(child_c, dev_encoded, parents.tokenizer)
            if condition != "C":
                del child_c
                gc.collect()
        report["step0_seed42"] = step0
        del child
        gc.collect()
    else:
        main._stage = "train_matrix"
        report["A_native_macro"] = native_macro(parents.recipient, dev_encoded, parents.tokenizer)
        runs = {}
        seam_states = {}
        for condition in ("C", "D", "E"):
            for seed in SEEDS:
                key = f"{condition}_s{seed}"
                main._stage = f"train_{key}"
                runs[key] = train_one(
                    out,
                    key,
                    condition,
                    seed,
                    SPAN,
                    build_foreign,
                    dev_encoded,
                    train_encoded,
                    parents,
                    report["A_native_macro"],
                )
                seam_states[key] = runs[key].pop("seam_state")
        main._stage = "train_H"
        runs["H_s42"] = train_one(
            out,
            "H_s42",
            "C",
            42,
            (20, 22),
            build_foreign,
            dev_encoded,
            train_encoded,
            parents,
            report["A_native_macro"],
        )
        seam_states["H_s42"] = runs["H_s42"].pop("seam_state")
        report["runs"] = runs

        main._stage = "primary_decision"
        decision = primary_decision(runs, report["A_native_macro"])
        report["primary_decision"] = decision
        write_json(out / "primary_partial.json", report)

        if decision["primary_success"]:
            main._stage = "heldout"
            report["heldout"] = run_heldout(data, parents, build_foreign, seam_states, runs)
        else:
            report["heldout"] = {"status": "NOT_RUN_NO_PROMOTION"}

    main._stage = "finalize"
    report["parents_unchanged"] = (
        module_fingerprint(parents.recipient) == recipient_before
        and module_fingerprint(parents.donor) == donor_before
    )
    report["runtime_seconds"] = time.monotonic() - clock
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(out / "results.json", report)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "mode": args.mode,
            "parents_unchanged": report["parents_unchanged"],
            "A_native_macro": report.get("A_native_macro"),
            "D0_pass": report.get("D0", {}).get("pass"),
            "B_pass": report.get("B_identity", {}).get("pass"),
            "primary_success": report.get("primary_decision", {}).get("primary_success"),
            "label_hint": report.get("primary_decision", {}).get("label_hint"),
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    print(f"Campaign 19 {args.mode} complete; parents_unchanged:", report["parents_unchanged"])


def train_one(
    out,
    key,
    condition,
    seed,
    donor_span,
    build_foreign,
    dev_encoded,
    train_encoded,
    parents,
    native,
):
    child, manifest = build_foreign(condition, seed, donor_span=donor_span)
    donor_fp_before = module_fingerprint(child.graft)
    entry = {"condition": condition, "seed": seed, "donor_span": list(donor_span)}
    entry["macro_step0"] = macro(evaluate_encoded(child, dev_encoded, parents.tokenizer))
    run_dir = out / "runs" / key
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {"experiment": key, "registry_sha256": stable_hash({"c19": key})}
    for mark in (32, 96):
        opts = TrainingOptions(
            batch_size=1,
            max_length=512,
            gradient_checkpointing=True,
            learning_rate=LR,
            steps=mark,
            seed=seed,
        )
        fit_resumable(
            child,
            parents.tokenizer,
            train_encoded,
            opts,
            validation=dev_encoded,
            output=run_dir / "calibration",
            contract=contract,
            checkpoint_every=8,
        )
        value = macro(evaluate_encoded(child, dev_encoded, parents.tokenizer))
        entry[f"macro_step{mark}"] = value
        if mark == 32:
            entry["macro_bypass_step32"] = macro(
                evaluate_encoded(child, dev_encoded, parents.tokenizer, ablation="bypass_graft")
            )
        entry[f"per_example_step{mark}"] = per_example_scores(child, dev_encoded, parents.tokenizer)
    entry["macro_bypass_step96"] = macro(
        evaluate_encoded(child, dev_encoded, parents.tokenizer, ablation="bypass_graft")
    )
    entry["donor_fingerprint_unchanged"] = module_fingerprint(child.graft) == donor_fp_before
    entry["trainable_parameters"] = sum(p.numel() for p in child.parameters() if p.requires_grad)
    entry["manifest"] = manifest
    entry["seam_state"] = {
        n: p.detach().cpu().clone()
        for n, p in child.named_parameters()
        if n.startswith(("input_seam.", "output_seam."))
    }
    write_json(run_dir / "result.json", {k: v for k, v in entry.items() if k != "seam_state"})
    del child
    gc.collect()
    return entry


def primary_decision(runs, native):
    per_seed = {}
    for seed in SEEDS:
        c = runs[f"C_s{seed}"]["macro_step32"]
        controls = {k: runs[f"{k}_s{seed}"]["macro_step32"] for k in ("D", "E")}
        best_name = min(controls, key=controls.get)
        advantage = controls[best_name] - c
        f_macro = runs[f"C_s{seed}"]["macro_bypass_step32"]
        per_seed[str(seed)] = {
            "C_step32": c,
            "best_control": best_name,
            "best_control_step32": controls[best_name],
            "advantage_step32": advantage,
            "C_preservation_damage_step32": c - native,
            "ablation_removal_fraction": (f_macro - c) / advantage if advantage > 0 else None,
        }
    advantages = [v["advantage_step32"] for v in per_seed.values()]
    removals = [
        v["ablation_removal_fraction"]
        for v in per_seed.values()
        if v["ablation_removal_fraction"] is not None
    ]
    success = bool(
        sum(advantages) / len(advantages) >= 0.25
        and sum(a > 0 for a in advantages) >= 2
        and removals
        and sum(removals) / len(removals) >= 0.5
        and all(v["C_preservation_damage_step32"] <= 4.50 for v in per_seed.values())
    )
    return {
        "per_seed": per_seed,
        "mean_advantage_step32": sum(advantages) / len(advantages),
        "seeds_positive": sum(a > 0 for a in advantages),
        "primary_success": success,
        "label_hint": "TRANSFER_EVIDENCE_POSITIVE" if success else "DONOR_ADVANTAGE_NOT_SHOWN",
    }


def run_heldout(data, parents, build_foreign, seam_states, runs):
    _, dev_math42 = gsm_splits(data["gsm8k"]["train"], seed=42, train_count=32, dev_count=16)
    dev_ids = {r["source_index"] for r in dev_math42}
    cand_train, cand_dev = gsm_splits(
        data["gsm8k"]["train"], seed=1719, train_count=32, dev_count=16
    )
    pool = [r for r in cand_dev + cand_train if r["source_index"] not in dev_ids]
    held_math = pool[:16]
    if "test" not in data["wikitext2"]:
        return {"status": "NOT_RUN_HELDOUT_SPLIT_UNAVAILABLE"}
    held_text = text_chunks(data["wikitext2"]["test"], characters=192, limit=16)
    encoded, _ = encode_math(held_math, parents.tokenizer, 512, filter_oversize=True)
    encoded += encode_text(held_text, parents.tokenizer, 512)
    results = {}
    for condition in ("C", "D", "E"):
        for seed in SEEDS:
            key = f"{condition}_s{seed}"
            child, _ = build_foreign(condition, seed)
            state = {n: p for n, p in child.named_parameters() if n in seam_states[key]}
            with torch.no_grad():
                for n, p in state.items():
                    p.copy_(seam_states[key][n].to(p.dtype))
            results[key] = macro(evaluate_encoded(child, encoded, parents.tokenizer))
            del child
            gc.collect()
    return {
        "status": "OPENED_ONCE_AFTER_PRIMARY_DECISION",
        "selection": "gsm_splits seed 1719 dev slice + WikiText-2 test chunks",
        "macro_by_run": results,
    }


if __name__ == "__main__":
    main()
