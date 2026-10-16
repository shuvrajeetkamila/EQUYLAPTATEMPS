#!/usr/bin/env python3
"""Campaign 18 structural damage map runner.

--set map: Stage-2 controls, invalid-manifest probes, inverse
controls, development (48) + validation (5) conditions, mechanism
subset. --set heldout: native anchor + identity control + the 30
sealed held-out conditions (run once, after predictor freeze).
Every condition is built through the Campaign 17 placement engine;
per-example scores, manifests, and block-hash audits are recorded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from equylaptatemps.audit import hash_state, module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.evaluation import collate, evaluate_encoded, predict
from equylaptatemps.model import child_from_parents
from equylaptatemps.native_placement import build_native_depth_child, validate_execution_order
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
DEPTH = 30
SETS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "campaigns/campaign_18/config/placement_sets.json"
    ).read_text()
)["sets"]


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
                    "nll_sum": float(losses.sum()),
                    "token_nll": float(losses.sum() / mask.sum()),
                }
            )
    return scores


def block_hashes(model_layers):
    return [hash_state(dict(block.named_parameters())) for block in model_layers]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", choices=["map", "heldout"], required=True)
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
    spec = acquire(MODEL_ID, Path("models/pretrained"), REVISION)
    roles = [spec, spec]
    base = Recipe(
        roles[0], roles[1], LayerSpan(24, 30), LayerSpan(24, 30), 42, positive_control=True
    )
    parents = load_parents(base, dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    recipient_before = module_fingerprint(parents.recipient)
    write_json(out / "provenance.json", parents.provenance)
    rec_hashes = block_hashes(list(parents.recipient.model.layers))

    acquire_benchmarks("data/local/pretrained-benchmarks")
    _, data = load_benchmarks("data/local/pretrained-benchmarks")
    _, dev_math = gsm_splits(data["gsm8k"]["train"], seed=42, train_count=32, dev_count=16)
    dev_text = text_chunks(data["wikitext2"]["validation"], characters=192, limit=16)
    from equylaptatemps.objectives import encode_math, encode_text

    dev_encoded, _ = encode_math(dev_math, parents.tokenizer, 512, filter_oversize=True)
    dev_encoded += encode_text(dev_text, parents.tokenizer, 512)
    batch_records = dev_encoded[:4]
    batch_ids_sha = hashlib.sha256(json.dumps([r.ids for r in batch_records]).encode()).hexdigest()

    def native_logits(record):
        ids = torch.tensor([record.ids])
        with torch.no_grad():
            return parents.recipient(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

    native_reference = [native_logits(r) for r in batch_records]

    def evaluate_model(model, *, child=True):
        metrics = evaluate_encoded(model, dev_encoded, parents.tokenizer)
        diffs = []
        if child:
            for record, ref in zip(batch_records, native_reference, strict=True):
                ids = torch.tensor([record.ids])
                with torch.no_grad():
                    logits = model(ids, torch.ones_like(ids)).logits
                diffs.append(
                    {
                        "max_abs_diff": float((logits - ref).abs().max()),
                        "mean_abs_diff": float((logits - ref).abs().mean()),
                    }
                )
        return {
            "macro_dev_nll": macro(metrics),
            "groups": metrics["groups"],
            "per_example": per_example_scores(model, dev_encoded, parents.tokenizer),
            "logit_diffs_vs_native": diffs,
        }

    def audit_child(child, execution_order):
        executed = list(child.prefix.layers) + list(child.graft.layers) + list(child.suffix.layers)
        changed_blocks = []
        for position, block in enumerate(executed):
            if hash_state(dict(block.named_parameters())) != rec_hashes[execution_order[position]]:
                changed_blocks.append(position)
        return {
            "blocks_checked": len(executed),
            "blocks_hash_identical": len(executed) - len(changed_blocks),
            "changed_block_positions": changed_blocks,
            "tensors_per_block": len(dict(executed[0].named_parameters())),
        }

    recipe_c = Recipe(
        roles[0], roles[1], LayerSpan(11, 13), LayerSpan(11, 13), 42, positive_control=True
    )
    report = {
        "source_commit": source,
        "set": args.set,
        "diagnostic_batch_ids_sha256": batch_ids_sha,
        "controls": {},
        "conditions": [],
    }
    clock = time.monotonic()

    def build(order, seam="bypass", donor=None, graft=(11, 13)):
        return build_native_depth_child(
            parents.recipient,
            execution_order=order,
            graft_span=graft,
            seam_mode=seam,
            recipe=recipe_c,
            donor_parent=donor,
        )

    def record_condition(cond):
        started = time.monotonic()
        try:
            child, manifest = build(cond["execution_order"])
            entry = evaluate_model(child)
            entry.update(
                {
                    "id": cond["id"],
                    "type": cond["type"],
                    "params": cond["params"],
                    "max_abs_displacement": cond["max_abs_displacement"],
                    "n_blocks_displaced": cond["n_blocks_displaced"],
                    "mean_source_position_norm": cond["mean_source_position_norm"],
                    "displacements": cond["displacements"],
                    "manifest": manifest,
                    "block_audit": audit_child(child, cond["execution_order"]),
                    "status": "scored",
                    "runtime_seconds": time.monotonic() - started,
                }
            )
            del child
        except Exception as error:  # recorded, never silently dropped
            entry = {"id": cond["id"], "status": "failed", "error": repr(error)}
        report["conditions"].append(entry)
        write_json(out / "results_partial.json", report)
        gc.collect()

    # Native anchor(s).
    main._stage = "native"
    report["controls"]["native_first"] = evaluate_model(parents.recipient, child=False)
    if args.set == "map":
        report["controls"]["native_second"] = evaluate_model(parents.recipient, child=False)
        report["controls"]["native_third"] = evaluate_model(parents.recipient, child=False)

        main._stage = "controls"
        child_b = child_from_parents(base, parents.recipient, parents.donor)
        child_b.input_seam = torch.nn.Identity()
        child_b.output_seam = torch.nn.Identity()
        b_entry = evaluate_model(child_b)
        b_entry["block_audit"] = audit_child(child_b, list(range(DEPTH)))
        report["controls"]["historical_reconstruction"] = b_entry
        del child_b
        gc.collect()

        for name, seam, donor in (
            ("engine_identity_residual", "residual", parents.donor),
            ("engine_noop_bypass", "bypass", None),
        ):
            child, manifest = build(list(range(DEPTH)), seam=seam, donor=donor)
            entry = evaluate_model(child)
            entry["block_audit"] = audit_child(child, list(range(DEPTH)))
            entry["manifest_moved"] = manifest["blocks_moved"]
            report["controls"][name] = entry
            del child
            gc.collect()

        main._stage = "invalid_probes"
        probes = {}

        def probe(fn):
            try:
                fn()
                return False
            except (ValueError, AssertionError):
                return True

        bad_dup = list(range(DEPTH))
        bad_dup[5] = bad_dup[4]
        bad_missing = list(range(DEPTH - 1))
        bad_unknown = list(range(DEPTH))
        bad_unknown[3] = 99
        declared = list(range(DEPTH))
        swapped = list(range(DEPTH))
        swapped[15], swapped[16] = swapped[16], swapped[15]
        _, probe_manifest = build(swapped)
        probes["duplicate_identity_rejected"] = probe(
            lambda: validate_execution_order(bad_dup, DEPTH)
        )
        probes["missing_block_rejected"] = probe(
            lambda: validate_execution_order(bad_missing, DEPTH)
        )
        probes["unknown_identity_rejected"] = probe(
            lambda: validate_execution_order(bad_unknown, DEPTH)
        )
        probes["order_mismatch_detected"] = probe_manifest["execution_order"] != declared
        probes["manifest_is_permutation"] = sorted(probe_manifest["execution_order"]) == declared
        report["controls"]["invalid_probes"] = probes

        main._stage = "inverse_controls"
        inverse = []
        dev_by_id = {c["id"]: c for c in SETS["development"]}
        for cid in ("dev-swap-14", "dev-move-b14-d+3", "dev-xchg-a4-c18-L2"):
            order = dev_by_id[cid]["execution_order"]
            inv = [0] * DEPTH
            for position, block in enumerate(order):
                inv[block] = position
            restored = [order[inv[i]] for i in range(DEPTH)]
            child, _ = build(inv)
            entry = evaluate_model(child)
            inverse.append(
                {
                    "condition": cid,
                    "permutation_composition_is_identity": restored == list(range(DEPTH)),
                    "macro_dev_nll": entry["macro_dev_nll"],
                    "logit_max_abs_vs_native": max(
                        d["max_abs_diff"] for d in entry["logit_diffs_vs_native"]
                    ),
                }
            )
            del child
            gc.collect()
        report["controls"]["inverse_operations"] = inverse

        main._stage = "map_development"
        for cond in SETS["development"]:
            record_condition(cond)
        main._stage = "map_validation"
        for cond in SETS["validation"]:
            record_condition(cond)

        main._stage = "mechanism"
        report["mechanism"] = mechanism_subset(
            parents, build, batch_records, dev_by_id, evaluate_model
        )
    else:
        main._stage = "heldout_identity_control"
        child, _ = build(list(range(DEPTH)), seam="residual", donor=parents.donor)
        entry = evaluate_model(child)
        entry["block_audit"] = audit_child(child, list(range(DEPTH)))
        report["controls"]["engine_identity_residual"] = entry
        del child
        gc.collect()
        main._stage = "heldout_conditions"
        for cond in SETS["heldout"]:
            record_condition(cond)

    main._stage = "finalize"
    report["parents_unchanged"] = module_fingerprint(parents.recipient) == recipient_before
    report["runtime_seconds"] = time.monotonic() - clock
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(out / "results.json", report)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "set": args.set,
            "parents_unchanged": report["parents_unchanged"],
            "native_macro": report["controls"]["native_first"]["macro_dev_nll"],
            "conditions_scored": sum(1 for c in report["conditions"] if c["status"] == "scored"),
            "conditions_failed": sum(1 for c in report["conditions"] if c["status"] != "scored"),
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    print(f"Campaign 18 {args.set} complete; parents_unchanged:", report["parents_unchanged"])


def mechanism_subset(parents, build, batch_records, dev_by_id, evaluate_model):
    """Position-wise hidden-state RMS ratio / cosine vs native (descriptive)."""
    wanted = ["dev-swap-14", "dev-move-b14-d+8", "dev-xchg-a4-c24-L2"]
    ids = torch.tensor([batch_records[0].ids])
    mask = torch.ones_like(ids)

    def capture(model, layers):
        store = []
        handles = []

        def hook(_m, _i, output):
            store.append(output[0].detach().float())

        for layer in layers:
            handles.append(layer.register_forward_hook(hook))
        with torch.no_grad():
            model(ids, mask)
        for h in handles:
            h.remove()
        return store

    native_layers = list(parents.recipient.model.layers)
    native_out = capture(parents.recipient, native_layers)
    results = {}
    for cid in wanted:
        cond = dev_by_id[cid]
        child, _ = build(cond["execution_order"])
        layers = list(child.prefix.layers) + list(child.graft.layers) + list(child.suffix.layers)
        out_states = capture(child, layers)
        rows = []
        for position in range(DEPTH):
            a, b = native_out[position], out_states[position]
            rows.append(
                {
                    "position": position,
                    "executed_block": cond["execution_order"][position],
                    "rms_ratio": float(b.norm() / a.norm()),
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
                    ),
                }
            )
        results[cid] = rows
        del child
        gc.collect()
    return results


if __name__ == "__main__":
    main()
