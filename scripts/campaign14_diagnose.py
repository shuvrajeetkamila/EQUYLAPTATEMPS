#!/usr/bin/env python3
"""Campaign 14 reconstruction diagnostics (no training is performed).

Loads SmolLM-135M once per role and evaluates the frozen variant
matrix: native recipient (twice), native-placement children with
bypassed / standard / residual-identity seams, the minimal cut
surgery, and the Campaign 13 placement with seams neutralized or
standard. Records per-example scores, parameter-integrity audits,
output-equivalence metrics, ablations, and graft activity.
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
from campaign02_align import measure_initial
from campaign03_screen import macro
from campaign13_runner import graft_delta, install
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.alignment import initialize_conditioned, install_interfaces
from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.evaluation import collate, evaluate_encoded, predict
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
DEPTH = 30

VARIANTS = {
    "V1": {"cut": (24, 30), "sector": (24, 30), "seams": "bypass"},
    "V2": {"cut": (24, 30), "sector": (24, 30), "seams": "standard"},
    "V3": {"cut": (24, 30), "sector": (24, 30), "seams": "residual"},
    "V4": {"cut": (11, 13), "sector": (11, 13), "seams": "residual"},
    "V5": {"cut": (11, 13), "sector": (24, 30), "seams": "residual"},
    "V6": {"cut": (11, 13), "sector": (24, 30), "seams": "standard"},
}


def per_example_scores(model, encoded, tokenizer):
    """Per-example token NLLs via the canonical collate/predict path."""
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


def block_params(module):
    return {name: p.detach() for name, p in module.named_parameters()}


def parameter_audit(child, recipient, cut, sector):
    """Tensor-by-tensor audit of the child against the recipient."""
    rec_blocks = list(recipient.model.layers)
    locations = []
    for i in range(cut[0]):
        locations.append((child.prefix.layers[i], i, f"prefix.{i}"))
    for j in range(sector[0], sector[1]):
        locations.append((child.graft.layers[j - sector[0]], j, f"graft.{j - sector[0]}"))
    for i in range(cut[1], DEPTH):
        locations.append((child.suffix.layers[i - cut[1]], i, f"suffix.{i - cut[1]}"))
    identical = changed = audited = 0
    changed_names = []
    used = []
    for module, rec_index, where in locations:
        used.append(rec_index)
        ref = block_params(rec_blocks[rec_index])
        mine = block_params(module)
        assert set(ref) == set(mine), f"tensor name mismatch at {where}"
        for name in ref:
            audited += 1
            if torch.equal(ref[name], mine[name]):
                identical += 1
            else:
                changed += 1
                changed_names.append(f"{where}.{name}")
    pairs = [
        ("embedding", child.embedding.weight, recipient.model.embed_tokens.weight),
        ("final_norm", child.final_norm.weight, recipient.model.norm.weight),
        ("lm_head", child.lm_head.weight, recipient.lm_head.weight),
    ]
    top = {}
    for name, mine, ref in pairs:
        top[name] = bool(torch.equal(mine.detach(), ref.detach()))
        audited += 1
        identical += int(top[name])
        changed += int(not top[name])
    removed = sorted(set(range(DEPTH)) - set(used))
    duplicated = sorted(i for i in set(used) if used.count(i) > 1)
    seam_params = sum(p.numel() for p in child.seam_parameters())
    return {
        "tensors_audited": audited,
        "tensors_identical": identical,
        "tensors_changed": changed,
        "changed_names": changed_names[:8],
        "top_level_identical": top,
        "recipient_blocks_removed": removed,
        "recipient_blocks_duplicated": duplicated,
        "recipient_tied_weights": bool(
            recipient.lm_head.weight.data_ptr() == recipient.model.embed_tokens.weight.data_ptr()
        ),
        "child_head_embedding_shared": bool(
            child.lm_head.weight.data_ptr() == child.embedding.weight.data_ptr()
        ),
        "extra_seam_parameters": seam_params,
    }


def build_child(spec, roles, parents, stats_text):
    recipe = Recipe(
        roles[0],
        roles[1],
        LayerSpan(*spec["cut"]),
        LayerSpan(*spec["sector"]),
        42,
        positive_control=True,
    )
    child = child_from_parents(recipe, parents.recipient, parents.donor)
    init_record = None
    if spec["seams"] == "bypass":
        child.input_seam = torch.nn.Identity()
        child.output_seam = torch.nn.Identity()
    elif spec["seams"] == "residual":
        install_interfaces(
            child, {"architecture": "residual_identity", "initialization": "residual_identity"}
        )
    elif spec["seams"] == "standard":
        install(child, {"architecture": "layernorm_linear"})
        stats = measure_initial(child, parents, stats_text)
        init_record = initialize_conditioned(
            child, "natgeom_in_trained_out", statistics=stats, seed=42
        )
        del stats
    return child, init_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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

    acquire_benchmarks("data/local/pretrained-benchmarks")
    _, data = load_benchmarks("data/local/pretrained-benchmarks")
    train_math, dev_math = gsm_splits(data["gsm8k"]["train"], seed=42, train_count=32, dev_count=16)
    train_text = text_chunks(data["wikitext2"]["train"], characters=192, limit=32)
    dev_text = text_chunks(data["wikitext2"]["validation"], characters=192, limit=16)
    stats_text = train_text[:8]
    dev_encoded, _ = encode_math(dev_math, parents.tokenizer, 512, filter_oversize=True)
    dev_encoded += encode_text(dev_text, parents.tokenizer, 512)
    batch_records = dev_encoded[:4]
    batch_ids_sha = hashlib.sha256(json.dumps([r.ids for r in batch_records]).encode()).hexdigest()

    def native_logits(record):
        ids = torch.tensor([record.ids])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            return parents.recipient(input_ids=ids, attention_mask=mask).logits

    native_reference = [native_logits(r) for r in batch_records]

    def evaluate_variant(model):
        metrics = evaluate_encoded(model, dev_encoded, parents.tokenizer)
        entry = {
            "macro_dev_nll": macro(metrics),
            "groups": metrics["groups"],
            "per_example": per_example_scores(model, dev_encoded, parents.tokenizer),
            "bypass_macro": macro(
                evaluate_encoded(model, dev_encoded, parents.tokenizer, ablation="bypass_graft")
            )
            if hasattr(model, "graft")
            else None,
            "zero_macro": macro(
                evaluate_encoded(model, dev_encoded, parents.tokenizer, ablation="zero_graft")
            )
            if hasattr(model, "graft")
            else None,
        }
        diffs = []
        if hasattr(model, "graft"):
            for record, ref in zip(batch_records, native_reference, strict=True):
                ids = torch.tensor([record.ids])
                mask = torch.ones_like(ids)
                with torch.no_grad():
                    logits = model(ids, mask).logits
                diffs.append(
                    {
                        "max_abs_diff": float((logits - ref).abs().max()),
                        "mean_abs_diff": float((logits - ref).abs().mean()),
                    }
                )
        entry["logit_diffs_vs_native"] = diffs
        return entry

    report = {
        "source_commit": source,
        "diagnostic_batch_ids_sha256": batch_ids_sha,
        "variants": {},
    }
    clock = time.monotonic()
    first = evaluate_variant(parents.recipient)
    second = evaluate_variant(parents.recipient)
    report["variants"]["V0"] = {
        "kind": "native_recipient",
        "first": first,
        "second": second,
        "macro_repeat_abs_diff": abs(first["macro_dev_nll"] - second["macro_dev_nll"]),
    }
    for name, vspec in VARIANTS.items():
        start = time.monotonic()
        child, init_record = build_child(vspec, roles, parents, stats_text)
        entry = evaluate_variant(child)
        entry.update(
            {
                "kind": "surgical_child",
                "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vspec.items()},
                "initialization": init_record,
                "parameter_audit": parameter_audit(
                    child, parents.recipient, vspec["cut"], vspec["sector"]
                ),
                "initial_graft_activity": graft_delta(child, parents.tokenizer, stats_text),
                "structure": child.structure_report(),
                "runtime_seconds": time.monotonic() - start,
            }
        )
        report["variants"][name] = entry
        del child
        gc.collect()
    report["parents_unchanged"] = module_fingerprint(parents.recipient) == recipient_before
    report["runtime_seconds"] = time.monotonic() - clock
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(out / "diagnostics.json", report)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "diagnostics_only": True,
            "parents_unchanged": report["parents_unchanged"],
            "macro": {
                k: v.get("macro_dev_nll", v.get("first", {}).get("macro_dev_nll"))
                for k, v in report["variants"].items()
            },
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    print("Campaign 14 diagnostics complete; parents_unchanged:", report["parents_unchanged"])


if __name__ == "__main__":
    main()
