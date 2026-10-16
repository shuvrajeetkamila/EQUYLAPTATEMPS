#!/usr/bin/env python3
"""Campaign 17 native-depth placement diagnostics (no training).

Frozen modes (campaigns/campaign_17/config/run_matrix.json): A native
recipient twice; B exact reconstruction through the historical
child_from_parents path (registered sector replaced by itself,
seams bypassed); C identity replacement through the native-depth
placement engine (same sector, residual-identity seams); D1
two-sector exchange [11,13)<->[24,26); D2 adjacent block swap 15<->16
(both bypassed, diagnostic only). Self-donor throughout.
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
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.evaluation import collate, evaluate_encoded, predict
from equylaptatemps.model import child_from_parents
from equylaptatemps.native_placement import build_native_depth_child
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
DEPTH = 30
MACRO_MAX = 0.01
LOGIT_MAX_MAX = 0.02
LOGIT_MEAN_MAX = 0.001
REPEAT_MAX = 0.001


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


def block_audit(child, recipient, donor, execution_order, graft_span, donor_role):
    """Tensor audit of the executed blocks against their sources."""
    executed = list(child.prefix.layers) + list(child.graft.layers) + list(child.suffix.layers)
    assert len(executed) == DEPTH
    rec_blocks = list(recipient.model.layers)
    donor_blocks = list(donor.model.layers)
    identical = changed = audited = 0
    changed_names = []
    for position, module in enumerate(executed):
        source_index = execution_order[position]
        from_donor = donor_role and graft_span[0] <= position < graft_span[1]
        reference = (donor_blocks if from_donor else rec_blocks)[source_index]
        mine = dict(module.named_parameters())
        refd = dict(reference.named_parameters())
        assert set(mine) == set(refd), f"tensor name mismatch at position {position}"
        for name in mine:
            audited += 1
            if torch.equal(mine[name].detach(), refd[name].detach()):
                identical += 1
            else:
                changed += 1
                changed_names.append(f"pos{position}.block{source_index}.{name}")
    top = {
        "embedding_identical": bool(
            torch.equal(
                child.embedding.weight.detach(), recipient.model.embed_tokens.weight.detach()
            )
        ),
        "final_norm_identical": bool(
            torch.equal(child.final_norm.weight.detach(), recipient.model.norm.weight.detach())
        ),
        "lm_head_identical": bool(
            torch.equal(child.lm_head.weight.detach(), recipient.lm_head.weight.detach())
        ),
    }
    return {
        "tensors_audited": audited,
        "tensors_identical": identical,
        "tensors_changed": changed,
        "changed_names": changed_names[:8],
        "top_level": top,
        "recipient_tied_weights": bool(
            recipient.lm_head.weight.data_ptr() == recipient.model.embed_tokens.weight.data_ptr()
        ),
        "child_head_embedding_shared": bool(
            child.lm_head.weight.data_ptr() == child.embedding.weight.data_ptr()
        ),
        "extra_seam_parameters": sum(p.numel() for p in child.seam_parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
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

    acquire_benchmarks("data/local/pretrained-benchmarks")
    _, data = load_benchmarks("data/local/pretrained-benchmarks")
    _, dev_math = gsm_splits(data["gsm8k"]["train"], seed=42, train_count=32, dev_count=16)
    dev_text = text_chunks(data["wikitext2"]["validation"], characters=192, limit=16)
    dev_encoded, _ = encode(dev_math, dev_text, parents)
    batch_records = dev_encoded[:4]
    batch_ids_sha = hashlib.sha256(json.dumps([r.ids for r in batch_records]).encode()).hexdigest()

    def native_logits(record):
        ids = torch.tensor([record.ids])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            return parents.recipient(input_ids=ids, attention_mask=mask).logits

    native_reference = [native_logits(r) for r in batch_records]

    def evaluate_model(model, *, child=True):
        metrics = evaluate_encoded(model, dev_encoded, parents.tokenizer)
        diffs = []
        if child:
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
        return {
            "macro_dev_nll": macro(metrics),
            "groups": metrics["groups"],
            "per_example": per_example_scores(model, dev_encoded, parents.tokenizer),
            "logit_diffs_vs_native": diffs,
            "logit_equivalence_pass": bool(diffs)
            and max(d["max_abs_diff"] for d in diffs) <= LOGIT_MAX_MAX
            and max(d["mean_abs_diff"] for d in diffs) <= LOGIT_MEAN_MAX,
        }

    report = {
        "source_commit": source,
        "diagnostic_batch_ids_sha256": batch_ids_sha,
        "frozen_tolerances": {
            "macro_exactness_max_abs_delta": MACRO_MAX,
            "logits_max_abs_max": LOGIT_MAX_MAX,
            "logits_mean_abs_max": LOGIT_MEAN_MAX,
            "native_repeat_max_delta": REPEAT_MAX,
        },
        "modes": {},
    }
    clock = time.monotonic()
    main._stage = "mode_a_native"
    first = evaluate_model(parents.recipient, child=False)
    second = evaluate_model(parents.recipient, child=False)
    report["modes"]["A_native"] = {
        "kind": "native_recipient",
        "first": first,
        "second": second,
        "macro_repeat_abs_diff": abs(first["macro_dev_nll"] - second["macro_dev_nll"]),
    }
    write_json(out / "diagnostics_partial.json", report)

    def record_mode(name, child, execution_order, graft_span, donor_role, manifest=None):
        entry = evaluate_model(child)
        entry.update(
            {
                "kind": "surgical_child",
                "execution_order": execution_order,
                "graft_span_execution_coordinates": list(graft_span),
                "block_audit": block_audit(
                    child, parents.recipient, parents.donor, execution_order, graft_span, donor_role
                ),
                "structure": child.structure_report(),
                "manifest": manifest,
            }
        )
        report["modes"][name] = entry
        write_json(out / "diagnostics_partial.json", report)

    main._stage = "mode_b_historical"
    child_b = child_from_parents(base, parents.recipient, parents.donor)
    child_b.input_seam = torch.nn.Identity()
    child_b.output_seam = torch.nn.Identity()
    record_mode(
        "B_historical_reconstruction",
        child_b,
        list(range(DEPTH)),
        (24, 30),
        donor_role=True,
    )
    del child_b
    gc.collect()

    recipe_c = Recipe(
        roles[0], roles[1], LayerSpan(11, 13), LayerSpan(11, 13), 42, positive_control=True
    )
    main._stage = "mode_c_identity"
    child_c, manifest_c = build_native_depth_child(
        parents.recipient,
        execution_order=list(range(DEPTH)),
        graft_span=(11, 13),
        seam_mode="residual",
        recipe=recipe_c,
        donor_parent=parents.donor,
    )
    record_mode("C_identity_replacement", child_c, list(range(DEPTH)), (11, 13), True, manifest_c)
    del child_c
    gc.collect()

    main._stage = "mode_d1_exchange"
    d1_order = list(range(11)) + [24, 25] + list(range(13, 24)) + [11, 12] + list(range(26, 30))
    child_d1, manifest_d1 = build_native_depth_child(
        parents.recipient,
        execution_order=d1_order,
        graft_span=(11, 13),
        seam_mode="bypass",
        recipe=recipe_c,
    )
    record_mode("D1_sector_exchange", child_d1, d1_order, (11, 13), False, manifest_d1)
    del child_d1
    gc.collect()

    main._stage = "mode_d2_swap"
    d2_order = list(range(DEPTH))
    d2_order[15], d2_order[16] = d2_order[16], d2_order[15]
    child_d2, manifest_d2 = build_native_depth_child(
        parents.recipient,
        execution_order=d2_order,
        graft_span=(15, 17),
        seam_mode="bypass",
        recipe=recipe_c,
    )
    record_mode("D2_block_swap", child_d2, d2_order, (15, 17), False, manifest_d2)
    del child_d2
    gc.collect()

    main._stage = "finalize"
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
                "A_native": report["modes"]["A_native"]["first"]["macro_dev_nll"],
                **{k: v["macro_dev_nll"] for k, v in report["modes"].items() if k != "A_native"},
            },
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    print("Campaign 17 diagnostics complete; parents_unchanged:", report["parents_unchanged"])


def encode(dev_math, dev_text, parents):
    from equylaptatemps.objectives import encode_math, encode_text

    encoded, _ = encode_math(dev_math, parents.tokenizer, 512, filter_oversize=True)
    encoded += encode_text(dev_text, parents.tokenizer, 512)
    return encoded, None


if __name__ == "__main__":
    main()
