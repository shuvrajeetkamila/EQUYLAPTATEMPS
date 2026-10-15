#!/usr/bin/env python3
"""Campaign 13 pre-training control validation (no training is performed).

Constructs the four frozen children (A intact, B within-row shuffled,
C full-tensor shuffled, D random twin) with the exact Campaign 12
construction path, records control fingerprints, initial graft
activity, shapes/parameter counts, and parent-immutability evidence,
and publishes the bundle. Confirmatory training must not start until
this validation has been imported and checked.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
from pathlib import Path

import torch
from campaign02_align import measure_initial
from campaign13_runner import graft_delta, install
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.alignment import initialize_conditioned
from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, load_benchmarks, text_chunks
from equylaptatemps.controls import (
    random_qwen_sector_control,
    row_shuffled_sector_control,
    shuffled_sector_control,
)
from equylaptatemps.model import child_from_parents
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe

CONDITIONS = {
    "A_intact": {},
    "B_row": {"row_shuffled_sector": True, "row_shuffle_seed": 20241014},
    "C_full": {"shuffled_sector": True, "shuffle_seed": 20241014},
    "D_rand": {"random_sector": True, "random_sector_seed": 1234},
}


def build(exp, recipe, parents):
    if exp.get("row_shuffled_sector"):
        return row_shuffled_sector_control(
            recipe, parents.recipient, parents.donor, shuffle_seed=exp["row_shuffle_seed"]
        )
    if exp.get("shuffled_sector"):
        return shuffled_sector_control(
            recipe, parents.recipient, parents.donor, shuffle_seed=exp["shuffle_seed"]
        )
    if exp.get("random_sector"):
        return random_qwen_sector_control(
            recipe, parents.recipient, parents.donor, initialization_seed=exp["random_sector_seed"]
        )
    return child_from_parents(recipe, parents.recipient, parents.donor)


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
    roles = []
    for model_id, revision in (
        ("HuggingFaceTB/SmolLM-135M", "1d461723eec654e65efdc40cf49301c89c0c92f4"),
        ("HuggingFaceTB/SmolLM-135M-Instruct", "DONOR_REVISION_SHA"),
    ):
        roles.append(acquire(model_id, Path("models/pretrained"), revision))
    base = Recipe(
        roles[0], roles[1], LayerSpan(11, 13), LayerSpan(24, 30), 42, positive_control=True
    )
    parents = load_parents(base, dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    frozen_before = {r: module_fingerprint(getattr(parents, r)) for r in ("recipient", "donor")}
    write_json(out / "provenance.json", parents.provenance)
    acquire_benchmarks("data/local/pretrained-benchmarks")
    _, data = load_benchmarks("data/local/pretrained-benchmarks")
    train_text = text_chunks(data["wikitext2"]["train"], characters=192, limit=32)
    stats_text = train_text[:8]

    report = {"source_commit": source, "conditions": {}}
    for name, extra in CONDITIONS.items():
        exp = {"seed": 42, "interface": {"architecture": "layernorm_linear"}, **extra}
        recipe = Recipe(
            roles[0], roles[1], LayerSpan(11, 13), LayerSpan(24, 30), 42, positive_control=True
        )
        child = build(exp, recipe, parents)
        install(child, exp["interface"])
        stats = measure_initial(child, parents, stats_text)
        initialize_conditioned(child, "natgeom_in_trained_out", statistics=stats, seed=42)
        graft_params = list(child.graft.parameters())
        entry = {
            "control_record": getattr(child, "control_record", None),
            "initial_statistics": stats["summary"],
            "initial_graft_activity": graft_delta(child, parents.tokenizer, stats_text),
            "graft_parameter_count": sum(p.numel() for p in graft_params),
            "graft_tensors": [
                {
                    "shape": list(p.shape),
                    "dtype": str(p.dtype),
                    "finite": bool(torch.isfinite(p).all()),
                }
                for p in graft_params
            ],
            "trainable_parameters": sum(p.numel() for p in child.parameters() if p.requires_grad),
            "all_graft_finite": all(bool(torch.isfinite(p).all()) for p in graft_params),
        }
        report["conditions"][name] = entry
        del child, stats
        gc.collect()
    frozen_after = {r: module_fingerprint(getattr(parents, r)) for r in ("recipient", "donor")}
    report["parents_unchanged"] = frozen_before == frozen_after
    report["parent_fingerprints"] = {"before": frozen_before, "after": frozen_after}
    write_json(out / "validation.json", report)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "validation_only": True,
            "parents_unchanged": report["parents_unchanged"],
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    print("Campaign 13 validation complete; parents_unchanged:", report["parents_unchanged"])


if __name__ == "__main__":
    main()
