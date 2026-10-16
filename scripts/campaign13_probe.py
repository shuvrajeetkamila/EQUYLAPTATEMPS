#!/usr/bin/env python3
"""Campaign 13 exploratory pair-selection probe (NOT confirmatory).

Resolves the immutable revisions of the candidate same-family pair,
verifies config/tokenizer compatibility, measures both parents
natively on the project dev splits, quantifies the donor sector's
weight delta vs the recipient's own corresponding layers, and
smoke-constructs the four frozen children (no training). Its bundle
is the evidence base for campaigns/campaign_13/model_pair_selection.md
and for pinning exact revisions in the preregistration.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

import torch
from campaign02_align import measure_initial
from campaign03_screen import macro
from campaign13_runner import graft_delta, install
from huggingface_hub import HfApi
from prepare_parents import acquire
from run_pretrained_study import publish_bundle

from equylaptatemps.alignment import initialize_conditioned
from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.benchdata import acquire_benchmarks, gsm_splits, load_benchmarks, text_chunks
from equylaptatemps.controls import (
    random_qwen_sector_control,
    row_shuffled_sector_control,
    shuffled_sector_control,
)
from equylaptatemps.evaluation import evaluate_encoded
from equylaptatemps.model import child_from_parents
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import LayerSpan, Recipe

RECIPIENT_ID = "HuggingFaceTB/SmolLM-135M"
DONOR_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
RECIPIENT_KNOWN_SHA = "1d461723eec654e65efdc40cf49301c89c0c92f4"


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

    api = HfApi()
    resolved = {
        RECIPIENT_ID: api.model_info(RECIPIENT_ID).sha,
        DONOR_ID: api.model_info(DONOR_ID).sha,
    }
    report = {
        "source_commit": source,
        "exploratory": True,
        "resolved_revisions": resolved,
        "recipient_revision_matches_recorded": resolved[RECIPIENT_ID] == RECIPIENT_KNOWN_SHA,
    }
    roles = [
        acquire(RECIPIENT_ID, Path("models/pretrained"), resolved[RECIPIENT_ID]),
        acquire(DONOR_ID, Path("models/pretrained"), resolved[DONOR_ID]),
    ]
    base = Recipe(
        roles[0], roles[1], LayerSpan(11, 13), LayerSpan(24, 30), 42, positive_control=True
    )
    parents = load_parents(base, dtype=torch.float32)
    parents.recipient.requires_grad_(False).eval()
    parents.donor.requires_grad_(False).eval()
    frozen_before = {r: module_fingerprint(getattr(parents, r)) for r in ("recipient", "donor")}
    write_json(out / "provenance.json", parents.provenance)
    report["provenance"] = parents.provenance

    cfg_a = json.loads((Path(roles[0].local_path) / "config.json").read_text())
    cfg_b = json.loads((Path(roles[1].local_path) / "config.json").read_text())
    keys = sorted(set(cfg_a) | set(cfg_b))
    report["config_differences"] = {
        k: {"recipient": cfg_a.get(k), "donor": cfg_b.get(k)}
        for k in keys
        if cfg_a.get(k) != cfg_b.get(k)
    }

    acquire_benchmarks("data/local/pretrained-benchmarks")
    _, data = load_benchmarks("data/local/pretrained-benchmarks")
    _, dev_math = gsm_splits(data["gsm8k"]["train"], seed=42, train_count=32, dev_count=16)
    train_text = text_chunks(data["wikitext2"]["train"], characters=192, limit=32)
    dev_text = text_chunks(data["wikitext2"]["validation"], characters=192, limit=16)
    stats_text = train_text[:8]

    native = {}
    for name, model, tok in (
        ("recipient", parents.recipient, parents.tokenizer),
        ("donor", parents.donor, parents.donor_tokenizer),
    ):
        encoded, _ = encode_math(dev_math, tok, 512, filter_oversize=True)
        encoded += encode_text(dev_text, tok, 512)
        metrics = evaluate_encoded(model, encoded, tok)
        native[name] = {"macro_dev_nll": macro(metrics), "groups": metrics["groups"]}
    report["native_development"] = native

    delta = {}
    rec_layers = parents.recipient.model.layers[24:30]
    don_layers = parents.donor.model.layers[24:30]
    for (name_a, p_a), (name_b, p_b) in zip(
        rec_layers.named_parameters(), don_layers.named_parameters(), strict=True
    ):
        assert name_a == name_b and p_a.shape == p_b.shape
        d = (p_b.float() - p_a.float()).norm().item()
        delta[name_a] = {
            "relative_rms_delta": d / p_a.float().norm().item(),
            "recipient_rms": p_a.float().pow(2).mean().sqrt().item(),
        }
    report["sector_delta_layers_24_30"] = delta
    report["sector_delta_mean_relative"] = sum(
        v["relative_rms_delta"] for v in delta.values()
    ) / len(delta)

    conditions = {
        "A_intact": {},
        "B_row": {"row_shuffled_sector": True, "row_shuffle_seed": 20241014},
        "C_full": {"shuffled_sector": True, "shuffle_seed": 20241014},
        "D_rand": {"random_sector": True, "random_sector_seed": 1234},
    }
    built = {}
    for name, extra in conditions.items():
        exp = {"seed": 42, "interface": {"architecture": "layernorm_linear"}, **extra}
        recipe = Recipe(
            roles[0], roles[1], LayerSpan(11, 13), LayerSpan(24, 30), 42, positive_control=True
        )
        child = build(exp, recipe, parents)
        install(child, exp["interface"])
        stats = measure_initial(child, parents, stats_text)
        initialize_conditioned(child, "natgeom_in_trained_out", statistics=stats, seed=42)
        graft_params = list(child.graft.parameters())
        built[name] = {
            "control_record": getattr(child, "control_record", None),
            "initial_graft_activity": graft_delta(child, parents.tokenizer, stats_text),
            "graft_parameter_count": sum(p.numel() for p in graft_params),
            "all_graft_finite": all(bool(torch.isfinite(p).all()) for p in graft_params),
            "structure": child.structure_report(),
        }
        del child, stats
        gc.collect()
    report["children_seed42"] = built
    frozen_after = {r: module_fingerprint(getattr(parents, r)) for r in ("recipient", "donor")}
    report["parents_unchanged"] = frozen_before == frozen_after
    write_json(out / "probe.json", report)
    write_json(
        out / "session_summary.json",
        {
            "source_commit": source,
            "exploratory_probe": True,
            "parents_unchanged": report["parents_unchanged"],
        },
    )
    if args.post_pr:
        publish_bundle(out, args.post_pr)
    print("Campaign 13 probe complete; donor revision:", resolved[DONOR_ID])


if __name__ == "__main__":
    main()
