#!/usr/bin/env python3
"""Reconstruct an actual bridge-only child using immutable, verified parent snapshots."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from prepare_parents import acquire

from equylaptatemps.artifacts import save_artifact
from equylaptatemps.benchmarks import generate_tokens
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import Recipe
from equylaptatemps.references import load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--parent-cache", default="models/pretrained")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", help="Optional new self-contained checkpoint directory (large)")
    parser.add_argument("--prompt", default="The study of mathematics")
    args = parser.parse_args()
    descriptor = json.loads((Path(args.reference) / "reference.json").read_text())
    recipe_data = descriptor["recipe"]
    local_data = json.loads(json.dumps(recipe_data))
    for role in ("recipient", "donor"):
        source = recipe_data[role]
        actual = acquire(source["model_id"], Path(args.parent_cache), source["revision"])
        local_data[role] = asdict(actual)
    parents = load_parents(Recipe.from_dict(local_data), dtype=torch.float32)
    child = load_reference(args.reference, parents).to(args.device)
    text, tokens = generate_tokens(
        child, parents.tokenizer, args.prompt, max_input_tokens=1024, max_new_tokens=16
    )
    print(
        json.dumps(
            {
                "output": text,
                "generated_ids": tokens,
                "capability_status": "UNPROVEN",
                "structure": child.structure_report(),
            }
        )
    )
    if args.out:
        save_artifact(
            child,
            parents.tokenizer,
            args.out,
            parents.provenance,
            training_status="seam_calibrated",
        )


if __name__ == "__main__":
    main()
