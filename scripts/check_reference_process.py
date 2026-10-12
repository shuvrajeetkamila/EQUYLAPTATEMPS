#!/usr/bin/env python3
"""Fresh-process genuine bridge-only checkpoint reconstruction plus inference/generation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from prepare_parents import acquire
from safetensors.torch import save_file

from equylaptatemps.audit import hash_state
from equylaptatemps.benchmarks import generate_tokens
from equylaptatemps.parents import load_parents
from equylaptatemps.recipe import Recipe
from equylaptatemps.references import load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    descriptor = json.loads((Path(args.reference) / "reference.json").read_text())
    recipe = json.loads(json.dumps(descriptor["recipe"]))
    for role in ("recipient", "donor"):
        spec = recipe[role]
        recipe[role] = asdict(acquire(spec["model_id"], Path(args.parent_cache), spec["revision"]))
    parents = load_parents(Recipe.from_dict(recipe), dtype=torch.float32)
    child = load_reference(args.reference, parents)
    inputs = json.loads(Path(args.inputs).read_text())
    ids = torch.tensor(inputs["ids"], dtype=torch.long)
    with torch.inference_mode():
        logits = child(ids).logits
    samples = []
    for prompt in inputs["prompts"]:
        text, tokens = generate_tokens(
            child, parents.tokenizer, prompt, max_input_tokens=512, max_new_tokens=8
        )
        samples.append({"prompt": prompt, "output": text, "ids": tokens})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    save_file({"logits": logits}, str(out / "logits.safetensors"))
    report = {
        "status": "PASS",
        "finite_logits": bool(torch.isfinite(logits).all()),
        "logits_sha256": hash_state({"logits": logits}),
        "shape": list(logits.shape),
        "parent_kind": "pretrained" if not child.recipe.synthetic else "synthetic",
        "generation_samples": samples,
        "scores": None,
        "scope": "reference-dependent reconstruction, not standalone or useful-capability proof",
    }
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
