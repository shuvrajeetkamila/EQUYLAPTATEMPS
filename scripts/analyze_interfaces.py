#!/usr/bin/env python3
"""Actual interface conditioning, not model performance or inheritance."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from equylaptatemps.audit import sha256_file
from equylaptatemps.model import DenseSeam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="examples/pretrained-pilot-reference")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.reference)
    descriptor = json.loads((root / "reference.json").read_text())
    file = root / "interfaces.safetensors"
    if sha256_file(file) != descriptor["interfaces_sha256"]:
        raise ValueError("Interface checkpoint changed")
    state = load_file(str(file))
    width_r = descriptor["structure"]["recipient_width"]
    width_d = descriptor["structure"]["donor_width"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(descriptor["recipe"]["seed"])
        initial_input = DenseSeam(width_r, width_d)
        initial_output = DenseSeam(width_d, width_r)
    output = {
        "kind": "measured_checkpoint_parameters_not_benchmark",
        "source_interfaces_sha256": sha256_file(file),
        "matrices": {},
        "warning": "Conditioning is not proof of semantic alignment or inheritance",
    }
    for role, module in (("input_seam", initial_input), ("output_seam", initial_output)):
        weight = state[role + ".projection.weight"].float()
        initial = module.projection.weight.detach()
        singular = torch.linalg.svdvals(weight)
        output["matrices"][role] = {
            "shape": list(weight.shape),
            "mean": float(weight.mean()),
            "std": float(weight.std()),
            "frobenius_norm": float(weight.norm()),
            "relative_frobenius_change_from_same_seed_initialization": float(
                (weight - initial).norm() / initial.norm()
            ),
            "singular_min": float(singular.min()),
            "singular_max": float(singular.max()),
            "condition_number": float(singular.max() / singular.min()),
            "numerical_rank": int(
                (
                    singular > singular.max() * max(weight.shape) * torch.finfo(weight.dtype).eps
                ).sum()
            ),
            "norm_scale_mean": float(state[role + ".norm.weight"].mean()),
            "norm_bias_abs_mean": float(state[role + ".norm.bias"].abs().mean()),
        }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Keep prior interface analysis; use a new output file")
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
