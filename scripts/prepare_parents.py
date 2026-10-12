#!/usr/bin/env python3
"""Run on a machine with Hugging Face access. Pin revisions and record exact local file hashes.

This script was NOT run against pretrained models in the Arena sandbox. No credentials are
requested, written, or printed. The standard Hub client's existing environment is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig

from equylaptatemps.artifacts import write_json
from equylaptatemps.audit import sha256_file
from equylaptatemps.provenance import SOURCE_RECORD, snapshot_files
from equylaptatemps.recipe import REGISTRY, LayerSpan, ParentSpec, Recipe

SUPPORT_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "LICENSE*",
    "NOTICE*",
    "README.md",
]


def acquire(model_id: str, root: Path) -> ParentSpec:
    if model_id not in REGISTRY or REGISTRY[model_id].synthetic:
        raise ValueError("Choose a pretrained parent from the reviewed provenance registry")
    info = HfApi().model_info(model_id, files_metadata=True)
    revision = info.sha
    # Record an immutable commit BEFORE downloading; never save a recipe that references 'main'.
    directory = (root / f"{model_id.rsplit('/', 1)[1]}--{revision[:12]}").resolve()
    spec = ParentSpec(model_id, str(directory), revision)
    spec.validate()
    names = [sibling.rfilename for sibling in info.siblings]
    if any(name.endswith(".safetensors") for name in names):
        weights = ["*.safetensors", "*.safetensors.index.json"]
    else:
        raise ValueError(
            f"{model_id} has no safetensors; this acquisition script refuses pickle weights"
        )
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(directory),
        allow_patterns=[*SUPPORT_FILES, *weights],
    )
    inventory = {
        relative: sha256_file(file) for relative, file in snapshot_files(directory).items()
    }
    if "config.json" not in inventory or not any(
        name.endswith(".safetensors") for name in inventory
    ):
        raise ValueError("Downloaded snapshot is incomplete")
    remote_hashes = {
        sibling.rfilename: sibling.lfs.sha256
        for sibling in info.siblings
        if sibling.lfs is not None and sibling.rfilename in inventory
    }
    for name, expected in remote_hashes.items():
        if inventory[name] != expected:
            raise ValueError(f"Downloaded file differs from Hub LFS SHA-256: {name}")
    write_json(
        directory / SOURCE_RECORD,
        {
            "format_version": 1,
            "model_id": model_id,
            "revision": revision,
            "files": inventory,
            "hub_lfs_sha256": remote_hashes,
        },
    )
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipient", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--donor", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/local-parents.json"))
    parser.add_argument(
        "--recipient-cut", type=int, nargs=2, default=[8, 16], metavar=("START", "STOP")
    )
    parser.add_argument(
        "--donor-sector", type=int, nargs=2, default=[8, 16], metavar=("START", "STOP")
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new --out recipe path; refusing to overwrite an existing recipe")
    # Reject cousins before downloading any bytes.
    if args.recipient not in REGISTRY or args.donor not in REGISTRY:
        parser.error("Both models must belong to the reviewed registry")
    a, b = REGISTRY[args.recipient], REGISTRY[args.donor]
    if a.family == b.family or a.lineage & b.lineage or a.synthetic or b.synthetic:
        parser.error("Choose two distinct pretrained architecture and lineage families")
    cut, sector = LayerSpan.from_list(args.recipient_cut), LayerSpan.from_list(args.donor_sector)
    recipient, donor = acquire(args.recipient, args.models), acquire(args.donor, args.models)
    recipe = Recipe(recipient, donor, cut, sector)
    configs = [
        AutoConfig.from_pretrained(spec.local_path, local_files_only=True, trust_remote_code=False)
        for spec in (recipient, donor)
    ]
    recipe.validate(*configs)
    write_json(args.out, recipe.to_dict())
    print(f"Pinned local-parent recipe written to {args.out}")
    print("Inspect parent licenses. No pretrained transplant or capability validation has run yet.")


if __name__ == "__main__":
    main()
