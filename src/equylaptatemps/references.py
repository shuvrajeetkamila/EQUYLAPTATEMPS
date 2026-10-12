"""Small bridge-only checkpoints: immutable parent downloads are required, never bundled here."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from .audit import frozen_fingerprint, module_fingerprint, sha256_file
from .model import child_from_parents
from .recipe import Recipe


def save_reference(child, target: str | Path, provenance: dict) -> dict:
    target = Path(target)
    if target.exists():
        raise FileExistsError("Use a new bridge-only checkpoint directory")
    child.assert_trainable_contract()
    target.mkdir(parents=True)
    seams = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in child.state_dict().items()
        if name.startswith(("input_seam.", "output_seam."))
    }
    save_file(seams, str(target / "interfaces.safetensors"))
    descriptor = {
        "format_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "self_contained": False,
        "parent_kind": "synthetic" if child.recipe.synthetic else "pretrained",
        "capability_status": "UNPROVEN",
        "recipe": child.recipe.to_dict(),
        "dtype": str(child.embedding.weight.dtype).removeprefix("torch."),
        "structure": child.structure_report(),
        "provenance": provenance,
        "frozen_sha256": frozen_fingerprint(child),
        "graft_sha256": module_fingerprint(child.graft.layers),
        "interfaces_sha256": sha256_file(target / "interfaces.safetensors"),
        "reconstruction": "Acquire/hash pinned parents at dtype, then call load_reference",
    }
    (target / "reference.json").write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
    return descriptor


def load_reference(target: str | Path, parents):
    """Require verified parents; source paths are descriptive only."""
    target = Path(target)
    descriptor = json.loads((target / "reference.json").read_text())
    if descriptor.get("format_version") != 1 or descriptor.get("self_contained") is not False:
        raise ValueError("Invalid bridge-only checkpoint format")
    if sha256_file(target / "interfaces.safetensors") != descriptor["interfaces_sha256"]:
        raise ValueError("Bridge-only checkpoint checksum mismatch")
    recipe = Recipe.from_dict(descriptor["recipe"])
    expected_kind = "synthetic" if recipe.synthetic else "pretrained"
    if descriptor["parent_kind"] != expected_kind:
        raise ValueError("Reference checkpoint parent-kind label mismatch")
    expected_tokens = descriptor["provenance"]["tokenizers"]
    if parents.provenance["tokenizers"] != expected_tokens:
        raise ValueError("Reference checkpoint tokenizer contracts differ")
    for role in ("recipient", "donor"):
        expected = descriptor["provenance"].get("snapshots", {}).get(role)
        actual = parents.provenance.get("snapshots", {}).get(role)
        if expected is not None and expected.get("snapshot_files_sha256") != actual.get(
            "snapshot_files_sha256"
        ):
            raise ValueError("Reference checkpoint parent snapshot hashes differ")
    child = child_from_parents(recipe, parents.recipient, parents.donor)
    if str(child.embedding.weight.dtype).removeprefix("torch.") != descriptor["dtype"]:
        raise ValueError("Reference checkpoint loading precision differs")
    if (
        frozen_fingerprint(child) != descriptor["frozen_sha256"]
        or module_fingerprint(child.graft.layers) != descriptor["graft_sha256"]
    ):
        raise ValueError("Reference checkpoint pretrained sectors differ")
    state = load_file(str(target / "interfaces.safetensors"), device=str(child.device))
    expected_names = {
        name
        for name, _ in child.named_parameters()
        if name.startswith(("input_seam.", "output_seam."))
    }
    if set(state) != expected_names:
        raise ValueError("Bridge checkpoint must contain exactly the new interface parameters")
    missing, unexpected = child.load_state_dict(state, strict=False)
    if unexpected or any(name in expected_names for name in missing):
        raise ValueError("Incomplete bridge-only loading")
    if child.structure_report() != descriptor["structure"]:
        raise ValueError("Reference checkpoint structure mismatch")
    return child.eval()
