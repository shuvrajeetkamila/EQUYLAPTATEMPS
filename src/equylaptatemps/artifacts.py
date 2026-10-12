"""Portable child-only safetensors checkpoints with tensor and vocabulary integrity checks."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_model, save_model

from . import __version__
from .audit import audit_child, sha256_file
from .model import SurgicalChild, empty_child
from .parents import DTYPES
from .recipe import Recipe
from .tokenization import Tokenizer, check_tokenizer, load_tokenizer, tokenizer_fingerprint


@dataclass
class StoredChild:
    model: SurgicalChild
    tokenizer: Tokenizer
    manifest: dict


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_artifact(
    child: SurgicalChild,
    tokenizer: Tokenizer,
    target: str | Path,
    provenance: dict,
    *,
    training_status: str = "uncalibrated",
    metrics: dict | None = None,
) -> dict:
    """Write atomically to a NEW directory. Never overwrite a user's existing checkpoint."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {target}")
    if training_status not in {"uncalibrated", "seam_calibrated"}:
        raise ValueError("Invalid training status")
    check_tokenizer(tokenizer, child.recipient_config.vocab_size)
    audit = audit_child(child)
    if (
        provenance.get("source_donor_sector_sha256", audit["donor_sector_sha256"])
        != audit["donor_sector_sha256"]
    ):
        raise ValueError("Donor weights changed after transplantation")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        save_model(
            child, str(staging / "model.safetensors"), metadata={"format": "equylaptatemps-v1"}
        )
        tokenizer.save_pretrained(str(staging / "tokenizer"))
        _parent_notices(staging, provenance)
        manifest = {
            "format_version": 1,
            "package_version": __version__,
            "software_license": "Apache-2.0; parent weight licenses apply separately",
            "recipe": child.recipe.to_dict(),
            "recipient_config": child.recipient_config.to_dict(),
            "donor_config": child.donor_config.to_dict(),
            "dtype": str(child.embedding.weight.dtype).removeprefix("torch."),
            "training_status": training_status,
            "capability_status": "synthetic_smoke_test"
            if child.recipe.synthetic
            else "unvalidated_pretrained_transplant",
            "provenance": provenance,
            "audit": audit,
            "recipient_tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "weights_sha256": sha256_file(staging / "model.safetensors"),
            "metrics": metrics,
        }
        write_json(staging / "child.json", manifest)
        # Guard against an output created during the save operation.
        if target.exists():
            raise FileExistsError(f"Checkpoint destination already exists: {target}")
        os.rename(staging, target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parent_notices(path: Path, provenance: dict, *, verify: bool = False) -> None:
    for role in ("recipient", "donor"):
        evidence = provenance.get("snapshots", {}).get(role, {})
        for name, document in evidence.get("license_notices", {}).items():
            if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
                raise ValueError("Unsafe parent license filename")
            destination = path / "parent-notices" / role / name
            if not verify:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(document["text"].encode("utf-8"))
            if sha256_file(destination) != document["sha256"]:
                raise ValueError("Parent license/notice integrity mismatch")


def read_manifest(path: str | Path) -> dict:
    manifest = json.loads((Path(path) / "child.json").read_text(encoding="utf-8"))
    required = {
        "format_version",
        "recipe",
        "recipient_config",
        "donor_config",
        "dtype",
        "training_status",
        "capability_status",
        "provenance",
        "audit",
        "recipient_tokenizer_sha256",
        "weights_sha256",
    }
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise ValueError("Invalid child manifest: missing required fields")
    if any(
        not isinstance(manifest[key], dict)
        for key in ("recipe", "recipient_config", "donor_config", "provenance", "audit")
    ):
        raise ValueError("Invalid child manifest: config/provenance/audit fields must be objects")
    if not isinstance(manifest["training_status"], str) or manifest["training_status"] not in {
        "uncalibrated",
        "seam_calibrated",
    }:
        raise ValueError("Invalid child training status")
    if not isinstance(manifest["dtype"], str):
        raise ValueError("Invalid checkpoint dtype")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != 1:
        raise ValueError("Unsupported child checkpoint format")
    return manifest


def load_artifact(path: str | Path, *, device: str = "cpu") -> StoredChild:
    path = Path(path)
    manifest = read_manifest(path)
    if sha256_file(path / "model.safetensors") != manifest["weights_sha256"]:
        raise ValueError("Checkpoint checksum mismatch; refusing changed or corrupt weights")
    recipe = Recipe.from_dict(manifest["recipe"])
    expected_status = (
        "synthetic_smoke_test" if recipe.synthetic else "unvalidated_pretrained_transplant"
    )
    if manifest["capability_status"] != expected_status:
        raise ValueError("Checkpoint capability label does not match its parent provenance")
    _parent_notices(path, manifest["provenance"], verify=True)
    if manifest["dtype"] not in DTYPES:
        raise ValueError("Unsupported checkpoint dtype")
    child = empty_child(recipe, manifest["recipient_config"], manifest["donor_config"])
    child.to(dtype=DTYPES[manifest["dtype"]])
    load_model(child, str(path / "model.safetensors"), strict=True, device="cpu")
    tokenizer = load_tokenizer(path / "tokenizer")
    check_tokenizer(tokenizer, child.recipient_config.vocab_size)
    if tokenizer_fingerprint(tokenizer) != manifest["recipient_tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer contract changed")
    if audit_child(child) != manifest["audit"]:
        raise ValueError("Checkpoint architecture or frozen-tensor audit mismatch")
    child.to(device).eval()
    return StoredChild(child, tokenizer, manifest)
