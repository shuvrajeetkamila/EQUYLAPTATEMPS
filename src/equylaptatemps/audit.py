"""Integrity evidence is about the actual tensors, not just declared method names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        # Byte view also supports bf16, unlike tensor.numpy() on a bf16 tensor.
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(raw).cast("B")
        for start in range(0, len(view), CHUNK_SIZE):
            digest.update(view[start : start + CHUNK_SIZE])
    return digest.hexdigest()


def module_fingerprint(module: nn.Module) -> str:
    return hash_state(module.state_dict())


def frozen_fingerprint(child) -> str:
    return hash_state(
        {
            key: value
            for key, value in child.state_dict().items()
            if not key.startswith(("input_seam.", "output_seam."))
        }
    )


def audit_child(child) -> dict:
    child.assert_trainable_contract()
    execution_config = {
        "recipe": child.recipe.to_dict(),
        "recipient_config": child.recipient_config.to_dict(),
        "donor_config": child.donor_config.to_dict(),
    }
    config_hash = hashlib.sha256(
        json.dumps(execution_config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {
        "execution_config_sha256": config_hash,
        "donor_sector_sha256": module_fingerprint(child.graft.layers),
        "all_frozen_tensors_sha256": frozen_fingerprint(child),
        "structure": child.structure_report(),
    }
