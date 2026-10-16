"""Load locally supplied, independently pinned parents; never fall back to a network download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .audit import module_fingerprint, sha256_file
from .backbones import extract_parent
from .contracts import parent_contract
from .model import child_from_parents
from .provenance import snapshot_evidence
from .recipe import Recipe
from .tokenization import (
    ByteTokenizer,
    Tokenizer,
    check_tokenizer,
    load_tokenizer,
    tokenizer_fingerprint,
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class LoadedParents:
    recipient: torch.nn.Module
    donor: torch.nn.Module
    tokenizer: Tokenizer
    provenance: dict
    donor_tokenizer: Tokenizer | None = None
    source_execution_contracts: dict | None = None


def load_local_model(path: str | Path, dtype: torch.dtype = torch.float32):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Local parent snapshot not found: {path}")
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if config.model_type not in {"mamba", "qwen2", "llama"}:
        raise ValueError(
            "Only stock Mamba, Qwen2, and Llama are supported; custom remote code is forbidden"
        )
    kwargs = {"attn_implementation": "eager"} if config.model_type in {"qwen2", "llama"} else {}
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        use_safetensors=True,
        **kwargs,
    ).eval()


def load_parents(
    recipe: Recipe, *, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> LoadedParents:
    recipe.validate()
    if recipe.recipient.local_path is None or recipe.donor.local_path is None:
        raise ValueError(
            "Local build requires both parent snapshot paths; use demo for random parents"
        )
    a_path, b_path = Path(recipe.recipient.local_path), Path(recipe.donor.local_path)
    snapshots = {
        "recipient": snapshot_evidence(recipe.recipient, a_path),
        "donor": snapshot_evidence(recipe.donor, b_path),
    }
    if not recipe.synthetic and any(
        evidence["revision_evidence"] != "locally_verified_pinned_download_record"
        for evidence in snapshots.values()
    ):
        raise ValueError(
            "Pretrained parents require verified pinned snapshot provenance; "
            "a manually declared revision is insufficient"
        )
    a_tokenizer, b_tokenizer = load_tokenizer(a_path), load_tokenizer(b_path)
    if not recipe.synthetic and any(
        isinstance(t, ByteTokenizer) for t in (a_tokenizer, b_tokenizer)
    ):
        raise ValueError("Synthetic byte-tokenizer fixtures cannot be labeled pretrained parents")
    a = load_local_model(a_path, dtype).to(device)
    b = load_local_model(b_path, dtype).to(device)
    recipe.validate(a.config, b.config)
    check_tokenizer(a_tokenizer, a.config.vocab_size)
    check_tokenizer(b_tokenizer, b.config.vocab_size)
    a_hash, b_hash = tokenizer_fingerprint(a_tokenizer), tokenizer_fingerprint(b_tokenizer)
    if a_hash == b_hash and not recipe.positive_control:
        raise ValueError("This strict stranger recipe also requires distinct tokenizer contracts")
    provenance = recipe.stranger_report()
    provenance["snapshots"] = snapshots
    provenance["tokenizers"] = {
        "recipient_sha256": a_hash,
        "donor_sha256": b_hash,
        "tokenizers_identical": a_hash == b_hash,
    }
    provenance["config_files"] = {
        "recipient_sha256": sha256_file(a_path / "config.json"),
        "donor_sha256": sha256_file(b_path / "config.json"),
    }
    return LoadedParents(
        a,
        b,
        a_tokenizer,
        provenance,
        b_tokenizer,
        {"recipient": parent_contract(a), "donor": parent_contract(b)},
    )


def build_local(recipe: Recipe, *, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    parents = load_parents(recipe, dtype=dtype, device=device)
    child = child_from_parents(recipe, parents.recipient, parents.donor)
    # Compare the exact transplanted weights BEFORE the original donor is released.
    span = recipe.donor_sector
    original_sector = torch.nn.ModuleList(
        extract_parent(parents.donor).layers[span.start : span.stop]
    )
    source_hash = module_fingerprint(original_sector)
    if module_fingerprint(child.graft.layers) != source_hash:
        raise RuntimeError("Donor transplant did not preserve every sector tensor")
    parents.provenance["source_donor_sector_sha256"] = source_hash
    return child, parents.tokenizer, parents.provenance
