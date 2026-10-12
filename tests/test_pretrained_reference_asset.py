"""Actual published NEW-interface integrity only; does not load gigabytes of parents."""

import json
from pathlib import Path

from safetensors import safe_open

from equylaptatemps.audit import sha256_file
from equylaptatemps.parents import LoadedParents
from equylaptatemps.tokenization import ByteTokenizer


def test_actual_pretrained_interface_asset_has_no_frozen_weights():
    root = Path(__file__).resolve().parents[1] / "examples/pretrained-structural-reference"
    descriptor = json.loads((root / "reference.json").read_text())
    path = root / "interfaces.safetensors"
    assert sha256_file(path) == descriptor["interfaces_sha256"]
    assert descriptor["parent_kind"] == "pretrained" and descriptor["self_contained"] is False
    assert descriptor["capability_status"] == "UNPROVEN"
    assert descriptor["structure"]["unique_parameters"] == 217553920
    assert descriptor["structure"]["trainable_dense_seam_parameters"] == 2366208
    assert path.stat().st_size == 9465560
    with safe_open(path, framework="pt") as stored:
        assert len(stored.keys()) == 8
        assert all(name.startswith(("input_seam.", "output_seam.")) for name in stored.keys())
        assert stored.get_slice("input_seam.projection.weight").get_shape() == [1536, 768]
        assert stored.get_slice("output_seam.projection.weight").get_shape() == [768, 1536]


def test_loaded_parents_exposes_distinct_donor_tokenizer_for_native_baselines(child_bundle):
    _, tokenizer, provenance, recipient, donor = child_bundle
    native_donor = ByteTokenizer()
    loaded = LoadedParents(recipient, donor, tokenizer, provenance, native_donor)
    assert loaded.donor_tokenizer is native_donor
    assert loaded.donor_tokenizer is not loaded.tokenizer
