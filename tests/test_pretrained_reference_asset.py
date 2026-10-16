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
    native_donor = ByteTokenizer(rotation=9)
    loaded = LoadedParents(recipient, donor, tokenizer, provenance, native_donor)
    assert loaded.donor_tokenizer is native_donor
    assert loaded.donor_tokenizer is not loaded.tokenizer


def test_actual_selected_negative_result_child_asset():
    root = Path(__file__).resolve().parents[1] / "examples/pretrained-pilot-reference"
    descriptor = json.loads((root / "reference.json").read_text())
    assert descriptor["recipe"]["donor_sector"] == [20, 22]
    assert descriptor["capability_status"] == "UNPROVEN"
    assert sha256_file(root / "interfaces.safetensors") == descriptor["interfaces_sha256"]
    assert descriptor["frozen_sha256"] == (
        "ba6c0f8331f18c222ed4c2b46016464e709702199bdf83e6236033b1a055867d"
    )
    with safe_open(root / "interfaces.safetensors", framework="pt") as tensors:
        assert len(tensors.keys()) == 8
        assert all(n.startswith(("input_seam.", "output_seam.")) for n in tensors.keys())
