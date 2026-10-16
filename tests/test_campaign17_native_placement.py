"""Campaign 17 native-depth placement engine: offline synthetic tests."""

import pytest
import torch

from equylaptatemps.model import child_from_parents
from equylaptatemps.native_placement import build_native_depth_child, validate_execution_order
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent

DEPTH = 4


def _recipe():
    return Recipe(
        ParentSpec("synthetic://llama", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 2),
        LayerSpan(1, 2),
        42,
        positive_control=True,
    )


def _parents():
    recipient, donor = tiny_parent("llama", 1), tiny_parent("llama", 2)
    donor.load_state_dict(recipient.state_dict())
    return recipient, donor


def test_order_must_be_exact_permutation():
    with pytest.raises(ValueError, match="exact permutation"):
        validate_execution_order([0, 1, 2, 2], DEPTH)  # duplication
    with pytest.raises(ValueError, match="exact permutation"):
        validate_execution_order([0, 1, 2], DEPTH)  # deletion
    validate_execution_order([3, 2, 1, 0], DEPTH)  # legal permutation


def test_identity_replacement_matches_native_forward():
    recipient, donor = _parents()
    child, manifest = build_native_depth_child(
        recipient,
        execution_order=[0, 1, 2, 3],
        graft_span=(1, 2),
        seam_mode="residual",
        recipe=_recipe(),
        donor_parent=donor,
    )
    ids = torch.randint(3, 250, (2, 9))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        expected = recipient(input_ids=ids, attention_mask=mask).logits
        actual = child(ids, mask).logits
    assert torch.allclose(expected, actual, atol=1e-5)
    assert manifest["depth_before"] == manifest["depth_after"] == DEPTH
    assert manifest["blocks_removed"] == [] and manifest["blocks_duplicated"] == []
    assert manifest["blocks_moved"] == []
    assert manifest["blocks"][1]["action"] == "replaced"
    assert manifest["blocks"][1]["source_role"] == "graft_donor"
    child.assert_trainable_contract()


def test_exchange_manifest_and_depth_preservation():
    recipient, _ = _parents()
    order = [0, 3, 2, 1]  # swap blocks 1 and 3
    child, manifest = build_native_depth_child(
        recipient,
        execution_order=order,
        graft_span=(1, 3),
        seam_mode="bypass",
        recipe=_recipe(),
    )
    assert manifest["execution_order"] == order
    assert sorted(manifest["blocks_moved"]) == ["block:1", "block:3"]
    assert manifest["depth_after"] == DEPTH
    ids = torch.randint(3, 250, (1, 6))
    with torch.no_grad():
        assert torch.isfinite(child(ids, torch.ones_like(ids)).logits).all()


def test_incompatible_candidate_block_rejected():
    recipient, _ = _parents()
    mamba = tiny_parent("mamba", 3)
    with pytest.raises(ValueError, match="incompatible|depth does not match"):
        build_native_depth_child(
            recipient,
            execution_order=[0, 1, 2, 3],
            graft_span=(1, 2),
            seam_mode="bypass",
            recipe=_recipe(),
            donor_parent=mamba,
        )


def test_engine_child_parameter_total_matches_native_plus_seams():
    recipient, donor = _parents()
    child, _ = build_native_depth_child(
        recipient,
        execution_order=[0, 1, 2, 3],
        graft_span=(1, 2),
        seam_mode="bypass",
        recipe=_recipe(),
        donor_parent=donor,
    )
    native_params = sum(p.numel() for p in recipient.parameters())
    child_params = sum(p.numel() for p in child.parameters())
    assert child_params == native_params  # bypassed seams add nothing


def test_historical_path_still_available_for_mode_b():
    recipient, donor = _parents()
    child = child_from_parents(_recipe(), recipient, donor)
    assert child.structure_report()["transplanted_blocks"] == 1
